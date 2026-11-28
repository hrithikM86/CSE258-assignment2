"""Matched gSASRec experiments; training uses validation only."""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from gsasrec_adaptive.data import load_training_data
from gsasrec_adaptive.evaluate import SPLIT_DIR, evaluate, load_validation
from gsasrec_adaptive.model import GSASRec
from gsasrec_adaptive.neighbors import load_or_build, file_sha256
from gsasrec_adaptive.sampling import SemanticSampler, AdaptiveSemanticSampler
from gsasrec_adaptive.training import gsasrec_loss, make_batch
from sasrec.protocol import verify_split

ROOT=Path(__file__).resolve().parents[1]


def metrics(ranks):
    values=list(ranks)
    if not values:
        return {"users":0,"recall_at_10":None,"ndcg_at_10":None}
    hits=[rank for rank in values if rank<=10]
    return {"users":len(values),"recall_at_10":len(hits)/len(values),
            "ndcg_at_10":sum(1/math.log2(rank+1) for rank in hits)/len(values)}


def summarize(ranks,no_ties,targets,training_items):
    return {"all_users":metrics(ranks.values()),
            "without_boundary_ties":metrics(r for u,r in ranks.items() if u in no_ties),
            "targets_unseen_in_training":sum(targets[u] not in training_items for u in ranks)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("uniform","fixed","curriculum","confidence"),required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--split-dir",type=Path,default=SPLIT_DIR)
    parser.add_argument("--embeddings",type=Path,default=SPLIT_DIR/"item_text_embeddings.npz")
    parser.add_argument("--cache-dir",type=Path,default=ROOT/"artifacts/adaptive_sampling")
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--patience",type=int,default=5)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--num-negatives",type=int,default=32)
    parser.add_argument("--t",type=float,default=0.5)
    parser.add_argument("--neighbors",type=int,default=100)
    parser.add_argument("--semantic-fraction",type=float,default=0.5)
    parser.add_argument("--warmup-epochs",type=int,default=3)
    parser.add_argument("--ramp-epochs",type=int,default=10)
    parser.add_argument("--max-batches",type=int)
    parser.add_argument("--max-eval-users",type=int)
    args=parser.parse_args()
    if min(args.epochs,args.patience,args.batch_size,args.threads,args.num_negatives,args.neighbors)<1:
        parser.error("Counts must be positive.")
    if not 0<=args.t<=1 or not 0<=args.semantic_fraction<1:
        parser.error("Invalid calibration or mixture fraction.")
    if any(v is not None and v<1 for v in (args.max_batches,args.max_eval_users)):
        parser.error("Smoke limits must be positive.")
    if args.warmup_epochs<0 or args.ramp_epochs<1:
        parser.error("Invalid warmup/ramp epochs.")
    if args.run_dir.exists():
        parser.error("Choose a new run-dir; existing experiments are never overwritten.")
    torch.set_num_threads(args.threads)
    fingerprint=verify_split(args.split_dir)
    histories,num_items=load_training_data(args.split_dir)
    if args.num_negatives>num_items-1:
        parser.error("Negative budget exceeds catalog minus current positive.")
    targets,no_ties=load_validation(args.split_dir,histories,num_items)
    if args.max_eval_users:
        targets=dict(list(targets.items())[:args.max_eval_users])
    training_items={i for history in histories.values() for i in history}
    sampler,neighbor_metadata=None,None
    if args.mode!="uniform":
        table,has_text,neighbor_metadata=load_or_build(args.embeddings,args.cache_dir,num_items,args.neighbors)
        if args.mode=="fixed":
            sampler=SemanticSampler(num_items,table,has_text,args.semantic_fraction,args.seed)
        else:
            sampler=AdaptiveSemanticSampler(num_items,table,has_text,args.semantic_fraction,args.seed,
                        mode=args.mode,warmup_epochs=args.warmup_epochs,ramp_epochs=args.ramp_epochs)
    # Cache building and sampler initialization do not alter the reference RNG.
    rng=random.Random(args.seed)
    torch.manual_seed(args.seed)
    model_config={"num_items":num_items,"max_length":200}
    model=GSASRec(**model_config)
    initial_hash=hashlib.sha256(b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    sequences=[h for h in histories.values() if len(h)>=2]
    smoke=args.max_batches is not None or args.max_eval_users is not None
    args.run_dir.mkdir(parents=True)
    metadata={"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "mode":args.mode,"seed":args.seed,"smoke_run":smoke,"model_config":model_config,
              "source_sha256":{name:file_sha256(Path(__file__).parent/name) for name in ("sampling.py","training.py","experiment.py","model.py","neighbors.py")},
              "initial_state_sha256":initial_hash,"manifest_sha256":fingerprint,
              "neighbor_cache":neighbor_metadata,"parameter_count":sum(p.numel() for p in model.parameters()),
              "versions":{"python":sys.version,"torch":str(torch.__version__),"numpy":str(np.__version__)},
              "negative_eligibility":"all catalog IDs except padding and current positive",
              "semantic_anchor":"current training target","calibration":"uniform theory; mixed sampling is heuristic"}
    (args.run_dir/"config.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2),flush=True)
    best,best_epoch=-math.inf,0
    start_run=perf_counter()
    for epoch in range(1,args.epochs+1):
        start=perf_counter()
        model.train(); rng.shuffle(sequences)
        total_loss,total_targets=0.,0
        if sampler is not None:
            if isinstance(sampler,AdaptiveSemanticSampler):
                sampler.begin_epoch(epoch)
            else:
                sampler.reset_stats()
        for batch_number,offset in enumerate(range(0,len(sequences),args.batch_size)):
            if args.max_batches is not None and batch_number>=args.max_batches:
                break
            inputs,labels=make_batch(sequences[offset:offset+args.batch_size],model.max_length)
            optimizer.zero_grad(set_to_none=True)
            loss=gsasrec_loss(model,inputs,labels,num_items,args.num_negatives,args.t,sampler)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss.")
            loss.backward(); optimizer.step()
            count=labels.count_nonzero().item()
            total_loss+=loss.item()*count; total_targets+=count
        training_seconds=perf_counter()-start
        start=perf_counter()
        ranks=evaluate(model,histories,targets,args.batch_size)
        result=summarize(ranks,no_ties,targets,training_items)
        stats=sampler.stats.copy() if sampler is not None else {"total_negatives":total_targets*args.num_negatives,"requested_semantic":0,"actual_semantic":0,"fallback_negatives":0}
        record={"epoch":epoch,"loss":total_loss/total_targets,"training_seconds":training_seconds,
                "validation_seconds":perf_counter()-start,"smoke_run":smoke,"sampling":stats,**result}
        with (args.run_dir/"metrics.jsonl").open("a") as file:
            file.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
        ndcg=result["all_users"]["ndcg_at_10"]
        if ndcg>best:
            best,best_epoch=ndcg,epoch
            torch.save({"model_type":"gsasrec_adaptive","model_config":model_config,
                        "model_state_dict":model.state_dict(),"epochs":epoch,"seed":args.seed,
                        "smoke_run":smoke,"metadata":metadata,"validation":result},args.run_dir/"best.pt")
        elif epoch-best_epoch>=args.patience:
            break
    summary={"best_epoch":best_epoch,"stopped_epoch":epoch,"best_validation_ndcg_at_10":best,
             "elapsed_seconds":perf_counter()-start_run,"smoke_run":smoke}
    (args.run_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
