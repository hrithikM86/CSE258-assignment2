"""Fit independent lecture baselines; freeze validation selection before test."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import scipy
from scipy import sparse
import torch
from course_baselines.bpr import BPR,TripletSampler
from course_baselines.item_cf import ItemCF,interaction_matrix
from course_baselines.protocol import SPLIT_DIR,load_data,evaluate,summarize


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+"\n")


class Popularity:
    def __init__(self,counts):
        self.counts=counts
    def score_all(self,users,histories):
        return np.broadcast_to(self.counts,(len(users),len(self.counts))).copy()


def train_bpr(training,history,targets,no_ties,num_items,factors,regularization,args,folder,fingerprint):
    folder.mkdir()
    torch.manual_seed(args.seed)
    sampler=TripletSampler(training,num_items,args.seed)
    config={"num_users":len(training),"num_items":num_items,"factors":factors}
    model=BPR(**config)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    best,best_epoch=-float("inf"),0
    begin=perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); start=perf_counter(); total_loss=0.; count=0
        for offset in range(0,len(sampler.users),args.batch_size):
            n=min(args.batch_size,len(sampler.users)-offset)
            users,positive,negative=sampler.sample(n)
            optimizer.zero_grad(set_to_none=True)
            loss=model.loss(users,positive,negative,regularization)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite BPR loss.")
            loss.backward(); optimizer.step()
            total_loss+=loss.item()*n; count+=n
        training_seconds=perf_counter()-start
        model.eval(); start=perf_counter()
        ranks=evaluate(model,history,targets,args.eval_batch_size)
        result=summarize(ranks,no_ties,targets,training)
        record={"epoch":epoch,"loss":total_loss/count,"training_seconds":training_seconds,
                "validation_seconds":perf_counter()-start,**result}
        with (folder/"metrics.jsonl").open("a") as file:
            file.write(json.dumps(record)+"\n")
        print(json.dumps({"model":"bpr","factors":factors,"regularization":regularization,**record}),flush=True)
        ndcg=result["all_users"]["ndcg_at_10"]
        if ndcg>best:
            best,best_epoch=ndcg,epoch
            torch.save({"model_type":"bpr_mf","model_config":config,"model_state_dict":model.state_dict(),
                        "regularization":regularization,"seed":args.seed,"selected_epoch":epoch,
                        "manifest_sha256":fingerprint,"validation":result},folder/"best.pt")
        elif epoch-best_epoch>=args.patience:
            break
    checkpoint=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
    summary={"model":"bpr","config":config,"regularization":regularization,"selected_epoch":best_epoch,
             "stopped_epoch":epoch,"elapsed_seconds":perf_counter()-begin,"validation":checkpoint["validation"],
             "parameter_count":sum(p.numel() for p in model.parameters())}
    save_json(folder/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--split-dir",type=Path,default=SPLIT_DIR)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--batch-size",type=int,default=2048)
    parser.add_argument("--eval-batch-size",type=int,default=256)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if min(args.epochs,args.patience,args.batch_size,args.eval_batch_size,args.threads)<1:
        parser.error("Counts must be positive.")
    if args.run_dir.exists():
        parser.error("Choose a fresh run directory to preserve earlier results.")
    torch.set_num_threads(args.threads)
    training,history,targets,no_ties,num_items,fingerprint=load_data(args.split_dir)
    args.run_dir.mkdir(parents=True)
    start_run=perf_counter()
    metadata={"seed":args.seed,"smoke_run":False,"manifest_sha256":fingerprint,
              "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "versions":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"torch":str(torch.__version__)},
              "source_sha256":{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ("run.py","bpr.py","item_cf.py","protocol.py")},
              "training_users":len(training),"training_interactions":sum(map(len,training.values())),"num_items":num_items,
              "design":json.loads((Path(__file__).parent/"design.json").read_text())}
    save_json(args.run_dir/"config.json",metadata)
    # Nonpersonalized sanity baseline; fit on training counts only.
    counts=np.asarray(interaction_matrix(training,num_items).sum(0)).ravel()
    np.save(args.run_dir/"popularity.npy",counts)
    start=perf_counter()
    pop_validation=summarize(evaluate(Popularity(counts),history,targets,args.eval_batch_size),no_ties,targets,training)
    print(json.dumps({"model":"popularity","validation":pop_validation}),flush=True)
    cf_results=[]; best_cf=None
    for metric in ("jaccard","cosine"):
        start=perf_counter(); model=ItemCF.fit(training,num_items,metric)
        fit_seconds=perf_counter()-start; start=perf_counter()
        result=summarize(evaluate(model,history,targets,args.eval_batch_size),no_ties,targets,training)
        entry={"model":"item_cf","metric":metric,"fit_seconds":fit_seconds,"validation_seconds":perf_counter()-start,
               "nonzero_similarities":model.similarity.nnz,"parameter_count":0,"validation":result}
        cf_results.append(entry); print(json.dumps(entry),flush=True)
        if best_cf is None or result["all_users"]["ndcg_at_10"]>best_cf["validation"]["all_users"]["ndcg_at_10"]:
            best_cf=entry
            sparse.save_npz(args.run_dir/"item_cf.npz",model.similarity)
            save_json(args.run_dir/"item_cf.json",{"manifest_sha256":fingerprint,**entry})
    save_json(args.run_dir/"item_cf_grid.json",cf_results)
    bpr_results=[]; best_bpr=None
    for factors in (32,64):
        for regularization in (1e-4,1e-3):
            name=f"bpr_f{factors}_reg{regularization:g}"
            result=train_bpr(training,history,targets,no_ties,num_items,factors,regularization,args,args.run_dir/name,fingerprint)
            result["run_name"]=name; bpr_results.append(result)
            if best_bpr is None or result["validation"]["all_users"]["ndcg_at_10"]>best_bpr["validation"]["all_users"]["ndcg_at_10"]:
                best_bpr=result
    save_json(args.run_dir/"bpr_grid.json",bpr_results)
    freeze={"frozen_at_utc":datetime.now(timezone.utc).isoformat(),"criterion":"all-user validation NDCG@10",
            "item_cf":best_cf,"bpr":best_bpr,"seed":args.seed,"test_not_used_for_selection":True}
    save_json(args.run_dir/"frozen_selection.json",freeze)
    print("VALIDATION SELECTION FROZEN",json.dumps(freeze),flush=True)
    # Only now load test targets. No retraining or refitting follows.
    _,test_history,test_targets,test_no_ties,_,test_fingerprint=load_data(args.split_dir,"test")
    assert test_fingerprint==fingerprint
    model=ItemCF(sparse.load_npz(args.run_dir/"item_cf.npz"))
    cf_test=summarize(evaluate(model,test_history,test_targets,args.eval_batch_size),test_no_ties,test_targets,training)
    checkpoint=torch.load(args.run_dir/best_bpr["run_name"]/"best.pt",map_location="cpu",weights_only=True)
    bpr=BPR(**checkpoint["model_config"]); bpr.load_state_dict(checkpoint["model_state_dict"]); bpr.eval()
    bpr_test=summarize(evaluate(bpr,test_history,test_targets,args.eval_batch_size),test_no_ties,test_targets,training)
    pop_test=summarize(evaluate(Popularity(counts),test_history,test_targets,args.eval_batch_size),test_no_ties,test_targets,training)
    import shutil
    shutil.copyfile(args.run_dir/best_bpr["run_name"]/"best.pt",args.run_dir/"bpr.pt")
    report={"status":"completed full-data baseline experiments","seed":args.seed,"smoke_run":False,
            "manifest_sha256":fingerprint,"elapsed_seconds":perf_counter()-start_run,"selection":freeze,
            "popularity":{"validation":pop_validation,"test":pop_test},
            "item_cf":{"metric":best_cf["metric"],"validation":best_cf["validation"],"test":cf_test},
            "bpr":{"config":best_bpr["config"],"regularization":best_bpr["regularization"],
                   "selected_epoch":best_bpr["selected_epoch"],"validation":best_bpr["validation"],"test":bpr_test}}
    save_json(args.run_dir/"results.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
