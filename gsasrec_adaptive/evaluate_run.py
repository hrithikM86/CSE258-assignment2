"""Evaluate a selected matched-experiment checkpoint without resampling negatives."""
import argparse
import json
from pathlib import Path

import torch
from gsasrec_adaptive.data import load_training_data
from gsasrec_adaptive.evaluate import SPLIT_DIR, evaluate, load_validation
from gsasrec_adaptive.experiment import summarize
from gsasrec_adaptive.model import GSASRec
from gsasrec_adaptive.test import load_held_out
from sasrec.protocol import verify_split


def evaluation_data(split_dir,split):
    histories,num_items=load_training_data(split_dir)
    targets,no_ties=load_validation(split_dir,histories,num_items)
    training_items={i for history in histories.values() for i in history}
    if split=="test":
        histories={u:history+[targets[u]] for u,history in histories.items()}
        records=load_held_out(Path(split_dir)/"test.csv",histories,num_items)
        targets={u:item for u,(item,_) in records.items()}
    elif split!="validation":
        raise ValueError("Unknown evaluation split.")
    return histories,targets,no_ties,training_items,num_items


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--split",choices=("validation","test"),default="validation")
    parser.add_argument("--split-dir",type=Path,default=SPLIT_DIR)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if min(args.batch_size,args.threads)<1:
        parser.error("Counts must be positive.")
    torch.set_num_threads(args.threads)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if checkpoint.get("model_type")!="gsasrec_adaptive":
        parser.error("Expected a matched gSASRec experiment checkpoint.")
    if checkpoint["metadata"]["manifest_sha256"]!=verify_split(args.split_dir):
        raise ValueError("Checkpoint split fingerprint mismatch.")
    histories,targets,no_ties,training_items,num_items=evaluation_data(args.split_dir,args.split)
    if checkpoint["model_config"]["num_items"]!=num_items:
        raise ValueError("Catalog mismatch.")
    model=GSASRec(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    ranks=evaluate(model,histories,targets,args.batch_size)
    result={"split":args.split,"mode":checkpoint["metadata"]["mode"],
            "seed":checkpoint["seed"],"selected_epoch":checkpoint["epochs"],
            "smoke_run":checkpoint["smoke_run"],**summarize(ranks,no_ties,targets,training_items)}
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
