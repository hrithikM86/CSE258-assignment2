"""Re-evaluate saved item-CF or BPR with the exact shared catalog protocol."""
import argparse
import json
from pathlib import Path
from scipy import sparse
import torch
from course_baselines.bpr import BPR
from course_baselines.item_cf import ItemCF
from course_baselines.protocol import SPLIT_DIR,load_data,evaluate,summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=("item_cf","bpr"),required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--split",choices=("validation","test"),default="validation")
    parser.add_argument("--split-dir",type=Path,default=SPLIT_DIR)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(); torch.set_num_threads(4)
    training,history,targets,no_ties,n,fingerprint=load_data(args.split_dir,args.split)
    if args.model=="item_cf":
        metadata=json.loads(args.checkpoint.with_suffix(".json").read_text())
        model=ItemCF(sparse.load_npz(args.checkpoint))
        if model.num_items!=n:
            raise ValueError("Catalog mismatch.")
    else:
        metadata=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
        config=metadata["model_config"]
        if config["num_items"]!=n or config["num_users"]!=len(training):
            raise ValueError("Catalog/user mapping mismatch.")
        model=BPR(**config); model.load_state_dict(metadata["model_state_dict"]); model.eval()
    if metadata["manifest_sha256"]!=fingerprint:
        raise ValueError("Checkpoint split mismatch.")
    result={"model":args.model,"split":args.split,**summarize(evaluate(model,history,targets),no_ties,targets,training)}
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
