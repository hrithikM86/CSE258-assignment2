"""Shared split, ranking, and full-catalog metrics; no model-specific data split."""
import csv
from pathlib import Path
import numpy as np
import torch
from gsasrec.data import load_training_data
from gsasrec.evaluate import SPLIT_DIR, load_validation, rank_targets, metrics_at_10
from gsasrec.test import load_held_out
from sasrec.protocol import verify_split


def load_data(split_dir=SPLIT_DIR, split="validation"):
    split_dir=Path(split_dir)
    fingerprint=verify_split(split_dir)
    training,num_items=load_training_data(split_dir)
    if sorted(training)!=list(range(1,len(training)+1)):
        raise ValueError("Expected shared consecutive user IDs.")
    targets,no_ties=load_validation(split_dir,training,num_items)
    history={u:list(items) for u,items in training.items()}
    if split=="test":
        history={u:items+[targets[u]] for u,items in history.items()}
        records=load_held_out(split_dir/"test.csv",training,num_items)
        targets={u:item for u,(item,_) in records.items()}
    elif split!="validation":
        raise ValueError("Unknown split.")
    return training,history,targets,no_ties,num_items,fingerprint


@torch.inference_mode()
def evaluate(scorer,histories,targets,batch_size=256):
    if batch_size<1 or not targets:
        raise ValueError("Expected positive batch size and nonempty targets.")
    users=list(targets); ranks={}
    for offset in range(0,len(users),batch_size):
        batch=users[offset:offset+batch_size]
        history=[histories[u] for u in batch]
        scores=scorer.score_all(batch,history)
        scores=torch.as_tensor(np.asarray(scores)) if not torch.is_tensor(scores) else scores
        if not torch.isfinite(scores).all():
            raise ValueError("Scorer returned nonfinite values.")
        scores=scores.clone()
        values=rank_targets(scores,history,[targets[u] for u in batch])
        ranks.update(zip(batch,values.tolist(),strict=True))
    return ranks


def summarize(ranks,no_ties,targets,training):
    items={item for h in training.values() for item in h}
    def group(values):
        if not values:
            return {"users":0,"recall_at_10":None,"ndcg_at_10":None}
        recall,ndcg=metrics_at_10(values)
        return {"users":len(values),"recall_at_10":recall,"ndcg_at_10":ndcg}
    return {"all_users":group(list(ranks.values())),
            "without_boundary_ties":group([r for u,r in ranks.items() if u in no_ties]),
            "targets_unseen_in_training":sum(targets[u] not in items for u in ranks)}
