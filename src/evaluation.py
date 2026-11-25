"""Shared deterministic full-catalog ranking contract."""
import math

def rank_items(scores, seen, k=10):
    return sorted((i for i in scores if i != 0 and i not in seen),
                  key=lambda i: (-scores[i], i))[:k]

def metrics(ranked, target, k=10):
    if target not in ranked[:k]:
        return {"ndcg@10": 0.0, "recall@10": 0.0}
    rank = ranked.index(target) + 1
    return {"ndcg@10": 1 / math.log2(rank + 1), "recall@10": 1.0}

def frequency_bin(count):
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    return "21+"
