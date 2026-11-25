"""Validation-first course baselines using the shared evaluator."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from .evaluation import rank_items, metrics, frequency_bin

def run(data, output, split):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (data / "sequences.jsonl").read_text().splitlines()]
    counts = {int(k): v for k, v in json.loads((data / "train_item_counts.json").read_text()).items()}
    item_users = defaultdict(set)
    user_items = {}
    transitions = defaultdict(Counter)
    for row in rows:
        ids = [e["item"] for e in row["train"]]
        user_items[row["user"]] = set(ids)
        for i in ids:
            item_users[i].add(row["user"])
        for a, b in zip(ids, ids[1:]):
            transitions[a][b] += 1
    summaries = {}
    for model in ("popularity", "item_cf", "markov"):
        started = time.perf_counter()
        results = []
        for row in rows:
            context = row["train"] + ([row["validation"]] if split == "test" else [])
            seen = {e["item"] for e in context}
            scores = dict(counts) if model == "popularity" else {i: 0.0 for i in counts}
            if model == "markov":
                last = context[-1]["item"]
                total = sum(transitions[last].values())
                if total:
                    scores.update({i: n / total for i, n in transitions[last].items()})
            if model == "item_cf":
                for event in context:
                    a = event["item"]
                    neighbors = Counter()
                    for u in item_users[a]:
                        neighbors.update(user_items[u])
                    for b, overlap in neighbors.items():
                        union = len(item_users[a]) + len(item_users[b]) - overlap
                        scores[b] += overlap / union if union else 0.0
            target = row[split]["item"]
            results.append({"user": row["user"], "target": target,
                            "frequency_bin": frequency_bin(counts[target]),
                            **metrics(rank_items(scores, seen), target)})
        summary = {key: sum(r[key] for r in results)/len(results)
                   for key in ("ndcg@10", "recall@10")}
        summary["by_frequency"] = {}
        for group in ("0", "1-5", "6-20", "21+"):
            subset = [r for r in results if r["frequency_bin"] == group]
            summary["by_frequency"][group] = {"users": len(subset), **{
                key: sum(r[key] for r in subset)/len(subset) if subset else None
                for key in ("ndcg@10", "recall@10")}}
        summary.update(model=model, split=split, users=len(results),
                       elapsed_seconds=time.perf_counter()-started,
                       manifest_sha256=hashlib.sha256((data/"manifest.json").read_bytes()).hexdigest())
        (output/f"{model}_{split}_per_user.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True)+"\n" for r in results))
        (output/f"{model}_{split}.json").write_text(json.dumps(summary, indent=2)+"\n")
        summaries[model] = summary
        print(json.dumps(summary), flush=True)
    return summaries

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--output", default="runs/baselines")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    args = p.parse_args()
    run(args.data, args.output, args.split)

if __name__ == "__main__":
    main()
