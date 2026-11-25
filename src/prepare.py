"""Deterministic, streaming Beauty preparation. See DATA_CONTRACT.md."""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield ast.literal_eval(line)

def flatten(value):
    if isinstance(value, (list, tuple)):
        return " ".join(flatten(x) for x in value)
    return str(value).strip() if value is not None else ""

def write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False) + "\n")

def prepare(reviews, metadata, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pairs = {}
    n_rows = 0
    for r in records(reviews):
        n_rows += 1
        key = (r["reviewerID"], r["asin"])
        event = (int(r["unixReviewTime"]), float(r["overall"]))
        if key not in pairs or event < pairs[key]:
            pairs[key] = event
    users = defaultdict(list)
    for (u, i), (timestamp, rating) in pairs.items():
        users[u].append((timestamp, i, rating))
    users = {u: sorted(events) for u, events in users.items() if len(events) >= 5}
    item_ids = sorted({i for events in users.values() for _, i, _ in events})
    item_map = {raw: i + 1 for i, raw in enumerate(item_ids)}
    user_map = {raw: i + 1 for i, raw in enumerate(sorted(users))}
    counts = Counter()
    lengths = Counter()
    ratings = Counter()
    timestamps = []
    ties = 0
    with (output / "sequences.jsonl").open("w") as stream:
        for u in sorted(users):
            events = users[u]
            encoded = [{"item": item_map[i], "timestamp": t, "rating": r}
                       for t, i, r in events]
            counts.update(x["item"] for x in encoded[:-2])
            lengths[len(events)] += 1
            ratings.update(str(r) for _, _, r in events)
            timestamps.extend(t for t, _, _ in events)
            ties += sum(a[0] == b[0] for a, b in zip(events, events[1:]))
            row = {"user": user_map[u], "train": encoded[:-2],
                   "validation": encoded[-2], "test": encoded[-1]}
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    texts = {}
    coverage = Counter()
    for r in records(metadata):
        raw = r.get("asin")
        if raw not in item_map:
            continue
        fields = {key: flatten(r.get(key)) for key in
                  ("title", "description", "brand", "categories")}
        text = " ".join(v for v in fields.values() if v)
        # Stable choice if metadata contains repeated product records.
        if raw not in texts or text < texts[raw]["text"]:
            texts[raw] = {"item": item_map[raw], "text": text, **fields}
    with (output / "items.jsonl").open("w") as stream:
        for raw in item_ids:
            row = texts.get(raw, {"item": item_map[raw], "text": ""})
            row = {**row, "missing_text": not bool(row["text"])}
            for key in ("title", "description", "brand", "categories", "text"):
                coverage[key] += bool(row.get(key))
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(output / "item_map.json", item_map)
    write_json(output / "user_map.json", user_map)
    write_json(output / "train_item_counts.json",
               {str(i): counts[i] for i in range(1, len(item_map) + 1)})
    audit = {"raw_reviews": n_rows, "duplicate_user_item_events": n_rows-len(pairs),
             "retained_users": len(users), "retained_items": len(item_map),
             "retained_interactions": sum(len(x) for x in users.values()),
             "sequence_length_histogram": dict(lengths), "rating_histogram": dict(ratings),
             "metadata_coverage_counts": dict(coverage),
             "adjacent_equal_timestamps": ties,
             "zero_training_items": sum(counts[i] == 0 for i in item_map.values()),
             "min_timestamp": min(timestamps) if timestamps else None,
             "max_timestamp": max(timestamps) if timestamps else None}
    write_json(output / "audit.json", audit)
    manifest = {"contract": "v1", "sources": {
        Path(reviews).name: digest(reviews), Path(metadata).name: digest(metadata)},
        "files": {p.name: digest(p) for p in sorted(output.iterdir())
                  if p.name not in ("manifest.json", "SHA256SUMS") and p.is_file()}}
    write_json(output / "manifest.json", manifest)
    paths = sorted(p for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in paths))
    return audit

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reviews", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", default="data/processed")
    args = p.parse_args()
    print(json.dumps(prepare(args.reviews, args.metadata, args.output), indent=2))

if __name__ == "__main__":
    main()
