"""SASRec input adapter using the team's unchanged ranking/metric functions."""
import argparse
import json
from pathlib import Path

import torch
from gsasrec.data import load_training_data
from gsasrec.evaluate import SPLIT_DIR, load_validation, rank_targets, metrics_at_10
from gsasrec.test import load_held_out
from sasrec.data import left_pad
from sasrec.protocol import verify_split


@torch.inference_mode()
def evaluate(model, histories, targets, batch_size=128):
    if batch_size < 1 or not targets:
        raise ValueError("Expected positive batch_size and nonempty targets.")
    model.eval()
    device = next(model.parameters()).device
    users = list(targets)
    ranks = {}
    for offset in range(0, len(users), batch_size):
        batch_users = users[offset:offset+batch_size]
        full_histories = [histories[u] for u in batch_users]
        inputs = left_pad(full_histories, model.max_length).to(device)
        states = model(inputs)[:, -1]
        scores = states @ model.item_embeddings.weight[1:].T
        batch_ranks = rank_targets(scores, full_histories, [targets[u] for u in batch_users])
        ranks.update(zip(batch_users, batch_ranks.cpu().tolist(), strict=True))
    return ranks


def summarize(ranks, no_ties, targets, training_items):
    def group(values):
        if not values:
            return {"users": 0, "recall_at_10": None, "ndcg_at_10": None}
        recall, ndcg = metrics_at_10(values)
        return {"users": len(values), "recall_at_10": recall, "ndcg_at_10": ndcg}
    return {
        "all_users": group(list(ranks.values())),
        "without_boundary_ties": group([r for u, r in ranks.items() if u in no_ties]),
        "targets_unseen_in_training": sum(targets[u] not in training_items for u in ranks),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SASRec with the shared full-catalog protocol.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error("batch-size and threads must be positive.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable.")
    torch.set_num_threads(args.threads)
    from sasrec.model import SASRec
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "sasrec":
        parser.error("Expected a SASRec checkpoint.")
    manifest_sha256 = verify_split(args.split_dir)
    if checkpoint["metadata"]["manifest_sha256"] != manifest_sha256:
        raise ValueError("Checkpoint was trained on a different split manifest.")
    histories, num_items = load_training_data(args.split_dir)
    targets, no_ties = load_validation(args.split_dir, histories, num_items)
    training_items = {i for h in histories.values() for i in h}
    if checkpoint["model_config"]["num_items"] != num_items:
        raise ValueError("Checkpoint catalog size differs from the split.")
    if args.split == "test":
        histories = {u: h + [targets[u]] for u, h in histories.items()}
        records = load_held_out(args.split_dir / "test.csv", histories, num_items)
        targets = {u: item for u, (item, _) in records.items()}
    model = SASRec(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ranks = evaluate(model, histories, targets, args.batch_size)
    result = {"split": args.split, "selected_epoch": checkpoint["epoch"],
              "seed": checkpoint["seed"], "smoke_run": checkpoint.get("smoke_run", False),
              **summarize(ranks, no_ties, targets, training_items)}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
