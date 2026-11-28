import argparse
import csv
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from gsasrec_adaptive.data import load_training_data
from gsasrec_adaptive.model import build_model


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "beauty_5core"


def load_validation(split_dir, histories, num_items):
    targets = {}
    with (split_dir / "validation.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            if user_id in targets or not 1 <= item_id <= num_items:
                raise ValueError(
                    f"Invalid or repeated validation row for user {user_id}."
                )
            targets[user_id] = item_id

    with (split_dir / "boundary_ties.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        no_ties = {
            int(row["user_id"])
            for row in csv.DictReader(file)
            if row["train_validation_tie"] == row["validation_test_tie"] == "0"
        }

    if not targets or targets.keys() != histories.keys():
        raise ValueError("Expected one validation target for each training user.")

    return targets, no_ties


def rank_targets(scores, histories, targets):
    """Return one-based ranks. Score column j corresponds to item ID j + 1."""
    for row, (history, target) in enumerate(zip(histories, targets, strict=True)):
        if target in history:
            raise ValueError("Validation target also appears in training history.")
        history_columns = torch.as_tensor(history, device=scores.device) - 1
        scores[row, history_columns] = -torch.inf

    item_ids = torch.arange(1, scores.size(1) + 1, device=scores.device)
    target_ids = torch.as_tensor(targets, device=scores.device)
    target_scores = scores.gather(1, (target_ids - 1).unsqueeze(1))

    # When scores tie, the smaller item ID ranks first.
    higher = scores > target_scores
    tied_ahead = (scores == target_scores) & (item_ids < target_ids.unsqueeze(1))
    return 1 + (higher | tied_ahead).sum(dim=1)


@torch.inference_mode()
def evaluate(model, histories, targets, batch_size):
    model.eval()
    ranks = {}
    users = list(targets)

    for offset in range(0, len(users), batch_size):
        batch_users = users[offset : offset + batch_size]
        full_histories = [histories[user] for user in batch_users]
        sequences = [
            torch.tensor(history[-model.max_length :], dtype=torch.long)
            for history in full_histories
        ]

        inputs = pad_sequence(sequences, batch_first=True)
        last_positions = inputs.count_nonzero(dim=1) - 1
        states = model(inputs)[torch.arange(len(batch_users)), last_positions]

        # Score the full catalog. Row 0 in the embedding table is padding.
        scores = states @ model.output_embeddings.weight[1:].T
        batch_targets = [targets[user] for user in batch_users]
        batch_ranks = rank_targets(scores, full_histories, batch_targets)
        ranks.update(zip(batch_users, batch_ranks.tolist(), strict=True))

    return ranks


def metrics_at_10(ranks):
    ranks = torch.tensor(ranks, dtype=torch.float32)
    hits = (ranks <= 10).float()
    recall = hits.mean().item()
    ndcg = (hits / torch.log2(ranks + 1)).mean().item()
    return recall, ndcg


def report(label, ranks):
    if not ranks:
        print(f"{label}: no users")
        return

    recall, ndcg = metrics_at_10(ranks)
    print(
        f"{label} ({len(ranks):,} users): "
        f"Recall@10={recall:.6f}, NDCG@10={ndcg:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate gSASRec on Beauty validation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Older baseline checkpoints were saved before model_type was added.
    model_type = checkpoint.get("model_type", "gsasrec")
    if model_type not in ("gsasrec", "gsasrec_text"):
        parser.error(f"Unsupported checkpoint model type: {model_type}")
    if model_type == "gsasrec_text" and args.text_embeddings is None:
        parser.error("This checkpoint requires --text-embeddings.")
    if model_type == "gsasrec" and args.text_embeddings is not None:
        parser.error("Omit --text-embeddings for an original gSASRec checkpoint.")

    histories, num_items = load_training_data(SPLIT_DIR)
    targets, no_ties = load_validation(SPLIT_DIR, histories, num_items)
    config = checkpoint["model_config"]

    if config["num_items"] != num_items:
        raise ValueError("The checkpoint catalog size differs from the split.")

    model = build_model(config, args.text_embeddings)
    model.load_state_dict(checkpoint["model_state_dict"])

    ranks = evaluate(model, histories, targets, args.batch_size)
    report("Validation", list(ranks.values()))
    report(
        "Validation without boundary ties",
        [rank for user, rank in ranks.items() if user in no_ties],
    )


if __name__ == "__main__":
    main()