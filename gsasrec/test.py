import argparse
import csv
from pathlib import Path

import torch

from gsasrec.data import load_training_data
from gsasrec.evaluate import (
    SPLIT_DIR,
    evaluate,
    load_validation,
    report,
)
from gsasrec.model import build_model


def load_held_out(path, users, num_items):
    records = {}

    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            user = int(row["user_id"])
            item = int(row["item_id"])
            rating = float(row["rating"])

            if user in records or not 1 <= item <= num_items:
                raise ValueError(f"Invalid or repeated row for user {user} in {path}.")
            if rating not in (1, 2, 3, 4, 5):
                raise ValueError(f"Invalid rating for user {user} in {path}.")

            records[user] = (item, int(rating))

    if records.keys() != users.keys():
        raise ValueError(f"Expected one interaction per training user in {path}.")

    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate final Beauty test results.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    model_type = checkpoint.get("model_type", "gsasrec")
    if model_type not in ("gsasrec", "gsasrec_text", "gsasrec_text_rating"):
        parser.error(f"Unsupported model type: {model_type}")
    if model_type != "gsasrec" and args.text_embeddings is None:
        parser.error("This checkpoint requires --text-embeddings.")
    if model_type == "gsasrec" and args.text_embeddings is not None:
        parser.error("Omit --text-embeddings for the original gSASRec checkpoint.")

    ratings = None
    if model_type == "gsasrec_text_rating":
        from gsasrec_text_rating.data import load_training_data_with_ratings

        histories, ratings, num_items = load_training_data_with_ratings(SPLIT_DIR)
    else:
        histories, num_items = load_training_data(SPLIT_DIR)

    _, no_ties = load_validation(SPLIT_DIR, histories, num_items)
    validation = load_held_out(
        SPLIT_DIR / "validation.csv", histories, num_items
    )
    test = load_held_out(SPLIT_DIR / "test.csv", histories, num_items)

    for user, (item, rating) in validation.items():
        if item in histories[user]:
            raise ValueError(f"Validation item repeats in history for user {user}.")
        histories[user].append(item)
        if ratings is not None:
            ratings[user].append(rating)

    targets = {user: item for user, (item, _) in test.items()}
    config = checkpoint["model_config"]
    if config["num_items"] != num_items:
        raise ValueError("Checkpoint catalog size differs from the dataset.")

    if model_type == "gsasrec_text_rating":
        from gsasrec_text_rating.evaluate import evaluate as evaluate_ratings
        from gsasrec_text_rating.model import TextRatingGSASRec

        model = TextRatingGSASRec(
            embeddings_path=args.text_embeddings, **config
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        ranks = evaluate_ratings(
            model, histories, ratings, targets, args.batch_size
        )
    else:
        model = build_model(config, args.text_embeddings)
        model.load_state_dict(checkpoint["model_state_dict"])
        ranks = evaluate(model, histories, targets, args.batch_size)

    print(f"Model: {model_type}, selected epoch: {checkpoint['epochs']}")
    report("Test", list(ranks.values()))
    report(
        "Test without boundary ties",
        [rank for user, rank in ranks.items() if user in no_ties],
    )


if __name__ == "__main__":
    main()