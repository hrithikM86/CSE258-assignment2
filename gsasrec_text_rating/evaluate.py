import argparse
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from gsasrec.evaluate import SPLIT_DIR, load_validation, rank_targets, report
from gsasrec_text_rating.data import load_training_data_with_ratings
from gsasrec_text_rating.model import TextRatingGSASRec


@torch.inference_mode()
def evaluate(model, histories, ratings, targets, batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    users = list(targets)
    ranks = {}

    for offset in range(0, len(users), batch_size):
        batch_users = users[offset : offset + batch_size]
        full_histories = [histories[user] for user in batch_users]

        item_sequences = []
        rating_sequences = []

        for user in batch_users:
            if len(histories[user]) != len(ratings[user]):
                raise ValueError(f"Items and ratings differ for user {user}.")

            item_sequences.append(
                torch.tensor(
                    histories[user][-model.max_length :],
                    dtype=torch.long,
                )
            )
            rating_sequences.append(
                torch.tensor(
                    ratings[user][-model.max_length :],
                    dtype=torch.long,
                )
            )

        inputs = pad_sequence(item_sequences, batch_first=True).to(device)
        input_ratings = pad_sequence(
            rating_sequences, batch_first=True
        ).to(device)

        last_positions = inputs.count_nonzero(dim=1) - 1
        batch_indices = torch.arange(len(batch_users), device=device)
        states = model(inputs, input_ratings)[batch_indices, last_positions]

        scores = states @ model.output_embeddings.weight[1:].T
        batch_targets = [targets[user] for user in batch_users]
        batch_ranks = rank_targets(scores, full_histories, batch_targets)
        ranks.update(zip(batch_users, batch_ranks.tolist(), strict=True))

    return ranks


def main():
    parser = argparse.ArgumentParser(
        description="Validate gSASRec with product text and historical ratings."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    histories, ratings, num_items = load_training_data_with_ratings(SPLIT_DIR)
    targets, no_ties = load_validation(SPLIT_DIR, histories, num_items)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("model_type") != "gsasrec_text_rating":
        raise ValueError("Expected a gSASRec text-and-rating checkpoint.")

    config = checkpoint["model_config"]
    if config["num_items"] != num_items:
        raise ValueError("The checkpoint catalog size differs from the split.")

    model = TextRatingGSASRec(
        embeddings_path=args.text_embeddings,
        **config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    ranks = evaluate(
        model,
        histories,
        ratings,
        targets,
        batch_size=args.batch_size,
    )
    report("Validation", list(ranks.values()))
    report(
        "Validation without boundary ties",
        [rank for user, rank in ranks.items() if user in no_ties],
    )


if __name__ == "__main__":
    main()