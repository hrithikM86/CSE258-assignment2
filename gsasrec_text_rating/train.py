import argparse
import random
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch

from gsasrec.evaluate import SPLIT_DIR, load_validation, metrics_at_10
from gsasrec_text_rating.data import load_training_data_with_ratings
from gsasrec_text_rating.evaluate import evaluate
from gsasrec_text_rating.model import TextRatingGSASRec
from gsasrec_text_rating.training import gsasrec_loss, make_batch


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Train gSASRec with product text and historical ratings."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--text-embeddings",
        type=Path,
        default=SPLIT_DIR / "item_text_embeddings.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs" / "gsasrec_text_rating" / "best_50ep.pt",
    )
    args = parser.parse_args()

    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("--epochs, --patience, and --batch-size must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this PyTorch environment")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    histories, ratings, num_items = load_training_data_with_ratings(SPLIT_DIR)
    targets, _ = load_validation(SPLIT_DIR, histories, num_items)
    users = [user for user in histories if len(histories[user]) >= 2]
    if not users:
        raise ValueError("Training needs at least one history with two items.")

    model_config = {"num_items": num_items, "max_length": 200}
    model = TextRatingGSASRec(
        embeddings_path=args.text_embeddings,
        **model_config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_ndcg = float("-inf")
    best_epoch = 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        model.train()
        rng.shuffle(users)
        total_loss = 0.0
        total_targets = 0

        for batch_number, offset in enumerate(range(0, len(users), args.batch_size)):
            if args.max_batches is not None and batch_number >= args.max_batches:
                break

            batch_users = users[offset : offset + args.batch_size]
            inputs, input_ratings, target_items = make_batch(
                [histories[user] for user in batch_users],
                [ratings[user] for user in batch_users],
                model.max_length,
            )
            target_count = target_items.count_nonzero().item()

            optimizer.zero_grad(set_to_none=True)
            loss = gsasrec_loss(
                model,
                inputs.to(device),
                input_ratings.to(device),
                target_items.to(device),
                num_items,
            )
            loss_value = loss.item()
            if not isfinite(loss_value):
                raise RuntimeError(
                    f"Non-finite loss in epoch {epoch}, batch {batch_number + 1}"
                )

            loss.backward()
            optimizer.step()
            total_loss += loss_value * target_count
            total_targets += target_count

        ranks = evaluate(
            model, histories, ratings, targets, batch_size=args.batch_size
        )
        recall, ndcg = metrics_at_10(list(ranks.values()))

        print(
            f"Epoch {epoch:02d}: loss={total_loss / total_targets:.4f}, "
            f"validation Recall@10={recall:.6f}, NDCG@10={ndcg:.6f}, "
            f"time={perf_counter() - start:.1f}s",
            flush=True,
        )

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = epoch
            torch.save(
                {
                    "model_type": "gsasrec_text_rating",
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "epochs": best_epoch,
                    "seed": args.seed,
                    "validation_recall_at_10": recall,
                    "validation_ndcg_at_10": ndcg,
                },
                args.checkpoint,
            )
        elif epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch}.")
            break

    print(f"Best epoch: {best_epoch}, validation NDCG@10={best_ndcg:.6f}")
    print(f"Best checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()