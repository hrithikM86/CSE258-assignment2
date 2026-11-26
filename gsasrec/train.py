import argparse
import random
from math import isfinite, log2
from pathlib import Path
from time import perf_counter

import torch

from gsasrec.data import load_training_data
from gsasrec.evaluate import SPLIT_DIR, evaluate, load_validation
from gsasrec.model import build_model
from gsasrec.training import gsasrec_loss, make_batch


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Train gSASRec on Beauty 5-core.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "runs" / "gsasrec" / "best.pt"
    )
    parser.add_argument("--text-embeddings", type=Path)
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

    histories, num_items = load_training_data(SPLIT_DIR)
    validation_targets, _ = load_validation(SPLIT_DIR, histories, num_items)
    sequences = [history for history in histories.values() if len(history) >= 2]
    if not sequences:
        raise ValueError("Training needs at least one history with two items.")

    model_config = {"num_items": num_items, "max_length": 200}
    model = build_model(model_config, args.text_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # The existing evaluator runs on CPU.
    validation_model = (
    model if device.type == "cpu" else build_model(model_config, args.text_embeddings))
    best_ndcg = float("-inf")
    best_epoch = 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        model.train()
        rng.shuffle(sequences)
        total_loss = 0.0
        total_targets = 0

        for batch_number, offset in enumerate(range(0, len(sequences), args.batch_size)):
            if args.max_batches is not None and batch_number >= args.max_batches:
                break

            batch = sequences[offset : offset + args.batch_size]
            inputs, targets = make_batch(batch, model.max_length)
            target_count = targets.count_nonzero().item()

            optimizer.zero_grad(set_to_none=True)
            loss = gsasrec_loss(model, inputs.to(device), targets.to(device), num_items)
            loss_value = loss.item()
            if not isfinite(loss_value):
                raise RuntimeError(
                    f"Non-finite loss in epoch {epoch}, batch {batch_number + 1}"
                )
            loss.backward()
            optimizer.step()

            total_loss += loss_value * target_count
            total_targets += target_count

        if validation_model is not model:
            validation_model.load_state_dict(model.state_dict())

        ranks = evaluate(validation_model, histories, validation_targets, args.batch_size)
        top_10_ranks = [rank for rank in ranks.values() if rank <= 10]
        recall = len(top_10_ranks) / len(ranks)
        ndcg = sum(1.0 / log2(rank + 1) for rank in top_10_ranks) / len(ranks)

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
                    "model_config": model_config,
                    "model_type": "gsasrec_text" if args.text_embeddings is not None else "gsasrec",
                    "model_state_dict": validation_model.state_dict(),
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