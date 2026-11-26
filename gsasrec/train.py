import argparse
import random
from pathlib import Path
from time import perf_counter

import torch

from gsasrec.data import load_training_data
from gsasrec.model import GSASRec
from gsasrec.training import gsasrec_loss, make_batch


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Train gSASRec on Beauty 5-core.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "runs" / "gsasrec" / "latest.pt"
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this PyTorch environment")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    histories, num_items = load_training_data(ROOT / "data/processed/beauty_5core")
    sequences = [history for history in histories.values() if len(history) >= 2]

    model_config = {"num_items": num_items, "max_length": 200}
    model = GSASRec(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss in epoch {epoch}, batch {batch_number + 1}"
                )
            loss.backward()
            optimizer.step()

            total_loss += loss_value * target_count
            total_targets += target_count

        print(
            f"Epoch {epoch}: loss={total_loss / total_targets:.4f}, "
            f"targets={total_targets:,}, time={perf_counter() - start:.1f}s"
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs,
            "seed": args.seed,
        },
        args.checkpoint,
    )
    print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()