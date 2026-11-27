"""Train SASRec; select exclusively on validation, never on test."""
import argparse
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from gsasrec.data import load_training_data
from gsasrec.evaluate import SPLIT_DIR, load_validation
from sasrec.data import TrainingSampler
from sasrec.evaluate import evaluate, summarize
from sasrec.training import sasrec_loss
from sasrec.protocol import REFERENCE_COMMIT, verify_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/sasrec/seed42"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=201)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=50)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2-emb", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, help="Smoke test only; marks saved artifacts as smoke runs.")
    parser.add_argument("--max-eval-users", type=int, help="Smoke test only; use full validation for reportable runs.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.eval_batch_size,
           args.max_length, args.hidden_size, args.num_blocks, args.num_heads, args.threads) < 1:
        parser.error("Counts and sizes must be positive.")
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.l2_emb < 0:
        parser.error("Invalid dropout, learning rate, or embedding regularization.")
    if args.hidden_size % args.num_heads:
        parser.error("hidden-size must be divisible by num-heads.")
    if any(v is not None and v < 1 for v in (args.max_batches, args.max_eval_users)):
        parser.error("Smoke limits must be positive.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable.")
    if args.run_dir.exists():
        parser.error("run-dir already exists; choose a new directory to preserve earlier experiments.")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_sha256 = verify_split(args.split_dir)
    histories, num_items = load_training_data(args.split_dir)
    targets, no_ties = load_validation(args.split_dir, histories, num_items)
    if args.max_eval_users:
        targets = dict(list(targets.items())[:args.max_eval_users])
    training_items = {i for h in histories.values() for i in h}
    sampler = TrainingSampler(histories, num_items, args.max_length, args.seed)
    from sasrec.model import SASRec
    config = dict(num_items=num_items, max_length=args.max_length,
                  hidden_size=args.hidden_size, num_blocks=args.num_blocks,
                  num_heads=args.num_heads, dropout=args.dropout)
    model = SASRec(**config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    # Original sampler makes floor(user_count/batch_size) full batches per epoch.
    batches = max(1, len(histories) // args.batch_size)
    if args.max_batches:
        batches = min(batches, args.max_batches)
    smoke = args.max_batches is not None or args.max_eval_users is not None
    args.run_dir.mkdir(parents=True)
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_config": config, "smoke_run": smoke,
        "python": sys.version, "torch": str(torch.__version__), "numpy": str(np.__version__),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "manifest_sha256": manifest_sha256,
        "evaluation": "full catalog; observed items filtered; item-ID tie break",
        "reference_commit": REFERENCE_COMMIT,
    }
    (args.run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    best_ndcg, best_epoch = -math.inf, 0
    start_run = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = perf_counter()
        total_loss, total_targets = 0.0, 0
        for _ in range(batches):
            inputs, positives, negatives = (v.to(args.device) for v in sampler.sample(args.batch_size))
            optimizer.zero_grad(set_to_none=True)
            loss = sasrec_loss(model, inputs, positives, negatives, args.l2_emb)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            count = positives.count_nonzero().item()
            total_loss += loss.item() * count
            total_targets += count
        training_seconds = perf_counter() - start
        start = perf_counter()
        ranks = evaluate(model, histories, targets, args.eval_batch_size)
        result = summarize(ranks, no_ties, targets, training_items)
        record = {"epoch": epoch, "loss": total_loss / total_targets,
                  "training_seconds": training_seconds, "validation_seconds": perf_counter()-start,
                  "smoke_run": smoke, **result}
        with (args.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        ndcg = result["all_users"]["ndcg_at_10"]
        if ndcg > best_ndcg:
            best_ndcg, best_epoch = ndcg, epoch
            torch.save({"model_type": "sasrec", "model_config": config,
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch, "seed": args.seed, "smoke_run": smoke,
                        "validation": result, "metadata": metadata}, args.run_dir / "best.pt")
        elif epoch - best_epoch >= args.patience:
            break
    summary = {"best_epoch": best_epoch, "best_validation_ndcg_at_10": best_ndcg,
               "elapsed_seconds": perf_counter()-start_run, "smoke_run": smoke}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
