# CSE 258 Assignment 2

We use Amazon Beauty reviews to predict the next product a user will review. We compare SASRec and gSASRec, then test whether product features or adaptive negative sampling improve gSASRec.

## Get started

From the repository root, restore the shared dataset and set up Python:

```sh
cat data/beauty_5core_bundle_parts/part-aa data/beauty_5core_bundle_parts/part-ab | tar -xzf -
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

The archive contains the raw downloads under `data/raw/` and the shared split and embeddings under `data/processed/beauty_5core/`. Use the extracted split for all experiments. `manifest.json` records the raw review and split-file checksums.

## How we prepared the data

1. We downloaded the 2014 Amazon Beauty 5-core reviews and product metadata. `inspect_data.py` audited required fields, duplicates, timestamp ties, and metadata coverage. All 198,502 reviews passed the field checks, and every reviewed product had a metadata record. [Dataset source](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
2. `preprocess.py` sorted each user's reviews by timestamp, preserving raw-file order when timestamps tied, as in SASRec's preprocessing. It held out the second-to-last review for validation and the last for test. All 22,363 users are included, with 153,776 training, 22,363 validation, and 22,363 test reviews. The script also saved user and item ID mappings and flagged ties at either split boundary. [SASRec preprocessing](https://github.com/kang205/SASRec/blob/master/data/DataProcessing.py)
3. `build_item_embeddings.py` joined product titles and descriptions to the 12,101 item IDs by ASIN. It generated normalized, 384-dimensional MiniLM embeddings for 12,094 products. In `item_text_embeddings.npz`, row number equals `item_id`, row 0 is padding, and the seven products without text have zero vectors and `has_text=False`.

## Four implementation tasks

| Task | Owner |
|---|---|
| [SASRec](https://arxiv.org/abs/1808.09781) | Vanshik |
| [gSASRec](https://arxiv.org/abs/2308.07192) | Hrithik |
| gSASRec with product text and historical numerical features | Hrithik |
| gSASRec with adaptive semantic negative sampling | Vanshik |

Both teammates use the same split, candidate catalog, and evaluator. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the experiment protocol.

## Results from seed 42

These experiments ran locally on Mac. The gSASRec experiments used a maximum of 50 epochs and early stopping after five epochs without better validation NDCG@10. SASRec used its reference defaults (maximum 201 epochs, patience 20), stopped at epoch 98, and selected epoch 78. The best validation checkpoint was evaluated on test. Evaluation ranks the full catalog and filters previously observed products. Test inputs include the validation interaction and its historical rating when applicable.

### All 22,363 users

| Model | Best epoch | Validation Recall@10 | Validation NDCG@10 | Test Recall@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|---:|
| SASRec | 78 | 0.064839 | 0.033872 | 0.051245 | 0.026014 |
| gSASRec | 15 | 0.016769 | 0.008701 | 0.011626 | 0.006171 |
| gSASRec + text | 17 | 0.017037 | 0.008330 | 0.012118 | 0.005911 |
| gSASRec + text + ratings | 17 | 0.018692 | 0.008792 | 0.013862 | 0.006455 |

### 8,003 users without ties at either split boundary

| Model | Validation Recall@10 | Validation NDCG@10 | Test Recall@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|
| SASRec | 0.045733 | 0.024203 | 0.037986 | 0.019449 |
| gSASRec | 0.010621 | 0.005132 | 0.008747 | 0.003943 |
| gSASRec + text | 0.011121 | 0.005424 | 0.009871 | 0.004608 |
| gSASRec + text + ratings | 0.011496 | 0.005574 | 0.010121 | 0.004737 |

The feature models add projected, frozen product text vectors to history inputs. The rating version also adds trainable embeddings for historical star ratings. Text plus ratings improved all-user test Recall@10 by 19.2% and NDCG@10 by 4.6% relative to gSASRec. Text alone improved recall but reduced all-user NDCG. These are single-seed results, and consistency across seeds remains to be measured.

The implementation uses an adapted PyTorch Transformer backbone with uniform negative sampling and gBCE. SASRec is implemented under `sasrec/`, with the full seed-42 results recorded in [sasrec/results/seed42.json](sasrec/results/seed42.json). Adaptive semantic sampling remains pending. Next, implement adaptive semantic sampling and run additional seeds using fixed settings. SASRec and gSASRec use different backbone and training settings; their comparison is not a controlled loss-only ablation.

## SASRec reproduction (Vanshik)

The `sasrec/` package is a PyTorch port of Kang and McAuley's
[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781).
Implementation details follow the [authors' original TensorFlow code](https://github.com/kang205/SASRec/tree/e3738967fddab206d6eeb4fda433e7a7034dd8b1),
reference commit `e3738967fddab206d6eeb4fda433e7a7034dd8b1`.
It does not subclass or reuse the adapted gSASRec backbone.
The upstream license is retained in `sasrec/LICENSE-SASRec.txt`; the original
attention module credits Kyubyong Park's Transformer implementation (June 2017).

### Isolated environment

From the repository root, use uv-managed Python rather than Conda:

```sh
uv python install 3.12
uv venv --managed-python --python 3.12 .venv
uv pip install --python .venv/bin/python -r sasrec/requirements-lock.txt
```

Restore the existing data archive using the command in Get started if it has
not already been extracted. Do not rerun `preprocess.py` or
`build_item_embeddings.py`. Training and evaluation verify all six prepared
split/mapping/tie CSV checksums against `manifest.json`; evaluation also checks
that the checkpoint was trained with the same manifest.

### Model and training fidelity

- Fixed-length, left-padded histories; zero is padding, newest item is last.
- One learned item table shared by sequence inputs and prediction. Input item
  vectors are scaled by the square root of the hidden dimension; learned
  position vectors are added before dropout.
- Explicit causal scaled dot-product attention. Following the reference code,
  queries use normalized states, keys/values use unnormalized states, and the
  attention residual adds normalized queries. There is no output projection.
- Two pointwise dense layers, ReLU, dropout after both layers, and a residual
  from the normalized feed-forward input. Padding is masked after each block,
  followed by final layer normalization. Layer normalization uses epsilon 1e-8.
- One uniform negative per valid next-item target, excluding every item in that
  user's complete training history. Users are sampled with replacement and
  each epoch has floor(training users / batch size) full batches. Held-out
  item identities are not supplied to the sampler.
- Positive plus negative binary cross-entropy, averaged over valid targets.
  Stable softplus replaces the reference's log-sigmoid arithmetic. Optional
  embedding L2 uses the TensorFlow half-squared-norm convention; padding row
  zero is fixed to zero. The default L2 coefficient is zero.
- Glorot-uniform embedding/projection initialization, zero linear biases, Adam
  with learning rate 0.001, betas (0.9, 0.98), and epsilon 1e-8.
- Beauty defaults: length 50, hidden dimension 50, two blocks, one attention
  head, dropout 0.5, batch size 128; maximum 201 epochs and patience 20.

SASRec trains item-ID and position embeddings. It does not load or regenerate
product title/description embeddings. The supplied MiniLM vectors remain
available unchanged for the separate adaptive sampling experiment.

### Evaluation and limits of reproduction

The model and training method follow the original SASRec reference. The
experiment uses this team's fixed Beauty 2014 5-core split and full-catalog
evaluation. The paper's Beauty dataset has different counts and its metrics
rank one target against 100 sampled negatives, so its numerical results are
not directly comparable. We do not claim to reproduce the paper's metric values.

The SASRec evaluator only adapts left-padded model inputs; it reuses
`gsasrec.evaluate.rank_targets` and `metrics_at_10` unchanged. It scores all
12,101 products, filters the full observed history (including items outside the
truncated input), and resolves tied scores by smaller item ID. Validation uses
training history; test adds the validation interaction. Reports include all
users, users without boundary ties, and target counts unseen in training.
Unlike the original script's periodic evaluation of both held-out sets, this
trainer evaluates validation each epoch and selects the best NDCG@10 checkpoint;
test evaluation is a separate explicit command.

This is a semantic PyTorch port, not a bitwise TensorFlow reproduction: RNG
streams, framework kernels, stable BCE arithmetic, and Adam epsilon placement
can differ. The sampler is synchronous for reproducibility rather than using
three asynchronous workers. Tests compare deterministic forward operations
against an independent NumPy transcription; TensorFlow parity has not been run.

### Commands and artifacts

```sh
.venv/bin/python -m unittest discover -s tests -v

# Small integration check; artifacts are explicitly marked smoke_run=true.
.venv/bin/python -m sasrec.train --epochs 2 --max-batches 2 --max-eval-users 256 --run-dir runs/sasrec/smoke

# Initial full-validation baseline, using the defaults above.
.venv/bin/python -m sasrec.train --device cpu --run-dir runs/sasrec/seed42

# Re-evaluate the selected checkpoint on validation.
.venv/bin/python -m sasrec.evaluate --checkpoint runs/sasrec/seed42/best.pt --split validation --output runs/sasrec/seed42/validation.json

# Only after configuration selection is complete:
.venv/bin/python -m sasrec.evaluate --checkpoint runs/sasrec/seed42/best.pt --split test --output runs/sasrec/seed42/test.json
```

`--device mps` is supported on Apple Silicon; `cuda` is supported when available.
CPU is the tested reference device for this port. Different devices need not
produce identical results. Run directories must be new to prevent overwrites.
Each directory contains `config.json` (arguments, versions, parameter count,
reference commit and split fingerprint), `metrics.jsonl` (validation metrics and
training/evaluation times), `best.pt`, and `summary.json`. Run artifacts are
ignored by Git. Smoke metrics must not be presented as final experiment results.

The adaptive semantic sampling extension remains a separate, unimplemented task.

### Saved baseline artifacts

The completed seed-42 baseline is included in this branch:

- `sasrec/results/seed42.json`: configuration, versions, validation/test results,
  split fingerprint, checkpoint checksum, and runtime.
- `sasrec/results/seed42_best.pt`: selected epoch-78 checkpoint (about 2.5 MB).
- `sasrec/results/seed42_metrics.jsonl`: all 98 epoch records.

Re-evaluate the saved baseline after restoring the existing shared data:

```sh
.venv/bin/python -m sasrec.evaluate --checkpoint sasrec/results/seed42_best.pt --split validation
.venv/bin/python -m sasrec.evaluate --checkpoint sasrec/results/seed42_best.pt --split test
```

The original `runs/` directory and the environment remain ignored. Only the
selected reportable checkpoint and epoch log are retained here; smoke artifacts
are excluded. See PROJECT_PLAN.md for the comparison audit and proposed sampling
experiment. Adaptive sampling has not been implemented or trained.

## Isolated negative-sampling experiments

On `vanshik-adaptive-sampling`, Hrithik's files were copied into
`gsasrec_adaptive/` and the original `gsasrec/` files remain unchanged.
The uniform control and a fixed semantic mixture have completed full seed-42
validation runs. The fixed mixture was worse on the primary all-user metric;
there is no adaptive improvement claim. The actual adaptive rule awaits the
user's curriculum-versus-confidence decision. See
[the implementation and experiment report](gsasrec_adaptive/README.md).
