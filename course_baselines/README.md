# Vanshik's lecture baselines

Branch: `vanshik-course-baselines`, created from `vanshik` (`f11a36a`) and merged
with Hrithik's latest `main` lecture-material/task commit. Neither prior Vanshik
branch is modified. Independent model code is in `course_baselines/`; original
SASRec/gSASRec implementations and shared data are unchanged.

## Task and course connection

Predict the next reviewed product, treating every training review as implicit
positive feedback irrespective of star rating. These are nonsequential lecture
baselines. No product text vectors, ratings, or SASRec/gSASRec model outputs are
used. Learned BPR ID factors are its own parameters, not regenerated text vectors.

The uploaded [Recommendation lecture](../slides/recommendation.pdf) discusses
Jaccard on page 28, cosine on page 29, and BPR's positive-versus-unobserved pairwise
ranking on pages 119-126. The [lecturer's Chapter 5 code](https://cseweb.ucsd.edu/~jmcauley/pml/code/chap5.html)
illustrates sparse implicit-feedback BPR. The [original BPR paper](https://arxiv.org/abs/1205.2618)
is the objective reference. This is our transparent implementation, not a claim
of bitwise replication of the lecture's `implicit` library or the paper's SGD.

## Item-based collaborative filtering

Fit a binary training user-item matrix X. Let c(i) be the number of training
users who reviewed item i and o(i,j) their user-set intersection count, computed
from X transpose times X.

- Jaccard: o(i,j) / (c(i) + c(j) - o(i,j)).
- Binary cosine: o(i,j) / sqrt(c(i) * c(j)).
- Self-similarities are zero. Zero-overlap pairs and training-unseen items have
  zero similarity; no division by zero is performed. All nonzero neighbors are
  retained (no top-k pruning, popularity blending, or unseen-item fallback).
- Score a candidate by summing similarity to every distinct observed history
  item. Rating magnitudes and timestamp order are ignored. Both similarity
  choices are evaluated on validation; primary all-user NDCG@10 selects one.
- Test query history includes the validation interaction. The similarity matrix
  itself remains training-only: validation is never added to co-occurrence counts.

The stored CSR matrix is a learned neighborhood structure, not neural parameters.
We report its nonzero count and fit time, with zero trainable parameters.

## BPR matrix factorization

Score s(u,i) = p(u) dot q(i) + b(i). User and item factors are trainable,
initialized independently from N(0, 0.01); item biases start at zero. Padding
row zero is fixed to zero and never sampled. No global/user bias is included
because it cancels in the pairwise score difference.

For each sampled training triple (u,i,j), optimize:

    softplus(-(s(u,i)-s(u,j)))
    + lambda/2 * (||p(u)||^2 + ||q(i)||^2 + ||q(j)||^2 + b(i)^2 + b(j)^2)

The minibatch objective averages both terms over sampled triples. This is an
explicit sampled L2 penalty, not Adam weight decay or a full-table penalty.
Positive user-item pairs are bootstrapped uniformly from training interactions.
One negative is sampled uniformly from all catalog IDs outside that user's
entire training history. Duplicates across triples are allowed. Held-out labels
are never supplied to the sampler; an item later held out can be a training
negative, as with ordinary implicit-feedback training.

Predefined validation grid in `design.json`: factors {32,64} times regularization
{1e-4,1e-3}; Adam lr=0.01, batch 2048, seed 42, maximum 100 epochs, patience 10.
Each epoch has exactly 153,776 sampled positive/negative triples. The last partial
batch is included. Each configuration has its own validation-best checkpoint;
the best checkpoint/configuration is then selected on all-user validation NDCG@10.

BPR is a user-ID model: test does not update its user factors with validation
feedback. The validation item is added to the observed-item filter; parameters
remain fitted on training only. This is deliberately different from supplying
additional history to a sequential encoder or CF query and is a baseline limitation.

## Shared evaluation and leakage controls

Both models reuse the existing split, ID mappings, full catalog, observed-item
filter, item-ID tie break, and Recall@10/NDCG@10 implementations. Main reports
cover 22,363 users, secondary reports the 8,003 users without boundary ties.
Training-unseen targets are retained (51 validation and 138 test). Zero scores
and cold-item ties are resolved by the same smaller-ID-first rule, not dropped.
No rating-prediction metric or sampled-candidate evaluation is substituted.

`protocol.py` verifies every prepared split/mapping/tie CSV against the existing
manifest. Training uses only `train.csv`; test identities are parsed only after
selection is frozen in `frozen_selection.json`. Integrity verification hashes
test bytes but does not expose labels to fitting or selection. No split, IDs,
raw files, or product text embeddings are regenerated.

A training-count popularity scorer is included as the project plan's sanity
baseline. It does not add a new assigned modeling task. Selection is independent
for item-CF and BPR, with no test-guided search or refitting afterward.

## Environment and commands

Use the uv-managed project environment, not Conda base. From the repository root:

```sh
uv python install 3.12
uv venv --managed-python --python 3.12 .venv
uv pip install --python .venv/bin/python -r course_baselines/requirements-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m course_baselines.run --run-dir runs/course_baselines/seed42
```

Restore the existing data archive using the root README if necessary. Run
folders must be new. The full run sequentially fits both CF choices, fits four
BPR configurations, freezes selections, and evaluates only the selected models
and popularity on test. Source hashes, split fingerprint, versions, settings,
training/validation times, parameter counts, and complete epoch logs are saved.

Standalone evaluation of saved models:

```sh
.venv/bin/python -m course_baselines.evaluate --model item_cf --checkpoint runs/course_baselines/seed42/item_cf.npz --split test
.venv/bin/python -m course_baselines.evaluate --model bpr --checkpoint runs/course_baselines/seed42/bpr.pt --split test
```

The item-CF `.json` sidecar must remain alongside its `.npz` file. BPR metadata
is inside its checkpoint. Both re-evaluators verify the split fingerprint.

## Verification and current status

Seven baseline checks cover hand-calculated Jaccard/cosine values, history
aggregation/zero cold-item rows, BPR negative exclusion and reproducibility,
explicit loss/bias/regularization, learning and fixed test user factors, shared
ranking/filtering, and a full synthetic pipeline including checkpoint reloads.
Together with the existing nine SASRec tests, all 16 checks passed before the
full-data run. The full-data experiments are complete; results and artifacts are below. The synthetic test uses
a temporary toy dataset and is never reported as Beauty model performance.

## Completed full-data report

Both assigned lecture baselines are implemented, trained/fitted, selected on
validation, and evaluated on test. Every eligible user and all 12,101 products
were included; the run is not smoke-limited. Total experiment runtime was
63.2 seconds on CPU with four threads.

| Model | Validation Recall@10 | Validation NDCG@10 | Test Recall@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|
| Popularity sanity check | 0.016322 | 0.007941 | 0.011403 | 0.005416 |
| Item-CF (Jaccard) | 0.072218 | 0.042517 | 0.056611 | 0.032962 |
| BPR-MF (64 factors) | 0.037294 | 0.019380 | 0.024192 | 0.011652 |

Jaccard beat cosine on validation NDCG@10 (0.042517 versus 0.040728), so Jaccard
was selected before test access. The fitted matrix has 1,668,388 nonzero item
similarities; its measured fit time was about 0.069 seconds. BPR selected 64
factors, L2=0.001, epoch 19; training stopped at epoch 29. It has 2,217,926
trainable parameters and that selected configuration took about 20.1 seconds
including validation. The complete four-configuration BPR grid was:

| Factors | L2 coefficient | Selected epoch | Stopped epoch | Validation NDCG@10 |
|---:|---:|---:|---:|---:|
| 32 | 0.0001 | 4 | 14 | 0.015274 |
| 32 | 0.001 | 4 | 14 | 0.015269 |
| 64 | 0.0001 | 19 | 29 | 0.018757 |
| 64 | 0.001 | 19 | 29 | 0.019380 |

Secondary test results for the 8,003 no-boundary-tie users:

| Model | Test Recall@10, no ties | Test NDCG@10, no ties |
|---|---:|---:|
| Popularity sanity check | 0.009371 | 0.004578 |
| Item-CF (Jaccard) | 0.033612 | 0.017600 |
| BPR-MF (64 factors) | 0.017119 | 0.008150 |

Both assigned models beat the popularity sanity baseline on the primary
all-user test metrics. Item-CF also exceeds our recorded SASRec seed-42
all-user test Recall/NDCG (0.051245/0.026014), but SASRec is higher on the
no-boundary-tie subgroup (0.037986/0.019449). This population dependence should
be presented, not suppressed. These one-seed results do not establish universal
model superiority or statistical significance. Different training settings and
the nonsequential BPR test-input limitation should accompany comparisons.

All 51 training-unseen validation targets and 138 training-unseen test targets
were retained. CSV hashes matched the shared manifest. Both selected models
were reloaded through the standalone evaluator and reproduced all-user and
no-tie test metrics exactly. Original model packages and existing branch refs
were checked unchanged. No product text embeddings were loaded or regenerated.

### Saved outputs and reproduction

`results/seed42/` contains:

- `results.json`: final validation/test results and frozen selection metadata.
- `config.json`, `frozen_selection.json`, and `artifact_manifest.json`: settings,
  versions, trained-source/split fingerprints, pre-test selection timestamp,
  and file hashes/sizes.
- `item_cf.npz` plus required `item_cf.json` sidecar: selected similarity model.
- `bpr.pt`: selected BPR model; all four grid-best checkpoints, full epoch logs,
  and summaries are retained in their respective `bpr_f*_reg*` subfolders.
- `item_cf_grid.json`, `bpr_grid.json`, training popularity counts, and independent
  checkpoint reload results. The original `runs/` files remain present and ignored.

```sh
.venv/bin/python -m course_baselines.evaluate --model item_cf --checkpoint course_baselines/results/seed42/item_cf.npz --split test
.venv/bin/python -m course_baselines.evaluate --model bpr --checkpoint course_baselines/results/seed42/bpr.pt --split test
```

No test-based retuning followed. Repeating selected settings across shared seeds,
combining with Hrithik's models, and preparing the final notebook/presentation
remain team-wide follow-up work; the two assigned implementation tasks are complete.
The existing `vanshik` and `vanshik-adaptive-sampling` branches remain untouched.
