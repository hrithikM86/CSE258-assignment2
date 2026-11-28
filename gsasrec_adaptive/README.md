# Isolated gSASRec sampling experiments

Branch: `vanshik-adaptive-sampling`, created from `vanshik` at `f11a36a`.
All original `gsasrec/*.py` files are preserved; their checksums are recorded
in `upstream_files.json`. All six Python source files were copied. Imports
in copied entry points refer to the isolated package. The original copied
trainer uses a separate output directory. Use `experiment.py` for controlled
sampling comparisons; use `evaluate_run.py` for its checkpoints.

## Completed implementation

- The model and data loader are unchanged copies of Hrithik's implementation.
- The copied loss accepts an optional sampler; the formula is unchanged.
- Deterministic cosine-neighbor cache from the existing normalized, frozen
  384-dimensional product embeddings. No text model is loaded or trained.
- Up to 100 neighbors per product, excluding padding, itself, and products
  without text. Similarity ties are resolved by ascending item ID.
- Uniform control: 32 random negatives per target, as in Hrithik's original.
- Fixed semantic mixture: 16 random plus 16 draws uniformly with replacement
  from the training target's 100 nearest text neighbors.
- Products without text fall back to uniform sampling for their semantic slots.
- Full-budget uniform draws occur in both modes; semantic replacements use a
  separate CPU RNG (seed + 104729). The model/dropout RNG stream is unchanged
  by semantic sampling. Initialization fingerprints and actual sampling counts
  are recorded. Duplicate negatives are allowed in both modes.
- Negative eligibility matches the original control: exclude padding and the
  current positive only, not the user's other training interactions. This can
  create false negatives, but prevents silently changing eligibility only for
  the semantic arm. Training never reads validation/test target identities to
  construct negative pools.
- CPU training with identical settings in all arms: 128 hidden dimensions,
  length 200, two Transformer layers, one head, dropout 0.5, Adam lr=0.001,
  batch 64, gBCE t=0.5, 32 negatives, maximum 50 epochs, patience 5, seed 42.
- Same full-catalog evaluator and checkpoint selection by validation NDCG@10.
  Test adds the validation interaction to history. Model architecture and
  output item embeddings are unchanged; text is used only for negative mining.
- Split and cache provenance, versions, parameter count, per-epoch metrics,
  sampler/fallback counts, runtime, and explicit smoke-run flags are logged.

## Adaptive rules implemented

Both rules were fixed in `design.json` before their full training runs. Both
use three uniform-only warmup epochs, the same 32-negative budget and catalog,
current training target as the text-neighbor anchor, unchanged gBCE, and a
maximum semantic allocation of 50%. Counts are rounded with floor(32*f+0.5).
At least one uniform slot is always retained; missing text falls back to uniform.

- `curriculum`: semantic fraction is zero for epochs 1-3, then
  `0.5 * min(1, (epoch-3)/10)`. This reaches 16 semantic/16 uniform slots at
  epoch 13. It is an epoch schedule, not confidence adaptation.
- `confidence`: after epoch 3, first draw the same 32 uniform candidates as
  the control. Compute detached current-model positive/negative scores from
  the existing training forward pass. Let
  `c = sigmoid(positive_logit - max(uniform_negative_logits))`.
  The per-target semantic fraction is `0.5 * clamp(2*c-1, 0, 1)`: targets below
  their strongest uniform negative receive no semantic slots; confident targets
  receive up to 16. These score margins are not calibrated probabilities.
  Decisions are detached from autograd; final candidate scores receive normal
  gradients. The signal uses training data only, no validation/test labels,
  and requires extra dot products but no extra sequence-model forward pass.

The confidence probe is a candidate selector, not an additional 32-negative
loss: only the final 32 candidates contribute to gBCE. Both adaptive modes
consume the same model RNG stream as the uniform control and use the isolated
semantic RNG for neighbor draws. Warmup loss and gradients match the original
uniform implementation exactly in tests.

Popularity sampling is an optional separate ablation, not part of this first
experiment. It does not automatically reduce popularity bias. Mixed sampling
with unchanged gBCE is a heuristic: the original calibration theorem assumes
uniform negatives. No calibration/CTR guarantee or novelty claim is made.

## Reproduce

Use the existing uv-managed `.venv` and `sasrec/requirements-lock.txt`. Restore
Hrithik's shared archive using the root README if necessary. Never regenerate
splits or text embeddings.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m gsasrec_adaptive.neighbors
.venv/bin/python -m gsasrec_adaptive.experiment --mode uniform --run-dir runs/adaptive/uniform_seed42
.venv/bin/python -m gsasrec_adaptive.experiment --mode fixed --run-dir runs/adaptive/fixed_seed42
.venv/bin/python -m gsasrec_adaptive.experiment --mode curriculum --run-dir runs/adaptive/curriculum_seed42
.venv/bin/python -m gsasrec_adaptive.experiment --mode confidence --run-dir runs/adaptive/confidence_seed42
.venv/bin/python -m gsasrec_adaptive.evaluate_run --checkpoint runs/adaptive/uniform_seed42/best.pt --split validation --output runs/adaptive/uniform_seed42/validation.json
```

Run directories must not already exist. Output: `config.json`, `metrics.jsonl`,
`summary.json`, and `best.pt`. Full runs have no max-batches/max-eval-users limits.
A separate one-batch check was marked `smoke_run=true` and is not reportable.
Evaluate the final selected configuration on test only after design/validation
selection is complete. Final test evaluation is complete for all four configurations after
validation-only selection was frozen.

The initial full uniform/fixed runs overlap in wall-clock time on the same CPU.
Their raw timings therefore are not a clean computational-efficiency comparison;
use a subsequent isolated timing measurement before making speed claims.
## Final full-run report (seed 42)

All four runs are complete: full training data, all 22,363 validation users,
full-catalog evaluation, no smoke limits. Each best checkpoint was selected
exclusively by validation NDCG@10, then reloaded to verify its validation metrics.
Overall selection (uniform) and adaptive selection (confidence) were frozen
before test evaluation in `results/frozen_selection.json`. No settings were
changed from test scores.

| Sampler | Best epoch | Stop epoch | Val Recall@10 | Val NDCG@10 | Test Recall@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 15 | 20 | 0.016813 | 0.008715 | 0.011626 | 0.006171 |
| Fixed semantic | 4 | 9 | 0.013102 | 0.006723 | 0.010419 | 0.005388 |
| Epoch curriculum | 4 | 9 | 0.013683 | 0.007072 | 0.010508 | 0.005400 |
| Confidence-driven | 15 | 20 | 0.016590 | 0.008377 | 0.011716 | 0.006122 |

Uniform remains best on the primary all-user validation and test NDCG@10.
Confidence-driven was strongest among the adaptive variants: test Recall@10
increased 0.77% (two extra hits among 22,363 users), while test NDCG@10 decreased
0.80%. This is a mixed single-seed result, not an overall improvement or
statistical-significance claim. Both curriculum and fixed sampling were worse
on all-user test recall and NDCG. No further tuning used the observed test set.

The curriculum stopped at epoch 9, before its epoch-13 maximum allocation.
Its selected checkpoint was epoch 4, requesting two semantic negatives out of
32. We retained patience 5 rather than extending it after seeing results.
Confidence sampling varied its allocation per training target; the epoch logs
record actual semantic/fallback counts and confidence statistics. This score
signal is not a calibrated interaction probability.

All four model initialization hashes match. Both adaptive warmup trajectories
match the uniform control exactly for three epochs. Each full epoch uses
4,205,184 negative loss terms. The original gSASRec source checksums, prepared
CSV checksums, and text embeddings remain unchanged. The uniform control's
test scores reproduce Hrithik's documented values to the reported precision.
Its small validation difference (one additional hit) has no established cause.

Twenty tests passed, covering original-file preservation, copied-model/loss/
gradient equivalence, semantic membership and budget, missing-text fallback,
isolated RNG, reproducibility, cosine neighbors/ties/cache, synthetic held-out
history handling, curriculum endpoints, confidence allocation without selector
gradients, exact warmup equivalence, and the existing SASRec checks.

### Saved artifacts

`results/` contains each configuration, every epoch record, summary, validation
and test results, comparison deltas, and the pre-test selection record. All four
selected checkpoints are saved under `results/checkpoints/{mode}_seed42.pt`;
SHA256 values are in `results/comparison.json`. The original files in `runs/`
remain present and ignored, along with smoke artifacts and the regenerable
neighbor cache. Saving artifacts does not delete or retire the experiment.

```sh
.venv/bin/python -m gsasrec_adaptive.evaluate_run --checkpoint gsasrec_adaptive/results/checkpoints/confidence_seed42.pt --split test
.venv/bin/python -m gsasrec_adaptive.evaluate_run --checkpoint gsasrec_adaptive/results/checkpoints/uniform_seed42.pt --split test
```

Uniform/fixed elapsed times were 264.7/131.2 seconds; curriculum/confidence
were 130.5/288.9 seconds. The first pair overlapped on CPU, while the adaptive
runs were sequential. They also stopped after different epoch counts. These
raw timings do not establish a fair computational-efficiency improvement.
Neighbor construction took about 2.9 seconds in the initial cache build.

Limitations: one seed, unchanged gBCE used heuristically under nonuniform
sampling, potentially false negatives (including other observed training items,
consistently allowed in all arms), one semantic anchor/neighborhood setting,
and no calibrated-probability or novelty guarantee. Some secondary no-tie
subgroup scores improve, but the predeclared all-user primary metric governs
selection. All subgroup metrics and unseen-target counts (51 validation, 138
test) are preserved in JSON.

The implementation and initial evaluation are complete. Additional shared
seeds are needed before robustness claims. Item-CF and BPR lecture baselines
are separate next tasks; no additional models or hyperparameter search were
started here.
