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

## Pending adaptive design

The user was asked to choose an epoch curriculum versus a model-confidence
rule. That required design choice remains pending. No adaptive rule is selected
or implemented yet; the uniform control and fixed mixture are independent
preparatory experiments. Recommended curriculum: retain 32 negatives total,
start with random draws, and ramp the semantic allocation to 50%. The exact
warmup/ramp must be fixed before training that arm. A curriculum should not be
presented as confidence-driven adaptation.

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
.venv/bin/python -m gsasrec_adaptive.evaluate_run --checkpoint runs/adaptive/uniform_seed42/best.pt --split validation --output runs/adaptive/uniform_seed42/validation.json
```

Run directories must not already exist. Output: `config.json`, `metrics.jsonl`,
`summary.json`, and `best.pt`. Full runs have no max-batches/max-eval-users limits.
A separate one-batch check was marked `smoke_run=true` and is not reportable.
Evaluate the final selected configuration on test only after design/validation
selection is complete. No real test-set evaluation has been performed for the
new experiments yet.

The initial full uniform/fixed runs overlap in wall-clock time on the same CPU.
Their raw timings therefore are not a clean computational-efficiency comparison;
use a subsequent isolated timing measurement before making speed claims.
## Full-run report (seed 42)

Both runs completed with all training users and all 22,363 validation users,
without max-batches or max-eval-users limits. Each configuration used its own
best validation checkpoint; reloading each checkpoint reproduced its metrics.

| Sampler | Selected epoch | Stopped epoch | Validation Recall@10 | Validation NDCG@10 |
|---|---:|---:|---:|---:|
| Uniform, 32 random | 15 | 20 | 0.016813 | 0.008715 |
| Fixed, 16 random + 16 semantic | 4 | 9 | 0.013102 | 0.006723 |

The fixed mixture reduced all-user Recall@10 by 22.1% and NDCG@10 by 22.9%
relative to this uniform control. This is a negative single-seed result, not
proof that semantic sampling can never help. On the 8,003 no-boundary-tie users,
uniform Recall/NDCG were 0.010621/0.005132 and fixed were 0.010246/0.005673.
The all-user primary metric governs selection; the secondary subgroup is not
used to reverse the conclusion after seeing the results.

The uniform checkpoint closely reproduces Hrithik's documented seed-42 result:
both selected epoch 15. His documented Recall/NDCG were 0.016769/0.008701;
this run gives 0.016813/0.008715 (one additional recall hit). The cause of this
small numerical difference has not been established. Exact source-model/loss/
gradient equivalence tests pass; no claim of identical framework RNG trajectories
or numerical backend execution is made across the original devices.

Uniform elapsed time was 264.7 seconds and fixed elapsed time was 131.2 seconds.
These include different early-stopping lengths and overlapping CPU execution,
so they must not be interpreted as a sampling-efficiency improvement.

Tracked `results/` contains per-run configuration, every epoch record, summary,
validation re-evaluation, and `comparison.json`. Checkpoints remain in ignored
`runs/adaptive/{uniform,fixed}_seed42/best.pt`; their hashes are recorded in the
comparison. The shared SASRec checkpoint from the previous branch remains
tracked under `sasrec/results/`. The neighbor cache remains ignored and is
reproducible from the unchanged source embeddings.

Seventeen tests passed: original file preservation, exact copied-model/loss/
gradient equivalence, fixed budget and valid semantic membership, missing-text
fallback, isolated sampler RNG and reproducibility, invalid neighbor rejection,
cosine similarity/tie handling and cache reuse, synthetic validation/test history
construction, plus the nine existing SASRec tests. The real test set has not
been evaluated for these new configurations.

Next: obtain the user's curriculum-versus-confidence decision, implement and
test that rule, run its full matched validation experiment, freeze the selected
configuration, and then perform the final test comparison. No adaptive result
or accuracy improvement is claimed at this stage.
