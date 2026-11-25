# Project definition and two-person ownership

## Scope and hypothesis

Predict the next reviewed product using the preceding review-event sequence.
These events are not a complete purchase history. Investigate whether product text
and historical feedback improve representations, whether adaptive semantic
negatives improve training, and whether the two changes complement one another.
Improvement and research novelty are hypotheses, not promised outcomes.

Primary metric: full-catalog NDCG@10. Secondary: Recall@10. Additional analysis:
training time, peak memory where measurable, parameter count, and performance by
training-item frequency. Do not compare our full-catalog numbers directly with
SASRec's sampled-candidate published scores.

## Ownership

Person A (provisionally M5): gSASRec + features.
- Own src/features.py, src/models/features.py, text cache, metadata audit, and EDA.
- Encode title/description/brand/categories once with a small frozen encoder.
- Implement IDs + text, IDs + historical rating, and both; time gaps optional.
- Start with projected additive fusion, then compare a small learned gate.
- Implement BPR matrix factorization as a course baseline.
- Explain feature availability, missingness, examples, and parameter behavior.

Person B (provisionally M2 Pro): gSASRec + adaptive sampling.
- Own src/sampling.py, sampling diagnostics, and sampler configurations.
- Implement uniform, semantic-only, score-based, and mixed samplers.
- Use cached text neighbors from A; an ID-only model can still use text in its
  training sampler, so explicitly report this information advantage.
- Build semi-hard candidate pools; do not deterministically choose only hardest.
- Warm up with uniform sampling, then increase the nonuniform mixture gradually.
- Own simple baseline execution and optionally FPMC after the required baselines.
- Keep negative budget constant; record realized mixture and duplicate rates.

Shared / coordinator-owned:
- src/prepare.py, src/evaluation.py, DATA_CONTRACT.md: freeze together before runs.
- Original SASRec/gSASRec port, trainer, checkpoint format: B leads; A reviews
  interface before feature work. First verify baseline parity, masking and loss.
- Combined variant: integrate only after both independent variants pass checks.
- Final notebook, related work, presentation, result interpretation: both.

Device assignments can swap based on measured free memory and pilot throughput;
chip names alone do not establish which device is faster for this workload.
Use branches codex/features and codex/adaptive-sampling; shared foundation changes
must be coordinated rather than independently rewritten.

## Milestones and acceptance checks

1. Foundation: download, preprocess, audit, validate hashes, run simple baselines.
   Verify deterministic splits, no held-out labels in train counts, and metrics.
2. Baseline: port gSASRec and reproduce its loss mathematically; smoke-test one
   batch on CPU and MPS, then establish a validation score and checkpoint parity.
3. Parallel extensions: A features; B sampling. Verify both use the frozen
   manifest and the original uniform sampler remains available unchanged.
4. Integration: run controlled ablations; choose configurations on validation.
5. Final experiment: rerun every selected model on ONE device, pinned environment,
   seeds [42,43,44], same data/config hashes. Open the test set only after freeze.
6. Deliverables: executed notebook and workbook.html; roughly 20-minute video;
   video_url.txt; later peer-grading responses.txt. Follow assignment instructions.

## Mac resource budget (initial settings, subject to pilot)

Two transformer blocks, d=64, max sequence length=50, batch=64; reduce batch if
needed on both final comparisons. Small frozen text encoder, cached embeddings,
no language-model fine-tuning. Score catalogs in chunks. Do not construct dense
all-item-by-all-item similarity matrices. Semantic neighbors: top 50 per item,
built in chunks. Persist data on disk; process gzip records as streams.

MPS is optional: CPU fallback is valid. Seeds do not guarantee identical results
across hardware/backends. Preliminary cross-device timings are not comparable.
Use the chosen final device for all reported timing and accuracy comparisons.

## Research caution for nonuniform negatives

gSASRec's calibration derivation assumes uniform sampling. A semantic/adaptive
sampler with unchanged gBCE is a heuristic variant, not a calibrated guarantee.
First test that variant explicitly. A corrected variant requires deriving and
checking sampling-probability weights under the actual replacement/exclusion
scheme; do not add an ad hoc correction and claim theoretical validity.
False negatives cannot be identified perfectly from similarity. Diagnose rather
than claiming they have been eliminated. Never use held-out interactions to mine
or exclude training negatives.

## Presentation outline (20 minutes)

Task and related work 3 min; data/EDA 3; baselines and gSASRec 3;
features 3; sampling 3; shared evaluation/results/limitations 5.
Both members explain their code, hypotheses and unsuccessful experiments.
