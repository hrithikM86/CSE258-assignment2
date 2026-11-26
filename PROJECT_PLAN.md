# Project plan

## Goal

Test whether product text, historical numerical features, and adaptive semantic
negative sampling improve next-item ranking over gSASRec. Improvement and novelty
are not established results.

## Dataset

Amazon Beauty 2014 five-core reviews plus product metadata. Predict the next
reviewed product, rather than claiming to observe all purchases.

## Shared data protocol (to implement together)

Both people use the full identical processed dataset, never separate halves.
Generate the split once, share the saved files and ID mappings, and verify hashes.
For each user, train on all but the last two events, validate on the second-last,
and test on the last. Resolve timestamp ties deterministically and document the
ambiguity. Fit preprocessing statistics using training data; never expose target
ratings/reviews to the model or use held-out outcomes for negative sampling.

## Responsibilities

Person A (M5): metadata audit and exploratory analysis; cached frozen product-text
embeddings; historical rating features; feature fusion and ablations; BPR baseline.
Time gaps are optional, price is deferred until suitability is checked.

Person B (M2 Pro): uniform and adaptive semantic/model-score samplers; sampling
ablations; popularity, item-CF and Markov baselines. Coordinate the common gSASRec
trainer with Person A before building either extension.

gSASRec calibration assumes uniform negative sampling. Nonuniform variants using
unchanged gBCE must be labeled heuristic unless a valid correction is derived.
Semantically similar unobserved products are not guaranteed true negatives.

Both: shared evaluator, integration, related work, result interpretation, notebook
and recorded presentation.

## Evaluation

Same saved splits, catalog, filtering, and evaluator for every model. Primary:
full-catalog NDCG@10; secondary: Recall@10. Report frequency-based analysis,
parameter counts, and computational cost. Select configurations on validation.
After freezing experiments, rerun selected models on one device with a pinned
environment and shared seed list, then evaluate test. Cross-device development
results and timings are preliminary.

## Steps

1. Verify the local environment.
2. Inspect raw reviews and metadata together.
3. Implement preprocessing, audit, and shared train/validation/test files.
4. Implement course baselines and verify SASRec/gSASRec.
5. Develop features and sampling independently; run validation ablations.
6. Integrate extensions and perform final single-device comparisons.
7. Produce workbook.html, approximately 20-minute presentation, video_url.txt,
   and the later peer-grading submission according to the assignment statement.

## References

- Data: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- SASRec: https://arxiv.org/abs/1808.09781
- gSASRec: https://arxiv.org/abs/2308.07192
