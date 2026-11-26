# Project plan

Test whether product text and historical numerical features or adaptive semantic negative sampling improve next-item ranking over gSASRec on Amazon Beauty 2014 5-core. The target is the next reviewed product.

## Four implementation tasks

| Task | Owner | Scope |
|---|---|---|
| SASRec | Vanshik | Establish the sequential recommendation baseline. |
| gSASRec | Hrithik | Use uniform negative sampling and generalized binary cross-entropy (gBCE). Share the gSASRec backbone with both extensions. |
| gSASRec + product features | Hrithik | Fuse catalog text embeddings and numerical features derived from the user's available history, starting with historical ratings. |
| gSASRec + adaptive semantic sampling | Vanshik | Use the shared embeddings to identify similar candidates. Keep a uniform component and introduce semantic or model-score negatives while holding the negative budget fixed. |

Coordinate the shared trainer, evaluator, and relevant course baselines. The feature and sampling extensions are separate experiments.

## Data and evaluation

- Use the shared split, ID mappings, and embeddings described in [README.md](README.md). Validation uses training history. Test uses training history plus the validation interaction.
- Use the same full catalog and observed-item filtering across models. Report NDCG@10 as the primary metric and Recall@10 as the secondary metric. Select configurations and checkpoints on validation before evaluating test.
- Report results for all eligible users and a secondary result for the 8,003 users with no ties at either split boundary. Ties affect 14,360 users at one or both boundaries. Raw-file order gives a reproducible split. The true order of those tied reviews is unknown.
- Retain targets whose products never appeared in training and report their counts alongside the main results. The current split has 51 such validation targets and 138 test targets.
- Fit feature statistics using training data and derive history features only from interactions available at prediction time. Use training information to select negatives. The original gBCE calibration assumes uniform sampling, so using unchanged gBCE with adaptive sampling is a heuristic.
- Develop on separate Macs and rerun selected experiments on one device with shared seeds. Compare accuracy by training item frequency and report parameter counts and runtime.

## Order of work

1. Verify the shared dataset and build the evaluator. Use popularity as a sanity check.
2. Implement SASRec and gSASRec, then add the feature and sampling extensions.
3. Compare models and ablations on validation. Freeze settings, rerun selected models, and evaluate on test.
4. Prepare the workbook and presentation.

## References

- [Amazon Beauty data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- [SASRec](https://arxiv.org/abs/1808.09781)
- [gSASRec](https://arxiv.org/abs/2308.07192)
