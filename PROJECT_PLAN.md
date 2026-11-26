# Project plan

Test whether product text and historical numerical features or adaptive semantic negative sampling improve next-item ranking over gSASRec on Amazon Beauty 2014 5-core. The target is the next reviewed product.

## Four implementation tasks

| Task | Owner | Scope |
|---|---|---|
| [SASRec](https://arxiv.org/abs/1808.09781) | Vanshik | Establish the sequential recommendation baseline. |
| [gSASRec](https://arxiv.org/abs/2308.07192) | Hrithik | Use uniform negative sampling and generalized binary cross-entropy (gBCE). Share the gSASRec backbone with both extensions. |
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

## Results from seed 42

Hrithik ran these experiments locally on Mac. Training used a maximum of 50 epochs and early stopping after five epochs without better validation NDCG@10. The best validation checkpoint was evaluated on test. Evaluation ranks the full catalog and filters previously observed products. Test inputs include the validation interaction and its historical rating when applicable.

### All 22,363 users

| Model | Best epoch | Validation Recall@10 | Validation NDCG@10 | Test Recall@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|---:|
| gSASRec | 15 | 0.016769 | 0.008701 | 0.011626 | 0.006171 |
| gSASRec + text | 17 | 0.017037 | 0.008330 | 0.012118 | 0.005911 |
| gSASRec + text + ratings | 17 | 0.018692 | 0.008792 | 0.013862 | 0.006455 |

### 8,003 users without ties at either split boundary

| Model | Validation Recall@10 | Validation NDCG@10 | Test Recall@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|
| gSASRec | 0.010621 | 0.005132 | 0.008747 | 0.003943 |
| gSASRec + text | 0.011121 | 0.005424 | 0.009871 | 0.004608 |
| gSASRec + text + ratings | 0.011496 | 0.005574 | 0.010121 | 0.004737 |

The feature models add projected, frozen product text vectors to history inputs. The rating version also adds trainable embeddings for historical star ratings. Text plus ratings improved all-user test Recall@10 by 19.2% and NDCG@10 by 4.6% relative to gSASRec. Text alone improved recall but reduced all-user NDCG. These are single-seed results, and consistency across seeds remains to be measured.

The implementation uses an adapted PyTorch Transformer backbone with uniform negative sampling and gBCE. SASRec and adaptive semantic sampling results from Vanshik are pending. Next, integrate those experiments with the shared evaluator and run additional seeds using fixed settings.
