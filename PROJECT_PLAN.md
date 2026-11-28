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

The implementation uses an adapted PyTorch Transformer backbone with uniform negative sampling and gBCE. SASRec is implemented under `sasrec/`, with the full seed-42 results recorded in [sasrec/results/seed42.json](sasrec/results/seed42.json). Adaptive sampling is complete on the separate [vanshik-adaptive-sampling](https://github.com/hrithikM86/CSE258-assignment2/tree/vanshik-adaptive-sampling) branch. Next, consolidate the team experiments and repeat selected settings across seeds. SASRec and gSASRec use different backbone and training settings; their comparison is not a controlled loss-only ablation.

## Baseline comparison audit and sampling proposal

Historical proposal from the SASRec baseline stage. The sampling work is now complete on [vanshik-adaptive-sampling](https://github.com/hrithikM86/CSE258-assignment2/tree/vanshik-adaptive-sampling); its final report supersedes the proposal below.
At the proposal stage, no adaptive/semantic/popularity sampler had been implemented. All Vanshik work
remains on `vanshik`; use separate commits for the sampling work after the
SASRec checkpoint commit. A separate branch is optional for parallel development,
not required for this experiment or desired under the current branch constraint.

### What the current comparison establishes

Both models target the next reviewed product and use the shared catalog,
prepared split, observed-item filtering, tie-breaking, and ranking metrics.
SASRec uses an input adapter for fixed-length left padding while calling the
existing shared ranking/metric functions. Its test Recall@10 is 0.051245 and
NDCG@10 is 0.026014. Hrithik's documented original gSASRec results are 0.011626
and 0.006171 respectively. This is a descriptive, single-seed comparison.
The gSASRec checkpoints/logs were not in the checkout, so those reported results
have not been independently reproduced here.

Inspection found these differences, all of which can affect the comparison:

| Setting | SASRec port | Current gSASRec |
|---|---|---|
| Backbone | Explicit original-reference attention/FFN operations | PyTorch TransformerEncoder adaptation |
| Item embeddings | Shared input/output | Separate input/output |
| Hidden size / maximum length | 50 / 50 | 128 / 200 |
| Trainable parameters | 633,600 | 3,323,136 |
| Padding / positions | Fixed-width left padding | Variable-width right padding |
| Negatives per positive | 1 | 32 |
| Negative exclusion | Whole user training history | Current positive only |
| User sampling | With replacement | Shuffled full user pass |
| Loss | BCE | gBCE, t=0.5 |
| Adam beta2 / batch size | 0.98 / 128 | 0.999 / 64 |
| Maximum epochs / patience | 201 / 20 | 50 / 5 |

The current gBCE positive term `beta * softplus(-positive_logit)` is
algebraically equivalent to `-log(sigmoid(positive_logit)**beta)` in the official
loss. Its beta expression also simplifies to the official expression. No loss
formula discrepancy was found in this inspection. Separate output embeddings
are also supported by the official gSASRec PyTorch repository; that choice alone
is not an implementation bug. The full architecture/training comparison is
still not a controlled reproduction of the gSASRec paper.

Do not infer that SASRec generally beats gSASRec from these numbers. Before any
loss-only claims, run BCE versus gBCE on an identical backbone, sampler,
negative budget, eligible-item policy, optimization protocol, and shared seeds.
Hrithik's original implementation and reported numbers are preserved.

The shared CSV checksums were verified again after evaluation; the existing
text embedding SHA256 is
`e84bfe6408a1761e5ca0fed8c494a145a23bc959ae93e864465871c682c1fcd1`,
matching its value before implementation. No split or text embeddings were
regenerated. The selected SASRec checkpoint is epoch 78, full run, not smoke.

### Interpretation of the suggested extension

Research question: under the same gSASRec model, loss, negative budget, training
protocol, and evaluator, does changing which negatives are sampled improve
next-item ranking, and at what computational cost?

A negative is an unobserved item used as a training contrast; it is not a
verified dislike. Semantic negatives are catalog products whose supplied text
vectors are similar to a training target or available history. These can be
informative but can also be false negatives. Semantics and model difficulty
are distinct: a similar item is not necessarily highly scored by the model.

The original pasted LLM proposal mentioned MovieLens-1M/Steam and variable
negative counts. Those suggestions do not replace the established Beauty split,
shared embeddings, or fixed-budget project protocol. Start with the existing
32-negative budget. Varying both counts and mixture proportions would confound
the first comparison.

Recommended staged experiment, subject to design agreement:

1. Establish/reproduce the uniform gSASRec control with saved configuration,
   checkpoint, and metrics. Coordinate with Hrithik before changing shared code.
2. Compare uniform-only sampling with a fixed uniform-plus-semantic mixture.
   Reuse the existing frozen MiniLM embeddings, with row-to-item mapping intact.
3. Compare that fixed mixture with an epoch-dependent mixture that gradually
   introduces more semantic negatives while retaining uniform support. Call it
   a curriculum; it is not model-confidence adaptation. If confidence-driven
   adaptation is selected instead, define the score statistic and update rule
   before implementation rather than inventing one after seeing results.
4. Treat popularity sampling as a separate optional ablation, not an automatic
   debiasing component. Use training counts only and report head/tail accuracy
   if making popularity-bias claims. Increasing the probability of popular
   negatives is different from inverse-popularity sampling; neither guarantees
   less bias.
5. Tune mixtures and neighbor settings only on validation, then freeze them
   before a final test evaluation. Repeat selected configurations with shared
   seeds and the same hardware. Include neighbor-precomputation time, training
   time, parameter count, and full-catalog Recall@10/NDCG@10.

Decisions still open: static versus confidence-driven adaptation, semantic
anchor (training target versus observed prefix), neighbor count, mixture
schedule/proportions, popularity distribution, and negative eligibility policy.
Use exactly the same eligibility policy in every sampler control. Excluding
all training-history items in an extension while excluding only the current
positive in its control would introduce another experimental difference.
Never consult held-out target identities to build or filter training negatives.
For items without text, use an explicitly documented uniform fallback.

### Calibration, feasibility, and prior work

The gSASRec theorem is derived for uniform negative sampling. Replacing that
with a semantic/popularity mixture does not retain its calibration guarantee.
An initial unchanged-gBCE extension must be described as a heuristic. A justified
importance-weighted estimator would require computing the actual proposal
probability (including exclusions, fallback, and any duplicate handling) and a
separate derivation; simply substituting a different alpha is not sufficient.
Keep that derivation outside the minimum implementation scope unless agreed.

The minimum evaluation should focus on ranking and runtime. A sigmoid score
is not automatically a calibrated probability. Any calibration metric needs a
clearly defined event/candidate population and evaluation weighting; naive ECE
on sampled negatives can be misleading. Do not promise calibrated preferences
or CTR from next-reviewed-item data.

The paper's MovieLens accuracy/runtime improvement and award are real, but do
not predict improvement on Beauty or justify the proposed sampler. Hard,
adaptive, mixed, popularity-based and false-negative-aware sampling already
have prior literature. This is a reasonable course-project extension and
ablation study; novelty or guaranteed accuracy improvements are not established.
A limited literature check found the following starting points, not an exhaustive
novelty review:

- [gSASRec paper, especially Sections 2.1 and 5](https://arxiv.org/html/2308.07192)
- [Official PyTorch loss and configuration documentation](https://github.com/asash/gSASRec-pytorch)
- [RecSys award record](https://recsys.acm.org/best-papers/)
- [Generating Negative Samples for Sequential Recommendation](https://arxiv.org/abs/2208.03645)
- [Evaluating Performance and Bias of Negative Sampling in Large-Scale Sequential Recommendation Models](https://arxiv.org/abs/2410.17276)
- [UFNRec: Utilizing False Negative Samples for Sequential Recommendation](https://arxiv.org/abs/2208.04116)

## Completed Vanshik lecture tasks

Both item-based CF and BPR matrix factorization are complete on
`vanshik-course-baselines`. The branch starts from `vanshik` and merges main's
lecture materials; existing branches and model source are preserved.
Validation selected Jaccard and BPR with 64 factors/L2=0.001/epoch 19.
Test Recall/NDCG are 0.056611/0.032962 for CF and 0.024192/0.011652 for BPR.
See [the full lecture-baseline report](course_baselines/README.md) for all trials,
subgroup results, objective definitions, saved models and checksums. Both models
fit training interactions only and use the shared full-catalog evaluator.
These are one-seed results; shared-seed robustness, team result consolidation,
and the final workbook/presentation remain follow-up work. Hrithik's separate
lecture model responsibilities are FPMC and feature-based factorization machines.
