# Experiment matrix and stopping rules

Required course baselines: popularity, item Jaccard CF, BPR-MF, Markov transitions.
Research baselines: SASRec (original BCE/one negative), gSASRec (gBCE/more negatives).
The SASRec vs gSASRec baseline intentionally differs in loss and negative count;
all gSASRec extension comparisons hold negative count fixed (initially 32).

IDs:
B0 popularity; B1 item_cf; B2 bpr_mf; B3 markov;
R0 sasrec; R1 gsasrec;
A1 gsasrec_text; A2 gsasrec_rating; A3 gsasrec_features;
S1 gsasrec_semantic; S2 gsasrec_adaptive; S3 gsasrec_mixed;
C1 gsasrec_features_mixed.

A1: ID + text with additive projection. A2: ID + historical rating.
A3: both, then compare gated vs additive fusion only if validation warrants it.
S1: uniform/semantic mixture without score adaptivity.
S2: uniform/model-score mixture without semantic neighbors.
S3: uniform/semantic/model-score mixture, warmed up gradually.
All S variants retain a uniform component and are labeled heuristic gBCE variants
until any nonuniform-sampling correction is independently derived and verified.
C1 combines the selected A3 and S3 definitions without additional test tuning.

Pilot: one seed, small validation-only run to check runtime and bugs.
Development: initial seed 42, at most 6 validation configurations per family;
use the same selection budget and document failed trials.
Final: seeds 42,43,44, all selected models rerun on one device. Select checkpoint
by validation NDCG@10, fixed max epochs and patience; no test-based selection.
Report per-user output, NDCG@10 and Recall@10 overall and by training-frequency
bin, mean/std over seeds, runtime, and parameter count. Use the same catalog.

The one-epoch smoke run, synthetic fixtures and preliminary cross-device results
are engineering checks, not reportable evidence of model superiority.
No results are supplied or invented in this scaffold.
