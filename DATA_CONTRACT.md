# Frozen data and evaluation contract v1

## Sharing between devices

Both devices receive the full identical processed bundle. Do NOT divide users,
products, or interactions between teammates: that would confound comparisons.
One producer downloads and preprocesses; transfer data/processed privately and
verify every file against SHA256SUMS. Both may download raw files independently
only if their SHA256 digests match the manifest. Share text embeddings and item
neighbor arrays with item-map hash, encoder revision, text-field definition,
normalization and dimension recorded. Raw data and caches stay outside Git.

## Deterministic preprocessing

Use the Beauty 2014 five-core reviews, not 2018 All Beauty. Parse JSON reviews and
Python-literal metadata safely (json.loads / ast.literal_eval, never eval).
Collapse repeated (user,item) events to the earliest timestamp; count removals.
Sort by (timestamp, raw item ID) to resolve ties reproducibly. Retain users with
at least five unique items after deduplication; no new iterative core is imposed.
IDs are lexicographically sorted raw IDs, indexed from 1; 0 is padding.
The retained catalog is fixed from the whole selected corpus (transductive),
including products with zero training interactions. Report those separately.
This preprocessing can differ from paper releases: do not claim exact replication.

For each user: all but last two -> train; second-last -> validation; last -> test.
Validation context = train. Test context = train + validation, never test.
Per-user time order is not a global time cutoff; disclose this limitation and
same-day timestamp ambiguity. At training position t the label is the next train
item; numeric/text inputs describe only context events, never target feedback.

Training negatives exclude the observed training positives for that user only;
do not look at held-out events to choose negatives. Real unobserved items may be
future positives. This ambiguity is inherent, not grounds to consult test labels.

Product text = title, description, brand, categories. Do not use target reviews,
helpfulness votes, aggregate ratings, salesRank, or also_bought graphs as product
features. Metadata is a crawl-time snapshot, not guaranteed historical; disclose
this assumption. Missing text gets a missing indicator/zero representation, not
model-specific item removal. Drop URLs/HTML from text before encoding if needed.

Historical rating is available on context events only. Time gaps use consecutive
context timestamps; never the unknown future target timestamp. Fit scalers on
training examples only. Price remains optional and is not enabled in v1.

## Metrics

For each validation/test user, score the SAME full catalog; remove padding and
all items in that user's observed context. Target is a novel item due to pair
deduplication. Ties break by ascending mapped item ID. No sampled test negatives.
Recall@10 = 1 if target in top 10, else 0.
NDCG@10 = 1/log2(rank+1) if rank <=10, else 0.
Average over the identical user set; store per-user scores for paired bootstrap.
Report mean/std over seeds and paired user bootstrap difference intervals with
seed results distinguished from user-level uncertainty. Tail bins are computed
from training counts only: 0, 1-5, 6-20, 21+ interactions. Publish support per bin;
merge/report inconclusive bins if too small, never tune boundaries on test gains.

## Cross-device experiment identity

Each result must record git commit, manifest SHA, config SHA, seed, Python/torch
versions, device/backend, trainable/total parameters, elapsed training time, best
validation epoch and score. Use independent RNG streams for model init, batches
and sampling. Keep final split, seed list, architecture and negative budget fixed.
Model-specific hyperparameters receive the same number of validation trials.
