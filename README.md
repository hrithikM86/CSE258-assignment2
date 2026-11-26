# CSE 258 Assignment 2

We use Amazon Beauty reviews to predict the next product a user will review. We compare SASRec and gSASRec, then test whether product features or adaptive negative sampling improve gSASRec.

## Get started

From the repository root, restore the shared dataset and set up Python:

```sh
cat data/beauty_5core_bundle_parts/part-aa data/beauty_5core_bundle_parts/part-ab | tar -xzf -
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

The archive contains the raw downloads under `data/raw/` and the shared split and embeddings under `data/processed/beauty_5core/`. Use the extracted split for all experiments. `manifest.json` records the raw review and split-file checksums.

## How we prepared the data

1. We downloaded the 2014 Amazon Beauty 5-core reviews and product metadata. `inspect_data.py` audited required fields, duplicates, timestamp ties, and metadata coverage. All 198,502 reviews passed the field checks, and every reviewed product had a metadata record. [Dataset source](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
2. `preprocess.py` sorted each user's reviews by timestamp, preserving raw-file order when timestamps tied, as in SASRec's preprocessing. It held out the second-to-last review for validation and the last for test. All 22,363 users are included, with 153,776 training, 22,363 validation, and 22,363 test reviews. The script also saved user and item ID mappings and flagged ties at either split boundary. [SASRec preprocessing](https://github.com/kang205/SASRec/blob/master/data/DataProcessing.py)
3. `build_item_embeddings.py` joined product titles and descriptions to the 12,101 item IDs by ASIN. It generated normalized, 384-dimensional MiniLM embeddings for 12,094 products. In `item_text_embeddings.npz`, row number equals `item_id`, row 0 is padding, and the seven products without text have zero vectors and `has_text=False`.

## Four implementation tasks

| Task | Owner |
|---|---|
| [SASRec](https://arxiv.org/abs/1808.09781) | Vanshik |
| [gSASRec](https://arxiv.org/abs/2308.07192) | Hrithik |
| gSASRec with product text and historical numerical features | Hrithik |
| gSASRec with adaptive semantic negative sampling | Vanshik |

Both teammates use the same split, candidate catalog, and evaluator. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the experiment protocol.
