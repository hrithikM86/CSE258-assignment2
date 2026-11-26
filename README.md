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

## Results from seed 42

These experiments ran locally on Mac. Training used a maximum of 50 epochs and early stopping after five epochs without better validation NDCG@10. The best validation checkpoint was evaluated on test. Evaluation ranks the full catalog and filters previously observed products. Test inputs include the validation interaction and its historical rating when applicable.

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
