# Semantic and adaptive extensions to gSASRec

Research question: Can product semantics, historical feedback, and adaptive
negative sampling improve next-item ranking over gSASRec at practical training cost?
Dataset: Amazon Beauty 2014 five-core reviews plus category metadata.

Start with [PROJECT_PLAN.md](PROJECT_PLAN.md), [DATA_CONTRACT.md](DATA_CONTRACT.md),
and [EXPERIMENTS.md](EXPERIMENTS.md).

## Initial runnable foundation

These commands use only the Python standard library:

```sh
python3 -m unittest discover -s tests -v
python3 -m src.download
python3 -m src.prepare --reviews data/raw/reviews_Beauty_5.json.gz --metadata data/raw/meta_Beauty.json.gz
python3 -m src.baselines --split validation
```

The downloader retrieves only the two Beauty files. Preparation generates stable
IDs, fixed per-user chronological splits, product text, training-only item counts,
a data audit, and checksums. Baselines currently implemented: popularity,
item-based Jaccard collaborative filtering, and first-order Markov transitions.
They run on validation by default. Do not run test experiments during development.

## Training environment (both Macs)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip freeze > environment-lock.txt
```

Use the same Python minor version and a shared resolved dependency lock for final
runs. requirements.txt is a provisional dependency range, not a reproducibility
lock. Model training and text encoding are the next implementation milestones;
this starter does not yet include a validated gSASRec implementation.

Both people use the ENTIRE processed dataset, not separate halves. Copy the exact
processed bundle from its designated producer and verify its SHA256SUMS. Do not
commit raw reviews, processed user histories, embeddings, or checkpoints to GitHub.

Research references:
- SASRec: https://arxiv.org/abs/1808.09781
- gSASRec: https://arxiv.org/abs/2308.07192
- Data: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- gSASRec code: https://github.com/asash/gsasrec

The root course folder has another Git remote. This directory is intended to be
an independent repository for https://github.com/hrithikM86/CSE258-assignment2.
