# CSE 258 Assignment 2

A step-by-step study of product features and adaptive negative sampling for gSASRec.

## Current stage

Local environment setup. Starter implementation and preliminary results have been
removed. We will implement and review preprocessing together before creating the
shared train/validation/test data.

## Team

- Person A (M5): gSASRec with product text and historical numerical features.
- Person B (M2 Pro): gSASRec with adaptive semantic negative sampling.
- Both: common baseline, evaluation, combined experiments, and presentation.

Both people will use the full same dataset and saved splits. Final selected
experiments will be rerun on one device. See PROJECT_PLAN.md.

## Environment

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

The lock file records the packages installed during local setup. Verify package
compatibility on the second Mac. requirements.txt lists the project dependencies.
Select .venv/bin/python as the VS Code interpreter.

Downloaded source data may already exist locally under data/raw/. Data and the
virtual environment are excluded from Git. No model implementation is present yet.
