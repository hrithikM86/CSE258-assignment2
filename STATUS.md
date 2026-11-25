# Project status

Foundation prepared November 25, 2026. gSASRec training and the two extensions
are not implemented yet; no transformer performance claim is made.

## Actual dataset audit

- 198,502 review events; 22,363 users; 12,101 items.
- 153,776 training events; 22,363 validation targets; 22,363 test targets.
- Metadata: 12,094 titles, 11,163 descriptions, 10,003 brands;
  category text available for all 12,101 items.
- 33 items have zero training interactions; retain and report separately.
- 77,402 adjacent event pairs have equal timestamps. Chronological ordering
  within a tied timestamp is unobserved; raw item ID is our deterministic rule.
- Manifest SHA256: 2af9f8ee1d2e5b2051eaf4199fd65accb5ec8b310ee7fdba06d91638ac6772ea

## Transfer to the other Mac

Send artifacts/beauty2014-contract-v1.zip through a private file transfer.
Its SHA256 is:
79b0e008f49a530009c468a6171117bdaab51d30bd2823d5e00f58149484eacc

On the other Mac, clone the repository, enter it, and run:

```sh
shasum -a 256 /path/to/beauty2014-contract-v1.zip
unzip /path/to/beauty2014-contract-v1.zip
(cd data/processed && shasum -a 256 -c SHA256SUMS)
python3 -m unittest discover -s tests -v
python3 -m src.baselines --split validation
```

Both users should obtain the same baseline accuracy, although execution time will
differ. Raw files are approximately 139 MB combined; the processed bundle is
4,719,345 bytes. No raw reviews or user-level processed data are published to GitHub.

## Next work

Freeze the model and trainer interface together; implement and validate the
uniform gSASRec baseline before running independent feature/sampler extensions.
Choose Python version and dependency lock, measure CPU/MPS pilot throughput on
both devices, then choose the one final reporting device. Branch ownership and
full milestones are in PROJECT_PLAN.md.
