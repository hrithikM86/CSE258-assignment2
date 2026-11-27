"""Integrity checks for the existing shared experiment files."""
import hashlib
import json
from pathlib import Path

REFERENCE_COMMIT = "e3738967fddab206d6eeb4fda433e7a7034dd8b1"


def verify_split(split_dir):
    split_dir = Path(split_dir)
    manifest_bytes = (split_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    required = {"train.csv", "validation.csv", "test.csv", "user_ids.csv",
                "item_ids.csv", "boundary_ties.csv"}
    if not required.issubset(manifest["output_sha256"]):
        raise ValueError("Manifest does not cover every shared split/mapping file.")
    for name in sorted(required):
        digest = hashlib.sha256((split_dir / name).read_bytes()).hexdigest()
        if digest != manifest["output_sha256"][name]:
            raise ValueError(f"Shared split checksum mismatch: {name}")
    return hashlib.sha256(manifest_bytes).hexdigest()
