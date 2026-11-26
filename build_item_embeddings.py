import ast
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "processed" / "beauty_5core"
METADATA_PATH = ROOT / "data" / "raw" / "meta_Beauty.json.gz"
OUTPUT_PATH = SPLIT_DIR / "item_text_embeddings.npz"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(value):
    if isinstance(value, list):
        value = " ".join(part for part in value if isinstance(part, str))
    return " ".join(value.split()) if isinstance(value, str) else ""


def main():
    with (SPLIT_DIR / "item_ids.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    item_ids = {row["asin"]: int(row["item_id"]) for row in rows}
    if len(item_ids) != len(rows) or sorted(item_ids.values()) != list(
        range(1, len(rows) + 1)
    ):
        raise ValueError("Expected unique ASINs and consecutive item IDs starting at 1.")

    titles = [""] * (len(rows) + 1)
    descriptions = [""] * (len(rows) + 1)

    with gzip.open(METADATA_PATH, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Older Amazon metadata can use Python dictionary syntax.
                record = ast.literal_eval(line)

            item_id = item_ids.get(record.get("asin"))
            if item_id is None:
                continue

            # Keep the first nonblank value for each field in metadata file order.
            titles[item_id] = titles[item_id] or clean_text(record.get("title"))
            descriptions[item_id] = descriptions[item_id] or clean_text(
                record.get("description")
            )

    texts = [
        ". ".join(part for part in (title, description) if part)
        for title, description in zip(titles, descriptions)
    ]
    has_text = np.array([bool(text) for text in texts], dtype=bool)
    present_ids = np.flatnonzero(has_text)
    if not present_ids.size:
        raise ValueError("No catalog text matched the item ID mapping.")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    vectors = model.encode(
        [texts[item_id] for item_id in present_ids],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Row 0 is padding. Products with no text also keep a zero vector.
    embeddings = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    embeddings[present_ids] = vectors

    np.savez_compressed(
        OUTPUT_PATH,
        embeddings=embeddings,
        has_text=has_text,
        model_name=MODEL_NAME,
    )

    print(f"Saved {OUTPUT_PATH}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Products with text: {present_ids.size:,}/{len(rows):,}")
    print(f"Products without text: {len(rows) - present_ids.size:,}")


if __name__ == "__main__":
    main()