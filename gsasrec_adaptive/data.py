import csv
from collections import defaultdict
from pathlib import Path


def load_training_data(split_dir):
    """Return training sequences in split-file order and the catalog size."""
    split_dir = Path(split_dir)
    with (split_dir / "item_ids.csv").open(encoding="utf-8", newline="") as file:
        item_ids = sorted(int(row["item_id"]) for row in csv.DictReader(file))

    if not item_ids or item_ids != list(range(1, len(item_ids) + 1)):
        raise ValueError("Expected consecutive item IDs starting at 1.")

    histories = defaultdict(list)
    with (split_dir / "train.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            if user_id < 1 or not 1 <= item_id <= len(item_ids):
                raise ValueError(f"Invalid training IDs: user={user_id}, item={item_id}")
            histories[user_id].append(item_id)

    if not histories:
        raise ValueError("The training split is empty.")

    return dict(histories), len(item_ids)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    histories, num_items = load_training_data(
        root / "data" / "processed" / "beauty_5core"
    )
    print(f"Training users: {len(histories):,}")
    print(f"Training reviews: {sum(map(len, histories.values())):,}")
    print(f"Catalog items: {num_items:,}")