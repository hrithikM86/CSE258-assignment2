import csv
from pathlib import Path

from gsasrec.data import load_training_data


def load_training_data_with_ratings(split_dir):
    """Load aligned item and rating histories in the saved split order."""
    split_dir = Path(split_dir)
    histories, num_items = load_training_data(split_dir)
    ratings = {user_id: [] for user_id in histories}

    with (split_dir / "train.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file):
            user_id = int(row["user_id"])
            rating = float(row["rating"])

            if rating not in (1, 2, 3, 4, 5):
                raise ValueError(
                    f"Expected a whole-star rating from 1 to 5, got {rating}."
                )

            ratings[user_id].append(int(rating))

    for user_id, history in histories.items():
        if len(history) != len(ratings[user_id]):
            raise ValueError(f"Items and ratings differ for user {user_id}.")

    return histories, ratings, num_items


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    histories, ratings, num_items = load_training_data_with_ratings(
        root / "data" / "processed" / "beauty_5core"
    )

    print(f"Training users: {len(histories):,}")
    print(f"Training reviews: {sum(map(len, histories.values())):,}")
    print(f"Training ratings: {sum(map(len, ratings.values())):,}")
    print(f"Catalog items: {num_items:,}")

    example_user = next(iter(histories))
    print(f"Example user: {example_user}")
    print(f"Items:   {histories[example_user]}")
    print(f"Ratings: {ratings[example_user]}")