import argparse
import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import mkdtemp


PROJECT_DIR = Path(__file__).resolve().parent
RAW_PATH = PROJECT_DIR / "data" / "raw" / "reviews_Beauty_5.json.gz"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "processed" / "beauty_5core"


def load_reviews(path):
    histories = defaultdict(list)
    users_per_product = Counter()
    seen_pairs = set()

    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                review = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error

            if not isinstance(review, dict):
                raise ValueError(f"Expected a review object on line {line_number}")

            user = review.get("reviewerID")
            asin = review.get("asin")
            rating = review.get("overall")
            timestamp = review.get("unixReviewTime")

            if not isinstance(user, str) or not user.strip():
                raise ValueError(f"Invalid reviewer ID on line {line_number}")
            if not isinstance(asin, str) or not asin.strip():
                raise ValueError(f"Invalid ASIN on line {line_number}")
            if type(rating) not in (int, float) or not 1 <= rating <= 5:
                raise ValueError(f"Invalid rating on line {line_number}")
            if type(timestamp) is not int or timestamp <= 0:
                raise ValueError(f"Invalid timestamp on line {line_number}")

            pair = (user, asin)
            if pair in seen_pairs:
                raise ValueError(f"Repeated user–product pair on line {line_number}")

            seen_pairs.add(pair)
            histories[user].append((timestamp, asin, rating))
            users_per_product[asin] += 1

    if not histories:
        raise ValueError("The input contains no reviews.")
    if min(map(len, histories.values())) < 5 or min(users_per_product.values()) < 5:
        raise ValueError("The input does not satisfy the expected 5-core rule.")

    return histories, users_per_product


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_split(histories, products, output_dir):
    # Keep 0 available as a padding ID for the recommendation model.
    user_ids = {user: index for index, user in enumerate(sorted(histories), start=1)}
    item_ids = {asin: index for index, asin in enumerate(sorted(products), start=1)}

    splits = {"train": [], "validation": [], "test": []}
    boundary_rows = []
    ties = Counter()

    for user, user_id in user_ids.items():
        # Python's stable sort preserves raw-file order when timestamps match.
        history = sorted(histories[user], key=lambda review: review[0])
        train_time, validation_time, test_time = (
            review[0] for review in history[-3:]
        )
        train_validation_tie = train_time == validation_time
        validation_test_tie = validation_time == test_time

        ties["train_validation"] += train_validation_tie
        ties["validation_test"] += validation_test_tie
        ties["either_boundary"] += train_validation_tie or validation_test_tie

        boundary_rows.append(
            (user_id, int(train_validation_tie), int(validation_test_tie))
        )

        for split_name, reviews in (
            ("train", history[:-2]),
            ("validation", history[-2:-1]),
            ("test", history[-1:]),
        ):
            for timestamp, asin, rating in reviews:
                splits[split_name].append(
                    (user_id, item_ids[asin], timestamp, rating)
                )

    counts = {name: len(rows) for name, rows in splits.items()}
    total_reviews = sum(map(len, histories.values()))
    if sum(counts.values()) != total_reviews:
        raise RuntimeError("The split did not preserve the total review count.")
    if counts["validation"] != len(user_ids) or counts["test"] != len(user_ids):
        raise RuntimeError("Expected one validation and one test review per user.")

    training_items = {row[1] for row in splits["train"]}
    unseen_targets = {
        name: sum(row[1] not in training_items for row in splits[name])
        for name in ("validation", "test")
    }

    files = []
    for name, rows in splits.items():
        filename = f"{name}.csv"
        write_csv(
            output_dir / filename,
            ("user_id", "item_id", "timestamp", "rating"),
            rows,
        )
        files.append(filename)

    write_csv(
        output_dir / "user_ids.csv",
        ("reviewer_id", "user_id"),
        user_ids.items(),
    )
    write_csv(
        output_dir / "item_ids.csv",
        ("asin", "item_id"),
        item_ids.items(),
    )
    write_csv(
        output_dir / "boundary_ties.csv",
        ("user_id", "train_validation_tie", "validation_test_tie"),
        boundary_rows,
    )
    files.extend(("user_ids.csv", "item_ids.csv", "boundary_ties.csv"))

    summary = {
        "source": RAW_PATH.name,
        "source_sha256": sha256(RAW_PATH),
        "split_rule": "Per-user last two, ordered by timestamp then raw-file order",
        "users": len(user_ids),
        "products": len(item_ids),
        "reviews": total_reviews,
        "split_rows": counts,
        "boundary_ties": dict(ties),
        "users_without_boundary_ties": len(user_ids) - ties["either_boundary"],
        "products_unseen_in_training": len(item_ids) - len(training_items),
        "targets_unseen_in_training": unseen_targets,
        "output_sha256": {name: sha256(output_dir / name) for name in files},
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2, sort_keys=True)
        destination.write("\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build the shared Beauty 5-core split.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if not RAW_PATH.is_file():
        raise SystemExit(f"Missing input file: {RAW_PATH}")
    if output_dir.exists():
        raise SystemExit(f"Output already exists. Choose another --output-dir: {output_dir}")

    histories, products = load_reviews(RAW_PATH)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(mkdtemp(prefix=".beauty-split-", dir=output_dir.parent))

    try:
        summary = create_split(histories, products, staging_dir)
        staging_dir.rename(output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print(f"Saved split to {output_dir}")
    for name, count in summary["split_rows"].items():
        print(f"{name}: {count:,} reviews")
    for name, count in summary["targets_unseen_in_training"].items():
        print(f"{name} targets unseen in training: {count:,}")
    print(f"Users without split-boundary ties: {summary['users_without_boundary_ties']:,}")
    print("Share the output directory and compare manifest.json checksums.")


if __name__ == "__main__":
    main()