import ast
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
REVIEWS_PATH = RAW_DIR / "reviews_Beauty_5.json.gz"
METADATA_PATH = RAW_DIR / "meta_Beauty.json.gz"


def read_records(path, row_counts):
    """Stream records from a gzip file and count unreadable rows."""
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            row_counts["lines"] += 1
            if not line.strip():
                row_counts["blank"] += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Older Amazon metadata can use Python dictionary syntax.
                try:
                    record = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    row_counts["malformed"] += 1
                    continue

            if not isinstance(record, dict):
                row_counts["non_object"] += 1
                continue

            row_counts["records"] += 1
            yield record


def valid_id(value):
    return isinstance(value, str) and bool(value.strip())


def valid_rating(value):
    return type(value) in (int, float) and 1 <= value <= 5


def valid_timestamp(value):
    return type(value) is int and value > 0


def has_text(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(isinstance(part, str) and part.strip() for part in value)
    return False


def print_row_counts(counts):
    print(f"Lines read: {counts['lines']:,}")
    print(f"Parsed records: {counts['records']:,}")
    print(f"Blank lines: {counts['blank']:,}")
    print(f"Malformed lines: {counts['malformed']:,}")
    print(f"Parsed values that were not records: {counts['non_object']:,}")


def print_distribution(label, counts):
    if not counts:
        print(f"{label}: no valid IDs")
        return

    values = counts.values()
    print(
        f"{label}: min={min(values):,}, "
        f"median={median(values):,.1f}, "
        f"mean={sum(values) / len(values):,.1f}, "
        f"max={max(values):,}"
    )


def print_share(label, count, total):
    percentage = f" ({count / total:.1%})" if total else ""
    print(f"{label}: {count:,}/{total:,}{percentage}")


def audit_split_boundaries(timestamps_by_user, total_users):
    """Count timestamp ties at the boundaries of the last-two split."""
    eligible_users = 0
    train_validation_ties = 0
    validation_test_ties = 0
    both_ties = 0

    for timestamps in timestamps_by_user.values():
        # At least one training review and two held-out reviews are required.
        if len(timestamps) < 3:
            continue

        eligible_users += 1
        last_train_time, validation_time, test_time = sorted(timestamps)[-3:]

        train_validation_tied = last_train_time == validation_time
        validation_test_tied = validation_time == test_time

        train_validation_ties += train_validation_tied
        validation_test_ties += validation_test_tied
        both_ties += train_validation_tied and validation_test_tied

    either_tie = train_validation_ties + validation_test_ties - both_ties

    print("\nPLANNED TRAIN / VALIDATION / TEST SPLIT")
    print(f"Users eligible for the split: {eligible_users:,}")
    print(f"Users with fewer than three valid reviews: {total_users - eligible_users:,}")
    print("Boundary percentages are based on eligible users.")
    print_share("Train–validation tie", train_validation_ties, eligible_users)
    print_share("Validation–test tie", validation_test_ties, eligible_users)
    print_share("Ties at both boundaries", both_ties, eligible_users)
    print_share("Tie at either boundary", either_tie, eligible_users)
    print_share("No ties at either boundary", eligible_users - either_tie, eligible_users)


def audit_reviews():
    row_counts = Counter()
    missing = Counter()
    invalid = Counter()
    seen_records = set()
    duplicate_rows = 0
    complete_rows = 0

    reviews_per_user = Counter()
    reviews_per_product = Counter()
    user_product_counts = Counter()
    user_timestamp_counts = Counter()
    rating_counts = Counter()
    timestamps_by_user = defaultdict(list)

    field_checks = {
        "reviewerID": valid_id,
        "asin": valid_id,
        "overall": valid_rating,
        "unixReviewTime": valid_timestamp,
    }

    for review in read_records(REVIEWS_PATH, row_counts):
        # Compare full records independently of dictionary key order.
        signature = json.dumps(review, sort_keys=True, separators=(",", ":"))
        if signature in seen_records:
            duplicate_rows += 1
        else:
            seen_records.add(signature)

        valid_fields = set()
        for field, check in field_checks.items():
            value = review.get(field)
            if value is None:
                missing[field] += 1
            elif check(value):
                valid_fields.add(field)
            else:
                invalid[field] += 1

        user_id = review.get("reviewerID")
        product_id = review.get("asin")
        timestamp = review.get("unixReviewTime")

        if len(valid_fields) == len(field_checks):
            complete_rows += 1
            timestamps_by_user[user_id].append(timestamp)

        if "reviewerID" in valid_fields:
            reviews_per_user[user_id] += 1
        if "asin" in valid_fields:
            reviews_per_product[product_id] += 1
        if {"reviewerID", "asin"} <= valid_fields:
            user_product_counts[user_id, product_id] += 1
        if {"reviewerID", "unixReviewTime"} <= valid_fields:
            user_timestamp_counts[user_id, timestamp] += 1
        if "overall" in valid_fields:
            rating_counts[review["overall"]] += 1

    repeated_pairs = [count for count in user_product_counts.values() if count > 1]
    tied_groups = [
        (user_id, count)
        for (user_id, _), count in user_timestamp_counts.items()
        if count > 1
    ]

    print("\nREVIEWS")
    print_row_counts(row_counts)
    print(f"Records with all four required fields valid: {complete_rows:,}")
    for field in field_checks:
        print(f"  {field}: {missing[field]:,} missing or null, {invalid[field]:,} invalid")

    print("\nDUPLICATES AND TIMESTAMP TIES")
    print(f"Exact duplicate rows beyond the first copy: {duplicate_rows:,}")
    print(f"User–product pairs with multiple rows: {len(repeated_pairs):,}")
    print(f"Extra rows across those pairs: {sum(c - 1 for c in repeated_pairs):,}")
    print(f"Users with at least one timestamp tie: {len({u for u, _ in tied_groups}):,}")
    print(f"Tied user–timestamp groups: {len(tied_groups):,}")
    print(f"Review rows in tied groups: {sum(c for _, c in tied_groups):,}")
    print("Pair and timestamp counts include exact duplicate rows.")

    print("\nINTERACTION COUNTS")
    print(f"Unique users with a valid ID: {len(reviews_per_user):,}")
    print(f"Unique products with a valid ID: {len(reviews_per_product):,}")
    print_distribution("Reviews per user", reviews_per_user)
    print_distribution("Reviews per product", reviews_per_product)

    print("\nVALID RATING COUNTS")
    for rating, count in sorted(rating_counts.items()):
        print(f"  {rating:g}: {count:,}")

    audit_split_boundaries(timestamps_by_user, len(reviews_per_user))
    return set(reviews_per_product)


def audit_metadata(reviewed_products):
    row_counts = Counter()
    invalid_id_rows = 0
    matched_products = set()
    products_with_title = set()
    products_with_description = set()

    for record in read_records(METADATA_PATH, row_counts):
        product_id = record.get("asin")
        if not valid_id(product_id):
            invalid_id_rows += 1
            continue
        if product_id not in reviewed_products:
            continue

        matched_products.add(product_id)
        if has_text(record.get("title")):
            products_with_title.add(product_id)
        if has_text(record.get("description")):
            products_with_description.add(product_id)

    total = len(reviewed_products)
    coverage = {
        "Metadata record": matched_products,
        "Nonblank title": products_with_title,
        "Nonblank description": products_with_description,
        "Nonblank title or description": products_with_title | products_with_description,
    }

    print("\nMETADATA")
    print_row_counts(row_counts)
    print(f"Metadata records with missing or invalid ASIN: {invalid_id_rows:,}")
    print(f"Coverage across {total:,} unique reviewed products with valid ASINs")
    for label, products in coverage.items():
        print_share(f"  {label}", len(products), total)


def main():
    for path in (REVIEWS_PATH, METADATA_PATH):
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")

    reviewed_products = audit_reviews()
    audit_metadata(reviewed_products)


if __name__ == "__main__":
    main()