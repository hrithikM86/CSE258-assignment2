import json
import math
from pathlib import Path
import tempfile
import unittest
from src.prepare import prepare
from src.evaluation import rank_items, metrics
from src.baselines import run

class ContractTests(unittest.TestCase):
    def test_rank_filter_ties_and_metrics(self):
        ranked = rank_items({0: 100, 1: 2, 2: 2, 3: 3}, {3})
        self.assertEqual(ranked, [1, 2])
        self.assertAlmostEqual(metrics(ranked, 2)["ndcg@10"], 1/math.log2(3))
        self.assertEqual(metrics(ranked, 9)["recall@10"], 0)

    def test_deterministic_split_and_no_heldout_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{"reviewerID": "u", "asin": str(i),
                        "unixReviewTime": i, "overall": 4} for i in range(1, 6)]
            records.append({**records[0], "unixReviewTime": 99})
            reviews, meta = root/"reviews", root/"metadata"
            reviews.write_text("".join(json.dumps(r)+"\n" for r in reversed(records)))
            meta.write_text("{'asin': '1', 'title': 'cream', 'categories': [['skin']]}\n")
            prepare(reviews, meta, root/"a")
            prepare(reviews, meta, root/"b")
            for p in (root/"a").iterdir():
                self.assertEqual(p.read_bytes(), (root/"b"/p.name).read_bytes())
            row = json.loads((root/"a/sequences.jsonl").read_text())
            self.assertEqual([e["item"] for e in row["train"]], [1, 2, 3])
            self.assertEqual(row["validation"]["item"], 4)
            self.assertEqual(row["test"]["item"], 5)
            counts = json.loads((root/"a/train_item_counts.json").read_text())
            self.assertEqual(counts["4"], 0)
            self.assertEqual(counts["5"], 0)
            summaries = run(root/"a", root/"runs", "validation")
            self.assertEqual(set(summaries), {"popularity", "item_cf", "markov"})

if __name__ == "__main__":
    unittest.main()
