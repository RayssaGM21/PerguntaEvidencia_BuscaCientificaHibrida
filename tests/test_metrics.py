from __future__ import annotations

import unittest

from src.metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


class MetricsTest(unittest.TestCase):
    def test_binary_relevance_metrics(self) -> None:
        ranking = ["d3", "d1", "d2", "d4"]
        qrels = {"d1": 1, "d2": 1}

        self.assertAlmostEqual(precision_at_k(ranking, qrels, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranking, qrels, 3), 1.0)
        self.assertAlmostEqual(reciprocal_rank(ranking, qrels), 0.5)
        self.assertAlmostEqual(average_precision(ranking, qrels), ((1 / 2) + (2 / 3)) / 2)

    def test_ndcg_uses_graded_relevance(self) -> None:
        ranking = ["d1", "d2"]
        qrels = {"d1": 1, "d2": 2}

        self.assertGreater(ndcg_at_k(ranking, qrels, 2), 0.0)
        self.assertLess(ndcg_at_k(ranking, qrels, 2), 1.0)


if __name__ == "__main__":
    unittest.main()
