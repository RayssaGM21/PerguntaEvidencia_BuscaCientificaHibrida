from __future__ import annotations

import unittest

import numpy as np

from src.retrievers.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs, batch_size: int, show_progress_bar: bool):
        return np.asarray([len(pair[1]) for pair in pairs], dtype=float)


class RerankerTest(unittest.TestCase):
    def test_reranker_reorders_existing_candidates_only(self) -> None:
        queries = {"q1": "query"}
        corpus = {
            "d1": {"title": "short", "text": ""},
            "d2": {"title": "much longer title", "text": "with text"},
            "d3": {"title": "not candidate", "text": ""},
        }
        candidates = {"q1": {"d1": 2.0, "d2": 1.0}}
        reranker = CrossEncoderReranker(model=FakeCrossEncoder())

        ranking, timing = reranker.rerank(queries, corpus, candidates, candidate_pool=2)

        self.assertEqual(list(ranking["q1"]), ["d2", "d1"])
        self.assertNotIn("d3", ranking["q1"])
        self.assertEqual(timing.total_pairs, 2)


if __name__ == "__main__":
    unittest.main()
