from __future__ import annotations

import unittest

from src.retrievers.hybrid import minmax_normalize, reciprocal_rank_fusion, weighted_score_fusion


class HybridTest(unittest.TestCase):
    def test_rrf_prefers_document_ranked_high_in_both_lists(self) -> None:
        lexical = {"q1": {"d1": 10.0, "d2": 9.0}}
        dense = {"q1": {"d2": 0.9, "d3": 0.8}}

        fused = reciprocal_rank_fusion(lexical, dense, k=60, top_k=3)

        self.assertEqual(next(iter(fused["q1"])), "d2")

    def test_weighted_fusion_uses_minmax_scores(self) -> None:
        lexical = {"q1": {"d1": 100.0, "d2": 50.0}}
        dense = {"q1": {"d2": 0.9, "d1": 0.1}}

        fused = weighted_score_fusion(lexical, dense, alpha=0.0, top_k=2)

        self.assertEqual(next(iter(fused["q1"])), "d2")

    def test_minmax_constant_scores(self) -> None:
        self.assertEqual(minmax_normalize({"d1": 2.0, "d2": 2.0}), {"d1": 1.0, "d2": 1.0})


if __name__ == "__main__":
    unittest.main()
