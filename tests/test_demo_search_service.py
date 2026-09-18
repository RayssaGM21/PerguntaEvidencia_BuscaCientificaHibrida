from __future__ import annotations

import unittest

from src.demo.search_service import (
    find_matching_query_id,
    format_results,
    shared_document_positions,
)


class DemoSearchServiceTest(unittest.TestCase):
    def test_find_matching_query_by_id_or_normalized_text(self) -> None:
        queries = {"q1": "How are knowledge graphs used in education?"}

        self.assertEqual(find_matching_query_id("q1", queries), "q1")
        self.assertEqual(find_matching_query_id("how are knowledge graphs used in education", queries), "q1")
        self.assertIsNone(find_matching_query_id("unseen query", queries))

    def test_format_results_marks_relevant_documents(self) -> None:
        ranking = {"d1": 2.0, "d2": 1.0}
        corpus = {
            "d1": {"title": "Relevant", "text": "A relevant abstract."},
            "d2": {"title": "Other", "text": "A different abstract."},
        }
        qrels = {"d1": 2}

        rows = format_results(ranking, corpus, qrels_for_query=qrels, top_k=2)

        self.assertTrue(rows[0]["is_relevant"])
        self.assertEqual(rows[0]["relevance"], 2)
        self.assertFalse(rows[1]["is_relevant"])
        self.assertIsNone(rows[1]["relevance"])

    def test_shared_document_positions_keeps_only_overlaps(self) -> None:
        positions = shared_document_positions(
            {
                "bm25": [{"document_id": "d1", "position": 1}, {"document_id": "d2", "position": 2}],
                "dense": [{"document_id": "d2", "position": 1}, {"document_id": "d3", "position": 2}],
            }
        )

        self.assertEqual(positions, {"d2": {"bm25": 2, "dense": 1}})


if __name__ == "__main__":
    unittest.main()
