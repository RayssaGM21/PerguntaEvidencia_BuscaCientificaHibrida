from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.config import Paths
from src.demo.search_service import (
    find_matching_query_id,
    format_results,
    load_dataset_bundle,
    rankings_for_free_query,
    shared_document_positions,
)


class StubBM25:
    def search(self, query: str, top_k: int = 100):
        return []


class DemoSearchServiceTests(unittest.TestCase):
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

    def test_bundled_dataset_is_used_when_full_dataset_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = replace(Paths(), data_dir=Path(temporary_directory))
            bundle = load_dataset_bundle("trec-covid", paths=paths)

        self.assertTrue(bundle.benchmark_only)
        self.assertEqual(len(bundle.queries), 50)
        self.assertEqual(len(bundle.qrels), 50)
        self.assertGreater(len(bundle.corpus), 0)

    def test_bm25_only_search_does_not_require_dense_resources(self) -> None:
        rankings = rankings_for_free_query(
            query="test query",
            corpus={},
            bm25=StubBM25(),
            dense_resources=None,
            reranker=None,
            selected_models=["bm25"],
            top_k=5,
        )

        self.assertEqual(rankings, {"bm25": {}})


if __name__ == "__main__":
    unittest.main()
