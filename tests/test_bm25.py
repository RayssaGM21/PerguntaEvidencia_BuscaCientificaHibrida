from __future__ import annotations

import unittest

from src.retrievers.bm25 import BM25Retriever


class BM25RetrieverTest(unittest.TestCase):
    def test_search_returns_matching_document_first(self) -> None:
        corpus = {
            "d1": {"title": "heart disease", "text": "cardiology treatment"},
            "d2": {"title": "graph algorithms", "text": "shortest paths"},
        }
        retriever = BM25Retriever().fit(corpus)

        results = retriever.search("cardiology heart", top_k=2)

        self.assertEqual(results[0].doc_id, "d1")
        self.assertGreater(results[0].score, 0.0)


if __name__ == "__main__":
    unittest.main()
