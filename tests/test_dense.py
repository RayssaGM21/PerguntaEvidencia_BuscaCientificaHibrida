from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrievers.dense import DenseRetriever, sanitize_model_name


class FakeEncoder:
    def encode(
        self,
        texts: list[str],
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            if "alpha" in text:
                vector = np.array([3.0, 0.0], dtype="float32")
            elif "beta" in text:
                vector = np.array([0.0, 4.0], dtype="float32")
            else:
                vector = np.array([1.0, 1.0], dtype="float32")
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors)


class DenseRetrieverTest(unittest.TestCase):
    def test_encode_texts_returns_normalized_embeddings(self) -> None:
        retriever = DenseRetriever(model_name="fake/model", encoder=FakeEncoder())

        embeddings = retriever.encode_texts(["alpha", "beta"])

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)

    def test_faiss_mapping_and_ranking_order(self) -> None:
        retriever = DenseRetriever(model_name="fake/model", encoder=FakeEncoder())
        corpus_embeddings = retriever.encode_texts(["alpha document", "beta document"])
        index = retriever.create_index(corpus_embeddings)
        query_embeddings = retriever.encode_texts(["beta query"])

        results = retriever.search_index(index, query_embeddings, ["d_alpha", "d_beta"], top_k=2)

        self.assertEqual(results[0][0][0], "d_beta")
        self.assertGreaterEqual(results[0][0][1], results[0][1][1])

    def test_top_k_size(self) -> None:
        retriever = DenseRetriever(model_name="fake/model", encoder=FakeEncoder())
        embeddings = retriever.encode_texts(["alpha", "beta", "other"])
        index = retriever.create_index(embeddings)

        results = retriever.search_index(index, retriever.encode_texts(["alpha"]), ["d1", "d2", "d3"], top_k=2)

        self.assertEqual(len(results[0]), 2)

    def test_cache_round_trip_and_id_validation(self) -> None:
        retriever = DenseRetriever(model_name="fake/model", encoder=FakeEncoder())
        embeddings = retriever.encode_texts(["alpha", "beta"])

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            retriever.save_cached_embeddings(cache_dir, "toy", "corpus", ["d1", "d2"], embeddings)

            cached = retriever.load_cached_embeddings(cache_dir, "toy", "corpus", ["d1", "d2"])
            self.assertIsNotNone(cached)
            cached_embeddings, cached_ids = cached
            self.assertEqual(cached_ids, ["d1", "d2"])
            np.testing.assert_allclose(cached_embeddings, embeddings)

            wrong_ids = retriever.load_cached_embeddings(cache_dir, "toy", "corpus", ["d2", "d1"])
            self.assertIsNone(wrong_ids)

    def test_sanitize_model_name(self) -> None:
        self.assertEqual(sanitize_model_name("sentence-transformers/a/b"), "sentence-transformers_a_b")


if __name__ == "__main__":
    unittest.main()
