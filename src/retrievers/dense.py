from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.utils import document_text


DEFAULT_DENSE_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


def detect_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class EmbeddingCache:
    embeddings_path: Path
    ids_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class DenseTiming:
    corpus_embedding_seconds: float = 0.0
    query_embedding_seconds: float = 0.0
    indexing_seconds: float = 0.0
    search_seconds: float = 0.0

    def as_dict(self, query_count: int) -> dict[str, float]:
        average_query = self.search_seconds / query_count if query_count else 0.0
        return {
            "corpus_embedding_seconds": self.corpus_embedding_seconds,
            "query_embedding_seconds": self.query_embedding_seconds,
            "indexing_seconds": self.indexing_seconds,
            "search_seconds": self.search_seconds,
            "average_search_seconds_per_query": average_query,
        }


class DenseRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        encoder: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or detect_device()
        self.batch_size = batch_size
        self.encoder = encoder

    def _load_encoder(self) -> Any:
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.model_name, device=self.device)
        return self.encoder

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        encoder = self._load_encoder()
        embeddings = encoder.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        return np.ascontiguousarray(embeddings)

    def cache_paths(self, cache_dir: Path, dataset: str, kind: str) -> EmbeddingCache:
        cache_dir = cache_dir / dataset
        stem = f"{dataset}_{sanitize_model_name(self.model_name)}_{kind}"
        return EmbeddingCache(
            embeddings_path=cache_dir / f"{stem}.npy",
            ids_path=cache_dir / f"{stem}_ids.json",
            metadata_path=cache_dir / f"{stem}_metadata.json",
        )

    def load_cached_embeddings(
        self,
        cache_dir: Path,
        dataset: str,
        kind: str,
        expected_ids: list[str],
    ) -> tuple[np.ndarray, list[str]] | None:
        cache = self.cache_paths(cache_dir, dataset, kind)
        if not (cache.embeddings_path.exists() and cache.ids_path.exists() and cache.metadata_path.exists()):
            return None

        metadata = json.loads(cache.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_name") != self.model_name or metadata.get("kind") != kind:
            return None

        ids = json.loads(cache.ids_path.read_text(encoding="utf-8"))
        if ids != expected_ids:
            return None

        embeddings = np.load(cache.embeddings_path)
        if embeddings.shape[0] != len(ids):
            return None
        return np.ascontiguousarray(embeddings.astype("float32")), ids

    def save_cached_embeddings(
        self,
        cache_dir: Path,
        dataset: str,
        kind: str,
        ids: list[str],
        embeddings: np.ndarray,
    ) -> None:
        cache = self.cache_paths(cache_dir, dataset, kind)
        cache.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache.embeddings_path, embeddings)
        cache.ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
        metadata = {
            "dataset": dataset,
            "model_name": self.model_name,
            "kind": kind,
            "embedding_shape": list(embeddings.shape),
            "normalized": True,
            "similarity": "cosine_via_inner_product",
        }
        cache.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def get_or_create_corpus_embeddings(
        self,
        corpus: dict[str, dict[str, str]],
        dataset: str,
        cache_dir: Path,
        force_rebuild: bool = False,
    ) -> tuple[np.ndarray, list[str], float, bool]:
        doc_ids = list(corpus.keys())
        if not force_rebuild:
            cached = self.load_cached_embeddings(cache_dir, dataset, "corpus", doc_ids)
            if cached is not None:
                embeddings, ids = cached
                return embeddings, ids, 0.0, True

        texts = [document_text(corpus[doc_id]) for doc_id in doc_ids]
        start = time.time()
        embeddings = self.encode_texts(texts)
        elapsed = time.time() - start
        self.save_cached_embeddings(cache_dir, dataset, "corpus", doc_ids, embeddings)
        return embeddings, doc_ids, elapsed, False

    def create_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.ascontiguousarray(embeddings.astype("float32")))
        return index

    def search_index(
        self,
        index: faiss.Index,
        query_embeddings: np.ndarray,
        doc_ids: list[str],
        top_k: int,
    ) -> dict[int, list[tuple[str, float]]]:
        scores, indices = index.search(np.ascontiguousarray(query_embeddings.astype("float32")), top_k)
        results: dict[int, list[tuple[str, float]]] = {}
        for query_position, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            rows: list[tuple[str, float]] = []
            for score, doc_position in zip(query_scores, query_indices):
                if doc_position < 0:
                    continue
                rows.append((doc_ids[int(doc_position)], float(score)))
            results[query_position] = rows
        return results

    def retrieve(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        dataset: str,
        cache_dir: Path,
        top_k: int = 100,
        force_rebuild: bool = False,
    ) -> tuple[dict[str, dict[str, float]], DenseTiming, dict[str, Any]]:
        corpus_embeddings, doc_ids, corpus_seconds, cache_hit = self.get_or_create_corpus_embeddings(
            corpus=corpus,
            dataset=dataset,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
        )

        query_ids = list(queries.keys())
        query_texts = [queries[query_id] for query_id in query_ids]

        query_start = time.time()
        query_embeddings = self.encode_texts(query_texts)
        query_seconds = time.time() - query_start

        index_start = time.time()
        index = self.create_index(corpus_embeddings)
        indexing_seconds = time.time() - index_start

        search_start = time.time()
        indexed_results = self.search_index(index, query_embeddings, doc_ids, top_k=top_k)
        search_seconds = time.time() - search_start

        rankings: dict[str, dict[str, float]] = {}
        for query_position, query_id in enumerate(query_ids):
            rankings[query_id] = {
                doc_id: score for doc_id, score in indexed_results.get(query_position, [])
            }

        timing = DenseTiming(
            corpus_embedding_seconds=corpus_seconds,
            query_embedding_seconds=query_seconds,
            indexing_seconds=indexing_seconds,
            search_seconds=search_seconds,
        )
        metadata = {
            "corpus_cache_hit": cache_hit,
            "embedding_dimension": int(corpus_embeddings.shape[1]),
            "corpus_embedding_count": int(corpus_embeddings.shape[0]),
            "query_embedding_count": int(query_embeddings.shape[0]),
        }
        return rankings, timing, metadata
