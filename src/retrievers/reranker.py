from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.utils import document_text, ranking_to_list


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"


@dataclass(frozen=True)
class RerankerTiming:
    scoring_seconds: float
    total_pairs: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "scoring_seconds": self.scoring_seconds,
            "total_pairs": self.total_pairs,
            "average_seconds_per_pair": self.scoring_seconds / self.total_pairs if self.total_pairs else 0.0,
        }


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = model

    def _load_model(self) -> Any:
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, device=self.device)
        return self.model

    def rerank(
        self,
        queries: dict[str, str],
        corpus: dict[str, dict[str, str]],
        candidate_rankings: dict[str, dict[str, float]],
        candidate_pool: int = 100,
    ) -> tuple[dict[str, dict[str, float]], RerankerTiming]:
        model = self._load_model()
        pairs: list[list[str]] = []
        pair_keys: list[tuple[str, str]] = []
        base_scores: dict[tuple[str, str], float] = {}

        for query_id, ranking in candidate_rankings.items():
            for doc_id, base_score in ranking_to_list(ranking)[:candidate_pool]:
                if query_id not in queries or doc_id not in corpus:
                    continue
                pairs.append([queries[query_id], document_text(corpus[doc_id])])
                pair_keys.append((query_id, doc_id))
                base_scores[(query_id, doc_id)] = base_score

        start = time.time()
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=True)
        elapsed = time.time() - start

        grouped: dict[str, dict[str, float]] = {}
        for (query_id, doc_id), score in zip(pair_keys, scores):
            grouped.setdefault(query_id, {})[doc_id] = float(score)

        reranked: dict[str, dict[str, float]] = {}
        for query_id, ranking in candidate_rankings.items():
            original = ranking_to_list(ranking)
            scored_prefix = grouped.get(query_id, {})
            ordered_prefix = sorted(scored_prefix.items(), key=lambda item: item[1], reverse=True)
            if ordered_prefix:
                floor = min(score for _, score in ordered_prefix) - 1.0
            else:
                floor = 0.0
            combined: dict[str, float] = dict(ordered_prefix)
            suffix_rank = 0
            for doc_id, _ in original:
                if doc_id in combined:
                    continue
                suffix_rank += 1
                combined[doc_id] = floor - suffix_rank * 1e-6
            reranked[query_id] = dict(sorted(combined.items(), key=lambda item: item[1], reverse=True)[: len(original)])
        return reranked, RerankerTiming(scoring_seconds=elapsed, total_pairs=len(pairs))
