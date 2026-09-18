from __future__ import annotations

from src.evaluation.io import ranking_to_list


def reciprocal_rank_fusion(
    lexical: dict[str, dict[str, float]],
    dense: dict[str, dict[str, float]],
    k: int = 60,
    top_k: int = 100,
) -> dict[str, dict[str, float]]:
    fused: dict[str, dict[str, float]] = {}
    query_ids = sorted(set(lexical) | set(dense))
    for query_id in query_ids:
        scores: dict[str, float] = {}
        for ranking in (lexical.get(query_id, {}), dense.get(query_id, {})):
            for rank, (doc_id, _) in enumerate(ranking_to_list(ranking), start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        fused[query_id] = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
    return fused


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return {doc_id: 1.0 for doc_id in scores}
    return {doc_id: (score - minimum) / (maximum - minimum) for doc_id, score in scores.items()}


def weighted_score_fusion(
    lexical: dict[str, dict[str, float]],
    dense: dict[str, dict[str, float]],
    alpha: float,
    top_k: int = 100,
) -> dict[str, dict[str, float]]:
    fused: dict[str, dict[str, float]] = {}
    query_ids = sorted(set(lexical) | set(dense))
    for query_id in query_ids:
        lexical_norm = minmax_normalize(lexical.get(query_id, {}))
        dense_norm = minmax_normalize(dense.get(query_id, {}))
        doc_ids = set(lexical_norm) | set(dense_norm)
        scores = {
            doc_id: alpha * lexical_norm.get(doc_id, 0.0) + (1.0 - alpha) * dense_norm.get(doc_id, 0.0)
            for doc_id in doc_ids
        }
        fused[query_id] = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
    return fused
