from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricSet:
    ndcg_at_10: float
    recall_at_100: float
    precision_at_10: float
    mrr: float
    map: float

    def as_dict(self) -> dict[str, float]:
        return {
            "ndcg@10": self.ndcg_at_10,
            "recall@100": self.recall_at_100,
            "precision@10": self.precision_at_10,
            "mrr": self.mrr,
            "map": self.map,
        }


def _dcg(relevances: list[int]) -> float:
    return sum(((2**rel - 1) / math.log2(index + 2)) for index, rel in enumerate(relevances))


def ndcg_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    gains = [qrels.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    ideal = sorted((score for score in qrels.values() if score > 0), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(gains) / ideal_dcg


def recall_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    if not relevant:
        return 0.0
    retrieved_relevant = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant)
    return retrieved_relevant / len(relevant)


def precision_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    retrieved_relevant = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant)
    return retrieved_relevant / k


def reciprocal_rank(ranked_doc_ids: list[str], qrels: dict[str, int]) -> float:
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    for index, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def average_precision(ranked_doc_ids: list[str], qrels: dict[str, int]) -> float:
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / index
    return precision_sum / len(relevant)


def evaluate_query(ranked_doc_ids: list[str], qrels: dict[str, int]) -> MetricSet:
    return MetricSet(
        ndcg_at_10=ndcg_at_k(ranked_doc_ids, qrels, 10),
        recall_at_100=recall_at_k(ranked_doc_ids, qrels, 100),
        precision_at_10=precision_at_k(ranked_doc_ids, qrels, 10),
        mrr=reciprocal_rank(ranked_doc_ids, qrels),
        map=average_precision(ranked_doc_ids, qrels),
    )


def evaluate_run(
    rankings: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> tuple[MetricSet, dict[str, MetricSet]]:
    per_query: dict[str, MetricSet] = {}
    for query_id, query_qrels in qrels.items():
        ranked_doc_ids = sorted(
            rankings.get(query_id, {}),
            key=lambda doc_id: rankings.get(query_id, {}).get(doc_id, float("-inf")),
            reverse=True,
        )
        per_query[query_id] = evaluate_query(ranked_doc_ids, query_qrels)

    if not per_query:
        raise ValueError("No queries were evaluated.")

    aggregate = MetricSet(
        ndcg_at_10=float(np.mean([metric.ndcg_at_10 for metric in per_query.values()])),
        recall_at_100=float(np.mean([metric.recall_at_100 for metric in per_query.values()])),
        precision_at_10=float(np.mean([metric.precision_at_10 for metric in per_query.values()])),
        mrr=float(np.mean([metric.mrr for metric in per_query.values()])),
        map=float(np.mean([metric.map for metric in per_query.values()])),
    )
    return aggregate, per_query
