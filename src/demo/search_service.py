from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, Paths
from src.data_loader import dataset_is_available, download_beir_dataset, load_beir_split
from src.metrics import MetricSet, evaluate_query
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DEFAULT_DENSE_MODEL, DenseRetriever
from src.retrievers.hybrid import weighted_score_fusion
from src.retrievers.reranker import CrossEncoderReranker
from src.utils import document_text, ranking_to_list, tokenize


DATASET_LABELS = {
    "nfcorpus": "NFCorpus",
    "scifact": "SciFact",
    "trec-covid": "TREC-COVID",
}

MODEL_LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "hybrid_reranker": "Hybrid + Reranker",
}

MODEL_ORDER = ["bm25", "dense", "hybrid", "hybrid_reranker"]
HYBRID_ALPHA = 0.5
RERANKER_CANDIDATE_POOL = 20
DEMO_DATA_DIR = PROJECT_ROOT / "demo_data"


def load_ranking(paths: Paths, dataset: str, model: str, split: str) -> dict[str, dict[str, float]]:
    ranking_path = paths.rankings_dir / f"{dataset}_{model}_{split}.json"
    with ranking_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@dataclass(frozen=True)
class DatasetBundle:
    dataset: str
    split: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    benchmark_only: bool = False


@dataclass(frozen=True)
class DenseResources:
    retriever: DenseRetriever
    index: object
    doc_ids: list[str]
    cache_hit: bool


def normalize_query(text: str) -> str:
    return " ".join(tokenize(text))


def load_dataset_bundle(dataset: str, split: str = "test", paths: Paths | None = None) -> DatasetBundle:
    paths = paths or Paths()
    source_paths = paths
    benchmark_only = False

    if not dataset_is_available(dataset, source_paths):
        bundled_paths = replace(paths, data_dir=DEMO_DATA_DIR)
        if dataset_is_available(dataset, bundled_paths):
            source_paths = bundled_paths
            benchmark_only = dataset == "trec-covid"
        else:
            download_beir_dataset(dataset, source_paths)

    corpus, queries, qrels = load_beir_split(dataset, split, source_paths)
    return DatasetBundle(
        dataset=dataset,
        split=split,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        benchmark_only=benchmark_only,
    )


def find_matching_query_id(query_text: str, queries: dict[str, str]) -> str | None:
    candidate = query_text.strip()
    if candidate in queries:
        return candidate

    normalized_candidate = normalize_query(candidate)
    if not normalized_candidate:
        return None

    for query_id, benchmark_query in queries.items():
        if normalize_query(benchmark_query) == normalized_candidate:
            return query_id
    return None


def load_saved_query_ranking(
    paths: Paths,
    dataset: str,
    model: str,
    query_id: str,
    split: str = "test",
) -> dict[str, float]:
    return load_ranking(paths, dataset, model, split).get(query_id, {})


def fit_bm25(corpus: dict[str, dict[str, str]]) -> BM25Retriever:
    return BM25Retriever().fit(corpus)


def bm25_search(retriever: BM25Retriever, query: str, top_k: int = 100) -> dict[str, float]:
    return {result.doc_id: result.score for result in retriever.search(query, top_k=top_k)}


def build_dense_resources(
    corpus: dict[str, dict[str, str]],
    dataset: str,
    cache_dir: Path,
    model_name: str = DEFAULT_DENSE_MODEL,
    device: str | None = None,
    batch_size: int = 256,
) -> DenseResources:
    retriever = DenseRetriever(model_name=model_name, device=device, batch_size=batch_size)
    embeddings, doc_ids, _, cache_hit = retriever.get_or_create_corpus_embeddings(
        corpus=corpus,
        dataset=dataset,
        cache_dir=cache_dir,
        force_rebuild=False,
    )
    index = retriever.create_index(embeddings)
    return DenseResources(retriever=retriever, index=index, doc_ids=doc_ids, cache_hit=cache_hit)


def dense_search(resources: DenseResources, query: str, top_k: int = 100) -> dict[str, float]:
    query_embeddings = resources.retriever.encode_texts([query])
    results = resources.retriever.search_index(resources.index, query_embeddings, resources.doc_ids, top_k=top_k)
    return dict(results.get(0, []))


def hybrid_search(lexical: dict[str, float], dense: dict[str, float], top_k: int = 100) -> dict[str, float]:
    fused = weighted_score_fusion(
        {"query": lexical},
        {"query": dense},
        alpha=HYBRID_ALPHA,
        top_k=top_k,
    )
    return fused["query"]


def rerank_search(
    reranker: CrossEncoderReranker,
    query: str,
    corpus: dict[str, dict[str, str]],
    candidates: dict[str, float],
) -> dict[str, float]:
    rankings, _ = reranker.rerank(
        queries={"query": query},
        corpus=corpus,
        candidate_rankings={"query": candidates},
        candidate_pool=RERANKER_CANDIDATE_POOL,
    )
    return rankings["query"]


def query_metrics(ranking: dict[str, float], qrels_for_query: dict[str, int]) -> MetricSet:
    ranked_doc_ids = [doc_id for doc_id, _ in ranking_to_list(ranking)]
    return evaluate_query(ranked_doc_ids, qrels_for_query)


def make_snippet(document: dict[str, str], max_chars: int = 420) -> str:
    text = document_text(document).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def format_results(
    ranking: dict[str, float],
    corpus: dict[str, dict[str, str]],
    qrels_for_query: dict[str, int] | None = None,
    top_k: int = 10,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    qrels_for_query = qrels_for_query or {}
    for position, (doc_id, score) in enumerate(ranking_to_list(ranking)[:top_k], start=1):
        document = corpus.get(doc_id, {"title": "", "text": ""})
        relevance = qrels_for_query.get(doc_id)
        rows.append(
            {
                "position": position,
                "document_id": doc_id,
                "title": document.get("title") or "(sem titulo)",
                "snippet": make_snippet(document),
                "score": float(score),
                "relevance": relevance,
                "is_relevant": relevance is not None and relevance > 0,
            }
        )
    return rows


def shared_document_positions(results_by_model: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, int]]:
    positions: dict[str, dict[str, int]] = {}
    for model, rows in results_by_model.items():
        for row in rows:
            doc_id = str(row["document_id"])
            positions.setdefault(doc_id, {})[model] = int(row["position"])
    return {doc_id: model_positions for doc_id, model_positions in positions.items() if len(model_positions) > 1}


def load_benchmark_results(paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or Paths()
    path = paths.metrics_dir / "benchmark_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["model", "dataset", "ndcg@10", "recall@100", "precision@10", "mrr", "map"])
    return pd.read_csv(path)


def benchmark_ndcg_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    table = results.pivot(index="dataset", columns="model", values="ndcg@10")
    table = table.reindex(index=list(DATASET_LABELS), columns=MODEL_ORDER)
    table.index = [DATASET_LABELS.get(dataset, dataset) for dataset in table.index]
    table.columns = [MODEL_LABELS.get(model, model) for model in table.columns]
    return table


def best_ndcg_by_dataset(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["dataset", "model", "ndcg@10"])
    idx = results.groupby("dataset")["ndcg@10"].idxmax()
    best = results.loc[idx, ["dataset", "model", "ndcg@10"]].copy()
    best["dataset"] = best["dataset"].map(DATASET_LABELS).fillna(best["dataset"])
    best["model"] = best["model"].map(MODEL_LABELS).fillna(best["model"])
    return best.sort_values("dataset")


def load_per_query_results(paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or Paths()
    path = paths.metrics_dir / "per_query_results.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "dataset",
                "query_id",
                "model",
                "ndcg@10",
                "recall@100",
                "precision@10",
                "reciprocal_rank",
                "average_precision",
            ]
        )
    return pd.read_csv(path)


def experiment_case_candidates(
    dataset: str,
    per_query: pd.DataFrame,
    limit: int = 3,
) -> dict[str, pd.DataFrame]:
    subset = per_query[per_query["dataset"] == dataset]
    if subset.empty:
        return {}

    pivot = subset.pivot_table(index="query_id", columns="model", values="ndcg@10", aggfunc="first")
    for model in MODEL_ORDER:
        if model not in pivot:
            pivot[model] = np.nan
    pivot = pivot.dropna(subset=MODEL_ORDER)
    if pivot.empty:
        return {}

    comparisons = {
        "bm25_beats_dense": pivot["bm25"] - pivot["dense"],
        "dense_beats_bm25": pivot["dense"] - pivot["bm25"],
        "hybrid_beats_both": pivot["hybrid"] - pivot[["bm25", "dense"]].max(axis=1),
        "reranker_improves": pivot["hybrid_reranker"] - pivot["hybrid"],
        "reranker_worsens": pivot["hybrid"] - pivot["hybrid_reranker"],
    }

    cases: dict[str, pd.DataFrame] = {}
    for name, delta in comparisons.items():
        rows = pivot.copy()
        rows["delta"] = delta
        rows = rows[rows["delta"] > 0].sort_values("delta", ascending=False).head(limit)
        cases[name] = rows.reset_index()
    return cases


def cosine_device_label() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    return "cpu"


def rankings_for_free_query(
    query: str,
    corpus: dict[str, dict[str, str]],
    bm25: BM25Retriever,
    dense_resources: DenseResources | None,
    reranker: CrossEncoderReranker | None,
    selected_models: list[str],
    top_k: int = 100,
) -> dict[str, dict[str, float]]:
    rankings: dict[str, dict[str, float]] = {}
    needs_bm25 = any(model in selected_models for model in ("bm25", "hybrid", "hybrid_reranker"))
    needs_dense = any(model in selected_models for model in ("dense", "hybrid", "hybrid_reranker"))

    if needs_bm25:
        rankings["bm25"] = bm25_search(bm25, query, top_k=top_k)
    if needs_dense:
        if dense_resources is None:
            raise ValueError("Dense resources are required for the selected models.")
        rankings["dense"] = dense_search(dense_resources, query, top_k=top_k)
    if any(model in selected_models for model in ("hybrid", "hybrid_reranker")):
        rankings["hybrid"] = hybrid_search(rankings["bm25"], rankings["dense"], top_k=top_k)
    if "hybrid_reranker" in selected_models:
        if reranker is None:
            rankings["hybrid_reranker"] = rankings["hybrid"]
        else:
            rankings["hybrid_reranker"] = rerank_search(reranker, query, corpus, rankings["hybrid"])

    return rankings


def rankings_for_existing_query(
    paths: Paths,
    dataset: str,
    query_id: str,
    selected_models: list[str],
    split: str = "test",
) -> dict[str, dict[str, float]]:
    return {
        model: load_saved_query_ranking(paths, dataset, model, query_id, split)
        for model in selected_models
    }
