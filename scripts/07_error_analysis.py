from __future__ import annotations

import csv

import pandas as pd

from _common import DATASETS, FINAL_MODELS, markdown_table
from src.config import Paths, ensure_output_dirs
from src.data_loader import load_beir_split
from src.evaluation.io import load_ranking, ranking_to_list


CATEGORIES = {
    "bm25_much_better_than_dense": lambda row: row.get("bm25", 0) - row.get("dense", 0),
    "dense_much_better_than_bm25": lambda row: row.get("dense", 0) - row.get("bm25", 0),
    "hybrid_beats_both": lambda row: row.get("hybrid", 0) - max(row.get("bm25", 0), row.get("dense", 0)),
    "reranker_improves": lambda row: row.get("hybrid_reranker", 0) - row.get("hybrid", 0),
    "reranker_worsens": lambda row: row.get("hybrid", 0) - row.get("hybrid_reranker", 0),
}


def first_docs(dataset: str, query_id: str, model: str, corpus: dict[str, dict[str, str]], qrels: dict[str, dict[str, int]], paths: Paths) -> list[dict]:
    ranking = load_ranking(paths, dataset, model, "test").get(query_id, {})
    rows = []
    for position, (doc_id, score) in enumerate(ranking_to_list(ranking)[:5], start=1):
        rows.append(
            {
                "model": model,
                "document_id": doc_id,
                "position": position,
                "title": corpus.get(doc_id, {}).get("title", ""),
                "relevance": qrels.get(query_id, {}).get(doc_id, 0),
                "score": score,
            }
        )
    return rows


def main() -> None:
    paths = Paths()
    ensure_output_dirs(paths)
    df = pd.read_csv(paths.metrics_dir / "per_query_results.csv")
    rows = []
    md_lines = ["# Error Analysis\n"]
    for dataset in DATASETS:
        corpus, queries, qrels = load_beir_split(dataset, "test", paths)
        pivot = df[df["dataset"] == dataset].pivot(index="query_id", columns="model", values="ndcg@10").fillna(0)
        low = pivot[pivot[FINAL_MODELS].max(axis=1) <= 0.05].copy() if all(model in pivot for model in FINAL_MODELS) else pd.DataFrame()
        categories = dict(CATEGORIES)
        selected: dict[str, list[str]] = {}
        for category, scorer in categories.items():
            scored = [(query_id, scorer(row)) for query_id, row in pivot.iterrows()]
            selected[category] = [query_id for query_id, score in sorted(scored, key=lambda item: item[1], reverse=True) if score > 0][:3]
        selected["all_models_low"] = list(low.index[:3])

        for category, query_ids in selected.items():
            md_lines.append(f"## {dataset} - {category}\n")
            for query_id in query_ids:
                md_lines.append(f"- `{query_id}`: {queries.get(query_id, '')}")
                for model in FINAL_MODELS:
                    for doc in first_docs(dataset, query_id, model, corpus, qrels, paths)[:1]:
                        rows.append({"dataset": dataset, "category": category, "query_id": query_id, "query": queries.get(query_id, ""), **doc})
                        md_lines.append(
                            f"  - {model}: pos {doc['position']} rel {doc['relevance']} score {float(doc['score']):.6f} `{doc['document_id']}` {doc['title']}"
                        )
            md_lines.append("")

    fieldnames = ["dataset", "category", "query_id", "query", "model", "document_id", "position", "title", "relevance", "score"]
    with (paths.analysis_dir / "error_analysis.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (paths.analysis_dir / "error_analysis.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {len(rows)} rows.")


if __name__ == "__main__":
    main()
