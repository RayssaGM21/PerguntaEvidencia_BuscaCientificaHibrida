from __future__ import annotations

import csv
import json

import pandas as pd

from _common import DATASETS, FINAL_MODELS, markdown_table
from src.config import Paths, ensure_output_dirs
from src.data_loader import load_corpus, load_qrels, load_queries


DISPLAY_MODEL = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "hybrid_reranker": "Hybrid + Reranker",
}


def consolidate_per_query(paths: Paths) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in FINAL_MODELS:
            path = paths.metrics_dir / f"{dataset}_{model}_test_per_query.csv"
            if not path.exists():
                continue
            rows.extend(pd.read_csv(path).to_dict("records"))
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(paths.metrics_dir / "per_query_results.csv", index=False)
    return df


def write_tables(paths: Paths, benchmark: pd.DataFrame) -> None:
    report_dir = paths.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in benchmark.iterrows():
        rows.append(
            [
                DISPLAY_MODEL.get(row["model"], row["model"]),
                row["dataset"],
                f"{row['ndcg@10']:.6f}",
                f"{row['recall@100']:.6f}",
                f"{row['precision@10']:.6f}",
                f"{row['mrr']:.6f}",
                f"{row['map']:.6f}",
            ]
        )
    full_table = markdown_table(["Modelo", "Dataset", "nDCG@10", "Recall@100", "P@10", "MRR", "MAP"], rows)

    compact_rows = []
    for model in FINAL_MODELS:
        model_rows = benchmark[benchmark["model"] == model].set_index("dataset")
        compact_rows.append(
            [
                DISPLAY_MODEL.get(model, model),
                *[f"{model_rows.loc[dataset, 'ndcg@10']:.6f}" if dataset in model_rows.index else "" for dataset in DATASETS],
            ]
        )
    compact = markdown_table(["Modelo", "NFCorpus nDCG@10", "SciFact nDCG@10", "TREC-COVID nDCG@10"], compact_rows)
    (report_dir / "tables.md").write_text(full_table + "\n\n" + compact + "\n", encoding="utf-8")


def dataset_stats(paths: Paths) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        corpus = load_corpus(dataset, paths)
        queries = load_queries(dataset, paths)
        qrels = load_qrels(dataset, "test", paths)
        qrels_count = sum(len(docs) for docs in qrels.values())
        rows.append({"dataset": dataset, "documents": len(corpus), "queries": len(qrels), "qrels": qrels_count})
    return rows


def main() -> None:
    paths = Paths()
    ensure_output_dirs(paths)
    per_query = consolidate_per_query(paths)
    benchmark = pd.read_csv(paths.metrics_dir / "benchmark_results.csv")
    write_tables(paths, benchmark)

    lines = ["# Results Summary\n"]
    lines.append("## Dataset Statistics\n")
    stats_rows = [[row["dataset"], str(row["documents"]), str(row["queries"]), str(row["qrels"])] for row in dataset_stats(paths)]
    lines.append(markdown_table(["Dataset", "Documents", "Queries", "Qrels"], stats_rows))

    lines.append("\n## Models\n")
    lines.append("- BM25: lexical baseline over title + text.")
    lines.append("- Dense: zero-shot `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` with normalized embeddings and FAISS inner product search.")
    lines.append("- Hybrid: configuration selected only on NFCorpus dev.")
    lines.append("- Hybrid + Reranker: CrossEncoder reranking of the Hybrid Top 100 candidates.")

    for dataset in DATASETS:
        lines.append(f"\n## {dataset} Results\n")
        subset = benchmark[benchmark["dataset"] == dataset]
        rows = [
            [DISPLAY_MODEL.get(row["model"], row["model"]), *[f"{row[col]:.6f}" for col in ["ndcg@10", "recall@100", "precision@10", "mrr", "map"]]]
            for _, row in subset.iterrows()
        ]
        lines.append(markdown_table(["Model", "nDCG@10", "Recall@100", "P@10", "MRR", "MAP"], rows))

    lines.append("\n## Best Model\n")
    best_rows = []
    for dataset in DATASETS:
        subset = benchmark[benchmark["dataset"] == dataset]
        best = subset.loc[subset["ndcg@10"].idxmax()]
        best_rows.append([dataset, DISPLAY_MODEL.get(best["model"], best["model"]), f"{best['ndcg@10']:.6f}"])
    lines.append(markdown_table(["Dataset", "Best nDCG@10 model", "nDCG@10"], best_rows))

    lines.append("\n## Relative Improvements\n")
    improvement_rows = []
    for dataset in DATASETS:
        subset = benchmark[benchmark["dataset"] == dataset]
        bm25 = float(subset[subset["model"] == "bm25"]["ndcg@10"].iloc[0])
        best = subset.loc[subset["ndcg@10"].idxmax()]
        improvement = (float(best["ndcg@10"]) - bm25) / bm25 * 100 if bm25 else 0.0
        improvement_rows.append([dataset, DISPLAY_MODEL.get(best["model"], best["model"]), f"{improvement:.2f}%"])
    lines.append(markdown_table(["Dataset", "Best model", "Relative nDCG@10 vs BM25"], improvement_rows))

    lines.append("\n## Statistical Analysis\n")
    sig_path = paths.analysis_dir / "significance.csv"
    if sig_path.exists():
        sig = pd.read_csv(sig_path)
        rows = [
            [row["dataset"], row["candidate"], row["baseline"], f"{row['mean_difference']:.6f}", f"[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]", f"{row['p_value']:.6f}"]
            for _, row in sig.iterrows()
        ]
        lines.append(markdown_table(["Dataset", "Candidate", "Baseline", "Mean diff", "95% CI", "p-value"], rows))

    lines.append("\n## Error Analysis\n")
    err_path = paths.analysis_dir / "error_analysis.md"
    if err_path.exists():
        text = err_path.read_text(encoding="utf-8").splitlines()
        lines.extend(text[:80])

    lines.append("\n## Runtime\n")
    runtime_rows = []
    for path in paths.metrics_dir.glob("*_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        timing = payload.get("metadata", {}).get("timing", {})
        if timing:
            runtime_rows.append([path.stem, json.dumps(timing)])
    lines.append(markdown_table(["Run", "Timing"], runtime_rows) if runtime_rows else "No comparable runtime metadata available.")

    lines.append("\n## Methodological Checks\n")
    lines.append("- Hybrid strategy/alpha selected only on NFCorpus dev.")
    lines.append("- NFCorpus test, SciFact test and TREC-COVID used only for final evaluation.")
    lines.append("- Qrels are used only for evaluation/statistical analysis, not as retrieval features.")
    lines.append("- Fine-tuning was not implemented in this experimental version.")
    (paths.report_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(paths.report_dir / "RESULTS_SUMMARY.md")


if __name__ == "__main__":
    main()
