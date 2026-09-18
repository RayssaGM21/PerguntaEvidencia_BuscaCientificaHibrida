from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.config import Paths, ensure_output_dirs


def bar_metric(df: pd.DataFrame, metric: str, path) -> None:
    pivot = df.pivot(index="dataset", columns="model", values=metric)
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel(metric)
    ax.set_xlabel("Dataset")
    ax.set_title(f"{metric} by model and dataset")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    paths = Paths()
    ensure_output_dirs(paths)
    df = pd.read_csv(paths.metrics_dir / "benchmark_results.csv")
    bar_metric(df, "ndcg@10", paths.plots_dir / "ndcg_at_10_by_model_dataset.png")
    bar_metric(df, "recall@100", paths.plots_dir / "recall_at_100_by_model_dataset.png")
    bar_metric(df, "precision@10", paths.plots_dir / "precision_at_10_by_model_dataset.png")

    alpha = pd.read_csv(paths.analysis_dir / "hybrid_alpha_dev.csv")
    weighted = alpha[alpha["method"] == "weighted"]
    plt.figure(figsize=(7, 4))
    plt.plot(weighted["alpha"], weighted["ndcg@10"], marker="o")
    plt.xlabel("alpha")
    plt.ylabel("nDCG@10")
    plt.title("Weighted hybrid alpha on NFCorpus dev")
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "hybrid_alpha_dev_ndcg.png", dpi=200)
    plt.close()

    per_query = pd.read_csv(paths.metrics_dir / "per_query_results.csv")
    pivot = per_query.pivot_table(index=["dataset", "query_id"], columns="model", values="ndcg@10").dropna()
    diff = pivot["hybrid_reranker"] - pivot["bm25"]
    plt.figure(figsize=(7, 4))
    plt.hist(diff, bins=40)
    plt.xlabel("nDCG@10 difference")
    plt.ylabel("Queries")
    plt.title("Hybrid + Reranker - BM25 per query")
    plt.tight_layout()
    plt.savefig(paths.plots_dir / "per_query_ndcg_diff_hybrid_reranker_minus_bm25.png", dpi=200)
    plt.close()

    timing_rows = []
    for path in paths.metrics_dir.glob("*_*.json"):
        try:
            import json
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metadata = payload.get("metadata", {})
        timing = metadata.get("timing", {})
        if "search_seconds" in timing or "scoring_seconds" in timing:
            timing_rows.append({"run": path.stem, "seconds": timing.get("search_seconds", timing.get("scoring_seconds", 0))})
    if timing_rows:
        timing_df = pd.DataFrame(timing_rows)
        ax = timing_df.plot(kind="bar", x="run", y="seconds", figsize=(10, 5), legend=False)
        ax.set_ylabel("seconds")
        ax.set_title("Available runtime measurements")
        plt.tight_layout()
        plt.savefig(paths.plots_dir / "runtime_available.png", dpi=200)
        plt.close()
    print(f"Plots saved to {paths.plots_dir}")


if __name__ == "__main__":
    main()
