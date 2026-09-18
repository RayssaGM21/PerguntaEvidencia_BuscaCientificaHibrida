from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from _common import DATASETS, FINAL_MODELS
from src.config import Paths, ensure_output_dirs
from src.data_loader import load_qrels
from src.evaluation.significance import paired_bootstrap


COMPARISONS = [
    ("dense", "bm25"),
    ("hybrid", "bm25"),
    ("hybrid", "dense"),
    ("hybrid_reranker", "hybrid"),
    ("hybrid_reranker", "bm25"),
    ("hybrid_reranker", "dense"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired bootstrap significance tests.")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    return parser.parse_args()


def normalize_per_query_file(path: Path, dataset: str, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["dataset"] = dataset
    frame["model"] = model
    if "reciprocal_rank" not in frame.columns and "mrr" in frame.columns:
        frame["reciprocal_rank"] = frame["mrr"]
    if "average_precision" not in frame.columns and "map" in frame.columns:
        frame["average_precision"] = frame["map"]
    required = [
        "dataset",
        "query_id",
        "model",
        "ndcg@10",
        "recall@100",
        "precision@10",
        "reciprocal_rank",
        "average_precision",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame[required]


def consolidate_and_validate_per_query(paths: Paths, datasets: list[str]) -> pd.DataFrame:
    frames = []
    validation_rows = []
    for dataset in datasets:
        expected_queries = len(load_qrels(dataset, "test", paths))
        for model in FINAL_MODELS:
            path = paths.metrics_dir / f"{dataset}_{model}_test_per_query.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing per-query metrics file: {path}")
            frame = normalize_per_query_file(path, dataset, model)
            observed_queries = frame["query_id"].nunique()
            validation_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "expected_queries": expected_queries,
                    "observed_queries": observed_queries,
                    "valid": observed_queries == expected_queries,
                }
            )
            if observed_queries != expected_queries:
                raise ValueError(
                    f"{dataset}/{model} has {observed_queries} queries in {path}, "
                    f"but qrels/test has {expected_queries}."
                )
            frames.append(frame)

    consolidated = pd.concat(frames, ignore_index=True)
    consolidated.to_csv(paths.metrics_dir / "per_query_results.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(paths.analysis_dir / "per_query_validation.csv", index=False)
    return consolidated


def main() -> None:
    args = parse_args()
    paths = Paths()
    ensure_output_dirs(paths)
    df = consolidate_and_validate_per_query(paths, args.datasets)
    rows = []
    for dataset in args.datasets:
        subset = df[df["dataset"] == dataset]
        for candidate, baseline in COMPARISONS:
            pivot = subset[subset["model"].isin([candidate, baseline])].pivot(
                index="query_id", columns="model", values="ndcg@10"
            ).dropna()
            expected_queries = len(load_qrels(dataset, "test", paths))
            if len(pivot) != expected_queries:
                raise ValueError(
                    f"{dataset} {candidate} vs {baseline} has {len(pivot)} paired queries; "
                    f"expected {expected_queries}."
                )
            result = paired_bootstrap(
                baseline=pivot[baseline].tolist(),
                candidate=pivot[candidate].tolist(),
                samples=args.samples,
                seed=42,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "candidate": candidate,
                    "baseline": baseline,
                    "n_queries": len(pivot),
                    "metric": "ndcg@10",
                    "mean_difference": result.mean_difference,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "ci_excludes_zero": result.ci_lower > 0 or result.ci_upper < 0,
                    "p_value": result.p_value,
                    "bootstrap_samples": args.samples,
                    "seed": 42,
                    "multiple_comparison_correction": "none",
                }
            )
    with (paths.analysis_dir / "significance.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
