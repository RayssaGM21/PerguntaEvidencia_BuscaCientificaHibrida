from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.config import Paths
from src.metrics import MetricSet, evaluate_run
from src.utils import ranking_to_list, write_json


METRIC_COLUMNS = ["ndcg@10", "recall@100", "precision@10", "mrr", "map"]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_metrics_csv(path: Path, model: str, dataset: str, split: str, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "dataset", "split", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerow({"model": model, "dataset": dataset, "split": split, **metrics})


def write_per_query_csv(path: Path, dataset: str, model: str, per_query: dict[str, MetricSet]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "dataset",
            "query_id",
            "model",
            "ndcg@10",
            "recall@100",
            "precision@10",
            "reciprocal_rank",
            "average_precision",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for query_id, metrics in sorted(per_query.items()):
            writer.writerow(
                {
                    "dataset": dataset,
                    "query_id": query_id,
                    "model": model,
                    "ndcg@10": metrics.ndcg_at_10,
                    "recall@100": metrics.recall_at_100,
                    "precision@10": metrics.precision_at_10,
                    "reciprocal_rank": metrics.mrr,
                    "average_precision": metrics.map,
                }
            )


def save_run_outputs(
    paths: Paths,
    dataset: str,
    split: str,
    model: str,
    rankings: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, MetricSet]]:
    aggregate, per_query = evaluate_run(rankings, qrels)
    metrics = aggregate.as_dict()
    stem = f"{dataset}_{model}_{split}"
    ranking_path = paths.rankings_dir / f"{stem}.json"
    metrics_json_path = paths.metrics_dir / f"{stem}.json"
    metrics_csv_path = paths.metrics_dir / f"{stem}.csv"
    per_query_path = paths.metrics_dir / f"{stem}_per_query.csv"

    write_json(ranking_path, rankings)
    write_json(
        metrics_json_path,
        {
            "model": model,
            "dataset": dataset,
            "split": split,
            "metrics": metrics,
            "metadata": metadata or {},
        },
    )
    write_metrics_csv(metrics_csv_path, model, dataset, split, metrics)
    write_per_query_csv(per_query_path, dataset, model, per_query)
    return metrics, per_query


def load_ranking(paths: Paths, dataset: str, model: str, split: str) -> dict[str, dict[str, float]]:
    return read_json(paths.rankings_dir / f"{dataset}_{model}_{split}.json")
