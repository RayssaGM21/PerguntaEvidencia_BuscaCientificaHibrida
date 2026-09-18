from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ensure_dataset, run_bm25_if_needed, run_dense_if_needed, write_rows_csv
from src.config import Paths, ensure_output_dirs
from src.data_loader import load_beir_split
from src.evaluation.io import METRIC_COLUMNS, load_ranking, save_run_outputs
from src.metrics import evaluate_run
from src.retrievers.hybrid import reciprocal_rank_fusion, weighted_score_fusion
from src.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval.")
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--tune-on-dev", action="store_true")
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


def evaluate_fusion(dataset: str, split: str, method: str, alpha: float | None, rrf_k: int, top_k: int) -> dict:
    paths = Paths()
    _, _, qrels = load_beir_split(dataset, split, paths)
    bm25 = load_ranking(paths, dataset, "bm25", split)
    dense = load_ranking(paths, dataset, "dense", split)
    if method == "rrf":
        ranking = reciprocal_rank_fusion(bm25, dense, k=rrf_k, top_k=top_k)
    else:
        ranking = weighted_score_fusion(bm25, dense, alpha=float(alpha), top_k=top_k)
    metrics, _ = evaluate_run(ranking, qrels)
    return {"ranking": ranking, "metrics": metrics.as_dict()}


def tune_on_dev(top_k: int, rrf_k: int) -> dict:
    paths = Paths()
    dataset = "nfcorpus"
    split = "dev"
    ensure_dataset(dataset, paths)
    run_bm25_if_needed(dataset, split, top_k=top_k)
    run_dense_if_needed(dataset, split, top_k=top_k)

    rows = []
    best = {"method": "", "alpha": None, "ndcg@10": -1.0}
    for alpha_i in range(11):
        alpha = alpha_i / 10
        result = evaluate_fusion(dataset, split, "weighted", alpha, rrf_k, top_k)
        row = {"method": "weighted", "alpha": alpha, **result["metrics"]}
        rows.append(row)
        if row["ndcg@10"] > best["ndcg@10"]:
            best = {"method": "weighted", "alpha": alpha, "ndcg@10": row["ndcg@10"]}

    rrf_result = evaluate_fusion(dataset, split, "rrf", None, rrf_k, top_k)
    rrf_row = {"method": "rrf", "alpha": "", **rrf_result["metrics"]}
    rows.append(rrf_row)
    if rrf_row["ndcg@10"] > best["ndcg@10"]:
        best = {"method": "rrf", "alpha": None, "ndcg@10": rrf_row["ndcg@10"]}

    write_rows_csv(paths.analysis_dir / "hybrid_alpha_dev.csv", ["method", "alpha", *METRIC_COLUMNS], rows)
    config = {"dataset": dataset, "split": split, "selected_method": best["method"], "alpha": best["alpha"], "rrf_k": rrf_k, "top_k": top_k}
    write_json(paths.analysis_dir / "hybrid_config.json", config)
    return config


def main() -> None:
    args = parse_args()
    paths = Paths()
    ensure_output_dirs(paths)

    config_path = paths.analysis_dir / "hybrid_config.json"
    if args.tune_on_dev or not config_path.exists():
        config = tune_on_dev(args.top_k, args.rrf_k)
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))

    ensure_dataset(args.dataset, paths)
    run_bm25_if_needed(args.dataset, args.split, top_k=args.top_k)
    run_dense_if_needed(args.dataset, args.split, top_k=args.top_k)
    _, _, qrels = load_beir_split(args.dataset, args.split, paths)
    bm25 = load_ranking(paths, args.dataset, "bm25", args.split)
    dense = load_ranking(paths, args.dataset, "dense", args.split)
    if config["selected_method"] == "rrf":
        ranking = reciprocal_rank_fusion(bm25, dense, k=int(config["rrf_k"]), top_k=args.top_k)
    else:
        ranking = weighted_score_fusion(bm25, dense, alpha=float(config["alpha"]), top_k=args.top_k)
    metrics, _ = save_run_outputs(paths, args.dataset, args.split, "hybrid", ranking, qrels, config)
    print(json.dumps({"config": config, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
