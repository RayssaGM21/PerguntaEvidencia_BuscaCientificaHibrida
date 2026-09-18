from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import BM25Config, Paths, ensure_output_dirs
from src.data_loader import dataset_is_available, download_beir_dataset, load_beir_split
from src.metrics import evaluate_run
from src.retrievers.bm25 import BM25Retriever
from src.utils import set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 baseline.")
    parser.add_argument("--dataset", default="nfcorpus", help="BEIR dataset name.")
    parser.add_argument("--split", default="test", help="qrels split to evaluate.")
    parser.add_argument("--top-k", type=int, default=1000, help="Number of documents to keep per query.")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    return parser.parse_args()


def write_metrics_csv(path: Path, model: str, dataset: str, split: str, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["model", "dataset", "split", "ndcg@10", "recall@100", "precision@10", "mrr", "map"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"model": model, "dataset": dataset, "split": split, **metrics})


def write_per_query_csv(path: Path, per_query: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["query_id", "ndcg@10", "recall@100", "precision@10", "mrr", "map"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for query_id, metrics in sorted(per_query.items()):
            writer.writerow({"query_id": query_id, **metrics.as_dict()})


def main() -> None:
    args = parse_args()
    paths = Paths()
    config = BM25Config(k1=args.k1, b=args.b, top_k=args.top_k)
    ensure_output_dirs(paths)
    set_seed(42)

    if not dataset_is_available(args.dataset, paths):
        download_beir_dataset(args.dataset, paths)

    start = time.time()
    corpus, queries, qrels = load_beir_split(args.dataset, args.split, paths)
    print(f"Loaded {args.dataset}/{args.split}: {len(corpus)} docs, {len(queries)} queries")

    retriever = BM25Retriever(k1=config.k1, b=config.b).fit(corpus)
    rankings = retriever.batch_search(queries, top_k=config.top_k)
    aggregate, per_query = evaluate_run(rankings, qrels)
    elapsed_seconds = time.time() - start

    stem = f"{args.dataset}_bm25_{args.split}"
    ranking_path = paths.rankings_dir / f"{stem}.json"
    metrics_json_path = paths.metrics_dir / f"{stem}.json"
    metrics_csv_path = paths.metrics_dir / f"{stem}.csv"
    per_query_path = paths.metrics_dir / f"{stem}_per_query.csv"

    write_json(ranking_path, rankings)
    metrics_payload = {
        "model": "bm25",
        "dataset": args.dataset,
        "split": args.split,
        "parameters": {"k1": config.k1, "b": config.b, "top_k": config.top_k},
        "elapsed_seconds": elapsed_seconds,
        "metrics": aggregate.as_dict(),
    }
    write_json(metrics_json_path, metrics_payload)
    write_metrics_csv(metrics_csv_path, "bm25", args.dataset, args.split, aggregate.as_dict())
    write_per_query_csv(per_query_path, per_query)

    print("BM25 metrics")
    for metric_name, value in aggregate.as_dict().items():
        print(f"{metric_name}: {value:.6f}")
    print(f"Saved ranking: {ranking_path}")
    print(f"Saved metrics: {metrics_json_path}")


if __name__ == "__main__":
    main()
