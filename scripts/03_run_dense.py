from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Paths, ensure_output_dirs
from src.data_loader import dataset_is_available, download_beir_dataset, load_beir_split
from src.metrics import evaluate_run
from src.retrievers.dense import DEFAULT_DENSE_MODEL, DenseRetriever
from src.utils import set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense retrieval baseline.")
    parser.add_argument("--dataset", default="nfcorpus", help="BEIR dataset name.")
    parser.add_argument("--split", default="test", help="qrels split to evaluate.")
    parser.add_argument("--model", default=DEFAULT_DENSE_MODEL, help="SentenceTransformers model.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of documents to keep per query.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cached corpus embeddings.")
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compare_with_bm25(paths: Paths, dataset: str, split: str, dense_metrics: dict[str, float]) -> list[dict[str, float | str]]:
    bm25_path = paths.metrics_dir / f"{dataset}_bm25_{split}.json"
    if not bm25_path.exists():
        return []

    bm25_metrics = load_json(bm25_path)["metrics"]
    rows: list[dict[str, float | str]] = []
    for metric_name, dense_value in dense_metrics.items():
        bm25_value = float(bm25_metrics[metric_name])
        absolute = dense_value - bm25_value
        percent = (absolute / bm25_value * 100.0) if bm25_value else 0.0
        rows.append(
            {
                "metric": metric_name,
                "bm25": bm25_value,
                "dense": dense_value,
                "absolute_diff": absolute,
                "percent_diff": percent,
            }
        )
    return rows


def print_comparison(rows: list[dict[str, float | str]]) -> None:
    if not rows:
        print("BM25 metrics file not found; skipping comparison.")
        return

    print("\nBM25 vs Dense")
    print("metric,bm25,dense,absolute_diff,percent_diff")
    for row in rows:
        print(
            f"{row['metric']},"
            f"{float(row['bm25']):.6f},"
            f"{float(row['dense']):.6f},"
            f"{float(row['absolute_diff']):.6f},"
            f"{float(row['percent_diff']):.2f}%"
        )


def print_top5_examples(
    paths: Paths,
    dataset: str,
    split: str,
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    dense_rankings: dict[str, dict[str, float]],
) -> None:
    bm25_path = paths.rankings_dir / f"{dataset}_bm25_{split}.json"
    if not bm25_path.exists():
        print("\nBM25 ranking file not found; skipping qualitative examples.")
        return

    bm25_rankings = load_json(bm25_path)
    example_query_ids = [
        query_id
        for query_id in sorted(queries)
        if len(bm25_rankings.get(query_id, {})) >= 5 and len(dense_rankings.get(query_id, {})) >= 5
    ][:3]

    print("\nQualitative examples")
    for query_id in example_query_ids:
        print(f"\nQuery {query_id}: {queries[query_id]}")
        for label, rankings, score_name in (
            ("Top 5 BM25", bm25_rankings, "score"),
            ("Top 5 Dense", dense_rankings, "cosine_similarity"),
        ):
            print(label)
            ranked = sorted(rankings[query_id].items(), key=lambda item: item[1], reverse=True)[:5]
            for position, (doc_id, score) in enumerate(ranked, start=1):
                title = corpus.get(doc_id, {}).get("title") or ""
                relevance = qrels.get(query_id, {}).get(doc_id, 0)
                print(
                    f"  {position}. doc_id={doc_id} rel={relevance} "
                    f"{score_name}={float(score):.6f} title={title}"
                )


def main() -> None:
    args = parse_args()
    paths = Paths()
    ensure_output_dirs(paths)
    set_seed(42)

    if not dataset_is_available(args.dataset, paths):
        download_beir_dataset(args.dataset, paths)

    corpus, queries, qrels = load_beir_split(args.dataset, args.split, paths)
    retriever = DenseRetriever(model_name=args.model, batch_size=args.batch_size)

    print(f"Model: {args.model}")
    print(f"Device: {retriever.device}")
    print(f"Loaded {args.dataset}/{args.split}: {len(corpus)} docs, {len(queries)} queries")

    total_start = time.time()
    rankings, timing, retrieval_metadata = retriever.retrieve(
        corpus=corpus,
        queries=queries,
        dataset=args.dataset,
        cache_dir=paths.embeddings_dir,
        top_k=args.top_k,
        force_rebuild=args.force_rebuild,
    )
    total_seconds = time.time() - total_start

    aggregate, per_query = evaluate_run(rankings, qrels)
    metrics = aggregate.as_dict()
    timing_payload = timing.as_dict(query_count=len(queries))
    timing_payload["total_seconds"] = total_seconds

    stem = f"{args.dataset}_dense_{args.split}"
    ranking_path = paths.rankings_dir / f"{stem}.json"
    metrics_json_path = paths.metrics_dir / f"{stem}.json"
    metrics_csv_path = paths.metrics_dir / f"{stem}.csv"
    per_query_path = paths.metrics_dir / f"{stem}_per_query.csv"

    write_json(ranking_path, rankings)
    payload = {
        "model": "dense",
        "sentence_transformers_model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "device": retriever.device,
        "parameters": {"top_k": args.top_k, "batch_size": args.batch_size, "normalize_embeddings": True},
        "retrieval_metadata": retrieval_metadata,
        "timing": timing_payload,
        "metrics": metrics,
    }
    write_json(metrics_json_path, payload)
    write_metrics_csv(metrics_csv_path, "dense", args.dataset, args.split, metrics)
    write_per_query_csv(per_query_path, per_query)

    print("\nDense metrics")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")

    print("\nTiming")
    for metric_name, value in timing_payload.items():
        print(f"{metric_name}: {value:.6f}")
    print(f"corpus_cache_hit: {retrieval_metadata['corpus_cache_hit']}")

    comparison_rows = compare_with_bm25(paths, args.dataset, args.split, metrics)
    print_comparison(comparison_rows)

    comparison_path = paths.metrics_dir / f"{args.dataset}_bm25_vs_dense_{args.split}.csv"
    if comparison_rows:
        with comparison_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["metric", "bm25", "dense", "absolute_diff", "percent_diff"])
            writer.writeheader()
            writer.writerows(comparison_rows)

    print_top5_examples(paths, args.dataset, args.split, corpus, queries, qrels, rankings)
    print(f"\nSaved ranking: {ranking_path}")
    print(f"Saved metrics: {metrics_json_path}")


if __name__ == "__main__":
    main()
