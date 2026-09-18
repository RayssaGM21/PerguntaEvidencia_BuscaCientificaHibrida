from __future__ import annotations

import argparse
import csv
import subprocess
import sys

from _common import DATASETS, FINAL_MODELS, markdown_table, metrics_for, run_bm25_if_needed, run_dense_if_needed
from src.config import Paths, ensure_output_dirs
from src.data_loader import download_beir_dataset
from src.evaluation.io import METRIC_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final benchmark matrix.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--skip-reranker", action="store_true")
    return parser.parse_args()


def run_script(script: str, *args: str) -> None:
    command = [sys.executable, script, *args]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    paths = Paths()
    ensure_output_dirs(paths)

    run_script("scripts/04_run_hybrid.py", "--dataset", "nfcorpus", "--split", "dev", "--tune-on-dev", "--top-k", str(args.top_k))

    for dataset in args.datasets:
        download_beir_dataset(dataset, paths)
        run_bm25_if_needed(dataset, "test", top_k=args.top_k)
        run_dense_if_needed(dataset, "test", top_k=args.top_k)
        run_script("scripts/04_run_hybrid.py", "--dataset", dataset, "--split", "test", "--top-k", str(args.top_k))
        if not args.skip_reranker:
            reranker_metrics = paths.metrics_dir / f"{dataset}_hybrid_reranker_test.json"
            if not reranker_metrics.exists():
                run_script(
                    "scripts/05_run_reranker.py",
                    "--dataset",
                    dataset,
                    "--split",
                    "test",
                    "--candidate-pool",
                    "20",
                    "--batch-size",
                    "32",
                )

    rows = []
    models = FINAL_MODELS if not args.skip_reranker else FINAL_MODELS[:-1]
    for dataset in args.datasets:
        for model in models:
            metrics = metrics_for(paths, dataset, model, "test")
            rows.append({"model": model, "dataset": dataset, **metrics})

    csv_path = paths.metrics_dir / "benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "dataset", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)

    md_rows = [
        [row["model"], row["dataset"], *[f"{row[column]:.6f}" for column in METRIC_COLUMNS]]
        for row in rows
    ]
    (paths.metrics_dir / "benchmark_results.md").write_text(
        markdown_table(["model", "dataset", *METRIC_COLUMNS], md_rows),
        encoding="utf-8",
    )
    print((paths.metrics_dir / "benchmark_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
