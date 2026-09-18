from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Paths, ensure_output_dirs
from src.data_loader import dataset_is_available, download_beir_dataset, load_beir_split
from src.evaluation.io import METRIC_COLUMNS, load_ranking, read_json, save_run_outputs
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DEFAULT_DENSE_MODEL, DenseRetriever
from src.utils import set_seed


DATASETS = ["nfcorpus", "scifact", "trec-covid"]
FINAL_MODELS = ["bm25", "dense", "hybrid", "hybrid_reranker"]


def ensure_dataset(dataset: str, paths: Paths) -> None:
    if not dataset_is_available(dataset, paths):
        download_beir_dataset(dataset, paths)


def run_bm25_if_needed(dataset: str, split: str, top_k: int = 100) -> None:
    paths = Paths()
    ensure_output_dirs(paths)
    output_path = paths.rankings_dir / f"{dataset}_bm25_{split}.json"
    if output_path.exists():
        return
    ensure_dataset(dataset, paths)
    corpus, queries, qrels = load_beir_split(dataset, split, paths)
    rankings = BM25Retriever().fit(corpus).batch_search(queries, top_k=top_k)
    save_run_outputs(paths, dataset, split, "bm25", rankings, qrels, {"top_k": top_k})


def run_dense_if_needed(dataset: str, split: str, top_k: int = 100, batch_size: int = 256) -> None:
    paths = Paths()
    ensure_output_dirs(paths)
    output_path = paths.rankings_dir / f"{dataset}_dense_{split}.json"
    if output_path.exists():
        return
    ensure_dataset(dataset, paths)
    corpus, queries, qrels = load_beir_split(dataset, split, paths)
    retriever = DenseRetriever(model_name=DEFAULT_DENSE_MODEL, batch_size=batch_size)
    rankings, timing, metadata = retriever.retrieve(corpus, queries, dataset, paths.embeddings_dir, top_k=top_k)
    save_run_outputs(
        paths,
        dataset,
        split,
        "dense",
        rankings,
        qrels,
        {
            "model_name": DEFAULT_DENSE_MODEL,
            "device": retriever.device,
            "top_k": top_k,
            "batch_size": batch_size,
            "timing": timing.as_dict(len(queries)),
            **metadata,
        },
    )


def write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metrics_for(paths: Paths, dataset: str, model: str, split: str = "test") -> dict[str, float]:
    return read_json(paths.metrics_dir / f"{dataset}_{model}_{split}.json")["metrics"]


def load_final_config(paths: Paths) -> dict:
    return read_json(paths.analysis_dir / "hybrid_config.json")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
