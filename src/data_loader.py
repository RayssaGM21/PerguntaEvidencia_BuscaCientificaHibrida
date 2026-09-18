from __future__ import annotations

import csv
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src.config import BEIR_DATASETS, Paths
from src.utils import read_jsonl


def dataset_dir(dataset: str, paths: Paths | None = None) -> Path:
    paths = paths or Paths()
    return paths.data_dir / dataset


def dataset_is_available(dataset: str, paths: Paths | None = None) -> bool:
    root = dataset_dir(dataset, paths)
    return (
        (root / "corpus.jsonl").exists()
        and (root / "queries.jsonl").exists()
        and (root / "qrels").exists()
    )


def download_beir_dataset(dataset: str, paths: Paths | None = None) -> Path:
    paths = paths or Paths()
    if dataset not in BEIR_DATASETS:
        valid = ", ".join(sorted(BEIR_DATASETS))
        raise ValueError(f"Unknown dataset '{dataset}'. Valid options: {valid}")

    target_dir = dataset_dir(dataset, paths)
    if dataset_is_available(dataset, paths):
        return target_dir

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = paths.data_dir / f"{dataset}.zip"
    url = BEIR_DATASETS[dataset].url

    print(f"Downloading {dataset} from {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "scientific-ir-benchmark/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with zip_path.open("wb") as file:
                shutil.copyfileobj(response, file)
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        print("SSL certificate verification failed; retrying BEIR download without certificate verification.")
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            with zip_path.open("wb") as file:
                shutil.copyfileobj(response, file)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(paths.data_dir)

    if not dataset_is_available(dataset, paths):
        raise RuntimeError(f"Dataset extraction finished, but expected files were not found in {target_dir}")

    return target_dir


def load_corpus(dataset: str, paths: Paths | None = None) -> dict[str, dict[str, str]]:
    root = dataset_dir(dataset, paths)
    corpus_path = root / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    corpus: dict[str, dict[str, str]] = {}
    for row in read_jsonl(corpus_path):
        doc_id = str(row["_id"])
        corpus[doc_id] = {
            "title": row.get("title") or "",
            "text": row.get("text") or "",
        }
    return corpus


def load_queries(dataset: str, paths: Paths | None = None) -> dict[str, str]:
    root = dataset_dir(dataset, paths)
    queries_path = root / "queries.jsonl"
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    queries: dict[str, str] = {}
    for row in read_jsonl(queries_path):
        queries[str(row["_id"])] = row.get("text") or ""
    return queries


def load_qrels(dataset: str, split: str, paths: Paths | None = None) -> dict[str, dict[str, int]]:
    root = dataset_dir(dataset, paths)
    qrels_path = root / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels file: {qrels_path}")

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            score = int(row["score"])
            qrels.setdefault(query_id, {})[corpus_id] = score
    return qrels


def load_beir_split(
    dataset: str,
    split: str,
    paths: Paths | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    corpus = load_corpus(dataset, paths)
    queries = load_queries(dataset, paths)
    qrels = load_qrels(dataset, split, paths)
    queries = {query_id: queries[query_id] for query_id in qrels if query_id in queries}
    return corpus, queries, qrels
