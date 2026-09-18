from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEMO_DIR = PROJECT_ROOT / "demo_data"
RANKINGS_DIR = PROJECT_ROOT / "outputs" / "rankings"


def copy_split_files(dataset: str) -> None:
    source = DATA_DIR / dataset
    target = DEMO_DIR / dataset
    (target / "qrels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "queries.jsonl", target / "queries.jsonl")
    shutil.copy2(source / "qrels" / "test.tsv", target / "qrels" / "test.tsv")


def ranked_document_ids(dataset: str) -> set[str]:
    document_ids: set[str] = set()
    for path in RANKINGS_DIR.glob(f"{dataset}_*_test.json"):
        rankings = json.loads(path.read_text(encoding="utf-8"))
        for ranking in rankings.values():
            document_ids.update(str(document_id) for document_id in ranking)
    return document_ids


def write_filtered_corpus(dataset: str, document_ids: set[str]) -> int:
    source = DATA_DIR / dataset / "corpus.jsonl"
    target = DEMO_DIR / dataset / "corpus.jsonl"
    written = 0
    with source.open("r", encoding="utf-8") as input_file, target.open("w", encoding="utf-8") as output_file:
        for line in input_file:
            row = json.loads(line)
            if str(row.get("_id")) in document_ids:
                output_file.write(line)
                written += 1
    return written


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in ("nfcorpus", "scifact"):
        copy_split_files(dataset)
        shutil.copy2(DATA_DIR / dataset / "corpus.jsonl", DEMO_DIR / dataset / "corpus.jsonl")
        print(f"{dataset}: full corpus bundled")

    dataset = "trec-covid"
    copy_split_files(dataset)
    document_ids = ranked_document_ids(dataset)
    written = write_filtered_corpus(dataset, document_ids)
    if written != len(document_ids):
        raise RuntimeError(f"TREC-COVID bundle is missing {len(document_ids) - written} ranked documents")
    print(f"{dataset}: {written} ranked documents bundled")


if __name__ == "__main__":
    main()
