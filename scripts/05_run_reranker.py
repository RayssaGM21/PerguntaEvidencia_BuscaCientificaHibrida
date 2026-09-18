from __future__ import annotations

import argparse
import json

from _common import ensure_dataset
from src.config import Paths, ensure_output_dirs
from src.data_loader import load_beir_split
from src.evaluation.io import load_ranking, save_run_outputs
from src.retrievers.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank hybrid candidates with a CrossEncoder.")
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths()
    ensure_output_dirs(paths)
    ensure_dataset(args.dataset, paths)
    corpus, queries, qrels = load_beir_split(args.dataset, args.split, paths)
    hybrid = load_ranking(paths, args.dataset, "hybrid", args.split)
    reranker = CrossEncoderReranker(model_name=args.model, batch_size=args.batch_size)
    ranking, timing = reranker.rerank(queries, corpus, hybrid, candidate_pool=args.candidate_pool)
    metadata = {
        "reranker_model": args.model,
        "candidate_pool": args.candidate_pool,
        "batch_size": args.batch_size,
        "timing": timing.as_dict(),
        "input_retriever": "hybrid",
    }
    metrics, _ = save_run_outputs(paths, args.dataset, args.split, "hybrid_reranker", ranking, qrels, metadata)
    print(json.dumps({"metadata": metadata, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
