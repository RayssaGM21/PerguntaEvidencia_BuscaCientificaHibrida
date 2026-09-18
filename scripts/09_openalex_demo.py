from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.openalex.client import OpenAlexClient
from src.retrievers.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAlex reranking demo.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--per-page", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    client = OpenAlexClient()
    works = client.search(args.query, per_page=args.per_page)
    reranker = CrossEncoderReranker(model_name=DEFAULT_RERANKER_MODEL)
    model = reranker._load_model()
    pairs = [[args.query, f"{work.title}. {work.abstract}".strip()] for work in works]
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    ranked = sorted(enumerate(zip(works, scores), start=1), key=lambda item: float(item[1][1]), reverse=True)

    print("Ranking produzido pelo modelo experimental; não há ground truth OpenAlex nesta demonstração.\n")
    for new_position, (original_position, (work, score)) in enumerate(ranked[: args.top_k], start=1):
        print(f"{new_position}. original_position={original_position} reranker_score={float(score):.6f}")
        print(f"   title={work.title}")
        print(f"   year={work.year} authors={work.authors}")
        print(f"   doi={work.doi} openalex_id={work.id} cited_by_count={work.cited_by_count}")


if __name__ == "__main__":
    main()
