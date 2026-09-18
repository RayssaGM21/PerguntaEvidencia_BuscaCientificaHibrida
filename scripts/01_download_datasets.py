from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import BEIR_DATASETS, Paths, ensure_output_dirs
from src.data_loader import download_beir_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BEIR datasets.")
    parser.add_argument(
        "--dataset",
        choices=sorted(BEIR_DATASETS),
        default="nfcorpus",
        help="Dataset to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths()
    ensure_output_dirs(paths)
    target_dir = download_beir_dataset(args.dataset, paths)
    print(f"Dataset ready: {target_dir}")


if __name__ == "__main__":
    main()
