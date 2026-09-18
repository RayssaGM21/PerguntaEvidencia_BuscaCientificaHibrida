from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    metrics_dir: Path = PROJECT_ROOT / "outputs" / "metrics"
    rankings_dir: Path = PROJECT_ROOT / "outputs" / "rankings"
    embeddings_dir: Path = PROJECT_ROOT / "outputs" / "embeddings"
    plots_dir: Path = PROJECT_ROOT / "outputs" / "plots"
    analysis_dir: Path = PROJECT_ROOT / "outputs" / "analysis"
    report_dir: Path = PROJECT_ROOT / "outputs" / "report"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    url: str


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 1000


SEED = 42

BEIR_DATASETS: dict[str, DatasetConfig] = {
    "nfcorpus": DatasetConfig(
        name="nfcorpus",
        url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    ),
    "scifact": DatasetConfig(
        name="scifact",
        url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    ),
    "trec-covid": DatasetConfig(
        name="trec-covid",
        url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
    ),
}


def ensure_output_dirs(paths: Paths | None = None) -> None:
    paths = paths or Paths()
    for directory in (
        paths.data_dir,
        paths.models_dir,
        paths.metrics_dir,
        paths.rankings_dir,
        paths.embeddings_dir,
        paths.plots_dir,
        paths.analysis_dir,
        paths.report_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
