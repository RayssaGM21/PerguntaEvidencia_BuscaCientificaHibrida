from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def document_text(document: dict[str, str]) -> str:
    title = document.get("title") or ""
    text = document.get("text") or ""
    return f"{title}. {text}".strip()


def ranking_to_list(ranking: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(ranking.items(), key=lambda item: item[1], reverse=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
