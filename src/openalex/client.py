from __future__ import annotations

import os
from dataclasses import dataclass

import requests


def inverted_index_to_text(index: dict | None) -> str:
    if not index:
        return ""
    positions = []
    for token, token_positions in index.items():
        for position in token_positions:
            positions.append((position, token))
    return " ".join(token for _, token in sorted(positions))


@dataclass(frozen=True)
class OpenAlexWork:
    id: str
    title: str
    abstract: str
    authors: str
    year: int | None
    doi: str
    cited_by_count: int


class OpenAlexClient:
    def __init__(self, email: str | None = None) -> None:
        self.email = email or os.getenv("OPENALEX_EMAIL", "")
        self.base_url = "https://api.openalex.org/works"

    def search(self, query: str, per_page: int = 50) -> list[OpenAlexWork]:
        params = {"search": query, "per-page": per_page}
        if self.email:
            params["mailto"] = self.email
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        works = []
        for item in response.json().get("results", []):
            authors = ", ".join(
                author.get("author", {}).get("display_name", "")
                for author in item.get("authorships", [])[:5]
                if author.get("author", {}).get("display_name")
            )
            works.append(
                OpenAlexWork(
                    id=item.get("id", ""),
                    title=item.get("title") or item.get("display_name") or "",
                    abstract=inverted_index_to_text(item.get("abstract_inverted_index")),
                    authors=authors,
                    year=item.get("publication_year"),
                    doi=item.get("doi") or "",
                    cited_by_count=int(item.get("cited_by_count") or 0),
                )
            )
        return works
