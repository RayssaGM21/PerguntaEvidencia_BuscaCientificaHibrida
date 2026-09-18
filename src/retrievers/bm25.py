from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from src.utils import document_text, tokenize


@dataclass
class BM25SearchResult:
    doc_id: str
    score: float


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0
        self.term_frequencies: list[Counter[str]] = []
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}

    def fit(self, corpus: dict[str, dict[str, str]]) -> "BM25Retriever":
        self.doc_ids = list(corpus.keys())
        self.doc_lengths = []
        self.term_frequencies = []
        self.inverted_index = defaultdict(list)
        self.idf = {}

        for doc_index, doc_id in enumerate(self.doc_ids):
            tokens = tokenize(document_text(corpus[doc_id]))
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.doc_lengths.append(len(tokens))
            for term, frequency in frequencies.items():
                self.inverted_index[term].append((doc_index, frequency))

        total_docs = len(self.doc_ids)
        self.avg_doc_length = sum(self.doc_lengths) / total_docs if total_docs else 0.0
        for term, postings in self.inverted_index.items():
            doc_frequency = len(postings)
            self.idf[term] = math.log(1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
        return self

    def search(self, query: str, top_k: int = 1000) -> list[BM25SearchResult]:
        if not self.doc_ids:
            raise RuntimeError("BM25Retriever must be fitted before search.")

        scores: dict[int, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))

        for term, query_frequency in query_terms.items():
            if term not in self.inverted_index:
                continue
            idf = self.idf[term]
            for doc_index, term_frequency in self.inverted_index[term]:
                doc_length = self.doc_lengths[doc_index]
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1e-9)
                )
                score = idf * ((term_frequency * (self.k1 + 1)) / denominator)
                scores[doc_index] += query_frequency * score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [BM25SearchResult(doc_id=self.doc_ids[doc_index], score=float(score)) for doc_index, score in ranked]

    def batch_search(self, queries: dict[str, str], top_k: int = 1000) -> dict[str, dict[str, float]]:
        rankings: dict[str, dict[str, float]] = {}
        for query_id, query in queries.items():
            rankings[query_id] = {result.doc_id: result.score for result in self.search(query, top_k=top_k)}
        return rankings
