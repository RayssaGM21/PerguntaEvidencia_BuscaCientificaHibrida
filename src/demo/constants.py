from __future__ import annotations


DATASET_STATS = {
    "nfcorpus": {"label": "NFCorpus", "queries": "323 queries", "documents": "3.633 documentos"},
    "scifact": {"label": "SciFact", "queries": "300 queries", "documents": "5.183 documentos"},
    "trec-covid": {"label": "TREC-COVID", "queries": "50 queries", "documents": "171.332 documentos"},
}

MODEL_DESCRIPTIONS = {
    "bm25": "Correspondencia lexical",
    "dense": "Similaridade semantica",
    "hybrid": "Lexical + semantico",
    "hybrid_reranker": "Reordenacao neural",
}

MODEL_COLORS = {
    "bm25": "#2563EB",
    "dense": "#0F766E",
    "hybrid": "#7C3AED",
    "hybrid_reranker": "#F97316",
}

MODEL_SOFT_COLORS = {
    "bm25": "#DBEAFE",
    "dense": "#CCFBF1",
    "hybrid": "#EDE9FE",
    "hybrid_reranker": "#FFEDD5",
}

EXAMPLE_QUERIES = [
    "knowledge graphs in education",
    "vaccine effectiveness",
    "disease transmission",
    "neural retrieval",
]

CASE_DEFINITIONS = {
    "bm25_beats_dense": {
        "title": "BM25 supera Dense",
        "description": "Consultas em que correspondencia lexical ordenou melhor os resultados.",
        "color": "#2563EB",
    },
    "dense_beats_bm25": {
        "title": "Dense supera BM25",
        "description": "Consultas em que similaridade semantica superou o baseline lexical.",
        "color": "#0F766E",
    },
    "hybrid_beats_both": {
        "title": "Hybrid supera ambos",
        "description": "Consultas em que a combinacao dos sinais venceu BM25 e Dense.",
        "color": "#7C3AED",
    },
    "reranker_improves": {
        "title": "Reranker melhora Hybrid",
        "description": "Consultas em que o CrossEncoder elevou a qualidade do ranking.",
        "color": "#F97316",
    },
    "reranker_worsens": {
        "title": "Reranker piora Hybrid",
        "description": "Consultas em que a reordenacao neural reduziu nDCG@10.",
        "color": "#EF4444",
    },
}
