from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

from src.demo.constants import (
    CASE_DEFINITIONS,
    DATASET_STATS,
    MODEL_COLORS,
    MODEL_DESCRIPTIONS,
    MODEL_SOFT_COLORS,
)
from src.demo.search_service import MODEL_LABELS, MODEL_ORDER


def _clean_html(fragment: str) -> str:
    """Remove Python indentation before concatenating HTML for Streamlit."""
    return dedent(fragment).strip()


@lru_cache(maxsize=1)
def _logo_data_uri() -> str:
    logo_path = Path(__file__).resolve().parents[2] / "static" / "da_pergunta_a_evidencia_header.png"
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_header() -> None:
    badges = "".join(
        f"<span class='badge'>{label}</span>"
        for label in ["BM25", "Dense Retrieval", "Hybrid Search", "CrossEncoder", "BEIR"]
    )
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-logo-wrap">
            <img
              class="hero-logo"
              src="{_logo_data_uri()}"
              alt="Da Pergunta a Evidencia - Busca Cientifica Hibrida"
            >
          </div>
          <div class="hero-content">
            <h1 class="hero-title">Scientific IR Benchmark</h1>
            <p class="hero-subtitle">Da pergunta a evidencia: compare diferentes estrategias de recuperacao de literatura cientifica</p>
            <div class="hero-badges">{badges}</div>
          </div>
          <div class="event-mark">Tech Summit 2026 / UNIMAR</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def sidebar_dataset_stat(dataset: str) -> None:
    stats = DATASET_STATS[dataset]
    st.markdown(
        f"""
        <div class="dataset-stat">
          <strong>{stats['label']}</strong><br>
          {stats['queries']} / {stats['documents']}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_reference(selected_models: list[str]) -> None:
    items = []
    for model in MODEL_ORDER:
        active = model in selected_models
        opacity = "1" if active else "0.52"
        items.append(
            _clean_html(
                f"""
            <div class="model-chip" style="opacity:{opacity}">
              <div class="model-chip-title">
                <span class="model-dot" style="background:{MODEL_COLORS[model]}"></span>
                {MODEL_LABELS[model]}
              </div>
              <div class="model-chip-desc">{MODEL_DESCRIPTIONS[model]}</div>
            </div>
            """
            )
        )
    st.markdown(f"<div class='model-list'>{''.join(items)}</div>", unsafe_allow_html=True)


def status_box(kind: str, title: str, text: str) -> None:
    if kind == "success":
        color = "#10B981"
        mark = "OK"
        bg = "#D1FAE5"
    else:
        color = "#2563EB"
        mark = "INFO"
        bg = "#DBEAFE"
    st.markdown(
        f"""
        <div class="status-box">
          <div class="status-icon" style="background:{bg};color:{color};">{mark}</div>
          <div>
            <div class="status-title">{escape(title)}</div>
            <div class="status-text">{escape(text)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metrics_grid(metrics_by_model: dict[str, object]) -> None:
    cards = []
    for model, metrics in metrics_by_model.items():
        color = MODEL_COLORS.get(model, "#2563EB")
        cards.append(
            _clean_html(
                f"""
            <div class="metric-card" style="border-top:4px solid {color}">
              <div class="metric-label">{MODEL_LABELS.get(model, model)}</div>
              <div class="metric-value">{metrics.ndcg_at_10:.3f}</div>
              <div class="status-text">nDCG@10</div>
              <div class="status-text">P@10 {metrics.precision_at_10:.3f} · R@100 {metrics.recall_at_100:.3f} · MRR {metrics.mrr:.3f}</div>
            </div>
            """
            )
        )
    st.markdown(f"<div class='metrics-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _relevance_badge(row: dict[str, object]) -> str:
    relevance = row.get("relevance")
    if relevance is None:
        return "<span class='mini-badge relevance-none'>Nao julgado</span>"
    if bool(row.get("is_relevant")):
        return f"<span class='mini-badge relevance-good'>Relevante ({relevance})</span>"
    return f"<span class='mini-badge relevance-bad'>Nao relevante ({relevance})</span>"


def result_card_html(
    row: dict[str, object],
    model: str,
    shared_positions: dict[str, dict[str, int]] | None = None,
) -> str:
    color = MODEL_COLORS.get(model, "#2563EB")
    soft = MODEL_SOFT_COLORS.get(model, "#DBEAFE")
    doc_id = str(row["document_id"])
    overlap = ""
    if shared_positions and doc_id in shared_positions:
        positions = shared_positions[doc_id]
        pieces = [
            f"{MODEL_LABELS.get(other_model, other_model)}: #{position}"
            for other_model, position in positions.items()
        ]
        overlap = (
            "<div class='overlap'>"
            f"Aparece em {len(positions)}/4 modelos<br>{escape(' / '.join(pieces))}"
            "</div>"
        )
    return _clean_html(
        f"""
    <article class="result-card" style="border-top:4px solid {color}">
      <div class="result-top">
        <div class="rank-circle" style="background:{color}">#{int(row['position'])}</div>
        <div>
          <div class="result-title">{escape(str(row['title']))}</div>
          <div class="snippet">{escape(str(row['snippet']))}</div>
        </div>
      </div>
      <div class="result-meta">
        <span class="mini-badge" style="background:{soft};border-color:{color};color:{color}">Score {float(row['score']):.4f}</span>
        <span class="mini-badge">ID: {escape(doc_id)}</span>
        {_relevance_badge(row)}
      </div>
      {overlap}</article>
    """
    )


def single_results(model: str, rows: list[dict[str, object]]) -> None:
    cards = "".join(result_card_html(row, model) for row in rows)
    st.markdown(f"<div class='results-list'>{cards}</div>", unsafe_allow_html=True)


def comparison_results(
    rows_by_model: dict[str, list[dict[str, object]]],
    shared_positions: dict[str, dict[str, int]],
) -> None:
    columns = []
    for model, rows in rows_by_model.items():
        cards = "".join(result_card_html(row, model, shared_positions) for row in rows)
        color = MODEL_COLORS[model]
        columns.append(
            _clean_html(
                f"""
            <section class="model-column">
              <div class="model-column-head" style="border-top:4px solid {color}">
                <div class="model-column-title">
                  <span>{MODEL_LABELS[model]}</span>
                  <span class="mini-badge" style="color:{color};border-color:{color}">Top {len(rows)}</span>
                </div>
                <div class="model-column-desc">{MODEL_DESCRIPTIONS[model]}</div>
              </div>
              <div class="model-column-body">
                <div class="results-list">{cards}</div>
              </div>
            </section>
            """
            )
        )
    st.markdown(f"<div class='compare-grid'>{''.join(columns)}</div>", unsafe_allow_html=True)


def benchmark_cards(best: pd.DataFrame) -> None:
    cards = []
    for _, row in best.iterrows():
        cards.append(
            _clean_html(
                f"""
            <div class="summary-card">
              <div class="summary-label">{escape(str(row['dataset']))}</div>
              <div class="summary-main">Melhor nDCG@10</div>
              <div class="summary-value">{float(row['ndcg@10']):.4f}</div>
              <div class="status-text">{escape(str(row['model']))}</div>
            </div>
            """
            )
        )
    st.markdown(f"<div class='summary-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def improvement_cards(results: pd.DataFrame) -> None:
    desired = [
        ("nfcorpus", "hybrid", "bm25", "Hybrid vs BM25 no NFCorpus"),
        ("trec-covid", "hybrid_reranker", "bm25", "Hybrid + Reranker vs BM25 no TREC-COVID"),
    ]
    cards = []
    for dataset, model, baseline, label in desired:
        subset = results[results["dataset"] == dataset]
        if subset.empty:
            continue
        model_rows = subset[subset["model"] == model]
        base_rows = subset[subset["model"] == baseline]
        if model_rows.empty or base_rows.empty:
            continue
        value = float(model_rows.iloc[0]["ndcg@10"])
        base = float(base_rows.iloc[0]["ndcg@10"])
        if base == 0:
            continue
        delta = (value - base) / base * 100.0
        cards.append(
            _clean_html(
                f"""
            <div class="summary-card">
              <div class="summary-label">{escape(label)}</div>
              <div class="summary-value">{delta:+.2f}%</div>
              <div class="status-text">Diferenca relativa em nDCG@10</div>
            </div>
            """
            )
        )
    if cards:
        st.markdown(f"<div class='summary-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def case_cards() -> None:
    cards = []
    for key, case in CASE_DEFINITIONS.items():
        cards.append(
            _clean_html(
                f"""
            <div class="case-card" style="border-top-color:{case['color']}">
              <div class="case-title">{escape(case['title'])}</div>
              <div class="case-desc">{escape(case['description'])}</div>
            </div>
            """
            )
        )
    st.markdown(f"<div class='case-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def pipeline_diagram() -> None:
    steps = [
        ("Pergunta", "Texto digitado pelo usuario ou query do benchmark"),
        ("BM25", "Palavras e termos exatos"),
        ("Dense", "Significado por embeddings"),
        ("Hybrid", "Combinacao dos dois sinais"),
        ("Top 20", "Candidatos para reordenacao"),
        ("CrossEncoder", "Pontua pares query-documento"),
    ]
    cards = [
        _clean_html(
            f"""
        <div class="method-card">
          <div class="method-title">{escape(title)}</div>
          <div class="method-text">{escape(text)}</div>
        </div>
        """
        )
        for title, text in steps
    ]
    st.markdown(f"<div class='pipeline'>{''.join(cards)}</div>", unsafe_allow_html=True)
