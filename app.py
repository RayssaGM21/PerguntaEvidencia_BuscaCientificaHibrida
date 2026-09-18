from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import Paths
from src.demo.components import (
    benchmark_cards,
    case_cards,
    comparison_results,
    improvement_cards,
    metrics_grid,
    pipeline_diagram,
    ranking_pulse,
    render_header,
    render_model_reference,
    sidebar_dataset_stat,
    single_results,
    status_box,
)
from src.demo.constants import CASE_DEFINITIONS, DATASET_STATS, EXAMPLE_QUERIES, MODEL_COLORS
from src.demo.search_service import (
    HYBRID_ALPHA,
    MODEL_LABELS,
    MODEL_ORDER,
    RERANKER_CANDIDATE_POOL,
    benchmark_ndcg_table,
    best_ndcg_by_dataset,
    build_dense_resources,
    cosine_device_label,
    experiment_case_candidates,
    find_matching_query_id,
    fit_bm25,
    format_results,
    load_benchmark_results,
    load_dataset_bundle,
    load_per_query_results,
    query_metrics,
    rankings_for_existing_query,
    rankings_for_free_query,
    shared_document_positions,
)
from src.demo.styles import APP_CSS
from src.retrievers.dense import DEFAULT_DENSE_MODEL
from src.retrievers.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker


st.set_page_config(
    page_title="Scientific IR Benchmark Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_dataset(dataset: str):
    return load_dataset_bundle(dataset, "test", Paths())


@st.cache_resource(show_spinner=False)
def cached_bm25(dataset: str):
    bundle = cached_dataset(dataset)
    return fit_bm25(bundle.corpus)


@st.cache_resource(show_spinner=False)
def cached_dense_resources(dataset: str, batch_size: int = 256):
    bundle = cached_dataset(dataset)
    return build_dense_resources(
        corpus=bundle.corpus,
        dataset=dataset,
        cache_dir=Paths().embeddings_dir,
        model_name=DEFAULT_DENSE_MODEL,
        device=None,
        batch_size=batch_size,
    )


@st.cache_resource(show_spinner=False)
def cached_reranker():
    return CrossEncoderReranker(model_name=DEFAULT_RERANKER_MODEL, device=None, batch_size=32)


@st.cache_data(show_spinner=False)
def cached_benchmark_results() -> pd.DataFrame:
    return load_benchmark_results(Paths())


@st.cache_data(show_spinner=False)
def cached_per_query_results() -> pd.DataFrame:
    return load_per_query_results(Paths())


def selected_model_keys(compare: bool) -> list[str]:
    if compare:
        keys: list[str] = []
        st.markdown('<div class="sidebar-title">Modelos</div>', unsafe_allow_html=True)
        for model in MODEL_ORDER:
            checked = st.checkbox(MODEL_LABELS[model], value=True, key=f"model_{model}")
            if checked:
                keys.append(model)
        render_model_reference(keys)
        return keys

    st.markdown('<div class="sidebar-title">Modelo</div>', unsafe_allow_html=True)
    label = st.radio(
        "Escolha um modelo",
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        index=2,
        label_visibility="collapsed",
    )
    key = next(model for model in MODEL_ORDER if MODEL_LABELS[model] == label)
    render_model_reference([key])
    return [key]


def run_search(
    dataset: str,
    query_text: str,
    models: list[str],
    top_k: int,
    compare: bool,
) -> None:
    bundle = cached_dataset(dataset)
    matching_query_id = find_matching_query_id(query_text, bundle.queries)
    has_ground_truth = matching_query_id is not None

    if has_ground_truth:
        status_box(
            "success",
            "Query reconhecida no benchmark",
            "Metricas e julgamentos de relevancia estao disponiveis para esta consulta.",
        )
    else:
        status_box(
            "info",
            "Consulta livre",
            "Esta pergunta nao pertence ao benchmark. Os resultados podem ser explorados, mas nao ha qrels para calcular metricas.",
        )

    if not models:
        st.warning("Escolha pelo menos um modelo.")
        return

    if not has_ground_truth and bundle.benchmark_only:
        status_box(
            "info",
            "TREC-COVID em modo benchmark",
            "Na versao publicada, este dataset usa os rankings pre-calculados. Digite uma query existente no benchmark para explorar os resultados sem recalcular o corpus completo na nuvem.",
        )
        return

    with st.spinner("Explorando documentos e preparando rankings..."):
        if has_ground_truth:
            model_rankings = rankings_for_existing_query(
                Paths(),
                dataset,
                matching_query_id,
                models,
                split="test",
            )
            qrels_for_query = bundle.qrels.get(matching_query_id, {})
            display_query = bundle.queries[matching_query_id]
        else:
            bm25 = cached_bm25(dataset)
            needs_dense = any(model in models for model in ("dense", "hybrid", "hybrid_reranker"))
            dense_resources = cached_dense_resources(dataset) if needs_dense else None
            reranker = cached_reranker() if "hybrid_reranker" in models else None
            model_rankings = rankings_for_free_query(
                query_text,
                bundle.corpus,
                bm25,
                dense_resources,
                reranker,
                models,
                top_k=100,
            )
            qrels_for_query = None
            display_query = query_text

    st.markdown(f"<div class='section-title'>Resultados para: {display_query}</div>", unsafe_allow_html=True)

    metrics_by_model = None
    if has_ground_truth:
        metrics_by_model = {
            model: query_metrics(ranking, qrels_for_query)
            for model, ranking in model_rankings.items()
        }
        metrics_grid(metrics_by_model)

    rows_by_model = {
        model: format_results(ranking, bundle.corpus, qrels_for_query=qrels_for_query, top_k=top_k)
        for model, ranking in model_rankings.items()
    }
    if compare:
        ranking_pulse(rows_by_model, metrics_by_model)
        comparison_results(rows_by_model, shared_document_positions(rows_by_model))
    else:
        model = models[0]
        single_results(model, rows_by_model[model])


def set_search_example_from_widget(state_key: str, widget_key: str) -> None:
    example = st.session_state.get(widget_key)
    if example:
        st.session_state[state_key] = example


def search_bar(dataset: str, state_key: str, default: str, button_label: str) -> tuple[str, bool]:
    bundle = cached_dataset(dataset)
    if dataset == "trec-covid":
        query_ids = list(bundle.queries)
        selected_query_id = st.selectbox(
            "Escolha uma pergunta oficial do TREC-COVID",
            query_ids,
            format_func=lambda query_id: f"Q{query_id} - {bundle.queries[query_id]}",
            key=f"{state_key}_benchmark_query",
        )
        clicked = st.button(
            button_label,
            type="primary",
            use_container_width=True,
            key=f"{state_key}_submit",
        )
        st.caption("Consulta oficial do benchmark: qrels e metricas por query estao disponiveis.")
        return bundle.queries[selected_query_id], clicked

    if state_key not in st.session_state:
        st.session_state[state_key] = default

    input_col, button_col = st.columns([5, 1])
    with input_col:
        query_text = st.text_input(
            "Digite uma pergunta cientifica",
            key=state_key,
            placeholder="Digite uma pergunta cientifica...",
            label_visibility="collapsed",
        )
    with button_col:
        clicked = st.button(button_label, type="primary", use_container_width=True, key=f"{state_key}_submit")

    example_key = f"{state_key}_example"
    st.selectbox(
        "Experimente uma consulta",
        [""] + EXAMPLE_QUERIES,
        format_func=lambda value: value or "Escolha um exemplo rapido",
        key=example_key,
        on_change=set_search_example_from_widget,
        args=(state_key, example_key),
    )
    return query_text, clicked


def show_results_page() -> None:
    benchmark = cached_benchmark_results()
    st.markdown('<div class="section-title">Resultados consolidados do experimento</div>', unsafe_allow_html=True)
    if benchmark.empty:
        st.warning("Arquivo outputs/metrics/benchmark_results.csv nao encontrado.")
        return

    benchmark_cards(best_ndcg_by_dataset(benchmark))
    improvement_cards(benchmark)

    ndcg = benchmark_ndcg_table(benchmark)
    st.markdown('<div class="section-title">nDCG@10 por modelo e dataset</div>', unsafe_allow_html=True)
    chart_data = benchmark[["dataset", "model", "ndcg@10"]].copy()
    chart_data["Dataset"] = chart_data["dataset"].map(
        {key: value["label"] for key, value in DATASET_STATS.items()}
    )
    chart_data["Modelo"] = chart_data["model"].map(MODEL_LABELS)
    model_labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    model_colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
    bars = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Dataset:N", sort=[DATASET_STATS[key]["label"] for key in DATASET_STATS], title=None),
            xOffset=alt.XOffset("Modelo:N", sort=model_labels),
            y=alt.Y("ndCG:Q", title="nDCG@10", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Modelo:N",
                sort=model_labels,
                scale=alt.Scale(domain=model_labels, range=model_colors),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("Dataset:N"),
                alt.Tooltip("Modelo:N"),
                alt.Tooltip("ndCG:Q", title="nDCG@10", format=".4f"),
            ],
        )
        .transform_calculate(ndCG="datum['ndcg@10']")
    )
    labels = bars.mark_text(dy=-8, color="#334155", fontSize=11).encode(
        text=alt.Text("ndCG:Q", format=".3f")
    )
    st.altair_chart((bars + labels).properties(height=380), use_container_width=True)
    with st.expander("Ver valores completos"):
        st.dataframe(ndcg.style.format("{:.6f}"), use_container_width=True)


def show_cases_page(dataset: str, top_k: int) -> None:
    st.markdown('<div class="section-title">Casos do experimento</div>', unsafe_allow_html=True)
    case_cards()
    cases = experiment_case_candidates(dataset, cached_per_query_results(), limit=3)
    if not cases:
        st.warning("Resultados por query nao encontrados para montar os casos.")
        return

    case_labels = [CASE_DEFINITIONS[key]["title"] for key in CASE_DEFINITIONS]
    selected_label = st.selectbox("Tipo de caso", case_labels)
    selected_key = next(key for key, value in CASE_DEFINITIONS.items() if value["title"] == selected_label)
    options = cases.get(selected_key)
    if options is None or options.empty:
        st.info("Nao ha casos positivos para esta categoria no dataset selecionado.")
        return

    query_choices = [
        f"{row.query_id} · delta {row.delta:.4f}"
        for row in options.itertuples(index=False)
    ]
    selected = st.selectbox("Query real", query_choices)
    row = options.iloc[query_choices.index(selected)]

    bundle = cached_dataset(dataset)
    query_id = str(row["query_id"])
    query_text = bundle.queries.get(query_id, query_id)

    cards = []
    for model in MODEL_ORDER:
        if model in row and pd.notna(row[model]):
            cards.append(
                dedent(
                    f"""
                <div class="metric-card">
                  <div class="metric-label">{MODEL_LABELS[model]}</div>
                  <div class="metric-value">{float(row[model]):.3f}</div>
                  <div class="status-text">nDCG@10</div>
                </div>
                """
                ).strip()
            )
    st.markdown(f"<div class='metrics-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
    st.caption(f"Delta usado na selecao: {float(row['delta']):.4f}")

    rankings = rankings_for_existing_query(Paths(), dataset, query_id, MODEL_ORDER, split="test")
    qrels_for_query = bundle.qrels.get(query_id, {})
    st.markdown(f"<div class='section-title'>{query_text}</div>", unsafe_allow_html=True)
    rows_by_model = {
        model: format_results(ranking, bundle.corpus, qrels_for_query=qrels_for_query, top_k=top_k)
        for model, ranking in rankings.items()
    }
    comparison_results(rows_by_model, shared_document_positions(rows_by_model))


def show_how_it_works() -> None:
    st.markdown('<div class="section-title">Como funciona</div>', unsafe_allow_html=True)
    pipeline_diagram()
    st.markdown(
        """
        <div class="panel">
          <p><strong>BM25</strong> usa correspondencia lexical: documentos sobem quando compartilham termos importantes com a consulta.</p>
          <p><strong>Dense Retrieval</strong> transforma consultas e documentos em vetores e busca proximidade semantica.</p>
          <p><strong>Hybrid</strong> combina os sinais lexical e semantico com normalizacao por query e alpha congelado no conjunto dev.</p>
          <p><strong>CrossEncoder</strong> recebe pares query-documento dos melhores candidatos e reordena o ranking final.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(APP_CSS, unsafe_allow_html=True)
render_header()

with st.sidebar:
    st.markdown('<div class="sidebar-title">Dataset</div>', unsafe_allow_html=True)
    dataset_label = st.selectbox(
        "Dataset",
        [DATASET_STATS[key]["label"] for key in DATASET_STATS],
        label_visibility="collapsed",
    )
    dataset = next(key for key, value in DATASET_STATS.items() if value["label"] == dataset_label)
    sidebar_dataset_stat(dataset)

    st.divider()
    st.markdown('<div class="sidebar-title">Consulta</div>', unsafe_allow_html=True)
    compare = st.toggle("Modo comparacao", value=True)

    st.divider()
    models = selected_model_keys(compare)

    st.divider()
    st.markdown('<div class="sidebar-title">Quantidade de resultados</div>', unsafe_allow_html=True)
    top_k = st.radio("Top N", [5, 10, 20], index=1, horizontal=True, label_visibility="collapsed")

    st.divider()
    st.caption(f"Dense: {DEFAULT_DENSE_MODEL}")
    st.caption(f"Hybrid: weighted, alpha={HYBRID_ALPHA}")
    st.caption(f"Reranker: {DEFAULT_RERANKER_MODEL}, pool={RERANKER_CANDIDATE_POOL}")
    st.caption(f"Device detectado: {cosine_device_label()}")

pages = ["Buscar", "Comparar", "Casos", "Resultados", "Como Funciona"]
page = st.segmented_control(
    "Navegacao principal",
    pages,
    default="Buscar",
    key="main_page",
    label_visibility="collapsed",
    selection_mode="single",
) or "Buscar"

if page == "Buscar":
    query_text, clicked = search_bar(dataset, "search_query", "How are knowledge graphs used in education?", "Buscar")
    if clicked:
        st.session_state.last_query = query_text
        st.session_state.last_query_dataset = dataset
    if st.session_state.get("last_query_dataset") == dataset and "last_query" in st.session_state:
        run_search(dataset, st.session_state.last_query, models[:1], top_k, False)
    else:
        status_box(
            "info",
            "Pesquisa pronta para iniciar",
            "Escolha a consulta e clique em Buscar para executar o modelo selecionado.",
        )
elif page == "Comparar":
    st.markdown('<div class="section-title">Comparacao lado a lado</div>', unsafe_allow_html=True)
    query_text, clicked = search_bar(dataset, "compare_query", "How are knowledge graphs used in education?", "Comparar")
    if clicked:
        st.session_state.last_compare_query = query_text
        st.session_state.last_compare_dataset = dataset
    compare_models = models if compare else MODEL_ORDER
    if st.session_state.get("last_compare_dataset") == dataset and "last_compare_query" in st.session_state:
        run_search(dataset, st.session_state.last_compare_query, compare_models, top_k, True)
    else:
        status_box(
            "info",
            "Comparacao pronta para iniciar",
            "Escolha a consulta e clique em Comparar para executar os modelos selecionados.",
        )
elif page == "Casos":
    show_cases_page(dataset, top_k)
elif page == "Resultados":
    show_results_page()
else:
    show_how_it_works()
