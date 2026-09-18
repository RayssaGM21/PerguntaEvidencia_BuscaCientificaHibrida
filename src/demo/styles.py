from __future__ import annotations


APP_CSS = """
<style>
:root {
    --blue: #2563EB;
    --blue-dark: #0F172A;
    --blue-mid: #3B82F6;
    --blue-light: #DBEAFE;
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --text: #0F172A;
    --muted: #64748B;
    --border: #E2E8F0;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 30rem),
        linear-gradient(180deg, #F8FAFC 0%, #EEF4FF 100%);
    color: var(--text);
}

.block-container {
    max-width: 1440px;
    padding: 1.0rem 1.6rem 2.4rem;
}

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--border);
    box-shadow: 8px 0 30px rgba(15, 23, 42, 0.04);
}

section[data-testid="stSidebar"] * {
    color: var(--text);
}

div[data-testid="stSidebarUserContent"] {
    padding-top: 1rem;
}

h1, h2, h3, h4, p, label, span, div {
    letter-spacing: 0;
}

.hero {
    display: grid;
    grid-template-columns: 148px minmax(0, 1fr) auto;
    gap: 1.25rem;
    align-items: center;
    padding: 0.7rem 1.15rem 0.7rem 0.8rem;
    margin-bottom: 0.9rem;
    border: 1px solid rgba(37, 99, 235, 0.18);
    border-radius: 16px;
    background:
        linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(255, 255, 255, 0.86)),
        #FFFFFF;
    box-shadow: 0 14px 36px rgba(15, 23, 42, 0.065);
}

.hero-logo-wrap {
    display: grid;
    place-items: center;
    align-self: stretch;
}

.hero-logo {
    display: block;
    width: 100%;
    max-width: 148px;
    height: auto;
    object-fit: contain;
    filter: drop-shadow(0 10px 18px rgba(15, 23, 42, 0.12));
}

.hero-content {
    min-width: 0;
}

.hero-title {
    margin: 0;
    color: var(--blue-dark);
    font-size: clamp(1.9rem, 3vw, 3.0rem);
    line-height: 1.02;
    font-weight: 850;
}

.hero-subtitle {
    color: #334155;
    font-size: clamp(0.96rem, 1.15vw, 1.08rem);
    margin: 0.55rem 0 0.75rem;
    max-width: 820px;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.badge {
    display: inline-flex;
    align-items: center;
    min-height: 1.65rem;
    padding: 0.26rem 0.56rem;
    border-radius: 999px;
    border: 1px solid rgba(37, 99, 235, 0.18);
    background: #FFFFFF;
    color: #1E40AF;
    font-size: 0.74rem;
    font-weight: 750;
}

.event-mark {
    color: var(--muted);
    font-weight: 750;
    white-space: nowrap;
    padding: 0.35rem 0.55rem;
    border-radius: 999px;
    background: rgba(37, 99, 235, 0.08);
    color: #1E40AF;
    font-size: 0.84rem;
}

.panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
    padding: 1rem;
}

.sidebar-block {
    margin-bottom: 1rem;
    padding: 0.9rem;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: #F8FAFC;
}

.sidebar-title {
    color: var(--blue-dark);
    font-size: 0.76rem;
    font-weight: 850;
    text-transform: uppercase;
    margin: 0.65rem 0 0.45rem;
}

.dataset-stat {
    margin-top: 0.5rem;
    color: var(--muted);
    font-size: 0.8rem;
    line-height: 1.35;
}

.model-list {
    display: grid;
    gap: 0.42rem;
    margin-top: 0.35rem;
}

.model-chip {
    border-radius: 13px;
    padding: 0.58rem 0.68rem;
    border: 1px solid var(--border);
    background: #FFFFFF;
}

.model-chip-title {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    font-weight: 820;
}

.model-dot {
    width: 0.66rem;
    height: 0.66rem;
    border-radius: 999px;
    display: inline-block;
}

.model-chip-desc {
    color: var(--muted);
    font-size: 0.74rem;
    margin-top: 0.18rem;
}

div[data-baseweb="tab-list"] {
    gap: 0.2rem;
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.25rem;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
}

div[data-testid="stButtonGroup"] {
    width: 100%;
    margin-bottom: 0.8rem;
}

div[data-testid="stButtonGroup"] [role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    width: 100%;
    gap: 0.2rem;
    padding: 0.25rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #FFFFFF;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
}

div[data-testid="stButtonGroup"] button {
    width: 100% !important;
    min-height: 2.45rem;
    border-radius: 10px;
    color: var(--muted);
    font-weight: 800;
}

div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {
    color: #FFFFFF;
    background: var(--blue);
}

button[data-baseweb="tab"] {
    border-radius: 10px;
    color: var(--muted);
    font-weight: 800;
    padding: 0.35rem 0.65rem;
    min-height: 2.45rem;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--blue);
    color: #FFFFFF;
}

.search-shell {
    display: grid;
    gap: 0.62rem;
    margin: 0.75rem 0 0.95rem;
}

.example-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    align-items: center;
    color: var(--muted);
    font-size: 0.9rem;
}

.status-box {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.74rem 0.86rem;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: #FFFFFF;
    margin: 0.72rem 0;
}

.status-icon {
    min-width: 2.35rem;
    height: 1.55rem;
    display: grid;
    place-items: center;
    border-radius: 999px;
    font-weight: 900;
    font-size: 0.68rem;
}

.status-title {
    color: var(--blue-dark);
    font-weight: 850;
}

.status-text {
    color: var(--muted);
    margin-top: 0.1rem;
    font-size: 0.92rem;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(150px, 1fr));
    gap: 0.7rem;
    margin: 0.8rem 0;
}

.metric-card {
    padding: 0.75rem;
    border-radius: 15px;
    background: #FFFFFF;
    border: 1px solid var(--border);
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
}

.metric-label {
    color: var(--muted);
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
}

.metric-value {
    color: var(--blue-dark);
    font-size: 1.45rem;
    font-weight: 850;
    margin-top: 0.15rem;
}

.compare-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
    gap: 0.85rem;
    align-items: start;
}

.model-column {
    border-radius: 15px;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.86);
    box-shadow: 0 14px 35px rgba(15, 23, 42, 0.06);
    overflow: hidden;
}

.model-column-head {
    padding: 0.78rem 0.88rem;
    border-bottom: 1px solid var(--border);
    background: #FFFFFF;
}

.model-column-title {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: center;
    font-weight: 900;
    color: var(--blue-dark);
}

.model-column-desc {
    color: var(--muted);
    font-size: 0.84rem;
    margin-top: 0.15rem;
}

.model-column-body {
    padding: 0.72rem;
    max-height: 72vh;
    overflow-y: auto;
}

.results-list {
    display: grid;
    gap: 0.68rem;
}

.result-card {
    position: relative;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: #FFFFFF;
    padding: 0.78rem;
    min-height: 140px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.052);
    transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
}

.result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 38px rgba(15, 23, 42, 0.10);
}

.result-top {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 0.6rem;
    align-items: start;
}

.rank-circle {
    width: 2rem;
    height: 2rem;
    display: grid;
    place-items: center;
    border-radius: 999px;
    color: #FFFFFF;
    font-weight: 900;
    flex: 0 0 auto;
}

.result-title {
    color: var(--blue-dark);
    font-weight: 850;
    line-height: 1.25;
    font-size: 0.92rem;
    overflow-wrap: anywhere;
}

.snippet {
    color: #475569;
    line-height: 1.48;
    font-size: 0.84rem;
    margin-top: 0.45rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.result-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
    margin-top: 0.62rem;
}

.mini-badge {
    border-radius: 999px;
    padding: 0.24rem 0.46rem;
    font-size: 0.7rem;
    font-weight: 800;
    border: 1px solid var(--border);
    background: #F8FAFC;
    color: #334155;
}

.relevance-good {
    background: #D1FAE5;
    border-color: #A7F3D0;
    color: #047857;
}

.relevance-bad {
    background: #FEE2E2;
    border-color: #FECACA;
    color: #B91C1C;
}

.relevance-none {
    background: #F1F5F9;
    border-color: #E2E8F0;
    color: #475569;
}

.overlap {
    margin-top: 0.55rem;
    padding: 0.48rem 0.55rem;
    border-radius: 12px;
    background: #F8FAFC;
    color: #334155;
    font-size: 0.78rem;
    line-height: 1.42;
}

.section-title {
    color: var(--blue-dark);
    font-size: 1.18rem;
    font-weight: 900;
    margin: 0.8rem 0 0.45rem;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.summary-card, .case-card, .method-card {
    border-radius: 16px;
    border: 1px solid var(--border);
    background: #FFFFFF;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    padding: 1rem;
}

.summary-label {
    color: var(--muted);
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
}

.summary-main {
    color: var(--blue-dark);
    font-size: 1.25rem;
    font-weight: 900;
    margin-top: 0.15rem;
}

.summary-value {
    color: var(--blue);
    font-size: 1.8rem;
    font-weight: 900;
    margin-top: 0.25rem;
}

.case-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(190px, 1fr));
    gap: 0.85rem;
}

.case-card {
    min-height: 142px;
    border-top: 4px solid var(--blue);
}

.case-title {
    font-weight: 900;
    color: var(--blue-dark);
    margin-bottom: 0.3rem;
}

.case-desc {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.42;
}

.pipeline {
    display: grid;
    grid-template-columns: repeat(6, minmax(120px, 1fr));
    gap: 0.75rem;
    align-items: stretch;
    margin: 1rem 0;
}

.method-card {
    text-align: center;
    min-height: 126px;
    display: grid;
    align-content: center;
}

.method-title {
    color: var(--blue-dark);
    font-weight: 900;
}

.method-text {
    color: var(--muted);
    font-size: 0.88rem;
    margin-top: 0.25rem;
}

div.stButton > button {
    border-radius: 12px;
    border: 1px solid var(--border);
    background: #FFFFFF;
    color: var(--blue-dark);
    font-weight: 800;
    transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    border-color: var(--blue);
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.16);
    color: var(--blue);
}

div.stButton > button[kind="primary"] {
    background: var(--blue);
    color: #FFFFFF;
    border-color: var(--blue);
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
}

div.stButton > button[kind="primary"]:hover {
    background: #1D4ED8;
    color: #FFFFFF;
}

div[data-testid="stTextInput"] input {
    border-radius: 13px;
    border: 1px solid var(--border);
    background: #FFFFFF;
    color: var(--blue-dark);
    min-height: 2.85rem;
    font-size: 0.96rem;
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}

div[data-baseweb="select"] > div,
div[data-baseweb="select"] div {
    background-color: #FFFFFF;
    color: var(--blue-dark);
}

div[data-baseweb="select"] > div {
    border-color: var(--border);
    border-radius: 12px;
}

div[data-baseweb="select"] input {
    color: var(--blue-dark);
}

label[data-testid="stWidgetLabel"],
div[data-testid="stCaptionContainer"],
div[data-testid="stMarkdownContainer"] p {
    color: var(--text);
}

div[data-testid="stCheckbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stToggle"] label {
    color: var(--text);
}

div[data-testid="stCheckbox"] input,
div[data-testid="stToggle"] input,
div[data-testid="stRadio"] input {
    accent-color: var(--blue);
}

div[data-testid="stCheckbox"] svg,
div[data-testid="stToggle"] svg {
    color: var(--blue);
    fill: var(--blue);
}

div[data-testid="stHorizontalBlock"] {
    gap: 0.65rem;
}

@media (max-width: 1450px) {
    .compare-grid { grid-template-columns: repeat(2, minmax(300px, 1fr)); }
    .case-grid { grid-template-columns: repeat(3, minmax(190px, 1fr)); }
}

@media (max-width: 900px) {
    .block-container { padding: 1rem; }
    .hero {
        grid-template-columns: 104px minmax(0, 1fr);
        gap: 0.85rem;
        padding: 0.7rem;
    }
    .hero-logo { max-width: 104px; }
    .event-mark { grid-column: 1 / -1; justify-self: start; }
    div[data-testid="stButtonGroup"] [role="radiogroup"] {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    div[data-testid="stButtonGroup"] button:last-child {
        grid-column: 1 / -1;
    }
    .compare-grid, .summary-grid, .metrics-grid, .pipeline, .case-grid {
        grid-template-columns: 1fr;
    }
    .model-column-body { max-height: none; }
}

@media (max-width: 620px) {
    .hero-title { font-size: 2rem; }
    .hero {
        grid-template-columns: 82px minmax(0, 1fr);
        padding: 0.65rem;
        border-radius: 14px;
    }
    .hero-logo { max-width: 82px; }
    .hero-subtitle { font-size: 0.86rem; }
    .result-card { min-height: auto; }
}
</style>
"""
