"""UI theming and session-state helpers (the only Streamlit-aware util)."""
from __future__ import annotations

import streamlit as st

PRIMARY = "#6C5CE7"
ACCENT = "#00B894"

CUSTOM_CSS = """
<style>
:root {
  --saturn-primary: #6C5CE7;
  --saturn-accent: #00B894;
  --saturn-blue: #0984E3;
}
/* Hero banner */
.saturn-hero {
  background: linear-gradient(120deg, #6C5CE7 0%, #0984E3 50%, #00B894 100%);
  padding: 2.6rem 2.2rem; border-radius: 20px; color: #fff;
  box-shadow: 0 10px 30px rgba(108,92,231,0.35); margin-bottom: 1.4rem;
}
.saturn-hero h1 { font-size: 2.5rem; margin: 0 0 .4rem 0; font-weight: 800; letter-spacing:-.5px;}
.saturn-hero p  { font-size: 1.05rem; opacity: .95; margin: 0; }
/* KPI cards */
.kpi-card {
  border-radius: 16px; padding: 1.1rem 1.2rem; color: #fff;
  box-shadow: 0 6px 18px rgba(0,0,0,0.12); transition: transform .15s ease;
  height: 100%;
}
.kpi-card:hover { transform: translateY(-4px); }
.kpi-value { font-size: 1.8rem; font-weight: 800; line-height: 1.1; }
.kpi-label { font-size: .82rem; opacity: .92; text-transform: uppercase; letter-spacing: .5px;}
.kpi-icon  { font-size: 1.4rem; margin-bottom: .3rem; }
.badge {
  display:inline-block; padding:.2rem .7rem; border-radius:999px;
  background:rgba(108,92,231,.12); color:var(--saturn-primary);
  font-size:.78rem; font-weight:600; margin:.15rem;
}
.section-card{
  border:1px solid rgba(108,92,231,.15); border-radius:14px;
  padding:1rem 1.2rem; margin-bottom:1rem; background:rgba(108,92,231,.03);
}
div[data-testid="stMetricValue"] { color: var(--saturn-primary); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
</style>
"""


def inject_theme() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='saturn-hero'><h1>🪐 {title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def init_state() -> None:
    """Ensure the keys we rely on exist in session_state."""
    defaults = {
        "raw_df": None,        # originally uploaded data
        "df": None,            # working (possibly cleaned) data
        "filename": None,
        "clean_log": [],       # audit trail of cleaning steps
        "last_model": None,    # cached trained-model result
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def has_data() -> bool:
    return st.session_state.get("df") is not None


def require_data() -> bool:
    """Render a friendly notice and return False if no dataset is loaded."""
    if not has_data():
        st.info("📁 No dataset loaded yet. Head to **Upload Dataset** to get started.")
        return False
    return True
