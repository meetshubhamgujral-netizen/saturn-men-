"""Data Explorer: profiling, dtypes, missingness, filtering."""
from __future__ import annotations

import os
import sys

# Make the project root importable on every host (incl. Streamlit Cloud).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from utils import analytics as an
from utils import data_loader as dl
from utils.theme import hero, init_state, inject_theme, require_data

st.set_page_config(page_title="Explorer · Saturn", page_icon="🔎", layout="wide")
inject_theme()
init_state()
hero("Data Explorer", "Profile, preview and interrogate your dataset.")

if not require_data():
    st.stop()

df = st.session_state["df"]
prof = dl.profile_dataframe(df)

tab_overview, tab_types, tab_missing, tab_filter = st.tabs(
    ["📋 Overview", "🧩 Types", "🕳️ Missingness", "🔍 Filter & Preview"]
)

with tab_overview:
    st.dataframe(
        dl.missing_table(df).rename(columns={"missing_pct": "missing %"}),
        use_container_width=True,
    )

with tab_types:
    type_rows = [{"column": c, "dtype": t} for c, t in prof["dtypes"].items()]
    st.dataframe(type_rows, use_container_width=True)

with tab_missing:
    st.plotly_chart(an.missing_heatmap(df), use_container_width=True)

with tab_filter:
    cat_cols = prof["categorical_cols"] + prof["boolean_cols"]
    work = df.copy()
    if cat_cols:
        fcol = st.selectbox("Filter by column", ["(none)"] + cat_cols)
        if fcol != "(none)":
            vals = st.multiselect(
                "Keep values", sorted(df[fcol].dropna().astype(str).unique())
            )
            if vals:
                work = work[work[fcol].astype(str).isin(vals)]
    st.caption(f"{len(work):,} row(s) after filtering.")
    st.dataframe(work.head(200), use_container_width=True)
