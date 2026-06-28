"""Upload page: accept CSV/Excel/JSON and profile the dataset."""
from __future__ import annotations

import os
import sys

# Make the project root importable on every host (incl. Streamlit Cloud).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from components.kpi import kpi_row
from utils import data_loader as dl
from utils.theme import hero, init_state, inject_theme

st.set_page_config(page_title="Upload · Saturn", page_icon="📁", layout="wide")
inject_theme()
init_state()
hero("Upload Dataset", "Bring your own data — CSV, Excel or JSON. Nothing is pre-loaded.")

uploaded = st.file_uploader(
    "Drop a file or browse", type=["csv", "tsv", "xlsx", "xls", "xlsm", "json"]
)

if uploaded is not None:
    try:
        with st.spinner("Reading and profiling your dataset…"):
            df = dl.load_dataframe(uploaded, uploaded.name)
            st.session_state["raw_df"] = df.copy()
            st.session_state["df"] = df.copy()
            st.session_state["filename"] = uploaded.name
            st.session_state["clean_log"] = []
            st.session_state["last_model"] = None
        st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {df.shape[1]} columns.")
    except Exception as exc:  # robust error surface
        st.error(f"Could not read this file: {exc}")
        st.stop()

if st.session_state.get("df") is not None:
    df = st.session_state["df"]
    prof = dl.profile_dataframe(df)
    st.markdown("### Dataset information")
    kpi_row(
        [
            {"label": "Rows", "value": f"{prof['rows']:,}", "icon": "📦"},
            {"label": "Columns", "value": prof["columns"], "icon": "🧱"},
            {"label": "Missing", "value": f"{prof['missing_values']:,}", "icon": "🕳️"},
            {"label": "Duplicates", "value": prof["duplicate_rows"], "icon": "♻️"},
            {"label": "Memory", "value": f"{prof['memory_mb']} MB", "icon": "💾"},
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Detected column types")
        st.markdown(
            "".join(f"<span class='badge'>🔢 {c}</span>" for c in prof["numeric_cols"])
            or "_none_",
            unsafe_allow_html=True,
        )
        st.markdown(
            "".join(f"<span class='badge'>🏷️ {c}</span>" for c in prof["categorical_cols"])
            or "",
            unsafe_allow_html=True,
        )
        if prof["datetime_cols"]:
            st.markdown(
                "".join(f"<span class='badge'>📅 {c}</span>" for c in prof["datetime_cols"]),
                unsafe_allow_html=True,
            )
        if prof["boolean_cols"]:
            st.markdown(
                "".join(f"<span class='badge'>✅ {c}</span>" for c in prof["boolean_cols"]),
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown("#### Suggested target variables")
        targets = dl.suggest_targets(df)
        if targets:
            st.markdown(
                "".join(f"<span class='badge'>🎯 {c}</span>" for c in targets[:12]),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No obvious target detected — clustering may suit this data.")

    st.markdown("#### Preview")
    st.dataframe(df.head(50), use_container_width=True)
else:
    st.info("Upload a file above to continue.")
