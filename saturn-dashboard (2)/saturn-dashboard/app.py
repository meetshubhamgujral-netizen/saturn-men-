"""Saturn – AI Market Understanding Dashboard.

Entry point. Run with:  ``streamlit run app.py``
Streamlit auto-discovers the multipage navigation from the ``pages/`` folder.
"""
from __future__ import annotations

import os
import sys

# Ensure the project root is importable no matter how the host launches the
# script (local, Streamlit Cloud, etc.). Harmless if it is already on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from components.kpi import kpi_row
from utils import data_loader as dl
from utils.theme import hero, init_state, inject_theme

st.set_page_config(
    page_title="Saturn AI Market Dashboard",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
init_state()

hero(
    "Saturn AI Market Understanding Dashboard",
    "Descriptive · Diagnostic · Machine Learning · Explainable AI · Conversational — "
    "all driven exclusively by the dataset you upload.",
)

if st.session_state.get("df") is None:
    st.subheader("Get started")
    st.write(
        "Upload your dataset to unlock the full analytics suite. "
        "Use the **📁 Upload Dataset** page in the sidebar, or the button below."
    )
    if st.button("📁 Go to Upload Dataset", type="primary", use_container_width=False):
        st.switch_page("pages/1_📁_Upload_Dataset.py")

    st.markdown("---")
    st.markdown("#### What you can do here")
    feat_cols = st.columns(3)
    features = [
        ("🔎 Data Explorer", "Profile types, preview rows, inspect missingness."),
        ("🧹 Data Cleaning", "Impute, dedupe, treat outliers, encode, scale."),
        ("📊 Descriptive Analytics", "Stats, correlations, distributions, pivots."),
        ("📈 Diagnostic Analytics", "Chi-square, ANOVA, t-tests, PCA, drivers."),
        ("🤖 Machine Learning", "Auto task detection, model zoo, comparison."),
        ("🧠 Explainable AI", "Permutation importance, SHAP, prediction breakdown."),
        ("💬 AI Data Chat", "Ask questions answered only from your data."),
        ("📑 Reports", "Export cleaned data, predictions and PDF reports."),
        ("⚙️ Settings", "Theme, reset, and session controls."),
    ]
    for i, (title, desc) in enumerate(features):
        with feat_cols[i % 3]:
            st.markdown(
                f"<div class='section-card'><b>{title}</b><br>"
                f"<span style='opacity:.8;font-size:.9rem'>{desc}</span></div>",
                unsafe_allow_html=True,
            )
else:
    df = st.session_state["df"]
    prof = dl.profile_dataframe(df)
    st.subheader(f"Loaded: {st.session_state.get('filename', 'dataset')}")
    kpi_row(
        [
            {"label": "Rows", "value": f"{prof['rows']:,}", "icon": "📦"},
            {"label": "Columns", "value": prof["columns"], "icon": "🧱"},
            {"label": "Numeric", "value": len(prof["numeric_cols"]), "icon": "🔢"},
            {"label": "Categorical", "value": len(prof["categorical_cols"]), "icon": "🏷️"},
            {"label": "Missing", "value": f"{prof['missing_pct']}%", "icon": "🕳️"},
            {"label": "Duplicates", "value": prof["duplicate_rows"], "icon": "♻️"},
        ]
    )
    st.markdown("---")
    st.markdown("##### Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption("Use the sidebar to move through the analytics modules.")
