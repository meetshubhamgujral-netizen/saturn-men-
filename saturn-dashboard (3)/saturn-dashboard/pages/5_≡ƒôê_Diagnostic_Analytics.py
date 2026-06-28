"""Diagnostic Analytics: statistical tests, PCA, drivers, segment analysis."""
from __future__ import annotations

import os
import sys

# Make the project root importable on every host (incl. Streamlit Cloud).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from utils import data_loader as dl
from utils import diagnostics as dg
from utils.theme import hero, init_state, inject_theme, require_data

st.set_page_config(page_title="Diagnostic · Saturn", page_icon="📈", layout="wide")
inject_theme()
init_state()
hero("Diagnostic Analytics", "Why do the patterns exist? Tests, drivers and structure.")

if not require_data():
    st.stop()

df = st.session_state["df"]
prof = dl.profile_dataframe(df)
num_cols = prof["numeric_cols"]
cat_cols = prof["categorical_cols"] + prof["boolean_cols"]

tabs = st.tabs(
    ["🎯 Drivers", "🧪 Chi-Square", "📊 ANOVA", "⚖️ T-Test", "🧬 PCA", "🧩 Segments"]
)

with tabs[0]:
    if num_cols or cat_cols:
        target = st.selectbox("Target", df.columns.tolist())
        feats = st.multiselect(
            "Features", [c for c in df.columns if c != target],
            default=[c for c in df.columns if c != target][:10],
        )
        if st.button("Compute drivers") and feats:
            with st.spinner("Fitting random forest…"):
                imp = dg.feature_importance(df, target, feats)
            fig = dg.importance_figure(imp)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(imp, use_container_width=True)
            else:
                st.info("Not enough complete data to estimate importance.")

with tabs[1]:
    if len(cat_cols) >= 2:
        a = st.selectbox("Variable A", cat_cols, key="chi_a")
        b = st.selectbox("Variable B", cat_cols, index=1, key="chi_b")
        if st.button("Run chi-square"):
            res = dg.chi_square_test(df, a, b)
            st.metric("p-value", f"{res['p_value']:.4f}")
            st.info(res["narrative"])
    else:
        st.info("Need at least two categorical columns.")

with tabs[2]:
    if cat_cols and num_cols:
        g = st.selectbox("Group", cat_cols, key="anova_g")
        v = st.selectbox("Numeric value", num_cols, key="anova_v")
        if st.button("Run ANOVA"):
            res = dg.anova_test(df, g, v)
            st.info(res["narrative"])
    else:
        st.info("Need a categorical group and a numeric value.")

with tabs[3]:
    if cat_cols and num_cols:
        g = st.selectbox("Binary group", cat_cols, key="tt_g")
        v = st.selectbox("Numeric value", num_cols, key="tt_v")
        if st.button("Run t-test"):
            res = dg.t_test(df, g, v)
            st.info(res["narrative"])

with tabs[4]:
    n = st.slider("Components", 2, max(2, min(10, len(num_cols))), 2)
    if st.button("Run PCA"):
        res = dg.run_pca(df, n)
        if res.get("figure"):
            st.plotly_chart(res["figure"], use_container_width=True)
        st.info(res["narrative"])

with tabs[5]:
    if cat_cols and num_cols:
        s = st.selectbox("Segment", cat_cols, key="seg_s")
        v = st.selectbox("Metric", num_cols, key="seg_v")
        st.dataframe(dg.segment_analysis(df, s, v), use_container_width=True)
