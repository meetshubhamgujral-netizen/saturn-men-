"""Descriptive Analytics: stats, correlation, distributions, pivots, trends."""
from __future__ import annotations

import os
import sys

# Make the project root importable on every host (incl. Streamlit Cloud).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from utils import analytics as an
from utils import data_loader as dl
from utils.theme import hero, init_state, inject_theme, require_data

st.set_page_config(page_title="Descriptive · Saturn", page_icon="📊", layout="wide")
inject_theme()
init_state()
hero("Descriptive Analytics", "What does the data look like? Stats, shapes and breakdowns.")

if not require_data():
    st.stop()

df = st.session_state["df"]
prof = dl.profile_dataframe(df)
num_cols, cat_cols = prof["numeric_cols"], prof["categorical_cols"]
dt_cols = prof["datetime_cols"]

tabs = st.tabs(
    ["📐 Summary", "🔗 Correlation", "📊 Distributions", "🧮 Pivot & Frequency",
     "🏆 Top / Bottom", "📈 Trend"]
)

with tabs[0]:
    st.dataframe(an.summary_statistics(df), use_container_width=True)

with tabs[1]:
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    fig = an.correlation_heatmap(df, method)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Covariance matrix"):
            st.dataframe(an.covariance_matrix(df), use_container_width=True)
    else:
        st.info("Need at least two numeric columns.")

with tabs[2]:
    col = st.selectbox("Column", num_cols + cat_cols)
    if col:
        st.plotly_chart(an.distribution_figure(df, col), use_container_width=True)

with tabs[3]:
    if cat_cols:
        c = st.selectbox("Category counts for", cat_cols, key="catcounts")
        st.dataframe(an.category_counts(df, c), use_container_width=True)
    if cat_cols and num_cols:
        st.markdown("**Pivot table**")
        p1, p2, p3 = st.columns(3)
        idx = p1.selectbox("Index", cat_cols, key="piv_idx")
        val = p2.selectbox("Values", num_cols, key="piv_val")
        agg = p3.selectbox("Aggregate", ["mean", "sum", "median", "count", "max", "min"])
        st.dataframe(an.pivot_table(df, idx, val, agg), use_container_width=True)
    if len(cat_cols) >= 2:
        st.markdown("**Frequency cross-tab**")
        f1, f2 = st.columns(2)
        r = f1.selectbox("Rows", cat_cols, key="freq_r")
        cc = f2.selectbox("Columns", cat_cols, index=1, key="freq_c")
        st.dataframe(an.frequency_table(df, r, cc), use_container_width=True)

with tabs[4]:
    if cat_cols and num_cols:
        g = st.selectbox("Group by", cat_cols, key="tb_g")
        v = st.selectbox("Metric", num_cols, key="tb_v")
        n = st.slider("How many", 3, 15, 5)
        top, bottom = an.top_bottom_categories(df, g, v, n)
        c1, c2 = st.columns(2)
        c1.markdown("**Top**"); c1.dataframe(top, use_container_width=True)
        c2.markdown("**Bottom**"); c2.dataframe(bottom, use_container_width=True)
    else:
        st.info("Need at least one categorical and one numeric column.")

with tabs[5]:
    if dt_cols and num_cols:
        d = st.selectbox("Date column", dt_cols)
        v = st.selectbox("Value", num_cols, key="trend_v")
        st.plotly_chart(an.trend_figure(df, d, v), use_container_width=True)
    else:
        st.info("No datetime column detected for trend analysis.")
