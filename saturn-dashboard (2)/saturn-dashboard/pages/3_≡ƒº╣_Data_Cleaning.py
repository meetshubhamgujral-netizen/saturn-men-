"""Data Cleaning: interactive missing/dup/outlier/encode/scale operations."""
from __future__ import annotations

import os
import sys

# Make the project root importable on every host (incl. Streamlit Cloud).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from utils import cleaning as cl
from utils import data_loader as dl
from utils.theme import hero, init_state, inject_theme, require_data

st.set_page_config(page_title="Cleaning · Saturn", page_icon="🧹", layout="wide")
inject_theme()
init_state()
hero("Data Cleaning", "Apply transformations step by step with a full audit trail.")

if not require_data():
    st.stop()

df = st.session_state["df"]
prof = dl.profile_dataframe(df)
num_cols, cat_cols = prof["numeric_cols"], prof["categorical_cols"]


def apply(new_df, message):
    st.session_state["df"] = new_df
    st.session_state["clean_log"].append(message)
    st.success(message)
    st.rerun()


t1, t2, t3, t4, t5 = st.tabs(
    ["🩹 Missing", "♻️ Duplicates & Outliers", "🔤 Encoding", "📏 Scaling", "🧮 Feature Select"]
)

with t1:
    strat = st.selectbox(
        "Strategy",
        ["drop_rows", "drop_cols", "mean", "median", "mode", "zero", "ffill"],
    )
    cols = st.multiselect("Columns (blank = all)", list(df.columns))
    if st.button("Apply missing-value handling"):
        new, msg = cl.handle_missing(df, strat, cols or None)
        apply(new, msg)

with t2:
    if st.button("Remove duplicate rows"):
        new, msg = cl.remove_duplicates(df)
        apply(new, msg)
    st.markdown("---")
    method = st.radio("Outlier method", ["iqr", "zscore"], horizontal=True)
    ocols = st.multiselect("Numeric columns for outlier removal", num_cols)
    if st.button("Remove outlier rows") and ocols:
        new, msg = cl.remove_outliers(df, ocols, method)
        apply(new, msg)

with t3:
    enc = st.radio("Encoding", ["onehot", "label"], horizontal=True)
    ecols = st.multiselect("Categorical columns", cat_cols)
    if st.button("Encode") and ecols:
        new, msg = cl.encode_categoricals(df, ecols, enc)
        apply(new, msg)

with t4:
    sc = st.radio("Scaler", ["standard", "minmax"], horizontal=True)
    scols = st.multiselect("Numeric columns to scale", num_cols)
    if st.button("Scale") and scols:
        new, msg = cl.scale_features(df, scols, sc)
        apply(new, msg)

with t5:
    thr = st.slider("Variance threshold", 0.0, 1.0, 0.0, 0.01)
    if st.button("Drop low-variance numeric features"):
        new, msg = cl.select_features_by_variance(df, num_cols, thr)
        apply(new, msg)

st.markdown("### Audit trail")
if st.session_state["clean_log"]:
    for i, step in enumerate(st.session_state["clean_log"], 1):
        st.markdown(f"{i}. {step}")
    cc1, cc2 = st.columns(2)
    if cc1.button("↩️ Reset to original upload"):
        st.session_state["df"] = st.session_state["raw_df"].copy()
        st.session_state["clean_log"] = []
        st.rerun()
else:
    st.caption("No cleaning steps applied yet.")

st.markdown("### Current data preview")
st.dataframe(st.session_state["df"].head(50), use_container_width=True)
