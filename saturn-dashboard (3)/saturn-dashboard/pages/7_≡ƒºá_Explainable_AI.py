"""Explainable AI: importance, permutation, optional SHAP, prediction breakdown."""
from __future__ import annotations

import os
import sys

# Make the project root importable on every host (incl. Streamlit Cloud).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from utils import explain as ex
from utils.theme import hero, init_state, inject_theme, require_data

st.set_page_config(page_title="Explainable AI · Saturn", page_icon="🧠", layout="wide")
inject_theme()
init_state()
hero("Explainable AI", "Open the black box: what drives the model's predictions.")

if not require_data():
    st.stop()

res = st.session_state.get("last_model")
if res is None or "model" not in res:
    st.info("Train a model on the **🤖 Machine Learning** page first, then return here.")
    st.stop()

model = res["model"]
feat_names = res["feature_names"]
st.markdown(
    f"Explaining: <span class='badge'>{res['model_name']}</span> "
    f"<span class='badge'>{res['task']}</span>"
    + ("" if ex.HAS_SHAP else "  <span class='badge'>SHAP not installed (optional)</span>"),
    unsafe_allow_html=True,
)

tabs = st.tabs(["⭐ Importance", "🔀 Permutation", "🌈 SHAP", "🔍 Single Prediction"])

with tabs[0]:
    native = ex.model_feature_importance(model, feat_names)
    if native is not None:
        from utils.diagnostics import importance_figure
        st.plotly_chart(importance_figure(native), use_container_width=True)
        st.dataframe(native, use_container_width=True)
    else:
        st.info("This estimator does not expose native feature importances.")

with tabs[1]:
    if st.button("Compute permutation importance"):
        with st.spinner("Shuffling features…"):
            perm = ex.permutation_importance_df(model, res["X_test"], res["y_test"])
        st.plotly_chart(ex.permutation_figure(perm), use_container_width=True)
        st.dataframe(perm, use_container_width=True)

with tabs[2]:
    if not ex.HAS_SHAP:
        st.warning("SHAP is optional and not installed. `pip install shap` to enable.")
    else:
        if st.button("Compute SHAP values (sampled)"):
            sample = res["X_test"].head(200)
            with st.spinner("Computing SHAP values…"):
                sv = ex.shap_summary_values(model, sample)
                fig = ex.shap_bar_figure(sv, feat_names)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SHAP could not explain this model/data combination.")

with tabs[3]:
    st.caption("Pick a row from the test set to explain its prediction.")
    idx = st.number_input("Test row index", 0, len(res["X_test"]) - 1, 0)
    row = res["X_test"].iloc[[idx]]
    breakdown = ex.explain_prediction(model, row, feat_names, res["task"])
    st.markdown(breakdown["narrative"])
    st.dataframe(row.T.rename(columns={row.index[0]: "value"}), use_container_width=True)
