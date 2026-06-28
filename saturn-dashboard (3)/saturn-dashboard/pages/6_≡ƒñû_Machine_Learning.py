"""Machine Learning: task detection, single-model training, comparison, clustering."""
from __future__ import annotations

import os
import sys

# Make the project root importable on every host (incl. Streamlit Cloud).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from utils import data_loader as dl
from utils import ml
from utils.theme import hero, init_state, inject_theme, require_data

st.set_page_config(page_title="ML · Saturn", page_icon="🤖", layout="wide")
inject_theme()
init_state()
hero("Machine Learning", "Auto task detection with a full model zoo and side-by-side comparison.")

if not require_data():
    st.stop()

df = st.session_state["df"]
prof = dl.profile_dataframe(df)
all_cols = df.columns.tolist()

target = st.selectbox("🎯 Target variable", ["(none / clustering)"] + all_cols)
task = ml.detect_task(df, target)
st.markdown(
    f"Detected task: <span class='badge'>🧠 {task.upper()}</span>"
    + ("" if ml.HAS_XGBOOST else "  <span class='badge'>XGBoost not installed (optional)</span>"),
    unsafe_allow_html=True,
)

if task != "clustering":
    feats = st.multiselect(
        "Features", [c for c in all_cols if c != target],
        default=[c for c in all_cols if c != target][:12],
    )
    mode = st.radio("Mode", ["Single model", "Compare models"], horizontal=True)
    models = ml.available_models(task)

    if mode == "Single model":
        model_name = st.selectbox("Algorithm", models)
        if st.button("🚀 Train", type="primary") and feats:
            with st.spinner(f"Training {model_name}…"):
                res = ml.train_supervised(df, target, feats, model_name, task)
            if "error" in res:
                st.error(res["error"])
            else:
                st.session_state["last_model"] = res
                st.success(f"{model_name} trained.")
                cols = st.columns(len(res["metrics"]))
                for c, (k, v) in zip(cols, res["metrics"].items()):
                    c.metric(k, v)
                if task == "classification":
                    st.plotly_chart(ml.confusion_figure(res["confusion"]),
                                    use_container_width=True)
                    if "roc" in res:
                        st.plotly_chart(ml.roc_figure(res["roc"]),
                                        use_container_width=True)
                else:
                    import numpy as np
                    st.plotly_chart(
                        ml.residual_figure(res["residuals"], np.asarray(res["predictions"])),
                        use_container_width=True,
                    )
    else:
        chosen = st.multiselect("Algorithms to compare", models, default=models[:4])
        if st.button("⚖️ Compare", type="primary") and feats and chosen:
            with st.spinner("Training models…"):
                table = ml.compare_models(df, target, feats, chosen, task)
            st.dataframe(table, use_container_width=True)
            sort_col = "Accuracy" if task == "classification" else "R2 Score"
            if sort_col in table.columns and not table.empty:
                best = table.sort_values(sort_col, ascending=False).iloc[0]["model"]
                st.success(f"Best by {sort_col}: **{best}**")
else:
    feats = st.multiselect(
        "Numeric features for clustering", prof["numeric_cols"],
        default=prof["numeric_cols"][:8],
    )
    algo = st.selectbox("Algorithm", ml.available_models("clustering"))
    k = st.slider("Clusters (k)", 2, 10, 4) if algo != "DBSCAN" else 0
    if st.button("🚀 Run clustering", type="primary") and feats:
        with st.spinner("Clustering…"):
            res = ml.run_clustering(df, feats, algo, k)
        if "error" in res:
            st.error(res["error"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Clusters found", res["n_found"])
            c2.metric("Silhouette", res.get("silhouette", "—"))
            c3.metric("Davies-Bouldin", res.get("davies_bouldin", "—"))
            st.plotly_chart(res["figure"], use_container_width=True)
