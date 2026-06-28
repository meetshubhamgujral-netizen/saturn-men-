"""Settings: session reset, data swap and app information."""
from __future__ import annotations

import os
import sys

# Make the project root importable on every host (incl. Streamlit Cloud).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from utils.theme import hero, init_state, inject_theme

st.set_page_config(page_title="Settings · Saturn", page_icon="⚙️", layout="wide")
inject_theme()
init_state()
hero("Settings", "Manage your session and review app information.")

st.markdown("### Session")
c1, c2 = st.columns(2)
with c1:
    if st.button("🔄 Reset cleaning to original upload", use_container_width=True):
        if st.session_state.get("raw_df") is not None:
            st.session_state["df"] = st.session_state["raw_df"].copy()
            st.session_state["clean_log"] = []
            st.success("Working data reset to the original upload.")
        else:
            st.info("No dataset loaded.")
with c2:
    if st.button("🗑️ Clear all session data", use_container_width=True):
        for k in ["raw_df", "df", "filename", "clean_log", "last_model", "chat_history"]:
            st.session_state[k] = None if k in ("raw_df", "df", "filename") else []
        st.success("Session cleared. Upload a new dataset to continue.")

st.markdown("### Theme")
st.caption(
    "Use the Streamlit menu (top-right ⋮ → Settings) to switch between light and dark "
    "mode. Saturn's palette adapts to both."
)

st.markdown("### About")
st.markdown(
    """
**Saturn – AI Market Understanding Dashboard**

- All insights are derived **only** from your uploaded dataset.
- The AI Data Chat is a deterministic, dataset-grounded engine — no external API,
  no outside knowledge, no data leaves your machine.
- Optional dependencies (**XGBoost**, **SHAP**, **reportlab**) unlock extra models,
  SHAP plots and PDF export when installed; the app degrades gracefully without them.
"""
)
