"""AI Data Chat: dataset-grounded Q&A with a free, key-free LLM + offline fallback."""
from __future__ import annotations

import os
import sys

# Make the project root importable on every host (incl. Streamlit Cloud).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from utils import chat
from utils.theme import hero, init_state, inject_theme, require_data

st.set_page_config(page_title="Data Chat · Saturn", page_icon="💬", layout="wide")
inject_theme()
init_state()
hero("AI Data Chat", "Ask anything about your data — answered from your data, not the internet.")

if not require_data():
    st.stop()

df = st.session_state["df"]

c1, c2 = st.columns([3, 1])
with c2:
    use_llm = st.toggle("✨ Use free AI", value=True,
                        help="Phrases answers using a free, no-API-key model "
                             "(Pollinations.ai), grounded in numbers computed from "
                             "your data. Falls back to the offline engine if it's "
                             "unavailable.")
with c1:
    st.caption(
        "Every answer's numbers are computed directly from your uploaded dataset. "
        "The AI only rewords those real results — it never invents data or uses "
        "outside knowledge."
    )

# Suggestion chips
st.markdown("**Try one:**")
examples = [
    "Summarise the data",
    "What correlates with Q13" if "Q13" in df.columns else "Show the correlation matrix",
    "Average Q17 by Emirate" if {"Q17", "Emirate"}.issubset(df.columns)
        else f"Average {df.select_dtypes('number').columns[0]} by {df.select_dtypes('object').columns[0]}"
        if len(df.select_dtypes("number").columns) and len(df.select_dtypes("object").columns)
        else "What columns are there",
    "Bar chart of Gender" if "Gender" in df.columns else "What columns are there",
]
ex_cols = st.columns(len(examples))
pending = None
for col, e in zip(ex_cols, examples):
    if col.button(e, use_container_width=True):
        pending = e

prompt = st.chat_input("Ask about your dataset…")
question = prompt or pending

if question:
    st.session_state["chat_history"].append(("user", question))
    spinner = "Asking the free AI…" if use_llm else "Computing from your data…"
    with st.spinner(spinner):
        ans = chat.chat_answer(
            df, question,
            history=[(r, m if isinstance(m, str) else m.get("text", ""))
                     for r, m in st.session_state["chat_history"][:-1]],
            use_llm=use_llm,
        )
    st.session_state["chat_history"].append(("assistant", ans))

# Render conversation
for role, payload in st.session_state["chat_history"]:
    with st.chat_message(role):
        if role == "user":
            st.write(payload)
        else:
            st.write(payload["text"])
            if payload.get("table") is not None:
                st.dataframe(payload["table"], use_container_width=True)
            if payload.get("figure") is not None:
                st.plotly_chart(payload["figure"], use_container_width=True)
            st.caption(f"source: {payload.get('source', 'offline engine')}")

if st.session_state["chat_history"]:
    if st.button("🧹 Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()
