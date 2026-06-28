"""Reusable presentational components (KPI cards, gradient helpers)."""
from __future__ import annotations

from typing import List

import streamlit as st

GRADIENTS = [
    "linear-gradient(135deg,#6C5CE7,#a29bfe)",
    "linear-gradient(135deg,#00B894,#55efc4)",
    "linear-gradient(135deg,#0984E3,#74b9ff)",
    "linear-gradient(135deg,#E17055,#fab1a0)",
    "linear-gradient(135deg,#E84393,#fd79a8)",
    "linear-gradient(135deg,#fdcb6e,#ffeaa7)",
]


def kpi_card(label: str, value, icon: str = "📊", grad_index: int = 0) -> str:
    grad = GRADIENTS[grad_index % len(GRADIENTS)]
    return (
        f"<div class='kpi-card' style='background:{grad}'>"
        f"<div class='kpi-icon'>{icon}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-label'>{label}</div></div>"
    )


def kpi_row(cards: List[dict]) -> None:
    """Render a responsive row of KPI cards.

    cards: list of {"label","value","icon"} dicts.
    """
    cols = st.columns(len(cards))
    for i, (col, card) in enumerate(zip(cols, cards)):
        with col:
            st.markdown(
                kpi_card(card["label"], card["value"], card.get("icon", "📊"), i),
                unsafe_allow_html=True,
            )
