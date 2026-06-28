"""Explainable-AI helpers.

Permutation importance always works (sklearn). SHAP is optional; if it is not
installed, the UI still functions and simply hides SHAP-specific plots.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

from .analytics import PALETTE

try:
    import shap

    HAS_SHAP = True
except Exception:  # pragma: no cover - depends on environment
    HAS_SHAP = False


def permutation_importance_df(
    model, X_test: pd.DataFrame, y_test, n_repeats: int = 10
) -> pd.DataFrame:
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=42
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_figure(imp: pd.DataFrame, top: int = 15) -> go.Figure:
    d = imp.head(top).iloc[::-1]
    fig = px.bar(
        d, x="importance", y="feature", orientation="h", error_x="std",
        color="importance", color_continuous_scale="Sunsetdark",
        title="Permutation importance (impact on score when shuffled)",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def model_feature_importance(model, feature_names: List[str]) -> pd.DataFrame | None:
    """Native importance / coefficients when the estimator exposes them."""
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.ravel(np.abs(model.coef_))
        if len(vals) != len(feature_names):
            return None
    else:
        return None
    return (
        pd.DataFrame({"feature": feature_names, "importance": vals})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def shap_summary_values(model, X_sample: pd.DataFrame):
    """Compute SHAP values for a sample; returns (values, expected) or None."""
    if not HAS_SHAP:
        return None
    try:
        explainer = shap.Explainer(model, X_sample)
        sv = explainer(X_sample)
        return sv
    except Exception:
        return None


def shap_bar_figure(shap_values, feature_names: List[str]) -> go.Figure | None:
    """Mean absolute SHAP value per feature as a Plotly bar (portable)."""
    if shap_values is None:
        return None
    try:
        arr = np.asarray(shap_values.values)
        if arr.ndim == 3:  # multiclass -> average across classes
            arr = np.abs(arr).mean(axis=2)
        mean_abs = np.abs(arr).mean(axis=0)
        d = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=True)
            .tail(15)
        )
        fig = px.bar(
            d, x="mean_abs_shap", y="feature", orientation="h",
            color="mean_abs_shap", color_continuous_scale="Viridis",
            title="SHAP feature impact (mean |value|)",
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
        return fig
    except Exception:
        return None


def explain_prediction(
    model, row: pd.DataFrame, feature_names: List[str], task: str
) -> Dict:
    """Return a single-row prediction with a plain-language breakdown."""
    pred = model.predict(row)[0]
    contributions = None
    native = model_feature_importance(model, feature_names)
    if native is not None:
        top = native.head(5)
        contributions = [
            f"{r.feature} (weight {r.importance:.3f})" for r in top.itertuples()
        ]
    narrative = (
        f"The model predicts **{pred}**. "
        + (
            "The features that most influence this kind of prediction are: "
            + ", ".join(contributions) + "."
            if contributions
            else "This model does not expose per-feature weights."
        )
    )
    return {"prediction": pred, "drivers": contributions, "narrative": narrative}
