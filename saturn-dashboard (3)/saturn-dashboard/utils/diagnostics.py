"""Diagnostic analytics: statistical tests and dimensionality reduction.

Every test returns a small result dict that includes a ``narrative`` field with
a plain-English interpretation, so the UI can surface "why" explanations.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .analytics import PALETTE


def _sig_word(p: float) -> str:
    return "statistically significant" if p < 0.05 else "not statistically significant"


def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str) -> Dict:
    table = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    narrative = (
        f"The association between '{col_a}' and '{col_b}' is {_sig_word(p)} "
        f"(chi-square = {chi2:.2f}, p = {p:.4f}). "
        + (
            "Knowing one variable tells you something about the other."
            if p < 0.05
            else "The two variables appear independent in this sample."
        )
    )
    return {"chi2": chi2, "p_value": p, "dof": int(dof), "narrative": narrative}


def anova_test(df: pd.DataFrame, group_col: str, value_col: str) -> Dict:
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return {"narrative": "Not enough groups with data to run ANOVA."}
    f_stat, p = stats.f_oneway(*groups)
    narrative = (
        f"Mean '{value_col}' differs across '{group_col}' groups: {_sig_word(p)} "
        f"(F = {f_stat:.2f}, p = {p:.4f}). "
        + (
            "At least one group mean stands apart from the others."
            if p < 0.05
            else "Group means are broadly similar."
        )
    )
    return {"f_stat": f_stat, "p_value": p, "narrative": narrative}


def t_test(df: pd.DataFrame, group_col: str, value_col: str) -> Dict:
    levels = df[group_col].dropna().unique()
    if len(levels) != 2:
        return {
            "narrative": f"'{group_col}' has {len(levels)} levels; a t-test needs exactly 2."
        }
    a = df.loc[df[group_col] == levels[0], value_col].dropna()
    b = df.loc[df[group_col] == levels[1], value_col].dropna()
    t_stat, p = stats.ttest_ind(a, b, equal_var=False)
    narrative = (
        f"Difference in '{value_col}' between '{levels[0]}' (mean {a.mean():.2f}) and "
        f"'{levels[1]}' (mean {b.mean():.2f}) is {_sig_word(p)} "
        f"(t = {t_stat:.2f}, p = {p:.4f})."
    )
    return {"t_stat": t_stat, "p_value": p, "narrative": narrative}


def feature_importance(
    df: pd.DataFrame, target: str, features: List[str]
) -> pd.DataFrame:
    """Random-forest importance for quick driver discovery."""
    data = df[features + [target]].dropna()
    if data.empty or len(data) < 10:
        return pd.DataFrame()
    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target]
    is_class = (not pd.api.types.is_numeric_dtype(y)) or y.nunique() <= 15
    model = (
        RandomForestClassifier(n_estimators=200, random_state=42)
        if is_class
        else RandomForestRegressor(n_estimators=200, random_state=42)
    )
    if is_class:
        y = y.astype(str)
    model.fit(X, y)
    imp = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp


def importance_figure(imp: pd.DataFrame, top: int = 15) -> go.Figure | None:
    if imp.empty:
        return None
    d = imp.head(top).iloc[::-1]
    fig = px.bar(
        d, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Tealrose",
        title="Feature importance (random-forest)",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def run_pca(df: pd.DataFrame, n_components: int = 2) -> Dict:
    num = df.select_dtypes(include=np.number).dropna()
    if num.shape[1] < 2 or num.shape[0] < 3:
        return {"narrative": "PCA needs at least 2 numeric columns and 3 rows."}
    n_components = min(n_components, num.shape[1])
    X = StandardScaler().fit_transform(num)
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    coords = pd.DataFrame(comps[:, :2], columns=["PC1", "PC2"]) if n_components >= 2 else None
    fig = None
    if coords is not None:
        fig = px.scatter(
            coords, x="PC1", y="PC2", opacity=0.6,
            color_discrete_sequence=[PALETTE[0]],
            title="PCA projection (first two components)",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    narrative = (
        f"The first {n_components} principal components capture "
        f"{100 * evr.sum():.1f}% of total variance "
        f"(PC1 = {100 * evr[0]:.1f}%"
        + (f", PC2 = {100 * evr[1]:.1f}%" if len(evr) > 1 else "")
        + "). High concentration suggests redundant, correlated features."
    )
    return {
        "explained_variance_ratio": evr.tolist(),
        "figure": fig,
        "narrative": narrative,
    }


def segment_analysis(df: pd.DataFrame, segment: str, value: str) -> pd.DataFrame:
    """Compare a numeric metric across segments with simple effect sizes."""
    grp = df.groupby(segment)[value].agg(["mean", "median", "std", "count"]).round(3)
    overall = df[value].mean()
    grp["vs_overall"] = (grp["mean"] - overall).round(3)
    return grp.sort_values("mean", ascending=False).reset_index()
