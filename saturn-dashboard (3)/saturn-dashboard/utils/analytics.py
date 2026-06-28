"""Descriptive analytics helpers.

Functions return either pandas objects or Plotly figures. No Streamlit imports,
so everything here is directly testable.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Shared colour palette used across the app for a consistent, vivid look.
PALETTE = [
    "#6C5CE7", "#00B894", "#0984E3", "#E17055", "#FDCB6E",
    "#E84393", "#00CEC9", "#A29BFE", "#FAB1A0", "#55EFC4",
]


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric describe() plus skewness and kurtosis."""
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T
    desc["skew"] = num.skew()
    desc["kurtosis"] = num.kurtosis()
    desc["missing"] = num.isna().sum()
    return desc.round(3)


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    num = df.select_dtypes(include=np.number)
    return num.corr(method=method).round(3) if num.shape[1] >= 2 else pd.DataFrame()


def covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=np.number)
    return num.cov().round(3) if num.shape[1] >= 2 else pd.DataFrame()


def correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> go.Figure | None:
    corr = correlation_matrix(df, method)
    if corr.empty:
        return None
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=f"{method.title()} Correlation Matrix",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def missing_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        df.isna().T,
        color_continuous_scale=["#0E1117", "#E84393"],
        aspect="auto",
        title="Missing-Value Map (highlighted = missing)",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def distribution_figure(df: pd.DataFrame, column: str) -> go.Figure:
    s = df[column].dropna()
    if pd.api.types.is_numeric_dtype(s):
        fig = px.histogram(
            s, x=column, nbins=30, marginal="box",
            color_discrete_sequence=[PALETTE[0]],
            title=f"Distribution of {column}",
        )
    else:
        counts = s.value_counts().head(25)
        fig = px.bar(
            x=counts.values, y=counts.index, orientation="h",
            color=counts.values, color_continuous_scale="Viridis",
            labels={"x": "Count", "y": column},
            title=f"Category counts: {column}",
        )
        fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def category_counts(df: pd.DataFrame, column: str, top: int = 20) -> pd.DataFrame:
    vc = df[column].value_counts(dropna=False).head(top)
    out = vc.rename_axis(column).reset_index(name="count")
    out["pct"] = (100 * out["count"] / len(df)).round(2)
    return out


def frequency_table(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    return pd.crosstab(df[row], df[col])


def pivot_table(
    df: pd.DataFrame, index: str, values: str, aggfunc: str = "mean"
) -> pd.DataFrame:
    return pd.pivot_table(
        df, index=index, values=values, aggfunc=aggfunc
    ).round(3).reset_index()


def top_bottom_categories(
    df: pd.DataFrame, group: str, value: str, n: int = 5, aggfunc: str = "mean"
):
    g = df.groupby(group)[value].agg(aggfunc).sort_values(ascending=False)
    return g.head(n).reset_index(), g.tail(n).reset_index()


def trend_figure(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
    tmp = df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna().sort_values(date_col)
    agg = tmp.groupby(date_col)[value_col].mean().reset_index()
    fig = px.line(
        agg, x=date_col, y=value_col, markers=True,
        color_discrete_sequence=[PALETTE[1]],
        title=f"Trend of {value_col} over {date_col}",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
