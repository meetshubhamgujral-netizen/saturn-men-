"""Interactive data-cleaning operations.

Each function returns a *new* DataFrame plus a short human-readable log entry so
the UI can show a before/after audit trail.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


def handle_missing(
    df: pd.DataFrame, strategy: str = "drop_rows", columns: List[str] | None = None
) -> Tuple[pd.DataFrame, str]:
    """Handle missing values using the chosen strategy."""
    out = df.copy()
    cols = columns or list(out.columns)
    before = int(out.isna().sum().sum())

    if strategy == "drop_rows":
        out = out.dropna(subset=cols)
    elif strategy == "drop_cols":
        out = out.drop(columns=[c for c in cols if out[c].isna().any()])
    elif strategy == "mean":
        for c in cols:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].mean())
    elif strategy == "median":
        for c in cols:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].median())
    elif strategy == "mode":
        for c in cols:
            mode = out[c].mode()
            if not mode.empty:
                out[c] = out[c].fillna(mode.iloc[0])
    elif strategy == "zero":
        for c in cols:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(0)
    elif strategy == "ffill":
        out[cols] = out[cols].ffill()
    after = int(out.isna().sum().sum())
    return out, f"Missing values: handled '{strategy}' ({before} -> {after} nulls)."


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, f"Removed {before - len(out)} duplicate row(s)."


def detect_outliers(
    df: pd.DataFrame, columns: List[str], method: str = "iqr", z_thresh: float = 3.0
) -> pd.DataFrame:
    """Return a boolean mask DataFrame marking outliers per numeric column."""
    mask = pd.DataFrame(False, index=df.index, columns=columns)
    for c in columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask[c] = (s < lo) | (s > hi)
        else:  # z-score
            std = s.std(ddof=0)
            if std and not np.isnan(std):
                mask[c] = ((s - s.mean()).abs() / std) > z_thresh
    return mask


def remove_outliers(
    df: pd.DataFrame, columns: List[str], method: str = "iqr"
) -> Tuple[pd.DataFrame, str]:
    mask = detect_outliers(df, columns, method)
    row_has_outlier = mask.any(axis=1)
    out = df.loc[~row_has_outlier].reset_index(drop=True)
    return out, f"Removed {int(row_has_outlier.sum())} outlier row(s) via {method}."


def encode_categoricals(
    df: pd.DataFrame, columns: List[str], method: str = "onehot"
) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    if not columns:
        return out, "Encoding: no categorical columns selected."
    if method == "label":
        for c in columns:
            le = LabelEncoder()
            out[c] = le.fit_transform(out[c].astype(str))
        return out, f"Label-encoded {len(columns)} column(s)."
    # one-hot
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    arr = enc.fit_transform(out[columns].astype(str))
    names = enc.get_feature_names_out(columns)
    dummies = pd.DataFrame(arr, columns=names, index=out.index).astype(int)
    out = pd.concat([out.drop(columns=columns), dummies], axis=1)
    return out, f"One-hot encoded {len(columns)} column(s) -> {len(names)} features."


def scale_features(
    df: pd.DataFrame, columns: List[str], method: str = "standard"
) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    cols = [c for c in columns if pd.api.types.is_numeric_dtype(out[c])]
    if not cols:
        return out, "Scaling: no numeric columns selected."
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    out[cols] = scaler.fit_transform(out[cols])
    return out, f"Applied {method} scaling to {len(cols)} column(s)."


def select_features_by_variance(
    df: pd.DataFrame, columns: List[str], threshold: float = 0.0
) -> Tuple[pd.DataFrame, str]:
    """Drop near-constant numeric features below a variance threshold."""
    drop = [
        c
        for c in columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].var(ddof=0) <= threshold
    ]
    out = df.drop(columns=drop)
    return out, f"Dropped {len(drop)} low-variance feature(s)."
