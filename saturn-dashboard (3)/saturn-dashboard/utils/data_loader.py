"""Dataset loading, type inference, and profiling.

This module is intentionally free of any Streamlit dependency so it can be unit
tested and reused. All functions take/return plain pandas / Python objects.
"""
from __future__ import annotations

import io
import json
from typing import Dict, List

import numpy as np
import pandas as pd


def load_dataframe(file_obj, filename: str) -> pd.DataFrame:
    """Load a CSV, Excel or JSON file-like object into a DataFrame.

    Parameters
    ----------
    file_obj : file-like
        An object exposing ``read``/``seek`` (e.g. a Streamlit UploadedFile or
        an open binary handle).
    filename : str
        Original filename, used to detect the format by extension.
    """
    name = (filename or "").lower()
    try:
        file_obj.seek(0)
    except Exception:
        pass

    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(file_obj)
    if name.endswith(".json"):
        raw = file_obj.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # Accept either a record dict or an orient='columns' style dict.
            try:
                return pd.json_normalize(parsed)
            except Exception:
                return pd.DataFrame(parsed)
        return pd.DataFrame(parsed)
    # default: CSV / TSV
    sep = "\t" if name.endswith(".tsv") else ","
    return pd.read_csv(file_obj, sep=sep)


def _is_boolean(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    vals = set(str(v).strip().lower() for v in series.dropna().unique())
    return len(vals) > 0 and vals.issubset({"true", "false", "yes", "no", "0", "1"})


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify each column as numeric, categorical, datetime or boolean."""
    numeric, categorical, datetime_cols, boolean = [], [], [], []
    for col in df.columns:
        s = df[col]
        if _is_boolean(s):
            boolean.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        else:
            # Attempt a date parse on object columns before falling back.
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() > 0.8:
                datetime_cols.append(col)
            else:
                categorical.append(col)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "boolean": boolean,
    }


def profile_dataframe(df: pd.DataFrame) -> Dict:
    """Return a dict of high level dataset metrics for the overview cards."""
    types = detect_column_types(df)
    missing = int(df.isna().sum().sum())
    cells = int(df.shape[0] * df.shape[1]) or 1
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": missing,
        "missing_pct": round(100 * missing / cells, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
        "numeric_cols": types["numeric"],
        "categorical_cols": types["categorical"],
        "datetime_cols": types["datetime"],
        "boolean_cols": types["boolean"],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


def suggest_targets(df: pd.DataFrame, max_card: int = 15) -> List[str]:
    """Suggest plausible target columns for modelling.

    Heuristic: low-cardinality columns make good classification targets;
    numeric columns with reasonable spread make good regression targets.
    """
    suggestions = []
    n = len(df)
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        nunique = s.nunique()
        if 2 <= nunique <= max_card and nunique < n:
            suggestions.append(col)  # classification-friendly
        elif pd.api.types.is_numeric_dtype(s) and nunique > max_card:
            suggestions.append(col)  # regression-friendly
    # Stable order, de-duplicated.
    seen, ordered = set(), []
    for c in suggestions:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value counts and percentages."""
    miss = df.isna().sum()
    out = pd.DataFrame(
        {
            "column": miss.index,
            "missing": miss.values,
            "missing_pct": (100 * miss.values / max(len(df), 1)).round(2),
            "dtype": [str(df[c].dtype) for c in miss.index],
        }
    )
    return out.sort_values("missing", ascending=False).reset_index(drop=True)
