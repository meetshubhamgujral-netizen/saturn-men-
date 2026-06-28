"""Dataset-grounded chat engine.

Two layers, designed so the chat box ALWAYS responds:

1. A strong deterministic engine (`local_answer`) that turns natural-language
   questions into real pandas computations over the uploaded DataFrame. This
   needs no network and never hallucinates.

2. An optional free, **key-free** LLM (`llm_answer`, via Pollinations.ai) that
   rephrases the real computed numbers into a natural reply. It is always fed
   the deterministic facts and told to use only those, so it stays grounded.
   If the endpoint is slow, rate-limited or offline, we fall back to layer 1.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

from .analytics import PALETTE

# Words we never treat as column references.
_STOP = {
    "the", "a", "an", "of", "by", "in", "on", "for", "to", "and", "or", "is",
    "are", "what", "which", "show", "me", "give", "tell", "about", "with",
    "highest", "lowest", "average", "mean", "median", "compare", "vs", "versus",
    "correlation", "between", "chart", "plot", "graph", "count", "how", "many",
    "top", "bottom", "most", "least", "value", "values", "group", "across",
    "per", "each", "distribution", "summary", "summarise", "summarize",
}


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s).lower()).strip()


def _resolve_columns(text: str, columns: List[str]) -> List[str]:
    """Find every column the text plausibly refers to, best matches first."""
    low = _norm(text)
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t and t not in _STOP]
    scored: List[Tuple[float, str]] = []
    for c in columns:
        nc = _norm(c)
        c_tokens = set(re.split(r"[^a-z0-9]+", nc))
        score = 0.0
        if re.search(r"\b" + re.escape(nc) + r"\b", low):   # whole name appears
            score = 100 + len(nc)
        elif any(t == nc for t in tokens):         # exact token == column
            score = 90
        elif c_tokens & set(tokens):               # share a significant word
            score = 50 + 5 * len(c_tokens & set(tokens))
        else:                                       # fuzzy last resort
            hit = get_close_matches(nc, tokens, n=1, cutoff=0.85)
            if hit:
                score = 30
        if score:
            scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    # de-dup, preserve order
    out, seen = [], set()
    for _, c in scored:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Deterministic engine
# --------------------------------------------------------------------------- #
def local_answer(df: pd.DataFrame, question: str) -> Dict:
    q = _norm(question)
    cols = list(df.columns)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]
    found = _resolve_columns(question, cols)
    fnum = [c for c in found if c in num_cols]
    fcat = [c for c in found if c in cat_cols]

    def out(text, table=None, figure=None):
        return {"text": text, "table": table, "figure": figure}

    # list columns
    if any(k in q for k in ["what columns", "list columns", "which columns", "fields"]):
        return out(
            f"The dataset has {len(cols)} columns: {', '.join(cols)}. "
            f"{len(num_cols)} are numeric, {len(cat_cols)} categorical."
        )

    # summary
    if any(k in q for k in ["summar", "overview", "describe the data", "tell me about the data"]):
        return out(
            f"{len(df):,} rows × {len(cols)} columns "
            f"({len(num_cols)} numeric, {len(cat_cols)} categorical). "
            f"Missing values: {int(df.isna().sum().sum()):,}. "
            f"Numeric columns include: {', '.join(num_cols[:10])}.",
            table=df.describe(include="all").T.reset_index().rename(columns={"index": "column"}),
        )

    # describe a single column
    if (q.startswith("describe ") or "tell me about" in q) and found:
        c = found[0]
        if c in num_cols:
            s = df[c]
            return out(
                f"'{c}' — mean {s.mean():.3f}, median {s.median():.3f}, "
                f"min {s.min():.3f}, max {s.max():.3f}, std {s.std():.3f}, "
                f"{int(s.isna().sum())} missing.",
                figure=_dist_fig(df, c),
            )
        vc = df[c].value_counts().head(15)
        return out(
            f"'{c}' has {df[c].nunique()} unique values. Most common: "
            f"{vc.index[0]} ({vc.iloc[0]}).",
            table=vc.rename_axis(c).reset_index(name="count"),
            figure=_dist_fig(df, c),
        )

    # correlation
    if "correlat" in q:
        if len(fnum) >= 2:
            a, b = fnum[0], fnum[1]
            r = df[[a, b]].corr().iloc[0, 1]
            strength = "strong" if abs(r) > 0.6 else "moderate" if abs(r) > 0.3 else "weak"
            return out(
                f"The correlation between '{a}' and '{b}' is {r:.2f} — a {strength} "
                f"{'positive' if r >= 0 else 'negative'} relationship."
            )
        if len(fnum) == 1 and len(num_cols) >= 2:
            target = fnum[0]
            corr = df[num_cols].corr()[target].drop(target).sort_values(key=abs, ascending=False)
            top = corr.head(5)
            tbl = top.reset_index(); tbl.columns = ["column", f"corr_with_{target}"]
            return out(
                f"Strongest correlations with '{target}': "
                + ", ".join(f"{i} ({v:.2f})" for i, v in top.items()) + ".",
                table=tbl.round(3),
            )
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, title="Correlation matrix")
            return out("Here is the correlation matrix for all numeric columns.", figure=fig)
        return out("I need at least two numeric columns to measure correlation.")

    # ranking: highest / lowest
    asc = any(k in q for k in ["lowest", "smallest", "bottom", "minimum", "least", "worst", "fewest"])
    if asc or any(k in q for k in ["highest", "largest", "top", "maximum", "most", "best"]):
        return _ranking(df, fnum, fcat, num_cols, cat_cols, asc, out)

    # average / mean / median by group
    if any(k in q for k in ["average", "mean", "median"]):
        agg = "median" if "median" in q else "mean"
        metric = fnum[0] if fnum else None
        if metric is None:
            return out("Which numeric column should I average? Available: " + ", ".join(num_cols[:12]) + ".")
        if fcat:
            grp = fcat[0]
            tbl = df.groupby(grp)[metric].agg(agg).round(3).sort_values(ascending=False).reset_index()
            fig = px.bar(tbl, x=grp, y=metric, color=grp, color_discrete_sequence=PALETTE,
                         title=f"{agg.title()} {metric} by {grp}")
            return out(f"{agg.title()} of '{metric}' by '{grp}':", table=tbl, figure=fig)
        return out(f"The {agg} of '{metric}' is {getattr(df[metric], agg)():.3f}.")

    # compare
    if "compare" in q or " vs " in q or "versus" in q:
        if fcat and fnum:
            grp, metric = fcat[0], fnum[0]
            tbl = df.groupby(grp)[metric].agg(["mean", "median", "count"]).round(3).reset_index()
            fig = px.bar(tbl, x=grp, y="mean", color=grp, color_discrete_sequence=PALETTE,
                         title=f"Mean {metric} by {grp}")
            return out(f"Comparison of '{metric}' across '{grp}':", table=tbl, figure=fig)
        if fcat:
            tbl = df[fcat[0]].value_counts().rename_axis(fcat[0]).reset_index(name="count")
            return out(f"Breakdown of '{fcat[0]}':", table=tbl,
                       figure=_dist_fig(df, fcat[0]))
        return out("To compare, name a category and (optionally) a numeric metric.")

    # chart
    if any(k in q for k in ["chart", "plot", "graph", "visual", "histogram", "bar", "pie"]):
        if not found:
            return out("Which column should I chart? Mention one: " + ", ".join(cols[:12]) + ".")
        return out(f"Chart for '{found[0]}':", figure=_chart(df, found, num_cols, cat_cols, q))

    # count / how many
    if any(k in q for k in ["how many", "count", "number of"]):
        if fcat:
            tbl = df[fcat[0]].value_counts().rename_axis(fcat[0]).reset_index(name="count")
            return out(f"Counts for '{fcat[0]}':", table=tbl, figure=_dist_fig(df, fcat[0]))
        return out(f"The dataset has {len(df):,} rows and {len(cols)} columns.")

    # fallback
    return out(
        "I answer from your dataset. Try: 'summarise the data', "
        "'which <category> has the highest <column>', 'average <column> by <category>', "
        "'compare <metric> across <category>', 'what correlates with <column>', "
        "or 'bar chart of <column>'. Columns: " + ", ".join(cols[:12])
        + ("…" if len(cols) > 12 else "") + "."
    )


def _ranking(df, fnum, fcat, num_cols, cat_cols, asc, out):
    word = "lowest" if asc else "highest"
    if fnum and fcat:
        metric, grp = fnum[0], fcat[0]
        g = df.groupby(grp)[metric].mean().sort_values(ascending=asc)
        tbl = g.reset_index().rename(columns={metric: f"mean_{metric}"}).round(3)
        fig = px.bar(tbl, x=grp, y=f"mean_{metric}", color=grp,
                     color_discrete_sequence=PALETTE, title=f"Mean {metric} by {grp}")
        return out(f"'{g.index[0]}' has the {word} average '{metric}' ({g.iloc[0]:.3f}).",
                   table=tbl, figure=fig)
    if fnum:
        m = fnum[0]
        idx = df[m].idxmin() if asc else df[m].idxmax()
        return out(f"The {('minimum' if asc else 'maximum')} '{m}' is {df[m].loc[idx]:.3f} (row {idx}).")
    if fcat:
        vc = df[fcat[0]].value_counts(ascending=asc)
        return out(f"'{vc.index[0]}' is the {('least' if asc else 'most')} common "
                   f"'{fcat[0]}' ({vc.iloc[0]} rows).",
                   table=vc.rename_axis(fcat[0]).reset_index(name="count"))
    return out("Name a numeric column (and optionally a category) to rank by.")


def _dist_fig(df, col):
    s = df[col].dropna()
    if pd.api.types.is_numeric_dtype(s):
        return px.histogram(df, x=col, nbins=30, color_discrete_sequence=[PALETTE[0]],
                            title=f"Distribution of {col}")
    vc = df[col].value_counts().head(20)
    fig = px.bar(x=vc.index, y=vc.values, color=vc.values, color_continuous_scale="Viridis",
                 labels={"x": col, "y": "count"}, title=f"Counts of {col}")
    fig.update_layout(coloraxis_showscale=False)
    return fig


def _chart(df, found, num_cols, cat_cols, q):
    col = found[0]
    second = next((c for c in found[1:] if c != col), None)
    if col in num_cols and second in cat_cols:
        tbl = df.groupby(second)[col].mean().reset_index()
        return px.bar(tbl, x=second, y=col, color=second, color_discrete_sequence=PALETTE,
                      title=f"Mean {col} by {second}")
    if "pie" in q and col in cat_cols:
        vc = df[col].value_counts().head(10)
        return px.pie(values=vc.values, names=vc.index, title=f"{col} share")
    return _dist_fig(df, col)


# --------------------------------------------------------------------------- #
# Free, key-free LLM layer (Pollinations.ai)
# --------------------------------------------------------------------------- #
def _build_context(df: pd.DataFrame, question: str, local: Dict) -> str:
    cols = ", ".join(f"{c}({str(t)})" for c, t in df.dtypes.items())
    facts = local["text"]
    if local.get("table") is not None:
        facts += "\nTable:\n" + local["table"].head(15).to_csv(index=False)
    return (
        f"Dataset: {len(df)} rows, {df.shape[1]} columns.\n"
        f"Columns: {cols}\n"
        f"Computed facts for the user's question:\n{facts}"
    )


def llm_answer(question: str, context: str, history: Optional[List] = None,
               timeout: int = 18) -> Optional[str]:
    """Call the free Pollinations text endpoint. Returns text or None on failure."""
    system = (
        "You are Saturn, a concise data analyst. Answer ONLY using the dataset "
        "facts provided. Never invent numbers or use outside knowledge. If the "
        "facts do not contain the answer, say so plainly. Keep it to a few sentences."
    )
    messages = [{"role": "system", "content": system}]
    for role, content in (history or [])[-4:]:
        messages.append({"role": role, "content": str(content)[:500]})
    messages.append({"role": "user",
                     "content": f"{context}\n\nUser question: {question}"})

    # Primary: OpenAI-compatible POST.
    try:
        payload = json.dumps({"model": "openai", "messages": messages,
                              "temperature": 0.3, "private": True}).encode()
        req = urllib.request.Request(
            "https://text.pollinations.ai/openai", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        text = data["choices"][0]["message"]["content"].strip()
        if text:
            return text
    except Exception:
        pass

    # Fallback: simple GET prompt endpoint.
    try:
        prompt = f"{system}\n\n{context}\n\nQuestion: {question}\nAnswer concisely:"
        url = "https://text.pollinations.ai/" + urllib.parse.quote(prompt[:1800])
        with urllib.request.urlopen(url, timeout=timeout) as r:
            text = r.read().decode().strip()
        return text or None
    except Exception:
        return None


def chat_answer(df: pd.DataFrame, question: str, history: Optional[List] = None,
                use_llm: bool = True) -> Dict:
    """Orchestrate: compute facts locally, optionally narrate with the free LLM."""
    local = local_answer(df, question)
    result = dict(local)
    result["source"] = "offline engine"
    if use_llm:
        context = _build_context(df, question, local)
        narrated = llm_answer(question, context, history)
        if narrated:
            result["text"] = narrated
            result["source"] = "free AI (Pollinations)"
    return result


# Backwards-compatible alias used by older imports.
def answer_question(df: pd.DataFrame, question: str) -> Dict:
    return chat_answer(df, question, use_llm=False)
