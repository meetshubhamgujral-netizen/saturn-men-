"""Export utilities for reports and processed data.

PDF generation uses reportlab when available and falls back to a self-contained
HTML report otherwise, so exports never hard-fail on a minimal install.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    HAS_REPORTLAB = True
except Exception:  # pragma: no cover
    HAS_REPORTLAB = False


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buf.getvalue()


def build_pdf_report(title: str, sections: List[Dict]) -> bytes:
    """Build a PDF (or HTML fallback) from a list of section dicts.

    Each section: {"heading": str, "text": str, "table": DataFrame|None}.
    """
    if HAS_REPORTLAB:
        return _pdf_reportlab(title, sections)
    return _html_fallback(title, sections).encode("utf-8")


def _pdf_reportlab(title: str, sections: List[Dict]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]),
             Paragraph(datetime.now().strftime("Generated %Y-%m-%d %H:%M"),
                       styles["Normal"]),
             Spacer(1, 0.6 * cm)]
    for sec in sections:
        story.append(Paragraph(sec.get("heading", ""), styles["Heading2"]))
        if sec.get("text"):
            story.append(Paragraph(str(sec["text"]), styles["Normal"]))
        tbl = sec.get("table")
        if isinstance(tbl, pd.DataFrame) and not tbl.empty:
            shown = tbl.head(25).round(3) if tbl.select_dtypes("number").shape[1] else tbl.head(25)
            data = [list(map(str, shown.columns))] + shown.astype(str).values.tolist()
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C5CE7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F2F0FF")]),
            ]))
            story.append(t)
        story.append(Spacer(1, 0.5 * cm))
    doc.build(story)
    return buf.getvalue()


def _html_fallback(title: str, sections: List[Dict]) -> str:
    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:40px;color:#222}",
        "h1{color:#6C5CE7}h2{color:#0984E3;border-bottom:2px solid #eee;padding-bottom:4px}",
        "table{border-collapse:collapse;margin:10px 0}td,th{border:1px solid #ccc;padding:6px 10px;font-size:12px}",
        "th{background:#6C5CE7;color:#fff}tr:nth-child(even){background:#F2F0FF}",
        "</style></head><body>",
        f"<h1>{title}</h1><p>{datetime.now():%Y-%m-%d %H:%M}</p>",
    ]
    for sec in sections:
        parts.append(f"<h2>{sec.get('heading','')}</h2>")
        if sec.get("text"):
            parts.append(f"<p>{sec['text']}</p>")
        tbl = sec.get("table")
        if isinstance(tbl, pd.DataFrame) and not tbl.empty:
            parts.append(tbl.head(25).to_html(index=False))
    parts.append("</body></html>")
    return "".join(parts)


def report_filename(prefix: str, ext: str) -> str:
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.{ext}"
