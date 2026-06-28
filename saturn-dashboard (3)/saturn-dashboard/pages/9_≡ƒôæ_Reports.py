"""Reports: export cleaned data, summary stats, predictions and a PDF report."""
from __future__ import annotations

import os
import sys

# Make the project root importable on every host (incl. Streamlit Cloud).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from utils import analytics as an
from utils import data_loader as dl
from utils import reporting as rp
from utils.theme import hero, init_state, inject_theme, require_data

st.set_page_config(page_title="Reports · Saturn", page_icon="📑", layout="wide")
inject_theme()
init_state()
hero("Reports", "Export your processed data, analysis and a shareable report.")

if not require_data():
    st.stop()

df = st.session_state["df"]
prof = dl.profile_dataframe(df)

st.markdown("### Data exports")
c1, c2 = st.columns(2)
c1.download_button(
    "⬇️ Cleaned data (CSV)", rp.dataframe_to_csv_bytes(df),
    file_name=rp.report_filename("saturn_data", "csv"), mime="text/csv",
    use_container_width=True,
)
c2.download_button(
    "⬇️ Cleaned data (Excel)", rp.dataframe_to_excel_bytes(df),
    file_name=rp.report_filename("saturn_data", "xlsx"),
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

summary = an.summary_statistics(df).reset_index().rename(columns={"index": "feature"})
st.download_button(
    "⬇️ Summary statistics (CSV)", rp.dataframe_to_csv_bytes(summary),
    file_name=rp.report_filename("saturn_summary", "csv"), mime="text/csv",
)

res = st.session_state.get("last_model")
if res and "predictions" in res:
    import pandas as pd
    preds = res["X_test"].copy()
    preds["actual"] = list(res["y_test"])
    preds["predicted"] = list(res["predictions"])
    st.download_button(
        "⬇️ Model predictions (CSV)", rp.dataframe_to_csv_bytes(preds),
        file_name=rp.report_filename("saturn_predictions", "csv"), mime="text/csv",
    )

st.markdown("---")
st.markdown("### Generate report")
title = st.text_input("Report title", "Saturn Market Understanding Report")
if st.button("📄 Build report", type="primary"):
    sections = [
        {
            "heading": "Dataset Overview",
            "text": (
                f"{prof['rows']:,} rows × {prof['columns']} columns. "
                f"{len(prof['numeric_cols'])} numeric and "
                f"{len(prof['categorical_cols'])} categorical features. "
                f"Missing values: {prof['missing_values']:,} ({prof['missing_pct']}%)."
            ),
            "table": None,
        },
        {"heading": "Summary Statistics", "text": "", "table": summary},
    ]
    corr = an.correlation_matrix(df)
    if not corr.empty:
        sections.append(
            {"heading": "Correlation Matrix", "text": "", "table": corr.reset_index()}
        )
    if res and "metrics" in res:
        import pandas as pd
        mtable = pd.DataFrame([res["metrics"]])
        sections.append(
            {"heading": f"Model: {res['model_name']}", "text": "Test-set metrics.",
             "table": mtable}
        )
    data = rp.build_pdf_report(title, sections)
    ext = "pdf" if rp.HAS_REPORTLAB else "html"
    mime = "application/pdf" if rp.HAS_REPORTLAB else "text/html"
    if not rp.HAS_REPORTLAB:
        st.caption("reportlab not installed — exporting an HTML report instead (install reportlab for PDF).")
    st.download_button(
        f"⬇️ Download report ({ext.upper()})", data,
        file_name=rp.report_filename("saturn_report", ext), mime=mime,
    )
