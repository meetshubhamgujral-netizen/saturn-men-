# 🪐 Saturn – AI Market Understanding Dashboard

A colourful, interactive **Streamlit** analytics platform that performs descriptive
analytics, diagnostic analytics, machine learning, explainable AI and
dataset-grounded conversational analysis — **exclusively on the dataset you
upload**. No sample data is bundled and no insight is ever invented from outside
knowledge.

Built to look like a modern BI tool (Power BI / Tableau / Looker Studio) rather
than a notebook: gradient hero, KPI cards, tabs, hover effects and Plotly
visuals throughout.

---

## ✨ Features

| Module | What it does |
| --- | --- |
| 🏠 **Home** | Hero landing + live KPI overview of the loaded dataset |
| 📁 **Upload Dataset** | CSV / Excel / JSON upload, auto type detection, target suggestions |
| 🔎 **Data Explorer** | Profiling, dtypes, missing-value map, filter & preview |
| 🧹 **Data Cleaning** | Missing values, duplicates, outliers, encoding, scaling, feature selection — with an audit trail and reset |
| 📊 **Descriptive Analytics** | Summary stats, correlation & covariance, distributions, pivots, frequency tables, top/bottom, trends |
| 📈 **Diagnostic Analytics** | Random-forest drivers, Chi-Square, ANOVA, T-Test, PCA, segment analysis — each with a plain-English interpretation |
| 🤖 **Machine Learning** | Auto classification/regression/clustering detection, a full model zoo, single-model training and side-by-side comparison |
| 🧠 **Explainable AI** | Native importance, permutation importance, optional SHAP, single-prediction breakdown |
| 💬 **AI Data Chat** | Deterministic, dataset-grounded Q&A — **no API key, no outside knowledge** |
| 📑 **Reports** | Export cleaned data (CSV/Excel), predictions, summary stats and a PDF/HTML report |
| ⚙️ **Settings** | Reset, clear session, theme & about |

---

## 🚀 Quick start

```bash
# 1. Clone
git clone <your-repo-url>.git
cd saturn-dashboard

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

The app opens at <http://localhost:8501>. Upload a file on the **Upload Dataset**
page and the rest of the modules unlock automatically.

> **Python 3.11 or 3.12 recommended.**

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. On <https://share.streamlit.io> click **New app**.
3. Point it at your repo, branch and `app.py`.
4. Deploy — `requirements.txt` is detected automatically. No code changes needed.

---

## 🧩 Optional dependencies

The core app runs on `requirements.txt` alone — those packages all ship prebuilt
wheels and install cleanly on Streamlit Community Cloud.

Three **optional** extras live in `requirements-optional.txt`. The app
**degrades gracefully** without them (the relevant control hides with a note),
and enables the feature automatically when present:

| Package | Unlocks |
| --- | --- |
| `xgboost` | XGBoost classifier / regressor in the model zoo |
| `shap` | SHAP feature-impact plots on the Explainable AI page |
| `reportlab` | True PDF report export (otherwise a styled HTML report) |

Install the extras locally with:

```bash
pip install -r requirements-optional.txt
```

> ⚠️ **Do not** copy the optional packages into the Cloud `requirements.txt`
> unless you have confirmed they build on the Cloud's Python version. `shap`
> pulls in `numba`/`llvmlite`, which frequently fail to compile on the newest
> Python and will abort the entire deploy. A `.python-version` file pinning
> Python **3.12** is included to keep builds predictable.

---

## 🗂️ Project structure

```
saturn-dashboard/
├── app.py                     # Home / entry point
├── pages/                     # One file per sidebar module (Streamlit multipage)
│   ├── 1_📁_Upload_Dataset.py
│   ├── 2_🔎_Data_Explorer.py
│   ├── 3_🧹_Data_Cleaning.py
│   ├── 4_📊_Descriptive_Analytics.py
│   ├── 5_📈_Diagnostic_Analytics.py
│   ├── 6_🤖_Machine_Learning.py
│   ├── 7_🧠_Explainable_AI.py
│   ├── 8_💬_AI_Data_Chat.py
│   ├── 9_📑_Reports.py
│   └── 10_⚙️_Settings.py
├── utils/                     # Streamlit-free computation core (unit-testable)
│   ├── data_loader.py         #   loading, type inference, profiling
│   ├── cleaning.py            #   cleaning transforms
│   ├── analytics.py           #   descriptive analytics + Plotly figures
│   ├── diagnostics.py         #   statistical tests, PCA, drivers
│   ├── ml.py                  #   task detection, model zoo, evaluation
│   ├── explain.py             #   permutation / SHAP / breakdown
│   ├── chat.py                #   deterministic NL query engine
│   ├── reporting.py           #   CSV / Excel / PDF exports
│   └── theme.py               #   CSS + session-state helpers (only ST-aware util)
├── components/                # Presentational widgets (KPI cards)
├── assets/                    # Reference stylesheet
├── data/  models/  exports/   # Runtime folders (git-ignored)
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

### Why the `utils/` layer avoids Streamlit
All heavy logic lives in `utils/` as plain functions over pandas/NumPy/Plotly.
That keeps the computation **testable without a running Streamlit server** and
makes the pages thin, readable UI wrappers.

---

## 💬 A note on the "AI Data Chat"

The chat has two layers and **always responds**:

1. A **deterministic engine** turns your question into a real pandas computation
   over your DataFrame (summaries, rankings, averages, comparisons, correlations,
   charts). It needs no network and never invents numbers.

2. An optional **free, key-free LLM** (via [Pollinations.ai](https://pollinations.ai))
   rewords those real computed numbers into a natural reply. It is always fed the
   computed facts and instructed to use only them, so it stays grounded in your
   data. Toggle **✨ Use free AI** on the chat page.

If the free endpoint is slow, rate-limited or offline, the chat automatically
falls back to layer 1 — so you always get an answer. No API key, ever.

> Want a different provider? Swap the call in `utils/chat.py → llm_answer()`.
> Anything that speaks the OpenAI message format (a local Ollama server, etc.)
> drops in with a one-line URL change; the page UI doesn't change.

---

## 📜 License

MIT — see [LICENSE](LICENSE).
