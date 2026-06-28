import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# ==========================================
# 1. GLOBAL SYSTEM CONFIGURATIONS & STYLES
# ==========================================
st.set_page_config(
    page_title="Saturn AI – Market Dashboard",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_premium_theme():
    """Injects high-fidelity custom SaaS styling mimicking Power BI/Looker Studio."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [data-testid="stSidebarNav"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Modern Gradient Hero Container */
        .hero-container {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            padding: 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        
        /* Premium Business KPI Cards */
        .kpi-wrapper {
            background: linear-gradient(145deg, #ffffff, #f3f4f6);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 6px solid #203a43;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            transition: transform 0.2s ease-in-out;
            margin-bottom: 1rem;
        }
        .kpi-wrapper:hover {
            transform: translateY(-3px);
        }
        
        /* Metric Typography Corrections */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #111827;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            padding: 10px 20px;
        }
        </style>
    """, unsafe_allow_html=True)

apply_premium_theme()

# ==========================================
# 2. APPLICATION CORE SESSION STATE STORAGE
# ==========================================
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# 3. ENHANCED SYSTEM ANALYTICAL ENGINES
# ==========================================
class SaturnDataEngine:
    @staticmethod
    @st.cache_data
    def inspect_dataset(df: pd.DataFrame) -> dict:
        """Inspects structural vector attributes, features, types and memory footprint."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Identify possible datetime sequences safely
        date_cols = []
        for col in categorical_cols[:]:
            try:
                sample = df[col].dropna().head(50)
                if pd.to_datetime(sample, errors='coerce').notna().sum() / max(len(sample), 1) > 0.8:
                    date_cols.append(col)
                    categorical_cols.remove(col)
            except Exception:
                pass

        suggested = categorical_cols[0] if categorical_cols else (numeric_cols[-1] if numeric_cols else df.columns[-1])
        
        return {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "missing": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "memory": f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB",
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "date_cols": date_cols,
            "suggested_target": suggested
        }

    @staticmethod
    def clean_dataset(df: pd.DataFrame, missing_strat: str, drop_dups: bool) -> pd.DataFrame:
        """Executes operational cleaning transformations on state data."""
        df_out = df.copy()
        if drop_dups:
            df_out = df_out.drop_duplicates()
        if missing_strat == "Drop Rows":
            df_out = df_out.dropna()
        elif missing_strat == "Mean/Mode Imputation":
            for col in df_out.columns:
                if df_out[col].dtype in [np.float64, np.int64]:
                    df_out[col] = df_out[col].fillna(df_out[col].mean())
                else:
                    if not df_out[col].mode().empty:
                        df_out[col] = df_out[col].fillna(df_out[col].mode()[0])
        return df_out

class SaturnDiagnosticEngine:
    @staticmethod
    def calculate_anova(df: pd.DataFrame, cat_col: str, num_col: str) -> dict:
        """Evaluates relationship significance via one-way Analysis of Variance."""
        try:
            groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
            if len(groups) < 2:
                return {"status": "Failure", "msg": "Insufficient groups for evaluation."}
            f_stat, p_val = stats.f_oneway(*groups)
            return {"status": "Success", "f_stat": f_stat, "p_value": p_val, "significant": p_val < 0.05}
        except Exception as e:
            return {"status": "Failure", "msg": str(e)}

# ==========================================
# 4. MOCK GENERATOR FOR INSTANT TESTING
# ==========================================
def load_mock_market_data():
    """Generates a premium synthetic corporate dataset if no data is uploaded."""
    np.random.seed(42)
    n = 250
    regions = ["Dubai", "Abu Dhabi", "Sharjah", "Riyadh"]
    segments = ["Corporate", "Enterprise", "Consumer"]
    
    mock_data = pd.DataFrame({
        "Transaction_ID": [f"TX-{1000+i}" for i in range(n)],
        "Region": np.random.choice(regions, size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "Segment": np.random.choice(segments, size=n, p=[0.5, 0.3, 0.2]),
        "Marketing_Spend_AED": np.random.randint(5000, 45000, size=n),
        "Conversion_Rate": np.random.uniform(1.2, 8.5, size=n),
        "Customer_Satisfaction": np.random.randint(1, 6, size=n),
        "Revenue_AED": np.random.randint(20000, 150000, size=n)
    })
    mock_data["Revenue_AED"] = (mock_data["Marketing_Spend_AED"] * 2.5) + (mock_data["Conversion_Rate"] * 5000) + np.random.normal(0, 5000, n)
    return mock_data

# ==========================================
# 5. SIDEBAR NAVIGATION CONTEXT PANEL
# ==========================================
st.sidebar.title(" Saturn Core Engine")
app_mode = st.sidebar.radio(
    "Control Panel Navigation",
    [
        "🏠 Home & Lander",
        "📁 Ingest & Inspect",
        "🧹 Pipeline Cleaning",
        "📊 Descriptive Suite",
        "📈 Diagnostics Engine",
        "🤖 Machine Learning",
        "💬 Conversational Data AI"
    ]
)

if st.sidebar.button("Load Premium Demo Market Data"):
    demo_df = load_mock_market_data()
    st.session_state.raw_df = demo_df
    st.session_state.cleaned_df = demo_df.copy()
    st.session_state.metadata = SaturnDataEngine.inspect_dataset(demo_df)
    st.sidebar.success("Demo dataset injected into memory.")

# ==========================================
# VIEW 1: HOME & HERO LANDER
# ==========================================
if app_mode == "🏠 Home & Lander":
    st.markdown("""
        <div class="hero-container">
            <h1 style='margin:0; font-size: 2.5rem; font-weight:700;'>Saturn – AI Market Understanding Dashboard</h1>
            <p style='font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem;'>
                Enterprise SaaS platform for statistical diagnostics, custom machine learning workflows, and deterministic business exploration.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Accelerated Analytical Infrastructure")
        st.write(
            "Saturn acts as an end-to-end environment that replaces disconnected data scripts with an interactive dashboard. "
            "It runs mathematical evaluations exclusively over your data context, ensuring security and structural truth."
        )
        st.info("💡 **Quickstart Workflow:** Go to **📁 Ingest & Inspect** in the sidebar to drop a CSV/XLSX file, or click the **Load Premium Demo Market Data** button to explore with mock variables instantly.")
    with col2:
        st.markdown("### Node Diagnostic Metrics")
        st.metric(label="Runtime Context Environment", value="Python 3.12+", delta="System Stable")
        st.metric(label="Execution Framework Core", value="Streamlit Native Monolithic", delta="Cloud Deployable")

# ==========================================
# VIEW 2: INGESTION & DATA INSPECTION
# ==========================================
elif app_mode == "📁 Ingest & Inspect":
    st.title("📁 Ingest & Structural Inspection Node")
    
    uploaded_file = st.file_uploader("Upload operational business dataset (CSV, XLSX, or JSON)", type=["csv", "xlsx", "json"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
                
            st.session_state.raw_df = df
            st.session_state.cleaned_df = df.copy()
            st.session_state.metadata = SaturnDataEngine.inspect_dataset(df)
            st.success(f"Pipeline Activated: successfully read {df.shape[0]} matrix rows.")
        except Exception as e:
            st.error(f"Ingestion System Malfunction: {str(e)}")

    if st.session_state.raw_df is not None:
        m = st.session_state.metadata
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="kpi-wrapper"><p style="margin:0;color:#6b7280;font-size:0.85rem;text-transform:uppercase;font-weight:600;">Total Records</p><h2 style="margin:0;color:#111827;">{m["rows"]:,}</h2></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="kpi-wrapper"><p style="margin:0;color:#6b7280;font-size:0.85rem;text-transform:uppercase;font-weight:600;">Dimensions</p><h2 style="margin:0;color:#111827;">{m["cols"]} Features</h2></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="kpi-wrapper"><p style="margin:0;color:#6b7280;font-size:0.85rem;text-transform:uppercase;font-weight:600;">Null Elements</p><h2 style="margin:0;color:#111827;">{m["missing"]:,} Cells</h2></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="kpi-wrapper"><p style="margin:0;color:#6b7280;font-size:0.85rem;text-transform:uppercase;font-weight:600;">RAM Footprint</p><h2 style="margin:0;color:#111827;">{m["memory"]}</h2></div>', unsafe_allow_html=True)
        
        st.markdown("### Data Struct Layout Profiler")
        st.json({
            "Continuous/Numeric Formats": m["numeric_cols"],
            "Categorical Text Groupings": m["categorical_cols"],
            "Inferred DateTime Dimensions": m["date_cols"],
            "Automated Target Label Selection Recommendation": m["suggested_target"]
        })
        
        st.markdown("### Raw Workspace Preview Matrix")
        st.dataframe(st.session_state.raw_df.head(15), use_container_width=True)
    else:
        st.info("System waiting for record configuration map ingestion. Mount an input pipeline or click 'Load Premium Demo Market Data'.")

# ==========================================
# VIEW 3: PIPELINE DATA CLEANING
# ==========================================
elif app_mode == "🧹 Pipeline Cleaning":
    st.title("🧹 Structural Pipeline Preprocessing & Transformation")
    
    if st.session_state.raw_df is None:
        st.warning("Data workspace empty. Ingest data components before applying cleaning logic.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Operations Configuration")
            missing_strat = st.selectbox("Missing Cell Resolution Strategy", ["Keep Unaltered", "Drop Rows", "Mean/Mode Imputation"])
            drop_dups = st.checkbox("Scrub/Remove Duplicate Vectors", value=True)
            
            if st.button("Execute Pipeline Optimization Routine"):
                st.session_state.cleaned_df = SaturnDataEngine.clean_dataset(st.session_state.raw_df, missing_strat, drop_dups)
                st.session_state.metadata = SaturnDataEngine.inspect_dataset(st.session_state.cleaned_df)
                st.success("Clean state pipeline processed and written successfully.")
                
        with col2:
            st.markdown("#### Before vs After Pipeline Audit")
            b_len = len(st.session_state.raw_df)
            a_len = len(st.session_state.cleaned_df)
            
            c1, c2 = st.columns(2)
            c1.metric("Input Registry Volume", b_len)
            c2.metric("Output Post-Cleaning Volume", a_len, delta=a_len - b_len)
            
            st.markdown("##### Preprocessed State Output Snippet")
            st.dataframe(st.session_state.cleaned_df.head(5), use_container_width=True)

# ==========================================
# VIEW 4: DESCRIPTIVE ANALYTICS SUITE
# ==========================================
elif app_mode == "📊 Descriptive Suite":
    st.title("📊 High-Fidelity Descriptive Analytics Suite")
    
    if st.session_state.cleaned_df is None:
        st.warning("Cleaned analytics matrix context absent. Initialize a dataset pipeline to continue.")
    else:
        df = st.session_state.cleaned_df
        m = st.session_state.metadata
        
        t1, t2, t3 = st.tabs(["Linear Correlation Models", "Distribution Visualizers", "Enterprise Cross-Tab Panels"])
        
        with t1:
            st.markdown("#### Continuous Dimension Covariance Matrix")
            if len(m["numeric_cols"]) > 1:
                corr = df[m["numeric_cols"]].corr()
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Correlation modeling requires multiple continuous numeric structural elements.")
                
        with t2:
            st.markdown("#### Frequency Distribution Maps")
            selected_feature = st.selectbox("Isolate Feature Node for Granular Split Analysis", df.columns)
            
            if df[selected_feature].dtype in [np.float64, np.int64]:
                fig = px.histogram(df, x=selected_feature, box=True, color_discrete_sequence=["#203a43"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                freq = df[selected_feature].value_counts().reset_index()
                fig = px.bar(freq, x=selected_feature, y="count", color="count", color_continuous_scale="Blugrn")
                st.plotly_chart(fig, use_container_width=True)
                
        with t3:
            st.markdown("#### Strategic Dynamic Pivot Synthesis Model")
            if len(m["categorical_cols"]) >= 2 and len(m["numeric_cols"]) >= 1:
                r_sel = st.