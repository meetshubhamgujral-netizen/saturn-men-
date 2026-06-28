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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');
        
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
                r_sel = st.selectbox("Horizontal Categorical Index Row Factor", m["categorical_cols"], index=0)
                c_sel = st.selectbox("Vertical Categorical Stratification Column Group", m["categorical_cols"], index=min(1, len(m["categorical_cols"])-1))
                v_sel = st.selectbox("Aggregation Target Metric Value", m["numeric_cols"], index=0)
                
                pivot_matrix = df.pivot_table(index=r_sel, columns=c_sel, values=v_sel, aggfunc='mean')
                st.markdown("##### Computed Aggregation Grid Matrix (Mean Value Calculation)")
                st.dataframe(pivot_matrix, use_container_width=True)
            else:
                st.info("Pivot matrices require at least two distinct category matrices and one value vector.")

# ==========================================
# VIEW 5: DIAGNOSTICS STATISTICAL ENGINE
# ==========================================
elif app_mode == "📈 Diagnostics Engine":
    st.title("📈 Diagnostic Root-Cause Variance Engine")
    
    if st.session_state.cleaned_df is None:
        st.warning("Data components not found. Run dataset configuration steps first.")
    else:
        df = st.session_state.cleaned_df
        m = st.session_state.metadata
        
        st.markdown("### Automated Inferential Hypothesis Verification Loop")
        if len(m["categorical_cols"]) > 0 and len(m["numeric_cols"]) > 0:
            ind_var = st.selectbox("Independent Group Factor Selection (X Axis Categorical)", m["categorical_cols"])
            dep_var = st.selectbox("Dependent Target Variance Metric (Y Axis Continuous)", m["numeric_cols"])
            
            res = SaturnDiagnosticEngine.calculate_anova(df, ind_var, dep_var)
            
            if res["status"] == "Success":
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Computed ANOVA F-Statistic Score", f"{res['f_stat']:.4f}")
                with c2:
                    st.metric("Probability P-Value Metrics", f"{res['p_value']:.4e}")
                
                if res["significant"]:
                    st.success(f"📊 **Automated Diagnostic Interpretation:** The variance distribution groups of **{ind_var}** exhibit structural deviations across the mean performance bands of **{dep_var}** that are statistically significant. This confirms a measurable correlation.")
                else:
                    st.info(f"📊 **Automated Diagnostic Interpretation:** Null Hypothesis Accepted. The cross-group drift profile inside **{ind_var}** contains marginal divergence thresholds that do not meet standard significance filters.")
                
                fig = px.box(df, x=ind_var, y=dep_var, color=ind_var, title=f"High-Density Spatial Diagnostic Distribution Matrix Map")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Diagnostic engines require an overlapping structured array composed of both distinct categorical parameters and numeric indicators.")

# ==========================================
# VIEW 6: ESTIMATION & MACHINE LEARNING
# ==========================================
elif app_mode == "🤖 Machine Learning":
    st.title("🤖 Enterprise Machine Learning Prototyper Node")
    
    if st.session_state.cleaned_df is None:
        st.warning("Analytical matrix arrays are unitialized. Inject a valid model file input matrix pattern.")
    else:
        df = st.session_state.cleaned_df
        m = st.session_state.metadata
        
        st.markdown("### Objective Targeting & Architectural Selections")
        target_y = st.selectbox("Select Target Optimization Variable (Y Label)", df.columns, index=df.columns.get_loc(m["suggested_target"]))
        
        candidate_features = [col for col in df.columns if col != target_y]
        selected_features = st.multiselect("Select Estimator Engine Input Metrics (X Features)", candidate_features, default=candidate_features[:4])
        
        if not selected_features:
            st.error("Select at least one continuous or categorical parameter node input feature.")
        else:
            is_numeric = df[target_y].dtype in [np.float64, np.int64] and df[target_y].nunique() > 10
            framework_type = "Regression Mode Engine" if is_numeric else "Classification Mode Engine"
            st.info(f"Dynamic structural analysis recommends execution framework alignment: **{framework_type}**")
            
            if st.button("Compile Estimator Pipeline Structure"):
                with st.spinner("Executing model engineering convergence routine..."):
                    X_processed = pd.get_dummies(df[selected_features], drop_first=True)
                    y_processed = df[target_y]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
                    
                    if is_numeric:
                        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
                        regressor.fit(X_train, y_train)
                        predictions = regressor.predict(X_test)
                        
                        r2_score_val = r2_score(y_test, predictions)
                        rmse_val = np.sqrt(mean_squared_error(y_test, predictions))
                        
                        st.markdown("#### Performance Metrics Output Matrix")
                        st.metric("R² Score (Variance Explanation Mapping)", f"{r2_score_val:.4f}")
                        st.metric("System Root Mean Squared Error (RMSE)", f"{rmse_val:.4f}")
                        
                        importance = regressor.feature_importances_
                        feat_imp_df = pd.DataFrame({"Feature Node": X_processed.columns, "Calculated Weight Variance Value": importance}).sort_values(by="Calculated Weight Variance Value", ascending=False)
                        
                        fig = px.bar(feat_imp_df, x="Calculated Weight Variance Value", y="Feature Node", orientation='h', title="Explainable AI Model Feature Importance Map")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                        classifier.fit(X_train, y_train)
                        predictions = classifier.predict(X_test)
                        
                        accuracy = accuracy_score(y_test, predictions)
                        st.markdown("#### Classification Model Report Metrics")
                        st.metric("Out-Of-Sample Model Prediction Accuracy Metric", f"{accuracy*100:.2f}%")
                        
                        st.markdown("##### Detailed Classification Report Summary Matrix")
                        st.text(classification_report(y_test, predictions))

# ==========================================
# VIEW 7: DETERMINISTIC AI CONVERSATIONAL CHAT
# ==========================================
elif app_mode == "💬 Conversational Data AI":
    st.title("💬 Saturn AI Local Deterministic Chat Subsystem")
    
    if st.session_state.cleaned_df is None:
        st.warning("Conversational subroutines remain deactivated. Please configure an analytical model frame.")
    else:
        df = st.session_state.cleaned_df
        
        st.markdown("""
            > ⚠️ **System Isolation Boundary Context Protocol:** This business intelligence assistant is bounded *strictly* to the uploaded data frame structure. It evaluates explicit string conditions and structures context without external information lookup patterns to prevent hallucinations.
        """)
        
        if not st.session_state.chat_history:
            st.session_state.chat_history = [
                {"role": "assistant", "content": f"Operational node ready. I have mapped {df.shape[0]} rows across {df.shape[1]} unique columns. How can I query the data for you?"}
            ]
            
        for conversation_block in st.session_state.chat_history:
            with st.chat_message(conversation_block["role"]):
                st.markdown(conversation_block["content"])
                
        if user_prompt := st.chat_input("Enter natural-language query parameter strings here:"):
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)
                
            normalized_query = user_prompt.lower()
            
            with st.chat_message("assistant"):
                if "column" in normalized_query or "feature" in normalized_query or "attributes" in normalized_query:
                    reply = f"The structural tracking attributes parsed inside the dataset consist of: `{', '.join(df.columns.tolist())}`."
                elif "missing" in normalized_query or "null" in normalized_query:
                    reply = f"The active matrix context map contains exactly {df.isna().sum().sum()} missing null coordinate values."
                elif "summary" in normalized_query or "describe" in normalized_query:
                    reply = f"The data structure contains {df.shape[0]} valid rows. Numeric boundaries for the key metric mean values are:\n\n"
                    num_summary = df.select_dtypes(include=[np.number]).mean().to_dict()
                    for k, v in num_summary.items():
                        reply += f"- **Mean {k}**: {v:.2f}\n"
                elif "region" in normalized_query and "region" in [c.lower() for c in df.columns]:
                    region_col = [c for c in df.columns if c.lower() == "region"][0]
                    top_reg = df[region_col].value_counts().idxmax()
                    reply = f"Analyzing category density: the category group with the highest frequency distribution inside **{region_col}** is **{top_reg}**."
                else:
                    reply = ("The isolated deterministic parsing engine could not safely extract a specific matching attribute from your request. "
                             "Please rephrase your query to focus explicitly on metrics like **'columns'**, **'missing data records'**, or **'describe summary stats'**.")
                
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
