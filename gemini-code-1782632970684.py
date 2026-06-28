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
                
        if user_
