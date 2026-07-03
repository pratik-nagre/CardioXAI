import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import torch
import sys
import os

# Ensure src is accessible
sys.path.append(os.path.abspath('src'))
from data_pipeline import get_train_test_splits, IMMUTABLE_FEATURES, ACTIONABLE_FEATURES, DIRECTIONAL_FEATURES
from optimizer import CounterfactualOptimizer

st.set_page_config(page_title="Cardiovascular Risk & Personalized Recourse", layout="wide")

@st.cache_resource
def load_resources():
    # Load data
    train_X, test_X, train_y, test_y = get_train_test_splits(filepath="data/frmgham2.csv", random_seed=42)
    
    # Load Models
    cph_model = joblib.load('models/cox_model.joblib')
    xgb_model = joblib.load('models/xgb_model.joblib')
    lgb_model = joblib.load('models/lgb_model.joblib')
    rf_model = joblib.load('models/rf_model.joblib')
    
    opt = CounterfactualOptimizer(models_dir='models')
    
    return train_X, test_X, train_y, test_y, cph_model, xgb_model, lgb_model, rf_model, opt

with st.spinner("Loading models and data..."):
    train_X, test_X, train_y, test_y, cph_model, xgb_model, lgb_model, rf_model, opt = load_resources()

# ---- 1. Sidebar: Patient Selection & Input ----
st.sidebar.title("Patient Profile")

# Select a sample patient
patient_options = [f"Sample Patient {i}" for i in range(len(test_X))]
selected_patient_name = st.sidebar.selectbox("Load Sample High-Risk Patient", patient_options, index=0)
patient_idx = int(selected_patient_name.split()[-1])

# Initialize session state if changing patient
if 'current_patient_idx' not in st.session_state or st.session_state.current_patient_idx != patient_idx:
    st.session_state.current_patient_idx = patient_idx
    patient_series = test_X.iloc[patient_idx].copy()
    for col in patient_series.index:
        st.session_state[col] = patient_series[col]
    st.session_state.patient_id = test_X.index[patient_idx]

st.sidebar.subheader("Adjust Parameters")

# Sliders for continuous & selectors for binary
col_ranges = {
    'SYSBP': (90.0, 220.0),
    'DIABP': (60.0, 140.0),
    'TOTCHOL': (120.0, 400.0),
    'BMI': (15.0, 60.0),
    'GLUCOSE': (60.0, 300.0),
    'HEARTRTE': (50.0, 140.0),
    'CIGPDAY': (0.0, 60.0),
    'AGE': (30.0, 90.0)
}

input_data = {}
for col in test_X.columns:
    val = st.session_state.get(col, test_X.iloc[patient_idx][col])
    if col in col_ranges:
        min_v, max_v = col_ranges[col]
        val = float(val)
        # Bounded slider, clamp initial value if needed
        val = max(min_v, min(max_v, val))
        input_data[col] = st.sidebar.slider(f"{col}", min_value=float(min_v), max_value=float(max_v), value=val, key=col)
    elif col in ['CURSMOKE', 'DIABETES', 'BPMEDS']:
        input_data[col] = st.sidebar.selectbox(f"{col}", options=[0, 1], index=int(val), key=col)
    else:
        # Keep immutable as is or read-only text
        input_data[col] = val

patient_df = pd.DataFrame([input_data], index=[st.session_state.patient_id])

# ---- 2. Sidebar Expander: Audit Log ----
with st.sidebar.expander("Algorithmic Recourse Integrity Audit Log", expanded=False):
    st.markdown("""
    **Methodology & Active Constraints**
    - `SYSBP`, `DIABP`, `TOTCHOL`, `GLUCOSE`, `BMI`: Capped at ±20% of baseline.
    - `HEARTRTE`: Capped at ±15% (Frailty confound override).
    - `CIGPDAY`: Capped at -50%.
    - `BPMEDS`: Increase-only (Cannot recommend stopping meds).
    - `DIABETES`: Immutable (Permanent diagnosis).
    - `AGE`: Immutable (Locked for point-in-time recommendation).
    - Demographic / Prior Events: Strictly Immutable.
    """)

# ---- 3. Main Panel: Baseline Risk & Ensemble Check ----
st.title("Cardiovascular Risk Prediction & Personalized Recourse")

cox_input = patient_df[opt.cox_feature_cols]
baseline_hazard = cph_model.predict_partial_hazard(cox_input).iloc[0]
baseline_log_hazard = np.log(baseline_hazard)

st.subheader("Baseline Risk Evaluation")
col1, col2 = st.columns([1, 1])

with col1:
    st.metric(label="Cox Log Partial Hazard", value=f"{baseline_log_hazard:.3f}")
    if baseline_log_hazard < 0.0:
        risk_category = "Low Risk"
        risk_color = "green"
    elif baseline_log_hazard < 0.5:
        risk_category = "Medium Risk"
        risk_color = "orange"
    else:
        risk_category = "High Risk"
        risk_color = "red"
        
    st.markdown(f"**Risk Category:** <span style='color:{risk_color}'>{risk_category}</span> *(Tied to Phase 1/2 optimal threshold)*", unsafe_allow_html=True)

with col2:
    st.markdown("**Cross-Model Validity Check**")
    xgb_prob = xgb_model.predict_proba(patient_df)[:, 1][0]
    lgb_prob = lgb_model.predict_proba(patient_df)[:, 1][0]
    rf_prob = rf_model.predict_proba(patient_df)[:, 1][0]
    
    threshold = 0.15 # Optimal F1 cutoff from Phase 1/2
    votes = sum([1 for p in [xgb_prob, lgb_prob, rf_prob] if p > threshold])
    st.info(f"{votes} of 3 independent ensemble models agree this patient is high-risk (>15% 10-yr probability).")

# ---- 4. Main Panel: Counterfactual Search & Trajectory ----
st.markdown("---")
st.subheader("Actionable Risk-Reduction Plan")

if st.button("Generate Personalized Risk-Reduction Plan", type="primary"):
    with st.spinner("Searching latent manifold for safe interventions..."):
        res = opt.optimize(patient_df)
        
    if not res['success']:
        st.warning("No clinically feasible recourse found within bounds. This patient's risk is primarily driven by immutable demographics and medical history.")
    else:
        cf_df = pd.DataFrame(res['x_cf'], columns=opt.ae_feature_cols)
        cf_cox_input = cf_df[opt.cox_feature_cols]
        cf_hazard = cph_model.predict_partial_hazard(cf_cox_input).iloc[0]
        cf_log_hazard = np.log(cf_hazard)
        
        st.success("Intervention plan generated successfully.")
        
        # Display prescription
        st.markdown("### Prescription")
        st.caption("Model-predicted risk reduction if changes are sustained.")
        
        orig_native = patient_df.iloc[0]
        cf_native = cf_df.iloc[0]
        
        diffs = []
        for feat in ACTIONABLE_FEATURES:
            if feat in orig_native and feat in cf_native:
                o_val = orig_native[feat]
                c_val = cf_native[feat]
                if abs(o_val - c_val) > 0.5:
                    diffs.append(f"**{feat}**: {o_val:.1f} &rarr; {c_val:.1f} ({c_val - o_val:+.1f})")
                    
        col_pres, col_metric = st.columns([2, 1])
        with col_pres:
            if len(diffs) > 0:
                for d in diffs:
                    st.markdown(f"- {d}")
            else:
                st.markdown("*No major actionable changes required.*")
                
        with col_metric:
            st.metric(label="Target Cox Log Partial Hazard", value=f"{cf_log_hazard:.3f}", delta=f"{cf_log_hazard - baseline_log_hazard:.3f} (Risk Drop)", delta_color="inverse")

        # 10-Year Survival Trajectory Plot
        st.markdown("### 10-Year Survival Trajectory")
        surv_baseline = cph_model.predict_survival_function(cox_input)
        surv_cf = cph_model.predict_survival_function(cf_cox_input)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=surv_baseline.index, y=surv_baseline.iloc[:, 0], mode='lines', name='Baseline Trajectory', line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=surv_cf.index, y=surv_cf.iloc[:, 0], mode='lines', name='Counterfactual Trajectory', line=dict(color='green', dash='dash', width=3)))
        
        # Add vertical markers for 1, 3, 5, 10 years
        for t in [1, 3, 5, 10]:
            fig.add_vline(x=t, line_width=1, line_dash="dot", line_color="gray")
            fig.add_annotation(x=t, y=0.52, text=f"{t}yr", showarrow=False, textangle=-90, font=dict(color="gray"))
            
        fig.update_layout(
            xaxis_title="Time (Years)",
            yaxis_title="Survival Probability (Event-Free)",
            title="Baseline vs. Recommended Counterfactual Survival Curves",
            xaxis=dict(range=[0, 10.5]),
            yaxis=dict(range=[0.5, 1.05]),
            height=500,
            margin=dict(l=50, r=50, b=50, t=50, pad=4)
        )
        st.plotly_chart(fig, use_container_width=True)
