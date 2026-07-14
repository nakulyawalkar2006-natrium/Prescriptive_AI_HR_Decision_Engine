import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model import train_attrition_model
from retension_check import analyze_retention

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="HR Retention AI", layout="wide", page_icon="🏢")
st.title("🏢 Enterprise HR Retention & AI Optimizer")

# ==========================================
# 2. LOAD AI ENGINE
# ==========================================
# This calls your cached function from model.py
assets = train_attrition_model()
x_test_raw = assets["x_test_raw"]

# ==========================================
# 3. APP LAYOUT & TABS
# ==========================================
tab1, tab2 = st.tabs(["👤 Individual Analysis & Interventions", "🗺️ Company Macro Heatmap"])

# ------------------------------------------
# TAB 1: INDIVIDUAL "WHAT-IF" & EXPLAINER
# ------------------------------------------
with tab1:
    st.markdown("### Search Employee Database")
    col_search, col_raise = st.columns([1, 1])
    
    with col_search:
        # User selects from the test set indices. We create a mapping of dropdown options to actual row positions.
        row_indices = range(len(x_test_raw))
        selected_position = st.selectbox("Select Employee ID:", row_indices)
        
    with col_raise:
        raise_pct = st.slider("Test Retention Raise (%)", min_value=5, max_value=50, value=15) / 100.0
    
    st.divider()

    # Isolate employee data based on positional index
    emp_data = x_test_raw.iloc[[selected_position]].copy()
    
    # SIDEBAR: What-If Simulator
    st.sidebar.header("🧪 What-If Simulator")
    st.sidebar.markdown("Test HR interventions to see if you can lower their flight risk.")
    
    if 'Annual_CTC_LPA' in emp_data.columns:
        new_ctc = st.sidebar.number_input("Adjust CTC (LPA)", value=float(emp_data['Annual_CTC_LPA'].values[0]))
        emp_data['Annual_CTC_LPA'] = new_ctc
    if 'Overwork_Strain' in emp_data.columns:
        new_strain = st.sidebar.slider("Adjust Overwork Strain", 0.0, 10.0, float(emp_data['Overwork_Strain'].values[0]))
        emp_data['Overwork_Strain'] = new_strain
    if 'EnvironmentSatisfaction' in emp_data.columns:
        new_env = st.sidebar.slider("Improve Env Satisfaction (1-4)", 1, 4, int(emp_data['EnvironmentSatisfaction'].values[0]))
        emp_data['EnvironmentSatisfaction'] = new_env

    # Run Prediction for this specific employee
    emp_clean = assets["selector"].transform(emp_data)
    current_prob = assets["model"].predict_proba(emp_clean)[0, 1]
    
    # Run Financial Calculator from economics.py
    metrics = analyze_retention(selected_position, current_prob, x_test_raw, raise_pct)

    # UI Rendering: Top Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Flight Risk", f"{current_prob*100:.1f}%")
    c2.metric("Expected Financial Risk", f"{metrics['risk_adjusted_loss']:.2f} LPA")
    
    decision = metrics['recommended_action']
    if decision == "Give Raise":
        c3.success(f"Strategy: {decision}")
    elif decision == "Allow Exit":
        c3.error(f"Strategy: {decision}")
    else:
        c3.info(f"Strategy: {decision}")

    # UI Rendering: AI Explainer
    st.markdown("### 🧠 AI Explainer: Why is this happening?")
    deviations = []
    
    for feature in assets["kept_features"]:
        emp_val = emp_data[feature].values[0]
        safe_val = assets["safe_averages"][feature]
        diff = (emp_val - safe_val) / (safe_val + 1e-9)
        deviations.append({
            "Feature": feature, 
            "Emp Value": emp_val, 
            "Avg Safe Value": round(safe_val, 2), 
            "Deviation %": round(diff*100, 1)
        })
    
    dev_df = pd.DataFrame(deviations)
    dev_df['Abs_Dev'] = abs(dev_df['Deviation %'])
    dev_df = dev_df.sort_values(by='Abs_Dev', ascending=False).head(5)
    
    st.write("Top factors driving this employee's risk compared to employees who stay:")
    st.dataframe(dev_df[['Feature', 'Emp Value', 'Avg Safe Value', 'Deviation %']], hide_index=True, use_container_width=True)

# ------------------------------------------
# TAB 2: COMPANY MACRO HEATMAP
# ------------------------------------------
with tab2:
    st.markdown("### 🗺️ Company-Wide Flight Risk Heatmap")
    
    # Predict for the entire test set
    all_clean = assets["selector"].transform(x_test_raw)
    all_probs = assets["model"].predict_proba(all_clean)[:, 1]
    
    summary = x_test_raw.copy()
    summary['Flight_Risk'] = all_probs
    
    role_cols = [c for c in summary.columns if 'JobRole_' in c]
    
    if len(role_cols) > 0:
        role_data = []
        for role in role_cols:
            role_name = role.replace('JobRole_', '')
            avg_risk = summary[summary[role] == 1]['Flight_Risk'].mean()
            role_data.append({"Department/Role": role_name, "Average Flight Risk": avg_risk})
            
        heat_df = pd.DataFrame(role_data).dropna().sort_values(by="Average Flight Risk", ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=heat_df, y="Department/Role", x="Average Flight Risk", palette="Reds_r", ax=ax)
        ax.set_title("Average Flight Risk by Job Role")
        ax.axvline(assets["optimal_threshold"], color='red', linestyle='--', label="Danger Threshold")
        ax.legend()
        
        st.pyplot(fig)
    else:
        st.write("Could not identify JobRole columns for heatmap grouping.")