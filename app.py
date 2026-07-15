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
# calls your cached function from model.py
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
    mode = st.radio("Analysis Mode:", ["🔍 Analyze Existing Employee", "➕ Check for a New Hire Profile"], horizontal=True)
    st.divider()
    
    if mode == "🔍 Analyze Existing Employee":
        col_search, col_raise = st.columns([1, 1])
        with col_search:
            # Create user-friendly IDs starting from 1
            employee_ids = range(1, len(x_test_raw) + 1)
            selected_id = st.selectbox("Select Employee ID:", employee_ids)
            # Map it back to the 0-based positional index
            selected_position = selected_id - 1
            emp_data = x_test_raw.iloc[[selected_position]].copy()
            df_for_calc = x_test_raw

        with col_raise:
            raise_pct = st.slider("Test Retention Raise (%)", min_value=5, max_value=50, value=15) / 100.0

        
        # SIDEBAR: What-If Simulator
        st.sidebar.header("🧪 What-If Simulator")
        st.sidebar.markdown("Test HR interventions to lower flight risk.")
        
        # --- 1. Financial & Workload Adjustments ---
        if 'Annual_CTC_LPA' in emp_data.columns:
            new_ctc = st.sidebar.number_input("Adjust CTC (LPA)", value=float(emp_data['Annual_CTC_LPA'].values[0]))
            emp_data['Annual_CTC_LPA'] = new_ctc
            
        if 'MonthlyRate' in emp_data.columns:
            new_mr = st.sidebar.number_input("Adjust Monthly Rate", value=float(emp_data['MonthlyRate'].values[0]), step=500.0)
            emp_data['MonthlyRate'] = new_mr

        if 'OverTime_Yes' in emp_data.columns:
            current_ot = bool(emp_data['OverTime_Yes'].values[0])
            new_ot = st.sidebar.checkbox("Change OverTime?", value=current_ot)
            emp_data['OverTime_Yes'] = int(new_ot)
            
        st.sidebar.divider()
        
        # --- 2. Environment & Satisfaction Adjustments ---
        if 'Overwork_Strain' in emp_data.columns:
            new_strain = st.sidebar.slider("Adjust Overwork Strain", 0.0, 10.0, float(emp_data['Overwork_Strain'].values[0]))
            emp_data['Overwork_Strain'] = new_strain
            
        if 'EnvironmentSatisfaction' in emp_data.columns:
            new_env = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, int(emp_data['EnvironmentSatisfaction'].values[0]))
            emp_data['EnvironmentSatisfaction'] = new_env
            
        if 'JobSatisfaction' in emp_data.columns:
            new_js = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, int(emp_data['JobSatisfaction'].values[0]))
            emp_data['JobSatisfaction'] = new_js
            
        if 'JobInvolvement' in emp_data.columns:
            new_ji = st.sidebar.slider("Job Involvement (1-4)", 1, 4, int(emp_data['JobInvolvement'].values[0]))
            emp_data['JobInvolvement'] = new_ji

        st.sidebar.divider()
        
        # --- 3. Business Travel Adjustment ---
        if 'BusinessTravel_Travel_Frequently' in emp_data.columns and 'BusinessTravel_Travel_Rarely' in emp_data.columns:
            # Check the dummy columns to figure out their current status
            if emp_data['BusinessTravel_Travel_Frequently'].values[0] == 1:
                current_travel = "Travel Frequently"
            elif emp_data['BusinessTravel_Travel_Rarely'].values[0] == 1:
                current_travel = "Travel Rarely"
            else:
                current_travel = "Non-Travel"
                
            new_travel = st.sidebar.radio(
                "Business Travel", 
                ["Non-Travel", "Travel Rarely", "Travel Frequently"], 
                index=["Non-Travel", "Travel Rarely", "Travel Frequently"].index(current_travel)
            )
            
            # Apply the new selection back into the machine-readable dummy columns
            emp_data['BusinessTravel_Travel_Frequently'] = 1 if new_travel == "Travel Frequently" else 0
            emp_data['BusinessTravel_Travel_Rarely'] = 1 if new_travel == "Travel Rarely" else 0
            
    else:
        # --- NEW HIRE BUILDER FORM ---
        st.markdown("### 📝 Configure Custom Employee Profile")
        st.caption("Fields left untouched automatically default to the company's healthy 'safe employee' averages.")
        
        # Start with a "safe" baseline to prevent the model from crashing on the other 35 missing columns
        baseline = assets["safe_averages"].copy()
        
        c_form1, c_form2, c_form3 = st.columns(3)
        with c_form1:
            baseline['Age'] = st.number_input("Age", 18, 65, 30)
            baseline['Annual_CTC_LPA'] = st.number_input("Starting CTC (LPA)", 1.0, 50.0, 12.0)
            ot_choice = st.radio("Requires OverTime?", ["No", "Yes"], horizontal=True)
            baseline['OverTime_Yes'] = 1.0 if ot_choice == "Yes" else 0.0
            
        with c_form2:
            baseline['DistanceFromHome'] = st.number_input("Commute Distance (km)", 1, 50, 5)
            baseline['YearsAtCompany'] = st.number_input("Years at Company (Tenure)", 0, 40, 0)
            travel_choice = st.selectbox("Expected Business Travel", ["Non-Travel", "Travel Rarely", "Travel Frequently"])
            baseline['BusinessTravel_Travel_Frequently'] = 1.0 if travel_choice == "Travel Frequently" else 0.0
            baseline['BusinessTravel_Travel_Rarely'] = 1.0 if travel_choice == "Travel Rarely" else 0.0
            
        with c_form3:
            raise_pct = st.slider("Test Future Retention Raise (%)", min_value=5, max_value=50, value=15) / 100.0
            baseline['JobSatisfaction'] = st.slider("Expected Job Satisfaction (1-4)", 1, 4, 3)
            baseline['EnvironmentSatisfaction'] = st.slider("Expected Env Satisfaction (1-4)", 1, 4, 3)
            
        # Convert the series back to a 1-row DataFrame matching the exact shape the model expects
        emp_data = pd.DataFrame([baseline], columns=x_test_raw.columns)
        
        # We need a dummy position for the economics calculator to use as an ID
        selected_position = 0
        df_for_calc = emp_data
    
    

    # Run Prediction for this specific employee
    emp_clean = assets["selector"].transform(emp_data)
    current_prob = assets["model"].predict_proba(emp_clean)[0, 1]
    
    # Run Financial Calculator
    metrics = analyze_retention(selected_position, current_prob, df_for_calc, raise_pct)

    # UI Rendering: Top Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Calculated Flight Risk", f"{current_prob*100:.1f}%")
    c2.metric("Expected Financial Exposure", f"{metrics['risk_adjusted_loss']:.2f} LPA")
    
    decision = metrics['recommended_action']
    if decision == "Give Raise":
        c3.success(f"Strategy: {decision}")
    elif decision == "Allow Exit":
        c3.error(f"Strategy: {decision}")
    else:
        c3.info(f"Strategy: {decision}")

    # ==========================================
    #  AI EXPLAINER
    # ==========================================
    st.markdown("### 🧠 AI Explainer: What is driving this risk?")
    deviations = []
    
    for feature in assets["kept_features"]:
        emp_val = emp_data[feature].values[0]
        safe_val = assets["safe_averages"][feature]
        
        # Calculate how far off this employee is from a normal "safe" employee
        diff = (emp_val - safe_val) / (safe_val + 1e-9)
        
        # Clean up the feature name to make it look professional (e.g., 'OverTime_Yes' -> 'OverTime')
        clean_feature = feature.replace('_Yes', '').replace('_', ' ')
        
        # Detect if it's a Yes/No categorical column (where the only possible values in the dataset are 0 and 1)
        is_category = set(x_test_raw[feature].dropna().unique()).issubset({0, 1})
        
        if is_category:
            emp_display = "Yes" if emp_val == 1 else "No"
            safe_display = f"{safe_val * 100:.0f}% of safe employees"
        else:
            # It's a standard number (like Job Satisfaction or Salary)
            emp_display = str(round(emp_val, 1))
            safe_display = str(round(safe_val, 1))
            
        deviations.append({
            "Risk Factors": clean_feature, 
            "Employee's Value": emp_display, 
            "Healthy Baseline (Avg)": safe_display, 
            "Abs_Dev": abs(diff) # We use this to sort, but hide it in the UI
        })
    
    # Create the dataframe, sort to find the top 5 biggest problems, then drop the sorting column
    dev_df = pd.DataFrame(deviations)
    dev_df = dev_df.sort_values(by='Abs_Dev', ascending=False).head(5)
    dev_df = dev_df.drop(columns=['Abs_Dev'])
    
    st.write("These are the top 5 areas where this employee deviates most negatively from the company's healthy baseline:")
    st.dataframe(dev_df, hide_index=True, use_container_width=True)


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