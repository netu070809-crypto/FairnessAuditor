import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from dataset_loader import load_clean_data
from audit_framework import calculate_audit_score, get_demographic_parity, run_chi_square

# --- Page Layout & Professional Header ---
st.set_page_config(page_title="Algorithmic Fairness Auditor", layout="centered")
st.title("Algorithmic Fairness Auditor")
st.subheader("Statistical auditing framework for evaluating bias in automated decision systems")
st.markdown("---")

# 1. Pipeline Audit Configurations
st.sidebar.header("Audit Configuration")
dataset_choice = st.sidebar.selectbox("Target Dataset", ["UCI German Credit Dataset"])

# UPGRADE: Real Model Comparison Selection
model_choice = st.sidebar.selectbox("Model to Pipeline", [
    "Random Forest Classifier (Ensemble)", 
    "Logistic Regression (Linear Baseline)",
    "Simulated Non-Compliant Model"
])

# 2. Compute Pipeline Execution
df, raw_sensitive_attr, target = load_clean_data()
sensitive_attr = raw_sensitive_attr.apply(lambda x: 'Male' if str(x).strip() in ['A91', 'A93', 'A94'] else 'Female')

# Cache calculations to keep cloud pipelines fast
@st.cache_data
def train_selected_model(strategy):
    X = df.drop(columns=[df.columns[-1]]) 
    X_numeric = pd.get_dummies(X, drop_first=True)
    X_numeric.columns = X_numeric.columns.astype(str)
    y_numeric = target.apply(lambda x: 1 if str(x).strip() in ['1', 'good'] else 0)
    
    if strategy == "Random Forest Classifier (Ensemble)":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_numeric.values, y_numeric.values)
        return clf.predict(X_numeric.values)
    elif strategy == "Logistic Regression (Linear Baseline)":
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_numeric.values, y_numeric.values)
        return clf.predict(X_numeric.values)
    else:
        # Simulated baseline showing heavy historical disparity
        np.random.seed(42)
        return np.where(sensitive_attr == 'Male', 
                        np.random.choice([0, 1], size=len(df), p=[0.22, 0.78]),
                        np.random.choice([0, 1], size=len(df), p=[0.70, 0.30]))

# Execute dynamic model pipeline
predictions = train_selected_model(model_choice)

# 3. Calculate Compliance Metrics
rates, ratio = get_demographic_parity(predictions, sensitive_attr)
p_val = run_chi_square(predictions, sensitive_attr)
score = calculate_audit_score(predictions, sensitive_attr)

# Calculate False Rejection Rate (FRR) Disparity
binary_target = target.apply(lambda x: 1 if str(x).strip() in ['1', 'good'] else 0)
df_metrics = pd.DataFrame({'Actual': binary_target, 'Predicted': predictions, 'Group': sensitive_attr})
frr_rates = {}
for group in ['Male', 'Female']:
    group_filter = df_metrics[df_metrics['Group'] == group]
    actual_goods = group_filter[group_filter['Actual'] == 1]
    false_rejections = actual_goods[actual_goods['Predicted'] == 0]
    frr_rates[group] = len(false_rejections) / len(actual_goods) if len(actual_goods) > 0 else 0

frr_disparity = abs(frr_rates['Male'] - frr_rates['Female']) * 100

# --- THE AUDIT PANEL DISPLAY ---
st.write("### 📋 Audit Target Metadata")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Pipeline Audited:** `{model_choice}`")
with col2:
    st.write(f"**Dataset Evaluated:** `{dataset_choice}`")

st.markdown("---")

st.write("### 📊 Quantitative Fairness Metrics")
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric(label="Demographic Parity Ratio", value=f"{ratio:.2f}", help="Legal boundary is 0.80")
m_col2.metric(label="Chi-Square p-value", value=f"{p_val:.4f}", help="p < 0.05 indicates statistically significant disparity")
m_col3.metric(label="False Rejection Disparity", value=f"{frr_disparity:.1f}%")

st.write("#### Vector Selection Breakdown:")
for group, rate in rates.items():
    st.write(f"- **{group} Group**: Approval Rate: `{rate*100:.1f}%` | False Rejection Rate: `{frr_rates[group]*100:.1f}%`")

st.markdown("---")

# 4. Calibrated Defensible Audit Verdict
st.write("### ⚖️ Regulatory Audit Verdict")
if ratio < 0.80 or p_val < 0.05:
    # RECALIBRATED LANGUAGE: Swapped "proven bias" for "statistically significant disparity"
    st.error(f"⚠️ DISPARITY ALERT: Detected a statistically significant disparity in model outcomes (p = {p_val:.4f}), warranting further fairness investigation and structural remediation.")
else:
    st.success(f"✅ COMPLIANCE PASS: No statistically significant systemic disparity detected across demographic vectors (p = {p_val:.4f}).")

st.markdown("---")

# UPGRADE: Add Audit Report Export Panel
st.write("### 📥 Compliance Registry Document")
st.write("Compile and generate an official data evaluation summary for organizational risk portfolios.")

# Generate text summary payload for the download action
report_content = f"""ALGORITHMIC COMPLIANCE AUDIT REPORT
==========================================
Target Model: {model_choice}
Evaluation Dataset: {dataset_choice}
------------------------------------------
FINDINGS:
- Demographic Parity Ratio: {ratio:.2f}
- Pearson Chi-Square p-value: {p_val:.4f}
- False Rejection Rate Disparity: {frr_disparity:.1f}%

METRIC SUMMARY BREAKDOWN:
- Female Approval Rate: {rates.get('Female', 0)*100:.1f}%
- Male Approval Rate: {rates.get('Male', 0)*100:.1f}%

VERDICT SUMMARY:
{'DISPARITY ALERT: Statistically significant outcomes observed.' if (ratio < 0.80 or p_val < 0.05) else 'PASS: Disparities remain within null-hypothesis limits.'}
==========================================
Report compiled via open-source Framework Auditor.
"""

st.download_button(
    label="Generate Audit Report",
    data=report_content,
    file_name=f"audit_report_{model_choice.lower().replace(' ', '_')}.txt",
    mime="text/plain"
)

with st.expander("View Framework Footnote"):
    st.write(f"Custom System Alignment Rating: **{score:.1f}/100**")