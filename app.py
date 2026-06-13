import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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
model_choice = st.sidebar.selectbox("Model to Pipeline", ["Random Forest Credit Classifier", "Baseline Classifier (Simulated)"])

# 2. Compute Pipeline Execution
df, raw_sensitive_attr, target = load_clean_data()
sensitive_attr = raw_sensitive_attr.apply(lambda x: 'Male' if str(x).strip() in ['A91', 'A93', 'A94'] else 'Female')

@st.cache_data
def train_production_random_forest():
    X = df.drop(columns=[df.columns[-1]]) 
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    y = LabelEncoder().fit_transform(target.astype(str))
    
    # Train a real Random Forest pipeline
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf.predict(X)

# Execute model configuration
if model_choice == "Random Forest Credit Classifier":
    predictions = train_production_random_forest()
else:
    # Non-compliant simulated baseline
    np.random.seed(42)
    predictions = np.where(sensitive_attr == 'Male', 
                           np.random.choice([0, 1], size=len(df), p=[0.22, 0.78]),
                           np.random.choice([0, 1], size=len(df), p=[0.70, 0.30]))

# 3. Calculate Audit Compliance Metrics
rates, ratio = get_demographic_parity(predictions, sensitive_attr)
p_val = run_chi_square(predictions, sensitive_attr)
score = calculate_audit_score(predictions, sensitive_attr)

# Calculate False Rejection Rate (FRR) Disparity assuming Target == 1 is 'Good Credit'
df_metrics = pd.DataFrame({'Actual': target, 'Predicted': predictions, 'Group': sensitive_attr})
frr_rates = {}
for group in ['Male', 'Female']:
    group_filter = df_metrics[df_metrics['Group'] == group]
    actual_goods = group_filter[group_filter['Actual'] == 1] # Real creditworthy individuals
    false_rejections = actual_goods[actual_goods['Predicted'] == 0]
    frr_rates[group] = len(false_rejections) / len(actual_goods) if len(actual_goods) > 0 else 0

frr_disparity = abs(frr_rates['Male'] - frr_rates['Female']) * 100

# --- THE AUDIT PANEL DISPLAY (The Reviewer's Layout) ---
st.write("### 📋 Audit Target Metadata")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Model Audited:** `{model_choice}`")
with col2:
    st.write(f"**Dataset Evaluated:** `{dataset_choice}`")

st.markdown("---")

st.write("### 📊 Quantitative Fairness Metrics")
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric(label="Demographic Parity Ratio", value=f"{ratio:.2f}", help="Legal threshold (4/5ths Rule) is 0.80")
m_col2.metric(label="Chi-Square p-value", value=f"{p_val:.4f}", help="p < 0.05 indicates statistically significant disparity")
m_col3.metric(label="False Rejection Disparity", value=f"{frr_disparity:.1f}%", help="The absolute gap in error rates across groups")

st.write("#### Group Breakdown Analysis:")
for group, rate in rates.items():
    st.write(f"- **{group} Selection Vector**: Approval Rate: `{rate*100:.1f}%` | False Rejection Rate: `{frr_rates[group]*100:.1f}%`")

st.markdown("---")

# 4. Definitive Regulatory Verdict Block
st.write("### ⚖️ Regulatory Audit Verdict")
if ratio < 0.80 or p_val < 0.05:
    st.error(f"❌ COMPLIANCE FAIL: Adverse impact detected via demographic disparity or significant variance (p = {p_val:.4f}). System requires structural model remediation.")
else:
    st.success(f"✅ COMPLIANCE PASS: No statistically significant systemic disparity detected (p = {p_val:.4f}). Vector variances fall within acceptable null-hypothesis boundaries.")

# Demote the arbitrary score to a minor visual footnote at the bottom
with st.expander("View System Score Footnote"):
    st.write(f"Custom Framework Alignment Rating (AAS): **{score:.1f}/100**")