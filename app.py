import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from dataset_loader import load_clean_data
from audit_framework import calculate_audit_score, get_demographic_parity, run_chi_square

# --- UI Layout Configurations & Branding ---
st.set_page_config(
    page_title="FairnessAuditor | Enterprise AI Compliance", 
    layout="wide", # Widens the page to look like a modern analytics desktop app
    initial_sidebar_state="expanded"
)

# Custom minimal styling for text readability
st.markdown("""
    <style>
    .big-font { font-size:18px !important; color: #555555; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #6c757d; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/fluency/96/shield-with-blockchain.png", width=60)
st.sidebar.title("Compliance Registry")
st.sidebar.markdown("Configure and execute real-time statistical audit pipelines on machine learning models.")
st.sidebar.markdown("---")

dataset_choice = st.sidebar.selectbox("Target Dataset Matrix", ["UCI German Credit Dataset"])
model_choice = st.sidebar.selectbox("Active Pipeline Strategy", [
    "Random Forest Classifier (Ensemble)", 
    "Logistic Regression (Linear Baseline)",
    "Simulated Non-Compliant Model"
])

st.sidebar.markdown("---")
st.sidebar.caption("v2.1.0-Alpha • Framework Protected")

# --- Core Compute Pipeline ---
df, raw_sensitive_attr, target = load_clean_data()
sensitive_attr = raw_sensitive_attr.apply(lambda x: 'Male' if str(x).strip() in ['A91', 'A93', 'A94'] else 'Female')

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
        np.random.seed(42)
        return np.where(sensitive_attr == 'Male', 
                        np.random.choice([0, 1], size=len(df), p=[0.22, 0.78]),
                        np.random.choice([0, 1], size=len(df), p=[0.70, 0.30]))

predictions = train_selected_model(model_choice)

# --- Calculation Routing ---
rates, ratio = get_demographic_parity(predictions, sensitive_attr)
p_val = run_chi_square(predictions, sensitive_attr)
score = calculate_audit_score(predictions, sensitive_attr)

binary_target = target.apply(lambda x: 1 if str(x).strip() in ['1', 'good'] else 0)
df_metrics = pd.DataFrame({'Actual': binary_target, 'Predicted': predictions, 'Group': sensitive_attr})
frr_rates = {}
for group in ['Male', 'Female']:
    group_filter = df_metrics[df_metrics['Group'] == group]
    actual_goods = group_filter[group_filter['Actual'] == 1]
    false_rejections = actual_goods[actual_goods['Predicted'] == 0]
    frr_rates[group] = len(false_rejections) / len(actual_goods) if len(actual_goods) > 0 else 0
frr_disparity = abs(frr_rates['Male'] - frr_rates['Female']) * 100

# --- MAIN INTERFACE RENDERING ---

# Main Dashboard Header
st.title("🛡️ Algorithmic Fairness Auditor")
st.markdown("<p class='big-font'>Independent statistical validation engine for regulatory compliance and AI risk mitigation.</p>", unsafe_allow_html=True)
st.markdown("---")

# Row 1: Metadata Badges
meta1, meta2, meta3 = st.columns(3)
with meta1:
    st.markdown(f"**AUDIT TARGET:** \n`{model_choice}`")
with meta2:
    st.markdown(f"**EVALUATION DATASET:** \n`{dataset_choice}`")
with meta3:
    status_color = "🔴 FAILED" if (ratio < 0.80 or p_val < 0.05) else "🟢 PASSED"
    st.markdown(f"**REGULATORY STATUS:** \n**{status_color}**")

st.markdown("---")

# Row 2: Modern Clean Metric Cards
st.write("### 📊 Compliance Telemetry")
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    r_delta = f"{ratio - 0.80:.2f} vs Threshold" if ratio >= 0.80 else f"{ratio - 0.80:.2f} Breach"
    st.metric(label="Demographic Parity Ratio", value=f"{ratio:.2f}", delta=r_delta, delta_color="normal" if ratio >= 0.80 else "inverse")