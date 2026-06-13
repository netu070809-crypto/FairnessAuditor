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
""", unsafe_base64=True)

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
    # Color metrics conditionally based on regulatory threshold passes
    r_delta = f"{ratio - 0.80:.2f} vs Threshold" if ratio >= 0.80 else f"{ratio - 0.80:.2f} Breach"
    st.metric(label="Demographic Parity Ratio", value=f"{ratio:.2f}", delta=r_delta, delta_color="normal" if ratio >= 0.80 else "inverse")
with m_col2:
    p_delta = "Statistically Significant" if p_val < 0.05 else "Statistically Stable"
    st.metric(label="Chi-Square p-value", value=f"{p_val:.4f}", delta=p_delta, delta_color="inverse" if p_val < 0.05 else "off")
with m_col3:
    st.metric(label="False Rejection Disparity", value=f"{frr_disparity:.1f}%", delta="Absolute Disparity Gap", delta_color="off")

st.markdown(" ")

# Row 3: Split Vector Data Frame View
col_left, col_right = st.columns([2, 1])

with col_left:
    st.write("#### Vector Selection Metrics")
    # Present data cleanly inside a beautifully styled table instead of a plain text list
    summary_data = {
        "Demographic Attribute": ["Female Group", "Male Group"],
        "Algorithmic Approval Rate": [f"{rates.get('Female', 0)*100:.1f}%", f"{rates.get('Male', 0)*100:.1f}%"],
        "False Rejection Rate (FRR)": [f"{frr_rates.get('Female', 0)*100:.1f}%", f"{frr_rates.get('Male', 0)*100:.1f}%"]
    }
    st.table(pd.DataFrame(summary_data))

with col_right:
    st.write("#### Audit Score")
    st.circular_progress(value=score/100, text=f"{score:.1f}/100")

st.markdown("---")

# Row 4: Callout Verdict Alerts
st.write("### ⚖️ Regulatory Evaluation Verdict")
if ratio < 0.80 or p_val < 0.05:
    st.error(
        f"**COMPLIANCE INFRACTION DETECTED** \n\n"
        f"The framework identified a statistically significant outcome disparity between protected demographic tracks "
        f"($p = {p_val:.4f}$). The target system fails automated validation baselines and requires "
        f"structural algorithm mitigation before live production deployment."
    )
else:
    st.success(
        f"**COMPLIANCE VALIDATION SUCCESS** \n\n"
        f"No statistically significant demographic disparities or vector variances were detected across the decision arrays "
        f"($p = {p_val:.4f}$). The pipeline outputs conform to the legal boundaries defined by the 4/5ths statutory rule."
    )

st.markdown("---")

# Row 5: Action Button / Export Section
st.write("### 📥 Compliance Logging")
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
{'FAIL: Statistically significant outcomes observed.' if (ratio < 0.80 or p_val < 0.05) else 'PASS: Disparities remain within null-hypothesis limits.'}
==========================================
Report compiled via open-source Framework Auditor.
"""

st.download_button(
    label="Download Formal Audit Ledger",
    data=report_content,
    file_name=f"audit_ledger_{model_choice.lower().replace(' ', '_')}.txt",
    mime="text/plain",
    use_container_width=True # Extends the button fully to look like a premium portal feature
)