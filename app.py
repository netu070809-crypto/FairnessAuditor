import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dataset_loader import load_clean_data
from audit_framework import get_demographic_parity, run_chi_square

# --- UI Layout Configurations & Branding ---
st.set_page_config(
    page_title="FairnessAuditor | Enterprise AI Compliance", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom minimal styling for text readability
st.markdown("""
    <style>
    .big-font { font-size:18px !important; color: #555555; }
    .interpretation-box { background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 5px solid #0068c9; margin-top: 15px; }
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
st.sidebar.caption("v2.3.0 • Framework Protected")

# --- Core Compute Pipeline ---
df, raw_sensitive_attr, target = load_clean_data()

# Prepare features and map sensitive attributes cleanly
X = df.drop(columns=[df.columns[-1]]) 
X_numeric = pd.get_dummies(X, drop_first=True)
X_numeric.columns = X_numeric.columns.astype(str)
y_numeric = target.apply(lambda x: 1 if str(x).strip() in ['1', 'good'] else 0)

# Robust Data Science Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test, attr_train, attr_test = train_test_split(
    X_numeric, y_numeric, raw_sensitive_attr, test_size=0.20, random_state=42
)

sensitive_attr_test = attr_test.apply(lambda x: 'Male' if str(x).strip() in ['A91', 'A93', 'A94'] else 'Female')

@st.cache_data
def train_and_evaluate_model(strategy):
    if strategy == "Random Forest Classifier (Ensemble)":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train.values, y_train.values)
        preds = clf.predict(X_test.values)
        probs = clf.predict_proba(X_test.values)[:, 1]
        
        acc = accuracy_score(y_test.values, preds)
        f1 = f1_score(y_test.values, preds)
        auc = roc_auc_score(y_test.values, probs)
        
        # Extract feature importances safely
        importances = clf.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(5)
        
        return preds, acc, f1, auc, feat_imp_df
        
    elif strategy == "Logistic Regression (Linear Baseline)":
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train.values, y_train.values)
        preds = clf.predict(X_test.values)
        probs = clf.predict_proba(X_test.values)[:, 1]
        
        acc = accuracy_score(y_test.values, preds)
        f1 = f1_score(y_test.values, preds)
        auc = roc_auc_score(y_test.values, probs)
        
        # Calculate pseudo-importance from absolute coefficients for linear strategy
        importances = np.abs(clf.coef_[0])
        feat_imp_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(5)
        
        return preds, acc, f1, auc, feat_imp_df
        
    else:
        np.random.seed(42)
        preds = np.where(sensitive_attr_test == 'Male', 
                        np.random.choice([0, 1], size=len(X_test), p=[0.22, 0.78]),
                        np.random.choice([0, 1], size=len(X_test), p=[0.70, 0.30]))
        
        # Create mock feature importance list for simulated model
        mock_df = pd.DataFrame({
            'Feature': ['Simulated Attribute Alpha', 'Simulated Attribute Beta', 'Noise Factor Vector'],
            'Importance': [0.65, 0.25, 0.10]
        })
        return preds, 0.685, 0.721, 0.642, mock_df

# Run pipeline execution
predictions, accuracy, f1, auc_score, feature_importance_df = train_and_evaluate_model(model_choice)

# --- Calculation Routing ---
rates, ratio = get_demographic_parity(predictions, sensitive_attr_test)
p_val = run_chi_square(predictions, sensitive_attr_test)

df_metrics = pd.DataFrame({'Actual': y_test.values, 'Predicted': predictions, 'Group': sensitive_attr_test.values})
frr_rates = {}
for group in ['Male', 'Female']:
    group_filter = df_metrics[df_metrics['Group'] == group]
    actual_goods = group_filter[group_filter['Actual'] == 1]
    false_rejections = actual_goods[actual_goods['Predicted'] == 0]
    frr_rates[group] = len(false_rejections) / len(actual_goods) if len(actual_goods) > 0 else 0
frr_disparity = abs(frr_rates['Male'] - frr_rates['Female']) * 100

# --- MAIN INTERFACE RENDERING ---

st.title("🛡️ Algorithmic Fairness Auditor")
st.markdown("<p class='big-font'>Independent statistical validation engine for regulatory compliance and AI risk mitigation.</p>", unsafe_allow_html=True)
st.markdown("---")

# Row 1: Metadata Badges
meta1, meta2, meta3 = st.columns(3)
with meta1:
    st.write(f"**AUDIT TARGET:** \n`{model_choice}`")
with meta2:
    st.write(f"**EVALUATION DATASET:** \n`{dataset_choice}` (Testing Partition)")
with meta3:
    status_signal = "⚠️ Disparity Detected" if (ratio < 0.80 or p_val < 0.05) else "✅ No Significant Disparity"
    st.write(f"**AUDIT SIGNAL:** \n**{status_signal}**")

st.markdown("---")

# Row 2: Operational Telemetry (UPGRADE: Integrated ROC-AUC Score column)
st.write("### 📊 Operational Telemetry")
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    r_delta = f"{ratio - 0.80:.2f} vs Threshold" if ratio >= 0.80 else f"{ratio - 0.80:.2f} Breach"
    st.metric(label="Demographic Parity Ratio", value=f"{ratio:.2f}", delta=r_delta, delta_color="normal" if ratio >= 0.80 else "inverse")
with m_col2:
    p_delta = "Significant Disparity" if p_val < 0.05 else "Statistically Stable"
    st.metric(label="Chi-Square p-value", value=f"{p_val:.4f}", delta=p_delta, delta_color="inverse" if p_val < 0.05 else "off")
with m_col3:
    st.metric(label="False Rejection Disparity", value=f"{frr_disparity:.1f}%")
with m_col4:
    st.metric(label="Pipeline Test Accuracy", value=f"{accuracy*100:.1f}%")
with m_col5:
    st.metric(label="ROC-AUC Performance Score", value=f"{auc_score:.3f}", help="Area Under the Receiver Operating Characteristic curve. Industry-standard for classification quality.")

st.markdown(" ")

# Row 3: Split Vector Data View vs UPGRADE: Feature Importance Analysis
col_left, col_right = st.columns([1, 1])

with col_left:
    st.write("#### Vector Selection Metrics")
    summary_data = {
        "Demographic Attribute": ["Female Group", "Male Group"],
        "Algorithmic Approval Rate": [f"{rates.get('Female', 0)*100:.1f}%", f"{rates.get('Male', 0)*100:.1f}%"],
        "False Rejection Rate (FRR)": [f"{frr_rates.get('Female', 0)*100:.1f}%", f"{frr_rates.get('Male', 0)*100:.1f}%"]
    }
    st.table(pd.DataFrame(summary_data))

with col_right:
    st.write("#### 🔍 Structural Feature Drivers (Top 5)")
    st.markdown("Identifies which structural fields exert the heaviest mathematical weight during inference operations.")
    st.dataframe(feature_importance_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Row 4: Callout Verdict Alerts
st.write("### ⚖️ Evaluation Analytics Summary")
if ratio < 0.80 or p_val < 0.05:
    st.error(
        f"**DISPARITY NOTICE REGISTERED** \n\n"
        f"The framework identified a statistically significant outcome disparity between protected demographic tracks "
        f"(p = {p_val:.4f}). The validation metrics fall below optimal baseline parameters, indicating that outcome variance "
        f"warrants secondary procedural fairness investigations and data-level remediation."
    )
else:
    st.success(
        f"**COMPLIANCE ASSESSMENT SEAMLESS** \n\n"
        f"No statistically significant demographic disparities or vector variances were observed across the unseen validation dataset "
        f"(p = {p_val:.4f}). The pipeline decisions display uniform mathematical distributions within expected limits."
    )

# UPGRADE: Grounded Scholarly Interpretation Section
st.markdown(
    """
    <div class='interpretation-box'>
        <strong>🔬 Methodological Interpretation & Causal Boundaries:</strong><br/>
        A low demographic parity ratio or a significant chi-square test result <em>does not automatically prove intentional discrimination</em> 
        by the algorithm. Observed disparities are frequently indicative of underlying non-uniformities or historical structural inequalities 
        deeply embedded within the source training dataset matrix itself. By cross-referencing the <strong>Top Predictive Feature Drivers</strong> 
        (displayed above) with outcome disparities, governance teams can evaluate whether the model is relying on proxy variables that 
        indirectly encode historical demographic discrepancies, allowing for targeted data-level remediation.
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown("---")

# Row 5: Action Button / Export Section
st.write("### 📥 Compliance Logging")
report_content = f"""ALGORITHMIC COMPLIANCE AUDIT REPORT
==========================================
Target Model: {model_choice}
Evaluation Dataset: {dataset_choice}
------------------------------------------
PERFORMANCE METRICS:
- Predictive Test Accuracy: {accuracy*100:.1f}%
- Validation ROC-AUC Score: {auc_score:.3f}

FAIRNESS & VALIDATION FINDINGS:
- Demographic Parity Ratio: {ratio:.2f}
- Pearson Chi-Square p-value: {p_val:.4f}
- False Rejection Rate Disparity: {frr_disparity:.1f}%

VERDICT SUMMARY:
{'ALERT: Statistically significant outcomes observed.' if (ratio < 0.80 or p_val < 0.05) else 'PASS: Disparities remain within expected limits.'}
==========================================
Report compiled via open-source Framework Auditor.
"""

st.download_button(
    label="Download Formal Audit Ledger",
    data=report_content,
    file_name=f"audit_ledger_{model_choice.lower().replace(' ', '_')}.txt",
    mime="text/plain",
    use_container_width=True
)