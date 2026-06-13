import streamlit as st
import numpy as np
from dataset_loader import load_clean_data
from audit_framework import calculate_audit_score, get_demographic_parity, run_chi_square

st.title("Algorithmic Fairness Auditor")
st.write("Auditing decision engines for systemic demographic exclusion.")

# 1. Load data
df, raw_sensitive_attr, target = load_clean_data()

# 2. Dynamic Mapping: Check if data is using UCI codes (A91, A92) or numeric formats (1, 2)
sample_val = str(raw_sensitive_attr.iloc[0]).strip()

if sample_val.startswith('A'):
    # UCI format
    sensitive_attr = raw_sensitive_attr.apply(lambda x: 'Male' if str(x).strip() in ['A91', 'A93', 'A94'] else 'Female')
else:
    # Numeric or alternate format (e.g., 1 for male, 2 for female)
    # Let's cleanly split them based on the unique values present
    unique_vals = raw_sensitive_attr.unique()
    sensitive_attr = raw_sensitive_attr.apply(lambda x: 'Group A' if x == unique_vals[0] else 'Group B')

# 3. Interactive Dropdown Menu
model_type = st.selectbox("Select Model to Audit", ["Baseline Classifier", "Complex Neural Net (Simulated)"])

# 4. Run simulations across BOTH groups dynamically
np.random.seed(42)
groups = sensitive_attr.unique()

if model_type == "Baseline Classifier":
    # Simulate an intentionally heavily biased engine
    simulated_preds = np.where((sensitive_attr == groups[0]) & (np.random.rand(len(df)) > 0.20), 1, 0)
    # Give the second group a significantly lower chance
    mask_group2 = (sensitive_attr == groups[1])
    simulated_preds[mask_group2] = np.where(np.random.rand(sum(mask_group2)) > 0.75, 1, 0)
else:
    # Simulate a balanced baseline model distribution
    simulated_preds = np.where(np.random.rand(len(df)) > 0.45, 1, 0)

# 5. Run the calculations
score = calculate_audit_score(simulated_preds, sensitive_attr)
rates, ratio = get_demographic_parity(simulated_preds, sensitive_attr)
p_val = run_chi_square(simulated_preds, sensitive_attr)

# 6. Render results to your screen
st.metric(label="Custom Algorithmic Audit Score (AAS)", value=f"{score:.1f} / 100")

st.write("### Demographic Selection Rates:")
for group, rate in rates.items():
    st.write(f"- **{group}**: {rate*100:.1f}% approval rate")

st.write(f"- **Demographic Parity Ratio**: {ratio:.2f}")

if p_val < 0.05:
    st.error(f"🚨 ALERT: Disparity is statistically significant (p = {p_val:.4f}). This algorithm requires compliance remediation.")
else:
    st.success(f"✅ PASS: Disparity is within acceptable baseline parameters (p = {p_val:.4f}).")