
import numpy as np
import scipy.stats as stats

def get_demographic_parity(y_pred, groups):
    unique_groups = np.unique(groups)
    rates = {}
    
    for g in unique_groups:
        group_mask = (groups == g)
        rates[g] = np.mean(y_pred[group_mask] == 1) if np.sum(group_mask) > 0 else 0.0
        
    val_list = [v for v in rates.values() if v > 0]
    if len(val_list) <= 1:
        return rates, 1.0
    parity_ratio = min(val_list) / max(val_list)
    return rates, parity_ratio

def run_chi_square(y_pred, groups):
    unique_groups = np.unique(groups)
    contingency_table = []
    
    for g in unique_groups:
        group_mask = (groups == g)
        approved = np.sum((y_pred == 1) & group_mask)
        rejected = np.sum((y_pred == 0) & group_mask)
        contingency_table.append([approved, rejected])
        
    # Check for zeros to prevent SciPy from throwing a ValueError
    contingency_matrix = np.array(contingency_table)
    if np.any(contingency_matrix == 0) or contingency_matrix.size == 0:
        return 1.0  # Safe fallback default (No statistically significant variance)
        
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    return p_value

def calculate_audit_score(y_pred, groups):
    _, parity_ratio = get_demographic_parity(y_pred, groups)
    p_value = run_chi_square(y_pred, groups)
    
    base_score = 100
    max_penalty = (1.0 - parity_ratio) * 50
    
    if p_value < 0.05:
        final_penalty = max_penalty
    else:
        final_penalty = max_penalty * 0.2
        
    return max(0, min(100, base_score - final_penalty))
