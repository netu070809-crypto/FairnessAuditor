# Algorithmic Fairness Auditor

A full-stack data science application designed to audit automated decision-making engines (such as credit lending algorithms) for systemic demographic and gender bias. 

## Live Application
Interact with the live, cloud-deployed dashboard here:(https://fairnessauditor-2kscypietjkjggzvdgqlsk.streamlit.app)

## The Problem
As machine learning models are increasingly deployed to automate high-stakes decisions (loans, hiring, criminal justice), they risk automating and magnifying human bias present in historical training data. This project provides compliance auditors with a mathematical framework to evaluate whether an algorithm unfairly excludes protected demographic groups.

## Methodology & Mathematical Framework
This tool utilizes the **UCI German Credit Dataset** to simulate model predictions and evaluate fairness across two primary statistical pillars:

1. **Demographic Parity Ratio (The Four-Fifths Rule):** Calculates the selection rate of a protected group against a baseline group. Following regulatory standards (like the US EEOC), a ratio below `0.80` indicates potential adverse impact.
2. **Chi-Square Independence Test ($\chi^2$):** Computes a $p$-value to determine if the variance in selection rates between groups is statistically significant or merely a result of random variation. A $p$-value $< 0.05$ flags the system for mandatory compliance remediation.

## Technologies Used
- **Backend:** Python 3, NumPy, SciPy (Statistical Functions)
- **Frontend/UI:** Streamlit Cloud Framework
- **Data Pipeline:** Pandas, UCI Machine Learning Repository
