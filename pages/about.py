"""About PredictIQ — Architecture, Methodology, and Attribution page."""

from __future__ import annotations

import streamlit as st

from utils.config import APP_SUBTITLE, APP_TAGLINE, APP_TITLE
from utils.helpers import render_page_header


def render() -> None:
    render_page_header(
        f"About {APP_TITLE}",
        f"{APP_TAGLINE} — {APP_SUBTITLE}",
    )

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Engineering & ML Design Principles")
            st.markdown(
                """
                1. **Zero Data Leakage**: Numerical median imputation, StandardScaler normalization, and
                   OneHotEncoder category mapping live inside a unified Scikit-learn `Pipeline` fitted
                   strictly on training splits and CV folds.
                2. **Recall-Conscious Evaluation**: Retaining an at-risk subscriber is typically far less costly
                   than acquiring a replacement customer. PredictIQ prioritizes **Recall**, **PR-AUC**, and
                   **ROC-AUC** alongside configurable probability thresholds.
                3. **Model-Agnostic & Model-Specific Explainability**: SHAP `LinearExplainer` and
                   `TreeExplainer` surface both global churn drivers and customer-specific waterfall attributions.
                4. **Reproducible Persistence**: Trained pipelines are serialized with `joblib`, and experiment
                   runs, evaluation splits, dataset metadata, and prediction logs are persisted in SQLite.
                """
            )

    with c2:
        with st.container(border=True):
            st.subheader("Algorithm Selection Rationale")
            st.markdown(
                """
                - **Logistic Regression**: Transparent linear baseline estimating calibrated log-odds coefficients;
                  fast to train and directly compatible with `shap.LinearExplainer`.
                - **Random Forest Classifier**: Bagged ensemble of decision trees robust to non-linear feature
                  interactions and outliers, supporting native balanced class weights.
                - **Gradient Boosting / XGBoost**: Sequential boosting algorithms that correct residual errors
                  on hard-to-classify churners, typically yielding strong ranking performance (ROC-AUC).
                """
            )

    with st.container(border=True):
        st.subheader("Dataset Attribution & Limitations")
        st.markdown(
            """
            - **Sample Dataset**: IBM Sample Data Sets — *Telco Customer Churn* (`7,043` customer records,
              `21` attributes covering demographics, subscribed services, contract terms, and billing).
            - **Limitations**: Cross-sectional snapshot without longitudinal time-series usage logs; class
              imbalance (~26.5% churn) affects precision-recall trade-offs; rule-based retention recommendations
              indicate correlation-informed targeting rather than proven causal uplift without A/B testing.
            """
        )
