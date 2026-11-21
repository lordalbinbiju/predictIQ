"""PredictIQ – Customer Churn Prediction & Explainable Analytics Platform."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="PredictIQ | Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from database.database import DatabaseManager
from pages import (
    about,
    business_insights,
    customer_predictions,
    data_management,
    explainable_ai,
    exploratory_analysis,
    model_evaluation,
    model_history,
    model_training,
    overview,
)
from src.data_loader import load_sample_dataset, prepare_dataset
from src.model_manager import ModelManager
from utils.config import APP_TAGLINE, APP_TITLE, NAVIGATION_ITEMS
from utils.helpers import load_css


def _init_session_state() -> None:
    """Initialize shared dataset, validation report, and database/model managers."""
    if "db" not in st.session_state:
        st.session_state["db"] = DatabaseManager()
    if "model_manager" not in st.session_state:
        st.session_state["model_manager"] = ModelManager(db=st.session_state["db"])
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Overview"

    if "raw_df" not in st.session_state:
        raw_df = load_sample_dataset()
        prepared_df, report = prepare_dataset(raw_df)
        st.session_state["raw_df"] = raw_df
        st.session_state["dataset_df"] = prepared_df
        st.session_state["validation_report"] = report
        st.session_state["dataset_name"] = "IBM Telco Customer Churn (Sample)"
        st.session_state["target_col"] = report.target_column or "Churn"


def main() -> None:
    load_css()
    _init_session_state()

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">{APP_TITLE}</div>
                <div class="sidebar-brand-tag">{APP_TAGLINE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            "Navigation",
            options=NAVIGATION_ITEMS,
            index=NAVIGATION_ITEMS.index(st.session_state["active_page"]),
            key="nav_radio",
        )
        if selected_page != st.session_state["active_page"]:
            st.session_state["active_page"] = selected_page

        st.markdown("---")
        model_mgr: ModelManager = st.session_state["model_manager"]
        active_bundle = model_mgr.load_model_bundle()
        dataset_name = st.session_state.get("dataset_name", "Sample Dataset")
        df = st.session_state["dataset_df"]

        st.caption("ACTIVE WORKSPACE")
        st.markdown(f"**Dataset:** {dataset_name} ({len(df):,} rows)")
        if active_bundle:
            test_rec = active_bundle["metrics_by_split"]["test"]["recall"]
            st.markdown(
                f"**Active Model:** {active_bundle['algorithm']}  \n"
                f"Test Recall: `{test_rec:.3f}`"
            )
        else:
            st.markdown("**Active Model:** Not trained yet")

    page_router = {
        "Overview": overview.render,
        "Data Management": data_management.render,
        "Exploratory Data Analysis": exploratory_analysis.render,
        "Model Training": model_training.render,
        "Model Evaluation": model_evaluation.render,
        "Customer Predictions": customer_predictions.render,
        "Explainable AI": explainable_ai.render,
        "Business Insights": business_insights.render,
        "Model History": model_history.render,
        "About": about.render,
    }

    page_fn = page_router.get(st.session_state["active_page"], overview.render)
    page_fn()


if __name__ == "__main__":
    main()
