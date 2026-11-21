"""SQLite Model Registry & Training History comparison page."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from database.database import DatabaseManager
from src.model_manager import ModelManager
from utils.helpers import render_page_header


def render() -> None:
    render_page_header(
        "Model History & Experiment Registry",
        "Audit past training runs, inspect hyperparameters, compare metrics across splits, and manage active models.",
    )

    model_mgr: ModelManager = st.session_state["model_manager"]
    db: DatabaseManager = st.session_state["db"]
    runs_df = model_mgr.list_models()

    if runs_df.empty:
        st.info("No model runs recorded yet. Train models from **Model Training** or **Overview**.")
        return

    display_cols = [
        "model_version",
        "algorithm",
        "dataset_name",
        "cv_folds",
        "cv_roc_auc_mean",
        "cv_recall_mean",
        "test_recall",
        "test_precision",
        "test_f1",
        "test_roc_auc",
        "test_accuracy",
        "is_active",
        "trained_at",
    ]
    view_df = runs_df[display_cols].copy()
    view_df["is_active"] = view_df["is_active"].map({1: "Active", 0: ""})
    for num_col in [
        "cv_roc_auc_mean",
        "cv_recall_mean",
        "test_recall",
        "test_precision",
        "test_f1",
        "test_roc_auc",
        "test_accuracy",
    ]:
        view_df[num_col] = view_df[num_col].round(4)

    st.subheader("Registered Model Runs")
    st.dataframe(view_df, use_container_width=True, hide_index=True)

    st.subheader("Manage & Inspect Run Details")
    versions = runs_df["model_version"].tolist()
    selected_version = st.selectbox("Select Model Version", options=versions)
    selected_row = runs_df[runs_df["model_version"] == selected_version].iloc[0]

    a1, a2, a3 = st.columns([1, 1, 2])
    with a1:
        if st.button("Set as Active Model", type="primary", use_container_width=True):
            model_mgr.set_active(selected_version)
            st.success(f"`{selected_version}` is now the active scoring model.")
            st.rerun()
    with a2:
        if st.button("Delete Model Run", use_container_width=True):
            model_mgr.delete_model(selected_version)
            st.warning(f"Deleted `{selected_version}` from SQLite and disk.")
            st.rerun()

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Hyperparameters & Configuration**")
        st.json(
            {
                "algorithm": selected_row["algorithm"],
                "test_size": selected_row["test_size"],
                "random_state": int(selected_row["random_state"]),
                "class_weighting": bool(selected_row["use_class_weight"]),
                "hyperparameters": json.loads(selected_row["hyperparameters_json"]),
                "model_path": selected_row["model_path"],
            }
        )
    with d2:
        st.markdown("**Train / Validation / Test Metrics (SQLite)**")
        metrics_df = db.get_metrics_for_run(int(selected_row["id"]))
        if not metrics_df.empty:
            m_view = metrics_df[
                ["split", "accuracy", "precision_score", "recall", "f1_score", "roc_auc"]
            ].copy()
            for c in ["accuracy", "precision_score", "recall", "f1_score", "roc_auc"]:
                m_view[c] = m_view[c].round(4)
            st.dataframe(m_view, use_container_width=True, hide_index=True)
