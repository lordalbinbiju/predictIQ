"""Dedicated Model Evaluation and comparison dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.model_evaluation import (
    build_classification_report_df,
    build_model_comparison_df,
    plot_confusion_matrix_heatmap,
    plot_model_comparison_chart,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.model_manager import ModelManager
from utils.helpers import render_kpi_card, render_page_header


def render() -> None:
    render_page_header(
        "Model Evaluation & Comparison",
        "Compare trained classifiers across held-out test, cross-validation, and training splits.",
    )

    model_mgr: ModelManager = st.session_state["model_manager"]
    bundles = model_mgr.load_all_bundles()

    if not bundles:
        st.warning("No trained models found. Navigate to **Model Training** or **Overview** to train models first.")
        return

    with st.expander("Metric Guide: Understanding Churn Evaluation Metrics", expanded=False):
        st.markdown(
            """
            - **Recall (Sensitivity) — Priority Metric**: Out of all customers who actually churned, how many did the model catch?
              Missing a churning customer (False Negative) results in lost recurring revenue without retention intervention.
            - **Precision**: Out of all customers flagged as likely to churn, how many actually churned? High precision limits wasted retention offers.
            - **F1-Score**: Harmonic mean of Precision and Recall, balancing missed churners against false alarms.
            - **ROC-AUC**: Ranking quality across all probability thresholds (0.5 = random guessing, 1.0 = perfect separation).
            - **Accuracy**: Overall fraction of correct predictions — can be misleading on imbalanced churn datasets.
            """
        )

    # Best model selection based on configurable metric (validation split prevents test leakage in model selection!)
    sel_col1, sel_col2, sel_col3 = st.columns([1.2, 1.2, 1])
    with sel_col1:
        selection_metric = st.selectbox(
            "Primary Selection Metric",
            options=["recall", "roc_auc", "f1_score", "precision", "accuracy"],
            index=0,
            format_func=lambda m: {
                "recall": "Recall (Recommended for Churn)",
                "roc_auc": "ROC-AUC",
                "f1_score": "F1-Score",
                "precision": "Precision",
                "accuracy": "Accuracy",
            }[m],
        )
    with sel_col2:
        selection_split = st.selectbox(
            "Selection Evaluation Split",
            options=["validation", "test"],
            index=0,
            format_func=lambda s: "Validation (Stratified CV OOF — Recommended)" if s == "validation" else "Held-Out Test Set",
        )
    with sel_col3:
        best_bundle = model_mgr.select_best_model(metric=selection_metric, split=selection_split)
        st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
        if best_bundle and st.button(
            f"Activate Best ({best_bundle['algorithm']})",
            type="primary",
            use_container_width=True,
        ):
            model_mgr.set_active(best_bundle["model_version"])
            st.success(f"Activated `{best_bundle['model_version']}` based on {selection_split} {selection_metric}.")
            st.rerun()

    if best_bundle:
        bm = best_bundle["metrics_by_split"]["test"]
        vm = best_bundle["metrics_by_split"]["validation"]
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card(
                f"Best by {selection_metric.upper()}",
                best_bundle["algorithm"],
                best_bundle["model_version"],
            )
        with k2:
            render_kpi_card("Validation Recall / ROC-AUC", f"{vm['recall']:.3f} / {vm['roc_auc']:.3f}", "Out-of-fold CV")
        with k3:
            render_kpi_card("Test Recall / F1", f"{bm['recall']:.3f} / {bm['f1_score']:.3f}", "Held-out test split")
        with k4:
            render_kpi_card("Test Precision / Accuracy", f"{bm['precision']:.3f} / {bm['accuracy']:.3f}", "Held-out test split")

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    tab_test, tab_val, tab_train = st.tabs(
        ["Held-Out Test Comparison", "Validation (CV OOF) Comparison", "Training Fit Check"]
    )
    with tab_test:
        test_df = build_model_comparison_df(bundles, split="test")
        st.dataframe(test_df, use_container_width=True, hide_index=True)
        st.plotly_chart(plot_model_comparison_chart(test_df, "Test"), use_container_width=True)

    with tab_val:
        val_df = build_model_comparison_df(bundles, split="validation")
        st.dataframe(val_df, use_container_width=True, hide_index=True)
        st.plotly_chart(plot_model_comparison_chart(val_df, "Validation CV OOF"), use_container_width=True)

    with tab_train:
        train_df = build_model_comparison_df(bundles, split="train")
        st.dataframe(train_df, use_container_width=True, hide_index=True)
        st.caption("Compare Training scores against Validation/Test scores to diagnose overfitting.")

    c_roc, c_pr = st.columns(2)
    with c_roc:
        st.plotly_chart(plot_roc_curves(bundles), use_container_width=True)
    with c_pr:
        st.plotly_chart(plot_precision_recall_curves(bundles), use_container_width=True)

    st.subheader("Detailed Single-Model Diagnostics (Test Set)")
    version_options = [b["model_version"] for b in bundles]
    selected_ver = st.selectbox("Inspect Model Run", options=version_options)
    inspected = next(b for b in bundles if b["model_version"] == selected_ver)

    d1, d2 = st.columns([1.1, 1])
    with d1:
        report_df = build_classification_report_df(
            inspected["y_test"],
            inspected["y_test_proba"],
            threshold=0.5,
        )
        st.markdown("**Classification Report (Threshold = 0.50)**")
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        cv_s = inspected["cv_summary"]
        st.markdown("**Cross-Validation Stability Summary**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "CV Folds": cv_s["cv_folds"],
                        "CV ROC-AUC Mean": round(cv_s["cv_roc_auc_mean"], 4),
                        "CV ROC-AUC Std": round(cv_s["cv_roc_auc_std"], 4),
                        "CV Recall Mean": round(cv_s["cv_recall_mean"], 4),
                        "CV F1 Mean": round(cv_s["cv_f1_mean"], 4),
                    }
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with d2:
        cm = inspected["metrics_by_split"]["test"]["confusion_matrix"]
        st.plotly_chart(
            plot_confusion_matrix_heatmap(cm, f"Confusion Matrix — {inspected['algorithm']}"),
            use_container_width=True,
        )
