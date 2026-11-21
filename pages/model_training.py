"""Machine learning training configuration and execution page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.model_manager import ModelManager
from src.model_training import (
    XGBOOST_AVAILABLE,
    get_available_algorithms,
    get_default_hyperparameters,
    train_churn_model,
)
from src.preprocessing import prepare_modeling_data
from utils.helpers import render_kpi_card, render_page_header


def render() -> None:
    render_page_header(
        "Model Training Engine",
        "Configure stratified splits, class imbalance weighting, hyperparameters, and train leakage-free ML pipelines.",
    )

    raw_df = st.session_state["raw_df"]
    target_col = st.session_state["target_col"]
    dataset_name = st.session_state["dataset_name"]
    model_mgr: ModelManager = st.session_state["model_manager"]

    _, y, num_cols, cat_cols, _ = prepare_modeling_data(raw_df, target_col=target_col, drop_duplicates=True)
    neg_cnt = int((y == 0).sum())
    pos_cnt = int((y == 1).sum())
    imbalance_ratio = neg_cnt / max(pos_cnt, 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Modeling Rows", f"{len(y):,}", "After deduplication")
    with c2:
        render_kpi_card("Class Balance", f"{pos_cnt / len(y) * 100:.1f}% Churn", f"Ratio {imbalance_ratio:.2f}:1 (Retained:Churned)")
    with c3:
        render_kpi_card("Raw Features", f"{len(num_cols) + len(cat_cols)}", f"{len(num_cols)} numeric, {len(cat_cols)} categorical")
    with c4:
        render_kpi_card("XGBoost Support", "Enabled" if XGBOOST_AVAILABLE else "Unavailable", "Optional boosting backend")

    st.info(
        "Data Leakage Prevention: All imputation, StandardScaler normalization, and OneHotEncoder transformations "
        "are encapsulated in a Scikit-learn `Pipeline` and fitted exclusively on training folds. The held-out test "
        "split is never used for cross-validation or hyperparameter selection."
    )

    with st.expander("Global Split, Cross-Validation & Imbalance Settings", expanded=True):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            test_size = st.slider(
                "Test Set Fraction",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
                help="Stratified hold-out split reserved strictly for final evaluation.",
            )
        with s2:
            cv_folds = st.slider(
                "Stratified CV Folds",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
            )
        with s3:
            random_state = int(
                st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)
            )
        with s4:
            use_class_weight = st.checkbox(
                "Apply Class Weighting",
                value=True,
                help="Uses class_weight='balanced' (or scale_pos_weight for XGBoost) to counter churn class imbalance.",
            )

    available_algos = get_available_algorithms()
    selected_algo = st.selectbox("Select Classification Algorithm", options=available_algos)
    defaults = get_default_hyperparameters(selected_algo)

    st.subheader(f"Hyperparameter Configuration — {selected_algo}")
    custom_params: dict[str, object] = {}
    p_cols = st.columns(min(4, len(defaults)))
    for idx, (param_name, default_val) in enumerate(defaults.items()):
        with p_cols[idx % len(p_cols)]:
            if isinstance(default_val, float):
                custom_params[param_name] = st.number_input(
                    param_name,
                    min_value=0.001,
                    max_value=100.0,
                    value=float(default_val),
                    step=0.01 if default_val < 1 else 0.5,
                    format="%.3f",
                )
            else:
                custom_params[param_name] = int(
                    st.number_input(
                        param_name,
                        min_value=1,
                        max_value=5000,
                        value=int(default_val),
                        step=10 if default_val >= 50 else 1,
                    )
                )

    set_active_on_train = st.checkbox(
        "Set newly trained model as Active Model",
        value=True,
    )

    btn1, btn2 = st.columns(2)
    with btn1:
        train_one = st.button(f"Train {selected_algo}", type="primary", use_container_width=True)
    with btn2:
        train_all = st.button(
            "Train All Available Algorithms (Default Suite)",
            use_container_width=True,
        )

    if train_one:
        progress = st.progress(0.0, text="Starting training pipeline...")
        try:
            res = train_churn_model(
                df=raw_df,
                algorithm=selected_algo,
                target_col=target_col,
                test_size=test_size,
                random_state=random_state,
                cv_folds=cv_folds,
                use_class_weight=use_class_weight,
                hyperparameters=custom_params,
                progress_callback=lambda msg, pct: progress.progress(pct, text=msg),
            )
            model_mgr.register_training_result(
                res,
                dataset_name=dataset_name,
                set_active=set_active_on_train,
            )
            st.success(
                f"Trained **{res.algorithm}** (`{res.model_version}`). "
                f"CV Recall: {res.cv_summary['cv_recall_mean']:.3f} | "
                f"Test Recall: {res.metrics_by_split['test']['recall']:.3f} | "
                f"Test ROC-AUC: {res.metrics_by_split['test']['roc_auc']:.3f}"
            )
            _render_split_summary_table(res.metrics_by_split, res.cv_summary)
        except Exception as exc:
            st.error(f"Model training failed: {exc}")

    if train_all:
        progress = st.progress(0.0, text="Starting multi-algorithm benchmark...")
        results = []
        try:
            for idx, algo in enumerate(available_algos):
                base = idx / len(available_algos)
                res = train_churn_model(
                    df=raw_df,
                    algorithm=algo,
                    target_col=target_col,
                    test_size=test_size,
                    random_state=random_state,
                    cv_folds=cv_folds,
                    use_class_weight=use_class_weight,
                    progress_callback=lambda msg, p, b=base, a=algo: progress.progress(
                        min(1.0, b + p / len(available_algos)),
                        text=f"[{a}] {msg}",
                    ),
                )
                results.append(res)

            best = max(results, key=lambda r: r.metrics_by_split["validation"]["recall"])
            for res in results:
                model_mgr.register_training_result(
                    res,
                    dataset_name=dataset_name,
                    set_active=set_active_on_train and (res.model_version == best.model_version),
                )
            progress.progress(1.0, text="All algorithms trained and persisted!")
            st.success(
                f"Trained {len(results)} models. Best validation Recall: **{best.algorithm}** "
                f"({best.metrics_by_split['validation']['recall']:.3f})."
            )
        except Exception as exc:
            st.error(f"Benchmark training failed: {exc}")


def _render_split_summary_table(
    metrics_by_split: dict[str, dict[str, object]],
    cv_summary: dict[str, float],
) -> None:
    rows = []
    for split_name in ["train", "validation", "test"]:
        m = metrics_by_split[split_name]
        rows.append(
            {
                "Split": "Validation (Stratified CV OOF)" if split_name == "validation" else split_name.title(),
                "Accuracy": round(float(m["accuracy"]), 4),
                "Precision": round(float(m["precision"]), 4),
                "Recall": round(float(m["recall"]), 4),
                "F1-Score": round(float(m["f1_score"]), 4),
                "ROC-AUC": round(float(m["roc_auc"]), 4),
            }
        )
    st.subheader("Training, Validation & Test Split Performance")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Cross-Validation ({cv_summary['cv_folds']} folds) ROC-AUC: "
        f"{cv_summary['cv_roc_auc_mean']:.4f} +/- {cv_summary['cv_roc_auc_std']:.4f}"
    )
