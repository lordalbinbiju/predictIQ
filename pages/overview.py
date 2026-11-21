"""Overview landing page for PredictIQ."""

from __future__ import annotations

import streamlit as st

from src.eda import compute_overview_stats, plot_churn_by_category, plot_churn_distribution
from src.model_manager import ModelManager
from src.model_training import train_churn_model
from utils.config import APP_SUBTITLE, APP_TAGLINE
from utils.helpers import format_percent, render_kpi_card, render_page_header


def _navigate_to(page_name: str) -> None:
    st.session_state["active_page"] = page_name
    st.rerun()


def render() -> None:
    render_page_header(
        "PredictIQ — Enterprise Churn Analytics",
        f"{APP_TAGLINE} | {APP_SUBTITLE}",
    )

    df = st.session_state["dataset_df"]
    target_col = st.session_state["target_col"]
    dataset_name = st.session_state["dataset_name"]
    model_mgr: ModelManager = st.session_state["model_manager"]
    models_df = model_mgr.list_models()
    active_bundle = model_mgr.load_model_bundle()

    stats = compute_overview_stats(df, target_col)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Total Customers",
            f"{stats['total_customers']:,}",
            f"Source: {dataset_name}",
        )
    with c2:
        render_kpi_card(
            "Observed Churn Rate",
            format_percent(float(stats["churn_rate"])),
            f"{stats['churned_customers']:,} churned / {stats['retained_customers']:,} retained",
        )
    with c3:
        render_kpi_card(
            "Trained Models",
            f"{len(models_df)}",
            "Persisted pipelines in registry",
        )
    with c4:
        if active_bundle:
            test_m = active_bundle["metrics_by_split"]["test"]
            render_kpi_card(
                "Active Model (Test Recall)",
                f"{test_m['recall'] * 100:.1f}%",
                f"{active_bundle['algorithm']} | ROC-AUC {test_m['roc_auc']:.3f}",
            )
        else:
            render_kpi_card(
                "Active Model",
                "Not Trained",
                "Train models below to unlock scoring",
            )

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])
    with left:
        with st.container(border=True):
            st.subheader("Platform Capabilities")
            st.markdown(
                """
                **PredictIQ** transforms raw subscription and billing records into predictive churn risk scores,
                interpretable SHAP explanations, and transparent retention insights:

                - **Data Validation & Leakage-Free Pipelines**: Automated missing-value detection, numeric coercion,
                  duplicate removal, and Scikit-learn `ColumnTransformer` pipelines fitted strictly on training folds.
                - **Multi-Algorithm Benchmarking**: Compare Logistic Regression, Random Forest, Gradient Boosting,
                  and XGBoost with stratified K-fold cross-validation and held-out test evaluation.
                - **Explainable AI (SHAP)**: Global driver rankings and customer-level waterfall explanations
                  grounded in calculated Shapley values.
                - **Persistent Analytics Registry**: SQLite storage for datasets, training runs, evaluation splits,
                  and prediction audit logs.
                """
            )

    with right:
        with st.container(border=True):
            st.subheader("Quick-Start Demonstration")
            if active_bundle is None:
                st.info(
                    "No trained model is registered yet. Click below to train the core benchmark suite "
                    "(Logistic Regression, Random Forest, and Gradient Boosting) on the active dataset."
                )
                if st.button(
                    "Train Core Benchmark Models Now",
                    type="primary",
                    use_container_width=True,
                ):
                    progress = st.progress(0.0, text="Initializing benchmark suite...")
                    core_algos = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
                    trained_results = []
                    for idx, algo in enumerate(core_algos):
                        base_pct = idx / len(core_algos)
                        res = train_churn_model(
                            df=st.session_state["raw_df"],
                            algorithm=algo,
                            target_col=target_col,
                            progress_callback=lambda msg, p, b=base_pct: progress.progress(
                                min(1.0, b + p / len(core_algos)),
                                text=f"[{algo}] {msg}",
                            ),
                        )
                        trained_results.append(res)

                    best_res = max(
                        trained_results,
                        key=lambda r: r.metrics_by_split["validation"]["recall"],
                    )
                    for res in trained_results:
                        model_mgr.register_training_result(
                            res,
                            dataset_name=dataset_name,
                            set_active=(res.model_version == best_res.model_version),
                        )
                    progress.progress(1.0, text="All 3 core models trained and persisted!")
                    st.success(
                        f"Benchmark complete. Active model set to **{best_res.algorithm}** "
                        f"(Validation Recall: {best_res.metrics_by_split['validation']['recall']:.3f}, "
                        f"Test Recall: {best_res.metrics_by_split['test']['recall']:.3f})."
                    )
                    st.rerun()
            else:
                st.success(
                    f"Active model **{active_bundle['algorithm']}** (`{active_bundle['model_version']}`) "
                    "is ready for live scoring and SHAP explainability."
                )

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Explore EDA Dashboard", use_container_width=True):
                    _navigate_to("Exploratory Data Analysis")
                if st.button("Run Customer Predictions", use_container_width=True):
                    _navigate_to("Customer Predictions")
            with b2:
                if st.button("Open Model Evaluation", use_container_width=True):
                    _navigate_to("Model Evaluation")
                if st.button("Inspect SHAP Explainability", use_container_width=True):
                    _navigate_to("Explainable AI")

    st.subheader("Dataset Snapshot")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            plot_churn_distribution(df, target_col),
            use_container_width=True,
        )
    with col_b:
        if "Contract" in df.columns:
            st.plotly_chart(
                plot_churn_by_category(df, "Contract", target_col, "Churn Volume by Contract Type"),
                use_container_width=True,
            )
