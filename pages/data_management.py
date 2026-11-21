"""Data ingestion, validation, and data quality reporting page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database.database import DatabaseManager
from src.data_loader import load_sample_dataset, load_uploaded_csv, prepare_dataset
from utils.helpers import format_percent, render_kpi_card, render_page_header
from utils.validation import ValidationReport


def _register_dataset_in_db(db: DatabaseManager, name: str, source: str, report: ValidationReport) -> None:
    retained = report.target_distribution.get("Retained (0)", 0)
    churned = report.target_distribution.get("Churned (1)", 0)
    total = retained + churned
    churn_rate = churned / total if total > 0 else 0.0
    db.record_dataset(
        name=name,
        source=source,
        row_count=report.total_rows,
        col_count=report.total_columns,
        missing_count=report.missing_values_total,
        duplicate_count=report.duplicate_rows,
        target_column=report.target_column or "Unknown",
        churn_rate=churn_rate,
    )


def render() -> None:
    render_page_header(
        "Data Management & Quality Validation",
        "Ingest customer churn datasets, inspect schema health, and verify preprocessing readiness.",
    )

    db: DatabaseManager = st.session_state["db"]

    src_col, opt_col = st.columns([1.4, 1])
    with src_col:
        source_choice = st.radio(
            "Data Source",
            options=["Bundled IBM Telco Sample Dataset", "Upload Custom CSV File"],
            horizontal=True,
        )

    with opt_col:
        remove_dups = st.checkbox(
            "Remove duplicate records on load",
            value=False,
            help="Duplicate records are always removed prior to train-test splitting to prevent leakage.",
        )

    if source_choice == "Bundled IBM Telco Sample Dataset":
        if st.button("Reload Sample Dataset", type="secondary"):
            raw_df = load_sample_dataset()
            prepared, report = prepare_dataset(raw_df, drop_duplicates=remove_dups)
            st.session_state["raw_df"] = raw_df
            st.session_state["dataset_df"] = prepared
            st.session_state["validation_report"] = report
            st.session_state["dataset_name"] = "IBM Telco Customer Churn (Sample)"
            st.session_state["target_col"] = report.target_column or "Churn"
            _register_dataset_in_db(db, st.session_state["dataset_name"], "sample", report)
            st.success("Sample dataset reloaded and registered.")
    else:
        uploaded = st.file_uploader(
            "Upload Customer Churn CSV",
            type=["csv"],
            help="CSV must contain customer attributes and a binary churn target column (e.g. 'Churn').",
        )
        if uploaded is not None:
            try:
                raw_df = load_uploaded_csv(uploaded)
                prepared, report = prepare_dataset(raw_df, drop_duplicates=remove_dups)
                if not report.is_valid:
                    for err in report.errors:
                        st.error(err)
                else:
                    st.session_state["raw_df"] = raw_df
                    st.session_state["dataset_df"] = prepared
                    st.session_state["validation_report"] = report
                    st.session_state["dataset_name"] = uploaded.name
                    st.session_state["target_col"] = report.target_column or "Churn"
                    _register_dataset_in_db(db, uploaded.name, "upload", report)
                    st.success(f"Loaded and validated '{uploaded.name}' ({report.total_rows:,} rows).")
            except Exception as exc:
                st.error(f"Could not parse uploaded CSV: {exc}")

    df: pd.DataFrame = st.session_state["dataset_df"]
    report: ValidationReport = st.session_state["validation_report"]

    # Target column selector override
    candidate_targets = [c for c in df.columns if c != "_churn_binary"]
    current_target = st.session_state.get("target_col", report.target_column)
    if current_target in candidate_targets:
        selected_target = st.selectbox(
            "Confirmed Target Column",
            options=candidate_targets,
            index=candidate_targets.index(current_target),
            help="Binary churn label column used for EDA and supervised learning.",
        )
        if selected_target != current_target:
            prepared, new_report = prepare_dataset(
                st.session_state["raw_df"],
                target_col=selected_target,
                drop_duplicates=remove_dups,
            )
            if new_report.is_valid:
                st.session_state["dataset_df"] = prepared
                st.session_state["validation_report"] = new_report
                st.session_state["target_col"] = selected_target
                report = new_report
                df = prepared
            else:
                for err in new_report.errors:
                    st.error(err)

    for warn in report.warnings:
        st.warning(warn)

    st.subheader("Data Quality Report")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Total Rows", f"{report.total_rows:,}")
    with k2:
        render_kpi_card("Total Columns", f"{report.total_columns}")
    with k3:
        render_kpi_card("Missing Values", f"{report.missing_values_total:,}")
    with k4:
        render_kpi_card("Duplicate Rows", f"{report.duplicate_rows:,}")
    with k5:
        retained = report.target_distribution.get("Retained (0)", 0)
        churned = report.target_distribution.get("Churned (1)", 0)
        total_t = retained + churned
        rate = churned / total_t if total_t else 0.0
        render_kpi_card("Churn Rate", format_percent(rate), f"{churned:,} positive")

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Schema & Feature Classification", "Missing & Invalid Values", "Data Preview", "Dataset Registry"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Numerical Features ({len(report.numerical_columns)})**")
            st.write(report.numerical_columns)
            st.markdown(f"**Identifier Columns ({len(report.id_columns)})**")
            st.write(report.id_columns if report.id_columns else "None detected")
        with c2:
            st.markdown(f"**Categorical Features ({len(report.categorical_columns)})**")
            st.write(report.categorical_columns)
            st.markdown("**Target Class Distribution**")
            st.json(report.target_distribution)

        schema_df = pd.DataFrame(
            {
                "Column": list(report.column_dtypes.keys()),
                "Detected Dtype": list(report.column_dtypes.values()),
                "Role": [
                    "Target"
                    if c == report.target_column
                    else (
                        "Identifier (Excluded from ML)"
                        if c in report.id_columns
                        else ("Numerical Feature" if c in report.numerical_columns else "Categorical Feature")
                    )
                    for c in report.column_dtypes.keys()
                ],
                "Missing Count": [report.missing_by_column.get(c, 0) for c in report.column_dtypes.keys()],
            }
        )
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

    with tab2:
        if report.missing_by_column or report.invalid_values:
            if report.missing_by_column:
                st.markdown("**Columns with Missing Values (After Numeric String Coercion)**")
                miss_df = pd.DataFrame(
                    [
                        {
                            "Column": col,
                            "Missing Count": cnt,
                            "Missing %": round(cnt / max(report.total_rows, 1) * 100, 3),
                        }
                        for col, cnt in report.missing_by_column.items()
                    ]
                )
                st.dataframe(miss_df, use_container_width=True, hide_index=True)
            if report.invalid_values:
                st.markdown("**Coerced / Invalid Values Detected**")
                inv_df = pd.DataFrame(
                    [{"Check": k, "Affected Rows": v} for k, v in report.invalid_values.items()]
                )
                st.dataframe(inv_df, use_container_width=True, hide_index=True)
            st.info(
                "During model training, missing numerical values are imputed with training-fold medians "
                "and missing categorical values with training-fold modes inside `ColumnTransformer` to prevent data leakage."
            )
        else:
            st.success("Zero missing or invalid values detected in the active dataset.")

    with tab3:
        display_df = df.drop(columns=["_churn_binary"], errors="ignore")
        st.dataframe(display_df.head(100), use_container_width=True)

    with tab4:
        hist_df = db.get_datasets()
        if hist_df.empty:
            st.info("No dataset ingestion events recorded yet. Reload or upload a dataset to log metadata.")
        else:
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
