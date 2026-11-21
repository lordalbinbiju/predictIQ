"""Individual and batch customer churn prediction page with SQLite persistence."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database.database import DatabaseManager
from src.data_loader import load_uploaded_csv
from src.model_manager import ModelManager
from src.prediction import predict_batch, predict_single_customer
from utils.config import PALETTE
from utils.helpers import render_kpi_card, render_page_header, risk_badge_html


def _build_gauge(probability: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Predicted Churn Probability", "font": {"size": 15, "color": PALETTE["navy"]}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": PALETTE["navy"]},
                "steps": [
                    {"range": [0, 35], "color": "#ECFDF5"},
                    {"range": [35, 65], "color": "#FFFBEB"},
                    {"range": [65, 100], "color": "#FEF2F2"},
                ],
                "threshold": {
                    "line": {"color": PALETTE["crimson"], "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render() -> None:
    render_page_header(
        "Customer Churn Predictions",
        "Score individual customer profiles or batch CSV files with configurable risk thresholds.",
    )

    model_mgr: ModelManager = st.session_state["model_manager"]
    db: DatabaseManager = st.session_state["db"]
    df_ref: pd.DataFrame = st.session_state["dataset_df"]
    bundles = model_mgr.load_all_bundles()

    if not bundles:
        st.warning("No trained model available. Please train a model in **Model Training** or **Overview** first.")
        return

    active_bundle = model_mgr.load_model_bundle()
    versions = [b["model_version"] for b in bundles]
    default_idx = versions.index(active_bundle["model_version"]) if active_bundle else 0

    cfg1, cfg2, cfg3, cfg4 = st.columns([1.4, 1, 1, 1])
    with cfg1:
        selected_version = st.selectbox("Scoring Model Version", options=versions, index=default_idx)
        bundle = next(b for b in bundles if b["model_version"] == selected_version)
    with cfg2:
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Customers with churn probability >= threshold are classified as Churn.",
        )
    with cfg3:
        low_max = st.slider("Low Risk Max Prob", min_value=0.10, max_value=0.50, value=0.35, step=0.05)
    with cfg4:
        high_min = st.slider(
            "High Risk Min Prob",
            min_value=max(low_max + 0.05, 0.40),
            max_value=0.90,
            value=max(0.65, low_max + 0.10),
            step=0.05,
        )

    st.caption(f"Scoring Pipeline: **{bundle['algorithm']}** (`{bundle['model_version']}`)")

    tab_ind, tab_batch, tab_hist = st.tabs(
        ["Individual Customer Prediction", "Batch CSV Prediction", "Prediction History"]
    )

    with tab_ind:
        with st.form("individual_prediction_form"):
            customer_id_input = st.text_input("Customer Identifier", value="CUST-DEMO-001")
            features: list[str] = bundle["feature_names"]
            num_cols = set(bundle.get("numerical_cols", []))

            customer_inputs: dict[str, object] = {}
            grid_cols = st.columns(3)
            for idx, feat in enumerate(features):
                with grid_cols[idx % 3]:
                    if feat in num_cols:
                        col_series = pd.to_numeric(df_ref[feat], errors="coerce").dropna() if feat in df_ref.columns else pd.Series([0.0])
                        unique_vals = sorted(col_series.unique().tolist())
                        if set(unique_vals).issubset({0, 1, 0.0, 1.0}) and len(unique_vals) <= 2:
                            customer_inputs[feat] = st.selectbox(
                                feat,
                                options=[0, 1],
                                index=0,
                                help="Binary indicator (0 = No, 1 = Yes)",
                            )
                        else:
                            min_v = float(col_series.min()) if not col_series.empty else 0.0
                            max_v = float(col_series.max() * 1.5) if not col_series.empty else 10000.0
                            med_v = float(col_series.median()) if not col_series.empty else 0.0
                            step_v = 1.0 if pd.api.types.is_integer_dtype(col_series) else 1.0
                            customer_inputs[feat] = st.number_input(
                                feat,
                                min_value=max(0.0, min_v),
                                max_value=max(max_v, 1000.0),
                                value=float(round(med_v, 2)),
                                step=step_v,
                            )
                    else:
                        options = (
                            sorted(df_ref[feat].dropna().astype(str).unique().tolist())
                            if feat in df_ref.columns
                            else ["Yes", "No"]
                        )
                        customer_inputs[feat] = st.selectbox(feat, options=options)

            submitted = st.form_submit_button("Predict Customer Churn Risk", type="primary")

        if submitted:
            try:
                res = predict_single_customer(
                    bundle=bundle,
                    customer_data=customer_inputs,
                    threshold=threshold,
                    low_max=low_max,
                    high_min=high_min,
                )
                db.save_prediction(
                    customer_id=customer_id_input or "CUST-UNKNOWN",
                    model_version=bundle["model_version"],
                    predicted_class=res["predicted_class"],
                    predicted_label=res["predicted_label"],
                    churn_probability=res["churn_probability"],
                    risk_category=res["risk_category"],
                    threshold=threshold,
                    input_features=customer_inputs,
                    source="individual",
                )
                st.session_state["last_predicted_customer"] = {
                    "customer_id": customer_id_input,
                    "row": res["aligned_row"],
                    "model_version": bundle["model_version"],
                }

                r1, r2 = st.columns([1.1, 1])
                with r1:
                    st.plotly_chart(
                        _build_gauge(res["churn_probability"], threshold),
                        use_container_width=True,
                    )
                with r2:
                    with st.container(border=True):
                        st.subheader(f"Prediction Result: {customer_id_input}")
                        st.markdown(f"**Predicted Outcome:** `{res['predicted_label']}` (Threshold: {threshold:.2f})")
                        st.markdown(f"**Churn Probability:** `{res['churn_probability'] * 100:.2f}%`")
                        st.markdown(
                            f"**Risk Tier:** {risk_badge_html(res['risk_category'])}",
                            unsafe_allow_html=True,
                        )
                        st.caption("Saved to SQLite prediction history. You can inspect this customer in Explainable AI.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with tab_batch:
        st.markdown("Upload a CSV file containing customer feature columns matching the trained model.")
        template_cols = [c for c in ["customerID"] + bundle["feature_names"]]
        available_template_cols = [c for c in template_cols if c in df_ref.columns]
        if available_template_cols:
            sample_batch_csv = df_ref[available_template_cols].head(25).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Sample Batch Input CSV (25 Customers)",
                data=sample_batch_csv,
                file_name="predictiq_sample_batch.csv",
                mime="text/csv",
            )

        batch_file = st.file_uploader("Upload Batch Customer CSV", type=["csv"], key="batch_csv_upload")
        if batch_file is not None:
            try:
                batch_df = load_uploaded_csv(batch_file)
                scored_df, warns = predict_batch(
                    bundle=bundle,
                    df=batch_df,
                    threshold=threshold,
                    low_max=low_max,
                    high_min=high_min,
                )
                for w in warns:
                    st.warning(w)

                # Persist batch summary and predictions in SQLite
                records_to_save = []
                feat_cols = bundle["feature_names"]
                for idx, row in scored_df.iterrows():
                    raw_feats = {
                        c: (None if pd.isna(batch_df.iloc[idx][c]) else batch_df.iloc[idx][c])
                        for c in feat_cols
                        if c in batch_df.columns
                    }
                    records_to_save.append(
                        {
                            "customer_id": row["Customer ID"],
                            "predicted_class": int(row["Predicted Class"]),
                            "predicted_label": row["Predicted Churn"],
                            "churn_probability": float(row["Churn Probability"]),
                            "risk_category": row["Risk Category"],
                            "input_features": raw_feats,
                        }
                    )

                db.save_batch_predictions(
                    batch_name=batch_file.name,
                    model_version=bundle["model_version"],
                    records=records_to_save,
                    threshold=threshold,
                )

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    render_kpi_card("Scored Customers", f"{len(scored_df):,}")
                with b2:
                    churn_cnt = int((scored_df["Predicted Class"] == 1).sum())
                    render_kpi_card("Predicted Churn", f"{churn_cnt:,}", f"{churn_cnt / len(scored_df) * 100:.1f}%")
                with b3:
                    high_cnt = int((scored_df["Risk Category"] == "High Risk").sum())
                    render_kpi_card("High Risk Customers", f"{high_cnt:,}")
                with b4:
                    render_kpi_card("Avg Churn Prob", f"{scored_df['Churn Probability'].mean() * 100:.1f}%")

                st.dataframe(scored_df, use_container_width=True, hide_index=True)

                out_csv = scored_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Prediction Results CSV",
                    data=out_csv,
                    file_name=f"predictiq_predictions_{bundle['model_version']}.csv",
                    mime="text/csv",
                    type="primary",
                )
            except Exception as exc:
                st.error(f"Batch scoring failed: {exc}")

    with tab_hist:
        st.subheader("Recent Individual & Batch Predictions (SQLite)")
        hist_df = db.get_prediction_history(limit=200)
        if hist_df.empty:
            st.info("No predictions logged yet.")
        else:
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

        batch_hist = db.get_batch_history(limit=50)
        if not batch_hist.empty:
            st.subheader("Batch Scoring Audit Log")
            st.dataframe(batch_hist, use_container_width=True, hide_index=True)
