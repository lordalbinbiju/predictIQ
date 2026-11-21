"""SHAP Global and Local Explainable AI page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.explainability import (
    compute_global_shap,
    compute_individual_shap,
    generate_individual_shap_narrative,
    plot_global_shap_bar,
    plot_individual_contributions_plotly,
    plot_individual_shap_waterfall_matplotlib,
    plot_shap_beeswarm_matplotlib,
)
from src.model_manager import ModelManager
from src.preprocessing import prepare_modeling_data
from utils.helpers import render_page_header


def render() -> None:
    render_page_header(
        "Explainable AI (SHAP)",
        "Inspect global feature attributions and customer-level Shapley explanations grounded in model math.",
    )

    model_mgr: ModelManager = st.session_state["model_manager"]
    bundles = model_mgr.load_all_bundles()
    if not bundles:
        st.warning("No trained model available. Please train a model first to compute SHAP explanations.")
        return

    active_bundle = model_mgr.load_model_bundle()
    versions = [b["model_version"] for b in bundles]
    default_idx = versions.index(active_bundle["model_version"]) if active_bundle else 0

    col_sel, col_n = st.columns([1.6, 1])
    with col_sel:
        selected_ver = st.selectbox("Select Model for SHAP Explanation", options=versions, index=default_idx)
        bundle = next(b for b in bundles if b["model_version"] == selected_ver)
    with col_n:
        sample_size = st.slider(
            "Global SHAP Sample Size",
            min_value=50,
            max_value=300,
            value=150,
            step=25,
            help="Representative sample of records used to compute global SHAP summaries.",
        )

    raw_df = st.session_state["raw_df"]
    target_col = st.session_state["target_col"]
    X_all, _, _, _, _ = prepare_modeling_data(raw_df, target_col=target_col, drop_duplicates=True)

    tab_global, tab_local = st.tabs(["Global Feature Explainability", "Individual Customer Explainability"])

    with tab_global:
        with st.spinner("Computing global SHAP attributions..."):
            global_shap = compute_global_shap(bundle, X_all, max_rows=sample_size)

        st.plotly_chart(
            plot_global_shap_bar(global_shap["importance_df"], top_n=15),
            use_container_width=True,
        )

        c_bee, c_tbl = st.columns([1.25, 1])
        with c_bee:
            st.markdown("**SHAP Summary Beeswarm Plot**")
            fig_bee = plot_shap_beeswarm_matplotlib(global_shap, max_display=15)
            st.pyplot(fig_bee, use_container_width=True)
        with c_tbl:
            st.markdown("**Global Feature Contribution Table**")
            display_imp = global_shap["importance_df"][
                ["Feature", "Mean |SHAP|", "Mean Signed SHAP"]
            ].copy()
            display_imp["Mean |SHAP|"] = display_imp["Mean |SHAP|"].round(5)
            display_imp["Mean Signed SHAP"] = display_imp["Mean Signed SHAP"].round(5)
            st.dataframe(display_imp, use_container_width=True, hide_index=True, height=440)

    with tab_local:
        last_pred = st.session_state.get("last_predicted_customer")
        source_options = ["Select Customer from Dataset"]
        if last_pred is not None:
            source_options.insert(0, f"Most Recent Prediction ({last_pred['customer_id']})")

        choice = st.radio("Customer Source", options=source_options, horizontal=True)

        if choice.startswith("Most Recent Prediction") and last_pred is not None:
            customer_row = last_pred["row"]
            cust_label = last_pred["customer_id"]
        else:
            max_idx = min(len(X_all) - 1, 500)
            row_idx = st.number_input(
                f"Dataset Row Index (0 to {max_idx})",
                min_value=0,
                max_value=max_idx,
                value=0,
                step=1,
            )
            customer_row = X_all.iloc[[int(row_idx)]]
            cust_label = (
                str(raw_df.iloc[int(row_idx)]["customerID"])
                if "customerID" in raw_df.columns
                else f"Row #{int(row_idx)}"
            )

        with st.spinner("Computing local SHAP waterfall explanation..."):
            local_shap = compute_individual_shap(bundle, customer_row)
            narrative = generate_individual_shap_narrative(local_shap)

        st.markdown(f'<div class="insight-item"><strong>{cust_label}:</strong> {narrative["narrative"]}</div>', unsafe_allow_html=True)

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Factors Increasing Churn Probability**")
            inc_df = pd.DataFrame(narrative["increasing_factors"])
            if not inc_df.empty:
                st.dataframe(
                    inc_df[["Feature", "SHAP Contribution"]].round(4),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No positive churn drivers detected for this profile.")
        with p2:
            st.markdown("**Factors Decreasing Churn Probability**")
            dec_df = pd.DataFrame(narrative["decreasing_factors"])
            if not dec_df.empty:
                st.dataframe(
                    dec_df[["Feature", "SHAP Contribution"]].round(4),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No negative churn drivers detected for this profile.")

        st.plotly_chart(
            plot_individual_contributions_plotly(local_shap, top_n=12),
            use_container_width=True,
        )

        st.markdown("**SHAP Waterfall Plot**")
        fig_wf = plot_individual_shap_waterfall_matplotlib(local_shap, max_display=12)
        st.pyplot(fig_wf, use_container_width=True)
