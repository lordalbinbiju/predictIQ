"""Interactive Exploratory Data Analysis dashboard page."""

from __future__ import annotations

import streamlit as st

from src.eda import (
    compute_overview_stats,
    generate_eda_insights,
    plot_churn_by_category,
    plot_churn_by_monthly_charges,
    plot_churn_by_tenure,
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_monthly_charges_distribution,
    plot_tenure_distribution,
)
from utils.helpers import format_percent, render_kpi_card, render_page_header


def render() -> None:
    render_page_header(
        "Exploratory Data Analysis",
        "Investigate churn drivers, customer demographics, billing distributions, and feature correlations.",
    )

    df = st.session_state["dataset_df"]
    target_col = st.session_state["target_col"]

    stats = compute_overview_stats(df, target_col)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Total Customers", f"{stats['total_customers']:,}")
    with k2:
        render_kpi_card("Churned Customers", f"{stats['churned_customers']:,}")
    with k3:
        render_kpi_card("Retained Customers", f"{stats['retained_customers']:,}")
    with k4:
        render_kpi_card("Overall Churn Rate", format_percent(float(stats["churn_rate"])))

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # Automated data-driven insights
    insights = generate_eda_insights(df, target_col)
    st.subheader("Automated Statistical Insights")
    for item in insights:
        st.markdown(f'<div class="insight-item">{item}</div>', unsafe_allow_html=True)

    tab_cat, tab_num, tab_corr = st.tabs(
        ["Categorical Churn Breakdown", "Tenure & Charges Analysis", "Correlation Matrix"]
    )

    with tab_cat:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.plotly_chart(plot_churn_distribution(df, target_col), use_container_width=True)
        with r1c2:
            if "gender" in df.columns:
                st.plotly_chart(
                    plot_churn_by_category(df, "gender", target_col, "Churn by Gender"),
                    use_container_width=True,
                )

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            if "Contract" in df.columns:
                st.plotly_chart(
                    plot_churn_by_category(df, "Contract", target_col, "Churn by Contract Type"),
                    use_container_width=True,
                )
        with r2c2:
            if "PaymentMethod" in df.columns:
                st.plotly_chart(
                    plot_churn_by_category(df, "PaymentMethod", target_col, "Churn by Payment Method"),
                    use_container_width=True,
                )

        if "InternetService" in df.columns:
            st.plotly_chart(
                plot_churn_by_category(df, "InternetService", target_col, "Churn by Internet Service"),
                use_container_width=True,
            )

    with tab_num:
        if "tenure" in df.columns:
            st.plotly_chart(plot_churn_by_tenure(df, target_col), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if "MonthlyCharges" in df.columns:
                st.plotly_chart(
                    plot_churn_by_monthly_charges(df, target_col),
                    use_container_width=True,
                )
        with c2:
            if "MonthlyCharges" in df.columns:
                st.plotly_chart(
                    plot_monthly_charges_distribution(df, target_col),
                    use_container_width=True,
                )

        if "tenure" in df.columns:
            st.plotly_chart(plot_tenure_distribution(df, target_col), use_container_width=True)

    with tab_corr:
        st.plotly_chart(plot_correlation_heatmap(df, target_col), use_container_width=True)
        st.caption(
            "Correlation coefficients measure linear association between numerical attributes and binary churn (0/1). "
            "Tree-based and ensemble models also capture non-linear interactions explored in Explainable AI."
        )
