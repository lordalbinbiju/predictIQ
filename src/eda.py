"""Exploratory Data Analysis calculations, Plotly figures, and dynamic insights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.config import PALETTE
from utils.validation import coerce_numeric_strings, normalize_target_series

CHURN_COLOR_MAP = {
    "Retained": PALETTE["primary_blue"],
    "Churned": PALETTE["crimson"],
}


def _prepare_eda_df(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    cleaned, _ = coerce_numeric_strings(df)
    if "_churn_binary" not in cleaned.columns:
        cleaned["_churn_binary"] = normalize_target_series(cleaned[target_col])
    cleaned["_ChurnLabel"] = np.where(
        cleaned["_churn_binary"] == 1, "Churned", "Retained"
    )
    return cleaned


def compute_overview_stats(df: pd.DataFrame, target_col: str) -> dict[str, float | int]:
    """Compute core customer overview KPIs."""
    eda_df = _prepare_eda_df(df, target_col)
    total = int(len(eda_df))
    churned = int((eda_df["_churn_binary"] == 1).sum())
    retained = int((eda_df["_churn_binary"] == 0).sum())
    churn_rate = churned / total if total > 0 else 0.0
    return {
        "total_customers": total,
        "churned_customers": churned,
        "retained_customers": retained,
        "churn_rate": churn_rate,
    }


def _apply_enterprise_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PALETTE["navy"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color=PALETTE["slate_900"], size=12),
        margin=dict(l=40, r=25, t=52, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            title_text="",
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor=PALETTE["slate_200"])
    fig.update_yaxes(gridcolor="#F1F5F9", linecolor=PALETTE["slate_200"])
    return fig


def plot_churn_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Bar chart of Retained vs Churned customers with percentages."""
    eda_df = _prepare_eda_df(df, target_col)
    counts = (
        eda_df["_ChurnLabel"]
        .value_counts()
        .reindex(["Retained", "Churned"], fill_value=0)
        .reset_index()
    )
    counts.columns = ["Status", "Customers"]
    total = counts["Customers"].sum()
    counts["Share"] = counts["Customers"].apply(
        lambda c: f"{c:,} ({(c / total * 100):.1f}%)" if total else "0 (0.0%)"
    )

    fig = px.bar(
        counts,
        x="Status",
        y="Customers",
        color="Status",
        text="Share",
        color_discrete_map=CHURN_COLOR_MAP,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Number of Customers", range=[0, counts["Customers"].max() * 1.18])
    fig.update_xaxes(title_text="Customer Status")
    return _apply_enterprise_layout(fig, "Overall Customer Churn Distribution")


def plot_churn_by_category(
    df: pd.DataFrame,
    category_col: str,
    target_col: str,
    title: str | None = None,
) -> go.Figure:
    """Grouped bar chart showing churn rate and volume across a categorical feature."""
    eda_df = _prepare_eda_df(df, target_col)
    grouped = (
        eda_df.groupby([category_col, "_ChurnLabel"], dropna=False)
        .size()
        .reset_index(name="Customers")
    )
    totals = grouped.groupby(category_col)["Customers"].transform("sum")
    grouped["RatePct"] = np.where(totals > 0, grouped["Customers"] / totals * 100, 0.0)
    grouped["Label"] = grouped.apply(
        lambda r: f"{int(r['Customers']):,} ({r['RatePct']:.1f}%)", axis=1
    )

    fig = px.bar(
        grouped,
        x=category_col,
        y="Customers",
        color="_ChurnLabel",
        barmode="group",
        color_discrete_map=CHURN_COLOR_MAP,
        hover_data={"RatePct": ":.1f", "Customers": ":,"},
    )
    fig.update_yaxes(title_text="Customers")
    fig.update_xaxes(title_text=category_col)
    return _apply_enterprise_layout(fig, title or f"Customer Churn by {category_col}")


def plot_churn_by_tenure(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Line/bar chart showing churn rate across customer tenure buckets."""
    eda_df = _prepare_eda_df(df, target_col)
    bins = [-1, 6, 12, 24, 36, 48, 60, 120]
    labels = ["0-6m", "7-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61m+"]
    eda_df["TenureBucket"] = pd.cut(eda_df["tenure"], bins=bins, labels=labels)
    summary = (
        eda_df.groupby("TenureBucket", observed=False)
        .agg(
            Total=("_churn_binary", "count"),
            Churned=("_churn_binary", "sum"),
        )
        .reset_index()
    )
    summary["ChurnRatePct"] = np.where(
        summary["Total"] > 0, summary["Churned"] / summary["Total"] * 100, 0.0
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["TenureBucket"].astype(str),
            y=summary["Total"],
            name="Total Customers",
            marker_color="#CBD5E1",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary["TenureBucket"].astype(str),
            y=summary["ChurnRatePct"],
            name="Churn Rate (%)",
            mode="lines+markers+text",
            text=[f"{v:.1f}%" for v in summary["ChurnRatePct"]],
            textposition="top center",
            line=dict(color=PALETTE["crimson"], width=3),
            marker=dict(size=8),
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Customer Count", showgrid=True, gridcolor="#F1F5F9"),
        yaxis2=dict(
            title="Churn Rate (%)",
            overlaying="y",
            side="right",
            range=[0, max(summary["ChurnRatePct"].max() * 1.25, 10)],
            showgrid=False,
        ),
        xaxis=dict(title="Customer Tenure Band"),
    )
    return _apply_enterprise_layout(fig, "Churn Rate Trend by Customer Tenure")


def plot_churn_by_monthly_charges(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Box plot comparing MonthlyCharges across Retained and Churned segments."""
    eda_df = _prepare_eda_df(df, target_col)
    fig = px.box(
        eda_df,
        x="_ChurnLabel",
        y="MonthlyCharges",
        color="_ChurnLabel",
        color_discrete_map=CHURN_COLOR_MAP,
        points=False,
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Customer Status")
    fig.update_yaxes(title_text="Monthly Charges ($)")
    return _apply_enterprise_layout(fig, "Monthly Charges Comparison by Churn Status")


def plot_monthly_charges_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Overlaid histogram of MonthlyCharges split by churn status."""
    eda_df = _prepare_eda_df(df, target_col)
    fig = px.histogram(
        eda_df,
        x="MonthlyCharges",
        color="_ChurnLabel",
        nbins=30,
        barmode="overlay",
        opacity=0.68,
        color_discrete_map=CHURN_COLOR_MAP,
    )
    fig.update_xaxes(title_text="Monthly Charges ($)")
    fig.update_yaxes(title_text="Customer Count")
    return _apply_enterprise_layout(fig, "Monthly Charges Distribution")


def plot_tenure_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Overlaid histogram of customer tenure split by churn status."""
    eda_df = _prepare_eda_df(df, target_col)
    fig = px.histogram(
        eda_df,
        x="tenure",
        color="_ChurnLabel",
        nbins=30,
        barmode="overlay",
        opacity=0.68,
        color_discrete_map=CHURN_COLOR_MAP,
    )
    fig.update_xaxes(title_text="Tenure (Months)")
    fig.update_yaxes(title_text="Customer Count")
    return _apply_enterprise_layout(fig, "Customer Tenure Distribution")


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Annotated correlation matrix heatmap for numerical features + binary churn target."""
    eda_df = _prepare_eda_df(df, target_col)
    num_df = eda_df.select_dtypes(include=[np.number]).copy()
    if "_churn_binary" in num_df.columns:
        num_df = num_df.rename(columns={"_churn_binary": "Churn (Binary)"})
    corr = num_df.corr(numeric_only=True).round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    return _apply_enterprise_layout(fig, "Numerical Feature Correlation Matrix")


def generate_eda_insights(df: pd.DataFrame, target_col: str) -> list[str]:
    """Derive data-driven analytical observations directly from the active dataset."""
    eda_df = _prepare_eda_df(df, target_col)
    insights: list[str] = []

    total = len(eda_df)
    churn_rate = float(eda_df["_churn_binary"].mean()) if total else 0.0
    churned_count = int(eda_df["_churn_binary"].sum())
    retained_count = total - churned_count
    if churned_count > 0:
        ratio = retained_count / churned_count
        insights.append(
            f"Overall churn rate is {churn_rate * 100:.1f}% ({churned_count:,} churned vs. {retained_count:,} retained; "
            f"class imbalance ratio {ratio:.2f}:1)."
        )

    if "Contract" in eda_df.columns:
        contract_rates = (
            eda_df.groupby("Contract")["_churn_binary"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )
        if len(contract_rates) >= 2:
            top_c = contract_rates.index[0]
            top_r = contract_rates.iloc[0]["mean"] * 100
            low_c = contract_rates.index[-1]
            low_r = contract_rates.iloc[-1]["mean"] * 100
            insights.append(
                f"Customers on '{top_c}' contracts exhibit the highest churn rate at {top_r:.1f}%, "
                f"compared to {low_r:.1f}% for '{low_c}' contracts."
            )

    if "tenure" in eda_df.columns:
        churn_tenure = eda_df.loc[eda_df["_churn_binary"] == 1, "tenure"].median()
        retain_tenure = eda_df.loc[eda_df["_churn_binary"] == 0, "tenure"].median()
        early_rate = eda_df.loc[eda_df["tenure"] <= 12, "_churn_binary"].mean() * 100
        insights.append(
            f"Median tenure for churned customers is {churn_tenure:.0f} months versus {retain_tenure:.0f} months for retained customers; "
            f"customers within their first 12 months show a {early_rate:.1f}% churn rate."
        )

    if "MonthlyCharges" in eda_df.columns:
        churn_mc = eda_df.loc[eda_df["_churn_binary"] == 1, "MonthlyCharges"].mean()
        retain_mc = eda_df.loc[eda_df["_churn_binary"] == 0, "MonthlyCharges"].mean()
        diff = churn_mc - retain_mc
        direction = "higher" if diff >= 0 else "lower"
        insights.append(
            f"Churned customers pay an average of ${churn_mc:.2f}/month, which is ${abs(diff):.2f} {direction} "
            f"than retained customers (${retain_mc:.2f}/month)."
        )

    if "PaymentMethod" in eda_df.columns:
        pm_rates = (
            eda_df.groupby("PaymentMethod")["_churn_binary"]
            .mean()
            .sort_values(ascending=False)
        )
        if not pm_rates.empty:
            insights.append(
                f"Among payment methods, '{pm_rates.index[0]}' records the highest churn rate at {pm_rates.iloc[0] * 100:.1f}%."
            )

    if "InternetService" in eda_df.columns:
        net_rates = (
            eda_df.groupby("InternetService")["_churn_binary"]
            .mean()
            .sort_values(ascending=False)
        )
        if not net_rates.empty:
            insights.append(
                f"Within internet service tiers, '{net_rates.index[0]}' experiences a {net_rates.iloc[0] * 100:.1f}% churn rate, "
                f"while '{net_rates.index[-1]}' experiences {net_rates.iloc[-1] * 100:.1f}%."
            )

    return insights
