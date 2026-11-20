"""Business intelligence metrics, segmentation, and rule-based retention insights."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.config import PALETTE, RISK_THRESHOLDS
from utils.helpers import classify_risk
from utils.validation import coerce_numeric_strings, normalize_target_series


def compute_business_intelligence(
    df: pd.DataFrame,
    target_col: str,
    bundle: dict[str, Any] | None = None,
    high_risk_threshold: float = RISK_THRESHOLDS["high_min"],
) -> dict[str, Any]:
    """Compute churn business metrics, customer segments, and rule-based recommendations.

    Distinguishes clearly between observed historical metrics and model-predicted risk
    segments, and provides transparent rule-based recommendations without claiming
    validated intervention outcomes.
    """
    working, _ = coerce_numeric_strings(df)
    working["_churn_binary"] = normalize_target_series(working[target_col])

    total_customers = int(len(working))
    overall_churn_rate = float(working["_churn_binary"].mean()) if total_customers else 0.0

    avg_monthly_churned = (
        float(working.loc[working["_churn_binary"] == 1, "MonthlyCharges"].mean())
        if "MonthlyCharges" in working.columns and (working["_churn_binary"] == 1).any()
        else 0.0
    )
    avg_monthly_retained = (
        float(working.loc[working["_churn_binary"] == 0, "MonthlyCharges"].mean())
        if "MonthlyCharges" in working.columns and (working["_churn_binary"] == 0).any()
        else 0.0
    )

    scored_with_model = False
    if bundle is not None:
        features = bundle["feature_names"]
        if all(f in working.columns for f in features):
            X_score = working[features].copy()
            for col in bundle.get("numerical_cols", []):
                if col in X_score.columns:
                    X_score[col] = pd.to_numeric(X_score[col], errors="coerce")
            probs = bundle["pipeline"].predict_proba(X_score)[:, 1]
            working["ChurnProbability"] = probs
            working["RiskCategory"] = [
                classify_risk(float(p), high_min=high_risk_threshold) for p in probs
            ]
            scored_with_model = True

    if not scored_with_model:
        # Fallback heuristic probability proxy only if no model is trained yet
        working["ChurnProbability"] = working["_churn_binary"].astype(float)
        working["RiskCategory"] = np.where(
            working["_churn_binary"] == 1, "High Risk", "Low Risk"
        )

    high_risk_df = working[working["RiskCategory"] == "High Risk"]
    high_risk_count = int(len(high_risk_df))
    high_risk_monthly_rev = (
        float(high_risk_df["MonthlyCharges"].sum())
        if "MonthlyCharges" in high_risk_df.columns
        else 0.0
    )
    high_risk_annual_rev = high_risk_monthly_rev * 12.0

    # Segmentation by Risk Category
    segment_cols: dict[str, Any] = {
        "Customers": ("_churn_binary", "count"),
        "Observed Churn Rate": ("_churn_binary", "mean"),
    }
    if "MonthlyCharges" in working.columns:
        segment_cols["Avg Monthly Charges"] = ("MonthlyCharges", "mean")
        segment_cols["Monthly Revenue"] = ("MonthlyCharges", "sum")
    if "tenure" in working.columns:
        segment_cols["Avg Tenure (mo)"] = ("tenure", "mean")

    risk_segments = (
        working.groupby("RiskCategory")
        .agg(**segment_cols)
        .reindex(["High Risk", "Medium Risk", "Low Risk"])
        .dropna(how="all")
        .reset_index()
    )

    # Contract segmentation
    contract_summary = pd.DataFrame()
    if "Contract" in working.columns:
        contract_summary = (
            working.groupby("Contract")
            .agg(
                Customers=("_churn_binary", "count"),
                ObservedChurnRate=("_churn_binary", "mean"),
                HighRiskCustomers=("RiskCategory", lambda s: int((s == "High Risk").sum())),
                AvgMonthlyCharges=("MonthlyCharges", "mean")
                if "MonthlyCharges" in working.columns
                else ("_churn_binary", "mean"),
            )
            .reset_index()
            .sort_values("ObservedChurnRate", ascending=False)
        )

    # Rule-based retention recommendations (transparent rules)
    recommendations = _generate_retention_recommendations(
        working=working,
        high_risk_df=high_risk_df,
        scored_with_model=scored_with_model,
    )

    return {
        "scored_with_model": scored_with_model,
        "total_customers": total_customers,
        "overall_churn_rate": overall_churn_rate,
        "high_risk_count": high_risk_count,
        "high_risk_monthly_revenue": high_risk_monthly_rev,
        "high_risk_annual_revenue": high_risk_annual_rev,
        "avg_monthly_churned": avg_monthly_churned,
        "avg_monthly_retained": avg_monthly_retained,
        "risk_segments": risk_segments,
        "contract_summary": contract_summary,
        "scored_df": working,
        "recommendations": recommendations,
    }


def _generate_retention_recommendations(
    working: pd.DataFrame,
    high_risk_df: pd.DataFrame,
    scored_with_model: bool,
) -> list[dict[str, str]]:
    """Transparent rule-based retention strategy recommendations derived from dataset segments."""
    recs: list[dict[str, str]] = []
    basis_label = "model-predicted High Risk cohort" if scored_with_model else "observed churned cohort"

    if "Contract" in high_risk_df.columns and not high_risk_df.empty:
        mtm = high_risk_df[high_risk_df["Contract"] == "Month-to-month"]
        if not mtm.empty:
            rev = float(mtm["MonthlyCharges"].sum()) if "MonthlyCharges" in mtm.columns else 0.0
            recs.append(
                {
                    "segment": "Month-to-Month High-Risk Accounts",
                    "rule": "RiskCategory == 'High Risk' AND Contract == 'Month-to-month'",
                    "volume": f"{len(mtm):,} customers (${rev:,.2f}/mo associated billing)",
                    "recommendation": (
                        f"Prioritize contract-migration offers (e.g., 1-year commitment discount) for these "
                        f"{len(mtm):,} {basis_label} accounts currently on flexible month-to-month billing."
                    ),
                }
            )

    if "MonthlyCharges" in high_risk_df.columns and not high_risk_df.empty:
        med_charge = float(working["MonthlyCharges"].median())
        high_spend = high_risk_df[high_risk_df["MonthlyCharges"] >= med_charge]
        if not high_spend.empty:
            rev = float(high_spend["MonthlyCharges"].sum())
            recs.append(
                {
                    "segment": "High-Spend High-Risk Accounts",
                    "rule": f"RiskCategory == 'High Risk' AND MonthlyCharges >= ${med_charge:.2f} (dataset median)",
                    "volume": f"{len(high_spend):,} customers (${rev:,.2f}/mo associated billing)",
                    "recommendation": (
                        "Route accounts with above-median monthly charges and elevated churn risk to "
                        "proactive account-manager outreach and targeted loyalty credits."
                    ),
                }
            )

    if "tenure" in high_risk_df.columns and not high_risk_df.empty:
        early_tenure = high_risk_df[high_risk_df["tenure"] <= 12]
        if not early_tenure.empty:
            recs.append(
                {
                    "segment": "Early-Tenure Onboarding Cohort (0–12 Months)",
                    "rule": "RiskCategory == 'High Risk' AND tenure <= 12 months",
                    "volume": f"{len(early_tenure):,} customers",
                    "recommendation": (
                        "Deploy structured onboarding check-ins and service-setup support during the first "
                        "year of tenure, where churn concentration is highest."
                    ),
                }
            )

    if "PaymentMethod" in high_risk_df.columns and not high_risk_df.empty:
        echeck = high_risk_df[
            high_risk_df["PaymentMethod"].astype(str).str.contains("Electronic check", case=False, na=False)
        ]
        if not echeck.empty:
            recs.append(
                {
                    "segment": "Electronic Check High-Risk Segment",
                    "rule": "RiskCategory == 'High Risk' AND PaymentMethod == 'Electronic check'",
                    "volume": f"{len(echeck):,} customers",
                    "recommendation": (
                        "Encourage enrollment in automated payment methods (auto-pay bank transfer or credit card) "
                        "to reduce billing friction associated with electronic check churn."
                    ),
                }
            )

    if not recs:
        recs.append(
            {
                "segment": "General Retention Monitoring",
                "rule": "Fallback rule when no specific high-risk sub-segments match",
                "volume": f"{len(high_risk_df):,} customers",
                "recommendation": "Continue monitoring churn probabilities and re-evaluate threshold calibration.",
            }
        )

    return recs


def plot_contract_risk_chart(contract_summary: pd.DataFrame) -> go.Figure:
    """Bar chart of observed churn rate by contract type."""
    df = contract_summary.copy()
    df["ChurnRatePct"] = df["ObservedChurnRate"] * 100
    fig = px.bar(
        df,
        x="Contract",
        y="ChurnRatePct",
        text=df["ChurnRatePct"].apply(lambda v: f"{v:.1f}%"),
        color="Contract",
        color_discrete_sequence=[PALETTE["crimson"], PALETTE["amber"], PALETTE["primary_blue"]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=dict(text="Churn Rate by Contract Type", font=dict(size=15, color=PALETTE["navy"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        yaxis_title="Observed Churn Rate (%)",
        xaxis_title="Contract Type",
        yaxis=dict(range=[0, max(df["ChurnRatePct"].max() * 1.2, 10)]),
    )
    return fig


def plot_risk_segment_revenue(risk_segments: pd.DataFrame) -> go.Figure:
    """Bar chart of monthly revenue exposure across predicted risk tiers."""
    if "Monthly Revenue" not in risk_segments.columns:
        return go.Figure()
    color_map = {
        "High Risk": PALETTE["crimson"],
        "Medium Risk": PALETTE["amber"],
        "Low Risk": PALETTE["emerald"],
    }
    fig = px.bar(
        risk_segments,
        x="RiskCategory",
        y="Monthly Revenue",
        color="RiskCategory",
        color_discrete_map=color_map,
        text=risk_segments["Monthly Revenue"].apply(lambda v: f"${v:,.0f}/mo"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=dict(text="Monthly Billing Exposure by Risk Segment", font=dict(size=15, color=PALETTE["navy"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        yaxis_title="Monthly Revenue ($)",
        xaxis_title="Predicted Risk Tier",
        yaxis=dict(range=[0, max(risk_segments["Monthly Revenue"].max() * 1.2, 100)]),
    )
    return fig
