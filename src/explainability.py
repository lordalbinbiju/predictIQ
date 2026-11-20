"""SHAP-based global and local Explainable AI engine for PredictIQ."""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from utils.config import PALETTE
from utils.helpers import clean_feature_name


def _extract_positive_shap_matrix(shap_vals: Any) -> np.ndarray:
    """Normalize SHAP output across binary TreeExplainer/LinearExplainer variants to (n, f)."""
    if isinstance(shap_vals, list):
        arr = np.asarray(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0], dtype=float)
    else:
        arr = np.asarray(shap_vals, dtype=float)

    if arr.ndim == 3:
        # Shape (n_samples, n_features, 2) -> positive class index 1
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def _extract_base_value(expected_value: Any) -> float:
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        arr = np.asarray(expected_value, dtype=float).ravel()
        return float(arr[1] if len(arr) > 1 else arr[0])
    return float(expected_value)


def _build_shap_explainer(bundle: dict[str, Any]) -> tuple[Any, str]:
    """Instantiate the appropriate SHAP explainer for the trained classifier."""
    pipeline = bundle["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    algorithm = bundle["algorithm"]
    background = bundle["background_transformed"]

    if algorithm == "Logistic Regression":
        return shap.LinearExplainer(classifier, background), "linear"

    if algorithm in ("Random Forest", "Gradient Boosting", "XGBoost"):
        try:
            explainer = shap.TreeExplainer(classifier)
            return explainer, "tree"
        except Exception:
            pass

    return shap.Explainer(classifier.predict_proba, background), "generic"


def compute_global_shap(
    bundle: dict[str, Any],
    X_sample_raw: pd.DataFrame,
    max_rows: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute global SHAP values and feature importance rankings."""
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names_raw = bundle.get("transformed_feature_names") or []

    sample_df = X_sample_raw[bundle["feature_names"]].copy()
    if len(sample_df) > max_rows:
        sample_df = sample_df.sample(n=max_rows, random_state=random_state)

    X_trans = np.asarray(preprocessor.transform(sample_df), dtype=float)
    if not feature_names_raw:
        feature_names_raw = [f"f_{i}" for i in range(X_trans.shape[1])]
    clean_names = [clean_feature_name(f) for f in feature_names_raw]

    explainer, kind = _build_shap_explainer(bundle)
    if kind == "generic":
        exp = explainer(X_trans)
        shap_matrix = _extract_positive_shap_matrix(exp.values)
        base_value = _extract_base_value(exp.base_values[0])
    else:
        raw_vals = explainer.shap_values(X_trans)
        shap_matrix = _extract_positive_shap_matrix(raw_vals)
        base_value = _extract_base_value(explainer.expected_value)

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    mean_signed = np.mean(shap_matrix, axis=0)

    importance_df = (
        pd.DataFrame(
            {
                "Feature": clean_names,
                "Raw Feature": feature_names_raw,
                "Mean |SHAP|": mean_abs,
                "Mean Signed SHAP": mean_signed,
            }
        )
        .sort_values("Mean |SHAP|", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "shap_matrix": shap_matrix,
        "X_transformed": X_trans,
        "feature_names": clean_names,
        "base_value": base_value,
        "importance_df": importance_df,
        "sample_size": len(sample_df),
    }


def compute_individual_shap(
    bundle: dict[str, Any],
    customer_row_raw: pd.DataFrame,
) -> dict[str, Any]:
    """Compute local SHAP explanation for a single customer record."""
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names_raw = bundle.get("transformed_feature_names") or []

    X_row = customer_row_raw[bundle["feature_names"]].iloc[[0]].copy()
    X_trans = np.asarray(preprocessor.transform(X_row), dtype=float)
    if not feature_names_raw:
        feature_names_raw = [f"f_{i}" for i in range(X_trans.shape[1])]
    clean_names = [clean_feature_name(f) for f in feature_names_raw]

    explainer, kind = _build_shap_explainer(bundle)
    if kind == "generic":
        exp = explainer(X_trans)
        shap_vec = _extract_positive_shap_matrix(exp.values)[0]
        base_value = _extract_base_value(exp.base_values[0])
    else:
        raw_vals = explainer.shap_values(X_trans)
        shap_vec = _extract_positive_shap_matrix(raw_vals)[0]
        base_value = _extract_base_value(explainer.expected_value)

    pred_prob = float(pipeline.predict_proba(X_row)[0, 1])

    contrib_df = pd.DataFrame(
        {
            "Feature": clean_names,
            "Transformed Value": X_trans[0],
            "SHAP Contribution": shap_vec,
            "Abs Contribution": np.abs(shap_vec),
        }
    )

    # Filter out inactive one-hot dummy levels (=0) when summarizing active customer attributes,
    # while keeping all features in raw explanation object for waterfall plot.
    active_mask = np.ones(len(contrib_df), dtype=bool)
    for idx, fname in enumerate(feature_names_raw):
        if fname.startswith("cat__") and abs(X_trans[0, idx]) < 1e-9:
            active_mask[idx] = False

    active_df = (
        contrib_df.loc[active_mask]
        .sort_values("Abs Contribution", ascending=False)
        .reset_index(drop=True)
    )
    all_sorted = contrib_df.sort_values("Abs Contribution", ascending=False).reset_index(drop=True)

    return {
        "shap_values": shap_vec,
        "X_transformed": X_trans[0],
        "feature_names": clean_names,
        "base_value": base_value,
        "predicted_probability": pred_prob,
        "contributions_df": all_sorted,
        "active_contributions_df": active_df,
        "raw_customer": X_row.iloc[0].to_dict(),
    }


def plot_global_shap_bar(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Interactive Plotly horizontal bar chart of top global SHAP feature importances."""
    top_df = importance_df.head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top_df["Mean |SHAP|"],
            y=top_df["Feature"],
            orientation="h",
            marker_color=PALETTE["primary_blue"],
            text=[f"{v:.4f}" for v in top_df["Mean |SHAP|"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Top {min(top_n, len(importance_df))} Global Churn Drivers (Mean |SHAP Value|)",
            font=dict(size=15, color=PALETTE["navy"]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis_title="Mean |SHAP Value| (Impact on Model Output)",
        yaxis_title="",
        margin=dict(l=20, r=45, t=50, b=40),
        height=max(380, min(top_n, len(importance_df)) * 28 + 100),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    return fig


def plot_shap_beeswarm_matplotlib(global_shap: dict[str, Any], max_display: int = 15) -> plt.Figure:
    """Generate a Matplotlib SHAP beeswarm summary figure."""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=120)
    plt.sca(ax)
    shap.summary_plot(
        global_shap["shap_matrix"],
        features=global_shap["X_transformed"],
        feature_names=global_shap["feature_names"],
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    ax.set_title("SHAP Summary Beeswarm Plot (Feature Impact & Direction)", fontsize=12, color=PALETTE["navy"], pad=12)
    fig.tight_layout()
    return fig


def plot_individual_shap_waterfall_matplotlib(
    local_shap: dict[str, Any],
    max_display: int = 12,
) -> plt.Figure:
    """Generate a Matplotlib SHAP waterfall figure for an individual customer."""
    plt.close("all")
    explanation = shap.Explanation(
        values=local_shap["shap_values"],
        base_values=local_shap["base_value"],
        data=local_shap["X_transformed"],
        feature_names=local_shap["feature_names"],
    )
    fig = plt.figure(figsize=(9.5, 5.5), dpi=120)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.title(
        f"Individual Customer SHAP Waterfall (Predicted Churn Prob: {local_shap['predicted_probability'] * 100:.1f}%)",
        fontsize=11,
        color=PALETTE["navy"],
        pad=12,
    )
    plt.tight_layout()
    return fig


def plot_individual_contributions_plotly(
    local_shap: dict[str, Any],
    top_n: int = 12,
) -> go.Figure:
    """Plotly horizontal bar chart showing positive vs negative SHAP drivers for one customer."""
    df = local_shap["active_contributions_df"]
    if df.empty:
        df = local_shap["contributions_df"]
    top = df.head(top_n).iloc[::-1]
    colors = [
        PALETTE["crimson"] if val > 0 else PALETTE["teal"]
        for val in top["SHAP Contribution"]
    ]
    fig = go.Figure(
        go.Bar(
            x=top["SHAP Contribution"],
            y=top["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in top["SHAP Contribution"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_width=1.5, line_color="#64748B")
    fig.update_layout(
        title=dict(
            text="Customer-Specific Feature Contributions (Red = Increases Risk, Teal = Decreases Risk)",
            font=dict(size=14, color=PALETTE["navy"]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis_title="SHAP Contribution to Churn Log-Odds / Probability",
        yaxis_title="",
        margin=dict(l=20, r=55, t=50, b=40),
        height=max(360, len(top) * 30 + 100),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    return fig


def generate_individual_shap_narrative(local_shap: dict[str, Any]) -> dict[str, Any]:
    """Generate transparent, value-grounded natural-language explanation from calculated SHAP values."""
    df = local_shap["active_contributions_df"]
    if df.empty:
        df = local_shap["contributions_df"]

    pos_df = df[df["SHAP Contribution"] > 0].sort_values("SHAP Contribution", ascending=False)
    neg_df = df[df["SHAP Contribution"] < 0].sort_values("SHAP Contribution", ascending=True)

    top_pos = pos_df.head(3).to_dict("records")
    top_neg = neg_df.head(3).to_dict("records")

    prob = local_shap["predicted_probability"]
    raw_customer = local_shap.get("raw_customer", {})

    def _describe_factor(rec: dict[str, Any]) -> str:
        feat = rec["Feature"]
        contrib = rec["SHAP Contribution"]
        if ":" in feat:
            return f"{feat} (SHAP {contrib:+.3f})"
        val = raw_customer.get(feat)
        if val is not None:
            return f"{feat} = {val} (SHAP {contrib:+.3f})"
        return f"{feat} (SHAP {contrib:+.3f})"

    pos_phrases = [_describe_factor(r) for r in top_pos]
    neg_phrases = [_describe_factor(r) for r in top_neg]

    summary_parts = [
        f"The model estimates a {prob * 100:.1f}% churn probability for this customer."
    ]
    if pos_phrases:
        summary_parts.append(
            "Primary factors increasing churn risk: " + "; ".join(pos_phrases) + "."
        )
    if neg_phrases:
        summary_parts.append(
            "Primary factors reducing churn risk: " + "; ".join(neg_phrases) + "."
        )

    return {
        "narrative": " ".join(summary_parts),
        "increasing_factors": top_pos,
        "decreasing_factors": top_neg,
    }
