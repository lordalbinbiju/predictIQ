"""Model evaluation visualizations, classification reports, and comparison metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from utils.config import PALETTE

MODEL_COLORS = [
    PALETTE["primary_blue"],
    PALETTE["teal"],
    PALETTE["amber"],
    PALETTE["accent_blue"],
    PALETTE["crimson"],
]


def _style_fig(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PALETTE["navy"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color=PALETTE["slate_900"], size=12),
        margin=dict(l=45, r=25, t=50, b=45),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", linecolor=PALETTE["slate_200"])
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", linecolor=PALETTE["slate_200"])
    return fig


def build_classification_report_df(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return a clean DataFrame representation of sklearn's classification_report."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Retained (0)", "Churned (1)"],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "Class / Aggregate": label,
                    "Precision": round(float(values["precision"]), 4),
                    "Recall": round(float(values["recall"]), 4),
                    "F1-Score": round(float(values["f1-score"]), 4),
                    "Support": int(values["support"]),
                }
            )
    return pd.DataFrame(rows)


def plot_confusion_matrix_heatmap(
    cm: list[list[int]] | np.ndarray,
    title: str = "Test Set Confusion Matrix",
) -> go.Figure:
    """Render an annotated confusion matrix heatmap using Plotly."""
    cm_arr = np.asarray(cm, dtype=int)
    labels = ["Retained (0)", "Churned (1)"]
    total = cm_arr.sum()
    annotations = [
        [
            f"{cm_arr[i, j]:,}<br>({(cm_arr[i, j] / total * 100):.1f}%)" if total else "0"
            for j in range(2)
        ]
        for i in range(2)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=["Predicted: Retained", "Predicted: Churned"],
            y=["Actual: Retained", "Actual: Churned"],
            text=annotations,
            texttemplate="%{text}",
            colorscale=[[0, "#EFF6FF"], [0.5, "#60A5FA"], [1, "#0A2540"]],
            showscale=True,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _style_fig(fig, title)


def plot_roc_curves(bundles: list[dict[str, Any]]) -> go.Figure:
    """Plot multi-model ROC curves on held-out test predictions."""
    fig = go.Figure()
    for idx, b in enumerate(bundles):
        y_test = b.get("y_test")
        y_proba = b.get("y_test_proba")
        if y_test is None or y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        label = f"{b['algorithm']} ({b['model_version'][-6:]}) — AUC={roc_auc:.3f}"
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=label,
                line=dict(width=2.5, color=MODEL_COLORS[idx % len(MODEL_COLORS)]),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline (AUC=0.500)",
            line=dict(dash="dash", color="#94A3B8", width=1.5),
        )
    )
    fig.update_xaxes(title_text="False Positive Rate (1 - Specificity)", range=[0, 1])
    fig.update_yaxes(title_text="True Positive Rate (Recall)", range=[0, 1.02])
    return _style_fig(fig, "ROC Curve Comparison (Held-Out Test Set)")


def plot_precision_recall_curves(bundles: list[dict[str, Any]]) -> go.Figure:
    """Plot multi-model Precision-Recall curves on held-out test predictions."""
    fig = go.Figure()
    for idx, b in enumerate(bundles):
        y_test = b.get("y_test")
        y_proba = b.get("y_test_proba")
        if y_test is None or y_proba is None:
            continue
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        label = f"{b['algorithm']} ({b['model_version'][-6:]}) — PR-AUC={pr_auc:.3f}"
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=label,
                line=dict(width=2.5, color=MODEL_COLORS[idx % len(MODEL_COLORS)]),
            )
        )

    fig.update_xaxes(title_text="Recall (Sensitivity)", range=[0, 1])
    fig.update_yaxes(title_text="Precision (Positive Predictive Value)", range=[0, 1.05])
    return _style_fig(fig, "Precision-Recall Curve Comparison (Held-Out Test Set)")


def build_model_comparison_df(
    bundles: list[dict[str, Any]],
    split: str = "test",
) -> pd.DataFrame:
    """Compile evaluation metrics across all trained models for a given split."""
    rows = []
    for b in bundles:
        m = b.get("metrics_by_split", {}).get(split, {})
        cv = b.get("cv_summary", {})
        rows.append(
            {
                "Model Version": b["model_version"],
                "Algorithm": b["algorithm"],
                "Accuracy": round(float(m.get("accuracy", 0.0)), 4),
                "Precision": round(float(m.get("precision", 0.0)), 4),
                "Recall": round(float(m.get("recall", 0.0)), 4),
                "F1-Score": round(float(m.get("f1_score", 0.0)), 4),
                "ROC-AUC": round(float(m.get("roc_auc", 0.0)), 4),
                "CV ROC-AUC (Mean)": round(float(cv.get("cv_roc_auc_mean", 0.0)), 4),
                "CV Recall (Mean)": round(float(cv.get("cv_recall_mean", 0.0)), 4),
                "Active": "Yes" if b.get("is_active") else "No",
            }
        )
    return pd.DataFrame(rows)


def plot_model_comparison_chart(
    comparison_df: pd.DataFrame,
    split_label: str = "Test",
) -> go.Figure:
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1-Score, and ROC-AUC."""
    melted = comparison_df.melt(
        id_vars=["Model Version", "Algorithm"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    melted["ModelLabel"] = (
        melted["Algorithm"] + " (" + melted["Model Version"].str[-6:] + ")"
    )
    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="ModelLabel",
        barmode="group",
        color_discrete_sequence=MODEL_COLORS,
        text_auto=".3f",
    )
    fig.update_yaxes(title_text="Score (0.0 – 1.0)", range=[0, 1.12])
    fig.update_xaxes(title_text="Evaluation Metric")
    return _style_fig(fig, f"Model Performance Comparison ({split_label} Set)")
