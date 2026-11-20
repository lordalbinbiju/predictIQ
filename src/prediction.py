"""Single-customer and batch churn prediction module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.config import RISK_THRESHOLDS
from utils.helpers import classify_risk
from utils.validation import (
    coerce_numeric_strings,
    detect_id_columns,
    validate_prediction_input,
)


def _align_features(
    df: pd.DataFrame,
    expected_features: list[str],
    numerical_cols: list[str],
) -> pd.DataFrame:
    aligned, _ = coerce_numeric_strings(df)
    aligned = aligned[expected_features].copy()
    for col in numerical_cols:
        if col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
    return aligned


def predict_single_customer(
    bundle: dict[str, Any],
    customer_data: dict[str, Any],
    threshold: float = 0.50,
    low_max: float = RISK_THRESHOLDS["low_max"],
    high_min: float = RISK_THRESHOLDS["high_min"],
) -> dict[str, Any]:
    """Generate churn class, probability, and risk category for a single customer."""
    if not 0.05 <= threshold <= 0.95:
        raise ValueError("Probability threshold must be between 0.05 and 0.95.")

    expected_features: list[str] = bundle["feature_names"]
    numerical_cols: list[str] = bundle.get("numerical_cols", [])
    pipeline = bundle["pipeline"]

    df_input = pd.DataFrame([customer_data])
    is_valid, errors, _ = validate_prediction_input(df_input, expected_features)
    if not is_valid:
        raise ValueError("; ".join(errors))

    X_row = _align_features(df_input, expected_features, numerical_cols)
    prob = float(pipeline.predict_proba(X_row)[0, 1])
    pred_class = int(prob >= threshold)
    pred_label = "Churn" if pred_class == 1 else "Retained"
    risk_category = classify_risk(prob, low_max=low_max, high_min=high_min)

    return {
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "churn_probability": prob,
        "risk_category": risk_category,
        "threshold": float(threshold),
        "model_version": bundle["model_version"],
        "algorithm": bundle["algorithm"],
        "aligned_row": X_row,
    }


def predict_batch(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    threshold: float = 0.50,
    low_max: float = RISK_THRESHOLDS["low_max"],
    high_min: float = RISK_THRESHOLDS["high_min"],
) -> tuple[pd.DataFrame, list[str]]:
    """Score a batch dataframe of customers using the trained preprocessing + model pipeline."""
    if not 0.05 <= threshold <= 0.95:
        raise ValueError("Probability threshold must be between 0.05 and 0.95.")

    expected_features: list[str] = bundle["feature_names"]
    numerical_cols: list[str] = bundle.get("numerical_cols", [])
    pipeline = bundle["pipeline"]

    is_valid, errors, warnings = validate_prediction_input(df, expected_features)
    if not is_valid:
        raise ValueError("; ".join(errors))

    id_cols = detect_id_columns(df)
    if id_cols:
        customer_ids = df[id_cols[0]].astype(str).tolist()
    else:
        customer_ids = [f"CUST-{idx + 1:05d}" for idx in range(len(df))]

    X_batch = _align_features(df, expected_features, numerical_cols)
    probs = pipeline.predict_proba(X_batch)[:, 1]
    preds = (probs >= threshold).astype(int)
    labels = np.where(preds == 1, "Churn", "Retained")
    risks = [classify_risk(float(p), low_max=low_max, high_min=high_min) for p in probs]

    result_df = pd.DataFrame(
        {
            "Customer ID": customer_ids,
            "Predicted Churn": labels,
            "Predicted Class": preds,
            "Churn Probability": np.round(probs, 4),
            "Risk Category": risks,
        }
    )

    # Preserve key context attributes when present
    for context_col in ["Contract", "tenure", "MonthlyCharges", "InternetService"]:
        if context_col in df.columns:
            result_df[context_col] = df[context_col].values

    return result_df, warnings
