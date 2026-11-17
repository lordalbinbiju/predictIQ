"""Dataset validation, type inference, and input schema checks for PredictIQ."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.config import (
    COMMON_ID_COLUMNS,
    COMMON_TARGET_COLUMNS,
    NEGATIVE_TARGET_VALUES,
    POSITIVE_TARGET_VALUES,
)


@dataclass
class ValidationReport:
    """Structured data quality and schema validation summary."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_rows: int = 0
    total_columns: int = 0
    missing_values_total: int = 0
    missing_by_column: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    invalid_values: dict[str, int] = field(default_factory=dict)
    numerical_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    target_column: str | None = None
    target_distribution: dict[str, int] = field(default_factory=dict)
    column_dtypes: dict[str, str] = field(default_factory=dict)


def detect_id_columns(df: pd.DataFrame) -> list[str]:
    """Identify likely customer/record identifier columns."""
    id_cols: list[str] = []
    for col in df.columns:
        normalized = col.strip().lower()
        if normalized in COMMON_ID_COLUMNS:
            id_cols.append(col)
        elif (
            df[col].nunique(dropna=True) == len(df)
            and len(df) > 10
            and df[col].dtype == object
            and ("id" in normalized)
        ):
            id_cols.append(col)
    return id_cols


def detect_target_column(df: pd.DataFrame) -> str | None:
    """Detect the churn target column using standard naming conventions."""
    col_map = {c.strip().lower(): c for c in df.columns}
    for candidate in COMMON_TARGET_COLUMNS:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    return None


def normalize_target_series(series: pd.Series) -> pd.Series:
    """Map binary churn labels (Yes/No, 1/0, True/False) to integers 1 and 0."""
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            return series.astype(int)
        raise ValueError(
            f"Numeric target column must contain only 0 and 1; found {sorted(unique_vals)}."
        )

    mapped = []
    unknown: set[str] = set()
    for val in series:
        if pd.isna(val):
            mapped.append(np.nan)
            continue
        text = str(val).strip().lower()
        if text in POSITIVE_TARGET_VALUES:
            mapped.append(1)
        elif text in NEGATIVE_TARGET_VALUES:
            mapped.append(0)
        else:
            unknown.add(str(val))
            mapped.append(np.nan)

    if unknown:
        raise ValueError(
            f"Unrecognized target values: {sorted(unknown)}. Expected Yes/No, 1/0, or True/False."
        )
    return pd.Series(mapped, index=series.index, dtype="float64")


def coerce_numeric_strings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Detect and coerce object columns that represent numbers (e.g., TotalCharges).

    Whitespace-only or non-numeric entries in numeric-like columns are converted to NaN
    and counted in invalid_values so they can be reported and imputed cleanly.
    """
    cleaned = df.copy()
    invalid_counts: dict[str, int] = {}

    for col in cleaned.columns:
        if cleaned[col].dtype == object or pd.api.types.is_string_dtype(cleaned[col]):
            series_str = cleaned[col].astype("string")
            non_null = series_str.dropna()
            if non_null.empty:
                continue

            stripped = non_null.str.strip()
            numeric_attempt = pd.to_numeric(stripped, errors="coerce")
            valid_ratio = numeric_attempt.notna().mean()

            # Treat as numeric if >= 85% of non-null entries parse as numbers
            if valid_ratio >= 0.85:
                blank_or_bad = int((numeric_attempt.isna()).sum())
                if blank_or_bad > 0:
                    invalid_counts[col] = blank_or_bad
                full_stripped = series_str.str.strip()
                cleaned[col] = pd.to_numeric(full_stripped, errors="coerce")

    return cleaned, invalid_counts


def validate_dataframe(
    df: pd.DataFrame,
    target_col: str | None = None,
) -> ValidationReport:
    """Run comprehensive data quality, schema, and consistency checks."""
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        return ValidationReport(
            is_valid=False,
            errors=["Dataset is empty. Please load a CSV with at least one row."],
        )

    if df.shape[1] < 2:
        errors.append("Dataset must contain at least one feature column and one target column.")

    coerced_df, invalid_values = coerce_numeric_strings(df)

    # Check for negative values in monetary/tenure-like columns
    for col in coerced_df.select_dtypes(include=[np.number]).columns:
        neg_count = int((coerced_df[col] < 0).sum())
        if neg_count > 0:
            invalid_values[f"{col} (negative)"] = neg_count
            warnings.append(f"Column '{col}' contains {neg_count} negative value(s).")

    resolved_target = target_col or detect_target_column(coerced_df)
    if resolved_target is None or resolved_target not in coerced_df.columns:
        errors.append(
            "Target column could not be detected. Expected a binary churn column such as 'Churn'."
        )

    id_cols = detect_id_columns(coerced_df)
    missing_by_col = {
        col: int(cnt) for col, cnt in coerced_df.isna().sum().items() if int(cnt) > 0
    }
    missing_total = int(coerced_df.isna().sum().sum())
    duplicate_rows = int(coerced_df.duplicated().sum())

    if duplicate_rows > 0:
        warnings.append(f"Detected {duplicate_rows} duplicate record(s).")

    for col, cnt in invalid_values.items():
        if "(negative)" not in col:
            warnings.append(
                f"Column '{col}' had {cnt} blank/non-numeric value(s) coerced to missing (NaN)."
            )

    excluded = set(id_cols)
    if resolved_target:
        excluded.add(resolved_target)

    numerical_cols = [
        c
        for c in coerced_df.select_dtypes(include=[np.number]).columns
        if c not in excluded
    ]
    categorical_cols = [
        c
        for c in coerced_df.columns
        if c not in excluded and c not in numerical_cols
    ]

    # Treat low-cardinality binary indicator columns like SeniorCitizen (0/1) cleanly as numeric
    target_dist: dict[str, int] = {}
    if resolved_target and resolved_target in coerced_df.columns:
        target_series = coerced_df[resolved_target]
        if target_series.isna().any():
            na_target = int(target_series.isna().sum())
            warnings.append(f"Target column '{resolved_target}' contains {na_target} missing value(s).")
        try:
            normalized = normalize_target_series(target_series.dropna())
            counts = normalized.value_counts().to_dict()
            target_dist = {
                "Retained (0)": int(counts.get(0, 0)),
                "Churned (1)": int(counts.get(1, 0)),
            }
            if len(counts) < 2:
                errors.append(
                    f"Target column '{resolved_target}' must contain both churned and retained classes."
                )
        except ValueError as exc:
            errors.append(str(exc))

    if len(coerced_df) < 20:
        warnings.append(
            "Dataset has fewer than 20 rows; model training and cross-validation may be unstable."
        )

    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        total_rows=int(coerced_df.shape[0]),
        total_columns=int(coerced_df.shape[1]),
        missing_values_total=missing_total,
        missing_by_column=missing_by_col,
        duplicate_rows=duplicate_rows,
        invalid_values=invalid_values,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        id_columns=id_cols,
        target_column=resolved_target,
        target_distribution=target_dist,
        column_dtypes={col: str(dtype) for col, dtype in coerced_df.dtypes.items()},
    )


def validate_prediction_input(
    df: pd.DataFrame,
    expected_features: list[str],
) -> tuple[bool, list[str], list[str]]:
    """Validate incoming batch or single-row prediction dataframe against trained model features."""
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        return False, ["Input data contains zero rows."], warnings

    missing_cols = [c for c in expected_features if c not in df.columns]
    if missing_cols:
        errors.append(
            f"Missing required feature column(s): {', '.join(missing_cols)}"
        )

    extra_cols = [
        c for c in df.columns if c not in expected_features and c.lower() not in COMMON_ID_COLUMNS and c.lower() != "churn"
    ]
    if extra_cols:
        warnings.append(
            f"Unused extra column(s) will be ignored during scoring: {', '.join(extra_cols)}"
        )

    return len(errors) == 0, errors, warnings
