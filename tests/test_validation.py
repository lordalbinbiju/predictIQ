"""Tests for CSV validation, schema detection, and invalid value checks."""

from __future__ import annotations

import pandas as pd

from src.data_loader import load_sample_dataset
from utils.validation import (
    coerce_numeric_strings,
    validate_dataframe,
    validate_prediction_input,
)


def test_sample_dataset_validation() -> None:
    df = load_sample_dataset()
    report = validate_dataframe(df)

    assert report.is_valid is True
    assert report.total_rows == 7043
    assert report.total_columns == 21
    assert report.target_column == "Churn"
    assert "customerID" in report.id_columns
    assert report.missing_by_column.get("TotalCharges") == 11
    assert report.invalid_values.get("TotalCharges") == 11
    assert "MonthlyCharges" in report.numerical_columns
    assert "Contract" in report.categorical_columns
    assert report.target_distribution == {"Retained (0)": 5174, "Churned (1)": 1869}


def test_empty_dataframe_fails_validation() -> None:
    report = validate_dataframe(pd.DataFrame())
    assert report.is_valid is False
    assert len(report.errors) >= 1


def test_missing_target_column_fails_validation() -> None:
    df = pd.DataFrame({"tenure": [1, 2, 3], "MonthlyCharges": [20.0, 30.0, 40.0]})
    report = validate_dataframe(df)
    assert report.is_valid is False
    assert any("Target column" in err for err in report.errors)


def test_duplicate_and_negative_value_detection() -> None:
    df = pd.DataFrame(
        {
            "tenure": [10, 10, -2, 5],
            "MonthlyCharges": [50.0, 50.0, 70.0, 30.0],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
            "Churn": ["Yes", "Yes", "No", "No"],
        }
    )
    report = validate_dataframe(df)
    assert report.is_valid is True
    assert report.duplicate_rows == 1
    assert report.invalid_values.get("tenure (negative)") == 1


def test_coerce_numeric_strings_handles_whitespace() -> None:
    df = pd.DataFrame({"TotalCharges": ["100.5", " ", "250.0", "bad"]})
    cleaned, invalid = coerce_numeric_strings(df)
    # 2/4 = 50% < 85% threshold -> stays object; test >=85% case
    assert invalid == {}

    df2 = pd.DataFrame(
        {"TotalCharges": ["100.5", "200.0", "300.0", "400.0", "500.0", "600.0", "700.0", " "]}
    )
    cleaned2, invalid2 = coerce_numeric_strings(df2)
    assert pd.api.types.is_numeric_dtype(cleaned2["TotalCharges"])
    assert invalid2["TotalCharges"] == 1
    assert cleaned2["TotalCharges"].isna().sum() == 1


def test_prediction_input_schema_validation() -> None:
    expected = ["tenure", "MonthlyCharges", "Contract"]
    valid_df = pd.DataFrame({"tenure": [5], "MonthlyCharges": [65.0], "Contract": ["One year"]})
    ok, errs, _ = validate_prediction_input(valid_df, expected)
    assert ok is True
    assert errs == []

    missing_df = pd.DataFrame({"tenure": [5]})
    ok_bad, errs_bad, _ = validate_prediction_input(missing_df, expected)
    assert ok_bad is False
    assert "MonthlyCharges" in errs_bad[0]
