"""Data loading and preparation utilities for PredictIQ."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import pandas as pd

from utils.config import SAMPLE_DATASET_PATH
from utils.validation import (
    ValidationReport,
    coerce_numeric_strings,
    normalize_target_series,
    validate_dataframe,
)


def load_sample_dataset(path: Path = SAMPLE_DATASET_PATH) -> pd.DataFrame:
    """Load the bundled IBM Telco Customer Churn CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Sample dataset not found at {path}. Ensure data/telco_churn.csv exists."
        )
    return pd.read_csv(path)


def load_uploaded_csv(file_obj: IO[bytes] | Path | str) -> pd.DataFrame:
    """Load a user-uploaded CSV file safely with encoding fallback."""
    try:
        return pd.read_csv(file_obj)
    except UnicodeDecodeError:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="latin-1")


def prepare_dataset(
    df: pd.DataFrame,
    target_col: str | None = None,
    drop_duplicates: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and coerce raw dataframe columns while preserving raw missingness for reporting."""
    report = validate_dataframe(df, target_col=target_col)
    prepared, _ = coerce_numeric_strings(df)

    if drop_duplicates and report.duplicate_rows > 0:
        prepared = prepared.drop_duplicates().reset_index(drop=True)

    if report.target_column and report.target_column in prepared.columns:
        try:
            prepared["_churn_binary"] = normalize_target_series(
                prepared[report.target_column]
            )
        except ValueError:
            pass

    return prepared, report
