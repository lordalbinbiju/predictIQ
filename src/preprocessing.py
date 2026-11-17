"""Leakage-free preprocessing pipelines and dataset splitting for PredictIQ."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from utils.validation import (
    coerce_numeric_strings,
    detect_id_columns,
    detect_target_column,
    normalize_target_series,
)


def prepare_modeling_data(
    df: pd.DataFrame,
    target_col: str | None = None,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], str]:
    """Clean raw dataframe into feature matrix X, binary target y, and feature column lists.

    Note: Only row-level structural cleanup (coercing numeric strings, dropping rows
    with missing target, deduplicating, dropping ID columns) is performed here.
    All statistical transformations (imputation, scaling, encoding) are encapsulated
    inside the Scikit-learn Pipeline and fitted strictly on training data.
    """
    working, _ = coerce_numeric_strings(df)
    if "_churn_binary" in working.columns:
        working = working.drop(columns=["_churn_binary"])

    resolved_target = target_col or detect_target_column(working)
    if not resolved_target or resolved_target not in working.columns:
        raise ValueError("Unable to identify target column in dataset.")

    if drop_duplicates:
        working = working.drop_duplicates().reset_index(drop=True)

    # Drop rows where target itself is missing
    working = working.dropna(subset=[resolved_target]).reset_index(drop=True)
    y = normalize_target_series(working[resolved_target]).astype(int)

    id_cols = detect_id_columns(working)
    drop_cols = set(id_cols) | {resolved_target}
    X = working.drop(columns=[c for c in drop_cols if c in working.columns])

    numerical_cols = list(X.select_dtypes(include=["number", "bool"]).columns)
    categorical_cols = [c for c in X.columns if c not in numerical_cols]

    # Convert booleans to int for uniform numerical handling
    for col in numerical_cols:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    return X, y, numerical_cols, categorical_cols, resolved_target


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train-test split."""
    if not 0.05 <= test_size <= 0.50:
        raise ValueError("test_size must be between 0.05 and 0.50.")
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(
    numerical_cols: list[str],
    categorical_cols: list[str],
    scale_numerical: bool = True,
) -> ColumnTransformer:
    """Build a Scikit-learn ColumnTransformer for numerical and categorical features."""
    num_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numerical:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(steps=num_steps)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numerical_cols:
        transformers.append(("num", num_pipeline, numerical_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract output feature names from a fitted ColumnTransformer."""
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return []
