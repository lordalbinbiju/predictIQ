"""Tests for preprocessing pipelines, unseen categories, and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_sample_dataset
from src.preprocessing import (
    build_preprocessor,
    get_transformed_feature_names,
    prepare_modeling_data,
    split_dataset,
)


def test_prepare_modeling_data_removes_duplicates_and_ids() -> None:
    df = pd.DataFrame(
        {
            "customerID": ["A1", "A1", "A2", "A3"],
            "tenure": [5, 5, 12, 24],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
            "Churn": ["Yes", "Yes", "No", "No"],
        }
    )
    X, y, num_cols, cat_cols, target = prepare_modeling_data(df, drop_duplicates=True)
    assert len(X) == 3
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert target == "Churn"
    assert num_cols == ["tenure"]
    assert cat_cols == ["Contract"]
    assert y.tolist() == [1, 0, 0]


def test_stratified_split_preserves_class_ratio() -> None:
    df = load_sample_dataset()
    X, y, _, _, _ = prepare_modeling_data(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)

    assert len(X_train) + len(X_test) == len(X)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_preprocessor_handles_missing_and_unseen_categories_without_leakage() -> None:
    X_train = pd.DataFrame(
        {
            "tenure": [2.0, 4.0, 6.0, np.nan],
            "Contract": ["Month-to-month", "One year", "One year", None],
        }
    )
    X_test = pd.DataFrame(
        {
            "tenure": [100.0, np.nan],
            "Contract": ["Five year", "Month-to-month"],
        }
    )

    pre = build_preprocessor(["tenure"], ["Contract"], scale_numerical=True)
    pre.fit(X_train)

    # Verify imputer median came ONLY from X_train [2, 4, 6] -> 4.0, NOT polluted by X_test's 100.0
    train_imputer = pre.named_transformers_["num"].named_steps["imputer"]
    assert train_imputer.statistics_[0] == 4.0

    transformed_test = pre.transform(X_test)
    assert transformed_test.shape[0] == 2
    assert not np.isnan(transformed_test).any()

    names = get_transformed_feature_names(pre)
    assert len(names) == transformed_test.shape[1]
