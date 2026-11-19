"""Machine learning training engine with leakage-free pipelines and CV."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    build_preprocessor,
    get_transformed_feature_names,
    prepare_modeling_data,
    split_dataset,
)
from utils.config import DEFAULT_CV_FOLDS, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, MODELS_DIR

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False


ProgressCallback = Callable[[str, float], None]


@dataclass
class TrainingResult:
    """Output container returned after training a classification pipeline."""

    model_version: str
    algorithm: str
    model_path: Path
    pipeline: Pipeline
    metrics_by_split: dict[str, dict[str, Any]]
    cv_summary: dict[str, float]
    hyperparameters: dict[str, Any]
    feature_names: list[str]
    transformed_feature_names: list[str]
    numerical_cols: list[str]
    categorical_cols: list[str]
    class_balance: dict[str, int]
    test_size: float
    random_state: int
    cv_folds: int
    use_class_weight: bool
    y_test: np.ndarray
    y_test_proba: np.ndarray
    y_val_proba: np.ndarray
    y_train: np.ndarray


def get_available_algorithms() -> list[str]:
    """Return list of supported classification algorithms."""
    algos = [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    ]
    if XGBOOST_AVAILABLE:
        algos.append("XGBoost")
    return algos


def get_default_hyperparameters(algorithm: str) -> dict[str, Any]:
    """Return sensible default hyperparameters for each supported algorithm."""
    defaults: dict[str, dict[str, Any]] = {
        "Logistic Regression": {
            "C": 1.0,
            "max_iter": 1000,
        },
        "Random Forest": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
        "Gradient Boosting": {
            "n_estimators": 150,
            "learning_rate": 0.08,
            "max_depth": 3,
            "subsample": 0.9,
        },
        "XGBoost": {
            "n_estimators": 180,
            "learning_rate": 0.06,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    }
    if algorithm not in defaults:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return dict(defaults[algorithm])


def _build_estimator(
    algorithm: str,
    hyperparameters: dict[str, Any],
    use_class_weight: bool,
    scale_pos_weight: float,
    random_state: int,
) -> Any:
    params = get_default_hyperparameters(algorithm)
    params.update(hyperparameters)

    if algorithm == "Logistic Regression":
        return LogisticRegression(
            C=float(params["C"]),
            max_iter=int(params["max_iter"]),
            class_weight="balanced" if use_class_weight else None,
            random_state=random_state,
        )

    if algorithm == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] in (None, 0) else int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced" if use_class_weight else None,
            random_state=random_state,
            n_jobs=-1,
        )

    if algorithm == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            random_state=random_state,
        )

    if algorithm == "XGBoost":
        if not XGBOOST_AVAILABLE or XGBClassifier is None:
            raise RuntimeError("XGBoost is not installed in the current environment.")
        return XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_churn_model(
    df: pd.DataFrame,
    algorithm: str,
    target_col: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    use_class_weight: bool = True,
    hyperparameters: dict[str, Any] | None = None,
    models_dir: Path = MODELS_DIR,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Train, cross-validate, evaluate, and persist a full Scikit-learn churn pipeline."""

    def _notify(msg: str, pct: float) -> None:
        if progress_callback is not None:
            progress_callback(msg, pct)

    _notify("Preparing features and stratified train-test split...", 0.10)
    X, y, numerical_cols, categorical_cols, _ = prepare_modeling_data(
        df, target_col=target_col, drop_duplicates=True
    )
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=test_size, random_state=random_state
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    resolved_params = get_default_hyperparameters(algorithm)
    if hyperparameters:
        resolved_params.update(hyperparameters)

    preprocessor = build_preprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        scale_numerical=True,
    )
    estimator = _build_estimator(
        algorithm=algorithm,
        hyperparameters=resolved_params,
        use_class_weight=use_class_weight,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    # Cap folds safely if dataset is very small (e.g. unit test fixtures)
    min_class_count = min(neg_count, pos_count)
    effective_folds = max(2, min(cv_folds, min_class_count))
    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)

    _notify(
        f"Running {effective_folds}-fold stratified cross-validation on training split...",
        0.25,
    )
    oof_proba = np.zeros(len(y_train), dtype=float)
    fold_metrics: dict[str, list[float]] = {
        "roc_auc": [],
        "recall": [],
        "precision": [],
        "f1": [],
        "accuracy": [],
    }
    y_train_arr = y_train.to_numpy()

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train_arr), start=1):
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        val_proba = fold_pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)
        oof_proba[val_idx] = val_proba

        y_val_true = y_train_arr[val_idx]
        fold_metrics["roc_auc"].append(float(roc_auc_score(y_val_true, val_proba)))
        fold_metrics["recall"].append(float(recall_score(y_val_true, val_pred, zero_division=0)))
        fold_metrics["precision"].append(float(precision_score(y_val_true, val_pred, zero_division=0)))
        fold_metrics["f1"].append(float(f1_score(y_val_true, val_pred, zero_division=0)))
        fold_metrics["accuracy"].append(float(accuracy_score(y_val_true, val_pred)))

        _notify(
            f"Completed CV fold {fold_idx}/{effective_folds}...",
            0.25 + 0.45 * (fold_idx / effective_folds),
        )

    oof_pred = (oof_proba >= 0.5).astype(int)

    _notify("Fitting final pipeline on full training dataset...", 0.75)
    pipeline.fit(X_train, y_train)

    train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)

    _notify("Evaluating held-out test set...", 0.88)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    metrics_by_split = {
        "train": _compute_classification_metrics(y_train_arr, train_pred, train_proba),
        "validation": _compute_classification_metrics(y_train_arr, oof_pred, oof_proba),
        "test": _compute_classification_metrics(y_test.to_numpy(), test_pred, test_proba),
    }

    cv_summary = {
        "cv_folds": int(effective_folds),
        "cv_roc_auc_mean": float(np.mean(fold_metrics["roc_auc"])),
        "cv_roc_auc_std": float(np.std(fold_metrics["roc_auc"])),
        "cv_recall_mean": float(np.mean(fold_metrics["recall"])),
        "cv_precision_mean": float(np.mean(fold_metrics["precision"])),
        "cv_f1_mean": float(np.mean(fold_metrics["f1"])),
        "cv_accuracy_mean": float(np.mean(fold_metrics["accuracy"])),
    }

    transformed_names = get_transformed_feature_names(pipeline.named_steps["preprocessor"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    slug = algorithm.lower().replace(" ", "_")
    model_version = f"{slug}_{timestamp}"

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_version}.joblib"

    # Keep a compact stratified background sample (transformed) for fast SHAP explanations
    X_bg_raw = X_train.sample(n=min(120, len(X_train)), random_state=random_state)
    X_bg_transformed = pipeline.named_steps["preprocessor"].transform(X_bg_raw)

    artifact = {
        "model_version": model_version,
        "algorithm": algorithm,
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "transformed_feature_names": transformed_names,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "hyperparameters": resolved_params,
        "use_class_weight": use_class_weight,
        "test_size": test_size,
        "random_state": random_state,
        "cv_summary": cv_summary,
        "metrics_by_split": metrics_by_split,
        "y_test": y_test.to_numpy(),
        "y_test_proba": test_proba,
        "y_val_proba": oof_proba,
        "y_train": y_train.to_numpy(),
        "background_transformed": np.asarray(X_bg_transformed, dtype=float),
        "background_raw": X_bg_raw.reset_index(drop=True),
    }

    _notify("Saving trained pipeline bundle to disk...", 0.95)
    joblib.dump(artifact, model_path)
    _notify("Training complete.", 1.0)

    return TrainingResult(
        model_version=model_version,
        algorithm=algorithm,
        model_path=model_path,
        pipeline=pipeline,
        metrics_by_split=metrics_by_split,
        cv_summary=cv_summary,
        hyperparameters=resolved_params,
        feature_names=list(X.columns),
        transformed_feature_names=transformed_names,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        class_balance={
            "retained": int((y == 0).sum()),
            "churned": int((y == 1).sum()),
        },
        test_size=test_size,
        random_state=random_state,
        cv_folds=effective_folds,
        use_class_weight=use_class_weight,
        y_test=y_test.to_numpy(),
        y_test_proba=test_proba,
        y_val_proba=oof_proba,
        y_train=y_train.to_numpy(),
    )
