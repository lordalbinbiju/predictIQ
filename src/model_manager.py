"""Model persistence, registry synchronization, and best-model selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from database.database import DatabaseManager
from src.model_training import TrainingResult
from utils.config import MODELS_DIR


class ModelManager:
    """Coordinates joblib model artifacts and SQLite model run records."""

    def __init__(
        self,
        db: DatabaseManager | None = None,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        self.db = db or DatabaseManager()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def register_training_result(
        self,
        result: TrainingResult,
        dataset_name: str,
        set_active: bool = True,
    ) -> int:
        """Persist a TrainingResult into SQLite and ensure consistency with its joblib file."""
        if not Path(result.model_path).exists():
            raise FileNotFoundError(f"Expected model artifact missing at {result.model_path}")

        return self.db.save_model_run(
            model_version=result.model_version,
            algorithm=result.algorithm,
            dataset_name=dataset_name,
            test_size=result.test_size,
            random_state=result.random_state,
            use_class_weight=result.use_class_weight,
            hyperparameters=result.hyperparameters,
            cv_folds=result.cv_folds,
            cv_roc_auc_mean=result.cv_summary["cv_roc_auc_mean"],
            cv_roc_auc_std=result.cv_summary["cv_roc_auc_std"],
            cv_recall_mean=result.cv_summary["cv_recall_mean"],
            cv_f1_mean=result.cv_summary["cv_f1_mean"],
            model_path=str(result.model_path),
            feature_names=result.feature_names,
            metrics_by_split=result.metrics_by_split,
            set_active=set_active,
        )

    def list_models(self) -> pd.DataFrame:
        """Return model runs whose serialized artifact exists on disk."""
        df = self.db.get_model_runs_with_test_metrics()
        if df.empty:
            return df
        exists_mask = df["model_path"].apply(lambda p: Path(p).exists())
        return df.loc[exists_mask].reset_index(drop=True)

    def load_model_bundle(self, model_version: str | None = None) -> dict[str, Any] | None:
        """Load the specified model version (or active/latest model) from disk."""
        df = self.list_models()
        if df.empty:
            return None

        if model_version:
            matches = df[df["model_version"] == model_version]
            if matches.empty:
                return None
            row = matches.iloc[0]
        else:
            active = df[df["is_active"] == 1]
            row = active.iloc[0] if not active.empty else df.iloc[0]

        path = Path(row["model_path"])
        if not path.exists():
            return None
        bundle = joblib.load(path)
        bundle["db_id"] = int(row["id"])
        bundle["is_active"] = bool(row["is_active"])
        bundle["trained_at"] = row["trained_at"]
        bundle["dataset_name"] = row["dataset_name"]
        return bundle

    def load_all_bundles(self) -> list[dict[str, Any]]:
        """Load all available persisted model bundles."""
        df = self.list_models()
        bundles: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            path = Path(row["model_path"])
            if path.exists():
                b = joblib.load(path)
                b["db_id"] = int(row["id"])
                b["is_active"] = bool(row["is_active"])
                b["trained_at"] = row["trained_at"]
                b["dataset_name"] = row["dataset_name"]
                bundles.append(b)
        return bundles

    def set_active(self, model_version: str) -> None:
        """Mark a persisted model version as active."""
        self.db.set_active_model(model_version)

    def delete_model(self, model_version: str) -> bool:
        """Delete a model record from SQLite and remove its joblib file."""
        model_path = self.db.delete_model_run(model_version)
        if model_path:
            p = Path(model_path)
            if p.exists():
                p.unlink()
            return True
        return False

    def select_best_model(
        self,
        metric: str = "recall",
        split: str = "validation",
    ) -> dict[str, Any] | None:
        """Select the best model based on a configurable metric on validation (CV/OOF) or test."""
        bundles = self.load_all_bundles()
        if not bundles:
            return None

        metric_key = "f1_score" if metric == "f1" else metric

        def _score(b: dict[str, Any]) -> float:
            split_metrics = b.get("metrics_by_split", {}).get(split, {})
            return float(split_metrics.get(metric_key, 0.0))

        return max(bundles, key=_score)
