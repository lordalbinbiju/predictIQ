"""Tests for model training, evaluation splits, serialization, and SQLite persistence."""

from __future__ import annotations

from pathlib import Path

from database.database import DatabaseManager
from src.data_loader import load_sample_dataset
from src.model_manager import ModelManager
from src.model_training import get_available_algorithms, train_churn_model


def test_train_core_models_and_persist_registry(tmp_path: Path) -> None:
    df = load_sample_dataset().sample(n=600, random_state=42).reset_index(drop=True)
    db = DatabaseManager(db_path=tmp_path / "test_predictiq.db")
    models_dir = tmp_path / "saved_models"
    mgr = ModelManager(db=db, models_dir=models_dir)

    algos = get_available_algorithms()
    assert "Logistic Regression" in algos
    assert "Random Forest" in algos
    assert "Gradient Boosting" in algos

    trained_versions = []
    for algo in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
        res = train_churn_model(
            df=df,
            algorithm=algo,
            test_size=0.20,
            random_state=42,
            cv_folds=3,
            use_class_weight=True,
            models_dir=models_dir,
        )
        assert res.model_path.exists()
        for split in ("train", "validation", "test"):
            m = res.metrics_by_split[split]
            for metric_key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
                assert 0.0 <= m[metric_key] <= 1.0

        mgr.register_training_result(res, dataset_name="Telco Sample 600", set_active=True)
        trained_versions.append(res.model_version)

    listed = mgr.list_models()
    assert len(listed) == 3

    best = mgr.select_best_model(metric="recall", split="validation")
    assert best is not None
    assert best["model_version"] in trained_versions

    loaded = mgr.load_model_bundle(trained_versions[0])
    assert loaded is not None
    assert loaded["model_version"] == trained_versions[0]

    # Deleting a model removes both DB record and joblib file
    deleted = mgr.delete_model(trained_versions[0])
    assert deleted is True
    assert len(mgr.list_models()) == 2
