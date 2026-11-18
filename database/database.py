"""SQLite persistence layer for datasets, model training runs, metrics, and predictions."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from utils.config import DB_PATH


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class DatabaseManager:
    """Manages SQLite persistence for PredictIQ metadata, models, and predictions."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    col_count INTEGER NOT NULL,
                    missing_count INTEGER NOT NULL,
                    duplicate_count INTEGER NOT NULL,
                    target_column TEXT NOT NULL,
                    churn_rate REAL NOT NULL,
                    uploaded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL UNIQUE,
                    algorithm TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    test_size REAL NOT NULL,
                    random_state INTEGER NOT NULL,
                    use_class_weight INTEGER NOT NULL,
                    hyperparameters_json TEXT NOT NULL,
                    cv_folds INTEGER NOT NULL,
                    cv_roc_auc_mean REAL,
                    cv_roc_auc_std REAL,
                    cv_recall_mean REAL,
                    cv_f1_mean REAL,
                    model_path TEXT NOT NULL,
                    feature_names_json TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    trained_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_run_id INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    roc_auc REAL NOT NULL,
                    confusion_matrix_json TEXT NOT NULL,
                    FOREIGN KEY (model_run_id) REFERENCES model_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS customer_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    predicted_class INTEGER NOT NULL,
                    predicted_label TEXT NOT NULL,
                    churn_probability REAL NOT NULL,
                    risk_category TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    input_features_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    total_customers INTEGER NOT NULL,
                    predicted_churn_count INTEGER NOT NULL,
                    high_risk_count INTEGER NOT NULL,
                    avg_probability REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def record_dataset(
        self,
        name: str,
        source: str,
        row_count: int,
        col_count: int,
        missing_count: int,
        duplicate_count: int,
        target_column: str,
        churn_rate: float,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO datasets (
                    name, source, row_count, col_count, missing_count,
                    duplicate_count, target_column, churn_rate, uploaded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    source,
                    int(row_count),
                    int(col_count),
                    int(missing_count),
                    int(duplicate_count),
                    target_column,
                    float(churn_rate),
                    _utc_now_iso(),
                ),
            )
            return int(cursor.lastrowid)

    def get_datasets(self, limit: int = 50) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM datasets ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )

    def save_model_run(
        self,
        model_version: str,
        algorithm: str,
        dataset_name: str,
        test_size: float,
        random_state: int,
        use_class_weight: bool,
        hyperparameters: dict[str, Any],
        cv_folds: int,
        cv_roc_auc_mean: float,
        cv_roc_auc_std: float,
        cv_recall_mean: float,
        cv_f1_mean: float,
        model_path: str,
        feature_names: list[str],
        metrics_by_split: dict[str, dict[str, Any]],
        set_active: bool = False,
    ) -> int:
        with self._connect() as conn:
            if set_active:
                conn.execute("UPDATE model_runs SET is_active = 0")

            cursor = conn.execute(
                """
                INSERT INTO model_runs (
                    model_version, algorithm, dataset_name, test_size, random_state,
                    use_class_weight, hyperparameters_json, cv_folds,
                    cv_roc_auc_mean, cv_roc_auc_std, cv_recall_mean, cv_f1_mean,
                    model_path, feature_names_json, is_active, trained_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_version,
                    algorithm,
                    dataset_name,
                    float(test_size),
                    int(random_state),
                    1 if use_class_weight else 0,
                    json.dumps(hyperparameters),
                    int(cv_folds),
                    float(cv_roc_auc_mean),
                    float(cv_roc_auc_std),
                    float(cv_recall_mean),
                    float(cv_f1_mean),
                    str(model_path),
                    json.dumps(feature_names),
                    1 if set_active else 0,
                    _utc_now_iso(),
                ),
            )
            run_id = int(cursor.lastrowid)

            for split, m in metrics_by_split.items():
                conn.execute(
                    """
                    INSERT INTO model_metrics (
                        model_run_id, split, accuracy, precision_score,
                        recall, f1_score, roc_auc, confusion_matrix_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        split,
                        float(m["accuracy"]),
                        float(m["precision"]),
                        float(m["recall"]),
                        float(m["f1_score"]),
                        float(m["roc_auc"]),
                        json.dumps(m["confusion_matrix"]),
                    ),
                )
            return run_id

    def set_active_model(self, model_version: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE model_runs SET is_active = 0")
            conn.execute(
                "UPDATE model_runs SET is_active = 1 WHERE model_version = ?",
                (model_version,),
            )

    def get_model_runs_with_test_metrics(self) -> pd.DataFrame:
        query = """
            SELECT
                r.id,
                r.model_version,
                r.algorithm,
                r.dataset_name,
                r.test_size,
                r.random_state,
                r.use_class_weight,
                r.hyperparameters_json,
                r.cv_folds,
                r.cv_roc_auc_mean,
                r.cv_roc_auc_std,
                r.cv_recall_mean,
                r.cv_f1_mean,
                r.model_path,
                r.feature_names_json,
                r.is_active,
                r.trained_at,
                m.accuracy AS test_accuracy,
                m.precision_score AS test_precision,
                m.recall AS test_recall,
                m.f1_score AS test_f1,
                m.roc_auc AS test_roc_auc,
                m.confusion_matrix_json AS test_confusion_matrix
            FROM model_runs r
            LEFT JOIN model_metrics m
                ON r.id = m.model_run_id AND m.split = 'test'
            ORDER BY r.id DESC
        """
        with self._connect() as conn:
            return pd.read_sql_query(query, conn)

    def get_metrics_for_run(self, model_run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM model_metrics WHERE model_run_id = ? ORDER BY id ASC",
                conn,
                params=(model_run_id,),
            )

    def delete_model_run(self, model_version: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT model_path FROM model_runs WHERE model_version = ?",
                (model_version,),
            ).fetchone()
            model_path = row["model_path"] if row else None
            conn.execute(
                "DELETE FROM model_runs WHERE model_version = ?",
                (model_version,),
            )
            return model_path

    def save_prediction(
        self,
        customer_id: str,
        model_version: str,
        predicted_class: int,
        predicted_label: str,
        churn_probability: float,
        risk_category: str,
        threshold: float,
        input_features: dict[str, Any],
        source: str = "individual",
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO customer_predictions (
                    customer_id, model_version, predicted_class, predicted_label,
                    churn_probability, risk_category, threshold,
                    input_features_json, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(customer_id),
                    model_version,
                    int(predicted_class),
                    predicted_label,
                    float(churn_probability),
                    risk_category,
                    float(threshold),
                    json.dumps(input_features),
                    source,
                    _utc_now_iso(),
                ),
            )
            return int(cursor.lastrowid)

    def save_batch_predictions(
        self,
        batch_name: str,
        model_version: str,
        records: list[dict[str, Any]],
        threshold: float,
    ) -> int:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO customer_predictions (
                    customer_id, model_version, predicted_class, predicted_label,
                    churn_probability, risk_category, threshold,
                    input_features_json, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'batch', ?)
                """,
                [
                    (
                        str(rec["customer_id"]),
                        model_version,
                        int(rec["predicted_class"]),
                        rec["predicted_label"],
                        float(rec["churn_probability"]),
                        rec["risk_category"],
                        float(threshold),
                        json.dumps(rec.get("input_features", {})),
                        now,
                    )
                    for rec in records
                ],
            )
            total = len(records)
            churn_cnt = sum(1 for r in records if int(r["predicted_class"]) == 1)
            high_cnt = sum(1 for r in records if r["risk_category"] == "High Risk")
            avg_prob = (
                sum(float(r["churn_probability"]) for r in records) / total
                if total > 0
                else 0.0
            )
            cursor = conn.execute(
                """
                INSERT INTO prediction_history (
                    batch_name, model_version, total_customers,
                    predicted_churn_count, high_risk_count, avg_probability, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_name,
                    model_version,
                    total,
                    churn_cnt,
                    high_cnt,
                    avg_prob,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def get_prediction_history(self, limit: int = 200) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT id, customer_id, model_version, predicted_class, predicted_label,
                       churn_probability, risk_category, threshold, source, created_at
                FROM customer_predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                conn,
                params=(limit,),
            )

    def get_batch_history(self, limit: int = 50) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
