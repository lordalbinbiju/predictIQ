"""Tests for individual/batch predictions, SHAP explainability, EDA, and business insights."""

from __future__ import annotations

from pathlib import Path

import pytest

from database.database import DatabaseManager
from src.business_insights import compute_business_intelligence
from src.data_loader import load_sample_dataset
from src.eda import generate_eda_insights
from src.explainability import (
    compute_global_shap,
    compute_individual_shap,
    generate_individual_shap_narrative,
)
from src.model_manager import ModelManager
from src.model_training import train_churn_model
from src.prediction import predict_batch, predict_single_customer
from src.preprocessing import prepare_modeling_data


@pytest.fixture()
def trained_bundle(tmp_path: Path):
    df = load_sample_dataset().sample(n=500, random_state=7).reset_index(drop=True)
    db = DatabaseManager(db_path=tmp_path / "pred_test.db")
    mgr = ModelManager(db=db, models_dir=tmp_path / "models")
    res = train_churn_model(
        df=df,
        algorithm="Random Forest",
        test_size=0.20,
        random_state=7,
        cv_folds=3,
        models_dir=tmp_path / "models",
        hyperparameters={"n_estimators": 40, "max_depth": 6},
    )
    mgr.register_training_result(res, dataset_name="Telco Sample 500", set_active=True)
    bundle = mgr.load_model_bundle(res.model_version)
    return df, bundle, db


def test_single_and_batch_prediction_and_db_logging(trained_bundle) -> None:
    df, bundle, db = trained_bundle
    sample_row = df.iloc[0].drop(labels=["customerID", "Churn"]).to_dict()

    out = predict_single_customer(bundle, sample_row, threshold=0.50)
    assert out["predicted_class"] in (0, 1)
    assert out["predicted_label"] in ("Churn", "Retained")
    assert 0.0 <= out["churn_probability"] <= 1.0
    assert out["risk_category"] in ("Low Risk", "Medium Risk", "High Risk")

    # Threshold boundary test
    out_strict = predict_single_customer(bundle, sample_row, threshold=0.95)
    if out["churn_probability"] < 0.95:
        assert out_strict["predicted_class"] == 0

    db.save_prediction(
        customer_id="CUST-TEST-1",
        model_version=bundle["model_version"],
        predicted_class=out["predicted_class"],
        predicted_label=out["predicted_label"],
        churn_probability=out["churn_probability"],
        risk_category=out["risk_category"],
        threshold=0.50,
        input_features=sample_row,
    )
    hist = db.get_prediction_history()
    assert len(hist) == 1
    assert hist.iloc[0]["customer_id"] == "CUST-TEST-1"

    batch_df = df.head(15).drop(columns=["Churn"])
    scored_df, _ = predict_batch(bundle, batch_df, threshold=0.50)
    assert len(scored_df) == 15
    assert set(["Customer ID", "Predicted Churn", "Churn Probability", "Risk Category"]).issubset(
        scored_df.columns
    )

    # Missing required feature raises ValueError cleanly
    with pytest.raises(ValueError, match="Missing required feature"):
        predict_single_customer(bundle, {"gender": "Female"})


def test_shap_global_and_individual_explanations(trained_bundle) -> None:
    df, bundle, _ = trained_bundle
    X, _, _, _, _ = prepare_modeling_data(df)

    global_shap = compute_global_shap(bundle, X, max_rows=60)
    assert global_shap["shap_matrix"].shape[0] == 60
    assert not global_shap["importance_df"].empty

    local_shap = compute_individual_shap(bundle, X.iloc[[0]])
    assert len(local_shap["shap_values"]) == len(local_shap["feature_names"])

    narrative = generate_individual_shap_narrative(local_shap)
    assert "churn probability" in narrative["narrative"].lower()
    assert len(narrative["increasing_factors"]) + len(narrative["decreasing_factors"]) > 0


def test_eda_insights_and_business_intelligence(trained_bundle) -> None:
    df, bundle, _ = trained_bundle
    insights = generate_eda_insights(df, "Churn")
    assert len(insights) >= 4
    assert any("churn rate" in s.lower() for s in insights)

    bi = compute_business_intelligence(df, "Churn", bundle=bundle)
    assert bi["scored_with_model"] is True
    assert bi["total_customers"] == len(df)
    assert bi["high_risk_monthly_revenue"] >= 0.0
    assert len(bi["recommendations"]) >= 1
