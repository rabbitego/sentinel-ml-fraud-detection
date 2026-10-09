"""Small generated fixtures test the real-data pipeline; not benchmark evidence."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_app import main
from fraud_app.model import CARD_FEATURES, ModelManager
from fraud_app.train_real import chronological_splits, load_creditcard, train_real_model


def fixture_frame():
    rng = np.random.default_rng(101)
    frame = pd.DataFrame(rng.normal(size=(600, 30)), columns=CARD_FEATURES)
    frame["Time"] = np.repeat(np.arange(300), 2)
    frame["Amount"] = rng.uniform(0, 200, 600)
    frame["Class"] = (np.arange(600) % 10 == 0).astype(int)
    frame["V1"] += frame["Class"] * 4
    return frame


@pytest.fixture(scope="module")
def real_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("real-schema-fixture")
    csv = directory / "creditcard.csv"
    fixture_frame().to_csv(csv, index=False)
    artifact = train_real_model(csv, data_dir=directory)
    return artifact


@pytest.fixture
def card_client(tmp_path, monkeypatch, real_fixture):
    monkeypatch.setenv("FRAUD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "manager", ModelManager())
    joblib.dump(real_fixture, tmp_path / "model.joblib")
    with TestClient(main.app) as client:
        yield client


def test_deduplication_and_chronological_boundaries(tmp_path):
    frame = fixture_frame()
    path = tmp_path / "card.csv"
    pd.concat([frame, frame.iloc[:2]]).to_csv(path, index=False)
    clean, sha, duplicate_count, original_count = load_creditcard(path)
    assert duplicate_count == 2 and original_count == 602 and len(sha) == 64
    train, validation, test = chronological_splits(clean)
    assert train.Time.max() < validation.Time.min() < test.Time.min()
    assert validation.Time.max() < test.Time.min()
    assert sum(map(len, (train, validation, test))) == len(clean)
    assert not set(train.Time) & set(validation.Time)
    assert not set(validation.Time) & set(test.Time)


def test_conflicting_labels_are_rejected(tmp_path):
    frame = fixture_frame()
    conflict = frame.iloc[:1].copy()
    conflict["Class"] = 1 - conflict["Class"]
    path = tmp_path / "card.csv"
    pd.concat([frame, conflict]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting"):
        load_creditcard(path)


@pytest.mark.parametrize("bad", ["missing", "label", "nan", "negative"])
def test_invalid_csv_fails_before_training(tmp_path, bad):
    frame = fixture_frame()
    if bad == "missing":
        frame = frame.drop(columns="V28")
    elif bad == "label":
        frame.loc[0, "Class"] = 2
    elif bad == "nan":
        frame.loc[0, "V1"] = np.nan
    else:
        frame.loc[0, "Amount"] = -1
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_creditcard(path)


def test_real_input_prediction_persistence_and_source_isolation(card_client):
    client = card_client
    examples = client.get("/api/examples").json()
    assert examples["source"] == "heldout"
    assert {e["label"] for e in examples["examples"]} == {0, 1}
    for example in examples["examples"]:
        features = example["features"]
        assert set(features) == set(CARD_FEATURES)
        assert "Class" not in features
        response = client.post("/api/predict/card", json=features)
        assert response.status_code == 200
        result = response.json()
        assert result["features"] == features
        assert result["amount"] == features["Amount"]
        assert "distance_km" not in result and "account_age_days" not in result
        assert 0 <= result["risk_score"] <= 1
    rows = client.get("/api/transactions").json()["transactions"]
    assert len(rows) == 2 and all(r["training_source"] == "real_creditcard" for r in rows)
    dashboard = client.get("/api/dashboard").json()
    assert dashboard["total_transactions"] == 2
    assert dashboard["drift"]["sample_count"] == 2
    assert dashboard["model"]["training_source"] == "real_creditcard"
    synthetic = dict(amount=1, hour=1, distance_km=0, transactions_24h=1, account_age_days=1, is_international=False)
    assert client.post("/api/predict", json=synthetic).status_code == 409


@pytest.mark.parametrize("key,value", [("Class", 1), ("Time", -1), ("Amount", -1), ("V1", True), ("V2", "1"), ("V3", None)])
def test_card_strict_validation(card_client, key, value):
    features = card_client.get("/api/examples").json()["examples"][0]["features"]
    features[key] = value
    assert card_client.post("/api/predict/card", json=features).status_code == 422
    assert card_client.get("/api/transactions").json()["transactions"] == []


def test_real_nonfinite_validation(card_client):
    features = card_client.get("/api/examples").json()["examples"][0]["features"]
    for invalid in (float("nan"), float("inf"), -float("inf")):
        features["V1"] = invalid
        assert card_client.post("/api/predict/card", content=json.dumps(features), headers={"Content-Type": "application/json"}).status_code == 422


def test_real_retrain_missing_source_does_not_revert_to_synthetic(card_client):
    before = card_client.get("/api/model").json()
    assert card_client.post("/api/retrain").status_code == 409
    after = card_client.get("/api/model").json()
    assert after["version"] == before["version"]
    assert after["training_source"] == "real_creditcard"


def test_threshold_and_examples_use_correct_partitions(real_fixture):
    metadata = real_fixture["metadata"]
    assert metadata["thresholds"]["review"] == metadata["validation"]["chosen_threshold"]
    assert metadata["thresholds"]["block"] >= metadata["thresholds"]["review"]
    assert "average_precision" in metadata["metrics"]
    test_start = metadata["splits"]["test"]["time_min"]
    assert all(e["features"]["Time"] >= test_start for e in real_fixture["examples"])
    matrix = metadata["metrics"]["confusion_matrix"]
    assert sum(map(sum, matrix)) == metadata["holdout_count"]


def test_real_retraining_preserves_real_ledger(card_client, tmp_path):
    fixture_frame().to_csv(tmp_path / "creditcard.csv", index=False)
    example = card_client.get("/api/examples").json()["examples"][0]["features"]
    saved = card_client.post("/api/predict/card", json=example).json()
    result = card_client.post("/api/retrain")
    assert result.status_code == 200
    metadata = card_client.get("/api/model").json()
    assert metadata["training_source"] == "real_creditcard"
    assert metadata["version"] != saved["model_version"]
    rows = card_client.get("/api/transactions").json()["transactions"]
    assert rows[0]["id"] == saved["id"]
    assert rows[0]["model_version"] == saved["model_version"]


def test_real_drift_uses_all_thirty_features(card_client):
    example = card_client.get("/api/examples").json()["examples"][0]["features"]
    for _ in range(50):
        assert card_client.post("/api/predict/card", json=example).status_code == 200
    drift = card_client.get("/api/dashboard").json()["drift"]
    assert drift["sample_count"] == 50
    assert drift["status"] == "drift_detected"
    assert set(drift["feature_scores"]) == set(CARD_FEATURES)
