from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_app import main
from fraud_app.model import ModelManager, drift_report, ensemble_scores
from fraud_app.storage import save_transaction
from fraud_app.train import generate_synthetic_data, train_model

NORMAL = dict(amount=45.0, hour=14, distance_km=2.0, transactions_24h=2,
              account_age_days=1500, is_international=False)
SUSPICIOUS = dict(amount=8500.0, hour=2, distance_km=8000.0, transactions_24h=40,
                  account_age_days=5, is_international=True)


@pytest.fixture(scope="session")
def artifact(tmp_path_factory):
    return train_model(tmp_path_factory.mktemp("training"), n_samples=4000)


@pytest.fixture
def client(tmp_path, monkeypatch, artifact):
    monkeypatch.setenv("FRAUD_DATA_DIR", str(tmp_path))
    joblib.dump(artifact, tmp_path / "model.joblib")
    monkeypatch.setattr(main, "manager", ModelManager())
    with TestClient(main.app) as app:
        yield app


def test_untrained_and_corrupt_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "manager", ModelManager())
    with TestClient(main.app) as app:
        assert app.get("/api/health").json()["model_ready"] is False
        assert app.post("/api/predict", json=NORMAL).status_code == 503
        assert app.get("/api/dashboard").status_code == 503
        joblib.dump({"metadata": {"features": []}}, tmp_path / "model.joblib")
        assert app.get("/api/health").json()["model_ready"] is False


def test_training_reproducibility_and_holdout(artifact):
    x, y = generate_synthetic_data(1000, 42)
    x2, y2 = generate_synthetic_data(1000, 42)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    metadata = artifact["metadata"]
    assert metadata["train_count"] == 3000
    assert metadata["holdout_count"] == 1000
    assert metadata["training_source"] == "synthetic"
    assert all(0 <= n <= 1 for n in metadata["metrics"].values())
    assert metadata["metrics"]["roc_auc"] > .65


def test_predict_save_read_and_aggregate(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["model_ready"] is True
    first = client.post("/api/predict", json=NORMAL)
    second = client.post("/api/predict", json=SUSPICIOUS)
    assert first.status_code == second.status_code == 200
    a, b = first.json(), second.json()
    assert a["risk_score"] < b["risk_score"]
    assert a["decision"] == "approve"
    assert b["decision"] == "block"
    for record in (a, b):
        assert 0 <= record["risk_score"] <= 1
        assert record["latency_ms"] >= 0
    records = client.get("/api/transactions?days=1").json()["transactions"]
    assert {r["id"] for r in records} == {a["id"], b["id"]}
    summary = client.get("/api/dashboard?days=1").json()
    assert summary["total_transactions"] == 2
    assert summary["flagged_transactions"] == 1
    assert summary["blocked_amount"] == SUSPICIOUS["amount"]
    assert sum(r["total"] for r in summary["series"]) == 2
    assert summary["drift"]["status"] == "insufficient_data"


@pytest.mark.parametrize("field,value", [
    ("amount", 0), ("amount", -1), ("amount", 1000001), ("amount", "12"),
    ("hour", 24), ("hour", 1.5), ("hour", True), ("distance_km", -1),
    ("transactions_24h", 0), ("account_age_days", 20001), ("is_international", "yes"),
])
def test_invalid_input(client, field, value):
    payload = {**NORMAL, field: value}
    assert client.post("/api/predict", json=payload).status_code == 422
    assert client.get("/api/transactions").json()["transactions"] == []


def test_nonfinite_missing_unknown_and_query_validation(client):
    for value in ("NaN", "Infinity", "-Infinity"):
        assert client.post("/api/predict", content='{"amount":' + value + '}',
                           headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/api/predict", json={}).status_code == 422
    assert client.post("/api/predict", json={**NORMAL, "unknown": 1}).status_code == 422
    assert client.get("/api/dashboard?days=2").status_code == 422
    assert client.get("/api/transactions?limit=1001").status_code == 422


def test_time_filter_and_restart_persistence(client, tmp_path):
    record = client.post("/api/predict", json=NORMAL).json()
    old = {**record, "id": "old-record", "created_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}
    save_transaction(tmp_path, old)
    assert len(client.get("/api/transactions?days=1").json()["transactions"]) == 1
    assert len(client.get("/api/transactions?days=7").json()["transactions"]) == 2
    with TestClient(main.app) as restarted:
        assert len(restarted.get("/api/transactions?days=7").json()["transactions"]) == 2


def test_drift_and_repeatable_scoring(artifact):
    features, _ = generate_synthetic_data(2000, 42)
    stable = drift_report(artifact, features.tolist())
    assert stable["status"] == "stable"
    shifted = [[900000, 23, 19999, 999, 19999, 1]] * 60
    assert drift_report(artifact, shifted)["status"] == "drift_detected"
    assert drift_report(artifact, shifted[:49])["status"] == "insufficient_data"
    np.testing.assert_allclose(ensemble_scores(artifact, features[:10])[0], ensemble_scores(artifact, features[:10])[0])


def test_retrain_keeps_ledger(client):
    saved = client.post("/api/predict", json=NORMAL).json()
    response = client.post("/api/retrain")
    assert response.status_code == 200
    assert response.json()["model_version"] != saved["model_version"]
    assert client.get("/api/model").json()["version"] == response.json()["model_version"]
    assert client.get("/api/transactions").json()["transactions"][0]["id"] == saved["id"]


def test_retrain_failure_keeps_model(client, monkeypatch):
    before = client.get("/api/model").json()["version"]
    def fail(**kwargs):
        raise RuntimeError("test training failure")
    monkeypatch.setattr(main, "train_model", fail)
    assert client.post("/api/retrain").status_code == 500
    assert client.get("/api/model").json()["version"] == before
    assert client.post("/api/predict", json=NORMAL).status_code == 200
    main.manager.retrain_lock.acquire()
    try:
        assert client.post("/api/retrain").status_code == 409
    finally:
        main.manager.retrain_lock.release()


def test_concurrent_predictions_have_unique_durable_records(client):
    def score(_):
        response = client.post("/api/predict", json=NORMAL)
        assert response.status_code == 200
        return response.json()["id"]
    with ThreadPoolExecutor(max_workers=4) as workers:
        ids = list(workers.map(score, range(12)))
    assert len(set(ids)) == 12
    records = client.get("/api/transactions?days=1").json()["transactions"]
    assert {row["id"] for row in records} == set(ids)
    assert client.get("/api/dashboard?days=1").json()["total_transactions"] == 12


def test_private_files_not_exposed(client):
    for path in ("/data/model.joblib", "/data/transactions.sqlite3", "/.env", "/fraud_app/main.py"):
        assert client.get(path).status_code == 404
