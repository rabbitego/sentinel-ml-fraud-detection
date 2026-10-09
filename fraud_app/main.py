"""Serve with uvicorn fraud_app.main:app. Training is always explicit."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, create_model

from .model import (CARD_FEATURES, FEATURES, ModelManager,
                    data_directory, drift_report, ensemble_scores, model_metadata)
from .storage import dashboard_data, database, list_transactions, save_transaction
from .train import train_model
from .train_real import train_real_model

logger = logging.getLogger(__name__)
manager = ModelManager()
ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    directory = data_directory()
    manager.load(directory)
    with database(directory):
        pass
    yield


app = FastAPI(title="Fraud Signal — local ML benchmark", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Pydantic's raw error input may itself contain NaN/infinity. Do not echo
    # non-JSON numbers back through Starlette's strict JSON serializer.
    details = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    amount: float = Field(gt=0, le=1000000, strict=True)
    hour: int = Field(ge=0, le=23, strict=True)
    distance_km: float = Field(ge=0, le=20000, strict=True)
    transactions_24h: int = Field(ge=1, le=1000, strict=True)
    account_age_days: int = Field(ge=0, le=20000, strict=True)
    is_international: StrictBool


CardInput = create_model(
    "CardInput", __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
    **{key: (float, Field(strict=True, **({"ge": 0} if key in ("Time", "Amount") else {})))
       for key in CARD_FEATURES},
)


def require_model():
    artifact = manager.snapshot(data_directory())
    if artifact is None:
        raise HTTPException(503, "Model unavailable. Train the benchmark with python -m fraud_app.train_real --csv data/creditcard.csv --data-dir data, or explicitly train the synthetic demonstration with python -m fraud_app.train.")
    return artifact


def validate_days(days: int):
    if days not in (1, 7, 30):
        raise HTTPException(422, "days must be one of 1, 7, or 30")


@app.get("/", include_in_schema=False)
def index():
    path = ROOT / "index.html"
    if not path.is_file():
        raise HTTPException(404, "Dashboard index.html not found")
    return FileResponse(path, media_type="text/html")


@app.get("/api/health")
def health():
    ready = manager.snapshot(data_directory()) is not None
    return {"status": "ok" if ready else "model_not_ready", "model_ready": ready}


@app.post("/api/predict")
def predict(transaction: TransactionInput):
    started = time.perf_counter()
    artifact = require_model()
    if artifact["metadata"]["training_source"] != "synthetic":
        raise HTTPException(409, "The active model uses real credit-card features. Use POST /api/predict/card.")
    values = transaction.model_dump()
    return score_record(artifact, values, started)


def score_record(artifact, values, started):
    metadata = artifact["metadata"]
    source = metadata["training_source"]
    features = np.array([[values[key] for key in metadata["features"]]], dtype=float)
    risk, xgb, anomaly = ensemble_scores(artifact, features)
    score = float(risk[0])
    thresholds = metadata["thresholds"]
    decision = "block" if score >= thresholds["block"] else "review" if score >= thresholds["review"] else "approve"
    record = {"id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
              "amount": values["Amount"] if source == "real_creditcard" else values["amount"],
              "training_source": source, "features": values,
              "risk_score": score, "xgb_score": float(xgb[0]), "anomaly_score": float(anomaly[0]),
              "decision": decision, "latency_ms": (time.perf_counter() - started) * 1000,
              "model_version": artifact["metadata"]["version"]}
    if source == "synthetic":
        record.update(values)
    save_transaction(data_directory(), record)
    return record


@app.post("/api/predict/card")
def predict_card(transaction: CardInput):
    started = time.perf_counter()
    artifact = require_model()
    if artifact["metadata"]["training_source"] != "real_creditcard":
        raise HTTPException(409, "The active model is synthetic. Train the real credit-card model before using this endpoint.")
    return score_record(artifact, transaction.model_dump(), started)


@app.get("/api/examples")
def examples():
    artifact = require_model()
    if artifact["metadata"]["training_source"] != "real_creditcard":
        raise HTTPException(409, "Held-out credit-card examples require the real credit-card model.")
    return {"source": "heldout", "examples": artifact["examples"]}


@app.get("/api/transactions")
def transactions(days: int = 7, limit: Annotated[int, Query(ge=1, le=1000)] = 100):
    validate_days(days)
    artifact = manager.snapshot(data_directory())
    source = artifact["metadata"]["training_source"] if artifact is not None else "synthetic"
    return {"transactions": list_transactions(data_directory(), days, limit, source)}


@app.get("/api/model")
def model():
    return model_metadata(require_model())


@app.get("/api/dashboard")
def dashboard(days: int = 7):
    validate_days(days)
    artifact = require_model()
    metadata = model_metadata(artifact)
    summary, current = dashboard_data(data_directory(), days, metadata["training_source"], metadata["features"])
    summary["model"] = {key: metadata[key] for key in ("version", "trained_at", "metrics", "training_source")}
    summary["drift"] = drift_report(artifact, current)
    return summary


@app.post("/api/retrain")
def retrain():
    if not manager.retrain_lock.acquire(blocking=False):
        raise HTTPException(409, "A retraining operation is already in progress")
    try:
        directory = data_directory()
        active = manager.snapshot(directory)
        if active is not None and active["metadata"]["training_source"] == "real_creditcard":
            csv_path = directory / "creditcard.csv"
            if not csv_path.is_file():
                raise HTTPException(409, f"Real model retraining requires the local source at {csv_path}. Restore creditcard.csv and retry; the active model remains available.")
            artifact = train_real_model(csv_path, data_dir=directory)
        else:
            artifact = train_model(data_dir=directory)
        manager.replace(directory, artifact)
        return {"status": "completed", "model_version": artifact["metadata"]["version"]}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Model retraining failed")
        raise HTTPException(500, "Training failed; any existing loaded model remains available.") from None
    finally:
        manager.retrain_lock.release()
