"""Feature contract, local artifact loading, scoring and custom drift monitoring."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import joblib
import numpy as np

FEATURES = ["amount", "hour", "distance_km", "transactions_24h", "account_age_days", "is_international"]
CARD_FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
REVIEW_THRESHOLD = 0.45
BLOCK_THRESHOLD = 0.75
DRIFT_THRESHOLD = 0.2
DRIFT_MIN_SAMPLES = 50
ARTIFACT_NAME = "model.joblib"


def data_directory() -> Path:
    return Path(os.environ.get("FRAUD_DATA_DIR", "data")).expanduser().resolve()


def ensemble_scores(artifact: dict, features: np.ndarray):
    xgb = artifact["classifier"].predict_proba(features)[:, 1]
    raw = -artifact["isolation_forest"].score_samples(features)
    reference = artifact["anomaly_reference"]
    anomaly = np.searchsorted(reference, raw, side="right") / len(reference)
    risk = np.clip(0.85 * xgb + 0.15 * anomaly, 0, 1)
    return risk, xgb, anomaly


def model_metadata(artifact: dict) -> dict:
    return dict(artifact["metadata"])


def drift_report(artifact: dict, current: list[list[float]]) -> dict:
    count = len(current)
    result = {"status": "insufficient_data", "score": None, "sample_count": count,
              "threshold": DRIFT_THRESHOLD, "minimum_samples": DRIFT_MIN_SAMPLES,
              "method": "custom_mean_histogram_total_variation", "max_samples": 5000}
    if count < DRIFT_MIN_SAMPLES:
        return result
    values = np.asarray(current, dtype=float)
    distances = []
    for index, reference in enumerate(artifact["drift_reference"]):
        histogram, _ = np.histogram(values[:, index], bins=reference["bins"])
        distribution = histogram / count
        distances.append(float(np.abs(distribution - reference["probabilities"]).sum() / 2))
    score = float(np.mean(distances))
    result.update(status="drift_detected" if score >= DRIFT_THRESHOLD else "stable", score=score,
                  feature_scores=dict(zip(artifact["metadata"]["features"], distances)))
    return result


class ModelManager:
    """An in-flight prediction keeps its immutable model even during a reload."""

    def __init__(self):
        self._lock = threading.RLock()
        self.retrain_lock = threading.Lock()
        self._artifact = None
        self._directory = None
        self.load_error = None

    def load(self, directory: Path) -> bool:
        path = directory / ARTIFACT_NAME
        artifact = None
        error = None
        if path.is_file():
            try:
                # joblib is pickle-based. Only load artifacts from this trusted local
                # directory; there is deliberately no artifact upload or URL API.
                artifact = joblib.load(path)
                metadata = artifact["metadata"]
                compatible = (
                    artifact["schema_version"] == 1 and metadata["features"] == FEATURES
                    and metadata.get("training_source") == "synthetic"
                ) or (
                    artifact["schema_version"] == 2 and metadata["features"] == CARD_FEATURES
                    and metadata.get("training_source") == "real_creditcard"
                )
                if not compatible:
                    raise ValueError("Incompatible model artifact")
            except Exception as exc:
                artifact = None
                error = type(exc).__name__
        with self._lock:
            self._artifact = artifact
            self._directory = directory
            self.load_error = error
        return artifact is not None

    def snapshot(self, directory: Path):
        with self._lock:
            if self._directory != directory or (self._artifact is None and (directory / ARTIFACT_NAME).is_file()):
                self.load(directory)
            return self._artifact

    def replace(self, directory: Path, artifact: dict):
        with self._lock:
            self._artifact = artifact
            self._directory = directory
            self.load_error = None
