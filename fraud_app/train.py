"""Explicit reproducible training: python -m fraud_app.train."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .model import ARTIFACT_NAME, BLOCK_THRESHOLD, FEATURES, REVIEW_THRESHOLD, data_directory, ensemble_scores


def generate_synthetic_data(n_samples=12000, seed=42):
    """Stochastic labels depend on features; no feature directly reveals a label."""
    if n_samples < 1000:
        raise ValueError("At least 1000 samples are required")
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(4.7, 1.5, n_samples), 0.01, 1000000)
    hour = rng.integers(0, 24, n_samples)
    international = rng.random(n_samples) < 0.2
    distance = np.clip(rng.exponential(65, n_samples) + international * rng.exponential(1600, n_samples), 0, 20000)
    velocity = np.clip(1 + rng.negative_binomial(2, 0.22, n_samples), 1, 1000)
    age = np.clip(rng.exponential(850, n_samples).astype(int), 0, 20000)
    logit = (-4.4 + 0.65 * np.log1p(amount / 180) + 1.05 * (hour < 5)
             + 0.9 * international + 0.00045 * distance + 0.105 * velocity
             + 1.55 * (age < 60) + 0.65 * ((amount > 1200) & international))
    probability = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    labels = (rng.random(n_samples) < probability).astype(int)
    features = np.column_stack([amount, hour, distance, velocity, age, international]).astype(float)
    return features, labels


def threshold_metrics(labels, scores, threshold):
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"threshold": threshold, "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "accuracy": float(accuracy_score(labels, predicted)),
            "true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp)}


def train_model(data_dir=None, seed=42, n_samples=12000):
    directory = Path(data_dir) if data_dir is not None else data_directory()
    directory.mkdir(parents=True, exist_ok=True)
    features, labels = generate_synthetic_data(n_samples, seed)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=seed, stratify=labels)
    classifier = XGBClassifier(n_estimators=160, max_depth=4, learning_rate=0.055,
                               subsample=0.9, colsample_bytree=0.9, reg_lambda=3,
                               objective="binary:logistic", eval_metric="logloss",
                               tree_method="hist", n_jobs=1, random_state=seed)
    classifier.fit(x_train, y_train)
    isolation = IsolationForest(n_estimators=120, contamination="auto", random_state=seed, n_jobs=1)
    isolation.fit(x_train[y_train == 0])
    anomaly_reference = np.sort(-isolation.score_samples(x_train[y_train == 0]))
    drift_reference = []
    for index in range(len(FEATURES)):
        if FEATURES[index] == "is_international":
            bins = np.array([-np.inf, 0.5, np.inf])
        else:
            interior = np.unique(np.quantile(x_train[:, index], np.linspace(0.1, 0.9, 9)))
            bins = np.concatenate(([-np.inf], interior, [np.inf]))
        counts, _ = np.histogram(x_train[:, index], bins=bins)
        drift_reference.append({"bins": bins, "probabilities": counts / len(x_train)})
    artifact = {"schema_version": 1, "classifier": classifier, "isolation_forest": isolation,
                "anomaly_reference": anomaly_reference, "drift_reference": drift_reference}
    risk, _, _ = ensemble_scores(artifact, x_test)
    review_metrics = threshold_metrics(y_test, risk, REVIEW_THRESHOLD)
    metrics = {key: review_metrics[key] for key in ("precision", "recall", "f1")}
    metrics["roc_auc"] = float(roc_auc_score(y_test, risk))
    trained_at = datetime.now(timezone.utc).isoformat()
    fingerprint = hashlib.sha256(f"{seed}:{n_samples}:{trained_at}".encode()).hexdigest()[:10]
    artifact["metadata"] = {
        "version": f"synthetic-v1-{fingerprint}", "trained_at": trained_at, "features": FEATURES,
        "training_source": "synthetic", "metrics": metrics,
        "threshold_metrics": {"review": review_metrics, "block": threshold_metrics(y_test, risk, BLOCK_THRESHOLD)},
        "thresholds": {"review": REVIEW_THRESHOLD, "block": BLOCK_THRESHOLD},
        "ensemble": {"xgboost_weight": 0.85, "isolation_forest_weight": 0.15,
                     "anomaly_normalization": "percentile against training legitimate transactions"},
        "seed": seed, "sample_count": n_samples, "train_count": len(x_train), "holdout_count": len(x_test),
        "train_fraud_rate": float(y_train.mean()), "holdout_fraud_rate": float(y_test.mean()),
        "evaluation": "Untouched stratified 25% synthetic holdout; fixed thresholds, no holdout tuning.",
        "limitations": "Synthetic demonstration only. Scores are not calibrated fraud probabilities; metrics do not establish real-world performance.",
        "drift_method": "Custom mean feature histogram total variation, not Evidently; minimum 50 current samples."
    }
    # A complete single artifact contains both estimators, references and metrics.
    # Atomic replacement lets concurrent readers see either complete generation.
    fd, temporary = tempfile.mkstemp(prefix="model-", suffix=".joblib", dir=directory)
    os.close(fd)
    try:
        joblib.dump(artifact, temporary)
        # Version-specific human-readable metrics cannot collide across runs.
        (directory / f"metrics-{artifact['metadata']['version']}.json").write_text(
            json.dumps(artifact["metadata"], indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, directory / ARTIFACT_NAME)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return artifact


def main():
    parser = argparse.ArgumentParser(description="Train the local synthetic fraud demonstration (not production fraud data).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    artifact = train_model(args.data_dir, args.seed, args.samples)
    print(json.dumps(artifact["metadata"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
