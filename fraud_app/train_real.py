"""Train on the historical ULB credit-card benchmark; never download implicitly."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

from .model import ARTIFACT_NAME, CARD_FEATURES, data_directory, ensemble_scores
from .train import threshold_metrics


def load_creditcard(csv_path):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Credit-card source CSV is missing: {path}. Place the ULB creditcard.csv there and retry.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), None)
    expected = [*CARD_FEATURES, "Class"]
    if header != expected:
        raise ValueError(f"CSV columns must be exactly {','.join(expected)} in that order")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError("Every CSV row must contain exactly 31 fields")
    if frame.empty or any(not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]) for c in expected):
        raise ValueError("CSV must contain nonempty numeric data in every column")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("CSV values must be finite; missing values are not allowed")
    if not frame["Class"].isin([0, 1]).all():
        raise ValueError("Class must be binary (0 or 1)")
    if (frame[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Time and Amount must be nonnegative")
    original_count = len(frame)
    duplicates = frame.duplicated(CARD_FEATURES, keep=False)
    if duplicates.any():
        groups = frame.loc[duplicates].groupby(CARD_FEATURES, sort=False)["Class"].nunique()
        if (groups > 1).any():
            raise ValueError("Identical feature rows have conflicting Class labels")
    frame = frame.drop_duplicates(CARD_FEATURES).sort_values("Time", kind="stable").reset_index(drop=True)
    return frame, digest.hexdigest(), original_count - len(frame), original_count


def chronological_splits(frame):
    times = frame["Time"].to_numpy()
    boundaries = np.flatnonzero(times[1:] != times[:-1]) + 1
    if len(boundaries) < 2:
        raise ValueError("At least three distinct Time groups are required for chronological splits")
    first = int(boundaries[:-1][np.argmin(np.abs(boundaries[:-1] - len(frame) * .6))])
    remaining = boundaries[boundaries > first]
    second = int(remaining[np.argmin(np.abs(remaining - len(frame) * .8))])
    splits = [frame.iloc[:first], frame.iloc[first:second], frame.iloc[second:]]
    for name, split in zip(("train", "validation", "test"), splits):
        if split["Class"].nunique() != 2:
            raise ValueError(f"Chronological {name} split must contain both legitimate and fraud labels")
    return splits


def train_real_model(csv_path, data_dir=None, seed=42):
    frame, sha256, duplicate_count, original_count = load_creditcard(csv_path)
    train, validation, test = chronological_splits(frame)
    x_train = train[CARD_FEATURES].to_numpy(dtype=float)
    y_train = train["Class"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    class_weight = float(np.sqrt(np.count_nonzero(y_train == 0) / np.count_nonzero(y_train == 1)))
    classifier = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.055,
                               subsample=.9, colsample_bytree=.9, reg_lambda=3,
                               scale_pos_weight=class_weight, objective="binary:logistic",
                               eval_metric="logloss", tree_method="hist", n_jobs=2, random_state=seed)
    classifier.fit(x_train, y_train)
    legitimate = x_train[y_train == 0]
    fit_sample = legitimate[rng.choice(len(legitimate), min(100000, len(legitimate)), replace=False)]
    calibration = legitimate[rng.choice(len(legitimate), min(30000, len(legitimate)), replace=False)]
    isolation = IsolationForest(n_estimators=120, contamination="auto", random_state=seed, n_jobs=2)
    isolation.fit(fit_sample)
    drift_reference = []
    for column in x_train.T:
        bins = np.concatenate(([-np.inf], np.unique(np.quantile(column, np.linspace(.1, .9, 9))), [np.inf]))
        counts, _ = np.histogram(column, bins=bins)
        drift_reference.append({"bins": bins, "probabilities": counts / len(x_train)})
    artifact = {"schema_version": 2, "classifier": classifier, "isolation_forest": isolation,
                "anomaly_reference": np.sort(-isolation.score_samples(calibration)), "drift_reference": drift_reference}
    val_labels = validation["Class"].to_numpy(dtype=int)
    val_scores, _, _ = ensemble_scores(artifact, validation[CARD_FEATURES].to_numpy(dtype=float))
    precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros_like(precision[:-1]), where=(precision[:-1] + recall[:-1]) > 0)
    # Deterministic tie rule: the lowest threshold attaining maximum validation F1.
    review = float(thresholds[np.argmax(f1)])
    block = max(.75, review)
    labels = test["Class"].to_numpy(dtype=int)
    scores, _, _ = ensemble_scores(artifact, test[CARD_FEATURES].to_numpy(dtype=float))
    review_metrics = threshold_metrics(labels, scores, review)
    metrics = {key: review_metrics[key] for key in ("precision", "recall", "f1")}
    metrics.update(roc_auc=float(roc_auc_score(labels, scores)),
                   average_precision=float(average_precision_score(labels, scores)),
                   confusion_matrix=[[review_metrics["true_negatives"], review_metrics["false_positives"]],
                                     [review_metrics["false_negatives"], review_metrics["true_positives"]]])
    examples = []
    for label in (0, 1):
        row = test.loc[test["Class"] == label].iloc[0]
        examples.append({"label": label, "features": {key: float(row[key]) for key in CARD_FEATURES}})
    trained_at = datetime.now(timezone.utc).isoformat()
    version = hashlib.sha256(f"{sha256}:{seed}:{trained_at}".encode()).hexdigest()[:10]
    artifact["metadata"] = {
        "schema_version": 2, "version": f"real-creditcard-v2-{version}", "trained_at": trained_at,
        "training_source": "real_creditcard", "features": CARD_FEATURES, "metrics": metrics,
        "thresholds": {"review": review, "block": block},
        "threshold_metrics": {"review": review_metrics, "block": threshold_metrics(labels, scores, block)},
        "validation": {"chosen_threshold": review, "selection": "maximum ensemble F1; lowest threshold breaks ties",
                       "metrics": threshold_metrics(val_labels, val_scores, review)},
        "dataset": {"sha256": sha256, "filename": Path(csv_path).name,
                    "source": "ULB Machine Learning Group / Worldline: European cardholder transactions, September 2013",
                    "source_url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
                    "mirror_url": "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
                    "original_count": original_count, "duplicate_count": duplicate_count},
        "duplicate_count": duplicate_count, "dataset_sha256": sha256,
        "splits": {name: {"count": len(part), "class_counts": {str(label): int((part["Class"] == label).sum()) for label in (0, 1)},
                           "time_min": float(part["Time"].min()), "time_max": float(part["Time"].max())}
                   for name, part in zip(("train", "validation", "test"), (train, validation, test))},
        "seed": seed, "sample_count": len(frame), "train_count": len(train), "validation_count": len(validation),
        "holdout_count": len(test), "train_fraud_rate": float(train["Class"].mean()), "holdout_fraud_rate": float(test["Class"].mean()),
        "training": {"scale_pos_weight": class_weight, "isolation_fit_count": len(fit_sample), "anomaly_calibration_count": len(calibration)},
        "ensemble": {"xgboost_weight": .85, "isolation_forest_weight": .15,
                     "anomaly_normalization": "percentile against bounded training legitimate sample"},
        "evaluation": "Feature-deduplicated chronological approximately 60/20/20 split; tied Time groups never cross boundaries. Training-only fitting; validation-only threshold selection; metrics on untouched test.",
        "limitations": "Historical public benchmark, not live production validation. V1–V28 are supplied PCA components; the original transformation is unavailable, so raw card transactions cannot be transformed here. Time and Amount are used as supplied. Scores are not calibrated fraud probabilities; drift on a historical Time coordinate may reflect elapsed time.",
        "drift_method": "Custom mean feature histogram total variation; training reference; minimum 50 current samples.",
        "example_selection": "First legitimate and first fraud row in chronological test order, without score-based selection. Labels are outside inference features.",
    }
    artifact["examples"] = examples
    directory = Path(data_dir) if data_dir is not None else data_directory()
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="model-", suffix=".joblib", dir=directory)
    os.close(fd)
    try:
        joblib.dump(artifact, temporary)
        (directory / f"metrics-{artifact['metadata']['version']}.json").write_text(
            json.dumps(artifact["metadata"], indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, directory / ARTIFACT_NAME)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    artifact = train_real_model(args.csv, args.data_dir, args.seed)
    print(json.dumps(artifact["metadata"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
