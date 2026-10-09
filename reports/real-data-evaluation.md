# Real-data training and verification

## Result

The active model is `real-creditcard-v2-dfa8a1d957`, trained at
`2026-10-09T19:02:39.414773+00:00` (2026-10-10 in Asia/Kolkata).
The app is served locally on http://localhost:8001 and its API docs at `/docs`.

## Dataset provenance

- ULB Machine Learning Group / Worldline Credit Card Fraud Detection benchmark.
- Public historical European cardholder transactions (September 2013), not synthetic.
- Provider: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Mirror: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
- Mirror verified against the download used in TensorFlow's official tutorial source:
  https://github.com/tensorflow/docs/blob/master/site/en/tutorials/structured_data/imbalanced_data.ipynb
- Download: 150,828,752 bytes; MD5 `e90efcb83d69faf99fcab8b0255024de`.
- SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
- Original rows: 284,807, including 492 frauds. Feature-duplicate rows removed: 1,081.
- Raw data remains in ignored `data/creditcard.csv`. Check provider terms before redistribution.

## Evaluation protocol

The CSV's exact numeric schema is validated. NaNs, infinities, negative Time/Amount,
invalid labels and conflicting duplicate labels are rejected. The Class column is
never passed to inference. Deduplication occurs before chronological splitting.

| Partition | Rows | Legitimate | Fraud | Time range (dataset seconds) |
|---|---:|---:|---:|---|
| Training | 170,236 | 169,894 | 342 | 0–120,393 |
| Validation | 56,744 | 56,687 | 57 | 120,394–145,233 |
| Test | 56,746 | 56,672 | 74 | 145,234–172,792 |

Equal-Time groups never cross partitions. XGBoost and Isolation Forest are fitted
on training data only. Isolation Forest fits 100,000 legitimate training rows and
normalizes anomaly scores against 30,000 legitimate training rows. XGBoost uses
200 depth-4 trees, seed 42, and square-root training class-ratio weighting.

Ensemble risk is 85% XGBoost plus 15% normalized anomaly score. Maximum validation
F1 selects review threshold 0.8655566426; the test set is not used for selection.
Block is max(0.75, review), so the two thresholds coincide for this run. It currently
acts as a two-outcome approve/block recommendation policy, not a three-way review policy.

## Untouched chronological test results

| Metric | Result |
|---|---:|
| Precision | 0.927273 |
| Recall | 0.689189 |
| F1 | 0.790698 |
| ROC AUC | 0.959815 |
| Average precision (AP) | 0.782770 |
| True negatives | 56,668 |
| False positives | 4 |
| False negatives | 23 |
| True positives | 51 |

AP is the sklearn average-precision statistic, not an unspecified trapezoidal PR-AUC.
The model missed 23 of 74 test frauds. Results are point estimates from a small,
historical positive sample, not a production guarantee. No test-driven tuning was
performed after these results. Accuracy is intentionally not the headline metric.

## Verification evidence

Executed using the project `.venv`:

```powershell
.\.venv\Scripts\python.exe -m fraud_app.download_data
.\.venv\Scripts\python.exe -m fraud_app.train_real --csv data/creditcard.csv --data-dir data
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q fraud_app
.\.venv\Scripts\python.exe -m pip check
```

- **39 tests passed** in the final suite. One upstream Starlette/httpx deprecation warning.
- Tests use small generated fixtures to exercise CSV validation, tied-time partition
  separation, duplicates/conflicting labels, label exclusion, source isolation,
  30-feature validation, ledger readback, real-source retraining and full-feature drift.
  These fixtures are **not** the source of the benchmark metrics above.
- Compilation passed; dependency check reported no broken requirements.
- The server was restarted with the real artifact. Live HTTP `/api/model` confirmed
  the version and source, and `/api/predict/card` scored the first test example from
  each historical class (chosen by chronological order, not by favorable scores).
- Legitimate example: risk 0.056905, approve, scoring latency 27.84 ms.
- Fraud example: risk 0.991157, block, scoring latency 11.96 ms.
- Both prediction UUIDs were found in `/api/transactions`; real dashboard count was 2.
- `/` returned 200; `/data/creditcard.csv` returned 404 (raw data is not web-served).
- Two requests do not establish throughput, p95 latency, or an inference SLA.
- Browser navigation repeatedly timed out. Full dashboard end-to-end interaction
  testing remains unverified. Docker and cloud deployment are also unverified.

Environment: Python 3.14.4, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1,
xgboost 3.4.1, FastAPI 0.143.0. Library ranges are not a fully locked environment.

## Remaining boundaries

- No authentication; bind to loopback and use only trusted local users.
- Historical PCA components cannot be reconstructed from ordinary payment fields.
- Supplied PCA preprocessing is outside this project's control.
- Time itself may create drift signals simply because time advances.
- No human-label feedback, scheduled retraining, production model promotion, or live payments.
- Real retraining repeats the same public dataset; it does not learn from unlabeled submissions.
- Test metrics have limited statistical precision (74 actual fraud cases).
