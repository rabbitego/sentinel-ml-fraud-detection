# Sentinel ML — local fraud detection lab

A runnable educational ML application, not a production payment gateway. Train an
XGBoost + Isolation Forest ensemble, score transactions with FastAPI, persist results
in SQLite, and explore them in a dependency-free web dashboard.

**Real historical benchmark support is now available and active locally.** Train on
the ULB/Worldline public credit-card dataset, or explicitly choose the older synthetic
demo. Scores are uncalibrated risk scores. `approve`, `review`, and `block` are
recommendations only: this app never authorizes, declines, or moves money.

## Quick start (Windows PowerShell)

Run from the project directory. Python 3.12+ is recommended (tested locally on 3.14).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m fraud_app.download_data
.\.venv\Scripts\python.exe -m fraud_app.train_real --csv data/creditcard.csv --data-dir data
.\.venv\Scripts\python.exe -m uvicorn fraud_app.main:app --host 127.0.0.1 --port 8001
```

Open **http://localhost:8001/**. Interactive API docs: **http://localhost:8001/docs**.
Do not serve `index.html` with `python -m http.server`: it needs the API on the same origin.

macOS/Linux equivalent:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m fraud_app.download_data
.venv/bin/python -m fraud_app.train_real --csv data/creditcard.csv --data-dir data
.venv/bin/python -m uvicorn fraud_app.main:app --host 127.0.0.1 --port 8001
```

### Try it

1. Open **Score transaction** and load the legitimate held-out benchmark example.
2. Score and save, then try the fraud-labeled held-out example. Historical labels
   are shown for context but are never sent as inference features; examples are
   not guaranteed correct predictions. You may paste another exact 30-feature JSON.
3. View saved IDs, scores and model versions in **Transaction history**. Filter
   by date or recommendation; history loads the latest 100 records and paginates locally.
4. Open **Overview** for actual ledger totals, scoring latency, chart and drift.
5. Open **Model & training** for measured holdout metrics and explicit retraining.

No training runs on import or startup. If the artifact is missing, the UI explains
the untrained state; run the real-data training CLI above. On an untrained installation,
the UI's train button explicitly creates the synthetic demo. Once a real model is
active, the retrain button stays on the real dataset and never silently switches sources.

## Real benchmark design and evaluation

Source: [ULB/Worldline Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
also used in [TensorFlow's imbalance tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
The downloader uses that tutorial's public Google Storage CSV mirror (~151 MB),
verifies its known MD5, and records SHA-256. It does not upload anything.
Review the dataset provider's current terms before redistribution or commercial use;
the raw CSV and model are ignored by Git and are not bundled with this source.

- 284,807 original historical transactions; 492 labeled frauds.
- 1,081 duplicate feature rows removed before splitting. Conflicting labels fail validation.
- Chronological train/validation/test split: 170,236 / 56,744 / 56,746 rows.
  Equal-Time groups stay in one split. Fraud counts: 342 / 57 / 74.
- Features are exactly `Time`, `V1`–`V28`, `Amount`; `Class` is the target, never input.
- XGBoost fits training labels with square-root class-ratio weighting. Isolation
  Forest fits only training legitimate rows. The ensemble remains 85% / 15%.
- Review threshold selected by maximum F1 on validation only; test untouched until evaluation.
- Test precision **92.73%**, recall **68.92%**, F1 **0.7907**, ROC AUC **0.9598**,
  average precision **0.7828**. Confusion matrix: TN 56,668; FP 4; FN 23; TP 51.
- The current review threshold is **0.86556**. Block is `max(0.75, review)`, so these
  thresholds coincide in this run: there is no intermediate review band.

This small historical fraud test set does not establish live fraud performance.
The original PCA transform for V1–V28 is unavailable: ordinary raw payment details
cannot be converted to this input schema by this app. The dataset's provided PCA
preprocessing cannot be independently audited here. See `reports/real-data-evaluation.md`.

## Optional synthetic demo

- Seed 42, 12,000 synthetic rows by default. Six validated features: amount, hour,
  distance, 24-hour transaction count, account age, international flag.
- Fraud labels are probabilistically generated from these features, with label noise.
- Stratified 75/25 train/holdout split; no fitting or threshold tuning on the holdout.
- XGBoost learns from the training labels. Isolation Forest fits legitimate training
  examples. Anomaly scores are normalized against legitimate training percentiles.
- Ensemble: `0.85 * xgboost_score + 0.15 * anomaly_percentile`.
- Fixed review threshold 0.45; block threshold 0.75. These are demo policy choices,
  not validated business thresholds.
- Metadata includes precision, recall, F1, ROC AUC and threshold-specific confusion counts.

The initial local default run produced ROC AUC **0.7946**, precision **0.5103**,
recall **0.2712**, and F1 **0.3542** at the review threshold on 3,000 holdout rows.
The low recall is a genuine limitation. Exact results can differ with library versions.
Accuracy alone would be misleading on this imbalanced dataset. Do not claim 35%
fewer false positives, production accuracy, or deployment improvements from this demo.

Training writes `data/model.joblib` and a version-specific metrics JSON. CLI options:

```sh
python -m fraud_app.train --samples 12000 --seed 42 --data-dir data
```

Model versions identify training runs; artifacts are atomically replaced. API
retraining updates the in-process model while preserving the prediction ledger.
If training via CLI while the API is running, restart the API to load the new model.

## Persistence and monitoring

`FRAUD_DATA_DIR` selects the artifact/SQLite directory (default `data` relative to the
working directory). Each successful prediction has a UUID, UTC timestamp, input
features, component scores, decision, model version, and measured scoring latency.

Date windows are rolling 1, 7 or 30 days. Chart buckets are UTC; history timestamps
display in browser local time. `blocked_amount` is the sum of amounts recommended
for blocking, not money actually recovered. Amounts have no assumed currency.

The custom drift monitor compares feature histograms with the training reference,
averaging total-variation distances. It uses up to the latest 5,000 predictions in
the selected window, requires at least 50, and signals drift at 0.20. It is **not
Evidently AI**. Small, manually chosen samples can legitimately trigger drift.

Retraining is **manual** and retains the active model's source. Real retraining reads
`<FRAUD_DATA_DIR>/creditcard.csv`; a missing file fails without replacing the model.
Synthetic and real ledgers are separate; dashboard/history display only the active
source. Drift does not imply new fraud labels, and rerunning the same dataset does
not adapt to a new population. Automatic retraining, labeled feedback, and promotion gates are
not implemented. No scheduled background task is installed.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Model readiness (HTTP 200 also covers untrained state; inspect `model_ready`) |
| POST | `/api/predict` | Synthetic model only: validate six fields, infer and persist |
| POST | `/api/predict/card` | Real model only: exact 30 benchmark numeric fields, no Class |
| GET | `/api/examples` | First legitimate and fraud test rows, labels outside features |
| GET | `/api/transactions?days=7&limit=100` | Latest saved records; API limit 1–1000 |
| GET | `/api/dashboard?days=7` | Ledger aggregates, series, model summary and drift |
| GET | `/api/model` | Full model provenance and holdout evaluation |
| POST | `/api/retrain` | Retrain active source synchronously; no implicit downloads |

Example JSON for synthetic-only `/api/predict` (real mode rejects it with 409):

```json
{
  "amount": 45.0,
  "hour": 14,
  "distance_km": 2.0,
  "transactions_24h": 2,
  "account_age_days": 1500,
  "is_international": false
}
```

Invalid or extra fields receive 422. Missing model receives 503 on scoring/model
endpoints. Concurrent retraining receives 409. Requests are not idempotent: after
an ambiguous network error, inspect history before submitting again.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q fraud_app
```

Tests train actual small models in temporary directories and cover prediction,
validation, persistence, date filtering, aggregate consistency, reproducible data,
drift, corrupt/untrained artifacts, and retraining failure recovery. Tests do not
touch the application ledger. Frontend browser interaction testing remains a
separate check; API tests are not a substitute for visual/browser validation.

### Local verification (2026-10-09 UTC)

- `python -m pytest -q`: **39 passed**. One upstream Starlette/httpx deprecation
  warning; no failing tests.
- `python -m compileall -q fraud_app`: passed.
- `python -m pip check`: no broken requirements.
- Live HTTP smoke test on port 8001: first legitimate test row scored 0.056905 (`approve`),
  first fraud test row scored 0.991157 (`block`); both read back from the real ledger.
- Observed scoring latency for those two requests: 27.84 ms and 11.96 ms. This is
  a two-request smoke check, **not** a latency benchmark or service guarantee.
- Private model/database paths are not served by the API.
- New dashboard browser interactions were not verified: browser navigation timed
  out. Docker/CI were not run locally. Real-data fixture tests are generated, not
  the source of reported benchmark performance; the full CSV was trained separately.

Dependencies use bounded ranges, not a fully locked environment. For reproducible
benchmarking, retain the exact installed versions alongside artifacts. GitHub
Actions runs the tests, Python compilation, and Docker build on Python 3.12.

## Docker

Docker must be installed separately. The supplied image runs as a non-root user;
Compose exposes it only on the local loopback interface and retains data in a volume.

```sh
docker compose build
docker compose run --rm fraud python -m fraud_app.download_data
docker compose run --rm fraud python -m fraud_app.train_real --csv data/creditcard.csv --data-dir data
docker compose up -d
```

Open http://localhost:8001. To stop without deleting data: `docker compose down`.
Docker is not available on the development machine, so this configuration has not
been locally build-tested. CI must pass before describing the image as verified.

## Security and production boundaries

- **Local, single-user use only.** There is no authentication, authorization, rate
  limiting, or abuse protection. Bind to `127.0.0.1`; never expose this directly to
  a public network. The training endpoint is intentionally local/admin-like.
- Use a **single Uvicorn worker**. Retraining locks and model reloads are process-local.
- Only load trusted local joblib artifacts: pickle-based files can execute code.
  There is no model-upload or remote model-download endpoint.
- Transactions contain potentially sensitive inputs. Use public benchmark or synthetic inputs only;
  the SQLite database is not encrypted and has no retention policy.
- No MLflow, Evidently, Azure ML, production deployment, zero-downtime guarantee,
  or real payment integration is implemented. CI configuration is not a deployment.
- Scoring latency excludes request validation, SQLite commit, and network time.
  No sub-100ms SLA or load-test claim is made.

## Publish safely

This project uses its own dedicated Git repository. Before staging, verify
`git rev-parse --show-toplevel` points to this project directory, not a parent user
folder. Inspect the staged diff and publish only the intended source files.

Include `fraud_app/`, `tests/`, `index.html`, `requirements.txt`, `README.md`,
`Dockerfile`, `compose.yaml`, `.dockerignore`, `.gitignore`, and `.github/workflows/`.
Exclude `.opencode/`, `.venv/`, `data/`, personal files, logs and credentials.
Choose an appropriate license before claiming the repository is open source.
