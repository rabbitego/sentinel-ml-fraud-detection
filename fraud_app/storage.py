"""SQLite ledger; each request owns a connection and commits its own write."""
from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .model import FEATURES


@contextmanager
def database(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / "transactions.sqlite3", timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("""CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
            amount REAL NOT NULL, hour INTEGER NOT NULL, distance_km REAL NOT NULL,
            transactions_24h INTEGER NOT NULL, account_age_days INTEGER NOT NULL,
            is_international INTEGER NOT NULL, risk_score REAL NOT NULL,
            xgb_score REAL NOT NULL, anomaly_score REAL NOT NULL, decision TEXT NOT NULL,
            latency_ms REAL NOT NULL, model_version TEXT NOT NULL)""")
        connection.execute("CREATE INDEX IF NOT EXISTS ix_transactions_created_at ON transactions(created_at)")
        connection.execute("""CREATE TABLE IF NOT EXISTS card_transactions (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, amount REAL NOT NULL,
            risk_score REAL NOT NULL, xgb_score REAL NOT NULL, anomaly_score REAL NOT NULL,
            decision TEXT NOT NULL, latency_ms REAL NOT NULL, model_version TEXT NOT NULL,
            training_source TEXT NOT NULL, features TEXT NOT NULL)""")
        connection.execute("CREATE INDEX IF NOT EXISTS ix_card_transactions_created_at ON card_transactions(created_at)")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def cutoff(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def save_transaction(directory: Path, record: dict):
    columns = ["id", "created_at", *FEATURES, "risk_score", "xgb_score", "anomaly_score", "decision", "latency_ms", "model_version"]
    table = "transactions"
    if record.get("training_source") == "real_creditcard":
        table = "card_transactions"
        columns = ["id", "created_at", "amount", "risk_score", "xgb_score", "anomaly_score", "decision", "latency_ms", "model_version", "training_source", "features"]
        record = {**record, "features": json.dumps(record["features"], allow_nan=False)}
    with database(directory) as connection:
        connection.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                           [record[key] for key in columns])


def list_transactions(directory: Path, days: int, limit: int, source="synthetic") -> list[dict]:
    table = "card_transactions" if source == "real_creditcard" else "transactions"
    with database(directory) as connection:
        rows = connection.execute(f"SELECT * FROM {table} WHERE created_at >= ? ORDER BY created_at DESC, id DESC LIMIT ?",
                                  (cutoff(days), limit)).fetchall()
    result = []
    for row in rows:
        record = dict(row)
        if source == "real_creditcard":
            record["features"] = json.loads(record["features"])
        else:
            record["is_international"] = bool(record["is_international"])
            record["training_source"] = "synthetic"
            record["features"] = {key: record[key] for key in FEATURES}
        result.append(record)
    return result


def dashboard_data(directory: Path, days: int, source="synthetic", features=None):
    table = "card_transactions" if source == "real_creditcard" else "transactions"
    features = FEATURES if features is None else features
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=days)).isoformat()
    with database(directory) as connection:
        # One SQLite read snapshot makes aggregate, series and drift internally consistent.
        connection.execute("BEGIN")
        summary = dict(connection.execute(f"""SELECT COUNT(*) AS total_transactions,
            COALESCE(SUM(CASE WHEN decision != 'approve' THEN 1 ELSE 0 END), 0) AS flagged_transactions,
            COALESCE(SUM(CASE WHEN decision = 'block' THEN amount ELSE 0 END), 0) AS blocked_amount,
            COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
            FROM {table} WHERE created_at >= ?""", (since,)).fetchone())
        # Hour buckets for 24h; date buckets for multi-day windows (UTC).
        length = 13 if days == 1 else 10
        buckets = connection.execute(f"""SELECT substr(created_at, 1, ?) AS label, COUNT(*) AS total,
            SUM(CASE WHEN decision != 'approve' THEN 1 ELSE 0 END) AS flagged
            FROM {table} WHERE created_at >= ? GROUP BY label ORDER BY label""", (length, since)).fetchall()
        columns = "features" if source == "real_creditcard" else ",".join(FEATURES)
        current = connection.execute(f"SELECT {columns} FROM {table} WHERE created_at >= ? ORDER BY created_at DESC LIMIT 5000", (since,)).fetchall()
    indexed = {row["label"]: dict(row) for row in buckets}
    start = now - timedelta(days=days)
    if days == 1:
        start = start.replace(minute=0, second=0, microsecond=0)
        step = timedelta(hours=1)
    else:
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        step = timedelta(days=1)
    series = []
    while start <= now:
        label = start.isoformat()[:length]
        series.append(indexed.get(label, {"label": label, "total": 0, "flagged": 0}))
        start += step
    summary["series"] = series
    if source == "real_creditcard":
        decoded = [json.loads(row["features"]) for row in current]
        return summary, [[row[key] for key in features] for row in decoded]
    return summary, [list(row) for row in current]
