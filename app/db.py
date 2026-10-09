"""SQLite persistence for scan history. All queries are parameterised.

Privacy: only the URL, verdict and score are stored (no IP addresses, no feature
vectors). Clear everything with:  python -m app.db --clear
"""
from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

DEFAULT_DB_PATH = Path(os.environ.get(
    "PHISHING_DB", Path(__file__).resolve().parent.parent / "data" / "scans.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    url            TEXT    NOT NULL,
    label          TEXT    NOT NULL CHECK (label IN ('phishing', 'legitimate')),
    risk_score     REAL    NOT NULL,
    risk_level     TEXT    NOT NULL,
    low_confidence INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans (created_at DESC);
"""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None) -> None:
    with closing(connect(db_path)) as conn, conn:
        conn.executescript(SCHEMA)


def save_scan(result: dict, db_path=None) -> int:
    with closing(connect(db_path)) as conn, conn:
        cur = conn.execute(
            "INSERT INTO scans (url, label, risk_score, risk_level, low_confidence) VALUES (?, ?, ?, ?, ?)",
            (result["url"], result["label"], result["risk_score"], result["risk_level"],
             int(result["low_confidence"])))
        return int(cur.lastrowid)


def recent_scans(limit: int = 20, db_path=None) -> list[dict]:
    with closing(connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, url, label, risk_score, risk_level, low_confidence, created_at "
            "FROM scans ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def stats(db_path=None) -> dict:
    with closing(connect(db_path)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(label = 'phishing'), 0) AS phishing, "
            "COALESCE(SUM(label = 'legitimate'), 0) AS legitimate FROM scans").fetchone()
    return {"total": row["total"], "phishing": row["phishing"], "legitimate": row["legitimate"]}


def clear(db_path=None) -> None:
    with closing(connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM scans")


if __name__ == "__main__":
    init_db()
    if "--clear" in sys.argv:
        clear()
        print("Scan history cleared.")
    else:
        print(f"Database ready at {DEFAULT_DB_PATH}. Use --clear to delete all stored scans.")
