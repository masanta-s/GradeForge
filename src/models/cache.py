"""Small JSON cache for model resolution results (data/resolution_cache.db).

Holds HF resolutions (30-day TTL), manual repo overrides, capability probe reports and the
override registry. Kept separate from the teacher's corrections: everything here can be
deleted and rebuilt.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from src import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_cache (
    kind      TEXT NOT NULL,
    key       TEXT NOT NULL,
    payload   TEXT NOT NULL,
    stored_at REAL NOT NULL,
    PRIMARY KEY (kind, key)
)
"""


class ModelCache:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.RESOLUTION_CACHE_DB)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(_SCHEMA)
        return conn

    def get(self, kind: str, key: str, max_age: float | None = None) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT payload, stored_at FROM model_cache WHERE kind = ? AND key = ?",
                               (kind, key)).fetchone()
        if row is None or (max_age is not None and time.time() - row[1] > max_age):
            return None
        return json.loads(row[0])

    def stored_at(self, kind: str, key: str) -> float | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT stored_at FROM model_cache WHERE kind = ? AND key = ?",
                               (kind, key)).fetchone()
        return row[0] if row else None

    def put(self, kind: str, key: str, payload: dict) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO model_cache VALUES (?, ?, ?, ?)",
                         (kind, key, json.dumps(payload), time.time()))

    def delete(self, kind: str, key: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM model_cache WHERE kind = ? AND key = ?", (kind, key))
