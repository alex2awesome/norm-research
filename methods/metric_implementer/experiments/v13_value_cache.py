"""Small content-addressed SQLite cache for the CR-3 v13.1 value campaign.

The cache is deliberately below the artifact layer.  A crash may leave a partially
filled cache, but every completed state cell remains reusable and immutable.  Reusing a
key with different bytes is an error rather than a last-writer-wins update.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Mapping


CACHE_SCHEMA = "cr3-value-cache-v13.1"


def canonical_json(payload: object) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def cache_key(kind: str, fields: Mapping[str, object]) -> str:
    return payload_sha256({"schema": CACHE_SCHEMA, "kind": str(kind), "fields": dict(fields)})


class ValueCache:
    """Immutable JSON rows keyed by a canonical description of one completed cell."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=120.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                key TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ValueCache":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def get(self, key: str) -> dict | None:
        row = self.connection.execute(
            "SELECT payload_json, payload_sha256 FROM entries WHERE key = ?", (str(key),)
        ).fetchone()
        if row is None:
            return None
        payload_json, observed_sha = row
        actual_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if actual_sha != observed_sha:
            raise RuntimeError(f"cache payload checksum mismatch for {key}")
        payload = json.loads(payload_json)
        # Reject non-canonical legacy/hand-edited rows as well as damaged bytes.
        if canonical_json(payload) != payload_json:
            raise RuntimeError(f"cache payload is not canonical for {key}")
        return payload

    def put(self, key: str, kind: str, payload: Mapping[str, object]) -> dict:
        payload_dict = dict(payload)
        payload_json = canonical_json(payload_dict)
        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        self.connection.execute(
            "INSERT OR IGNORE INTO entries(key, kind, payload_json, payload_sha256) "
            "VALUES (?, ?, ?, ?)",
            (str(key), str(kind), payload_json, digest),
        )
        self.connection.commit()
        stored = self.get(str(key))
        if stored != payload_dict:
            raise RuntimeError(
                f"repeated cache key {key} produced non-identical {kind} evidence"
            )
        return stored

    def count(self, kind: str | None = None) -> int:
        if kind is None:
            row = self.connection.execute("SELECT COUNT(*) FROM entries").fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM entries WHERE kind = ?", (str(kind),)
            ).fetchone()
        return int(row[0])
