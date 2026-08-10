#!/usr/bin/env python3
"""Ingest read-only Claude label sessions into an immutable raw JSONL artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def extract_structured_labels(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Return the final StructuredOutput labels and models recorded by a session."""
    payloads: list[dict[str, Any]] = []
    models: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid session JSON at {path}:{line_no}") from exc
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            model = message.get("model")
            if isinstance(model, str) and model:
                models.add(model)
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use" or block.get("name") != "StructuredOutput":
                    continue
                value = block.get("input")
                if isinstance(value, dict) and isinstance(value.get("labels"), list):
                    payloads.append(value)
    if not payloads:
        raise ValueError(f"no StructuredOutput labels found in {path}")
    if len(payloads) != 1:
        raise ValueError(f"expected one StructuredOutput payload in {path}, found {len(payloads)}")
    labels = payloads[0]["labels"]
    if not all(isinstance(label, dict) for label in labels):
        raise ValueError(f"non-object label in {path}")
    return labels, models


def collect(
    session_paths: list[Path], expected_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    expected_uids = [str(row["norm_uid"]) for row in expected_rows]
    if len(expected_uids) != len(set(expected_uids)):
        raise ValueError("expected items contain duplicate norm_uid values")
    by_uid: dict[str, dict[str, Any]] = {}
    models: dict[str, list[str]] = {}
    for path in session_paths:
        labels, seen_models = extract_structured_labels(path)
        models[str(path)] = sorted(seen_models)
        for label in labels:
            uid = str(label.get("norm_uid") or "")
            if not uid:
                raise ValueError(f"missing norm_uid in {path}")
            if uid in by_uid:
                raise ValueError(f"duplicate label for {uid}")
            by_uid[uid] = label
    missing = [uid for uid in expected_uids if uid not in by_uid]
    extra = sorted(set(by_uid) - set(expected_uids))
    if missing or extra:
        raise ValueError(
            f"session labels do not equal expected items: missing={missing[:3]}, extra={extra[:3]}"
        )
    return [by_uid[uid] for uid in expected_uids], models


def apply_overrides(
    rows: list[dict[str, Any]], override_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply explicit human adjudications without changing row membership/order."""
    by_uid = {str(row["norm_uid"]): dict(row) for row in rows}
    seen: set[str] = set()
    for override in override_rows:
        uid = str(override.get("norm_uid") or "")
        if uid not in by_uid:
            raise ValueError(f"override norm_uid is absent from session labels: {uid!r}")
        if uid in seen:
            raise ValueError(f"duplicate override for {uid}")
        seen.add(uid)
        by_uid[uid] = dict(override)
    return [by_uid[str(row["norm_uid"])] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-log", action="append", required=True)
    parser.add_argument("--expected-items", required=True)
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    session_paths = [Path(value).resolve() for value in args.session_log]
    expected_path = Path(args.expected_items).resolve()
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"immutable output already exists: {output_path}")
    rows, models = collect(session_paths, list(read_jsonl(expected_path)))
    override_paths = [Path(value).resolve() for value in args.overrides]
    override_rows = [
        row for path in override_paths for row in read_jsonl(path)
    ]
    rows = apply_overrides(rows, override_rows)
    write_jsonl(output_path, rows)
    meta = {
        "schema_version": "silver-match-v3-independent-session-ingest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "expected_items": str(expected_path),
        "expected_items_sha256": sha256_file(expected_path),
        "session_logs": [str(path) for path in session_paths],
        "session_log_sha256": {
            str(path): sha256_file(path) for path in session_paths
        },
        "models_by_session": models,
        "overrides": [str(path) for path in override_paths],
        "override_count": len(override_rows),
        "override_sha256": {
            str(path): sha256_file(path) for path in override_paths
        },
        "output_sha256": sha256_file(output_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
