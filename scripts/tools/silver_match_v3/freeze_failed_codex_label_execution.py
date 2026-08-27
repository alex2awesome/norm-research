#!/usr/bin/env python3
"""Fail a timed-out independent Codex plan closed and bind its replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


def _ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _artifacts(root: Path) -> dict[str, Any]:
    root = root.resolve()
    validation = root / "validation.json"
    logs = sorted((root / "logs").glob("part-*.log"))
    raw = sorted((root / "raw_labels").glob("part-*.json"))
    if not logs:
        raise ValueError(f"failed execution has no chunk logs: {root}")
    if (root / "labels.validated.jsonl").exists():
        raise ValueError(f"failed execution already has promoted labels: {root}")
    return {
        "root": str(root),
        "validation": _ref(validation),
        "logs": [_ref(path) for path in logs],
        "raw_labels": [_ref(path) for path in raw],
        "completed_raw_chunk_count": len(raw),
        "validated_labels_present": False,
    }


def _pass_hashes(plan: dict[str, Any], key: str) -> tuple[str, str]:
    staged = plan["inputs"][f"staged_pass_{key.lower()}"]
    return str(staged["items"]["sha256"]), str(staged["bank"]["sha256"])


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    old_path = Path(args.failed_plan).resolve()
    replacement_path = Path(args.replacement_plan).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    old = json.loads(old_path.read_text(encoding="utf-8"))
    replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
    expected_schema = "silver-match-v3-independent-codex-label-execution-plan-v1"
    if old.get("schema_version") != expected_schema or replacement.get("schema_version") != expected_schema:
        raise ValueError("failed/replacement inputs are not independent Codex plans")
    if old.get("task") != replacement.get("task") or old.get("row_count") != replacement.get("row_count"):
        raise ValueError("replacement changes task or row universe")
    for key in ("model", "reasoning_effort", "timeout_seconds"):
        if old["runtime"].get(key) != replacement["runtime"].get(key):
            raise ValueError(f"replacement changes frozen runtime semantic: {key}")
    for key in ("labeling_guide", "isolation_guide", "output_schema"):
        if old["implementation"][key]["sha256"] != replacement["implementation"][key]["sha256"]:
            raise ValueError(f"replacement changes frozen labeling input: {key}")
    for key in ("A", "B"):
        if _pass_hashes(old, key) != _pass_hashes(replacement, key):
            raise ValueError(f"replacement changes item or bank order for pass {key}")

    passes = {"A": _artifacts(Path(args.pass_a_root)), "B": _artifacts(Path(args.pass_b_root))}
    for key in ("A", "B"):
        if passes[key]["validation"]["sha256"] != old["inputs"][f"staged_pass_{key.lower()}"]["validation"]["sha256"]:
            raise ValueError(f"failed pass root differs from frozen plan: {key}")
    completed = sum(row["completed_raw_chunk_count"] for row in passes.values())
    if completed:
        raise ValueError("timeout replacement requires zero completed chunks in the failed plan")

    report = {
        "schema_version": "silver-match-v3-failed-codex-label-execution-v1",
        "status": "FAILED_CLOSED_SUPERSEDED_BY_UID_ORDER_PRESERVING_REPARTITION",
        "task": old["task"],
        "failure_kind": "CHUNK_TIMEOUT",
        "observed_timeout_seconds": args.observed_timeout_seconds,
        "scoped_processes_terminated": True,
        "raw_labels_promoted": False,
        "completed_raw_chunk_count": completed,
        "semantic_settings_unchanged": True,
        "item_and_bank_order_unchanged_per_pass": True,
        "inputs": {
            "failed_plan": _ref(old_path),
            "replacement_plan": _ref(replacement_path),
            "failed_passes": passes,
        },
        "replacement": {
            "failed_chunks_per_pass": len(old["inputs"]["staged_pass_a"]["chunks"]),
            "replacement_chunks_per_pass": len(replacement["inputs"]["staged_pass_a"]["chunks"]),
            "failed_chunk_attempts": old["runtime"]["chunk_attempts"],
            "replacement_chunk_attempts": replacement["runtime"]["chunk_attempts"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "output": _ref(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-plan", required=True)
    parser.add_argument("--replacement-plan", required=True)
    parser.add_argument("--pass-a-root", required=True)
    parser.add_argument("--pass-b-root", required=True)
    parser.add_argument("--observed-timeout-seconds", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.observed_timeout_seconds < 1:
        parser.error("--observed-timeout-seconds must be positive")
    print(json.dumps(freeze(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
