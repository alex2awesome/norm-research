#!/usr/bin/env python3
"""Derive an append-only activation lock from audited v2 truth releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


RELEASE_SCHEMA = "silver-match-v3-clean-gepa-exact-truth-release-v2"
RELEASE_STATUS = "FROZEN_EXACT_TRUTH_RELEASE_AUDITED"


def _release(path: Path, task: str, role: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("scientific_contract") or {}
    if (
        payload.get("schema_version") != RELEASE_SCHEMA
        or payload.get("status") != RELEASE_STATUS
        or payload.get("task") != task
        or payload.get("role") != role
        or contract.get("strict_transcript_pass_required_for_every_consensus_pass")
        is not True
        or contract.get("cross_workspace_artifacts_hash_equivalent") is not True
        or contract.get("legacy_transcripts_allowed") is not False
    ):
        raise ValueError(f"invalid strict v2 truth release: {task}/{role}/{path}")
    return sha256_file(path)


def derive(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_lock).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    lock = json.loads(source.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version")
        != "silver-match-v3-task-local-gepa-predeclaration-v1"
        or lock.get("status") != "FROZEN_AND_EXECUTION_AUTHORIZED"
        or set(lock.get("tasks") or {}) != {"code-review", "math-stackexchange"}
    ):
        raise ValueError("source lock is not the authorized Code/Math activation")
    releases = {
        ("code-review", "optimize"): Path(args.code_optimize_release).resolve(),
        ("code-review", "select"): Path(args.code_select_release).resolve(),
        ("math-stackexchange", "optimize"): Path(args.math_optimize_release).resolve(),
        ("math-stackexchange", "select"): Path(args.math_select_release).resolve(),
    }
    release_hashes = {key: _release(path, *key) for key, path in releases.items()}
    for task in ("code-review", "math-stackexchange"):
        evidence = lock["tasks"][task].get("execution_evidence") or {}
        if not evidence:
            raise ValueError(f"source activation lacks execution evidence: {task}")
        evidence["optimize_truth_release_sha256"] = release_hashes[(task, "optimize")]
        evidence["select_truth_release_sha256"] = release_hashes[(task, "select")]
        lock["tasks"][task]["execution_evidence"] = evidence
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "source_lock": {"path": str(source), "sha256": sha256_file(source)},
        "release_hashes": {
            f"{task}/{role}": value
            for (task, role), value in sorted(release_hashes.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--code-optimize-release", required=True)
    parser.add_argument("--code-select-release", required=True)
    parser.add_argument("--math-optimize-release", required=True)
    parser.add_argument("--math-select-release", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(derive(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
