#!/usr/bin/env python3
"""Seal exact all-corpus production-candidate coverage for one task.

The per-corpus candidate auditor proves retrieval semantics. This task-level
auditor proves that every canonical corpus for a task is present exactly once,
rechecks the sealed candidate/meta hashes, and totals the canonical rows. With
``--wait`` it can be launched before a frozen producer queue and will run as
soon as all expected per-corpus audits exist.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from .audit_alltask_release_coverage import _verify_candidates
from .common import sha256_file


SCHEMA = "silver-match-v3-task-candidate-coverage-audit-v1"


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _pid_is_alive(pidfile: Path) -> bool:
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def wait_for_paths(
    paths: Iterable[Path], *, poll_seconds: int, producer_pidfile: Path | None = None
) -> None:
    paths = [path.resolve() for path in paths]
    poll_seconds = min(60, max(5, int(poll_seconds)))
    while True:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            return
        if producer_pidfile is not None and not _pid_is_alive(producer_pidfile):
            raise RuntimeError(
                f"producer exited before task coverage completed; missing={missing[:3]}"
            )
        print(
            json.dumps(
                {
                    "status": "WAITING_FOR_CANDIDATE_AUDITS",
                    "present": len(paths) - len(missing),
                    "expected": len(paths),
                    "missing_sample": [str(path) for path in missing[:3]],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def audit_task_candidate_coverage(
    *, manifest_path: Path, task: str, candidate_audits: Iterable[Path]
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task not in (manifest.get("banks") or {}):
        raise KeyError(f"unknown task: {task}")
    expected = {
        corpus: meta
        for corpus, meta in (manifest.get("corpora") or {}).items()
        if meta.get("task") == task
    }
    audit_paths = [path.resolve() for path in candidate_audits]
    if len(audit_paths) != len(set(audit_paths)):
        raise ValueError("duplicate candidate-audit path")
    verified = _verify_candidates(manifest_path, manifest, audit_paths)
    records = verified["corpora"]
    if set(records) != set(expected):
        raise ValueError(
            f"task candidate-audit corpus mismatch: "
            f"missing={sorted(set(expected) - set(records))} "
            f"foreign={sorted(set(records) - set(expected))}"
        )
    covered_count = sum(int(row["count"]) for row in records.values())
    expected_count = sum(int(row["count"]) for row in expected.values())
    if covered_count != expected_count:
        raise ValueError("task candidate row total differs from canonical manifest")
    return {
        "schema_version": SCHEMA,
        "complete": True,
        "task": task,
        "manifest": _artifact(manifest_path),
        "bank_source_sha256": manifest["banks"][task]["source_sha256"],
        "expected_corpora": len(expected),
        "complete_corpora": len(records),
        "expected_count": expected_count,
        "covered_count": covered_count,
        "corpora": records,
        "implementation": _artifact(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate-audit", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--producer-pidfile")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    audit_paths = [Path(value).resolve() for value in args.candidate_audit]
    if args.wait:
        wait_for_paths(
            audit_paths,
            poll_seconds=args.poll_seconds,
            producer_pidfile=(
                Path(args.producer_pidfile).resolve() if args.producer_pidfile else None
            ),
        )
    payload = audit_task_candidate_coverage(
        manifest_path=Path(args.manifest),
        task=args.task,
        candidate_audits=audit_paths,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": _artifact(output), "coverage": payload["covered_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
