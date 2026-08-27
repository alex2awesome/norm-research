#!/usr/bin/env python3
"""Fail an independent label execution closed after transcript-audit violations."""

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


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "FROZEN_BEFORE_EITHER_INDEPENDENT_LABEL_PASS":
        raise ValueError("execution plan is not a frozen independent-label plan")
    roots = {"A": Path(args.pass_a_root).resolve(), "B": Path(args.pass_b_root).resolve()}
    audits = {"A": Path(args.audit_a).resolve(), "B": Path(args.audit_b).resolve()}
    passes: dict[str, Any] = {}
    all_violations: list[dict[str, Any]] = []
    for name in ("A", "B"):
        root, audit_path = roots[name], audits[name]
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        validation = root / "validation.json"
        raw = sorted((root / "raw_labels").glob("part-*.json"))
        logs = sorted((root / "logs").glob("part-*.log"))
        if (
            audit.get("schema_version")
            != "silver-match-v3-isolated-labeler-transcript-audit-v1"
            or audit.get("status") != "FAIL"
            or not audit.get("violations")
            or (audit.get("pack_validation") or {}).get("sha256")
            != sha256_file(validation)
            or sha256_file(validation)
            != plan["inputs"][f"staged_pass_{name.lower()}"]["validation"]["sha256"]
            or not raw
            or not logs
        ):
            raise ValueError(f"pass {name} does not prove a failed frozen transcript audit")
        if (root / "labels.validated.jsonl").exists():
            raise ValueError(f"pass {name} already contains promoted labels")
        tagged = [{"pass": name, **row} for row in audit["violations"]]
        all_violations.extend(tagged)
        passes[name] = {
            "root": str(root),
            "validation": _ref(validation),
            "transcript_audit": _ref(audit_path),
            "violation_count": len(tagged),
            "raw_labels": [_ref(path) for path in raw],
            "logs": [_ref(path) for path in logs],
        }
    payload = {
        "schema_version": "silver-match-v3-failed-transcript-label-execution-v1",
        "status": "FAILED_CLOSED_TRANSCRIPT_ISOLATION_VIOLATION",
        "task": plan["task"],
        "row_count": plan["row_count"],
        "failure_kind": "TRANSCRIPT_ISOLATION_VIOLATION",
        "violations": all_violations,
        "canonical_usage": {
            "eligible_for_truth": False,
            "eligible_for_prompt_or_model_selection": False,
            "eligible_for_training": False,
            "eligible_for_reporting": False,
            "raw_labels_promoted": False,
            "preserve_append_only_for_failure_analysis": True,
        },
        "inputs": {"execution_plan": _ref(plan_path), "passes": passes},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {**payload, "output": _ref(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pass-a-root", required=True)
    parser.add_argument("--pass-b-root", required=True)
    parser.add_argument("--audit-a", required=True)
    parser.add_argument("--audit-b", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(freeze(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
