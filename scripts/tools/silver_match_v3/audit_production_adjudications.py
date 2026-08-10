#!/usr/bin/env python3
"""Audit a frozen task's two-order production adjudication outputs exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def _check_artifact(value: dict[str, Any], label: str) -> None:
    path = Path(str(value.get("path") or ""))
    expected = str(value.get("sha256") or "")
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError(f"frozen artifact changed: {label}={path}")


def _check_plan_artifacts(plan: dict[str, Any]) -> None:
    for key in (
        "manifest",
        "candidate_union",
        "candidate_union_meta",
        "retriever_selection",
    ):
        _check_artifact(plan[key], key)
    for role in ("adjudicator", "verifier"):
        block = plan[role]
        for key in ("selection", "implementation"):
            _check_artifact(block[key], f"{role}.{key}")
        for path, value in (block.get("prompt_components") or {}).items():
            _check_artifact({"path": path, **value}, f"{role}.prompt_component")
    _check_artifact(plan["verifier"]["production_policy"], "verifier.production_policy")
    for order, value in (
        plan["adjudicator"].get("selected_dev_run_meta") or {}
    ).items():
        _check_artifact(value, f"adjudicator.selected_dev_run_meta.{order}")
    for path, expected in (plan.get("candidate_audits") or {}).items():
        _check_artifact({"path": path, "sha256": expected}, "candidate_audit")
    for label, value in (plan.get("lineage_artifacts") or {}).items():
        _check_artifact(value, f"lineage_artifacts.{label}")
    for key in ("finalizer_implementation", "runner_implementation"):
        if plan.get(key):
            _check_artifact(plan[key], key)
    verifier = plan.get("verifier") or {}
    if verifier.get("combiner_implementation"):
        _check_artifact(
            verifier["combiner_implementation"], "verifier.combiner_implementation"
        )
    for order, value in (verifier.get("selected_dev_run_meta") or {}).items():
        _check_artifact(value, f"verifier.selected_dev_run_meta.{order}")


def _unique_candidates(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for row in read_jsonl(path):
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in output:
            raise ValueError(f"missing/duplicate candidate norm_uid: {uid!r}")
        output[uid] = row
    return output


def audit(
    *,
    plan_path: Path,
    original_path: Path,
    hashed_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "FROZEN_READY_FOR_UNLABELED_PRODUCTION":
        raise ValueError("task plan is not frozen for production")
    _check_plan_artifacts(plan)
    candidate_path = Path(plan["candidate_union"]["path"])
    candidates = _unique_candidates(candidate_path)
    expected_count = int(plan["expected_count"])
    if len(candidates) != expected_count:
        raise ValueError("candidate union count differs from frozen plan")
    expected_uids = set(candidates)
    rendering = plan["adjudicator"]["prompt_rendering"]
    expected_prompt = plan["adjudicator"]["prompt_sha256"]
    expected_model = plan["adjudicator"]["model"]
    expected_bank = plan["bank_source_sha256"]
    depth = int(plan["adjudicator"]["candidate_depth"])

    order_artifacts = {}
    outputs: dict[str, dict[str, dict[str, Any]]] = {}
    for order, path in (("original", original_path), ("hashed", hashed_path)):
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("input_candidates_sha256") != plan["candidate_union"]["sha256"]
            or meta.get("output_sha256") != sha256_file(path)
            or meta.get("prompt_sha256") != expected_prompt
            or meta.get("model") != expected_model
            or meta.get("order_mode") != order
            or int(meta.get("max_candidates", -1)) != depth
            or (meta.get("prompt_rendering") or {}) != rendering
        ):
            raise ValueError(f"{order} adjudication metadata differs from frozen plan")
        rows = {}
        actual_invalid = 0
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in rows:
                raise ValueError(f"missing/duplicate {order} norm_uid: {uid!r}")
            rows[uid] = row
        if set(rows) != expected_uids:
            raise ValueError(f"{order} adjudication coverage differs from candidate union")
        for uid, row in rows.items():
            candidate = candidates[uid]
            candidate_ids = [
                str(value["metric_id"]) for value in candidate.get("candidates") or []
            ][:depth]
            output_ids = [str(value) for value in row.get("candidate_ids") or []]
            decision = str(row.get("decision") or "")
            parse_error = row.get("parse_error")
            if decision == "INVALID_OUTPUT":
                actual_invalid += 1
                if not parse_error:
                    raise ValueError(f"{order} INVALID_OUTPUT lacks parse error: {uid}")
            elif parse_error:
                raise ValueError(f"{order} parsed row carries parse error: {uid}")
            if (
                row.get("task") != plan["task"]
                or row.get("corpus") not in plan["corpora"]
                or row.get("prompt_sha256") != expected_prompt
                or row.get("model") != expected_model
                or row.get("order_mode") != order
                or row.get("candidate_bank_source_sha256") != expected_bank
                or len(output_ids) != depth
                or set(output_ids) != set(candidate_ids)
            ):
                raise ValueError(f"{order} adjudication row provenance mismatch: {uid}")
        if actual_invalid != int(meta.get("invalid_count", -1)):
            raise ValueError(f"{order} invalid count differs from metadata")
        outputs[order] = rows
        order_artifacts[order] = {
            "output": {"path": str(path), "sha256": sha256_file(path)},
            "meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
            "count": len(rows),
            "invalid_count": actual_invalid,
        }
    if set(outputs["original"]) != set(outputs["hashed"]):
        raise AssertionError("two-order coverage differs")

    report = {
        "schema_version": "silver-match-v3-production-adjudication-audit-v1",
        "complete": True,
        "task": plan["task"],
        "count": expected_count,
        "orders": order_artifacts,
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "candidate_union_sha256": plan["candidate_union"]["sha256"],
        "prompt_sha256": expected_prompt,
        "model": expected_model,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(
        plan_path=Path(args.plan).resolve(),
        original_path=Path(args.original).resolve(),
        hashed_path=Path(args.hashed).resolve(),
        output_path=Path(args.output).resolve(),
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
