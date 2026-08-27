#!/usr/bin/env python3
"""Freeze a byte-identical independent-label execution relocation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .common import sha256_file


SCHEMA = "silver-match-v3-independent-codex-label-execution-plan-v1"


def _ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _pack_identity(plan: dict[str, Any], name: str) -> dict[str, Any]:
    value = plan["inputs"][name]
    return {
        "task": value["task"],
        "count": value["count"],
        "seed": value["seed"],
        "source_pack": value["source_pack"],
        "validation_sha256": value["validation"]["sha256"],
        "items_sha256": value["items"]["sha256"],
        "bank_sha256": value["bank"]["sha256"],
        "chunk_sha256s": [row["sha256"] for row in value["chunks"]],
    }


def _normalized_command(plan: dict[str, Any], key: str) -> dict[str, Any]:
    command = plan["commands"][key]
    argv = list(command["argv"])
    argv[0] = "<PYTHON>"
    argv[argv.index("--pack-root") + 1] = "<WORKSPACE>/pack"
    return {
        "cwd": "<WORKSPACE>",
        "environment": {"PYTHONPATH": "<IMPLEMENTATION_ROOT>"},
        "argv": argv,
    }


def _without_host_paths(value: Any) -> Any:
    """Retain scientific values and hashes while removing host path strings."""

    if isinstance(value, dict):
        return {
            key: _without_host_paths(child)
            for key, child in sorted(value.items())
            if key not in {"path", "root", "cwd"}
        }
    if isinstance(value, list):
        return [_without_host_paths(child) for child in value]
    return value


def _normalized_implementation(plan: dict[str, Any]) -> dict[str, Any]:
    value = _without_host_paths(plan["implementation"])
    # Newer freezer versions make the already-frozen isolation guide explicit
    # as a one-element boundary-guide list.  Treat that backward-compatible
    # spelling as equivalent to the older plan rather than as a prompt change.
    value.setdefault("boundary_guides", [value["isolation_guide"]])
    return value


def _audit_identity(path: Path) -> dict[str, Any]:
    value = _load(path)
    passes = value.get("passes") or {}
    return {
        "schema_version": value.get("schema_version"),
        "status": value.get("status"),
        "task": value.get("task"),
        "count": value.get("count"),
        "bank_metric_count": value.get("bank_metric_count"),
        "same_uid_set": value.get("same_uid_set"),
        "same_canonical_item_content_by_uid": value.get(
            "same_canonical_item_content_by_uid"
        ),
        "same_bank_leaf_set": value.get("same_bank_leaf_set"),
        "distinct_seeds": value.get("distinct_seeds"),
        "distinct_item_order": value.get("distinct_item_order"),
        "distinct_bank_order": value.get("distinct_bank_order"),
        "candidate_proposals_exposed_to_either_pass": value.get(
            "candidate_proposals_exposed_to_either_pass"
        ),
        "prior_truth_or_predictions_exposed_to_either_pass": value.get(
            "prior_truth_or_predictions_exposed_to_either_pass"
        ),
        "pass_predictions_mutually_visible": value.get(
            "pass_predictions_mutually_visible"
        ),
        "passes": {
            key: {
                "seed": passes[key]["seed"],
                "validation_sha256": passes[key]["validation_sha256"],
                "items_sha256": passes[key]["items_sha256"],
                "bank_sha256": passes[key]["bank_sha256"],
            }
            for key in ("A", "B")
        },
        "usage_contract": value.get("usage_contract"),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_plan).resolve()
    target_path = Path(args.relocated_plan).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source, target = _load(source_path), _load(target_path)
    for plan in (source, target):
        if plan.get("schema_version") != SCHEMA:
            raise ValueError("unexpected independent execution-plan schema")
        if plan.get("status") != "FROZEN_BEFORE_EITHER_INDEPENDENT_LABEL_PASS":
            raise ValueError("execution plan was not frozen before labeling")

    equalities = {
        "task": source["task"] == target["task"],
        "row_count": source["row_count"] == target["row_count"],
        "pass_count": source["pass_count"] == target["pass_count"],
        "runtime_semantics": {
            key: value
            for key, value in source["runtime"].items()
            if key != "python"
        }
        == {
            key: value
            for key, value in target["runtime"].items()
            if key != "python"
        },
        "implementation_hashes_and_nonpath_fields": _normalized_implementation(
            source
        )
        == _normalized_implementation(target),
        "contracts": source["contracts"] == target["contracts"],
        "source_pass_a": _pack_identity(source, "source_pass_a")
        == _pack_identity(target, "source_pass_a"),
        "source_pass_b": _pack_identity(source, "source_pass_b")
        == _pack_identity(target, "source_pass_b"),
        "staged_pass_a": _pack_identity(source, "staged_pass_a")
        == _pack_identity(target, "staged_pass_a"),
        "staged_pass_b": _pack_identity(source, "staged_pass_b")
        == _pack_identity(target, "staged_pass_b"),
        "command_a_except_host_paths": _normalized_command(source, "A")
        == _normalized_command(target, "A"),
        "command_b_except_host_paths": _normalized_command(source, "B")
        == _normalized_command(target, "B"),
    }
    known_inputs = {
        "prelabel_independence_audit",
        "source_pass_a",
        "source_pass_b",
        "staged_pass_a",
        "staged_pass_b",
    }
    source_extra_inputs = {
        key: value
        for key, value in source["inputs"].items()
        if key not in known_inputs and value is not None
    }
    target_extra_inputs = {
        key: value
        for key, value in target["inputs"].items()
        if key not in known_inputs and value is not None
    }
    equalities["additional_frozen_inputs"] = _without_host_paths(
        source_extra_inputs
    ) == _without_host_paths(target_extra_inputs)
    source_audit = Path(source["inputs"]["prelabel_independence_audit"]["path"])
    target_audit = Path(target["inputs"]["prelabel_independence_audit"]["path"])
    # The source plan may have been copied from an unmounted host.  Its audit
    # bytes are copied alongside the plan and supplied explicitly when needed.
    if args.source_prelabel_audit:
        source_audit = Path(args.source_prelabel_audit).resolve()
    equalities["prelabel_independence_audit"] = _audit_identity(source_audit) == _audit_identity(
        target_audit
    )

    flat = []
    for key, value in equalities.items():
        if isinstance(value, dict):
            flat.extend((f"{key}.{child}", result) for child, result in value.items())
        else:
            flat.append((key, value))
    failed = [name for name, value in flat if value is not True]
    if failed:
        raise ValueError(f"relocation changes frozen execution semantics: {failed}")

    for key in ("A", "B"):
        pack = Path(target["commands"][key]["argv"][target["commands"][key]["argv"].index("--pack-root") + 1])
        forbidden = [
            pack / "raw_labels",
            pack / "logs",
            pack / "labels.validated.jsonl",
            pack / "predictions.jsonl",
        ]
        observed = [str(path) for path in forbidden if path.exists()]
        if observed:
            raise ValueError(f"relocated workspace already has runtime artifacts: {observed}")

    codex = Path(args.codex_bin).resolve()
    auth = Path(args.auth_file).resolve()
    status = subprocess.run(
        [str(codex), "login", "status"],
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    auth_text = (status.stdout + status.stderr).strip()
    if status.returncode != 0 or "Logged in using ChatGPT" not in auth_text:
        raise ValueError(f"relocated Codex runner is not authenticated: {auth_text}")

    report = {
        "schema_version": "silver-match-v3-independent-codex-plan-relocation-audit-v1",
        "status": "FROZEN_APPEND_ONLY_BYTE_IDENTICAL_EXECUTION_RELOCATION",
        "task": source["task"],
        "scientific_or_labeling_semantics_changed": False,
        "only_changes": [
            "host_filesystem_paths",
            "python_executable_and_pythonpath_host_paths",
            "authenticated_codex_executable_host_path",
        ],
        "equalities": equalities,
        "authenticated_local_runtime": {
            "codex": _ref(codex),
            "auth_file": _ref(auth),
            "login_status": "Logged in using ChatGPT",
            "python": target["runtime"]["python"],
        },
        "inputs": {
            "source_plan": _ref(source_path),
            "relocated_plan": _ref(target_path),
            "source_prelabel_audit": _ref(source_audit),
            "relocated_prelabel_audit": _ref(target_audit),
        },
        "execution_disposition": {
            "eligible_after_external_local_slot_allocation": True,
            "launched_by_this_audit": False,
            "remote_auth_failed_attempts_may_not_be_reused": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "audit_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--relocated-plan", required=True)
    parser.add_argument("--source-prelabel-audit")
    parser.add_argument("--codex-bin", required=True)
    parser.add_argument("--auth-file", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(parser.parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
