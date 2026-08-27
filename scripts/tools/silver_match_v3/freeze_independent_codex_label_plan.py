#!/usr/bin/env python3
"""Freeze two isolated full-bank Codex label passes before either pass runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _load_pack(root: Path) -> dict[str, Any]:
    root = root.resolve()
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    schema = validation.get("schema_version")
    if schema not in {
        "silver-match-v3-permuted-independent-teacher-pack-v1",
        "silver-match-v3-truth-hidden-uid-subset-pack-v1",
    }:
        raise ValueError(f"not a permuted independent teacher pack: {root}")
    if schema == "silver-match-v3-truth-hidden-uid-subset-pack-v1" and (
        validation.get("truth_hidden") is not True
        or validation.get("prior_decisions_metric_ids_and_proposals_hidden") is not True
    ):
        raise ValueError(f"UID-subset pack is not truth/proposal hidden: {root}")
    outputs = validation.get("outputs") or {}
    for name in ("items", "bank"):
        path = root / f"{name}.json" if name == "bank" else root / "items.jsonl"
        if sha256_file(path) != (outputs.get(name) or {}).get("sha256"):
            raise ValueError(f"{name} hash drift: {root}")
    chunks = sorted((root / "chunks").glob("part-*.jsonl"))
    recorded = outputs.get("chunks") or {}
    if not chunks or len(chunks) != len(recorded):
        raise ValueError(f"chunk inventory drift: {root}")
    recorded_by_name = {Path(path).name: value for path, value in recorded.items()}
    if len(recorded_by_name) != len(recorded):
        raise ValueError(f"source chunk inventory has duplicate basenames: {root}")
    for path in chunks:
        if sha256_file(path) != recorded_by_name.get(path.name):
            raise ValueError(f"chunk hash drift: {path}")
    for forbidden in ("raw_labels", "logs", "labels.validated.jsonl", "predictions.jsonl"):
        if (root / forbidden).exists():
            raise ValueError(f"pack already has label/runtime artifacts: {root / forbidden}")
    return {
        "root": str(root),
        "validation": _ref(validation_path),
        "items": _ref(root / "items.jsonl"),
        "bank": _ref(root / "bank.json"),
        "chunks": [_ref(path) for path in chunks],
        "seed": validation.get("seed"),
        "task": validation.get("task"),
        "count": validation.get("count"),
        "source_pack": validation.get("source_pack")
        or (validation.get("inputs") or {}).get("source_pack_validation"),
    }


def _assert_workspace_copy(source: dict[str, Any], workspace: Path) -> dict[str, Any]:
    root = workspace.resolve() / "pack"
    staged = _load_pack(root)
    if (
        source["task"] != staged["task"]
        or source["count"] != staged["count"]
        or source["seed"] != staged["seed"]
        or source["items"]["sha256"] != staged["items"]["sha256"]
        or source["bank"]["sha256"] != staged["bank"]["sha256"]
        or [row["sha256"] for row in source["chunks"]]
        != [row["sha256"] for row in staged["chunks"]]
    ):
        raise ValueError(f"staged workspace pack differs from source: {workspace}")
    return staged


def _assert_staged_file(implementation_root: Path, workspace: Path, relative: str) -> None:
    source = (implementation_root / relative).resolve()
    staged = (workspace / relative).resolve()
    if not source.is_file() or not staged.is_file() or sha256_file(source) != sha256_file(staged):
        raise ValueError(f"staged implementation input differs from source: {relative}")


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source_a = _load_pack(Path(args.pass_a_source))
    source_b = _load_pack(Path(args.pass_b_source))
    if (
        source_a["task"] != args.task
        or source_b["task"] != args.task
        or source_a["count"] != source_b["count"]
        or source_a["source_pack"] != source_b["source_pack"]
        or source_a["seed"] == source_b["seed"]
    ):
        raise ValueError("independent source packs do not define one distinct-view universe")
    audit_path = Path(args.prelabel_audit).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version") != "silver-match-v3-independent-pack-view-audit-v1"
        or audit.get("status") != "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING"
        or audit.get("task") != args.task
        or audit.get("count") != source_a["count"]
        or audit.get("candidate_proposals_exposed_to_either_pass") is not False
        or audit.get("prior_truth_or_predictions_exposed_to_either_pass") is not False
        or audit.get("pass_predictions_mutually_visible") is not False
    ):
        raise ValueError("prelabel mutual-isolation audit is invalid")
    workspaces = {
        "A": Path(args.workspace_a).resolve(),
        "B": Path(args.workspace_b).resolve(),
    }
    if workspaces["A"] == workspaces["B"]:
        raise ValueError("independent passes require distinct workspaces")
    staged = {
        "A": _assert_workspace_copy(source_a, workspaces["A"]),
        "B": _assert_workspace_copy(source_b, workspaces["B"]),
    }
    python = Path(args.python).resolve()
    implementation_root = Path(args.implementation_root).resolve()
    runner = implementation_root / "scripts/tools/silver_match_v3/run_codex_pack_labels.py"
    guide = implementation_root / "scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"
    isolation_guide = (
        implementation_root
        / "scripts/tools/silver_match_v3/ISOLATED_LABELER_NO_DISCOVERY_GUIDE.md"
    )
    schema = (
        implementation_root
        / "scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json"
    )
    extra_guides = [
        (implementation_root / value).resolve()
        for value in args.extra_boundary_guide
    ]
    boundary_guides = [isolation_guide, *extra_guides]
    for path in (python, runner, guide, isolation_guide, schema, *extra_guides):
        if not path.is_file():
            raise FileNotFoundError(path)
    staged_relatives = [
        str(path.relative_to(implementation_root))
        for path in (guide, schema, *boundary_guides)
    ]
    for workspace in workspaces.values():
        for relative in staged_relatives:
            _assert_staged_file(implementation_root, workspace, relative)
    commands = {}
    for key, pass_name in (("A", args.pass_name_a), ("B", args.pass_name_b)):
        workspace = workspaces[key]
        boundary_argv = [
            value
            for relative in [
                str(path.relative_to(implementation_root))
                for path in boundary_guides
            ]
            for value in ("--boundary-guide", relative)
        ]
        commands[key] = {
            "cwd": str(workspace),
            "environment": {"PYTHONPATH": str(implementation_root)},
            "argv": [
                str(python),
                "-u",
                "-m",
                "scripts.tools.silver_match_v3.run_codex_pack_labels",
                "--pack-root",
                str(workspace / "pack"),
                "--task",
                args.task,
                "--pass-name",
                pass_name,
                *boundary_argv,
                "--concurrency",
                str(args.concurrency),
                "--model",
                args.model,
                "--reasoning-effort",
                args.reasoning_effort,
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--chunk-attempts",
                str(args.chunk_attempts),
                "--output-schema",
                "scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json",
            ],
        }
    payload = {
        "schema_version": "silver-match-v3-independent-codex-label-execution-plan-v1",
        "status": "FROZEN_BEFORE_EITHER_INDEPENDENT_LABEL_PASS",
        "task": args.task,
        "row_count": source_a["count"],
        "pass_count": 2,
        "runtime": {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "concurrency_per_pass": args.concurrency,
            "timeout_seconds": args.timeout_seconds,
            "chunk_attempts": args.chunk_attempts,
            "python": _ref(python),
        },
        "implementation": {
            "runner": _ref(runner),
            "labeling_guide": _ref(guide),
            "isolation_guide": _ref(isolation_guide),
            "output_schema": _ref(schema),
            "boundary_guides": [_ref(path) for path in boundary_guides],
        },
        "inputs": {
            "prelabel_independence_audit": _ref(audit_path),
            "external_policy": _ref(Path(args.policy)) if args.policy else None,
            "source_pass_a": source_a,
            "source_pass_b": source_b,
            "staged_pass_a": staged["A"],
            "staged_pass_b": staged["B"],
        },
        "commands": commands,
        "contracts": {
            "separate_processes_and_workspaces": True,
            "other_pass_predictions_unavailable": True,
            "retrieval_candidates_unavailable_to_truth_labelers": True,
            "prior_truth_labels_and_outcomes_unavailable": True,
            "full_bank_required_for_every_item": True,
            "strict_transcript_audit_required_before_validation": True,
            "prompt_or_command_mutation_after_freeze_allowed": False,
            "mi_or_outcome_use_allowed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**payload, "plan_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--pass-a-source", required=True)
    parser.add_argument("--pass-b-source", required=True)
    parser.add_argument("--prelabel-audit", required=True)
    parser.add_argument("--workspace-a", required=True)
    parser.add_argument("--workspace-b", required=True)
    parser.add_argument("--implementation-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--pass-name-a", required=True)
    parser.add_argument("--pass-name-b", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--extra-boundary-guide", action="append", default=[])
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--chunk-attempts", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.timeout_seconds < 1 or args.chunk_attempts < 1:
        parser.error("concurrency, timeout, and attempts must be positive")
    print(json.dumps(freeze(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
