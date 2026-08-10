#!/usr/bin/env python3
"""Author and freeze one identity-stripped PR verifier prompt.

The author sees a single inline payload derived exclusively from a frozen
optimize-role packet.  It cannot choose among prompt variants after observing
fresh verifier-dev labels: exactly one authored prompt is frozen before any
such join.  A tool-using author transcript fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


EXPECTED_DECISIONS = [
    "CONFIRM_MATCH",
    "AMBIGUOUS_MATCH",
    "BETTER_CANDIDATE",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
]
EXPECTED_CONFIDENCES = ["high", "medium", "low"]
TOOL_EVENT_RE = re.compile(
    r"(?m)^(?:exec|apply_patch|web|mcp|view_image|imagegen|tool)\s*$"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_and_sanitize_packet(
    report_path: Path, examples_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = report_path.resolve()
    examples_path = examples_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = list(read_jsonl(examples_path))
    expected_examples = (report.get("outputs") or {}).get("examples") or {}
    if (
        report.get("schema_version")
        != "silver-match-v3-verifier-author-training-packet-v1"
        or report.get("status") != "FROZEN_OPTIMIZE_ONLY_AUTHORSHIP_EVIDENCE"
        or report.get("task") != "press-releases"
        or report.get("fresh_verifier_dev_truth_read") is not False
        or report.get("blind_audit_truth_read") is not False
        or int(report.get("count", -1)) != len(rows)
        or int(report.get("source_groups", -1)) != len(rows)
        or expected_examples.get("sha256") != sha256_file(examples_path)
        or len(rows) < 20
    ):
        raise ValueError("unsupported, incomplete, or drifted optimize author packet")

    sanitized: list[dict[str, Any]] = []
    targets: Counter[str] = Counter()
    groups: set[str] = set()
    for row in rows:
        target = str(row.get("target") or "")
        group = str(row.get("source_group") or "")
        use = row.get("use_contract") or {}
        if (
            row.get("schema_version")
            != "silver-match-v3-verifier-author-example-v1"
            or row.get("task") != "press-releases"
            or row.get("gepa_role") != "optimize"
            or row.get("predeclared_split") != "train"
            or target not in {"CONFIRM_MATCH", "REJECT"}
            or not group
            or group in groups
            or use.get("verifier_prompt_authorship_or_gepa_optimize_only") is not True
            or use.get("verifier_selection") is not False
            or use.get("final_blind_audit") is not False
            or use.get("mi_or_outcome_estimation") is not False
            or use.get("retriever_training") is not False
            or (row.get("proposal") or {}).get("decision") != "MATCH"
            or not str((row.get("proposal") or {}).get("metric_id") or "")
            or not isinstance(row.get("metric_cards"), dict)
            or not row.get("metric_cards")
        ):
            raise ValueError("invalid or contaminated optimize author example")
        groups.add(group)
        targets[target] += 1
        proposal = row["proposal"]
        gold = row.get("gold") or {}
        sanitized.append(
            {
                "norm": row.get("norm"),
                "context": row.get("context"),
                "proposal": {
                    "decision": "MATCH",
                    "metric_id": proposal.get("metric_id"),
                    "reason": proposal.get("reason"),
                },
                "gold": {
                    "decision": gold.get("decision"),
                    "metric_id": gold.get("metric_id"),
                },
                "target": target,
                "metric_cards": row["metric_cards"],
            }
        )
    expected_targets = {
        str(key): int(value) for key, value in (report.get("target_counts") or {}).items()
    }
    if dict(sorted(targets.items())) != dict(sorted(expected_targets.items())):
        raise ValueError("author packet target counts drifted")
    if not {"CONFIRM_MATCH", "REJECT"} <= set(targets):
        raise ValueError("author packet does not support both confirmation and rejection")
    payload = {
        "schema_version": "silver-match-v3-pr-verifier-sanitized-author-evidence-v1",
        "task": "press-releases",
        "role": "optimize_only_prompt_authorship",
        "identity_fields_removed": ["norm_uid", "source_group"],
        "fresh_verifier_dev_truth_included": False,
        "select_test_blind_mi_or_outcomes_included": False,
        "count": len(sanitized),
        "target_counts": dict(sorted(targets.items())),
        "examples": sanitized,
    }
    return report, payload


def audit_tool_free_transcript(log_text: str) -> dict[str, Any]:
    tool_events = TOOL_EVENT_RE.findall(log_text)
    if tool_events:
        raise ValueError(f"isolated verifier author used tools: {tool_events}")
    return {
        "status": "PASS_TOOL_FREE_INLINE_AUTHOR",
        "tool_event_count": 0,
        "filesystem_reads_requested_by_author": 0,
    }


def validate_author_output(
    value: dict[str, Any], *, expected_hashes: dict[str, str]
) -> None:
    parse = value.get("parse_rule") or {}
    selection = value.get("selection_rule") or {}
    provenance = value.get("provenance") or {}
    prompt_text = str(value.get("prompt_text") or "")
    prompt_lower = prompt_text.lower()
    required_hashes = {
        "training_packet_report_sha256": expected_hashes["report"],
        "training_examples_sha256": expected_hashes["examples"],
        "sanitized_evidence_sha256": expected_hashes["evidence"],
        "input_freeze_sha256": expected_hashes["input_freeze"],
    }
    if (
        value.get("schema_version")
        != "silver-match-v3-pr-verifier-fresh-author-v1"
        or not (900 <= len(prompt_text) <= 14000)
        or parse.get("allowed_decisions") != EXPECTED_DECISIONS
        or parse.get("confirm_metric_id_is_proposal") is not True
        or parse.get("better_candidate_metric_id_is_supplied_alternative") is not True
        or parse.get("other_decisions_metric_id_null") is not True
        or parse.get("confidence_values") != EXPECTED_CONFIDENCES
        or parse.get("confirm_requires_explicit_criterion") is not True
        or parse.get("confirm_requires_exact_leaf_contrast") is not True
        or int(parse.get("reason_max_words", -1)) != 24
        or re.search(
            r"confidence.{0,100}high.{0,60}medium.{0,60}low",
            prompt_lower,
            flags=re.DOTALL,
        )
        is None
        or "24 words" not in prompt_lower
        or "number from 0 to 1" in prompt_lower
        or selection.get("variant_count") != 1
        or selection.get("choose_without_verifier_dev_truth") is not True
        or selection.get("optimize_only_authoring") is not True
        or selection.get("promotion_requires_frozen_fresh_dev_gates") is not True
        or selection.get("precision_dominates_yield") is not True
        or selection.get("no_candidate_fits_route") != "full_bank_rescue"
        or provenance.get("used_only_inline_identity_stripped_optimize_evidence")
        is not True
        or provenance.get("tools_called") is not False
        or provenance.get("verifier_dev_truth_read") is not False
        or provenance.get("select_test_or_blind_material_read") is not False
        or provenance.get("mi_or_outcomes_read") is not False
        or any(provenance.get(key) != expected for key, expected in required_hashes.items())
    ):
        raise ValueError("fresh verifier author output violates its frozen contract")


def run_author(args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(args.workspace).resolve()
    if workspace.exists():
        raise FileExistsError(workspace)
    report_path = Path(args.training_report).resolve()
    examples_path = Path(args.training_examples).resolve()
    instructions_path = Path(args.instructions).resolve()
    schema_path = Path(args.schema).resolve()
    report, evidence = validate_and_sanitize_packet(report_path, examples_path)

    workspace.mkdir(parents=True)
    local_instructions = workspace / "AUTHOR_INSTRUCTIONS.md"
    local_schema = workspace / "OUTPUT_SCHEMA.json"
    evidence_path = workspace / "SANITIZED_OPTIMIZE_EVIDENCE.json"
    shutil.copyfile(instructions_path, local_instructions)
    shutil.copyfile(schema_path, local_schema)
    _write_json(evidence_path, evidence)
    input_freeze_path = workspace / "INPUT_FREEZE.json"
    input_freeze = {
        "schema_version": "silver-match-v3-pr-verifier-author-input-freeze-v1",
        "status": "FROZEN_INLINE_OPTIMIZE_ONLY_BEFORE_VERIFIER_DEV_JOIN",
        "task": "press-releases",
        "variant_count": 1,
        "source_packet": {
            "report_sha256": sha256_file(report_path),
            "examples_sha256": sha256_file(examples_path),
            "count": report["count"],
            "target_counts": report["target_counts"],
        },
        "sanitized_evidence": {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
            "identity_fields_removed": True,
        },
        "instructions": {
            "path": str(local_instructions),
            "sha256": sha256_file(local_instructions),
        },
        "schema": {"path": str(local_schema), "sha256": sha256_file(local_schema)},
        "contracts": {
            "verifier_dev_truth_joined_to_author": False,
            "select_test_blind_mi_or_outcomes_joined_to_author": False,
            "author_receives_inline_evidence_and_must_not_use_tools": True,
        },
    }
    _write_json(input_freeze_path, input_freeze)
    hashes = {
        "report": sha256_file(report_path),
        "examples": sha256_file(examples_path),
        "evidence": sha256_file(evidence_path),
        "input_freeze": sha256_file(input_freeze_path),
    }
    provenance_hashes = {
        "training_packet_report_sha256": hashes["report"],
        "training_examples_sha256": hashes["examples"],
        "sanitized_evidence_sha256": hashes["evidence"],
        "input_freeze_sha256": hashes["input_freeze"],
    }
    prompt = (
        "You are in a sealed prompt-authoring call. Do not call any tool and do not "
        "read the filesystem. Everything you may use is inline below. Return only one "
        "JSON object conforming to the supplied output schema.\n\n"
        + local_instructions.read_text(encoding="utf-8")
        + "\nThe output provenance hashes must be exactly:\n"
        + json.dumps(provenance_hashes, sort_keys=True)
        + "\n\nFROZEN IDENTITY-STRIPPED OPTIMIZE EVIDENCE:\n"
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    )
    output_path = workspace / "author_output.json"
    log_path = workspace / "author.log"
    command = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "-m",
        args.model,
        "-c",
        f'model_reasoning_effort="{args.reasoning_effort}"',
        "--output-schema",
        str(local_schema),
        "-o",
        str(output_path),
        "-",
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            input=prompt,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(f"isolated verifier author failed: {completed.returncode}")
    transcript_audit = audit_tool_free_transcript(log_path.read_text(encoding="utf-8"))
    value = json.loads(output_path.read_text(encoding="utf-8"))
    validate_author_output(value, expected_hashes=hashes)
    freeze = {
        "schema_version": "silver-match-v3-pr-verifier-author-output-freeze-v1",
        "status": "FROZEN_CONTEXT_ISOLATED_BEFORE_VERIFIER_DEV_TRUTH_JOIN",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": "press-releases",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "variant_count": 1,
        "verifier_dev_truth_joined_to_author": False,
        "input_freeze": {
            "path": str(input_freeze_path),
            "sha256": sha256_file(input_freeze_path),
        },
        "author_output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        },
        "author_log": {"path": str(log_path), "sha256": sha256_file(log_path)},
        "transcript_audit": transcript_audit,
        "promotion_contract": {
            "fresh_verifier_dev_may_select_only_this_frozen_variant": True,
            "fresh_verifier_dev_cannot_mutate_prompt": True,
            "final_blind_audit_still_required": True,
        },
    }
    freeze_path = workspace / "OUTPUT_FREEZE.json"
    _write_json(freeze_path, freeze)
    return {**freeze, "output_freeze_sha256": sha256_file(freeze_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--training-examples", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--instructions",
        default="scripts/tools/silver_match_v3/prompts/pr_verifier_fresh_author_instructions_v1.md",
    )
    parser.add_argument(
        "--schema",
        default="scripts/tools/silver_match_v3/schemas/pr_verifier_prompt_author_v1.schema.json",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    print(json.dumps(run_author(args), sort_keys=True))


if __name__ == "__main__":
    main()
