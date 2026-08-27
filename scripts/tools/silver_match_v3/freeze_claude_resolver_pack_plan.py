#!/usr/bin/env python3
"""Freeze one exact-frontier resolver pack before an isolated Claude pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file
from .freeze_claude_independent_pack_plan import _pack, _ref


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    pack_root = Path(args.pack_root).resolve()
    frontier_path = Path(args.frontier).resolve()
    report_path = Path(args.consensus_report).resolve()
    validation_path = pack_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    unresolved_ref = (report.get("outputs") or {}).get("unresolved") or {}
    report_source = (report.get("inputs") or {}).get("source_pack_validation") or {}
    resolver_inputs = validation.get("inputs") or {}
    if (
        validation.get("schema_version")
        != "silver-match-v3-exact-unresolved-resolver-pack-v1"
        or validation.get("truth_hidden") is not True
        or validation.get("prior_decisions_and_metric_ids_hidden") is not True
        or validation.get("selection_rule")
        != "all_and_only_current_exact_consensus_unresolved_uids"
        or (resolver_inputs.get("unresolved") or {}).get("sha256")
        != sha256_file(frontier_path)
        or unresolved_ref.get("sha256") != sha256_file(frontier_path)
        or (resolver_inputs.get("source_pack_validation") or {}).get("sha256")
        != report_source.get("sha256")
        or int(validation.get("count") or -1)
        != int(report.get("unresolved_count") or -2)
        or report.get("complete") is not False
    ):
        raise ValueError("resolver pack is not the exact truth-hidden consensus frontier")
    pack, _ = _pack(pack_root, args.output_namespace)
    repo = Path(args.repo).resolve()
    implementation_paths = [
        repo / "scripts/tools/silver_match_v3/freeze_claude_resolver_pack_plan.py",
        repo / "scripts/tools/silver_match_v3/prepare_exact_unresolved_resolver_pack.py",
        repo / "scripts/tools/silver_match_v3/run_claude_pack_labels.py",
        repo / "scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md",
        repo / "scripts/tools/silver_match_v3/ISOLATED_LABELER_NO_DISCOVERY_GUIDE.md",
        repo
        / "scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json",
    ]
    result = {
        "schema_version": "silver-match-v3-independent-claude-label-execution-freeze-v1",
        "status": "FROZEN_BEFORE_CLAUDE_RESOLVER_PASS",
        "task": args.task,
        "model": args.model,
        "effort": args.effort,
        "output_namespace": args.output_namespace,
        "row_count": pack["count"],
        "passes": {args.pass_name: pack},
        "prior_consensus_frontier": _ref(frontier_path),
        "source_consensus_report": _ref(report_path),
        "implementation": [_ref(path) for path in implementation_paths],
        "runtime_contract": {
            "separate_process_per_chunk": True,
            "only_read_tool_available": True,
            "safe_mode": True,
            "no_session_persistence": True,
            "strict_empty_mcp_config": True,
            "pass_outputs_mutually_hidden": True,
            "prior_labels_hidden_from_resolver": True,
            "prior_truth_proposals_model_outputs_mi_and_outcomes_hidden": True,
            "Gemma_baseline_outputs_available_to_labelers": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "output": str(output), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--consensus-report", required=True)
    parser.add_argument("--pass-name", required=True)
    parser.add_argument("--output-namespace", default="claude_sonnet_v1")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args), sort_keys=True))


if __name__ == "__main__":
    main()
