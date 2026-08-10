#!/usr/bin/env python3
"""Freeze two mutually hidden full-bank packs before independent Claude labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def _ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _pack(root: Path, output_namespace: str) -> tuple[dict[str, Any], set[str]]:
    root = root.resolve()
    validation_path = root / "validation.json"
    bank_path = root / "bank.json"
    items_path = root / "items.jsonl"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    chunks = sorted((root / "chunks").glob("part-*.jsonl"))
    full_source_pack = (
        validation.get("status") == "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING"
    )
    exact_resolver_pack = (
        validation.get("schema_version")
        == "silver-match-v3-exact-unresolved-resolver-pack-v1"
        and validation.get("prior_decisions_and_metric_ids_hidden") is True
        and validation.get("selection_rule")
        == "all_and_only_current_exact_consensus_unresolved_uids"
    )
    if (
        not (full_source_pack or exact_resolver_pack)
        or validation.get("truth_hidden") is not True
        or not chunks
        or (root / output_namespace).exists()
    ):
        raise ValueError(f"pack is not clean and truth-hidden before Claude: {root}")
    chunk_rows = []
    uids: set[str] = set()
    for chunk in chunks:
        rows = list(read_jsonl(chunk))
        observed = {str(row.get("norm_uid") or "") for row in rows}
        if "" in observed or len(observed) != len(rows) or uids & observed:
            raise ValueError(f"missing or duplicate UIDs in {chunk}")
        uids.update(observed)
        chunk_rows.append({**_ref(chunk), "rows": len(rows)})
    item_uids = {str(row.get("norm_uid") or "") for row in read_jsonl(items_path)}
    if item_uids != uids or len(item_uids) != int(validation.get("count", -1)):
        raise ValueError(f"pack item/chunk universe drift: {root}")
    return (
        {
            "root": str(root),
            "validation": _ref(validation_path),
            "bank": _ref(bank_path),
            "items": _ref(items_path),
            "chunks": chunk_rows,
            "count": len(uids),
            "output_namespace_absent_before_freeze": True,
        },
        uids,
    )


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    independence_path = Path(args.independence_audit).resolve()
    independence = json.loads(independence_path.read_text(encoding="utf-8"))
    if (
        independence.get("status")
        != "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING"
        or independence.get("post_label_artifacts_present") is not False
        or independence.get("prior_truth_or_predictions_exposed_to_either_pass")
        is not False
        or independence.get("candidate_proposals_exposed_to_either_pass") is not False
        or independence.get("pass_predictions_mutually_visible") is not False
        or independence.get("distinct_bank_order") is not True
        or independence.get("distinct_item_order") is not True
        or independence.get("same_uid_set") is not True
        or independence.get("same_bank_leaf_set") is not True
    ):
        raise ValueError("prelabel mutual-independence audit is not clean")
    pass_a, uids_a = _pack(Path(args.pass_a), args.output_namespace)
    pass_b, uids_b = _pack(Path(args.pass_b), args.output_namespace)
    if uids_a != uids_b or len(uids_a) != int(independence.get("count", -1)):
        raise ValueError("two packs do not cover the same exact UID universe")
    observed = {"A": pass_a["validation"]["sha256"], "B": pass_b["validation"]["sha256"]}
    expected = {
        name: str(row.get("validation_sha256") or "")
        for name, row in (independence.get("passes") or {}).items()
    }
    if observed != expected:
        raise ValueError("pack validations differ from the prelabel independence audit")
    repo = Path(args.repo).resolve()
    implementation_paths = [
        repo / "scripts/tools/silver_match_v3/run_claude_pack_labels.py",
        repo / "scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md",
        repo / "scripts/tools/silver_match_v3/ISOLATED_LABELER_NO_DISCOVERY_GUIDE.md",
        repo / "scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json",
    ]
    result = {
        "schema_version": "silver-match-v3-independent-claude-label-execution-freeze-v1",
        "status": "FROZEN_BEFORE_EITHER_CLAUDE_LABEL_PASS",
        "task": args.task,
        "model": args.model,
        "effort": args.effort,
        "output_namespace": args.output_namespace,
        "row_count": len(uids_a),
        "passes": {"A": pass_a, "B": pass_b},
        "prelabel_independence_audit": _ref(independence_path),
        "implementation": [_ref(path) for path in implementation_paths],
        "runtime_contract": {
            "separate_process_per_chunk": True,
            "only_read_tool_available": True,
            "safe_mode": True,
            "no_session_persistence": True,
            "strict_empty_mcp_config": True,
            "pass_outputs_mutually_hidden": True,
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
    parser.add_argument("--pass-a", required=True)
    parser.add_argument("--pass-b", required=True)
    parser.add_argument("--independence-audit", required=True)
    parser.add_argument("--output-namespace", default="claude_sonnet_v1")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args), sort_keys=True))


if __name__ == "__main__":
    main()
