#!/usr/bin/env python3
"""Freeze a proposal-hidden full-bank Codex verifier pack for the PR frontier."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


FORBIDDEN_ITEM_FIELDS = {
    "target",
    "decision",
    "metric_id",
    "proposal_metric_id",
    "candidate_ids",
    "candidates",
    "prediction",
    "reason",
    "raw_response",
}


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    values = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in values or len(values) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return values


def _ref(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-freeze", required=True)
    parser.add_argument("--frontier-candidates", required=True)
    parser.add_argument("--source-validation", required=True)
    parser.add_argument("--source-items", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--rejection-freeze", required=True)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--seed", type=int, default=2026071327)
    parser.add_argument("--chunk-size", type=int, default=22)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--expected-tokens", type=int, default=116000)
    parser.add_argument("--token-budget", type=int, default=132000)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.chunk_size > 25:
        parser.error("--chunk-size must be in [1, 25]")
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "pair_freeze",
            "frontier_candidates",
            "source_validation",
            "source_items",
            "source_bank",
            "rejection_freeze",
            "guide",
            "schema",
        )
    }
    workspace = Path(args.workspace).resolve()
    plan_path = Path(args.plan_output).resolve()
    if workspace.exists() or plan_path.exists():
        raise FileExistsError("refusing to overwrite verifier workspace or plan")

    pair_freeze = json.loads(paths["pair_freeze"].read_text(encoding="utf-8"))
    rejected = json.loads(paths["rejection_freeze"].read_text(encoding="utf-8"))
    source_validation = json.loads(
        paths["source_validation"].read_text(encoding="utf-8")
    )
    if (
        pair_freeze.get("status") != "FROZEN_BALANCED_BEFORE_VERIFIER_INFERENCE"
        or pair_freeze.get("task") != "press-releases"
        or rejected.get("status")
        != "REJECTED_APPEND_ONLY_NO_PROMOTION_OR_RETUNING_ON_CONSUMED_DEV"
        or int(rejected.get("consumed_verifier_dev_count", -1))
        != int(pair_freeze.get("selected_count", -2))
        or source_validation.get("truth_hidden") is not True
        or source_validation.get("task") != "press-releases"
        or (source_validation.get("outputs") or {}).get("items", {}).get("sha256")
        != sha256_file(paths["source_items"])
        or (source_validation.get("outputs") or {}).get("bank", {}).get("sha256")
        != sha256_file(paths["source_bank"])
        or (pair_freeze.get("outputs") or {}).get("candidates", {}).get("sha256")
        != sha256_file(paths["frontier_candidates"])
    ):
        raise ValueError("fallback inputs are not the frozen rejected PR frontier")

    frontier = _index(paths["frontier_candidates"])
    source = _index(paths["source_items"])
    if len(frontier) != int(pair_freeze["selected_count"]) or not set(frontier) <= set(
        source
    ):
        raise ValueError("frontier UID coverage differs from source pack")
    rng = random.Random(args.seed)
    uids = sorted(frontier)
    rng.shuffle(uids)
    items = []
    for uid in uids:
        row = source[uid]
        if FORBIDDEN_ITEM_FIELDS & set(row):
            raise ValueError(f"source item exposes labels/proposals: {uid}")
        items.append(
            {
                **row,
                "split": "dev",
                "predeclared_split": "dev",
                "gepa_role": "verifier_dev_codex_fallback",
                "truth_hidden": True,
                "candidate_proposals_hidden": True,
            }
        )
    bank = json.loads(paths["source_bank"].read_text(encoding="utf-8"))
    if bank.get("task") != "press-releases" or not bank.get("metrics"):
        raise ValueError("invalid PR bank")
    metrics = list(bank["metrics"])
    rng.shuffle(metrics)
    bank = {**bank, "metrics": metrics}

    pack = workspace / "pack"
    chunks_root = pack / "chunks"
    scripts_root = workspace / "scripts" / "tools" / "silver_match_v3"
    schema_root = scripts_root / "schemas"
    chunks_root.mkdir(parents=True)
    schema_root.mkdir(parents=True)
    items_path, bank_path = pack / "items.jsonl", pack / "bank.json"
    write_jsonl(items_path, items)
    bank_path.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    guide_output = scripts_root / "INDEPENDENT_LABELING_GUIDE.md"
    schema_output = schema_root / paths["schema"].name
    shutil.copy2(paths["guide"], guide_output)
    shutil.copy2(paths["schema"], schema_output)
    chunks = []
    for start in range(0, len(items), args.chunk_size):
        chunk = chunks_root / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(chunk, items[start : start + args.chunk_size])
        chunks.append(chunk)
    validation = {
        "schema_version": "silver-match-v3-pr-independent-codex-verifier-pack-v1",
        "status": "FROZEN_PROPOSAL_TARGET_AND_TRUTH_HIDDEN_BEFORE_LABELING",
        "task": "press-releases",
        "role": "verifier_dev_codex_fallback",
        "count": len(items),
        "source_groups": len({str(row["source_group"]) for row in items}),
        "bank_metric_count": len(metrics),
        "bank_source_sha256": source_validation["bank_source_sha256"],
        "truth_hidden": True,
        "candidate_proposals_hidden": True,
        "target_balance_hidden": True,
        "prior_labels_predictions_mi_and_outcomes_not_read_by_labeler": True,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "inputs": {
            "source_validation": _ref(paths["source_validation"]),
            "pair_freeze": _ref(paths["pair_freeze"]),
            "rejection_freeze": _ref(paths["rejection_freeze"]),
        },
        "outputs": {
            "items": _ref(items_path),
            "bank": _ref(bank_path),
            "chunks": {str(path): sha256_file(path) for path in chunks},
            "guide": _ref(guide_output),
            "schema": _ref(schema_output),
        },
    }
    validation_path = pack / "validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plan = {
        "schema_version": "silver-match-v3-pr-independent-codex-verifier-plan-v1",
        "status": "FROZEN_BEFORE_ANY_FALLBACK_CODEX_LABEL",
        "task": "press-releases",
        "role": "verifier_dev_codex_fallback",
        "frontier_count": len(items),
        "bank_metric_count": len(metrics),
        "chunk_count": len(chunks),
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "cost_estimate": {
            "codex_calls": len(chunks),
            "expected_tokens": args.expected_tokens,
            "token_budget": args.token_budget,
            "basis": "65891 observed tokens per prior 25-item PR full-bank chunk, scaled to 44 items with overhead",
        },
        "keep_rule": {
            "name": "proposal_hidden_independent_exact_match",
            "retain_only_if_label_decision": "MATCH",
            "retain_only_if_independent_metric_id_equals_hidden_proposal": True,
            "accepted_confidences": ["high", "medium"],
            "other_decisions_or_leafs": "ABSTAIN",
        },
        "gate_rule": {
            "minimum_retained_proposals": 20,
            "minimum_exact_precision": 0.9,
            "minimum_wilson_lower_95": 0.8,
            "all_three_must_pass": True,
        },
        "contracts": {
            "proposal_not_in_label_workspace": True,
            "truth_or_target_not_in_label_workspace": True,
            "target_balance_not_in_label_workspace": True,
            "full_bank_required_for_every_item": True,
            "strict_transcript_audit_required": True,
            "score_exactly_once_without_tuning": True,
            "gemma_v4_ineligible_and_advisory_only": True,
            "successful_dev_gate_requires_new_independent_blind_audit": True,
        },
        "inputs": {name: _ref(path) for name, path in paths.items()},
        "workspace": str(workspace),
        "pack_validation": _ref(validation_path),
        "pack_outputs": validation["outputs"],
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {**plan, "plan_sha256": sha256_file(plan_path)}, sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
