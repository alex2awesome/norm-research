#!/usr/bin/env python3
"""Freeze the truth-hidden three-order R4 proposal run on PR verifier-dev.

This plan is deliberately independent of verifier-dev labels.  It binds the
pre-label identity freeze, the selected K50 retriever output, the already
frozen optimize-authored R4 adjudicator prompt, the exact Gemma snapshot, and
all three candidate-order runs before any proposal is generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


ORDERS = ("original", "hashed", "reverse")
POLICY_SCHEMA = "silver-match-v3-press-releases-verifier-dev-policy-amendment-v3"


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    values = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in values or len(values) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return values


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "policy",
            "manifest",
            "identity_freeze",
            "identities",
            "candidate_source",
            "candidate_subset",
            "candidate_subset_meta",
            "candidate_coverage_audit",
            "retriever_selection",
            "adjudicator_prompt",
            "adjudicator_prompt_meta",
            "runner",
            "model_inventory",
        )
    }
    output_root = Path(args.output_root).resolve()
    output = Path(args.output).resolve()
    if output.exists() or output_root.exists():
        raise FileExistsError("refusing to overwrite or reuse an R4 verifier-dev run")

    policy = json.loads(paths["policy"].read_text(encoding="utf-8"))
    base_ref = policy.get("base_policy") or {}
    base_path = Path(str(base_ref.get("path") or "")).resolve()
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or policy.get("status") != "FROZEN_BEFORE_FRESH_DEV_PREDICTIONS_LABELS_OR_TRUTH"
        or policy.get("task") != "press-releases"
        or not base_path.is_file()
        or sha256_file(base_path) != str(base_ref.get("sha256") or "")
    ):
        raise ValueError("unsupported or drifted verifier-dev policy")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    canonical = base.get("canonical_inputs") or {}

    manifest_sha = sha256_file(paths["manifest"])
    identity_sha = sha256_file(paths["identities"])
    identity_freeze_sha = sha256_file(paths["identity_freeze"])
    if (
        ((canonical.get("manifest") or {}).get("sha256")) != manifest_sha
        or ((canonical.get("fresh_dev_identities") or {}).get("sha256"))
        != identity_sha
        or ((canonical.get("fresh_dev_identity_freeze") or {}).get("sha256"))
        != identity_freeze_sha
    ):
        raise ValueError("manifest or fixed verifier-dev identity differs from policy")
    identity_freeze = json.loads(paths["identity_freeze"].read_text(encoding="utf-8"))
    identities = _index(paths["identities"])
    if (
        identity_freeze.get("status")
        != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or identity_freeze.get("task") != "press-releases"
        or identity_freeze.get("role") != "verifier_dev"
        or int(identity_freeze.get("selected_count") or -1) != 300
        or len(identities) != 300
        or any(
            row.get("task") != "press-releases"
            or row.get("gepa_role") != "verifier_dev"
            or row.get("upstream_split") != "train"
            for row in identities.values()
        )
    ):
        raise ValueError("fixed verifier-dev identity freeze is invalid")

    subset = _index(paths["candidate_subset"])
    subset_meta = json.loads(paths["candidate_subset_meta"].read_text(encoding="utf-8"))
    source_sha = sha256_file(paths["candidate_source"])
    subset_sha = sha256_file(paths["candidate_subset"])
    bank_sha = str(canonical.get("bank_source_sha256") or "")
    if (
        set(subset) != set(identities)
        or any(
            row.get("task") != "press-releases"
            or row.get("bank_source_sha256") != bank_sha
            or len(row.get("candidates") or []) != 50
            or ""
            in {
                str(value.get("metric_id") or "")
                for value in row.get("candidates") or []
            }
            or len(
                {
                    str(value.get("metric_id") or "")
                    for value in row.get("candidates") or []
                }
            )
            != 50
            for row in subset.values()
        )
        or subset_meta.get("schema_version")
        != "silver-match-v3-jsonl-reference-subset-v1"
        or ((subset_meta.get("inputs") or {}).get("input") or {}).get("sha256")
        != source_sha
        or ((subset_meta.get("inputs") or {}).get("reference") or {}).get("sha256")
        != identity_sha
        or (subset_meta.get("output") or {}).get("sha256") != subset_sha
        or int(subset_meta.get("output_count") or -1) != 300
    ):
        raise ValueError("truth-hidden K50 subset is incomplete or not identity-bound")

    coverage = json.loads(paths["candidate_coverage_audit"].read_text(encoding="utf-8"))
    coverage_inputs = coverage.get("candidate_inputs") or {}
    if not any(
        Path(str(candidate)).resolve() == paths["candidate_source"]
        and str((meta or {}).get("sha256") or "") == source_sha
        for candidate, meta in coverage_inputs.items()
    ):
        raise ValueError("candidate source is not bound by the production coverage audit")
    selection = json.loads(paths["retriever_selection"].read_text(encoding="utf-8"))
    chosen = selection.get("chosen") or selection.get("selection") or {}
    chosen_kind = str(
        chosen.get("kind")
        or chosen.get("chosen_kind")
        or (selection.get("selection") or {}).get("chosen_kind")
        or ""
    )
    if chosen_kind != "adapter":
        raise ValueError("PR verifier-dev candidates do not use the selected adapter")

    prompt_meta = json.loads(paths["adjudicator_prompt_meta"].read_text(encoding="utf-8"))
    prompt_sha = sha256_file(paths["adjudicator_prompt"])
    if (
        prompt_meta.get("status") != "MATERIALIZED_WITHOUT_PROMPT_MUTATION"
        or int(prompt_meta.get("variant_count") or -1) != 1
        or (prompt_meta.get("prompt") or {}).get("sha256") != prompt_sha
        or prompt_meta.get("select_material_joined_at_authoring") is not False
    ):
        raise ValueError("R4 adjudicator prompt is not the frozen optimize-authored prompt")

    model_inventory = json.loads(paths["model_inventory"].read_text(encoding="utf-8"))
    if args.model_snapshot not in json.dumps(model_inventory, sort_keys=True):
        raise ValueError("requested Gemma snapshot is absent from the frozen inventory")

    output_root.mkdir(parents=True, exist_ok=False)
    outputs = {order: str(output_root / f"{order}.jsonl") for order in ORDERS}
    plan = {
        "schema_version": "silver-match-v3-pr-verifier-dev-r4-proposal-plan-v1",
        "status": "FROZEN_BEFORE_DEV_PROPOSAL_INFERENCE",
        "task": "press-releases",
        "role": "verifier_dev_truth_hidden_proposals",
        "row_count": 300,
        "candidate_depth": 50,
        "orders": list(ORDERS),
        "model": args.model_snapshot,
        "rendering": {
            "context_chars": 1200,
            "description_chars": 260,
            "example_chars": 80,
            "max_examples": 0,
            "max_tokens": 220,
            "max_model_len": 8192,
            "batch_size": 32,
            "gpu_memory_utilization": 0.88,
            "seed": 2026071311,
        },
        "inputs": {name: _artifact(path) for name, path in paths.items()},
        "outputs": outputs,
        "contracts": {
            "fixed_identity_not_substituted": True,
            "candidate_and_proposals_hidden_from_truth_labelers": True,
            "truth_labels_or_predictions_read_by_plan": False,
            "all_three_orders_frozen_before_inference": True,
            "strict_proposals_require_exact_match_leaf_consensus": True,
            "prompt_may_not_be_edited_from_dev_results": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**plan, "plan_sha256": sha256_file(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "policy",
        "manifest",
        "identity-freeze",
        "identities",
        "candidate-source",
        "candidate-subset",
        "candidate-subset-meta",
        "candidate-coverage-audit",
        "retriever-selection",
        "adjudicator-prompt",
        "adjudicator-prompt-meta",
        "runner",
        "model-inventory",
        "model-snapshot",
        "output-root",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(freeze(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
