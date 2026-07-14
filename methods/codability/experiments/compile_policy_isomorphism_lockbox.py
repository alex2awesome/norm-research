#!/usr/bin/env python
"""Freeze the final direct-policy candidates before any residual-lockbox scoring."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.compile_residual_isomorphism_bank import BEST_SOURCE


PRIMARY = {
    "N_humor_49": [
        ("confirm_self_contrastive", "rule_contrastive_v0_from_self"),
        ("confirm_behavior_promptfold", "rule_contrastive_v1_from_prompt_selection"),
        ("confirm_behavior_unitfold", "rule_contrastive_v1_from_unit_certification"),
    ],
}
EXPLORATORY = {
    "N_cw_27": [
        ("explore_cw_rank_promptfold",
         "source_then_revision_rank_repair_v1_slot1_from_prompt_selection"),
        ("explore_cw_rank_unitfold",
         "source_then_revision_rank_repair_v1_slot1_from_unit_certification"),
    ],
}


def _frontier_recipe(frontier: dict, cell_id: str, recipe_id: str) -> dict:
    cell = next(row for row in frontier["cells"] if row["cell_id"] == cell_id)
    return next(row for row in cell["ranked_recipes"] if row["recipe_id"] == recipe_id)


def _copy_arm(arm: dict, *, new_id: str, source_bank_sha256: str,
              semantic_audit: str, claim_role: str) -> dict:
    result = copy.deepcopy(arm)
    result["original_arm_id"] = result["id"]
    result["id"] = new_id
    result["source_bank_sha256"] = source_bank_sha256
    result["semantic_audit"] = semantic_audit
    result["claim_role"] = claim_role
    # Final frozen texts are no longer fold-gated at execution. Their training provenance remains
    # separately recorded in original_source_partition.
    result["original_source_partition"] = result.get("source_partition")
    result["source_partition"] = None
    return result


def compile_lockbox(*, source_bank_path: str, rule_bank_path: str,
                    revision_bank_path: str, rule_frontier_path: str,
                    revision_frontier_path: str, packet_manifest_path: str) -> tuple[dict, dict, dict]:
    source = json.loads(Path(source_bank_path).read_text())
    rules = json.loads(Path(rule_bank_path).read_text())
    revisions = json.loads(Path(revision_bank_path).read_text())
    rule_frontier = json.loads(Path(rule_frontier_path).read_text())
    revision_frontier = json.loads(Path(revision_frontier_path).read_text())
    banks = {
        "source": {cell["id"]: {arm["id"]: arm for arm in cell["arms"]}
                   for cell in source["cells"]},
        "rule": {cell["id"]: {arm["id"]: arm for arm in cell["arms"]}
                 for cell in rules["cells"]},
        "revision": {cell["id"]: {arm["id"]: arm for arm in cell["arms"]}
                     for cell in revisions["cells"]},
    }
    hashes = {"source": sha256_file(source_bank_path), "rule": sha256_file(rule_bank_path),
              "revision": sha256_file(revision_bank_path)}

    # Freeze only recipes that crossed the declared valid two-fold public margin frontier.
    for recipe_id in ("rule_contrastive_v0_from_self",
                      "rule_contrastive_v1_from_crossfit_source"):
        row = _frontier_recipe(rule_frontier, "N_humor_49", recipe_id)
        if not row["stable_identity_margin_frontier_improvement"]:
            raise ValueError(f"primary recipe did not pass public frontier: {recipe_id}")
    cw_recipe = _frontier_recipe(
        revision_frontier, "N_cw_27",
        "source_then_revision_rank_repair_v1_slot1_from_crossfit_source")
    if not cw_recipe["stable_identity_margin_frontier_improvement"]:
        raise ValueError("CW exploratory recipe did not pass public point frontier")

    output_cells, selection_cells = [], []
    for source_cell in source["cells"]:
        cell_id = source_cell["id"]
        source_arms = banks["source"][cell_id]
        name = copy.deepcopy(source_arms["name"])
        name["source_partition"] = None
        name["claim_role"] = "small_sparse_baseline"
        incumbent = _copy_arm(
            source_arms[BEST_SOURCE[cell_id]], new_id="incumbent_source",
            source_bank_sha256=hashes["source"], semantic_audit="source-arm-preserved",
            claim_role="intact_text_incumbent")
        arms = [name, incumbent]
        confirmatory_ids, exploratory_ids = [], []
        for new_id, old_id in PRIMARY.get(cell_id, []):
            arms.append(_copy_arm(
                banks["rule"][cell_id][old_id], new_id=new_id,
                source_bank_sha256=hashes["rule"],
                semantic_audit="pass-construct-relevant-no-item-copy",
                claim_role="confirmatory_policy_isomorphism"))
            confirmatory_ids.append(new_id)
        for new_id, old_id in EXPLORATORY.get(cell_id, []):
            arms.append(_copy_arm(
                banks["revision"][cell_id][old_id], new_id=new_id,
                source_bank_sha256=hashes["revision"],
                semantic_audit="behavioral-only-general-story-quality-drift-risk",
                claim_role="exploratory_behavioral_frontier"))
            exploratory_ids.append(new_id)
        output_cells.append({"id": cell_id, "domain": source_cell["domain"],
                             "gi": source_cell["gi"], "construct": source_cell["construct"],
                             "arms": arms})
        selection_cells.append({"cell_id": cell_id,
                                "allowed_arm_ids": [arm["id"] for arm in arms],
                                "confirmatory_candidate_ids": confirmatory_ids,
                                "exploratory_candidate_ids": exploratory_ids,
                                "control_ids": ["name", "incumbent_source"]})
    bank = {
        "schema": "policy_isomorphism_lockbox_arm_bank/v1",
        "status": "frozen-before-residual-lockbox-target-or-executor-scoring",
        "objective": "direct 3B textual reconstruction of the fixed 8B name-only policy",
        "anchor_policy": "no external ground truth or evaluator",
        "source_banks": {key: {"path": path, "sha256": hashes[key]} for key, path in {
            "source": source_bank_path, "rule": rule_bank_path,
            "revision": revision_bank_path}.items()},
        "public_frontiers": {
            "rule": {"path": rule_frontier_path, "sha256": sha256_file(rule_frontier_path)},
            "revision": {"path": revision_frontier_path,
                         "sha256": sha256_file(revision_frontier_path)},
        },
        "selection_rule": ("primary: all exact realizations of valid stable-margin wordplay "
                           "recipes; exploratory: both realizations of the lowest-worst-MAE valid "
                           "CW recipe, excluded from tacit-content claims after semantic audit"),
        "cells": output_cells,
    }
    n_primary = sum(len(row["confirmatory_candidate_ids"]) for row in selection_cells)
    selection = {
        "schema": "policy_isomorphism_lockbox_selection/v1",
        "status": "frozen-before-residual-lockbox-target-or-executor-scoring",
        "arm_bank_sha256": None,
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "lockbox_partition": "residual_lockbox",
        "confirmatory_family_size": n_primary,
        "candidatewise_confidence": 1.0 - 0.05 / n_primary,
        "multiplicity_rule": "Bonferroni across exact confirmatory articulation candidates",
        "cells": selection_cells,
        "claim_boundary": ("Only confirmatory_candidate_ids enter the tacit-content family. CW is "
                           "behavioral-only; other cells supply baselines and target-health checks."),
    }
    manifest = {
        "schema": "policy_isomorphism_lockbox_execution_manifest/v1",
        "status": "frozen-before-residual-lockbox-target-or-executor-scoring",
        "arm_bank_sha256": None,
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "phases": {"lockbox": ["residual_lockbox"]},
        "model_jobs": [
            {"id": "llama8_lockbox_target", "model": "meta-llama/Llama-3.1-8B-Instruct",
             "role": "big_sparse", "arm_policy": "name_only"},
            {"id": "llama3_lockbox_executor", "model": "meta-llama/Llama-3.2-3B-Instruct",
             "role": "small", "arm_policy": "all"},
        ],
        "primary_estimand": "direct 3B articulated to 8B sparse policy isomorphism",
        "lockbox_status": "authorized only with the exact frozen selection artifact",
    }
    return bank, selection, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--rule-bank", required=True)
    parser.add_argument("--revision-bank", required=True)
    parser.add_argument("--rule-frontier", required=True)
    parser.add_argument("--revision-frontier", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--out-bank", required=True)
    parser.add_argument("--out-selection", required=True)
    parser.add_argument("--out-manifest", required=True)
    args = parser.parse_args()
    bank, selection, manifest = compile_lockbox(
        source_bank_path=args.source_bank, rule_bank_path=args.rule_bank,
        revision_bank_path=args.revision_bank, rule_frontier_path=args.rule_frontier,
        revision_frontier_path=args.revision_frontier,
        packet_manifest_path=args.packet_manifest)
    bank_path = Path(args.out_bank)
    bank_path.write_text(json.dumps(bank, indent=1))
    bank_sha = sha256_file(bank_path)
    selection["arm_bank_sha256"] = bank_sha
    manifest["arm_bank_sha256"] = bank_sha
    selection_path = Path(args.out_selection)
    manifest_path = Path(args.out_manifest)
    selection_path.write_text(json.dumps(selection, indent=1))
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"bank": str(bank_path), "bank_sha256": bank_sha,
                      "selection": str(selection_path),
                      "selection_sha256": sha256_file(selection_path),
                      "manifest": str(manifest_path),
                      "manifest_sha256": sha256_file(manifest_path),
                      "confirmatory_family_size": selection["confirmatory_family_size"],
                      "candidatewise_confidence": selection["candidatewise_confidence"]}, indent=1))


if __name__ == "__main__":
    main()
