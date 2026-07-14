#!/usr/bin/env python
"""Certify cross-fitted item-adaptive ostension against the fixed larger-Llama policy."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.policy_data import (
    _align_orbit,
    _average_repetitions,
    _orbits,
    load_public_index,
)
from methods.codability.experiments.policy_isomorphism import (
    _orbit_point,
    certify_policy_isomorphism,
)
from methods.codability.experiments.synthesize_residual_policy_revisions import identity_loss


PARTITIONS = ("residual_prompt_selection", "residual_unit_certification")
MODEL_JOB = "llama3_adaptive_ostensive_tf"


def recipe_family(arm_id: str) -> str:
    for partition in PARTITIONS:
        suffix = f"_from_{partition.removeprefix('residual_')}"
        if arm_id.endswith(suffix):
            return arm_id.removesuffix(suffix)
    return arm_id


def _delta(base: dict, candidate: dict) -> dict:
    return {
        "mae_gain": float(base["mae_tvd"] - candidate["mae_tvd"]),
        "rho_gain": float(candidate["spearman"] - base["spearman"]),
        "flip_gain": float(base["binary_flip_rate"] - candidate["binary_flip_rate"]),
        "bias_gain": float(base["absolute_bias"] - candidate["absolute_bias"]),
    }


def _baselines(*, partition: str, teaching_partition: str, target_hashes: list[str],
               source_index: dict, rule_index: dict) -> dict[str, dict]:
    source = _average_repetitions(source_index[("llama3_small", "humor")])
    source_orbits = _orbits(source["scores"], source["meta"], cell_id="N_humor_49")
    rules = _average_repetitions(rule_index[("llama3_target_policy_rules", "humor")])
    rule_orbits = _orbits(rules["scores"], rules["meta"], cell_id="N_humor_49")
    teaching_suffix = teaching_partition.removeprefix("residual_")
    values = {
        "demos_only": (source_orbits["name"], source["hashes"]),
        "source_explanation": (source_orbits["source_explanation"], source["hashes"]),
        "self_contrastive": (rule_orbits["rule_contrastive_v0_from_self"], rules["hashes"]),
        "behavior_contrastive": (
            rule_orbits[f"rule_contrastive_v1_from_{teaching_suffix}"], rules["hashes"]),
    }
    return {key: _align_orbit(orbit, hashes, target_hashes)
            for key, (orbit, hashes) in values.items()}


def run(*, bank_path: str, adaptive_shard_root: str, target_shard_root: str,
        rule_shard_root: str, n_boot: int = 2000, seed: int = 20260726) -> dict:
    bank = json.loads(Path(bank_path).read_text())
    evaluations = {row["evaluation_partition"]: row for row in bank["evaluations"]}
    fold_rows = {}
    artifacts = []
    for fold_index, partition in enumerate(PARTITIONS):
        evaluation = evaluations[partition]
        target_index = load_public_index(target_shard_root, partition)
        adaptive_index = load_public_index(adaptive_shard_root, partition)
        rule_index = load_public_index(rule_shard_root, partition)
        target_data = _average_repetitions(target_index[("llama8_big_sparse", "humor")])
        target = _orbits(target_data["scores"], target_data["meta"],
                         cell_id="N_humor_49")["name"]
        target_hashes = target_data["hashes"]
        sparse_data = _average_repetitions(target_index[("llama3_small", "humor")])
        sparse = _align_orbit(
            _orbits(sparse_data["scores"], sparse_data["meta"],
                    cell_id="N_humor_49")["name"],
            sparse_data["hashes"], target_hashes)
        adaptive_data = _average_repetitions(adaptive_index[(MODEL_JOB, "humor")])
        adaptive_orbits = _orbits(adaptive_data["scores"], adaptive_data["meta"],
                                  cell_id="N_humor_49")
        baselines = _baselines(
            partition=partition,
            teaching_partition=evaluation["teaching_partition"],
            target_hashes=target_hashes, source_index=target_index, rule_index=rule_index)
        baseline_points = {key: _orbit_point(target, orbit)["candidate_robust"]
                           for key, orbit in baselines.items()}
        arm_specs = {arm["id"]: arm for arm in evaluation["arms"]}
        rows = []
        for arm_index, (arm_id, orbit) in enumerate(sorted(adaptive_orbits.items())):
            arm = arm_specs[arm_id]
            aligned = _align_orbit(orbit, adaptive_data["hashes"], target_hashes)
            point = _orbit_point(target, aligned)
            candidate = point["candidate_robust"]
            base = baseline_points[arm["parent_id"]]
            delta = _delta(base, candidate)
            certificate = certify_policy_isomorphism(
                target, aligned, sparse_orbit=sparse, n_boot=n_boot,
                seed=seed + 10_007 * fold_index + arm_index)
            rows.append({
                "recipe_id": recipe_family(arm_id),
                "actual_arm_id": arm_id,
                "teaching_partition": evaluation["teaching_partition"],
                "parent_id": arm["parent_id"],
                "retrieval_id": arm["retrieval_id"],
                "semantic_content_word_count": arm["semantic_content_word_count"],
                "parent_robust": base,
                "candidate_robust": candidate,
                "candidate_quotient": point["quotient"],
                "identity_loss": identity_loss(point),
                "delta_from_direct_parent": delta,
                "point_pareto_improves_parent_all_four": bool(
                    all(delta[key] >= 0 for key in (
                        "mae_gain", "rho_gain", "flip_gain", "bias_gain"))
                    and (delta["mae_gain"] > 0 or delta["rho_gain"] > 0)),
                "certificate": certificate,
            })
        fold_rows[partition] = rows
        artifacts.append({
            "partition": partition,
            "adaptive_shards": adaptive_data["shard_sha256"],
            "target_shards": target_data["shard_sha256"],
            "n_items": len(target_hashes),
        })

    grouped = defaultdict(list)
    for partition, rows in fold_rows.items():
        for row in rows:
            grouped[row["recipe_id"]].append({"partition": partition, **row})
    stable = []
    for recipe_id, rows in grouped.items():
        if len(rows) != len(PARTITIONS):
            continue
        deltas = [row["delta_from_direct_parent"] for row in rows]
        stable.append({
            "recipe_id": recipe_id,
            "folds": rows,
            "max_identity_loss": float(max(row["identity_loss"] for row in rows)),
            "mean_identity_loss": float(np.mean([row["identity_loss"] for row in rows])),
            "improves_mae_and_rho_over_parent_both": bool(
                all(delta["mae_gain"] > 0 and delta["rho_gain"] > 0 for delta in deltas)),
            "point_pareto_improves_parent_all_four_both": bool(
                all(row["point_pareto_improves_parent_all_four"] for row in rows)),
            "policy_isomorphic_both": bool(
                all(row["certificate"]["policy_isomorphic"] for row in rows)),
        })
    stable.sort(key=lambda row: (row["max_identity_loss"], row["mean_identity_loss"],
                                 row["recipe_id"]))
    return {
        "schema": "adaptive_ostensive_isomorphism/v1",
        "status": "public-crossfit-development; not confirmation",
        "estimand": ("direct 3B policy reconstruction from item-adaptive explicit examples of "
                      "the fixed 8B name-only policy"),
        "model_family": bank["model_family"],
        "anchor_policy": bank["anchor_policy"],
        "bank": {"path": bank_path, "sha256": sha256_file(bank_path)},
        "shard_roots": {"adaptive": adaptive_shard_root, "target": target_shard_root,
                        "rules": rule_shard_root},
        "bootstrap": {"n": n_boot, "seed": seed, "confidence": 0.95},
        "artifacts": artifacts,
        "fold_rows": fold_rows,
        "stable_recipes": stable,
        "summary": {
            "n_fold_candidates": sum(len(rows) for rows in fold_rows.values()),
            "n_stable_recipes": len(stable),
            "n_improve_mae_and_rho_both": sum(
                row["improves_mae_and_rho_over_parent_both"] for row in stable),
            "n_point_pareto_parent_both": sum(
                row["point_pareto_improves_parent_all_four_both"] for row in stable),
            "n_policy_isomorphic_both": sum(row["policy_isomorphic_both"] for row in stable),
        },
        "claim_boundary": ("All target-policy examples are cross-fitted from the opposite public "
                           "fold. This is an exploratory atlas; a chosen recipe must be frozen "
                           "before any new untouched-panel claim."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", required=True)
    parser.add_argument("--adaptive-shard-root", required=True)
    parser.add_argument("--target-shard-root", required=True)
    parser.add_argument("--rule-shard-root", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(
        bank_path=args.bank, adaptive_shard_root=args.adaptive_shard_root,
        target_shard_root=args.target_shard_root, rule_shard_root=args.rule_shard_root,
        n_boot=args.n_boot, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), **result["summary"]}, indent=1))


if __name__ == "__main__":
    main()
