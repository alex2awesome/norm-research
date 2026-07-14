#!/usr/bin/env python
"""Select source-only name arms on the isolated public partition; never read lockbox shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.policy_data import (
    PUBLIC_DEVELOPMENT_PARTITIONS,
    _align_orbit,
    _average_repetitions,
    _orbits,
    load_public_index,
    require_partition,
)
from methods.codability.experiments.target_articulation_frontier import (
    SCORE_KEY,
    bootstrap_orbit_values,
    orbit_recovery,
    paired_substitution_test,
)


def _ci(values: np.ndarray, confidence: float = 0.95) -> list[float] | None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))]


def choose_candidate(rows: list[dict], *, tie_width: float = 0.01) -> dict:
    eligible = [row for row in rows if row["selection_eligible"]]
    pool = eligible or rows
    if not pool:
        raise ValueError("no source candidates")
    best_lower = max(row["score_CI"][0] for row in pool)
    tied = [row for row in pool if row["score_CI"][0] >= best_lower - tie_width]
    chosen = min(tied, key=lambda row: (row["semantic_content_word_count"], row["arm_id"]))
    return {"chosen": chosen, "had_eligible_candidate": bool(eligible),
            "best_lower_bound": best_lower, "tie_width": tie_width,
            "rule": ("maximize lower oriented-recovery bound among polarity/signature-eligible "
                     "source arms; within tie width choose fewer content words then arm id")}


def _candidate_report(target: np.ndarray, orbit: dict[str, np.ndarray], *,
                      samples: np.ndarray, n_boot: int) -> dict:
    point = orbit_recovery(target, orbit, divergence="tvd", min_target_information=0.01)
    if not point.get("valid"):
        return {"valid": False, "error": point.get("error"), "recovery": point}
    draws = bootstrap_orbit_values(target, orbit, samples, divergence="tvd",
                                   min_target_information=0.01)
    score_ci = _ci(draws[SCORE_KEY])
    rho_ci = _ci(draws["spearman"])
    robust = point["robust"]
    return {"valid": True, "recovery": point, "score_CI": score_ci, "rho_CI": rho_ci,
            "n_boot": n_boot,
            "selection_eligible": bool(robust["all_positive_polarity"] and rho_ci
                                       and rho_ci[0] >= 0.5)}


def select(*, target_shard_root: str, executor_shard_root: str,
           arm_bank_path: str, packet_manifest_path: str,
           partition: str = "residual_prompt_selection", n_boot: int = 5000,
           seed: int = 1207, audit_all_source_arms: bool = False) -> dict:
    require_partition(
        partition,
        allowed=PUBLIC_DEVELOPMENT_PARTITIONS,
        operation="fresh name-arm selection",
    )
    target_index = load_public_index(target_shard_root, partition)
    executor_index = load_public_index(executor_shard_root, partition)
    arm_bank = json.loads(Path(arm_bank_path).read_text())
    output_rows = []
    for cell in arm_bank["cells"]:
        domain, cell_id = cell["domain"], cell["id"]
        small = _average_repetitions(executor_index[("llama3_small", domain)])
        big = _average_repetitions(executor_index[("llama8_big_sparse", domain)])
        small_orbits = _orbits(small["scores"], small["meta"], cell_id=cell_id)
        big_orbits = _orbits(big["scores"], big["meta"], cell_id=cell_id)
        for target_job in cell["target_model_jobs"]:
            target_bundle = _average_repetitions(target_index[(target_job, domain)])
            target_orbit = _orbits(target_bundle["scores"], target_bundle["meta"],
                                   cell_id=cell_id)["target"]
            target = np.mean(list(target_orbit.values()), axis=0)
            target_hashes = target_bundle["hashes"]
            small_aligned = {arm: _align_orbit(orbit, small["hashes"], target_hashes)
                             for arm, orbit in small_orbits.items()}
            big_name = _align_orbit(big_orbits["name"], big["hashes"], target_hashes)
            rng_seed = seed + int(hashlib.sha256(f"{cell_id}|{target_job}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(rng_seed)
            samples = rng.integers(0, len(target), size=(n_boot, len(target)))
            arm_meta = {arm["id"]: arm for arm in cell["arms"]}
            candidates = []
            for arm_id, orbit in small_aligned.items():
                if arm_meta[arm_id]["provenance"] != "source_telling":
                    continue
                report = _candidate_report(target, orbit, samples=samples, n_boot=n_boot)
                candidates.append({"arm_id": arm_id,
                                   "semantic_content_word_count": arm_meta[arm_id][
                                       "semantic_content_word_count"], **report})
            decision = choose_candidate([row for row in candidates if row.get("valid")])
            chosen = decision["chosen"]
            suffix = chosen["arm_id"].removeprefix("source_")
            controls = [f"control_wrong_{suffix}", f"control_inert_{suffix}"]
            paired = paired_substitution_test(
                target, small_sparse_orbit=small_aligned["name"],
                big_sparse_orbit=big_name, articulated_orbit=small_aligned[chosen["arm_id"]],
                n_boot=n_boot, seed=rng_seed, min_target_information=0.01)
            source_arm_atlas = None
            if audit_all_source_arms:
                # This is explicitly a public-development atlas, not a second selection rule and
                # not confirmation.  It diagnoses whether maximizing fixed-target recovery is
                # misaligned with the stricter two-sided cross-reader isomorphism objective.
                source_arm_atlas = {
                    arm_id: paired_substitution_test(
                        target, small_sparse_orbit=small_aligned["name"],
                        big_sparse_orbit=big_name, articulated_orbit=small_aligned[arm_id],
                        n_boot=n_boot, seed=rng_seed, min_target_information=0.01)
                    for arm_id in sorted(small_aligned)
                    if arm_meta[arm_id]["provenance"] == "source_telling"
                }
            specificity = {}
            for control in controls:
                specificity[control] = paired_substitution_test(
                    target, small_sparse_orbit=small_aligned["name"],
                    big_sparse_orbit=big_name,
                    articulated_orbit=small_aligned[chosen["arm_id"]],
                    control_orbit=small_aligned[control], n_boot=n_boot,
                    seed=rng_seed, min_target_information=0.01)
            output_rows.append({
                "cell_id": cell_id, "domain": domain, "target_model_job": target_job,
                "selected_arm_id": chosen["arm_id"], "matched_control_ids": controls,
                "cuf_status": "U1-U2-fresh-scores-available; U3-U5-not-yet-certified",
                "selection": decision, "all_source_candidates": candidates,
                "development_substitution": paired,
                "development_specificity": specificity,
                "public_development_source_arm_atlas": source_arm_atlas,
                "target_public_shards": target_bundle["shard_sha256"],
                "small_public_shards": small["shard_sha256"],
                "big_public_shards": big["shard_sha256"],
                "n_public_items": len(target), "bootstrap_seed": rng_seed,
            })
    return {
        "schema": "fresh_name_arm_selection/v1", "partition": partition,
        "arm_bank_sha256": sha256_file(arm_bank_path),
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "n_boot": n_boot, "seed": seed, "cells": output_rows,
        "all_source_arm_audit": bool(audit_all_source_arms),
        "lockbox_status": "selection frozen; lockbox shards were not opened",
        "claim_boundary": ("Selection targets confirmation candidates. Raw words are not certified "
                           "units; U3-U5 remain required for a finite certified debt claim."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-shard-root", required=True)
    parser.add_argument("--executor-shard-root", required=True)
    parser.add_argument("--arm-bank", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--partition", default="residual_prompt_selection")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument("--audit-all-source-arms", action="store_true",
                        help="persist an exploratory public-development substitution atlas")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = select(target_shard_root=args.target_shard_root,
                    executor_shard_root=args.executor_shard_root,
                    arm_bank_path=args.arm_bank, packet_manifest_path=args.packet_manifest,
                    partition=args.partition, n_boot=args.n_boot, seed=args.seed,
                    audit_all_source_arms=args.audit_all_source_arms)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"out": str(out), "n_target_cells": len(report["cells"]),
                      "selections": [{"cell": row["cell_id"],
                                      "target": row["target_model_job"],
                                      "arm": row["selected_arm_id"]}
                                     for row in report["cells"]]}, indent=1))


if __name__ == "__main__":
    main()
