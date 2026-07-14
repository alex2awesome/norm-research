#!/usr/bin/env python
"""Paired-bootstrap promising cross-fitted rules against the intact-text incumbent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.compile_residual_isomorphism_bank import BEST_SOURCE
from methods.codability.experiments.policy_data import (
    _align_orbit,
    _average_repetitions,
    _orbits,
    load_public_index,
)
from methods.codability.experiments.policy_isomorphism import (
    _bootstrap_orbit,
    _ci,
    _orbit_point,
)


SECONDARY_MARGINS = {"spearman": 0.05, "binary_flip_rate": 0.02,
                     "absolute_bias": 0.02}


def paired_frontier_draws(target_orbit: dict, incumbent_orbit: dict,
                          challenger_orbit: dict, *, samples: np.ndarray) -> dict:
    incumbent = _bootstrap_orbit(target_orbit, incumbent_orbit, samples)["candidate"]
    challenger = _bootstrap_orbit(target_orbit, challenger_orbit, samples)["candidate"]
    return {
        "mae_gain": incumbent["mae_tvd"] - challenger["mae_tvd"],
        "rho_gain": challenger["spearman"] - incumbent["spearman"],
        "flip_gain": incumbent["binary_flip_rate"] - challenger["binary_flip_rate"],
        "bias_gain": incumbent["absolute_bias"] - challenger["absolute_bias"],
    }


def certify_frontier_gain(target_orbit: dict, incumbent_orbit: dict,
                          challenger_orbit: dict, *, n_boot: int,
                          seed: int, confidence: float = 0.95
                          ) -> tuple[dict, dict[str, np.ndarray]]:
    incumbent = _orbit_point(target_orbit, incumbent_orbit)["candidate_robust"]
    challenger = _orbit_point(target_orbit, challenger_orbit)["candidate_robust"]
    rng = np.random.default_rng(seed)
    n_items = len(next(iter(target_orbit.values())))
    samples = rng.integers(0, n_items, size=(n_boot, n_items))
    draws = paired_frontier_draws(target_orbit, incumbent_orbit, challenger_orbit,
                                  samples=samples)
    points = {
        "mae_gain": incumbent["mae_tvd"] - challenger["mae_tvd"],
        "rho_gain": challenger["spearman"] - incumbent["spearman"],
        "flip_gain": incumbent["binary_flip_rate"] - challenger["binary_flip_rate"],
        "bias_gain": incumbent["absolute_bias"] - challenger["absolute_bias"],
    }
    estimates = {key: {"point": float(value), "CI": _ci(draws[key], confidence)}
                 for key, value in points.items()}
    estimates["incumbent"] = incumbent
    estimates["challenger"] = challenger
    estimates["gates"] = {
        "mae_superior": estimates["mae_gain"]["CI"][0] > 0,
        "rho_noninferior": estimates["rho_gain"]["CI"][0]
        >= -SECONDARY_MARGINS["spearman"],
        "flip_noninferior": estimates["flip_gain"]["CI"][0]
        >= -SECONDARY_MARGINS["binary_flip_rate"],
        "bias_noninferior": estimates["bias_gain"]["CI"][0]
        >= -SECONDARY_MARGINS["absolute_bias"],
    }
    estimates["paired_frontier_improvement"] = all(estimates["gates"].values())
    estimates["confidence"] = confidence
    return estimates, draws


def run(*, frontier_path: str, target_shard_root: str, incumbent_shard_root: str,
        challenger_shard_root: str, target_job: str, incumbent_job: str,
        challenger_job: str, n_boot: int, seed: int) -> dict:
    frontier = json.loads(Path(frontier_path).read_text())
    partitions = frontier["folds"]
    target_indexes = {p: load_public_index(target_shard_root, p) for p in partitions}
    incumbent_indexes = {p: load_public_index(incumbent_shard_root, p) for p in partitions}
    challenger_indexes = {p: load_public_index(challenger_shard_root, p) for p in partitions}
    cells = []
    for cell_index, cell in enumerate(frontier["cells"]):
        cell_id = cell["cell_id"]
        candidates = [row for row in cell["ranked_recipes"]
                      if row["stable_identity_margin_frontier_improvement"]]
        candidate_rows = []
        for candidate_index, candidate in enumerate(candidates):
            fold_reports, fold_draws = [], []
            for fold_index, fold in enumerate(partitions):
                domain = cell["domain"]
                incumbent_data = _average_repetitions(
                    incumbent_indexes[fold][(incumbent_job, domain)])
                incumbent_orbits = _orbits(incumbent_data["scores"], incumbent_data["meta"],
                                           cell_id=cell_id)
                target_data = _average_repetitions(target_indexes[fold][(target_job, domain)])
                challenger_data = _average_repetitions(
                    challenger_indexes[fold][(challenger_job, domain)])
                target_orbits = _orbits(target_data["scores"], target_data["meta"],
                                         cell_id=cell_id)
                challenger_orbits = _orbits(challenger_data["scores"], challenger_data["meta"],
                                             cell_id=cell_id)
                hashes = target_data["hashes"]
                target = target_orbits["name"]
                incumbent = _align_orbit(incumbent_orbits[BEST_SOURCE[cell_id]],
                                         incumbent_data["hashes"], hashes)
                actual_arm = next(row["arm_id"] for row in candidate["folds"]
                                  if row["partition"] == fold)
                challenger = _align_orbit(challenger_orbits[actual_arm],
                                          challenger_data["hashes"], hashes)
                report, draws = certify_frontier_gain(
                    target, incumbent, challenger, n_boot=n_boot,
                    seed=seed + 101 * cell_index + 17 * candidate_index + fold_index)
                fold_reports.append({"partition": fold, "arm_id": actual_arm, **report})
                fold_draws.append(draws)
            combined = {}
            for key in ("mae_gain", "rho_gain", "flip_gain", "bias_gain"):
                values = np.mean(np.stack([draw[key] for draw in fold_draws]), axis=0)
                points = [row[key]["point"] for row in fold_reports]
                combined[key] = {"point": float(np.mean(points)), "CI": _ci(values)}
            combined["gates"] = {
                "mae_superior": combined["mae_gain"]["CI"][0] > 0,
                "rho_noninferior": combined["rho_gain"]["CI"][0]
                >= -SECONDARY_MARGINS["spearman"],
                "flip_noninferior": combined["flip_gain"]["CI"][0]
                >= -SECONDARY_MARGINS["binary_flip_rate"],
                "bias_noninferior": combined["bias_gain"]["CI"][0]
                >= -SECONDARY_MARGINS["absolute_bias"],
            }
            combined["paired_frontier_improvement"] = all(combined["gates"].values())
            candidate_rows.append({"recipe_id": candidate["recipe_id"],
                                   "folds": fold_reports, "two_fold_mean": combined})
        cells.append({"cell_id": cell_id, "incumbent_arm_id": BEST_SOURCE[cell_id],
                      "candidates": candidate_rows})
    return {
        "schema": "paired_policy_frontier/v1",
        "estimand": "paired direct-policy fidelity gain over the intact-text incumbent",
        "frontier": {"path": frontier_path, "sha256": sha256_file(frontier_path)},
        "shard_roots": {"target": target_shard_root, "incumbent": incumbent_shard_root,
                        "challenger": challenger_shard_root},
        "jobs": {"target": target_job, "incumbent": incumbent_job,
                 "challenger": challenger_job},
        "bootstrap": {"n": n_boot, "seed": seed, "confidence": 0.95,
                      "two_fold_rule": "mean of independently within-fold resampled gains"},
        "secondary_equivalence_margins": SECONDARY_MARGINS,
        "cells": cells,
        "claim_boundary": ("Public-fold paired uncertainty is a promotion/search diagnostic. "
                           "It is not lockbox confirmation."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--target-shard-root", required=True)
    parser.add_argument("--incumbent-shard-root", required=True)
    parser.add_argument("--challenger-shard-root", required=True)
    parser.add_argument("--target-job", default="llama8_big_sparse")
    parser.add_argument("--incumbent-job", default="llama3_small")
    parser.add_argument("--challenger-job", default="llama3_target_policy_rules")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(
        frontier_path=args.frontier, target_shard_root=args.target_shard_root,
        incumbent_shard_root=args.incumbent_shard_root,
        challenger_shard_root=args.challenger_shard_root,
        target_job=args.target_job, incumbent_job=args.incumbent_job,
        challenger_job=args.challenger_job, n_boot=args.n_boot, seed=args.seed)
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), "n_cells": len(result["cells"]),
                      "n_candidates": sum(len(cell["candidates"])
                                          for cell in result["cells"]),
                      "n_combined_pass": sum(
                          row["two_fold_mean"]["paired_frontier_improvement"]
                          for cell in result["cells"] for row in cell["candidates"])}, indent=1))


if __name__ == "__main__":
    main()
