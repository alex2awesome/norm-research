#!/usr/bin/env python
"""Selection-aware readout for the frozen policy-isomorphism lockbox.

The exact certificate remains primary.  This readout keeps confirmatory and exploratory arms
separate, adds paired reconstruction-gain estimates against the small name-only and intact-text
baselines, and reports two complementary gap-closure quantities:

* direct quotient gap closed: movement toward the larger model's mean item policy; and
* adverse-form excess removed: movement toward the larger model's own form-identity band.

Neither partial quantity is allowed to promote an arm into the exact isomorphism fiber.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
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
    articulation_distance,
    pairwise_policy_fidelity,
)


SECONDARY_MARGINS = {
    "spearman": 0.05,
    "binary_flip_rate": 0.02,
    "absolute_bias": 0.02,
}


def _paired_comparison(
        target_orbit: Mapping[str, Sequence[float]],
        baseline_orbit: Mapping[str, Sequence[float]],
        candidate_orbit: Mapping[str, Sequence[float]], *,
        samples: np.ndarray, confidence: float) -> dict:
    """Paired adverse-form fidelity gain; positive values favor the candidate."""
    baseline_point = _orbit_point(target_orbit, baseline_orbit)["candidate_robust"]
    candidate_point = _orbit_point(target_orbit, candidate_orbit)["candidate_robust"]
    baseline = _bootstrap_orbit(target_orbit, baseline_orbit, samples)["candidate"]
    candidate = _bootstrap_orbit(target_orbit, candidate_orbit, samples)["candidate"]
    point = {
        "mae_gain": baseline_point["mae_tvd"] - candidate_point["mae_tvd"],
        "rho_gain": candidate_point["spearman"] - baseline_point["spearman"],
        "flip_gain": (baseline_point["binary_flip_rate"]
                      - candidate_point["binary_flip_rate"]),
        "bias_gain": baseline_point["absolute_bias"] - candidate_point["absolute_bias"],
    }
    draws = {
        "mae_gain": baseline["mae_tvd"] - candidate["mae_tvd"],
        "rho_gain": candidate["spearman"] - baseline["spearman"],
        "flip_gain": baseline["binary_flip_rate"] - candidate["binary_flip_rate"],
        "bias_gain": baseline["absolute_bias"] - candidate["absolute_bias"],
    }
    estimates = {
        key: {"point": float(value), "CI": _ci(draws[key], confidence)}
        for key, value in point.items()
    }
    gates = {
        "mae_superior": estimates["mae_gain"]["CI"][0] > 0.0,
        "rho_noninferior": (estimates["rho_gain"]["CI"][0]
                            >= -SECONDARY_MARGINS["spearman"]),
        "flip_noninferior": (estimates["flip_gain"]["CI"][0]
                             >= -SECONDARY_MARGINS["binary_flip_rate"]),
        "bias_noninferior": (estimates["bias_gain"]["CI"][0]
                             >= -SECONDARY_MARGINS["absolute_bias"]),
    }
    return {
        "baseline": baseline_point,
        "candidate": candidate_point,
        "estimates": estimates,
        "gates": gates,
        "paired_frontier_improvement": bool(all(gates.values())),
        "confidence": confidence,
        "scope": "secondary paired adverse-form comparison",
    }


def _ratio_summary(numerator_point: float, denominator_point: float,
                   numerator_draws: np.ndarray, denominator_draws: np.ndarray, *,
                   confidence: float) -> dict:
    numerator_draws = np.asarray(numerator_draws, float)
    denominator_draws = np.asarray(denominator_draws, float)
    finite = np.isfinite(numerator_draws) & np.isfinite(denominator_draws)
    nonzero = finite & (np.abs(denominator_draws) > np.finfo(float).eps)
    denominator_ci = _ci(denominator_draws[finite], confidence)
    denominator_stable_positive = bool(
        denominator_point > 0.0 and denominator_ci and denominator_ci[0] > 0.0
    )
    # Conditioning the interval on positive denominator draws discards adverse paired resamples
    # and biases closure upward.  Keep every finite nonzero draw, and decline ratio inference when
    # the reference gap itself is not stably positive.
    ratios = np.divide(numerator_draws[nonzero], denominator_draws[nonzero])
    return {
        "point": (None if denominator_point <= 0.0 else
                  float(numerator_point / denominator_point)),
        "CI": (_ci(ratios, confidence) if denominator_stable_positive else None),
        "denominator_point": float(denominator_point),
        "denominator_CI": denominator_ci,
        "denominator_stable_positive": denominator_stable_positive,
        "valid_bootstrap_fraction": float(np.mean(nonzero)),
        "nonpositive_denominator_fraction": float(np.mean(
            finite & (denominator_draws <= 0.0)
        )),
        "inference_status": (
            "ratio_interval_valid"
            if denominator_stable_positive
            else "ratio_interval_undefined_reference_gap_not_stably_positive"
        ),
    }


def _quotient_draws(target_orbit: Mapping[str, Sequence[float]],
                    candidate_orbit: Mapping[str, Sequence[float]],
                    samples: np.ndarray) -> dict[str, np.ndarray]:
    target_values = np.mean(np.stack(list(target_orbit.values())), axis=0)
    candidate_values = np.mean(np.stack(list(candidate_orbit.values())), axis=0)
    q = target_values[samples]
    p = candidate_values[samples]
    q_binary = target_values >= 0.5

    # Match the fixed-full-panel-midrank bootstrap used by the exact certificate.
    from methods.codability.grid_auc_report import _rank

    q_rank = _rank(target_values)[samples]
    p_rank = _rank(candidate_values)[samples]
    q_centered = q_rank - np.mean(q_rank, axis=1, keepdims=True)
    p_centered = p_rank - np.mean(p_rank, axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(q_centered ** 2, axis=1)
                          * np.sum(p_centered ** 2, axis=1))
    rho = np.divide(np.sum(q_centered * p_centered, axis=1), denominator,
                    out=np.full(len(samples), np.nan), where=denominator > 0.0)
    return {
        "mae_tvd": np.mean(np.abs(p - q), axis=1),
        "spearman": rho,
        "binary_flip_rate": np.mean(
            (candidate_values[samples] >= 0.5) != q_binary[samples], axis=1),
        "absolute_bias": np.abs(np.mean(p - q, axis=1)),
    }


def _gap_closure(target_orbit: Mapping[str, Sequence[float]],
                 name_orbit: Mapping[str, Sequence[float]],
                 candidate_orbit: Mapping[str, Sequence[float]], *,
                 samples: np.ndarray, confidence: float) -> dict:
    """Report direct target closure and closure relative to target form variability."""
    name_point = _orbit_point(target_orbit, name_orbit)
    candidate_point = _orbit_point(target_orbit, candidate_orbit)
    target_robust = name_point["target_self_robust"]
    name_robust = name_point["candidate_robust"]
    candidate_robust = candidate_point["candidate_robust"]

    target_draws = _bootstrap_orbit(target_orbit, target_orbit, samples)["target_self"]
    name_draws = _bootstrap_orbit(target_orbit, name_orbit, samples)["candidate"]
    candidate_draws = _bootstrap_orbit(target_orbit, candidate_orbit, samples)["candidate"]

    adverse_mae = _ratio_summary(
        name_robust["mae_tvd"] - candidate_robust["mae_tvd"],
        name_robust["mae_tvd"] - target_robust["mae_tvd"],
        name_draws["mae_tvd"] - candidate_draws["mae_tvd"],
        name_draws["mae_tvd"] - target_draws["mae_tvd"],
        confidence=confidence,
    )
    adverse_rho = _ratio_summary(
        candidate_robust["spearman"] - name_robust["spearman"],
        target_robust["spearman"] - name_robust["spearman"],
        candidate_draws["spearman"] - name_draws["spearman"],
        target_draws["spearman"] - name_draws["spearman"],
        confidence=confidence,
    )
    adverse_flip = _ratio_summary(
        name_robust["binary_flip_rate"] - candidate_robust["binary_flip_rate"],
        name_robust["binary_flip_rate"] - target_robust["binary_flip_rate"],
        name_draws["binary_flip_rate"] - candidate_draws["binary_flip_rate"],
        name_draws["binary_flip_rate"] - target_draws["binary_flip_rate"],
        confidence=confidence,
    )
    adverse_bias = _ratio_summary(
        name_robust["absolute_bias"] - candidate_robust["absolute_bias"],
        name_robust["absolute_bias"] - target_robust["absolute_bias"],
        name_draws["absolute_bias"] - candidate_draws["absolute_bias"],
        name_draws["absolute_bias"] - target_draws["absolute_bias"],
        confidence=confidence,
    )

    name_quotient = name_point["quotient"]
    candidate_quotient = candidate_point["quotient"]
    name_q_draws = _quotient_draws(target_orbit, name_orbit, samples)
    candidate_q_draws = _quotient_draws(target_orbit, candidate_orbit, samples)
    quotient_mae = _ratio_summary(
        name_quotient["mae_tvd"] - candidate_quotient["mae_tvd"],
        name_quotient["mae_tvd"],
        name_q_draws["mae_tvd"] - candidate_q_draws["mae_tvd"],
        name_q_draws["mae_tvd"],
        confidence=confidence,
    )
    quotient_rho = _ratio_summary(
        candidate_quotient["spearman"] - name_quotient["spearman"],
        1.0 - name_quotient["spearman"],
        candidate_q_draws["spearman"] - name_q_draws["spearman"],
        1.0 - name_q_draws["spearman"],
        confidence=confidence,
    )
    quotient_flip = _ratio_summary(
        name_quotient["binary_flip_rate"] - candidate_quotient["binary_flip_rate"],
        name_quotient["binary_flip_rate"],
        name_q_draws["binary_flip_rate"] - candidate_q_draws["binary_flip_rate"],
        name_q_draws["binary_flip_rate"],
        confidence=confidence,
    )
    quotient_bias = _ratio_summary(
        name_quotient["absolute_bias"] - candidate_quotient["absolute_bias"],
        name_quotient["absolute_bias"],
        name_q_draws["absolute_bias"] - candidate_q_draws["absolute_bias"],
        name_q_draws["absolute_bias"],
        confidence=confidence,
    )
    return {
        "direct_target_quotient": {
            "mae_gap_closed": quotient_mae,
            "rho_gap_closed": quotient_rho,
            "flip_gap_closed": quotient_flip,
            "bias_gap_closed": quotient_bias,
            "name": name_quotient,
            "candidate": candidate_quotient,
            "target_reference": {"mae_tvd": 0.0, "spearman": 1.0},
        },
        "adverse_form_identity_band": {
            "mae_excess_removed": adverse_mae,
            "rho_gap_recovered": adverse_rho,
            "flip_excess_removed": adverse_flip,
            "bias_excess_removed": adverse_bias,
            "target_self": target_robust,
            "name": name_robust,
            "candidate": candidate_robust,
        },
        "interpretation_boundary": (
            "A closure fraction is a dimension-specific partial reconstruction statistic. "
            "Values above one mean the candidate crosses past the target self-form radius on that "
            "coordinate; negative values mean articulation moves away. Only the joint exact "
            "certificate can establish policy isomorphism."
        ),
    }


def _failed_gates(certificate: Mapping) -> list[str]:
    exact = {
        "target_identity_valid", "mae_inside_identity_band", "rho_inside_identity_band",
        "flip_inside_identity_band", "bias_inside_identity_band", "positive_polarity",
    }
    return sorted(key for key in exact if not certificate["gates"].get(key, False))


def summarize(*, exact_report_path: str, selection_path: str, arm_bank_path: str,
              shard_root: str, n_boot: int | None = None, seed: int = 20260712) -> dict:
    exact = json.loads(Path(exact_report_path).read_text())
    selection = json.loads(Path(selection_path).read_text())
    bank = json.loads(Path(arm_bank_path).read_text())
    bank_sha = sha256_file(arm_bank_path)
    if bank_sha != selection["arm_bank_sha256"] or bank_sha != exact["arm_bank_sha256"]:
        raise ValueError("arm-bank checksum does not match frozen selection and exact report")
    if exact["partition"] != selection["lockbox_partition"]:
        raise ValueError("exact report did not use the frozen lockbox partition")
    confidence = float(selection["candidatewise_confidence"])
    if not np.isclose(exact["config"]["confidence"], confidence):
        raise ValueError("exact report did not use the frozen multiplicity-adjusted confidence")
    n_boot = int(n_boot or exact["config"]["n_boot"])

    partition = selection["lockbox_partition"]
    index = load_public_index(shard_root, partition)
    target_job = exact["config"]["big_job"]
    executor_job = exact["config"]["small_job"]
    report_cells = {cell["cell_id"]: cell for cell in exact["cells"]}
    bank_cells = {cell["id"]: cell for cell in bank["cells"]}

    confirmatory, exploratory, target_health = [], [], []
    confirm_orbits: dict[str, dict[str, Mapping[str, Sequence[float]]]] = {}
    for cell_index, selected_cell in enumerate(selection["cells"]):
        cell_id = selected_cell["cell_id"]
        cell = bank_cells[cell_id]
        report_cell = report_cells[cell_id]
        allowed = set(selected_cell["allowed_arm_ids"])
        if {arm["id"] for arm in cell["arms"]} != allowed:
            raise ValueError(f"lockbox arm set drifted for {cell_id}")
        if {row["arm_id"] for row in report_cell["rows"]} != allowed - {"name"}:
            raise ValueError(f"exact report arm set drifted for {cell_id}")

        domain = cell["domain"]
        target_data = _average_repetitions(index[(target_job, domain)])
        executor_data = _average_repetitions(index[(executor_job, domain)])
        target_orbits = _orbits(target_data["scores"], target_data["meta"], cell_id=cell_id)
        executor_orbits = _orbits(
            executor_data["scores"], executor_data["meta"], cell_id=cell_id)
        target = target_orbits["name"]
        aligned = {
            arm_id: _align_orbit(orbit, executor_data["hashes"], target_data["hashes"])
            for arm_id, orbit in executor_orbits.items()
        }
        rows = {row["arm_id"]: row for row in report_cell["rows"]}
        first_certificate = next(iter(rows.values()))["certificate"]
        target_health.append({
            "cell_id": cell_id,
            "domain": domain,
            "n_items": report_cell["n_items"],
            "target_identity_valid": first_certificate["gates"]["target_identity_valid"],
            "target_information": first_certificate["point"]["target_information"],
            "target_self_robust": first_certificate["point"]["target_self_robust"],
        })
        incumbent_id = "incumbent_source" if "incumbent_source" in allowed else None
        arm_specs = {arm["id"]: arm for arm in cell["arms"]}
        for family, destination in (
                (selected_cell["confirmatory_candidate_ids"], confirmatory),
                (selected_cell["exploratory_candidate_ids"], exploratory)):
            for candidate_index, arm_id in enumerate(family):
                rng = np.random.default_rng(seed + 1009 * cell_index + 37 * candidate_index)
                samples = rng.integers(
                    0, len(target_data["hashes"]), size=(n_boot, len(target_data["hashes"])))
                certificate = rows[arm_id]["certificate"]
                result = {
                    "cell_id": cell_id,
                    "domain": domain,
                    "arm_id": arm_id,
                    "provenance": arm_specs[arm_id]["provenance"],
                    "channel": arm_specs[arm_id]["channel"],
                    "semantic_content_word_count": arm_specs[arm_id][
                        "semantic_content_word_count"],
                    "policy_isomorphic": certificate["policy_isomorphic"],
                    "articulation_rescue": certificate["articulation_rescue"],
                    "failed_exact_gates": _failed_gates(certificate),
                    "exact_point": certificate["point"],
                    "exact_differences": certificate["differences"],
                    "exact_gates": certificate["gates"],
                    "functional": certificate.get("functional"),
                    "paired_vs_name": _paired_comparison(
                        target, aligned["name"], aligned[arm_id], samples=samples,
                        confidence=confidence),
                    "gap_closure_from_name": _gap_closure(
                        target, aligned["name"], aligned[arm_id], samples=samples,
                        confidence=confidence),
                }
                if incumbent_id:
                    result["paired_vs_intact_incumbent"] = _paired_comparison(
                        target, aligned[incumbent_id], aligned[arm_id], samples=samples,
                        confidence=confidence)
                destination.append(result)
                if destination is confirmatory:
                    confirm_orbits.setdefault(cell_id, {})[arm_id] = aligned[arm_id]

    exact_members = [row for row in confirmatory if row["policy_isomorphic"]]
    fiber_pairs = []
    for left, right in itertools.combinations(exact_members, 2):
        if left["cell_id"] != right["cell_id"]:
            continue
        cell_specs = {arm["id"]: arm for arm in bank_cells[left["cell_id"]]["arms"]}
        distance = articulation_distance(cell_specs[left["arm_id"]],
                                         cell_specs[right["arm_id"]])
        fiber_pairs.append({
            "cell_id": left["cell_id"], "left": left["arm_id"], "right": right["arm_id"],
            "articulation_surface_distance": distance,
            "equal_but_different": distance >= 0.35,
            "behavior": pairwise_policy_fidelity(
                confirm_orbits[left["cell_id"]][left["arm_id"]],
                confirm_orbits[right["cell_id"]][right["arm_id"]]),
        })

    best = min(confirmatory, key=lambda row: (
        row["exact_point"]["candidate_robust"]["mae_tvd"], row["arm_id"]))
    summary = {
        "n_confirmatory_candidates": len(confirmatory),
        "n_exact_policy_isomorphic": len(exact_members),
        "n_articulation_rescues": sum(row["articulation_rescue"] for row in confirmatory),
        "n_familywise_mae_superior_to_name": sum(
            row["paired_vs_name"]["gates"]["mae_superior"] for row in confirmatory),
        "n_familywise_mae_superior_to_intact_incumbent": sum(
            row.get("paired_vs_intact_incumbent", {}).get("gates", {}).get(
                "mae_superior", False) for row in confirmatory),
        "n_observed_functional_ordinal": sum(
            bool(row.get("functional", {}).get(
                "observed_functional_ordinal_isomorphism"))
            for row in confirmatory),
        "n_certified_functional_ordinal": sum(
            bool(row.get("functional", {}).get(
                "certified_functional_ordinal_isomorphism"))
            for row in confirmatory),
        "n_observed_functional_policy_substitutions": sum(
            bool(row.get("functional", {}).get(
                "observed_functional_policy_substitution"))
            for row in confirmatory),
        "n_certified_functional_policy_substitutions": sum(
            bool(row.get("functional", {}).get(
                "certified_functional_policy_substitution"))
            for row in confirmatory),
        "best_adverse_mae_candidate": best["arm_id"],
        "best_adverse_mae_tvd": best["exact_point"]["candidate_robust"]["mae_tvd"],
        "n_exploratory_candidates": len(exploratory),
        "n_valid_target_cells": sum(row["target_identity_valid"] for row in target_health),
    }
    return {
        "schema": "policy_isomorphism_lockbox_summary/v3",
        "estimand": "unsupervised direct reconstruction of the larger name-only item policy",
        "frozen_selection": {"path": selection_path, "sha256": sha256_file(selection_path)},
        "exact_report": {"path": exact_report_path, "sha256": sha256_file(exact_report_path)},
        "arm_bank": {"path": arm_bank_path, "sha256": bank_sha},
        "partition": partition,
        "models": {"target": target_job, "executor": executor_job},
        "bootstrap": {"n": n_boot, "seed": seed, "confidence": confidence,
                      "multiplicity": selection["multiplicity_rule"]},
        "target_health": target_health,
        "confirmatory_family": confirmatory,
        "confirmatory_isomorphism_fiber": {
            "members": [row["arm_id"] for row in exact_members],
            "pairs": fiber_pairs,
            "n_equal_but_different_pairs": sum(
                row["equal_but_different"] for row in fiber_pairs),
        },
        "exploratory_behavioral_only": exploratory,
        "summary": summary,
        "claim_boundary": (
            "No human labels, corpus labels, or third-model judgments enter the target or loss. "
            "Paired gains and dimension-specific closure broaden the partial-reconstruction claim "
            "but cannot substitute for the frozen joint isomorphism certificate. Ratio intervals "
            "are reported only when the paired-bootstrap reference gap is stably positive; no "
            "denominator-sign draws are discarded. Exploratory CW arms remain excluded from the "
            "tacit-content claim after semantic audit. The .70 functional rank tier was added "
            "after this lockbox was opened and is retrospective here; only a future frozen panel "
            "can make it confirmatory."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exact-report", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--arm-bank", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = summarize(
        exact_report_path=args.exact_report, selection_path=args.selection,
        arm_bank_path=args.arm_bank, shard_root=args.shard_root,
        n_boot=args.n_boot, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), **result["summary"]}, indent=1))


if __name__ == "__main__":
    main()
