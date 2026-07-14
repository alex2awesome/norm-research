#!/usr/bin/env python
"""Cross-fit pairwise ordering into direct 3B policy orbits and audit exact 8B fidelity."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import lru_cache
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
from methods.codability.grid_auc_report import _rank, spearman


PARTITIONS = ("residual_prompt_selection", "residual_unit_certification")
FORMS = ("canonical", "question", "boilerplate")
RANK_METHODS = {
    "borda": "borda_scores",
    "bradley_terry": "bradley_terry_scores",
}
AGGREGATIONS = ("matching_form", "mean_rank", "median_rank")
ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return (_rank(values) - 1.0) / max(len(values) - 1.0, 1.0)


def quantile_reorder(base: np.ndarray, ordering_signal: np.ndarray) -> np.ndarray:
    """Change only item order: preserve the direct policy's probability multiset exactly."""
    base = np.asarray(base, float)
    ordering_signal = np.asarray(ordering_signal, float)
    if base.shape != ordering_signal.shape or base.ndim != 1:
        raise ValueError("base and ordering signal must be aligned vectors")
    order = np.argsort(ordering_signal, kind="mergesort")
    result = np.empty_like(base)
    result[order] = np.sort(base, kind="mergesort")
    return result


def transplant_orbit(base_orbit: dict[str, np.ndarray],
                     pair_orbit: dict[str, np.ndarray], *, aggregation: str,
                     alpha: float) -> dict[str, np.ndarray]:
    if set(base_orbit) != set(pair_orbit):
        raise ValueError("direct and comparative form orbits differ")
    pair_ranks = {form: percentile_rank(pair_orbit[form]) for form in base_orbit}
    stack = np.stack([pair_ranks[form] for form in sorted(pair_ranks)])
    if aggregation == "mean_rank":
        shared = np.mean(stack, axis=0)
    elif aggregation == "median_rank":
        shared = np.median(stack, axis=0)
    elif aggregation == "matching_form":
        shared = None
    else:
        raise ValueError(f"unknown aggregation {aggregation}")
    result = {}
    for form, base in base_orbit.items():
        pair = pair_ranks[form] if shared is None else shared
        signal = (1.0 - alpha) * percentile_rank(base) + alpha * pair
        result[form] = quantile_reorder(base, signal)
    return result


def recipe_family(arm_id: str) -> str:
    for partition in PARTITIONS:
        suffix = f"_from_{partition.removeprefix('residual_')}"
        if arm_id.endswith(suffix):
            return arm_id.removesuffix(suffix)
    return arm_id


def _load_pairwise(path: str, target_hashes: list[str]) -> tuple[dict, list[dict]]:
    with np.load(path, allow_pickle=True) as artifact:
        hashes = [str(value) for value in artifact["probe_sha256"]]
        meta = [json.loads(str(value)) for value in artifact["meta"]]
        if len(hashes) != len(set(hashes)):
            raise ValueError("pairwise artifact contains duplicate probe hashes")
        if len(target_hashes) != len(set(target_hashes)):
            raise ValueError("target shard contains duplicate probe hashes")
        location = {value: index for index, value in enumerate(hashes)}
        if not set(target_hashes) <= set(location):
            raise ValueError("pairwise artifact and target shard probes differ")
        order = [location[value] for value in target_hashes]
        result = defaultdict(lambda: defaultdict(dict))
        for method, key in RANK_METHODS.items():
            scores = np.asarray(artifact[key], float)
            for row_index, row in enumerate(meta):
                if row["form"] in result[row["arm_id"]][method]:
                    raise ValueError(
                        f"duplicate pairwise row for {row['arm_id']}/{method}/{row['form']}"
                    )
                result[row["arm_id"]][method][row["form"]] = scores[row_index, order]
    return {arm: {method: dict(orbit) for method, orbit in methods.items()}
            for arm, methods in result.items()}, meta


def _metric_delta(base: dict, candidate: dict) -> dict:
    return {
        "mae_gain": float(base["mae_tvd"] - candidate["mae_tvd"]),
        "rho_gain": float(candidate["spearman"] - base["spearman"]),
        "flip_gain": float(base["binary_flip_rate"] - candidate["binary_flip_rate"]),
        "bias_gain": float(base["absolute_bias"] - candidate["absolute_bias"]),
    }


@lru_cache(maxsize=None)
def _index(root: str, partition: str) -> dict:
    return load_public_index(root, partition)


def _load_base(arm: dict, *, partition: str, target_hashes: list[str]) -> dict:
    source = arm["base_policy_source"]
    data = _average_repetitions(
        _index(source["shard_root"], partition)[(source["model_job"], "humor")])
    orbit = _orbits(data["scores"], data["meta"], cell_id="N_humor_49")[
        source["arm_id"]]
    return _align_orbit(orbit, data["hashes"], target_hashes)


def _raw_pairwise_diagnostic(target: dict, pairwise: dict) -> list[dict]:
    target_mean = np.mean(np.stack(list(target.values())), axis=0)
    rows = []
    for arm_id, methods in sorted(pairwise.items()):
        for method, orbit in sorted(methods.items()):
            form_rho = {form: spearman(target_mean, values)
                        for form, values in sorted(orbit.items())}
            consensus = np.mean(np.stack(
                [percentile_rank(values) for values in orbit.values()]), axis=0)
            rows.append({
                "arm_id": arm_id,
                "method": method,
                "form_spearman": form_rho,
                "robust_spearman": float(min(form_rho.values())),
                "consensus_spearman": float(spearman(target_mean, consensus)),
            })
    return rows


def run(*, bank_path: str, pairwise_score_dir: str, target_shard_root: str,
        n_boot: int = 1000, seed: int = 20260724, top_certify: int = 12) -> dict:
    bank = json.loads(Path(bank_path).read_text())
    arm_specs = {arm["id"]: arm for arm in bank["cell"]["arms"]}
    fold_rows = {}
    candidate_orbits = {}
    fold_context = {}
    raw_diagnostics = {}
    score_artifacts = []
    for fold_index, partition in enumerate(PARTITIONS):
        target_data = _average_repetitions(
            _index(target_shard_root, partition)[("llama8_big_sparse", "humor")])
        target = _orbits(target_data["scores"], target_data["meta"],
                         cell_id="N_humor_49")["name"]
        target_hashes = target_data["hashes"]
        sparse_data = _average_repetitions(
            _index(target_shard_root, partition)[("llama3_small", "humor")])
        sparse = _align_orbit(
            _orbits(sparse_data["scores"], sparse_data["meta"],
                    cell_id="N_humor_49")["name"],
            sparse_data["hashes"], target_hashes)
        pair_path = str(Path(pairwise_score_dir) / f"pairwise_{partition}.npz")
        pairwise, _meta = _load_pairwise(pair_path, target_hashes)
        score_artifacts.append({"partition": partition, "path": pair_path,
                                "sha256": sha256_file(pair_path)})
        raw_diagnostics[partition] = _raw_pairwise_diagnostic(target, pairwise)
        rows = []
        for arm_id, arm in arm_specs.items():
            if arm.get("source_partition") == partition:
                continue
            base = _load_base(arm, partition=partition, target_hashes=target_hashes)
            base_point = _orbit_point(target, base)
            base_robust = base_point["candidate_robust"]
            family = recipe_family(arm_id)
            for method in RANK_METHODS:
                pair_orbit = pairwise[arm_id][method]
                for aggregation in AGGREGATIONS:
                    for alpha in ALPHAS:
                        candidate = transplant_orbit(
                            base, pair_orbit, aggregation=aggregation, alpha=alpha)
                        point = _orbit_point(target, candidate)
                        robust = point["candidate_robust"]
                        recipe_id = (f"{family}|{method}|{aggregation}|"
                                     f"alpha={alpha:.2f}")
                        row = {
                            "recipe_id": recipe_id,
                            "family": family,
                            "actual_arm_id": arm_id,
                            "source_partition": arm.get("source_partition"),
                            "claim_channel": ("elicitation_control" if family == "name"
                                              else "explicit_articulation"),
                            "rank_method": method,
                            "aggregation": aggregation,
                            "alpha": alpha,
                            "base_robust": base_robust,
                            "candidate_robust": robust,
                            "candidate_quotient": point["quotient"],
                            "identity_loss": identity_loss(point),
                            "delta_from_same_direct_policy": _metric_delta(base_robust, robust),
                            "point_inside_all_identity_margins": bool(
                                robust["mae_tvd"] - point["target_self_robust"]["mae_tvd"] <= 0.02
                                and robust["spearman"] - point["target_self_robust"]["spearman"] >= -0.05
                                and robust["binary_flip_rate"]
                                - point["target_self_robust"]["binary_flip_rate"] <= 0.02
                                and robust["absolute_bias"]
                                - point["target_self_robust"]["absolute_bias"] <= 0.02
                                and robust["all_positive_polarity"]),
                        }
                        rows.append(row)
                        candidate_orbits[(partition, recipe_id)] = candidate
        fold_rows[partition] = rows
        fold_context[partition] = {"target": target, "sparse": sparse,
                                   "n_items": len(target_hashes),
                                   "target_shards": target_data["shard_sha256"]}

    grouped = defaultdict(list)
    for partition, rows in fold_rows.items():
        for row in rows:
            grouped[row["recipe_id"]].append({"partition": partition, **row})
    stable = []
    for recipe_id, rows in grouped.items():
        if len(rows) != len(PARTITIONS):
            continue
        deltas = [row["delta_from_same_direct_policy"] for row in rows]
        stable.append({
            "recipe_id": recipe_id,
            "claim_channel": rows[0]["claim_channel"],
            "folds": rows,
            "max_identity_loss": float(max(row["identity_loss"] for row in rows)),
            "mean_identity_loss": float(np.mean([row["identity_loss"] for row in rows])),
            "improves_mae_and_rho_both_folds": bool(
                all(delta["mae_gain"] > 0 and delta["rho_gain"] > 0 for delta in deltas)),
            "point_pareto_improves_all_four_both_folds": bool(all(
                all(delta[key] >= 0 for key in (
                    "mae_gain", "rho_gain", "flip_gain", "bias_gain"))
                and (delta["mae_gain"] > 0 or delta["rho_gain"] > 0)
                for delta in deltas)),
            "point_inside_all_identity_margins_both_folds": bool(
                all(row["point_inside_all_identity_margins"] for row in rows)),
        })
    stable.sort(key=lambda row: (row["max_identity_loss"], row["mean_identity_loss"],
                                 row["recipe_id"]))

    certifications = []
    for recipe_index, recipe in enumerate(stable[:top_certify]):
        folds = []
        for fold_index, partition in enumerate(PARTITIONS):
            context = fold_context[partition]
            certificate = certify_policy_isomorphism(
                context["target"], candidate_orbits[(partition, recipe["recipe_id"])],
                sparse_orbit=context["sparse"], n_boot=n_boot,
                seed=seed + 1009 * recipe_index + fold_index)
            folds.append({"partition": partition, "certificate": certificate})
        certifications.append({
            "recipe_id": recipe["recipe_id"],
            "folds": folds,
            "policy_isomorphic_both_folds": bool(
                all(row["certificate"]["policy_isomorphic"] for row in folds)),
        })

    return {
        "schema": "pairwise_policy_isomorphism_atlas/v1",
        "status": "public-development-crossfit; not confirmation",
        "estimand": ("whether content-conditioned 3B comparative ordering can improve or exactly "
                      "reconstruct the fixed 8B name-only policy while preserving each direct "
                      "3B articulation's probability marginal"),
        "model_family": bank["model_family"],
        "anchor_policy": bank["anchor_policy"],
        "bank": {"path": bank_path, "sha256": sha256_file(bank_path)},
        "pairwise_scores": score_artifacts,
        "target_shard_root": target_shard_root,
        "design": {
            "partitions": list(PARTITIONS),
            "rank_methods": list(RANK_METHODS),
            "aggregations": list(AGGREGATIONS),
            "alphas": list(ALPHAS),
            "crossfit_rule": ("exclude an articulation from the same residual fold that generated "
                              "its content; compare normalized recipe families across both folds"),
            "calibration_rule": ("rank transplant preserves the exact multiset of direct 3B "
                                 "probabilities separately in every form"),
            "n_boot": n_boot,
            "seed": seed,
            "top_certify": top_certify,
        },
        "raw_pairwise_rank_diagnostic": raw_diagnostics,
        "fold_atlas": fold_rows,
        "stable_recipes": stable,
        "certifications": certifications,
        "summary": {
            "n_fold_candidates": sum(len(rows) for rows in fold_rows.values()),
            "n_stable_recipes": len(stable),
            "n_improve_mae_and_rho_both": sum(
                row["improves_mae_and_rho_both_folds"] for row in stable),
            "n_point_pareto_all_four_both": sum(
                row["point_pareto_improves_all_four_both_folds"] for row in stable),
            "n_point_inside_identity_both": sum(
                row["point_inside_all_identity_margins_both_folds"] for row in stable),
            "n_certified_isomorphic_both": sum(
                row["policy_isomorphic_both_folds"] for row in certifications),
        },
        "claim_boundary": ("This atlas uses only public residual folds and performs no external-"
                           "label fitting. Recipe ranking is exploratory; any promotion requires "
                           "freezing a small family before a new untouched panel."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", required=True)
    parser.add_argument("--pairwise-score-dir", required=True)
    parser.add_argument("--target-shard-root", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--top-certify", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(bank_path=args.bank, pairwise_score_dir=args.pairwise_score_dir,
                 target_shard_root=args.target_shard_root, n_boot=args.n_boot,
                 seed=args.seed, top_certify=args.top_certify)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), **result["summary"]}, indent=1))


if __name__ == "__main__":
    main()
