#!/usr/bin/env python
"""Cross-fit convex portfolios of explicit 3B prompt policies against the fixed 8B target."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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
from methods.codability.experiments.paired_policy_frontier import certify_frontier_gain
from methods.codability.experiments.synthesize_residual_policy_revisions import identity_loss


PARTITIONS = ("residual_prompt_selection", "residual_unit_certification")
FORMS = ("canonical", "question", "boilerplate")
SOURCE_SPECS = (
    ("source", "fresh_name_arm_score_shards_v1", "llama3_small"),
    ("ostensive", "residual_policy_isomorphism_score_shards_v1",
     "llama3_residual_isomorphism"),
    ("rules", "target_policy_rule_score_shards_v1", "llama3_target_policy_rules"),
    ("revisions", "residual_policy_revision_score_shards_v1", "llama3_policy_revision"),
    ("textbooks", "hierarchical_policy_textbook_score_shards_v1",
     "llama3_policy_textbook"),
    ("rank", "rank_contrast_score_shards_v1", "llama3_rank_contrasts"),
)
ALGORITHMS = (
    ("simplex_mse", "mse", None),
    ("simplex_huber", "huber", None),
    ("simplex_rank", "rank", None),
    ("simplex_rank_mse", "rank_mse", None),
    ("simplex_mse_top3", "mse", 3),
    ("simplex_mse_top5", "mse", 5),
    ("simplex_mse_top10", "mse", 10),
    ("simplex_huber_top5", "huber", 5),
)
ARTICULATION_PROVENANCE = frozenset({
    "source_telling",
    "ostensive_teaching",
    "target_self_articulation",
    "target_behavior_articulation",
    "target_residual_revision",
    "target_hierarchical_articulation",
    "target_rank_articulation",
})
FEATURE_SCOPES = {
    # These texts do not depend on either public item fold. Their portfolio weights can therefore
    # be learned on one fold and tested on the other without reciprocal-content leakage.
    "fold_independent": frozenset({"source_telling", "target_self_articulation"}),
    # Exploratory recipe transfer: fold-derived texts are evaluated only on the opposite fold, but
    # the normalized counterpart recipe exists in both directions. This is not a strict holdout.
    "reciprocal_all": ARTICULATION_PROVENANCE,
}


def arm_origin(arm_id: str) -> str | None:
    for partition in PARTITIONS:
        suffix = f"_from_{partition.removeprefix('residual_')}"
        if arm_id.endswith(suffix):
            return partition
    return None


def normalize_arm_id(arm_id: str) -> str:
    origin = arm_origin(arm_id)
    if origin is None:
        return arm_id
    return arm_id.removesuffix(f"_from_{origin.removeprefix('residual_')}")


def portfolio_loss(prediction: np.ndarray, target: np.ndarray, *, loss: str) -> float:
    prediction = np.asarray(prediction, float)
    target = np.asarray(target, float)
    residual = prediction - target
    if loss == "mse":
        return float(np.mean(residual ** 2))
    if loss == "huber":
        delta = 0.05
        absolute = np.abs(residual)
        values = np.where(
            absolute <= delta,
            0.5 * residual ** 2,
            delta * (absolute - 0.5 * delta),
        )
        return float(np.mean(values))
    if loss in {"rank", "rank_mse"}:
        # Pearson correlation against fixed target midranks is a smooth listwise proxy for the
        # held-out Spearman gate. A small MSE term in rank_mse discourages rank-by-overshoot.
        from methods.codability.grid_auc_report import _rank

        target_rank = _rank(target)
        centered_target = target_rank - np.mean(target_rank)
        centered_prediction = prediction - np.mean(prediction)
        denominator = np.sqrt(
            np.sum(centered_target ** 2) * np.sum(centered_prediction ** 2)
        )
        if denominator <= np.finfo(float).eps:
            correlation_loss = 2.0
        else:
            correlation = np.sum(centered_target * centered_prediction) / denominator
            correlation_loss = 1.0 - float(correlation)
        if loss == "rank_mse":
            correlation_loss += 2.0 * float(np.mean(residual ** 2))
        return correlation_loss
    raise ValueError(f"unknown portfolio loss {loss}")


def fit_simplex(features: np.ndarray, target: np.ndarray, *, loss: str,
                top_k: int | None = None) -> tuple[np.ndarray, dict]:
    """Fit a shared nonnegative, sum-one prompt mixture; optional sparse refit."""
    features = np.asarray(features, float)
    target = np.asarray(target, float)
    if features.ndim != 2 or target.shape != (len(features),):
        raise ValueError("portfolio features and target are not aligned")
    n_features = features.shape[1]

    def objective(weights: np.ndarray) -> float:
        return portfolio_loss(features @ weights, target, loss=loss)

    def solve(active: np.ndarray, initial: np.ndarray | None = None):
        active_features = features[:, active]

        def active_objective(weights):
            return portfolio_loss(active_features @ weights, target, loss=loss)

        x0 = (np.full(len(active), 1.0 / len(active)) if initial is None else initial)
        result = minimize(
            active_objective, x0, method="SLSQP", bounds=[(0.0, 1.0)] * len(active),
            constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
            options={"maxiter": 1000, "ftol": 1e-12})
        if not result.success:
            raise RuntimeError(f"simplex optimization failed: {result.message}")
        full = np.zeros(n_features, float)
        full[active] = result.x
        return full, result

    active = np.arange(n_features)
    weights, result = solve(active)
    if top_k is not None and top_k < n_features:
        active = np.sort(np.argsort(weights)[-top_k:])
        initial = weights[active]
        initial = initial / np.sum(initial)
        weights, result = solve(active, initial)
    return weights, {
        "success": bool(result.success),
        "objective": objective(weights),
        "n_nonzero": int(np.sum(weights > 1e-8)),
        "weight_sum": float(np.sum(weights)),
    }


def _feature_bundle(*, data_root: str, partition: str, target_hashes: list[str],
                    allowed_provenance: frozenset[str]) -> dict:
    features = {}
    feature_meta = {}
    artifacts = []
    for source_id, relative_root, job in SOURCE_SPECS:
        root = str(Path(data_root) / relative_root)
        data = _average_repetitions(load_public_index(root, partition)[(job, "humor")])
        orbits = _orbits(data["scores"], data["meta"], cell_id="N_humor_49")
        arm_meta = {}
        for row in data["meta"]:
            if row["cell_id"] != "N_humor_49":
                continue
            arm_id = row.get("arm_id", "target")
            summary = {key: row.get(key) for key in (
                "provenance", "channel", "semantic_content_word_count")}
            if arm_id in arm_meta and arm_meta[arm_id] != summary:
                raise ValueError(f"inconsistent portfolio metadata for {source_id}:{arm_id}")
            arm_meta[arm_id] = summary
        for arm_id, orbit in orbits.items():
            if arm_origin(arm_id) == partition:
                continue
            if set(orbit) != set(FORMS):
                continue
            provenance = arm_meta[arm_id]["provenance"]
            # Retain one direct-name baseline, never six duplicate copies. Controls are useful
            # diagnostics but cannot support an articulation-specific portfolio claim.
            if arm_id == "name":
                if source_id != "source":
                    continue
            elif provenance not in allowed_provenance:
                continue
            key = f"{source_id}:{normalize_arm_id(arm_id)}"
            if key in features:
                raise ValueError(f"duplicate normalized portfolio feature {key}")
            features[key] = _align_orbit(orbit, data["hashes"], target_hashes)
            feature_meta[key] = arm_meta[arm_id]
        artifacts.append({"source_id": source_id, "root": root, "job": job,
                          "shards": data["shard_sha256"]})
    return {"features": features, "feature_meta": feature_meta, "artifacts": artifacts}


def _matrix(features: dict[str, dict], keys: list[str], form: str) -> np.ndarray:
    return np.column_stack([features[key][form] for key in keys])


def run(*, data_root: str, n_boot: int = 2000, seed: int = 20260727,
        feature_scope: str = "fold_independent",
        functional_rho_floor: float = 0.70) -> dict:
    if feature_scope not in FEATURE_SCOPES:
        raise ValueError(f"unknown portfolio feature scope {feature_scope!r}")
    allowed_provenance = FEATURE_SCOPES[feature_scope]
    target_root = str(Path(data_root) / "fresh_name_arm_score_shards_v1")
    contexts = {}
    all_artifacts = {}
    for partition in PARTITIONS:
        index = load_public_index(target_root, partition)
        target_data = _average_repetitions(index[("llama8_big_sparse", "humor")])
        target = _orbits(target_data["scores"], target_data["meta"],
                         cell_id="N_humor_49")["name"]
        target_hashes = target_data["hashes"]
        sparse_data = _average_repetitions(index[("llama3_small", "humor")])
        sparse = _align_orbit(
            _orbits(sparse_data["scores"], sparse_data["meta"],
                    cell_id="N_humor_49")["name"],
            sparse_data["hashes"], target_hashes)
        bundle = _feature_bundle(
            data_root=data_root, partition=partition, target_hashes=target_hashes,
            allowed_provenance=allowed_provenance)
        contexts[partition] = {"target": target, "sparse": sparse,
                               "features": bundle["features"],
                               "feature_meta": bundle["feature_meta"],
                               "n_items": len(target_hashes)}
        all_artifacts[partition] = {
            "target_shards": target_data["shard_sha256"],
            "feature_sources": bundle["artifacts"],
        }
    common_keys = sorted(set.intersection(
        *(set(contexts[partition]["features"]) for partition in PARTITIONS)))
    if not common_keys:
        raise ValueError("no common cross-fitted portfolio features")
    provenance_counts = Counter(
        contexts[PARTITIONS[0]]["feature_meta"][key]["provenance"]
        for key in common_keys
    )

    algorithm_rows = []
    for algorithm_index, (algorithm_id, loss, top_k) in enumerate(ALGORITHMS):
        directions = []
        for train_index, train_partition in enumerate(PARTITIONS):
            test_partition = next(p for p in PARTITIONS if p != train_partition)
            train = contexts[train_partition]
            test = contexts[test_partition]
            train_matrix = np.vstack([
                _matrix(train["features"], common_keys, form) for form in FORMS])
            train_target_mean = np.mean(np.stack(list(train["target"].values())), axis=0)
            train_target = np.tile(train_target_mean, len(FORMS))
            weights, optimization = fit_simplex(
                train_matrix, train_target, loss=loss, top_k=top_k)
            single_losses = []
            for feature_index in range(len(common_keys)):
                single_losses.append(portfolio_loss(
                    train_matrix[:, feature_index], train_target, loss=loss
                ))
            single_index = min(
                range(len(common_keys)),
                key=lambda index: (single_losses[index], common_keys[index]),
            )
            single_key = common_keys[single_index]
            train_orbit = {form: _matrix(
                train["features"], common_keys, form) @ weights for form in FORMS}
            test_orbit = {form: _matrix(
                test["features"], common_keys, form) @ weights for form in FORMS}
            test_single_orbit = {
                form: test["features"][single_key][form] for form in FORMS
            }
            train_point = _orbit_point(train["target"], train_orbit)
            certificate = certify_policy_isomorphism(
                test["target"], test_orbit, sparse_orbit=test["sparse"],
                functional_rho_floor=functional_rho_floor,
                n_boot=n_boot, seed=seed + 1009 * algorithm_index + train_index)
            portfolio_vs_single, _draws = certify_frontier_gain(
                test["target"], test_single_orbit, test_orbit,
                n_boot=n_boot,
                seed=seed + 100_003 + 1009 * algorithm_index + train_index,
            )
            nonzero = [(common_keys[index], float(weight))
                       for index, weight in enumerate(weights) if weight > 1e-8]
            nonzero.sort(key=lambda row: (-row[1], row[0]))
            directions.append({
                "train_partition": train_partition,
                "test_partition": test_partition,
                "optimization": optimization,
                "weights": [{"feature": key, "weight": weight} for key, weight in nonzero],
                "training_point": train_point,
                "training_identity_loss": identity_loss(train_point),
                "selected_single_feature": single_key,
                "selected_single_training_objective": single_losses[single_index],
                "selected_single_test_point": _orbit_point(
                    test["target"], test_single_orbit
                ),
                "portfolio_vs_selected_single": portfolio_vs_single,
                "test_certificate": certificate,
            })
        algorithm_rows.append({
            "algorithm_id": algorithm_id,
            "loss": loss,
            "top_k": top_k,
            "directions": directions,
            "policy_isomorphic_both_directions": bool(all(
                row["test_certificate"]["policy_isomorphic"] for row in directions)),
            "improves_mae_over_sparse_both_directions": bool(all(
                row["test_certificate"]["gates"]["mae_improves_over_small_sparse"]
                for row in directions)),
            "paired_frontier_over_single_both_directions": bool(all(
                row["portfolio_vs_selected_single"]["paired_frontier_improvement"]
                for row in directions
            )),
            "observed_functional_substitution_both_directions": bool(all(
                row["test_certificate"]["functional"][
                    "observed_functional_policy_substitution"]
                for row in directions
            )),
            "certified_functional_substitution_both_directions": bool(all(
                row["test_certificate"]["functional"][
                    "certified_functional_policy_substitution"]
                for row in directions
            )),
            "max_test_identity_loss": float(max(
                identity_loss(row["test_certificate"]["point"]) for row in directions)),
        })
    algorithm_rows.sort(key=lambda row: (row["max_test_identity_loss"], row["algorithm_id"]))
    return {
        "schema": "prompt_portfolio_crossfit/v5",
        "status": ("strict-public-fold-transfer; not confirmation"
                   if feature_scope == "fold_independent"
                   else "reciprocal-public-development; not strict holdout or confirmation"),
        "estimand": ("system-level direct policy reconstruction by a convex portfolio of explicit "
                      "3B prompt policies, with one shared weight vector across prompt forms"),
        "model_family": "Llama: 3B executor, fixed 8B name-only target",
        "external_target": "none; weights use only the opposite public fold of the fixed target",
        "compiler_boundary": "no generated code or metric_seam artifact enters this experiment",
        "data_root": data_root,
        "feature_scope": feature_scope,
        "source_specs": [{"id": source_id, "relative_root": root, "job": job}
                         for source_id, root, job in SOURCE_SPECS],
        "artifacts": all_artifacts,
        "feature_policy": {
            "n_common_features": len(common_keys),
            "common_features": common_keys,
            "provenance_counts": dict(sorted(provenance_counts.items())),
            "allowed_articulation_provenance": sorted(allowed_provenance),
            "controls_excluded": True,
            "one_name_baseline_only": True,
            "same_form_only": True,
            "shared_weights_across_forms": True,
            "simplex": True,
            "crossfit": (
                "fold-independent texts and one name baseline are weight-fit on one public fold "
                "and transferred unchanged to the other"
                if feature_scope == "fold_independent"
                else "fold-derived texts are excluded on their source fold and normalized "
                     "counterpart recipes transfer reciprocally; exploratory, not strict holdout"
            ),
        },
        "bootstrap": {"n": n_boot, "seed": seed, "confidence": 0.95,
                      "functional_rho_floor": functional_rho_floor},
        "algorithms": algorithm_rows,
        "summary": {
            "n_algorithms": len(algorithm_rows),
            "n_isomorphic_both": sum(
                row["policy_isomorphic_both_directions"] for row in algorithm_rows),
            "n_improve_mae_both": sum(
                row["improves_mae_over_sparse_both_directions"] for row in algorithm_rows),
            "n_paired_frontier_over_single_both": sum(
                row["paired_frontier_over_single_both_directions"]
                for row in algorithm_rows
            ),
            "n_observed_functional_substitution_both": sum(
                row["observed_functional_substitution_both_directions"]
                for row in algorithm_rows
            ),
            "n_certified_functional_substitution_both": sum(
                row["certified_functional_substitution_both_directions"]
                for row in algorithm_rows
            ),
        },
        "claim_boundary": ("This is a prompt-system result, not evidence that any single text is "
                           "isomorphic. It averages multiple small-model prompt executions and is "
                           "not interchangeable with one larger prompt. Controls are excluded, but "
                           "algorithm selection is public exploration; an untouched claim requires "
                           "frozen sources, weights, and panel."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--feature-scope", choices=sorted(FEATURE_SCOPES),
                        default="fold_independent")
    parser.add_argument("--functional-rho-floor", type=float, default=0.70)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(data_root=args.data_root, n_boot=args.n_boot, seed=args.seed,
                 feature_scope=args.feature_scope,
                 functional_rho_floor=args.functional_rho_floor)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), **result["summary"],
                      "n_features": result["feature_policy"]["n_common_features"]}, indent=1))


if __name__ == "__main__":
    main()
