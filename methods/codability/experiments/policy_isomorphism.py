#!/usr/bin/env python
"""Direct policy-isomorphism certificates and equal-but-different articulation fibers.

The larger reader's sparse soft verdict orbit is the target.  A smaller-reader articulation is
isomorphic when its adverse-form item-level policy lies inside the target's own form-identity band,
up to frozen smallest-effect margins.  This is deliberately stricter than equal aggregate recovery
against a third target.

The primary distance is mean Bernoulli TVD, equal here to mean absolute difference between soft
verdict probabilities.  Rank fidelity, threshold flips, calibration bias, and polarity are retained
as joint gates.  Prompt cost and unit count never enter the isomorphism objective.
"""
from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from methods.codability.grid_auc_report import spearman
from methods.codability.experiments.target_articulation_frontier import validate_orbit


SCHEMA = "policy_isomorphism/v4"
ORACLE_MEAN_SHIFT_SALT = "policy-isomorphism-oracle-mean-shift-v1"


def _ci(values: np.ndarray, confidence: float = 0.95) -> list[float] | None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))]


def _validate_bootstrap_args(n_boot: int, confidence: float) -> None:
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")


def _rowwise_midrank(values: np.ndarray) -> np.ndarray:
    """Return midranks computed independently inside every bootstrap resample."""
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("values must have shape (n_draws, positive sample size)")
    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    positions = np.arange(1, values.shape[1] + 1, dtype=float)[None, :]

    group_start = np.ones(values.shape, dtype=bool)
    group_start[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    starts = np.maximum.accumulate(np.where(group_start, positions, 0.0), axis=1)

    group_end = np.ones(values.shape, dtype=bool)
    group_end[:, :-1] = sorted_values[:, :-1] != sorted_values[:, 1:]
    ends = np.minimum.accumulate(
        np.where(group_end, positions, values.shape[1] + 1.0)[:, ::-1], axis=1
    )[:, ::-1]

    sorted_ranks = (starts + ends) / 2.0
    ranks = np.empty(values.shape, dtype=float)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    return ranks


def _n_jointly_finite(*draws: np.ndarray) -> int:
    if not draws:
        return 0
    arrays = [np.asarray(draw, float) for draw in draws]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("bootstrap draw arrays must have identical shapes")
    return int(np.sum(np.logical_and.reduce([np.isfinite(array) for array in arrays])))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def lexical_jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def articulation_distance(left: Mapping, right: Mapping) -> float:
    """Declared surface/provenance diversity; not a claim of semantic distance."""
    left_text = next(form["prompt"] for form in left["forms"] if form["id"] == "canonical")
    right_text = next(form["prompt"] for form in right["forms"] if form["id"] == "canonical")
    lexical = 1.0 - lexical_jaccard(left_text, right_text)
    channel = float(left.get("channel") != right.get("channel"))
    provenance = float(left.get("provenance") != right.get("provenance"))
    return float(0.6 * lexical + 0.3 * channel + 0.1 * provenance)


def _orbit_point(target_orbit: Mapping[str, Sequence[float]],
                 candidate_orbit: Mapping[str, Sequence[float]]) -> dict:
    target = validate_orbit(target_orbit, name="target_orbit")
    n_items = len(next(iter(target.values())))
    candidate = validate_orbit(candidate_orbit, n_items=n_items, name="candidate_orbit")
    q = np.mean(np.stack(list(target.values())), axis=0)
    p = np.mean(np.stack(list(candidate.values())), axis=0)

    def per_form(orbit: dict[str, np.ndarray]) -> dict[str, dict]:
        rows = {}
        for form, values in sorted(orbit.items()):
            rho = spearman(q, values)
            rows[form] = {
                "mae_tvd": float(np.mean(np.abs(values - q))),
                "spearman": None if rho is None or not np.isfinite(rho) else float(rho),
                "binary_flip_rate": float(np.mean((values >= 0.5) != (q >= 0.5))),
                "absolute_bias": float(abs(np.mean(values - q))),
                "covariance": float(np.mean(q * values) - np.mean(q) * np.mean(values)),
            }
        return rows

    target_forms, candidate_forms = per_form(target), per_form(candidate)

    def robust(rows: dict[str, dict]) -> dict:
        rhos = [row["spearman"] for row in rows.values()]
        return {
            "mae_tvd": float(max(row["mae_tvd"] for row in rows.values())),
            "spearman": (float(min(rhos)) if all(value is not None for value in rhos) else None),
            "binary_flip_rate": float(max(row["binary_flip_rate"] for row in rows.values())),
            "absolute_bias": float(max(row["absolute_bias"] for row in rows.values())),
            "all_positive_polarity": bool(all(row["covariance"] > 0 for row in rows.values())),
            "n_forms": len(rows),
        }

    quotient_rho = spearman(q, p)
    return {
        "n_items": n_items,
        "target_information": float(np.mean(np.abs(q - np.mean(q)))),
        "target_positive_rate": float(np.mean(q >= 0.5)),
        "target_self_forms": target_forms,
        "target_self_robust": robust(target_forms),
        "candidate_forms": candidate_forms,
        "candidate_robust": robust(candidate_forms),
        "quotient": {
            "mae_tvd": float(np.mean(np.abs(p - q))),
            "spearman": (None if quotient_rho is None or not np.isfinite(quotient_rho)
                         else float(quotient_rho)),
            "binary_flip_rate": float(np.mean((p >= 0.5) != (q >= 0.5))),
            "absolute_bias": float(abs(np.mean(p - q))),
        },
    }


def _stable_hash_crossfit_split(
        item_hashes: Sequence[str], *, salt: str = ORACLE_MEAN_SHIFT_SALT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an append-stable, outcome-blind 50/50 calibration/evaluation split."""
    hashes = [str(value) for value in item_hashes]
    if len(hashes) < 8 or len(hashes) != len(set(hashes)):
        raise ValueError("item_hashes must contain at least eight unique values")
    threshold = 1 << 255
    calibration = np.asarray([
        int(hashlib.sha256(f"{salt}\0{value}".encode()).hexdigest(), 16) < threshold
        for value in hashes
    ], dtype=bool)
    evaluation = ~calibration
    if int(np.sum(calibration)) < 4 or int(np.sum(evaluation)) < 4:
        raise ValueError("stable hash split produced fewer than four items in one fold")
    return np.flatnonzero(calibration), np.flatnonzero(evaluation)


def _fit_bounded_mean_shift(
        target_mean: np.ndarray, candidate_orbit: Mapping[str, Sequence[float]],
        calibration_indexes: np.ndarray,
) -> float:
    """Fit one clipped additive intercept, using target scores only on calibration items."""
    candidate = validate_orbit(candidate_orbit, n_items=len(target_mean),
                               name="candidate_orbit")
    target_level = float(np.mean(target_mean[calibration_indexes]))
    values = np.stack(list(candidate.values()))[:, calibration_indexes]
    low, high = -1.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        observed = float(np.mean(np.clip(values + middle, 0.0, 1.0)))
        if observed < target_level:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def oracle_mean_shift_diagnostic(
        target_orbit: Mapping[str, Sequence[float]],
        candidate_orbits: Mapping[str, Mapping[str, Sequence[float]]], *,
        item_hashes: Sequence[str], name_arm_id: str = "name",
        salt: str = ORACLE_MEAN_SHIFT_SALT, n_boot: int = 5000,
        seed: int = 1207, confidence: float = 0.98,
) -> dict:
    """Price articulation against a cross-fitted one-scalar target-score oracle.

    This is intentionally a diagnostic reference, not an unsupervised reconstruction arm: the
    scalar intercept is fitted from the larger model's calibration-fold scores.  It transmits no
    item-specific information and is evaluated only on the disjoint stable-hash fold.
    """
    _validate_bootstrap_args(n_boot, confidence)
    target = validate_orbit(target_orbit, name="target_orbit")
    n_items = len(next(iter(target.values())))
    if len(item_hashes) != n_items:
        raise ValueError("item_hashes and policy orbits are unaligned")
    if name_arm_id not in candidate_orbits:
        raise ValueError(f"candidate_orbits omit name arm {name_arm_id!r}")
    candidates = {
        str(arm_id): validate_orbit(
            orbit, n_items=n_items, name=f"candidate_orbits[{arm_id!r}]"
        )
        for arm_id, orbit in candidate_orbits.items()
    }
    calibration_indexes, evaluation_indexes = _stable_hash_crossfit_split(
        item_hashes, salt=salt)
    q = np.mean(np.stack(list(target.values())), axis=0)
    shift = _fit_bounded_mean_shift(
        q, candidates[name_arm_id], calibration_indexes)

    def sliced(orbit: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {form: np.asarray(values, float)[evaluation_indexes]
                for form, values in orbit.items()}

    target_evaluation = sliced(target)
    raw_name_evaluation = sliced(candidates[name_arm_id])
    shifted_full = {
        form: np.clip(np.asarray(values, float) + shift, 0.0, 1.0)
        for form, values in candidates[name_arm_id].items()
    }
    shifted_evaluation = sliced(shifted_full)
    evaluation_orbits = {
        arm_id: sliced(orbit) for arm_id, orbit in candidates.items()
    }
    reference_id = "name_plus_crossfit_mean_shift"
    evaluation_orbits[reference_id] = shifted_evaluation

    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0, len(evaluation_indexes), size=(n_boot, len(evaluation_indexes)))
    points = {
        arm_id: _orbit_point(target_evaluation, orbit)
        for arm_id, orbit in evaluation_orbits.items()
    }
    draws = {
        arm_id: _bootstrap_orbit(target_evaluation, orbit, samples)["candidate"]
        for arm_id, orbit in evaluation_orbits.items()
    }

    def improvement(candidate_id: str, comparator_id: str) -> dict:
        candidate_point = points[candidate_id]["candidate_robust"]
        comparator_point = points[comparator_id]["candidate_robust"]
        candidate_draws, comparator_draws = draws[candidate_id], draws[comparator_id]
        definitions = {
            "spearman": (
                candidate_point["spearman"], comparator_point["spearman"],
                candidate_draws["spearman"] - comparator_draws["spearman"]),
            "mae_tvd": (
                comparator_point["mae_tvd"], candidate_point["mae_tvd"],
                comparator_draws["mae_tvd"] - candidate_draws["mae_tvd"]),
            "binary_flip_rate": (
                comparator_point["binary_flip_rate"],
                candidate_point["binary_flip_rate"],
                comparator_draws["binary_flip_rate"]
                - candidate_draws["binary_flip_rate"]),
            "absolute_bias": (
                comparator_point["absolute_bias"], candidate_point["absolute_bias"],
                comparator_draws["absolute_bias"]
                - candidate_draws["absolute_bias"]),
        }
        result = {}
        for metric, (positive, negative, values) in definitions.items():
            point = (None if positive is None or negative is None
                     else float(positive - negative))
            result[metric] = {
                "point": point,
                "CI": _ci(values, confidence),
                "positive_means_candidate_improves": True,
            }
        return result

    rows = []
    reference_point = points[reference_id]["candidate_robust"]
    for arm_id in evaluation_orbits:
        point = points[arm_id]
        row = {
            "arm_id": arm_id,
            "heldout_robust": point["candidate_robust"],
            "heldout_quotient": point["quotient"],
            "improvement_over_raw_name": (
                None if arm_id == name_arm_id else improvement(arm_id, name_arm_id)
            ),
            "improvement_over_oracle_mean_shift": (
                None if arm_id == reference_id else improvement(arm_id, reference_id)
            ),
        }
        if arm_id not in {name_arm_id, reference_id}:
            robust = point["candidate_robust"]
            row["oracle_mean_shift_point_dominates_all_four_coordinates"] = bool(
                reference_point["spearman"] >= robust["spearman"]
                and reference_point["mae_tvd"] <= robust["mae_tvd"]
                and reference_point["binary_flip_rate"] <= robust["binary_flip_rate"]
                and reference_point["absolute_bias"] <= robust["absolute_bias"]
            )
        rows.append(row)

    raw_values = np.stack(list(candidates[name_arm_id].values()))
    shifted_values = np.stack(list(shifted_full.values()))
    calibration_target_level = float(np.mean(q[calibration_indexes]))
    calibration_shifted_level = float(
        np.mean(shifted_values[:, calibration_indexes]))
    return {
        "schema": "policy_isomorphism_oracle_recalibration/v1",
        "status": "retrospective_target-score-oracle-diagnostic",
        "claim_eligible_as_unsupervised_reconstruction": False,
        "reason": (
            "the one-scalar intercept is fitted from larger-model calibration scores; "
            "it is a supervision price line, not an admissible articulation arm"
        ),
        "split": {
            "method": "sha256(salt + NUL + item_sha256) below the 2^255 threshold",
            "salt": salt,
            "n_total": n_items,
            "n_calibration": int(len(calibration_indexes)),
            "n_evaluation": int(len(evaluation_indexes)),
            "calibration_item_set_sha256": hashlib.sha256(
                "\n".join(sorted(str(item_hashes[index])
                                 for index in calibration_indexes)).encode()
            ).hexdigest(),
            "evaluation_item_set_sha256": hashlib.sha256(
                "\n".join(sorted(str(item_hashes[index])
                                 for index in evaluation_indexes)).encode()
            ).hexdigest(),
        },
        "oracle": {
            "kind": "one_scalar_bounded_additive_mean_matching",
            "fitted_shift": shift,
            "calibration_target_level": calibration_target_level,
            "calibration_shifted_level": calibration_shifted_level,
            "calibration_level_absolute_error": abs(
                calibration_target_level - calibration_shifted_level),
            "fraction_clipped_low": float(np.mean(raw_values + shift < 0.0)),
            "fraction_clipped_high": float(np.mean(raw_values + shift > 1.0)),
        },
        "bootstrap": {
            "n": n_boot,
            "seed": seed,
            "confidence": confidence,
            "method": "paired item bootstrap on the stable-hash evaluation fold",
        },
        "target_heldout": points[name_arm_id]["target_self_robust"],
        "rows": rows,
        "summary": {
            "n_articulation_arms": len(evaluation_orbits) - 2,
            "n_oracle_dominated_on_all_four_point_coordinates": sum(
                bool(row.get(
                    "oracle_mean_shift_point_dominates_all_four_coordinates"))
                for row in rows
            ),
            "n_articulation_rank_gains_over_oracle_with_ci_above_zero": sum(
                bool(row.get("improvement_over_oracle_mean_shift")
                     and row["improvement_over_oracle_mean_shift"]["spearman"]["CI"]
                     and row["improvement_over_oracle_mean_shift"]["spearman"]["CI"][0]
                     > 0.0)
                for row in rows
            ),
        },
    }


def _bootstrap_orbit_matrix(target_orbit: Mapping[str, Sequence[float]],
                            candidate_orbit: Mapping[str, Sequence[float]],
                            samples: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized orbit bootstrap for draws with a common item count."""
    target = validate_orbit(target_orbit, name="target_orbit")
    n_items = len(next(iter(target.values())))
    candidate = validate_orbit(candidate_orbit, n_items=n_items, name="candidate_orbit")
    sample_idx = np.asarray(samples, int)
    if sample_idx.ndim != 2 or sample_idx.shape[1] == 0:
        raise ValueError("samples must have shape (n_boot, positive sample size)")
    if sample_idx.size and (sample_idx.min() < 0 or sample_idx.max() >= n_items):
        raise ValueError("bootstrap sample index is out of range")
    q = np.mean(np.stack(list(target.values())), axis=0)
    q_sample = q[sample_idx]
    q_binary = q >= 0.5
    q_rank = _rowwise_midrank(q_sample)
    q_centered = q_rank - np.mean(q_rank, axis=1, keepdims=True)
    q_ss = np.sum(q_centered ** 2, axis=1)

    def draws(orbit: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        maes, rhos, flips, biases = [], [], [], []
        for values in orbit.values():
            value_sample = values[sample_idx]
            maes.append(np.mean(np.abs(value_sample - q_sample), axis=1))
            flips.append(np.mean((values[sample_idx] >= 0.5) != q_binary[sample_idx], axis=1))
            biases.append(np.abs(np.mean(value_sample - q_sample, axis=1)))
            value_rank = _rowwise_midrank(value_sample)
            value_centered = value_rank - np.mean(value_rank, axis=1, keepdims=True)
            denom = np.sqrt(q_ss * np.sum(value_centered ** 2, axis=1))
            rhos.append(np.divide(np.sum(q_centered * value_centered, axis=1), denom,
                                  out=np.full(len(sample_idx), np.nan), where=denom > 0))
        rho_matrix = np.stack(rhos)
        robust_rho = np.min(rho_matrix, axis=0)
        robust_rho[~np.all(np.isfinite(rho_matrix), axis=0)] = np.nan
        quotient_sample = np.mean(np.stack(list(orbit.values())), axis=0)[sample_idx]
        quotient_rank = _rowwise_midrank(quotient_sample)
        quotient_centered = quotient_rank - np.mean(
            quotient_rank, axis=1, keepdims=True
        )
        quotient_denom = np.sqrt(
            q_ss * np.sum(quotient_centered ** 2, axis=1)
        )
        quotient_rho = np.divide(
            np.sum(q_centered * quotient_centered, axis=1),
            quotient_denom,
            out=np.full(len(sample_idx), np.nan),
            where=quotient_denom > 0,
        )
        return {
            "mae_tvd": np.max(maes, axis=0),
            "spearman": robust_rho,
            "binary_flip_rate": np.max(flips, axis=0),
            "absolute_bias": np.max(biases, axis=0),
            "quotient_spearman": quotient_rho,
        }

    return {"target_self": draws(target), "candidate": draws(candidate)}


def _bootstrap_orbit(target_orbit: Mapping[str, Sequence[float]],
                     candidate_orbit: Mapping[str, Sequence[float]],
                     samples: np.ndarray | Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Bootstrap an orbit, accepting fixed- or variable-length paired item draws.

    A one-stage cluster bootstrap can contain a different number of items in each draw when
    source groups have unequal sizes.  Draws of equal length are batched so Spearman midranks are
    still recomputed inside each resample without reducing the clustered draw to group means.
    """
    if isinstance(samples, np.ndarray):
        return _bootstrap_orbit_matrix(target_orbit, candidate_orbit, samples)
    sample_rows = [np.asarray(row, int) for row in samples]
    if not sample_rows or any(row.ndim != 1 or not len(row) for row in sample_rows):
        raise ValueError("samples must contain positive-length one-dimensional draws")
    by_length: dict[int, list[int]] = {}
    for draw_index, row in enumerate(sample_rows):
        by_length.setdefault(len(row), []).append(draw_index)
    result = {
        orbit: {metric: np.empty(len(sample_rows), dtype=float)
                for metric in (
                    "mae_tvd", "spearman", "binary_flip_rate", "absolute_bias",
                    "quotient_spearman",
                )}
        for orbit in ("target_self", "candidate")
    }
    for draw_indexes in by_length.values():
        matrix = np.stack([sample_rows[index] for index in draw_indexes])
        batched = _bootstrap_orbit_matrix(target_orbit, candidate_orbit, matrix)
        for orbit in result:
            for metric in result[orbit]:
                result[orbit][metric][draw_indexes] = batched[orbit][metric]
    return result


def _validated_labels(labels: Sequence[str | int], *, n_items: int,
                      name: str) -> list[str | int]:
    values = list(labels)
    if len(values) != n_items:
        raise ValueError(f"{name} must have one label per item")
    for value in values:
        if not isinstance(value, (str, int, np.integer)):
            raise ValueError(f"{name} labels must be strings or integers")
    return values


def _index_groups(labels: Sequence[str | int], indexes: np.ndarray) -> list[np.ndarray]:
    groups: dict[str | int, list[int]] = {}
    for index in indexes:
        groups.setdefault(labels[int(index)], []).append(int(index))
    return [np.asarray(values, int) for values in groups.values()]


def _bootstrap_samples(
    *,
    rng: np.random.Generator,
    n_boot: int,
    n_items: int,
    strata: Sequence[str | int] | None = None,
    clusters: Sequence[str | int] | None = None,
) -> tuple[np.ndarray | list[np.ndarray], dict]:
    """Generate paired item or one-stage cluster draws, optionally within fixed strata."""
    all_indexes = np.arange(n_items)
    if strata is None:
        stratum_groups = [all_indexes]
    else:
        stratum_values = _validated_labels(
            strata, n_items=n_items, name="bootstrap_strata")
        stratum_groups = _index_groups(stratum_values, all_indexes)

    if clusters is None:
        samples = np.concatenate([
            group[rng.integers(0, len(group), size=(n_boot, len(group)))]
            for group in stratum_groups
        ], axis=1)
        return samples, {
            "sampling": (
                "paired item bootstrap"
                if strata is None
                else "paired item bootstrap stratified by supplied labels"
            ),
            "n_strata": len(stratum_groups),
            "resampling_unit": "item",
            "n_resampling_units": n_items,
            "point_estimand": "item-weighted policy metrics on the original scored panel",
        }

    cluster_values = _validated_labels(
        clusters, n_items=n_items, name="bootstrap_clusters")
    clusters_by_stratum = [
        _index_groups(cluster_values, stratum_group) for stratum_group in stratum_groups
    ]
    cluster_sizes = [
        len(cluster) for stratum_clusters in clusters_by_stratum for cluster in stratum_clusters
    ]
    all_singleton = all(size == 1 for size in cluster_sizes)
    if all_singleton:
        # This is exactly the former item bootstrap, including RNG call shape and item ordering.
        samples = np.concatenate([
            group[rng.integers(0, len(group), size=(n_boot, len(group)))]
            for group in stratum_groups
        ], axis=1)
    else:
        selections = [
            rng.integers(0, len(stratum_clusters), size=(n_boot, len(stratum_clusters)))
            for stratum_clusters in clusters_by_stratum
        ]
        samples = []
        for draw_index in range(n_boot):
            pieces = [
                stratum_clusters[selected]
                for stratum_clusters, selected_draw in zip(
                    clusters_by_stratum, selections
                )
                for selected in selected_draw[draw_index]
            ]
            samples.append(np.concatenate(pieces))
    sampling = (
        "paired one-stage source-group cluster bootstrap"
        if strata is None
        else "paired one-stage source-group cluster bootstrap stratified by supplied labels"
    )
    if all_singleton:
        sampling += "; singleton groups exactly reproduce paired item-bootstrap draws"
    return samples, {
        "sampling": sampling,
        "n_strata": len(stratum_groups),
        "resampling_unit": "source_group_with_all_member_items_retained",
        "n_resampling_units": len(cluster_sizes),
        "n_source_groups": len(cluster_sizes),
        "n_singleton_source_groups": sum(size == 1 for size in cluster_sizes),
        "min_source_group_size": min(cluster_sizes),
        "max_source_group_size": max(cluster_sizes),
        "mean_source_group_size": float(np.mean(cluster_sizes)),
        "all_source_groups_singleton": all_singleton,
        "point_estimand": "item-weighted policy metrics on the original scored panel",
        "draw_construction": (
            "sample source groups with replacement within each stratum; retain every member "
            "item for each sampled occurrence; recompute all metrics on expanded item draws"
        ),
    }


def _canonical_bootstrap_labels(
        labels: Sequence[str | int] | None, *, n_items: int, name: str
) -> tuple[str | int, ...] | None:
    """Return an immutable label representation with NumPy integers normalized."""
    if labels is None:
        return None
    return tuple(
        int(value) if isinstance(value, np.integer) else value
        for value in _validated_labels(labels, n_items=n_items, name=name)
    )


def _freeze_bootstrap_draws(draws: dict[str, np.ndarray] | dict[str, dict]) -> None:
    """Make cached draw arrays read-only so one certificate cannot corrupt another."""
    for value in draws.values():
        if isinstance(value, dict):
            _freeze_bootstrap_draws(value)
        elif isinstance(value, np.ndarray):
            value.setflags(write=False)


@dataclass(frozen=True, slots=True, init=False)
class PolicyBootstrapContext:
    """Exact reusable paired-bootstrap state for one item panel and resampling design.

    Confidence is deliberately absent from the context identity: nominal and multiplicity-adjusted
    intervals must use the same paired draws.  Every other draw-defining argument is frozen and
    checked on every use.  Orbit cache keys contain the complete validated float64 bytes (including
    form order), rather than a probabilistic digest.  Reusing and then mutating the same mapping or
    vector object fails closed instead of silently returning stale draws.

    The context does not change certificate serialization.  ``cache_info`` is an out-of-band,
    non-scientific performance diagnostic only.
    """

    _n_items: int
    _n_boot: int
    _seed: int
    _bootstrap_strata: tuple[str | int, ...] | None
    _bootstrap_clusters: tuple[str | int, ...] | None
    _samples: np.ndarray | tuple[np.ndarray, ...]
    _bootstrap_design: dict
    _orbit_bundles: dict
    _pairwise_draws: dict
    _identity_registry: dict
    _stats: dict[str, int]

    def __init__(
            self, *, n_items: int, n_boot: int = 2000, seed: int = 0,
            bootstrap_strata: Sequence[str | int] | None = None,
            bootstrap_clusters: Sequence[str | int] | None = None,
    ) -> None:
        if (isinstance(n_items, bool)
                or not isinstance(n_items, (int, np.integer))
                or n_items < 4):
            raise ValueError("n_items must be an integer of at least four")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        _validate_bootstrap_args(n_boot, 0.95)
        normalized_n_items = int(n_items)
        normalized_n_boot = int(n_boot)
        normalized_seed = int(seed)
        strata = _canonical_bootstrap_labels(
            bootstrap_strata,
            n_items=normalized_n_items,
            name="bootstrap_strata",
        )
        clusters = _canonical_bootstrap_labels(
            bootstrap_clusters,
            n_items=normalized_n_items,
            name="bootstrap_clusters",
        )
        samples, design = _bootstrap_samples(
            rng=np.random.default_rng(normalized_seed),
            n_boot=normalized_n_boot,
            n_items=normalized_n_items,
            strata=strata,
            clusters=clusters,
        )
        if isinstance(samples, np.ndarray):
            samples.setflags(write=False)
            frozen_samples: np.ndarray | tuple[np.ndarray, ...] = samples
        else:
            for row in samples:
                row.setflags(write=False)
            frozen_samples = tuple(samples)
        object.__setattr__(self, "_n_items", normalized_n_items)
        object.__setattr__(self, "_n_boot", normalized_n_boot)
        object.__setattr__(self, "_seed", normalized_seed)
        object.__setattr__(self, "_bootstrap_strata", strata)
        object.__setattr__(self, "_bootstrap_clusters", clusters)
        object.__setattr__(self, "_samples", frozen_samples)
        object.__setattr__(self, "_bootstrap_design", design)
        object.__setattr__(self, "_orbit_bundles", {})
        object.__setattr__(self, "_pairwise_draws", {})
        object.__setattr__(self, "_identity_registry", {})
        object.__setattr__(self, "_stats", {
            "orbit_bundle_hits": 0,
            "orbit_bundle_misses": 0,
            "pairwise_draw_hits": 0,
            "pairwise_draw_misses": 0,
        })

    @property
    def n_items(self) -> int:
        return self._n_items

    @property
    def n_boot(self) -> int:
        return self._n_boot

    @property
    def seed(self) -> int:
        return self._seed

    def _assert_compatible(
            self, *, n_items: int, n_boot: int, seed: int,
            bootstrap_strata: Sequence[str | int] | None,
            bootstrap_clusters: Sequence[str | int] | None,
    ) -> None:
        if int(n_items) != self._n_items:
            raise ValueError(
                f"bootstrap_context n_items mismatch: {self._n_items} != {n_items}"
            )
        if (isinstance(n_boot, bool)
                or not isinstance(n_boot, (int, np.integer))
                or int(n_boot) != self._n_boot):
            raise ValueError(
                f"bootstrap_context n_boot mismatch: {self._n_boot} != {n_boot}"
            )
        if (isinstance(seed, bool)
                or not isinstance(seed, (int, np.integer))
                or int(seed) != self._seed):
            raise ValueError(
                f"bootstrap_context seed mismatch: {self._seed} != {seed}"
            )
        strata = _canonical_bootstrap_labels(
            bootstrap_strata, n_items=self._n_items, name="bootstrap_strata"
        )
        clusters = _canonical_bootstrap_labels(
            bootstrap_clusters, n_items=self._n_items, name="bootstrap_clusters"
        )
        if strata != self._bootstrap_strata:
            raise ValueError("bootstrap_context bootstrap_strata mismatch")
        if clusters != self._bootstrap_clusters:
            raise ValueError("bootstrap_context bootstrap_clusters mismatch")

    def _guard_identity(self, value: object, fingerprint: object, *, name: str) -> None:
        identity = id(value)
        prior = self._identity_registry.get(identity)
        if prior is None:
            # Retaining the object also prevents a later Python object from reusing this id.
            self._identity_registry[identity] = (value, fingerprint, name)
            return
        if prior[0] is not value:
            raise RuntimeError("bootstrap context identity registry collision")
        if prior[1] != fingerprint:
            raise ValueError(
                f"{name} was mutated after binding to bootstrap_context"
            )

    def _canonical_orbit(
            self, orbit: Mapping[str, Sequence[float]], *, name: str
    ) -> tuple[dict[str, np.ndarray], tuple]:
        validated = validate_orbit(orbit, n_items=self._n_items, name=name)
        key_parts = []
        snapshots: dict[str, np.ndarray] = {}
        for form, values in validated.items():
            snapshot = np.array(values, dtype=float, order="C", copy=True)
            snapshot.setflags(write=False)
            value_bytes = snapshot.tobytes(order="C")
            key_parts.append((form, snapshot.shape, value_bytes))
            snapshots[form] = snapshot
        key = tuple(key_parts)
        self._guard_identity(orbit, key, name=name)
        # Guard shared mutable vectors as well as their enclosing mapping.  This catches an array
        # mutated through an alias and subsequently supplied inside a fresh mapping.
        for values in orbit.values():
            normalized = np.asarray(values, dtype=float)
            vector_key = (normalized.shape, normalized.tobytes(order="C"))
            self._guard_identity(values, vector_key, name=f"{name} vector")
        return snapshots, key

    def _orbit_bundle(
            self, target_orbit: Mapping[str, Sequence[float]],
            candidate_orbit: Mapping[str, Sequence[float]],
    ) -> dict[str, dict[str, np.ndarray]]:
        target, target_key = self._canonical_orbit(target_orbit, name="target_orbit")
        candidate, candidate_key = self._canonical_orbit(
            candidate_orbit, name="candidate_orbit"
        )
        key = (target_key, candidate_key)
        cached = self._orbit_bundles.get(key)
        if cached is not None:
            self._stats["orbit_bundle_hits"] += 1
            return cached
        bundle = _bootstrap_orbit(target, candidate, self._samples)
        _freeze_bootstrap_draws(bundle)
        self._orbit_bundles[key] = bundle
        self._stats["orbit_bundle_misses"] += 1
        return bundle

    def _pairwise_bundle(
            self, left_orbit: Mapping[str, Sequence[float]],
            right_orbit: Mapping[str, Sequence[float]],
    ) -> dict[str, np.ndarray]:
        left, left_key = self._canonical_orbit(left_orbit, name="left_orbit")
        right, right_key = self._canonical_orbit(right_orbit, name="right_orbit")
        if set(left) != set(right):
            raise ValueError("pairwise policy fidelity requires identical form-id sets")
        key = (left_key, right_key)
        cached = self._pairwise_draws.get(key)
        if cached is not None:
            self._stats["pairwise_draw_hits"] += 1
            return cached
        left_mean = np.mean(np.stack(list(left.values())), axis=0)
        right_mean = np.mean(np.stack(list(right.values())), axis=0)
        draws = _pairwise_quotient_bootstrap_draws(
            left_mean, right_mean, self._samples
        )
        _freeze_bootstrap_draws(draws)
        self._pairwise_draws[key] = draws
        self._stats["pairwise_draw_misses"] += 1
        return draws

    def cache_info(self) -> dict[str, int]:
        """Return non-scientific cache diagnostics; never embedded in certificates."""
        arrays: list[np.ndarray] = []
        if isinstance(self._samples, np.ndarray):
            arrays.append(self._samples)
        else:
            arrays.extend(self._samples)
        for bundle in self._orbit_bundles.values():
            arrays.extend(
                values
                for orbit in bundle.values()
                for values in orbit.values()
            )
        for bundle in self._pairwise_draws.values():
            arrays.extend(bundle.values())
        # Count aliased arrays once if a future implementation interns target-self bundles.
        storage_bytes = sum(array.nbytes for array in {id(a): a for a in arrays}.values())
        return {
            **self._stats,
            "orbit_bundles": len(self._orbit_bundles),
            "pairwise_draw_bundles": len(self._pairwise_draws),
            "registered_mutable_objects": len(self._identity_registry),
            "array_storage_bytes": int(storage_bytes),
        }


def _bootstrap_state(
        *, n_items: int, n_boot: int, seed: int,
        bootstrap_strata: Sequence[str | int] | None,
        bootstrap_clusters: Sequence[str | int] | None,
        bootstrap_context: PolicyBootstrapContext | None,
) -> tuple[np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray], dict]:
    """Resolve an exact fresh or context-owned paired resampling design."""
    if bootstrap_context is None:
        return _bootstrap_samples(
            rng=np.random.default_rng(seed),
            n_boot=n_boot,
            n_items=n_items,
            strata=bootstrap_strata,
            clusters=bootstrap_clusters,
        )
    if not isinstance(bootstrap_context, PolicyBootstrapContext):
        raise TypeError("bootstrap_context must be a PolicyBootstrapContext or None")
    bootstrap_context._assert_compatible(
        n_items=n_items,
        n_boot=n_boot,
        seed=seed,
        bootstrap_strata=bootstrap_strata,
        bootstrap_clusters=bootstrap_clusters,
    )
    return bootstrap_context._samples, dict(bootstrap_context._bootstrap_design)


def _cached_orbit_bundle(
        target_orbit: Mapping[str, Sequence[float]],
        candidate_orbit: Mapping[str, Sequence[float]],
        samples: np.ndarray | Sequence[np.ndarray],
        bootstrap_context: PolicyBootstrapContext | None,
) -> dict[str, dict[str, np.ndarray]]:
    if bootstrap_context is None:
        return _bootstrap_orbit(target_orbit, candidate_orbit, samples)
    return bootstrap_context._orbit_bundle(target_orbit, candidate_orbit)


def certify_policy_isomorphism(
        target_orbit: Mapping[str, Sequence[float]],
        candidate_orbit: Mapping[str, Sequence[float]], *,
        sparse_orbit: Mapping[str, Sequence[float]] | None = None,
        bootstrap_strata: Sequence[str | int] | None = None,
        bootstrap_clusters: Sequence[str | int] | None = None,
        mae_margin: float = 0.02, rho_margin: float = 0.05,
        flip_margin: float = 0.02, bias_margin: float = 0.02,
        functional_rho_floor: float = 0.70,
        min_target_information: float = 0.01, min_target_self_rho: float = 0.5,
        max_target_self_mae: float = 0.25, n_boot: int = 2000,
        seed: int = 0, confidence: float = 0.95,
        bootstrap_context: PolicyBootstrapContext | None = None) -> dict:
    """Certify one articulation against the larger-reader sparse policy itself."""
    _validate_bootstrap_args(n_boot, confidence)
    if not -1.0 <= functional_rho_floor <= 1.0:
        raise ValueError("functional_rho_floor must lie in [-1, 1]")
    point = _orbit_point(target_orbit, candidate_orbit)
    sparse_point = _orbit_point(target_orbit, sparse_orbit) if sparse_orbit is not None else None
    samples, bootstrap_design = _bootstrap_state(
        n_boot=n_boot,
        n_items=point["n_items"],
        seed=seed,
        bootstrap_strata=bootstrap_strata,
        bootstrap_clusters=bootstrap_clusters,
        bootstrap_context=bootstrap_context,
    )
    draws = _cached_orbit_bundle(
        target_orbit, candidate_orbit, samples, bootstrap_context
    )
    sparse_draws = (_cached_orbit_bundle(
                        target_orbit, sparse_orbit, samples, bootstrap_context)
                    if sparse_orbit is not None else None)
    rank_draw_counts = {
        "candidate_valid": _n_jointly_finite(draws["candidate"]["spearman"]),
        "candidate_quotient_valid": _n_jointly_finite(
            draws["candidate"]["quotient_spearman"]
        ),
        "target_self_valid": _n_jointly_finite(draws["target_self"]["spearman"]),
        "candidate_target_self_paired_valid": _n_jointly_finite(
            draws["candidate"]["spearman"], draws["target_self"]["spearman"]
        ),
    }
    if sparse_draws is not None:
        rank_draw_counts.update({
            "small_sparse_valid": _n_jointly_finite(
                sparse_draws["candidate"]["spearman"]),
            "candidate_small_sparse_paired_valid": _n_jointly_finite(
                draws["candidate"]["spearman"],
                sparse_draws["candidate"]["spearman"],
            ),
        })

    candidate_rho = point["candidate_robust"]["spearman"]
    target_self_rho = point["target_self_robust"]["spearman"]

    differences = {
        "mae_excess_over_target_self": {
            "point": float(point["candidate_robust"]["mae_tvd"]
                           - point["target_self_robust"]["mae_tvd"]),
            "CI": _ci(draws["candidate"]["mae_tvd"]
                      - draws["target_self"]["mae_tvd"], confidence),
        },
        "rho_minus_target_self": {
            "point": (None if candidate_rho is None or target_self_rho is None else
                      float(candidate_rho - target_self_rho)),
            "CI": _ci(draws["candidate"]["spearman"]
                      - draws["target_self"]["spearman"], confidence),
        },
        "flip_excess_over_target_self": {
            "point": float(point["candidate_robust"]["binary_flip_rate"]
                           - point["target_self_robust"]["binary_flip_rate"]),
            "CI": _ci(draws["candidate"]["binary_flip_rate"]
                      - draws["target_self"]["binary_flip_rate"], confidence),
        },
        "bias_excess_over_target_self": {
            "point": float(point["candidate_robust"]["absolute_bias"]
                           - point["target_self_robust"]["absolute_bias"]),
            "CI": _ci(draws["candidate"]["absolute_bias"]
                      - draws["target_self"]["absolute_bias"], confidence),
        },
    }
    if sparse_point is not None and sparse_draws is not None:
        differences["mae_gain_over_small_sparse"] = {
            "point": float(sparse_point["candidate_robust"]["mae_tvd"]
                           - point["candidate_robust"]["mae_tvd"]),
            "CI": _ci(sparse_draws["candidate"]["mae_tvd"]
                      - draws["candidate"]["mae_tvd"], confidence),
        }

    target_rho = target_self_rho
    target_valid = bool(point["target_information"] >= min_target_information
                        and target_rho is not None and target_rho >= min_target_self_rho
                        and point["target_self_robust"]["mae_tvd"] <= max_target_self_mae)
    mae_ci = differences["mae_excess_over_target_self"]["CI"]
    rho_ci = differences["rho_minus_target_self"]["CI"]
    flip_ci = differences["flip_excess_over_target_self"]["CI"]
    bias_ci = differences["bias_excess_over_target_self"]["CI"]
    gates = {
        "target_identity_valid": target_valid,
        "mae_inside_identity_band": bool(mae_ci and mae_ci[1] <= mae_margin),
        "rho_inside_identity_band": bool(rho_ci and rho_ci[0] >= -rho_margin),
        "flip_inside_identity_band": bool(flip_ci and flip_ci[1] <= flip_margin),
        "bias_inside_identity_band": bool(bias_ci and bias_ci[1] <= bias_margin),
        "positive_polarity": bool(point["candidate_robust"]["all_positive_polarity"]),
    }
    if "mae_gain_over_small_sparse" in differences:
        gain_ci = differences["mae_gain_over_small_sparse"]["CI"]
        gates["mae_improves_over_small_sparse"] = bool(gain_ci and gain_ci[0] > 0.0)

    candidate_rho_ci = _ci(draws["candidate"]["spearman"], confidence)
    sparse_rho_ci = (
        _ci(sparse_draws["candidate"]["spearman"], confidence)
        if sparse_draws is not None else None
    )
    quotient_rho = point["quotient"]["spearman"]
    quotient_rho_ci = _ci(
        draws["candidate"]["quotient_spearman"], confidence
    )
    mae_gain_point = (differences.get("mae_gain_over_small_sparse", {})
                      .get("point"))
    functional_gates = {
        "target_identity_valid": target_valid,
        "positive_polarity": bool(point["candidate_robust"]["all_positive_polarity"]),
        "adverse_rank_point_at_least_floor": bool(
            candidate_rho is not None and candidate_rho >= functional_rho_floor
        ),
        "adverse_rank_lower_CI_at_least_floor": bool(
            candidate_rho_ci and candidate_rho_ci[0] >= functional_rho_floor
        ),
        "quotient_rank_point_at_least_floor": bool(
            quotient_rho is not None and quotient_rho >= functional_rho_floor
        ),
        "quotient_rank_lower_CI_at_least_floor": bool(
            quotient_rho_ci and quotient_rho_ci[0] >= functional_rho_floor
        ),
        "mae_point_improves_over_small_sparse": bool(
            mae_gain_point is not None and mae_gain_point > 0.0
        ),
        "mae_CI_improves_over_small_sparse": bool(
            gates.get("mae_improves_over_small_sparse", False)
        ),
        "small_sparse_point_below_functional_floor": bool(
            sparse_point is not None
            and sparse_point["candidate_robust"]["spearman"] is not None
            and sparse_point["candidate_robust"]["spearman"] < functional_rho_floor
        ),
        "small_sparse_upper_CI_below_functional_floor": bool(
            sparse_rho_ci and sparse_rho_ci[1] < functional_rho_floor
        ),
    }
    observed_functional_ordinal = bool(all(functional_gates[key] for key in (
        "target_identity_valid", "positive_polarity",
        "adverse_rank_point_at_least_floor", "quotient_rank_point_at_least_floor",
    )))
    certified_functional_ordinal = bool(all(functional_gates[key] for key in (
        "target_identity_valid", "positive_polarity",
        "adverse_rank_lower_CI_at_least_floor",
        "quotient_rank_lower_CI_at_least_floor",
    )))
    observed_functional_substitution = bool(
        observed_functional_ordinal
        and functional_gates["mae_point_improves_over_small_sparse"]
        and functional_gates["small_sparse_point_below_functional_floor"]
    )
    certified_functional_substitution = bool(
        certified_functional_ordinal
        and functional_gates["mae_CI_improves_over_small_sparse"]
        and functional_gates["small_sparse_upper_CI_below_functional_floor"]
    )
    policy_isomorphic = bool(all(gates[key] for key in (
        "target_identity_valid", "mae_inside_identity_band", "rho_inside_identity_band",
        "flip_inside_identity_band", "bias_inside_identity_band", "positive_polarity")))
    sparse_isomorphic = None
    if sparse_point is not None and sparse_draws is not None:
        # This is exactly the policy-isomorphic gate set that a recursive sparse-as-candidate
        # call used to compute.  Reuse the already paired sparse and target-self draws instead:
        # the recursive call regenerated identical samples from the same seed, doubling runtime
        # (and becoming especially expensive for unequal-size source-group cluster draws).
        sparse_rho = sparse_point["candidate_robust"]["spearman"]
        sparse_target_rho = sparse_point["target_self_robust"]["spearman"]
        sparse_identity_cis = {
            metric: _ci(
                sparse_draws["candidate"][metric]
                - sparse_draws["target_self"][metric],
                confidence,
            )
            for metric in (
                "mae_tvd", "spearman", "binary_flip_rate", "absolute_bias")
        }
        sparse_identity_gates = {
            "target_identity_valid": target_valid,
            "mae_inside_identity_band": bool(
                sparse_identity_cis["mae_tvd"]
                and sparse_identity_cis["mae_tvd"][1] <= mae_margin
            ),
            "rho_inside_identity_band": bool(
                sparse_identity_cis["spearman"]
                and sparse_identity_cis["spearman"][0] >= -rho_margin
            ),
            "flip_inside_identity_band": bool(
                sparse_identity_cis["binary_flip_rate"]
                and sparse_identity_cis["binary_flip_rate"][1] <= flip_margin
            ),
            "bias_inside_identity_band": bool(
                sparse_identity_cis["absolute_bias"]
                and sparse_identity_cis["absolute_bias"][1] <= bias_margin
            ),
            "positive_polarity": bool(
                sparse_point["candidate_robust"]["all_positive_polarity"]),
        }
        # Keep the point values alive as a guard against accidental semantic drift toward a
        # rank-only shortcut: undefined robust rank necessarily fails the CI gate above, just as
        # in the former recursive certificate.
        if sparse_rho is None or sparse_target_rho is None:
            sparse_identity_gates["rho_inside_identity_band"] = False
        sparse_isomorphic = bool(all(sparse_identity_gates.values()))
    return {
        "schema": SCHEMA,
        "estimand": "direct larger-sparse-policy transplantation",
        "point": point,
        "small_sparse_point": sparse_point,
        "differences": differences,
        "margins": {"mae": mae_margin, "rho": rho_margin, "flip": flip_margin,
                    "bias": bias_margin},
        "bootstrap": {"n": n_boot, "n_requested": n_boot,
                      "rank_draw_counts": rank_draw_counts,
                      "seed": seed, "confidence": confidence,
                      **bootstrap_design,
                      "rank_method": (
                          "paired resampling with midranks recomputed within each resample "
                          "after expansion to member items"
                      )},
        "gates": gates,
        "functional": {
            "estimand": ("approximate ordinal policy reconstruction plus direct MAE "
                          "improvement; distinct from target-self-band near-identity"),
            "epsilon_rank_loss": float(1.0 - functional_rho_floor),
            "adverse_rho_floor": float(functional_rho_floor),
            "adverse_rho_point": candidate_rho,
            "adverse_rho_CI": candidate_rho_ci,
            "small_sparse_adverse_rho_CI": sparse_rho_ci,
            "quotient_rho_point": quotient_rho,
            "quotient_rho_CI": quotient_rho_ci,
            "gates": functional_gates,
            "observed_functional_ordinal_isomorphism": observed_functional_ordinal,
            "certified_functional_ordinal_isomorphism": certified_functional_ordinal,
            "observed_functional_policy_substitution": observed_functional_substitution,
            "certified_functional_policy_substitution": certified_functional_substitution,
            "claim_boundary": (
                "Observed requires both adverse-form and form-quotient point ranks to clear "
                "the floor; certified requires both bootstrap lower bounds to clear it. Policy "
                "substitution additionally requires direct MAE improvement over "
                "the small name-only policy and a baseline rank gap below the declared floor "
                "at the corresponding point/interval grade. Neither status implies "
                "target-self-band identity."
            ),
        },
        "policy_isomorphic": policy_isomorphic,
        "small_sparse_isomorphic": sparse_isomorphic,
        "articulation_rescue": bool(policy_isomorphic and sparse_isomorphic is False
                                    and gates.get("mae_improves_over_small_sparse")),
    }


def pairwise_policy_fidelity(left_orbit: Mapping[str, Sequence[float]],
                             right_orbit: Mapping[str, Sequence[float]]) -> dict:
    """Mutual quotient-policy distances, independent of the pair's shared target."""
    left = validate_orbit(left_orbit, name="left_orbit")
    n_items = len(next(iter(left.values())))
    right = validate_orbit(right_orbit, n_items=n_items, name="right_orbit")
    if set(left) != set(right):
        raise ValueError("pairwise policy fidelity requires identical form-id sets")
    left_mean = np.mean(np.stack(list(left.values())), axis=0)
    right_mean = np.mean(np.stack(list(right.values())), axis=0)
    rho = spearman(left_mean, right_mean)
    return {
        "quotient_mae_tvd": float(np.mean(np.abs(left_mean - right_mean))),
        "quotient_spearman": None if rho is None or not np.isfinite(rho) else float(rho),
        "quotient_binary_flip_rate": float(np.mean((left_mean >= 0.5) != (right_mean >= 0.5))),
        "quotient_absolute_bias": float(np.abs(np.mean(right_mean - left_mean))),
    }


def _pairwise_quotient_draw_matrix(
        left_mean: np.ndarray, right_mean: np.ndarray,
        sample_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized mutual quotient-policy metrics for equal-length paired draws."""
    left_sample = left_mean[sample_idx]
    right_sample = right_mean[sample_idx]
    left_rank = _rowwise_midrank(left_sample)
    right_rank = _rowwise_midrank(right_sample)
    left_centered = left_rank - np.mean(left_rank, axis=1, keepdims=True)
    right_centered = right_rank - np.mean(right_rank, axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(left_centered ** 2, axis=1)
        * np.sum(right_centered ** 2, axis=1)
    )
    rho = np.divide(
        np.sum(left_centered * right_centered, axis=1),
        denominator,
        out=np.full(len(sample_idx), np.nan),
        where=denominator > 0,
    )
    return {
        "quotient_spearman": rho,
        "quotient_mae_tvd": np.mean(np.abs(left_sample - right_sample), axis=1),
        "quotient_binary_flip_rate": np.mean(
            (left_sample >= 0.5) != (right_sample >= 0.5), axis=1),
        "quotient_absolute_bias": np.abs(
            np.mean(right_sample - left_sample, axis=1)),
    }


def _pairwise_quotient_bootstrap_draws(
        left_mean: np.ndarray, right_mean: np.ndarray,
        samples: np.ndarray | Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Mutual quotient draws for fixed-length item or variable-length cluster bootstraps."""
    if isinstance(samples, np.ndarray):
        return _pairwise_quotient_draw_matrix(left_mean, right_mean, samples)
    sample_rows = [np.asarray(row, int) for row in samples]
    by_length: dict[int, list[int]] = {}
    for draw_index, row in enumerate(sample_rows):
        by_length.setdefault(len(row), []).append(draw_index)
    metrics = (
        "quotient_spearman", "quotient_mae_tvd",
        "quotient_binary_flip_rate", "quotient_absolute_bias",
    )
    result = {metric: np.empty(len(sample_rows), dtype=float) for metric in metrics}
    for draw_indexes in by_length.values():
        matrix = np.stack([sample_rows[index] for index in draw_indexes])
        batch = _pairwise_quotient_draw_matrix(left_mean, right_mean, matrix)
        for metric in metrics:
            result[metric][draw_indexes] = batch[metric]
    return result


def certify_pairwise_policy_fidelity(
        left_orbit: Mapping[str, Sequence[float]],
        right_orbit: Mapping[str, Sequence[float]], *,
        bootstrap_strata: Sequence[str | int] | None = None,
        bootstrap_clusters: Sequence[str | int] | None = None,
        rho_floor: float = 0.90,
        rho_sensitivity_floor: float = 0.85,
        min_rank_valid_fraction: float = 0.99,
        mae_margin: float = 0.02,
        flip_margin: float = 0.02,
        bias_margin: float = 0.02,
        n_boot: int = 2000, seed: int = 0, confidence: float = 0.95,
        bootstrap_context: PolicyBootstrapContext | None = None) -> dict:
    """Certify the mutual quotient rank of two independently admissible articulations."""
    _validate_bootstrap_args(n_boot, confidence)
    for name, value in (("rho_floor", rho_floor),
                        ("rho_sensitivity_floor", rho_sensitivity_floor)):
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [-1, 1]")
    if rho_sensitivity_floor > rho_floor:
        raise ValueError("rho_sensitivity_floor cannot exceed rho_floor")
    if (not np.isfinite(min_rank_valid_fraction)
            or not 0.0 < min_rank_valid_fraction <= 1.0):
        raise ValueError("min_rank_valid_fraction must lie in (0, 1]")
    for name, value in (("mae_margin", mae_margin), ("flip_margin", flip_margin),
                        ("bias_margin", bias_margin)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    left = validate_orbit(left_orbit, name="left_orbit")
    n_items = len(next(iter(left.values())))
    right = validate_orbit(right_orbit, n_items=n_items, name="right_orbit")
    if set(left) != set(right):
        raise ValueError("pairwise policy fidelity requires identical form-id sets")
    point = pairwise_policy_fidelity(left, right)
    samples, bootstrap_design = _bootstrap_state(
        n_boot=n_boot,
        n_items=n_items,
        seed=seed,
        bootstrap_strata=bootstrap_strata,
        bootstrap_clusters=bootstrap_clusters,
        bootstrap_context=bootstrap_context,
    )
    if bootstrap_context is None:
        left_mean = np.mean(np.stack(list(left.values())), axis=0)
        right_mean = np.mean(np.stack(list(right.values())), axis=0)
        draws = _pairwise_quotient_bootstrap_draws(
            left_mean, right_mean, samples)
    else:
        draws = bootstrap_context._pairwise_bundle(left_orbit, right_orbit)
    rho_draws = draws["quotient_spearman"]
    rho_ci = _ci(rho_draws, confidence)
    mae_ci = _ci(draws["quotient_mae_tvd"], confidence)
    flip_ci = _ci(draws["quotient_binary_flip_rate"], confidence)
    bias_ci = _ci(draws["quotient_absolute_bias"], confidence)
    rho_point = point["quotient_spearman"]
    n_rank_valid = int(np.sum(np.isfinite(rho_draws)))
    rank_valid_fraction = n_rank_valid / n_boot
    rank_validity_pass = rank_valid_fraction >= min_rank_valid_fraction
    point_vector_equivalent = bool(
        rho_point is not None
        and rho_point >= rho_floor
        and point["quotient_mae_tvd"] <= mae_margin
        and point["quotient_binary_flip_rate"] <= flip_margin
        and point["quotient_absolute_bias"] <= bias_margin
    )
    certified_vector_equivalent = bool(
        rank_validity_pass
        and rho_ci and rho_ci[0] >= rho_floor
        and mae_ci and mae_ci[1] <= mae_margin
        and flip_ci and flip_ci[1] <= flip_margin
        and bias_ci and bias_ci[1] <= bias_margin
    )
    return {
        "schema": "pairwise_policy_fidelity/v3_interval_vector",
        "estimand": (
            "mutual item-order fidelity between two articulation-induced form-quotient policies"
        ),
        "n_items": n_items,
        "point": point,
        "quotient_spearman_CI": rho_ci,
        "quotient_mae_tvd_CI": mae_ci,
        "quotient_binary_flip_rate_CI": flip_ci,
        "quotient_absolute_bias_CI": bias_ci,
        "bootstrap": {
            "n": n_boot,
            "n_requested": n_boot,
            "n_rank_valid": n_rank_valid,
            "rank_valid_fraction": rank_valid_fraction,
            "min_rank_valid_fraction": min_rank_valid_fraction,
            "rank_validity_pass": rank_validity_pass,
            "seed": seed,
            "confidence": confidence,
            **bootstrap_design,
            "rank_method": (
                "paired resampling with midranks recomputed within each resample after "
                "expansion to member items"
            ),
        },
        "floors": {
            "primary": float(rho_floor),
            "sensitivity": float(rho_sensitivity_floor),
        },
        "vector_equivalence_margins": {
            "quotient_mae_tvd": mae_margin,
            "quotient_binary_flip_rate": flip_margin,
            "quotient_absolute_bias": bias_margin,
        },
        "gates": {
            "point_at_least_primary_floor": bool(
                rho_point is not None and rho_point >= rho_floor),
            "lower_CI_at_least_primary_floor": bool(
                rank_validity_pass and rho_ci and rho_ci[0] >= rho_floor),
            "point_at_least_sensitivity_floor": bool(
                rho_point is not None and rho_point >= rho_sensitivity_floor),
            "lower_CI_at_least_sensitivity_floor": bool(
                rank_validity_pass and rho_ci and rho_ci[0] >= rho_sensitivity_floor),
            "point_quotient_mae_within_margin": bool(
                point["quotient_mae_tvd"] <= mae_margin),
            "upper_CI_quotient_mae_within_margin": bool(
                mae_ci and mae_ci[1] <= mae_margin),
            "point_quotient_flip_within_margin": bool(
                point["quotient_binary_flip_rate"] <= flip_margin),
            "upper_CI_quotient_flip_within_margin": bool(
                flip_ci and flip_ci[1] <= flip_margin),
            "point_quotient_bias_within_margin": bool(
                point["quotient_absolute_bias"] <= bias_margin),
            "upper_CI_quotient_bias_within_margin": bool(
                bias_ci and bias_ci[1] <= bias_margin),
            "point_vector_equivalent": point_vector_equivalent,
            "certified_vector_equivalent": certified_vector_equivalent,
        },
        "claim_boundary": (
            "The primary mutual gate certifies quotient rank only. The nested vector-equivalence "
            "gate additionally bounds quotient MAE, threshold flips, and calibration bias, but "
            "does not assert matched-form or semantic equality. Each articulation must "
            "independently pass the declared target, native-gap, gain, and content-specificity "
            "gates before either pair grade is a scale-substitution fiber."
        ),
    }


def compare_articulation_to_matched_control(
        target_orbit: Mapping[str, Sequence[float]],
        source_orbit: Mapping[str, Sequence[float]],
        control_orbit: Mapping[str, Sequence[float]], *,
        bootstrap_strata: Sequence[str | int] | None = None,
        bootstrap_clusters: Sequence[str | int] | None = None,
        n_boot: int = 2000, seed: int = 0, confidence: float = 0.95,
        bootstrap_context: PolicyBootstrapContext | None = None) -> dict:
    """Paired source-minus-control certificate under the same target and item bootstrap."""
    _validate_bootstrap_args(n_boot, confidence)
    source_point = _orbit_point(target_orbit, source_orbit)
    control_point = _orbit_point(target_orbit, control_orbit)
    if source_point["n_items"] != control_point["n_items"]:
        raise ValueError("source and control item counts differ")
    samples, bootstrap_design = _bootstrap_state(
        n_boot=n_boot,
        n_items=source_point["n_items"],
        seed=seed,
        bootstrap_strata=bootstrap_strata,
        bootstrap_clusters=bootstrap_clusters,
        bootstrap_context=bootstrap_context,
    )
    source_draws = _cached_orbit_bundle(
        target_orbit, source_orbit, samples, bootstrap_context
    )["candidate"]
    control_draws = _cached_orbit_bundle(
        target_orbit, control_orbit, samples, bootstrap_context
    )["candidate"]
    source_rho = source_point["candidate_robust"]["spearman"]
    control_rho = control_point["candidate_robust"]["spearman"]
    rho_draws = source_draws["spearman"] - control_draws["spearman"]
    mae_draws = control_draws["mae_tvd"] - source_draws["mae_tvd"]
    rho_ci = _ci(rho_draws, confidence)
    mae_ci = _ci(mae_draws, confidence)
    rho_point = None if source_rho is None or control_rho is None else float(
        source_rho - control_rho)
    mae_point = float(
        control_point["candidate_robust"]["mae_tvd"]
        - source_point["candidate_robust"]["mae_tvd"])
    return {
        "schema": "matched_control_policy_comparison/v1",
        "estimand": "same-target explicit-content advantage over matched control",
        "n_items": source_point["n_items"],
        "bootstrap": {
            "n": n_boot,
            "n_requested": n_boot,
            "n_paired_rank_valid": _n_jointly_finite(
                source_draws["spearman"], control_draws["spearman"]
            ),
            "seed": seed,
            "confidence": confidence,
            **bootstrap_design,
            "rank_method": (
                "paired resampling with midranks recomputed within each resample after "
                "expansion to member items"
            ),
        },
        "source": source_point["candidate_robust"],
        "control": control_point["candidate_robust"],
        "differences": {
            "rho_advantage_source_minus_control": {"point": rho_point, "CI": rho_ci},
            "mae_advantage_control_minus_source": {"point": mae_point, "CI": mae_ci},
        },
        "gates": {
            "source_rank_better_point": bool(rho_point is not None and rho_point > 0.0),
            "source_rank_better_CI": bool(rho_ci and rho_ci[0] > 0.0),
            "source_mae_better_point": bool(mae_point > 0.0),
            "source_mae_better_CI": bool(mae_ci and mae_ci[0] > 0.0),
        },
    }


def certify_scale_step_substitution(
        target_orbit: Mapping[str, Sequence[float]],
        small_sparse_orbit: Mapping[str, Sequence[float]],
        candidate_orbit: Mapping[str, Sequence[float]],
        larger_sparse_orbit: Mapping[str, Sequence[float]], *,
        bootstrap_strata: Sequence[str | int] | None = None,
        bootstrap_clusters: Sequence[str | int] | None = None,
        endpoint_mae_margin: float = 0.02,
        endpoint_rho_margin: float = 0.05,
        endpoint_flip_margin: float = 0.02,
        endpoint_bias_margin: float = 0.02,
        functional_rho_floor: float = 0.70,
        min_rho_gain: float = 0.0,
        min_mae_gain: float = 0.0,
        min_target_information: float = 0.01,
        min_target_self_rho: float = 0.5,
        max_target_self_mae: float = 0.25,
        n_boot: int = 2000,
        seed: int = 0,
        confidence: float = 0.95,
        bootstrap_context: PolicyBootstrapContext | None = None) -> dict:
    """Certify explicit articulation as a replacement for one native scale step.

    All four policies are evaluated on the same items against the same fixed larger-policy
    target, and every bootstrap draw resamples those items (or source groups) once for all
    policies.  A direct endpoint-isomorphic local scale substitution requires four logically
    separate results:

    1. the larger sparse/name policy improves over the smaller sparse/name policy;
    2. the articulated smaller policy improves over its own sparse/name policy; and
    3. the articulated endpoint is compared with the larger sparse/name endpoint under both
       one-sided fixed-target noninferiority and two-sided fixed-target equivalence; and
    4. the articulated policy is compared directly with the larger sparse/name policy.

    Rank and MAE are the primary scale-step coordinates.  A stricter vector grade also requires
    endpoint noninferiority/equivalence for threshold flips and absolute calibration bias.
    Absolute fidelity to the fixed target and direct fidelity to the larger sparse endpoint are
    reported separately.  The legacy local-primary/vector keys are retained as explicitly
    one-sided noninferiority-recovery aliases; only the new direct endpoint grades can support an
    endpoint-policy-isomorphism claim.
    """
    _validate_bootstrap_args(n_boot, confidence)
    margins = {
        "mae": endpoint_mae_margin,
        "rho": endpoint_rho_margin,
        "flip": endpoint_flip_margin,
        "bias": endpoint_bias_margin,
    }
    for name, value in {
        **{f"endpoint_{key}_margin": value for key, value in margins.items()},
        "min_rho_gain": min_rho_gain,
        "min_mae_gain": min_mae_gain,
        "min_target_information": min_target_information,
        "max_target_self_mae": max_target_self_mae,
    }.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not -1.0 <= functional_rho_floor <= 1.0:
        raise ValueError("functional_rho_floor must lie in [-1, 1]")
    if not -1.0 <= min_target_self_rho <= 1.0:
        raise ValueError("min_target_self_rho must lie in [-1, 1]")

    policy_orbits = {
        "small_sparse": small_sparse_orbit,
        "candidate": candidate_orbit,
        "larger_sparse": larger_sparse_orbit,
    }
    policy_points = {
        name: _orbit_point(target_orbit, orbit)
        for name, orbit in policy_orbits.items()
    }
    n_items = policy_points["candidate"]["n_items"]
    # _orbit_point validates every candidate against the target item count; keep an explicit
    # assertion here because the shared-draw interpretation is central to this certificate.
    if any(point["n_items"] != n_items for point in policy_points.values()):
        raise ValueError("scale-step policies must have identical item counts")

    samples, bootstrap_design = _bootstrap_state(
        n_boot=n_boot,
        n_items=n_items,
        seed=seed,
        bootstrap_strata=bootstrap_strata,
        bootstrap_clusters=bootstrap_clusters,
        bootstrap_context=bootstrap_context,
    )
    bootstrap_bundles = {
        name: _cached_orbit_bundle(
            target_orbit, orbit, samples, bootstrap_context)
        for name, orbit in policy_orbits.items()
    }
    policy_draws = {
        name: bundle["candidate"] for name, bundle in bootstrap_bundles.items()
    }
    target_draws = bootstrap_bundles["candidate"]["target_self"]

    # The fixed-target comparisons above answer whether each executor performs well against a
    # third policy.  They do not answer whether the articulated smaller executor reproduces the
    # *larger sparse endpoint itself*: a candidate can overshoot the larger endpoint against the
    # fixed target and thereby pass every one-sided noninferiority gate while remaining behaviorally
    # unlike that endpoint.  Compute that direct relation on the exact same paired bootstrap sample
    # indexes.  This is intentionally inlined rather than delegated to
    # certify_policy_isomorphism(), which would generate a second set of bootstrap draws.
    direct_endpoint_point = _orbit_point(larger_sparse_orbit, candidate_orbit)
    direct_endpoint_small_point = _orbit_point(
        larger_sparse_orbit, small_sparse_orbit)
    direct_endpoint_bundle = _cached_orbit_bundle(
        larger_sparse_orbit, candidate_orbit, samples, bootstrap_context)
    direct_endpoint_small_bundle = _cached_orbit_bundle(
        larger_sparse_orbit, small_sparse_orbit, samples, bootstrap_context)
    direct_endpoint_draws = direct_endpoint_bundle["candidate"]
    direct_endpoint_self_draws = direct_endpoint_bundle["target_self"]
    direct_endpoint_small_draws = direct_endpoint_small_bundle["candidate"]

    robust_points = {
        name: point["candidate_robust"] for name, point in policy_points.items()
    }
    target_point = policy_points["candidate"]
    target_robust = target_point["target_self_robust"]

    def difference(
            left_policy: str, right_policy: str, metric: str, *,
            label: str) -> tuple[str, dict]:
        """Return an improvement-oriented left-minus-right paired difference."""
        left_point = robust_points[left_policy][metric]
        right_point = robust_points[right_policy][metric]
        point_difference = (
            None if left_point is None or right_point is None
            else float(left_point - right_point)
        )
        return label, {
            "point": point_difference,
            "CI": _ci(
                policy_draws[left_policy][metric]
                - policy_draws[right_policy][metric],
                confidence,
            ),
            "direction": (
                "positive favors the beneficiary named in this improvement-oriented label"
            ),
        }

    # For errors, left-minus-right is reversed so every positive value still means improvement.
    difference_items = [
        difference("larger_sparse", "small_sparse", "spearman",
                   label="native_rho_larger_minus_small"),
        difference("small_sparse", "larger_sparse", "mae_tvd",
                   label="native_mae_improvement_larger_over_small"),
        difference("candidate", "small_sparse", "spearman",
                   label="articulation_rho_candidate_minus_small"),
        difference("small_sparse", "candidate", "mae_tvd",
                   label="articulation_mae_improvement_candidate_over_small"),
        difference("candidate", "larger_sparse", "spearman",
                   label="endpoint_rho_candidate_minus_larger"),
        difference("larger_sparse", "candidate", "mae_tvd",
                   label="endpoint_mae_improvement_candidate_over_larger"),
        difference("larger_sparse", "candidate", "binary_flip_rate",
                   label="endpoint_flip_improvement_candidate_over_larger"),
        difference("larger_sparse", "candidate", "absolute_bias",
                   label="endpoint_bias_improvement_candidate_over_larger"),
    ]
    differences = dict(difference_items)

    def target_difference(metric: str, *, error_metric: bool) -> dict:
        candidate_point = robust_points["candidate"][metric]
        target_value = target_robust[metric]
        if error_metric:
            point_difference = float(candidate_point - target_value)
            draw_difference = policy_draws["candidate"][metric] - target_draws[metric]
            direction = "positive is candidate excess over target self"
        else:
            point_difference = (
                None if candidate_point is None or target_value is None
                else float(candidate_point - target_value)
            )
            draw_difference = policy_draws["candidate"][metric] - target_draws[metric]
            direction = "positive favors candidate over target self"
        return {
            "point": point_difference,
            "CI": _ci(draw_difference, confidence),
            "direction": direction,
        }

    target_differences = {
        "mae_excess_over_target_self": target_difference("mae_tvd", error_metric=True),
        "rho_minus_target_self": target_difference("spearman", error_metric=False),
        "flip_excess_over_target_self": target_difference(
            "binary_flip_rate", error_metric=True),
        "bias_excess_over_target_self": target_difference(
            "absolute_bias", error_metric=True),
    }

    def point_above(name: str, threshold: float, *, strict: bool) -> bool:
        value = differences[name]["point"]
        if value is None:
            return False
        return bool(value > threshold if strict else value >= threshold)

    def lower_above(name: str, threshold: float, *, strict: bool) -> bool:
        interval = differences[name]["CI"]
        if not interval:
            return False
        return bool(interval[0] > threshold if strict else interval[0] >= threshold)

    def point_inside_symmetric_margin(name: str, margin: float) -> bool:
        value = differences[name]["point"]
        return bool(value is not None and -margin <= value <= margin)

    def ci_inside_symmetric_margin(name: str, margin: float) -> bool:
        interval = differences[name]["CI"]
        return bool(interval and interval[0] >= -margin and interval[1] <= margin)

    target_self_rho = target_robust["spearman"]
    target_valid = bool(
        target_point["target_information"] >= min_target_information
        and target_self_rho is not None
        and target_self_rho >= min_target_self_rho
        and target_robust["mae_tvd"] <= max_target_self_mae
    )
    polarity_valid = {
        name: bool(point["all_positive_polarity"])
        for name, point in robust_points.items()
    }

    observed_gates = {
        "target_identity_valid": target_valid,
        "candidate_positive_polarity": polarity_valid["candidate"],
        "larger_sparse_positive_polarity": polarity_valid["larger_sparse"],
        "native_rank_gap": point_above(
            "native_rho_larger_minus_small", min_rho_gain, strict=True),
        "native_mae_gap": point_above(
            "native_mae_improvement_larger_over_small", min_mae_gain, strict=True),
        "articulation_rank_gain": point_above(
            "articulation_rho_candidate_minus_small", min_rho_gain, strict=True),
        "articulation_mae_gain": point_above(
            "articulation_mae_improvement_candidate_over_small",
            min_mae_gain,
            strict=True,
        ),
        "endpoint_rank_noninferior": point_above(
            "endpoint_rho_candidate_minus_larger", -endpoint_rho_margin, strict=False),
        "endpoint_mae_noninferior": point_above(
            "endpoint_mae_improvement_candidate_over_larger",
            -endpoint_mae_margin,
            strict=False,
        ),
        "endpoint_flip_noninferior": point_above(
            "endpoint_flip_improvement_candidate_over_larger",
            -endpoint_flip_margin,
            strict=False,
        ),
        "endpoint_bias_noninferior": point_above(
            "endpoint_bias_improvement_candidate_over_larger",
            -endpoint_bias_margin,
            strict=False,
        ),
        "endpoint_rank_two_sided_equivalent": point_inside_symmetric_margin(
            "endpoint_rho_candidate_minus_larger", endpoint_rho_margin),
        "endpoint_mae_two_sided_equivalent": point_inside_symmetric_margin(
            "endpoint_mae_improvement_candidate_over_larger", endpoint_mae_margin),
        "endpoint_flip_two_sided_equivalent": point_inside_symmetric_margin(
            "endpoint_flip_improvement_candidate_over_larger", endpoint_flip_margin),
        "endpoint_bias_two_sided_equivalent": point_inside_symmetric_margin(
            "endpoint_bias_improvement_candidate_over_larger", endpoint_bias_margin),
    }
    certified_gates = {
        "target_identity_valid": target_valid,
        "candidate_positive_polarity": polarity_valid["candidate"],
        "larger_sparse_positive_polarity": polarity_valid["larger_sparse"],
        "native_rank_gap": lower_above(
            "native_rho_larger_minus_small", min_rho_gain, strict=True),
        "native_mae_gap": lower_above(
            "native_mae_improvement_larger_over_small", min_mae_gain, strict=True),
        "articulation_rank_gain": lower_above(
            "articulation_rho_candidate_minus_small", min_rho_gain, strict=True),
        "articulation_mae_gain": lower_above(
            "articulation_mae_improvement_candidate_over_small",
            min_mae_gain,
            strict=True,
        ),
        "endpoint_rank_noninferior": lower_above(
            "endpoint_rho_candidate_minus_larger", -endpoint_rho_margin, strict=False),
        "endpoint_mae_noninferior": lower_above(
            "endpoint_mae_improvement_candidate_over_larger",
            -endpoint_mae_margin,
            strict=False,
        ),
        "endpoint_flip_noninferior": lower_above(
            "endpoint_flip_improvement_candidate_over_larger",
            -endpoint_flip_margin,
            strict=False,
        ),
        "endpoint_bias_noninferior": lower_above(
            "endpoint_bias_improvement_candidate_over_larger",
            -endpoint_bias_margin,
            strict=False,
        ),
        "endpoint_rank_two_sided_equivalent": ci_inside_symmetric_margin(
            "endpoint_rho_candidate_minus_larger", endpoint_rho_margin),
        "endpoint_mae_two_sided_equivalent": ci_inside_symmetric_margin(
            "endpoint_mae_improvement_candidate_over_larger", endpoint_mae_margin),
        "endpoint_flip_two_sided_equivalent": ci_inside_symmetric_margin(
            "endpoint_flip_improvement_candidate_over_larger", endpoint_flip_margin),
        "endpoint_bias_two_sided_equivalent": ci_inside_symmetric_margin(
            "endpoint_bias_improvement_candidate_over_larger", endpoint_bias_margin),
    }

    native_keys = ("native_rank_gap", "native_mae_gap")
    articulation_keys = ("articulation_rank_gain", "articulation_mae_gain")
    endpoint_primary_keys = ("endpoint_rank_noninferior", "endpoint_mae_noninferior")
    endpoint_vector_keys = endpoint_primary_keys + (
        "endpoint_flip_noninferior", "endpoint_bias_noninferior")
    endpoint_equivalent_primary_keys = (
        "endpoint_rank_two_sided_equivalent", "endpoint_mae_two_sided_equivalent")
    endpoint_equivalent_vector_keys = endpoint_equivalent_primary_keys + (
        "endpoint_flip_two_sided_equivalent", "endpoint_bias_two_sided_equivalent")
    common_keys = (
        "target_identity_valid", "candidate_positive_polarity",
        "larger_sparse_positive_polarity",
    )

    def evidence_grades(gates: Mapping[str, bool]) -> dict[str, bool]:
        native = bool(all(gates[key] for key in native_keys))
        articulation = bool(all(gates[key] for key in articulation_keys))
        endpoint_primary = bool(all(gates[key] for key in endpoint_primary_keys))
        endpoint_vector = bool(all(gates[key] for key in endpoint_vector_keys))
        endpoint_equivalent_primary = bool(
            all(gates[key] for key in endpoint_equivalent_primary_keys))
        endpoint_equivalent_vector = bool(
            all(gates[key] for key in endpoint_equivalent_vector_keys))
        common = bool(all(gates[key] for key in common_keys))
        primary_noninferiority_recovery = bool(
            common and native and articulation and endpoint_primary)
        vector_noninferiority_recovery = bool(
            common and native and articulation and endpoint_vector)
        return {
            "native_scale_gap": native,
            "articulation_gain": articulation,
            "endpoint_noninferior_primary": endpoint_primary,
            "endpoint_noninferior_vector": endpoint_vector,
            "endpoint_two_sided_equivalent_primary": endpoint_equivalent_primary,
            "endpoint_two_sided_equivalent_vector": endpoint_equivalent_vector,
            "local_primary_one_sided_noninferiority_recovery": (
                primary_noninferiority_recovery),
            "local_vector_one_sided_noninferiority_recovery": (
                vector_noninferiority_recovery),
            "local_primary_two_sided_equivalence_recovery": bool(
                common and native and articulation and endpoint_equivalent_primary),
            "local_vector_two_sided_equivalence_recovery": bool(
                common and native and articulation and endpoint_equivalent_vector),
            # Backward-audit aliases.  These keys predate the direct endpoint relation and must
            # never be described as policy isomorphism on their own.
            "local_primary_scale_substitution": primary_noninferiority_recovery,
            "local_vector_scale_substitution": vector_noninferiority_recovery,
        }

    observed = evidence_grades(observed_gates)
    certified = evidence_grades(certified_gates)

    candidate_rho = robust_points["candidate"]["spearman"]
    candidate_rho_ci = _ci(policy_draws["candidate"]["spearman"], confidence)
    small_sparse_rho = robust_points["small_sparse"]["spearman"]
    small_sparse_rho_ci = _ci(
        policy_draws["small_sparse"]["spearman"], confidence)
    candidate_quotient_rho = policy_points["candidate"]["quotient"]["spearman"]
    candidate_quotient_rho_ci = _ci(
        policy_draws["candidate"]["quotient_spearman"], confidence)

    def target_gate_point(name: str, threshold: float, *, upper: bool) -> bool:
        value = target_differences[name]["point"]
        if value is None:
            return False
        return bool(value <= threshold if upper else value >= threshold)

    def target_gate_ci(name: str, threshold: float, *, upper: bool) -> bool:
        interval = target_differences[name]["CI"]
        if not interval:
            return False
        return bool(interval[1] <= threshold if upper else interval[0] >= threshold)

    observed_target_gates = {
        "target_identity_valid": target_valid,
        "candidate_positive_polarity": polarity_valid["candidate"],
        "adverse_rank_at_functional_floor": bool(
            candidate_rho is not None and candidate_rho >= functional_rho_floor),
        "quotient_rank_at_functional_floor": bool(
            candidate_quotient_rho is not None
            and candidate_quotient_rho >= functional_rho_floor),
        "small_sparse_adverse_rank_below_functional_floor": bool(
            small_sparse_rho is not None
            and small_sparse_rho < functional_rho_floor),
        "mae_inside_target_self_band": target_gate_point(
            "mae_excess_over_target_self", endpoint_mae_margin, upper=True),
        "rho_inside_target_self_band": target_gate_point(
            "rho_minus_target_self", -endpoint_rho_margin, upper=False),
        "flip_inside_target_self_band": target_gate_point(
            "flip_excess_over_target_self", endpoint_flip_margin, upper=True),
        "bias_inside_target_self_band": target_gate_point(
            "bias_excess_over_target_self", endpoint_bias_margin, upper=True),
    }
    certified_target_gates = {
        "target_identity_valid": target_valid,
        "candidate_positive_polarity": polarity_valid["candidate"],
        "adverse_rank_at_functional_floor": bool(
            candidate_rho_ci and candidate_rho_ci[0] >= functional_rho_floor),
        "quotient_rank_at_functional_floor": bool(
            candidate_quotient_rho_ci
            and candidate_quotient_rho_ci[0] >= functional_rho_floor),
        "small_sparse_adverse_rank_below_functional_floor": bool(
            small_sparse_rho_ci
            and small_sparse_rho_ci[1] < functional_rho_floor),
        "mae_inside_target_self_band": target_gate_ci(
            "mae_excess_over_target_self", endpoint_mae_margin, upper=True),
        "rho_inside_target_self_band": target_gate_ci(
            "rho_minus_target_self", -endpoint_rho_margin, upper=False),
        "flip_inside_target_self_band": target_gate_ci(
            "flip_excess_over_target_self", endpoint_flip_margin, upper=True),
        "bias_inside_target_self_band": target_gate_ci(
            "bias_excess_over_target_self", endpoint_bias_margin, upper=True),
    }
    functional_keys = (
        "target_identity_valid", "candidate_positive_polarity",
        "adverse_rank_at_functional_floor", "quotient_rank_at_functional_floor",
    )
    # Near-identity is a genuinely nested refinement of the declared functional tier.
    identity_keys = functional_keys + (
        "mae_inside_target_self_band", "rho_inside_target_self_band",
        "flip_inside_target_self_band", "bias_inside_target_self_band",
    )

    def fidelity_grades(gates: Mapping[str, bool]) -> dict[str, bool]:
        return {
            "functional_ordinal": bool(all(gates[key] for key in functional_keys)),
            "target_self_band_near_identity": bool(
                all(gates[key] for key in identity_keys)),
        }

    observed_fidelity = fidelity_grades(observed_target_gates)
    certified_fidelity = fidelity_grades(certified_target_gates)

    direct_candidate = direct_endpoint_point["candidate_robust"]
    direct_small = direct_endpoint_small_point["candidate_robust"]
    direct_self = direct_endpoint_point["target_self_robust"]
    direct_candidate_rho = direct_candidate["spearman"]
    direct_small_rho = direct_small["spearman"]
    direct_candidate_quotient_rho = direct_endpoint_point["quotient"]["spearman"]
    direct_small_quotient_rho = direct_endpoint_small_point["quotient"]["spearman"]
    direct_candidate_rho_ci = _ci(
        direct_endpoint_draws["spearman"], confidence)
    direct_small_rho_ci = _ci(
        direct_endpoint_small_draws["spearman"], confidence)
    direct_candidate_quotient_rho_ci = _ci(
        direct_endpoint_draws["quotient_spearman"], confidence)
    direct_small_quotient_rho_ci = _ci(
        direct_endpoint_small_draws["quotient_spearman"], confidence)

    direct_differences = {
        "mae_excess_over_larger_endpoint_self": {
            "point": float(direct_candidate["mae_tvd"] - direct_self["mae_tvd"]),
            "CI": _ci(
                direct_endpoint_draws["mae_tvd"]
                - direct_endpoint_self_draws["mae_tvd"],
                confidence,
            ),
            "direction": "positive is candidate excess over larger endpoint self",
        },
        "rho_minus_larger_endpoint_self": {
            "point": (
                None if direct_candidate_rho is None or direct_self["spearman"] is None
                else float(direct_candidate_rho - direct_self["spearman"])
            ),
            "CI": _ci(
                direct_endpoint_draws["spearman"]
                - direct_endpoint_self_draws["spearman"],
                confidence,
            ),
            "direction": "positive favors candidate over larger endpoint self",
        },
        "flip_excess_over_larger_endpoint_self": {
            "point": float(
                direct_candidate["binary_flip_rate"]
                - direct_self["binary_flip_rate"]),
            "CI": _ci(
                direct_endpoint_draws["binary_flip_rate"]
                - direct_endpoint_self_draws["binary_flip_rate"],
                confidence,
            ),
            "direction": "positive is candidate excess over larger endpoint self",
        },
        "bias_excess_over_larger_endpoint_self": {
            "point": float(
                direct_candidate["absolute_bias"] - direct_self["absolute_bias"]),
            "CI": _ci(
                direct_endpoint_draws["absolute_bias"]
                - direct_endpoint_self_draws["absolute_bias"],
                confidence,
            ),
            "direction": "positive is candidate excess over larger endpoint self",
        },
        "mae_improvement_over_small_sparse": {
            "point": float(direct_small["mae_tvd"] - direct_candidate["mae_tvd"]),
            "CI": _ci(
                direct_endpoint_small_draws["mae_tvd"]
                - direct_endpoint_draws["mae_tvd"],
                confidence,
            ),
            "direction": "positive favors articulated candidate over small sparse",
        },
        "small_sparse_mae_excess_over_larger_endpoint_self": {
            "point": float(direct_small["mae_tvd"] - direct_self["mae_tvd"]),
            "CI": _ci(
                direct_endpoint_small_draws["mae_tvd"]
                - direct_endpoint_self_draws["mae_tvd"],
                confidence,
            ),
            "direction": "positive is small-sparse excess over larger endpoint self",
        },
        "small_sparse_rho_minus_larger_endpoint_self": {
            "point": (
                None if direct_small_rho is None or direct_self["spearman"] is None
                else float(direct_small_rho - direct_self["spearman"])
            ),
            "CI": _ci(
                direct_endpoint_small_draws["spearman"]
                - direct_endpoint_self_draws["spearman"],
                confidence,
            ),
            "direction": "positive favors small sparse over larger endpoint self",
        },
        "small_sparse_flip_excess_over_larger_endpoint_self": {
            "point": float(
                direct_small["binary_flip_rate"] - direct_self["binary_flip_rate"]),
            "CI": _ci(
                direct_endpoint_small_draws["binary_flip_rate"]
                - direct_endpoint_self_draws["binary_flip_rate"],
                confidence,
            ),
            "direction": "positive is small-sparse excess over larger endpoint self",
        },
        "small_sparse_bias_excess_over_larger_endpoint_self": {
            "point": float(
                direct_small["absolute_bias"] - direct_self["absolute_bias"]),
            "CI": _ci(
                direct_endpoint_small_draws["absolute_bias"]
                - direct_endpoint_self_draws["absolute_bias"],
                confidence,
            ),
            "direction": "positive is small-sparse excess over larger endpoint self",
        },
    }

    direct_self_rho = direct_self["spearman"]
    direct_endpoint_valid = bool(
        direct_endpoint_point["target_information"] >= min_target_information
        and direct_self_rho is not None
        and direct_self_rho >= min_target_self_rho
        and direct_self["mae_tvd"] <= max_target_self_mae
    )

    def direct_point_inside(name: str, threshold: float, *, upper: bool) -> bool:
        value = direct_differences[name]["point"]
        if value is None:
            return False
        return bool(value <= threshold if upper else value >= threshold)

    def direct_ci_inside(name: str, threshold: float, *, upper: bool) -> bool:
        interval = direct_differences[name]["CI"]
        if not interval:
            return False
        return bool(interval[1] <= threshold if upper else interval[0] >= threshold)

    def interval_lower_at_floor(interval: list[float] | None) -> bool:
        return bool(interval and interval[0] >= functional_rho_floor)

    def interval_upper_below_floor(interval: list[float] | None) -> bool:
        return bool(interval and interval[1] < functional_rho_floor)

    def direct_small_point_outside_identity_band() -> bool:
        checks = (
            direct_point_inside(
                "small_sparse_mae_excess_over_larger_endpoint_self",
                endpoint_mae_margin,
                upper=True,
            ),
            direct_point_inside(
                "small_sparse_rho_minus_larger_endpoint_self",
                -endpoint_rho_margin,
                upper=False,
            ),
            direct_point_inside(
                "small_sparse_flip_excess_over_larger_endpoint_self",
                endpoint_flip_margin,
                upper=True,
            ),
            direct_point_inside(
                "small_sparse_bias_excess_over_larger_endpoint_self",
                endpoint_bias_margin,
                upper=True,
            ),
            bool(direct_small["all_positive_polarity"]),
        )
        return not all(checks)

    def direct_small_certified_outside_identity_band() -> bool:
        # Certify the complement of the joint identity region by putting at least one whole
        # confidence interval beyond its permitted edge (or by a deterministic polarity fail).
        error_keys_and_margins = (
            ("small_sparse_mae_excess_over_larger_endpoint_self", endpoint_mae_margin),
            ("small_sparse_flip_excess_over_larger_endpoint_self", endpoint_flip_margin),
            ("small_sparse_bias_excess_over_larger_endpoint_self", endpoint_bias_margin),
        )
        error_outside = any(
            direct_differences[key]["CI"]
            and direct_differences[key]["CI"][0] > margin
            for key, margin in error_keys_and_margins
        )
        rank_interval = direct_differences[
            "small_sparse_rho_minus_larger_endpoint_self"]["CI"]
        rank_outside = bool(
            rank_interval and rank_interval[1] < -endpoint_rho_margin)
        return bool(
            error_outside
            or rank_outside
            or not direct_small["all_positive_polarity"])

    direct_observed_gates = {
        "larger_endpoint_identity_valid": direct_endpoint_valid,
        "candidate_positive_polarity": bool(direct_candidate["all_positive_polarity"]),
        "adverse_rank_at_functional_floor": bool(
            direct_candidate_rho is not None
            and direct_candidate_rho >= functional_rho_floor),
        "quotient_rank_at_functional_floor": bool(
            direct_candidate_quotient_rho is not None
            and direct_candidate_quotient_rho >= functional_rho_floor),
        "direct_mae_improvement_over_small_sparse": bool(
            direct_differences["mae_improvement_over_small_sparse"]["point"] > 0.0),
        "small_sparse_adverse_rank_below_functional_floor": bool(
            direct_small_rho is not None
            and direct_small_rho < functional_rho_floor),
        "small_sparse_outside_functional_region": bool(
            direct_small_rho is not None
            and direct_small_rho < functional_rho_floor),
        "small_sparse_outside_functional_region_any_coordinate_descriptive": bool(
            direct_small_rho is None
            or direct_small_quotient_rho is None
            or direct_small_rho < functional_rho_floor
            or direct_small_quotient_rho < functional_rho_floor),
        "small_sparse_rank_outside_near_identity_region": bool(
            direct_differences[
                "small_sparse_rho_minus_larger_endpoint_self"]["point"] is not None
            and direct_differences[
                "small_sparse_rho_minus_larger_endpoint_self"]["point"]
            < -endpoint_rho_margin),
        "small_sparse_outside_near_identity_region": bool(
            direct_differences[
                "small_sparse_rho_minus_larger_endpoint_self"]["point"] is not None
            and direct_differences[
                "small_sparse_rho_minus_larger_endpoint_self"]["point"]
            < -endpoint_rho_margin),
        "small_sparse_outside_near_identity_region_any_coordinate_descriptive": (
            direct_small_point_outside_identity_band()),
        "mae_inside_larger_endpoint_self_band": direct_point_inside(
            "mae_excess_over_larger_endpoint_self", endpoint_mae_margin, upper=True),
        "rho_inside_larger_endpoint_self_band": direct_point_inside(
            "rho_minus_larger_endpoint_self", -endpoint_rho_margin, upper=False),
        "flip_inside_larger_endpoint_self_band": direct_point_inside(
            "flip_excess_over_larger_endpoint_self", endpoint_flip_margin, upper=True),
        "bias_inside_larger_endpoint_self_band": direct_point_inside(
            "bias_excess_over_larger_endpoint_self", endpoint_bias_margin, upper=True),
    }
    direct_certified_gates = {
        "larger_endpoint_identity_valid": direct_endpoint_valid,
        "candidate_positive_polarity": bool(direct_candidate["all_positive_polarity"]),
        "adverse_rank_at_functional_floor": interval_lower_at_floor(
            direct_candidate_rho_ci),
        "quotient_rank_at_functional_floor": interval_lower_at_floor(
            direct_candidate_quotient_rho_ci),
        "direct_mae_improvement_over_small_sparse": bool(
            direct_differences["mae_improvement_over_small_sparse"]["CI"]
            and direct_differences["mae_improvement_over_small_sparse"]["CI"][0] > 0.0),
        # Predeclare adverse rank as the sole exclusion coordinate.  This avoids an implicit
        # union test over adverse rank, quotient rank, and the four identity-band coordinates.
        "small_sparse_adverse_rank_below_functional_floor": (
            interval_upper_below_floor(direct_small_rho_ci)),
        "small_sparse_outside_functional_region": (
            interval_upper_below_floor(direct_small_rho_ci)),
        "small_sparse_outside_functional_region_any_coordinate_descriptive": bool(
            interval_upper_below_floor(direct_small_rho_ci)
            or interval_upper_below_floor(direct_small_quotient_rho_ci)),
        "small_sparse_rank_outside_near_identity_region": bool(
            direct_differences[
                "small_sparse_rho_minus_larger_endpoint_self"]["CI"]
            and direct_differences[
                "small_sparse_rho_minus_larger_endpoint_self"]["CI"][1]
            < -endpoint_rho_margin),
        "small_sparse_outside_near_identity_region": bool(
            direct_differences[
                "small_sparse_rho_minus_larger_endpoint_self"]["CI"]
            and direct_differences[
                "small_sparse_rho_minus_larger_endpoint_self"]["CI"][1]
            < -endpoint_rho_margin),
        "small_sparse_outside_near_identity_region_any_coordinate_descriptive": (
            direct_small_certified_outside_identity_band()),
        "mae_inside_larger_endpoint_self_band": direct_ci_inside(
            "mae_excess_over_larger_endpoint_self", endpoint_mae_margin, upper=True),
        "rho_inside_larger_endpoint_self_band": direct_ci_inside(
            "rho_minus_larger_endpoint_self", -endpoint_rho_margin, upper=False),
        "flip_inside_larger_endpoint_self_band": direct_ci_inside(
            "flip_excess_over_larger_endpoint_self", endpoint_flip_margin, upper=True),
        "bias_inside_larger_endpoint_self_band": direct_ci_inside(
            "bias_excess_over_larger_endpoint_self", endpoint_bias_margin, upper=True),
    }
    direct_functional_keys = (
        "larger_endpoint_identity_valid", "candidate_positive_polarity",
        "adverse_rank_at_functional_floor", "quotient_rank_at_functional_floor",
    )
    # Direct endpoint near-identity likewise cannot be earned below the functional floor.
    direct_identity_keys = direct_functional_keys + (
        "mae_inside_larger_endpoint_self_band",
        "rho_inside_larger_endpoint_self_band",
        "flip_inside_larger_endpoint_self_band",
        "bias_inside_larger_endpoint_self_band",
    )

    def direct_endpoint_grades(gates: Mapping[str, bool]) -> dict[str, bool]:
        functional = bool(all(gates[key] for key in direct_functional_keys))
        direct_substitution = bool(
            functional
            and gates["direct_mae_improvement_over_small_sparse"]
            and gates["small_sparse_outside_functional_region"]
        )
        near_identity = bool(all(gates[key] for key in direct_identity_keys))
        return {
            "functional_ordinal_fidelity": functional,
            "direct_mae_improvement_over_small_sparse": gates[
                "direct_mae_improvement_over_small_sparse"],
            "small_sparse_outside_functional_region": gates[
                "small_sparse_outside_functional_region"],
            "small_sparse_outside_near_identity_region": gates[
                "small_sparse_outside_near_identity_region"],
            "functional_policy_substitution": direct_substitution,
            "target_self_band_near_identity": near_identity,
            "near_identity_policy_substitution": bool(
                near_identity
                and gates["direct_mae_improvement_over_small_sparse"]
                and gates["small_sparse_outside_near_identity_region"]),
        }

    direct_observed = direct_endpoint_grades(direct_observed_gates)
    direct_certified = direct_endpoint_grades(direct_certified_gates)

    def fidelity_tier(grades: Mapping[str, bool]) -> str:
        if grades["target_self_band_near_identity"]:
            return "target_self_band_near_identity"
        if grades["functional_ordinal"]:
            return "functional_ordinal"
        return "below_functional_ordinal"

    observed.update({
        "functional_target_scale_substitution": bool(
            all(observed_gates[key] for key in common_keys)
            and observed["native_scale_gap"]
            and observed["articulation_gain"]
            and observed_fidelity["functional_ordinal"]
            and observed_target_gates[
                "small_sparse_adverse_rank_below_functional_floor"]),
        "near_identity_target_scale_substitution": bool(
            all(observed_gates[key] for key in common_keys)
            and observed["native_scale_gap"]
            and observed["articulation_gain"]
            and observed_fidelity["target_self_band_near_identity"]
            and observed_target_gates[
                "small_sparse_adverse_rank_below_functional_floor"]),
        "local_functional_endpoint_isomorphic_scale_substitution": bool(
            all(observed_gates[key] for key in common_keys)
            and observed["native_scale_gap"]
            and observed["articulation_gain"]
            and direct_observed["functional_policy_substitution"]),
        "local_functional_isomorphic_scale_substitution": bool(
            all(observed_gates[key] for key in common_keys)
            and observed["native_scale_gap"]
            and observed["articulation_gain"]
            and direct_observed["functional_policy_substitution"]),
        "local_functional_endpoint_equivalent_scale_substitution": bool(
            all(observed_gates[key] for key in common_keys)
            and observed["native_scale_gap"]
            and observed["articulation_gain"]
            and observed["endpoint_two_sided_equivalent_primary"]
            and direct_observed["functional_policy_substitution"]),
        "local_near_identity_isomorphic_scale_substitution": bool(
            all(observed_gates[key] for key in common_keys)
            and observed["native_scale_gap"]
            and observed["articulation_gain"]
            and direct_observed["near_identity_policy_substitution"]),
    })
    certified.update({
        "functional_target_scale_substitution": bool(
            all(certified_gates[key] for key in common_keys)
            and certified["native_scale_gap"]
            and certified["articulation_gain"]
            and certified_fidelity["functional_ordinal"]
            and certified_target_gates[
                "small_sparse_adverse_rank_below_functional_floor"]),
        "near_identity_target_scale_substitution": bool(
            all(certified_gates[key] for key in common_keys)
            and certified["native_scale_gap"]
            and certified["articulation_gain"]
            and certified_fidelity["target_self_band_near_identity"]
            and certified_target_gates[
                "small_sparse_adverse_rank_below_functional_floor"]),
        "local_functional_endpoint_isomorphic_scale_substitution": bool(
            all(certified_gates[key] for key in common_keys)
            and certified["native_scale_gap"]
            and certified["articulation_gain"]
            and direct_certified["functional_policy_substitution"]),
        "local_functional_isomorphic_scale_substitution": bool(
            all(certified_gates[key] for key in common_keys)
            and certified["native_scale_gap"]
            and certified["articulation_gain"]
            and direct_certified["functional_policy_substitution"]),
        "local_functional_endpoint_equivalent_scale_substitution": bool(
            all(certified_gates[key] for key in common_keys)
            and certified["native_scale_gap"]
            and certified["articulation_gain"]
            and certified["endpoint_two_sided_equivalent_primary"]
            and direct_certified["functional_policy_substitution"]),
        "local_near_identity_isomorphic_scale_substitution": bool(
            all(certified_gates[key] for key in common_keys)
            and certified["native_scale_gap"]
            and certified["articulation_gain"]
            and direct_certified["near_identity_policy_substitution"]),
    })

    for grades in (observed, certified):
        grades["joint_fixed_target_and_endpoint_functional_isomorphic_scale_substitution"] = (
            bool(
                grades["functional_target_scale_substitution"]
                and grades["local_functional_endpoint_isomorphic_scale_substitution"]
            )
        )
        grades["joint_fixed_target_and_endpoint_functional_equivalent_scale_substitution"] = (
            bool(
                grades["functional_target_scale_substitution"]
                and grades["local_functional_endpoint_equivalent_scale_substitution"]
            )
        )

    rank_arrays = [
        target_draws["spearman"],
        policy_draws["small_sparse"]["spearman"],
        policy_draws["candidate"]["spearman"],
        policy_draws["larger_sparse"]["spearman"],
        policy_draws["candidate"]["quotient_spearman"],
        direct_endpoint_draws["spearman"],
        direct_endpoint_draws["quotient_spearman"],
        direct_endpoint_small_draws["spearman"],
        direct_endpoint_small_draws["quotient_spearman"],
    ]

    def descriptive_step_closure(articulation_key: str, native_key: str) -> dict:
        articulation_gain = differences[articulation_key]["point"]
        native_gap = differences[native_key]["point"]
        ratio = (
            float(articulation_gain / native_gap)
            if articulation_gain is not None and native_gap is not None and native_gap > 0.0
            else None
        )
        return {
            "articulation_gain": articulation_gain,
            "native_scale_gap": native_gap,
            "chi_articulation_gain_over_native_gap": ratio,
            "defined": ratio is not None,
            "interpretation": (
                "1 closes the point native step; above 1 overshoots; this ratio is descriptive "
                "and never replaces the paired endpoint gates"
            ),
        }

    descriptive_closure = {
        "rank": descriptive_step_closure(
            "articulation_rho_candidate_minus_small",
            "native_rho_larger_minus_small",
        ),
        "mae": descriptive_step_closure(
            "articulation_mae_improvement_candidate_over_small",
            "native_mae_improvement_larger_over_small",
        ),
    }
    return {
        "schema": "scale_step_policy_substitution/v2",
        "estimand": (
            "paired fixed-target replacement of a native sparse/name model-scale step by "
            "explicit articulation supplied to the smaller executor"
        ),
        "n_items": n_items,
        "point": {
            "target_information": target_point["target_information"],
            "target_positive_rate": target_point["target_positive_rate"],
            "target_self_robust": target_robust,
            "small_sparse": robust_points["small_sparse"],
            "candidate": robust_points["candidate"],
            "larger_sparse": robust_points["larger_sparse"],
            "candidate_quotient": policy_points["candidate"]["quotient"],
        },
        "differences": differences,
        "descriptive_step_closure": descriptive_closure,
        "target_relative_endpoint_margins": margins,
        # Backward-audit alias used by v1 readers.  In v2 these margins parameterize explicit
        # one-sided noninferiority and two-sided equivalence gates; they are not by themselves a
        # direct policy-isomorphism criterion.
        "endpoint_margins": margins,
        "minimum_superiority_gains": {
            "rho": min_rho_gain,
            "mae": min_mae_gain,
        },
        "bootstrap": {
            "n": n_boot,
            "n_requested": n_boot,
            "n_joint_rank_valid": _n_jointly_finite(*rank_arrays),
            "rank_draw_counts": {
                "target_self_valid": _n_jointly_finite(target_draws["spearman"]),
                "small_sparse_valid": _n_jointly_finite(
                    policy_draws["small_sparse"]["spearman"]),
                "candidate_valid": _n_jointly_finite(
                    policy_draws["candidate"]["spearman"]),
                "larger_sparse_valid": _n_jointly_finite(
                    policy_draws["larger_sparse"]["spearman"]),
                "candidate_quotient_valid": _n_jointly_finite(
                    policy_draws["candidate"]["quotient_spearman"]),
            },
            "seed": seed,
            "confidence": confidence,
            **bootstrap_design,
            "joint_draw_contract": (
                "one shared paired item/source-group draw indexes target, small sparse/name, "
                "articulated candidate, and larger sparse/name policies; all Spearman ranks "
                "are recomputed after item expansion inside each draw"
            ),
        },
        "evidence": {
            "observed_gates": observed_gates,
            "certified_gates": certified_gates,
            "observed": observed,
            "certified": certified,
            "grade_meaning": {
                "observed": "all named gates hold at the original-panel point estimates",
                "certified": (
                    "all named superiority, one-sided noninferiority, two-sided equivalence, "
                    "and direct endpoint-fidelity gates hold at their corresponding paired-"
                    "bootstrap confidence-interval edges"
                ),
            },
        },
        "target_fidelity": {
            "functional_rho_floor": functional_rho_floor,
            "epsilon_rank_loss": float(1.0 - functional_rho_floor),
            "candidate_adverse_rho_point": candidate_rho,
            "candidate_adverse_rho_CI": candidate_rho_ci,
            "small_sparse_adverse_rho_point": small_sparse_rho,
            "small_sparse_adverse_rho_CI": small_sparse_rho_ci,
            "candidate_quotient_rho_point": candidate_quotient_rho,
            "candidate_quotient_rho_CI": candidate_quotient_rho_ci,
            "differences_from_target_self": target_differences,
            "observed_gates": observed_target_gates,
            "certified_gates": certified_target_gates,
            "observed": {
                **observed_fidelity,
                "tier": fidelity_tier(observed_fidelity),
            },
            "certified": {
                **certified_fidelity,
                "tier": fidelity_tier(certified_fidelity),
            },
        },
        "direct_endpoint_isomorphism": {
            "schema": "direct_larger_sparse_endpoint_isomorphism/v1",
            "estimand": (
                "direct behavioral reconstruction of the larger sparse/name endpoint by the "
                "articulated smaller executor"
            ),
            "functional_rho_floor": functional_rho_floor,
            "epsilon_rank_loss": float(1.0 - functional_rho_floor),
            "point": {
                "larger_endpoint_information": direct_endpoint_point[
                    "target_information"],
                "larger_endpoint_positive_rate": direct_endpoint_point[
                    "target_positive_rate"],
                "larger_endpoint_self_robust": direct_self,
                "candidate": direct_candidate,
                "candidate_quotient": direct_endpoint_point["quotient"],
                "small_sparse": direct_small,
                "small_sparse_quotient": direct_endpoint_small_point["quotient"],
            },
            "rank_intervals": {
                "candidate_adverse_rho_CI": direct_candidate_rho_ci,
                "candidate_quotient_rho_CI": direct_candidate_quotient_rho_ci,
                "small_sparse_adverse_rho_CI": direct_small_rho_ci,
                "small_sparse_quotient_rho_CI": direct_small_quotient_rho_ci,
            },
            "differences": direct_differences,
            "margins": margins,
            "observed_gates": direct_observed_gates,
            "certified_gates": direct_certified_gates,
            "observed": direct_observed,
            "certified": direct_certified,
            "bootstrap": {
                "shared_with_scale_step": True,
                "n": n_boot,
                "n_requested": n_boot,
                "rank_draw_counts": {
                    "larger_endpoint_self_valid": _n_jointly_finite(
                        direct_endpoint_self_draws["spearman"]),
                    "candidate_adverse_valid": _n_jointly_finite(
                        direct_endpoint_draws["spearman"]),
                    "candidate_quotient_valid": _n_jointly_finite(
                        direct_endpoint_draws["quotient_spearman"]),
                    "small_sparse_adverse_valid": _n_jointly_finite(
                        direct_endpoint_small_draws["spearman"]),
                    "small_sparse_quotient_valid": _n_jointly_finite(
                        direct_endpoint_small_draws["quotient_spearman"]),
                    "all_direct_rank_coordinates_jointly_valid": _n_jointly_finite(
                        direct_endpoint_self_draws["spearman"],
                        direct_endpoint_draws["spearman"],
                        direct_endpoint_draws["quotient_spearman"],
                        direct_endpoint_small_draws["spearman"],
                        direct_endpoint_small_draws["quotient_spearman"],
                    ),
                },
                "seed": seed,
                "confidence": confidence,
                **bootstrap_design,
            },
            "claim_boundary": (
                "Functional endpoint isomorphism requires both adverse-form and quotient rank "
                "against the larger sparse endpoint to clear the declared floor. Functional "
                "endpoint substitution additionally requires direct MAE improvement and uses "
                "predeclared adverse rank to establish that the small sparse baseline is outside "
                "the functional region. Near-identity substitution requires the candidate inside "
                "the larger endpoint's self band across MAE, rank, threshold flips, and bias, "
                "while the predeclared baseline-exclusion coordinate is again adverse rank. "
                "Any-coordinate exclusion diagnostics are descriptive only and never enter a "
                "substitution gate. Observed uses point estimates; certified uses the "
                "corresponding confidence-interval edges."
            ),
        },
        "claim_boundary": (
            "The backward-compatible local_primary_scale_substitution and local_vector_"
            "scale_substitution keys are one-sided target-relative noninferiority-recovery "
            "grades, not policy-isomorphism grades: a superior candidate can pass them by "
            "overshooting the larger endpoint. Two-sided endpoint-equivalence grades require "
            "the fixed-target endpoint differences and their intervals to stay within both "
            "margin edges. Direct endpoint isomorphism instead compares the articulated policy "
            "to the larger sparse policy itself. The local functional endpoint-isomorphic grade "
            "uses the accepted direct adverse-plus-quotient rank region; its stricter endpoint-"
            "equivalent grade additionally requires two-sided fixed-target rank-plus-MAE "
            "equivalence. Fixed-target reconstruction and direct endpoint reconstruction remain "
            "separate and are also exposed as explicit joint grades. All scale-substitution "
            "grades require a demonstrated native gap and articulation gain. This single-"
            "candidate interval is nominal unless a surrounding selection procedure supplies "
            "multiplicity control."
        ),
    }


def summarize_isomorphism_fiber(rows: list[dict], arm_specs: Mapping[str, Mapping],
                                arm_orbits: Mapping[str, Mapping[str, Sequence[float]]], *,
                                performance_slack: float = 0.01,
                                distinctness_floor: float = 0.35,
                                functional_pairwise_rho_floor: float = 0.90,
                                near_identity_pairwise_rho_floor: float | None = None,
                                pairwise_rho_sensitivity_floors: Sequence[float] = (
                                    0.70, 0.80, 0.85, 0.90,
                                ),
                                max_selected: int = 5) -> dict:
    """Describe the target preimage and choose diversity only near the best isomorphism point."""
    if near_identity_pairwise_rho_floor is None:
        near_identity_pairwise_rho_floor = functional_pairwise_rho_floor
    named_floors = {
        "functional_pairwise_rho_floor": functional_pairwise_rho_floor,
        "near_identity_pairwise_rho_floor": near_identity_pairwise_rho_floor,
    }
    for name, value in named_floors.items():
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [-1, 1]")
    sensitivity_floors = []
    for value in pairwise_rho_sensitivity_floors:
        numeric = float(value)
        if not np.isfinite(numeric) or not -1.0 <= numeric <= 1.0:
            raise ValueError("pairwise_rho_sensitivity_floors must lie in [-1, 1]")
        sensitivity_floors.append(numeric)
    sensitivity_floors = sorted(set(
        sensitivity_floors + list(named_floors.values())
    ))
    control_provenances = {"wrong_construct_control", "inert_length_control"}

    def is_matched_control(row: Mapping) -> bool:
        spec = arm_specs.get(row.get("arm_id"), {})
        return bool(
            row.get("control_for")
            or spec.get("control_for")
            or row.get("provenance") in control_provenances
            or spec.get("provenance") in control_provenances
        )

    excluded_controls = sorted(
        row.get("arm_id") for row in rows if is_matched_control(row)
    )
    usable = [
        row for row in rows
        if row.get("certificate", {}).get("point") and not is_matched_control(row)
    ]
    if not usable:
        return {
            "n_tested": 0,
            "n_controls_excluded_from_fiber": len(excluded_controls),
            "controls_excluded_from_fiber": excluded_controls,
            "n_isomorphic": 0,
            "members": [],
            "observed_functional_members": [],
            "certified_functional_members": [],
            "observed_functional_component_minimal_members": [],
            "certified_functional_component_minimal_members": [],
            "n_observed_functional_component_minimal_members": 0,
            "n_certified_functional_component_minimal_members": 0,
            "near_frontier": [],
            "selected_diverse": [],
            "equal_but_different_pairs": [],
            "n_equal_but_different_pairs": 0,
            "observed_functional_equal_but_different_pairs": [],
            "certified_functional_equal_but_different_pairs": [],
            "n_observed_functional_equal_but_different_pairs": 0,
            "n_certified_functional_equal_but_different_pairs": 0,
            "functional_pairwise_rho_floor": functional_pairwise_rho_floor,
            "near_identity_pairwise_rho_floor": near_identity_pairwise_rho_floor,
            "pairwise_behavior_gate_grade": "point_only",
            "pairwise_behavior_threshold_sensitivity": [],
            "diversity_pool": [],
            "fiber_status": "no_eligible_arms",
        }
    best_mae = min(row["certificate"]["point"]["candidate_robust"]["mae_tvd"] for row in usable)
    members = [row for row in usable if row["certificate"]["policy_isomorphic"]]
    observed_functional_members = [
        row for row in usable
        if row["certificate"].get("functional", {}).get(
            "observed_functional_policy_substitution")
    ]
    certified_functional_members = [
        row for row in usable
        if row["certificate"].get("functional", {}).get(
            "certified_functional_policy_substitution")
    ]

    def component_minimal(candidate_rows: list[dict]) -> list[str]:
        component_sets = {
            row["arm_id"]: set(arm_specs[row["arm_id"]].get(
                "components", [f"__arm__:{row['arm_id']}"]))
            for row in candidate_rows
        }
        return sorted(
            arm_id for arm_id, components in component_sets.items()
            if not any(
                other_components < components
                for other_id, other_components in component_sets.items()
                if other_id != arm_id
            )
        )

    observed_component_minimal = component_minimal(observed_functional_members)
    certified_component_minimal = component_minimal(certified_functional_members)
    frontier = [row for row in usable
                if row["certificate"]["point"]["candidate_robust"]["mae_tvd"]
                <= best_mae + performance_slack]
    best_member_mae = (
        min(row["certificate"]["point"]["candidate_robust"]["mae_tvd"]
            for row in members)
        if members else None
    )
    member_frontier = [
        row for row in members
        if row["certificate"]["point"]["candidate_robust"]["mae_tvd"]
        <= best_member_mae + performance_slack
    ] if best_member_mae is not None else []
    pool = member_frontier if members else frontier
    diversity_pool_basis = (
        "near-identity members within performance_slack of the best near-identity member"
        if members else
        "eligible arms within performance_slack of the best tested arm; no near-identity member"
    )

    selected = []
    if pool:
        selected = [min(pool, key=lambda row: (
            row["certificate"]["point"]["candidate_robust"]["mae_tvd"], row["arm_id"]))]
        remaining = [row for row in pool if row is not selected[0]]
        while remaining and len(selected) < max_selected:
            choice = max(remaining, key=lambda row: (
                min(articulation_distance(arm_specs[row["arm_id"]],
                                          arm_specs[chosen["arm_id"]])
                    for chosen in selected),
                -row["certificate"]["point"]["candidate_robust"]["mae_tvd"],
                row["arm_id"],
            ))
            selected.append(choice)
            remaining.remove(choice)

    def distinct_pair_candidates(candidate_rows: list[dict]) -> list[dict]:
        result = []
        for left, right in itertools.combinations(
                sorted(candidate_rows, key=lambda row: row["arm_id"]), 2):
            distance = articulation_distance(
                arm_specs[left["arm_id"]], arm_specs[right["arm_id"]]
            )
            if distance < distinctness_floor:
                continue
            behavior = pairwise_policy_fidelity(
                arm_orbits[left["arm_id"]], arm_orbits[right["arm_id"]]
            )
            result.append({
                "left": left["arm_id"], "right": right["arm_id"],
                "articulation_surface_distance": distance,
                "behavior": behavior,
                "pairwise_gate_grade": "point_only",
            })
        return result

    def gate_pair_candidates(pair_candidates: list[dict], rho_floor: float) -> list[dict]:
        return [
            {
                **pair,
                "behavior_rho_floor": rho_floor,
                "pairwise_behavior_gate_pass": True,
            }
            for pair in pair_candidates
            if pair["behavior"]["quotient_spearman"] is not None
            and pair["behavior"]["quotient_spearman"] >= rho_floor
        ]

    near_identity_pair_candidates = distinct_pair_candidates(members)
    observed_pair_candidates = distinct_pair_candidates(observed_functional_members)
    certified_pair_candidates = distinct_pair_candidates(certified_functional_members)
    equal_different = gate_pair_candidates(
        near_identity_pair_candidates, near_identity_pairwise_rho_floor
    )
    observed_functional_pairs = gate_pair_candidates(
        observed_pair_candidates, functional_pairwise_rho_floor
    )
    certified_functional_pairs = gate_pair_candidates(
        certified_pair_candidates, functional_pairwise_rho_floor
    )

    def compact_pairs(pairs: list[dict]) -> list[dict]:
        return [
            {
                "left": pair["left"],
                "right": pair["right"],
                "quotient_spearman": pair["behavior"]["quotient_spearman"],
            }
            for pair in pairs
        ]

    threshold_sensitivity = []
    for rho_floor in sensitivity_floors:
        near_pairs = gate_pair_candidates(near_identity_pair_candidates, rho_floor)
        observed_pairs = gate_pair_candidates(observed_pair_candidates, rho_floor)
        certified_pairs = gate_pair_candidates(certified_pair_candidates, rho_floor)
        threshold_sensitivity.append({
            "rho_floor": rho_floor,
            "pairwise_gate_grade": "point_only",
            "near_identity": {
                "n_pairs": len(near_pairs), "pairs": compact_pairs(near_pairs),
            },
            "observed_functional": {
                "n_pairs": len(observed_pairs), "pairs": compact_pairs(observed_pairs),
            },
            "certified_functional": {
                "n_pairs": len(certified_pairs), "pairs": compact_pairs(certified_pairs),
            },
        })
    return {
        "n_tested": len(usable), "best_adverse_mae_tvd": best_mae,
        "n_controls_excluded_from_fiber": len(excluded_controls),
        "controls_excluded_from_fiber": excluded_controls,
        "best_near_identity_member_adverse_mae_tvd": best_member_mae,
        "performance_slack": performance_slack,
        "fiber_status": "certified_members" if members else "near_frontier_only",
        "n_isomorphic": len(members), "members": [row["arm_id"] for row in members],
        "observed_functional_members": [
            row["arm_id"] for row in observed_functional_members],
        "certified_functional_members": [
            row["arm_id"] for row in certified_functional_members],
        "observed_functional_component_minimal_members": observed_component_minimal,
        "certified_functional_component_minimal_members": certified_component_minimal,
        "n_observed_functional_component_minimal_members": len(
            observed_component_minimal),
        "n_certified_functional_component_minimal_members": len(
            certified_component_minimal),
        "near_frontier": [row["arm_id"] for row in frontier],
        "diversity_pool": [row["arm_id"] for row in pool],
        "diversity_pool_basis": diversity_pool_basis,
        "selected_diverse": [row["arm_id"] for row in selected],
        "equal_but_different_pairs": equal_different,
        "n_equal_but_different_pairs": len(equal_different),
        "observed_functional_equal_but_different_pairs": observed_functional_pairs,
        "certified_functional_equal_but_different_pairs": certified_functional_pairs,
        "n_observed_functional_equal_but_different_pairs": len(observed_functional_pairs),
        "n_certified_functional_equal_but_different_pairs": len(certified_functional_pairs),
        "functional_pairwise_rho_floor": functional_pairwise_rho_floor,
        "near_identity_pairwise_rho_floor": near_identity_pairwise_rho_floor,
        "pairwise_behavior_gate_grade": "point_only",
        "pairwise_behavior_threshold_sensitivity": threshold_sensitivity,
        "pairwise_claim_boundary": (
            "Member status is evaluated against the target. Mutual pairwise behavioral "
            "fidelity is separately gated on point Spearman only; even pairs whose members "
            "are individually certified do not have an interval-certified mutual gate."
        ),
        "distinctness_scope": ("lexical/channel/provenance surface distance; independent semantic "
                                "fidelity certification remains required"),
        "selection_priority": "isomorphism first; diversity only within the best-performance band",
    }
