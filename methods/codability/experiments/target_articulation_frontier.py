#!/usr/bin/env python
"""Target-indexed articulation frontiers and held-out substitution tests.

This is the target/view-agnostic measurement layer described in
``notes/2026-07-12__target-indexed-articulation-frontier-and-duality.md``.  It preserves the
fixed-target DPI quantity from :mod:`methods.metric_implementer.vinfo`, but does not mistake
unsigned mutual information for verdict isomorphism: an inverted candidate has the same MI as an
aligned one.  Recovery is therefore oriented by the target--candidate covariance and accompanied
by a direct Spearman signature gate.

The module knows nothing about prompt text or model APIs.  Callers provide a frozen target vector
and a mapping from articulation-form identifiers to candidate vectors on the same held-out items.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from methods.codability.grid_auc_report import _rank, spearman
from methods.metric_implementer.vinfo import fixed_target_channel_certificate


SCHEMA = "target_articulation_frontier/v1"
MANIFEST_PATH = Path(__file__).with_name("target_articulation_manifest_v1.json")
SCORE_KEY = "oriented_recovery_fraction"


def manifest_sha256() -> str:
    """Return the exact manifest hash recorded in every analysis artifact."""
    return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def validate_target_spec(spec: Mapping) -> dict:
    """Validate the provenance fields that keep target views scientifically distinct."""
    manifest = load_manifest()
    missing = [field for field in manifest["required_target_fields"] if field not in spec]
    if missing:
        raise ValueError(f"target spec missing required fields: {missing}")
    if spec["target_view"] not in manifest["target_views"]:
        raise ValueError(f"unknown target view {spec['target_view']!r}")
    return dict(spec)


def _vector(value: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.size < 4:
        raise ValueError(f"{name} must be a one-dimensional vector with at least four items")
    if not np.isfinite(arr).all() or not ((0.0 <= arr).all() and (arr <= 1.0).all()):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return arr


def validate_orbit(orbit: Mapping[str, Sequence[float]], *, n_items: int | None = None,
                   name: str = "candidate_orbit") -> dict[str, np.ndarray]:
    if not orbit:
        raise ValueError(f"{name} must contain at least one form")
    out = {str(form): _vector(values, f"{name}[{form!r}]")
           for form, values in orbit.items()}
    lengths = {len(v) for v in out.values()}
    if len(lengths) != 1 or (n_items is not None and lengths != {n_items}):
        raise ValueError(f"{name} forms are not aligned to one item set")
    return out


def target_orbit_mean(target_orbit: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Form-quotient target used by the adverse-form candidate readout."""
    orbit = validate_orbit(target_orbit, name="target_orbit")
    return np.mean(np.stack(list(orbit.values()), axis=0), axis=0)


def recovery_point(target: Sequence[float], candidate: Sequence[float], *,
                   divergence: str = "tvd", min_target_information: float = 1e-6) -> dict:
    """One fixed-target recovery point with polarity and direct-fidelity diagnostics.

    ``recovery_fraction`` is the ordinary non-negative ``R/T`` DPI fraction.  The primary
    ``oriented_recovery_fraction`` multiplies it by the covariance sign, so a policy inversion is
    negative instead of looking perfectly recovered.  This orientation is a measurement gate,
    not a new information-theoretic bound.
    """
    if divergence not in {"tvd", "shannon"}:
        raise ValueError("divergence must be 'tvd' or 'shannon'")
    q, p = _vector(target, "target"), _vector(candidate, "candidate")
    if q.shape != p.shape:
        raise ValueError("target and candidate must be aligned")
    cert = fixed_target_channel_certificate(q, p)
    if not cert.get("valid"):
        return {"valid": False, "error": cert.get("error", "invalid certificate")}
    row = cert[divergence]
    target_information = float(row["T_target"])
    if target_information < min_target_information:
        return {
            "valid": False,
            "error": "target_information_below_floor",
            "target_information": target_information,
            "min_target_information": float(min_target_information),
        }
    covariance = float(np.mean(q * p) - np.mean(q) * np.mean(p))
    orientation = 1 if covariance > 1e-15 else (-1 if covariance < -1e-15 else 0)
    recovery = float(row["R"])
    fraction = float(np.clip(recovery / target_information, 0.0, 1.0))
    rho = spearman(q, p)
    return {
        "valid": True,
        "divergence": divergence,
        "R": recovery,
        "T_target": target_information,
        "T_candidate": float(row["T_candidate"]),
        "dpi_upper": float(row["dpi_upper"]),
        "dpi_ok": bool(row["dpi_ok"]),
        "recovery_fraction": fraction,
        SCORE_KEY: float(orientation * fraction),
        "covariance": covariance,
        "polarity": orientation,
        "positive_polarity": bool(orientation > 0),
        "spearman": None if rho is None or not np.isfinite(rho) else float(rho),
        "mean_absolute_error": float(np.mean(np.abs(q - p))),
        "mean_candidate_minus_target": float(np.mean(p - q)),
        "n_items": int(q.size),
        "scope": cert["scope"],
    }


def orbit_recovery(target: Sequence[float], candidate_orbit: Mapping[str, Sequence[float]], *,
                   divergence: str = "tvd", min_target_information: float = 1e-6) -> dict:
    """Evaluate all articulation forms and take the adverse (worst) form."""
    q = _vector(target, "target")
    orbit = validate_orbit(candidate_orbit, n_items=len(q))
    forms = {form: recovery_point(q, values, divergence=divergence,
                                  min_target_information=min_target_information)
             for form, values in sorted(orbit.items())}
    invalid = [form for form, row in forms.items() if not row.get("valid")]
    if invalid:
        return {"valid": False, "error": "invalid_forms", "invalid_forms": invalid,
                "forms": forms}
    scores = [row[SCORE_KEY] for row in forms.values()]
    rhos = [row["spearman"] for row in forms.values()]
    robust_rho = None if any(v is None for v in rhos) else float(min(rhos))
    return {
        "valid": True,
        "forms": forms,
        "robust": {
            SCORE_KEY: float(min(scores)),
            "recovery_fraction_unsigned": float(min(row["recovery_fraction"]
                                                     for row in forms.values())),
            "spearman": robust_rho,
            "mean_absolute_error": float(max(row["mean_absolute_error"]
                                              for row in forms.values())),
            "all_positive_polarity": bool(all(row["positive_polarity"]
                                                for row in forms.values())),
            "adverse_form": min(forms, key=lambda form: forms[form][SCORE_KEY]),
            "n_forms": len(forms),
        },
        "aggregation": "adverse_candidate_form_against_form_quotient_target",
    }


def _percentile_interval(values: np.ndarray, confidence: float = 0.95) -> list[float] | None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))]


def bootstrap_orbit_values(target: np.ndarray, candidate_orbit: dict[str, np.ndarray],
                           samples: np.ndarray, *, divergence: str,
                           min_target_information: float) -> dict[str, np.ndarray]:
    """Raw paired bootstrap draws for callers assembling multi-reader surfaces.

    ``samples`` is supplied by the caller so every reader/arm on a fixed target can use exactly the
    same resampled item indices. Invalid target-information resamples are omitted identically for
    every aligned arm.
    """
    q = _vector(target, "target")
    orbit = validate_orbit(candidate_orbit, n_items=len(q))
    sample_idx = np.asarray(samples, int)
    if sample_idx.ndim != 2 or sample_idx.shape[1] == 0:
        raise ValueError("samples must have shape (n_boot, positive sample size)")
    if sample_idx.size and (sample_idx.min() < 0 or sample_idx.max() >= len(q)):
        raise ValueError("bootstrap sample index is out of range")

    qb = q[sample_idx]
    q_mean = np.mean(qb, axis=1)
    if divergence == "tvd":
        target_information = np.mean(np.abs(qb - q_mean[:, None]), axis=1)
    elif divergence == "shannon":
        def h(value):
            value = np.asarray(value, float)
            out = np.zeros_like(value)
            keep = (value > 1e-15) & (value < 1.0 - 1e-15)
            out[keep] = -(value[keep] * np.log2(value[keep])
                          + (1.0 - value[keep]) * np.log2(1.0 - value[keep]))
            return out
        target_information = h(q_mean) - np.mean(h(qb), axis=1)
    else:
        raise ValueError("divergence must be 'tvd' or 'shannon'")
    valid = target_information >= min_target_information

    q_rank = _rank(q)
    qrb = q_rank[sample_idx]
    qrc = qrb - np.mean(qrb, axis=1, keepdims=True)
    qss = np.sum(qrc ** 2, axis=1)
    form_scores, form_rhos, form_maes = [], [], []
    for candidate in orbit.values():
        pb = candidate[sample_idx]
        p_mean = np.mean(pb, axis=1)
        covariance = np.mean(qb * pb, axis=1) - q_mean * p_mean
        if divergence == "tvd":
            recovery = 2.0 * np.abs(covariance)
        else:
            joint = np.stack([
                np.mean((1.0 - qb) * (1.0 - pb), axis=1),
                np.mean((1.0 - qb) * pb, axis=1),
                np.mean(qb * (1.0 - pb), axis=1),
                np.mean(qb * pb, axis=1),
            ], axis=1).reshape(-1, 2, 2)
            row = joint.sum(axis=2)
            col = joint.sum(axis=1)
            prod = row[:, :, None] * col[:, None, :]
            term = np.zeros_like(joint)
            nonzero = joint > 0
            term[nonzero] = joint[nonzero] * np.log2(joint[nonzero] / prod[nonzero])
            recovery = np.sum(term, axis=(1, 2))
        fraction = np.divide(recovery, target_information,
                             out=np.full_like(recovery, np.nan), where=valid)
        fraction = np.clip(fraction, 0.0, 1.0)
        form_scores.append(np.sign(covariance) * fraction)

        # Point Spearman remains the exact midrank statistic. For uncertainty, rank once on the
        # held-out panel and bootstrap the paired rank scores. This avoids reranking hundreds of
        # duplicate-heavy resamples and is the declared surface-bootstrap estimator.
        p_rank = _rank(candidate)
        prb = p_rank[sample_idx]
        prc = prb - np.mean(prb, axis=1, keepdims=True)
        denom = np.sqrt(qss * np.sum(prc ** 2, axis=1))
        rho = np.divide(np.sum(qrc * prc, axis=1), denom,
                        out=np.full(len(samples), np.nan), where=denom > 0)
        form_rhos.append(rho)
        form_maes.append(np.mean(np.abs(qb - pb), axis=1))

    score_matrix = np.stack(form_scores)
    rho_matrix = np.stack(form_rhos)
    mae_matrix = np.stack(form_maes)
    robust_score = np.min(score_matrix, axis=0)
    robust_rho = np.min(rho_matrix, axis=0)
    robust_rho[~np.all(np.isfinite(rho_matrix), axis=0)] = np.nan
    robust_mae = np.max(mae_matrix, axis=0)
    return {SCORE_KEY: robust_score[valid], "spearman": robust_rho[valid],
            "mean_absolute_error": robust_mae[valid]}


def direct_orbit_fidelity(reference_orbit: Mapping[str, Sequence[float]],
                          candidate_orbit: Mapping[str, Sequence[float]]) -> dict:
    """Direct cross-reader signature fidelity under an adverse form-pair quotient.

    Equal recovery against a third target does not imply that two readers implement the same
    item-level policy.  We therefore compare their score signatures directly.  All form pairs are
    included because legacy arms do not always expose matching form identifiers; taking the worst
    pair is conservative and makes the quotient explicit.
    """
    reference = validate_orbit(reference_orbit, name="reference_orbit")
    n_items = len(next(iter(reference.values())))
    candidate = validate_orbit(candidate_orbit, n_items=n_items, name="candidate_orbit")
    pairs = {}
    for reference_form, reference_values in sorted(reference.items()):
        for candidate_form, candidate_values in sorted(candidate.items()):
            rho = spearman(reference_values, candidate_values)
            pairs[f"{reference_form}::{candidate_form}"] = {
                "spearman": None if rho is None or not np.isfinite(rho) else float(rho),
                "mean_absolute_error": float(np.mean(np.abs(reference_values - candidate_values))),
            }
    rhos = [row["spearman"] for row in pairs.values()]
    return {
        "valid": bool(pairs) and all(value is not None for value in rhos),
        "pairs": pairs,
        "robust": {
            "spearman": (float(min(rhos)) if pairs and all(value is not None for value in rhos)
                         else None),
            "mean_absolute_error": (float(max(row["mean_absolute_error"]
                                                for row in pairs.values())) if pairs else None),
            "adverse_spearman_pair": (min(pairs, key=lambda key: pairs[key]["spearman"])
                                      if pairs and all(value is not None for value in rhos) else None),
            "n_form_pairs": len(pairs),
        },
        "aggregation": "adverse_pair_over_reference_form_x_candidate_form",
    }


def bootstrap_direct_orbit_fidelity(reference_orbit: Mapping[str, Sequence[float]],
                                    candidate_orbit: Mapping[str, Sequence[float]],
                                    samples: np.ndarray) -> dict[str, np.ndarray]:
    """Paired bootstrap draws for :func:`direct_orbit_fidelity`."""
    reference = validate_orbit(reference_orbit, name="reference_orbit")
    n_items = len(next(iter(reference.values())))
    candidate = validate_orbit(candidate_orbit, n_items=n_items, name="candidate_orbit")
    sample_idx = np.asarray(samples, int)
    if sample_idx.ndim != 2 or sample_idx.shape[1] == 0:
        raise ValueError("samples must have shape (n_boot, positive sample size)")
    if sample_idx.size and (sample_idx.min() < 0 or sample_idx.max() >= n_items):
        raise ValueError("bootstrap sample index is out of range")

    pair_rhos, pair_maes = [], []
    for reference_values in reference.values():
        reference_rank = _rank(reference_values)[sample_idx]
        reference_centered = reference_rank - np.mean(reference_rank, axis=1, keepdims=True)
        reference_ss = np.sum(reference_centered ** 2, axis=1)
        for candidate_values in candidate.values():
            candidate_rank = _rank(candidate_values)[sample_idx]
            candidate_centered = candidate_rank - np.mean(candidate_rank, axis=1, keepdims=True)
            denom = np.sqrt(reference_ss * np.sum(candidate_centered ** 2, axis=1))
            rho = np.divide(np.sum(reference_centered * candidate_centered, axis=1), denom,
                            out=np.full(len(sample_idx), np.nan), where=denom > 0)
            pair_rhos.append(rho)
            pair_maes.append(np.mean(np.abs(reference_values[sample_idx]
                                             - candidate_values[sample_idx]), axis=1))
    rho_matrix = np.stack(pair_rhos)
    robust_rho = np.min(rho_matrix, axis=0)
    robust_rho[~np.all(np.isfinite(rho_matrix), axis=0)] = np.nan
    return {"spearman": robust_rho, "mean_absolute_error": np.max(pair_maes, axis=0)}


def bootstrap_orbit_recovery(target: Sequence[float],
                             candidate_orbit: Mapping[str, Sequence[float]], *,
                             divergence: str = "tvd", min_target_information: float = 1e-6,
                             n_boot: int = 1000, seed: int = 0,
                             confidence: float = 0.95) -> dict:
    """IID-item bootstrap uncertainty for a frozen target and candidate orbit.

    This is an uncertainty interval on the declared empirical process, not an all-prompt bound and
    not a correction for finite stochastic passes used to estimate either channel.
    """
    q = _vector(target, "target")
    orbit = validate_orbit(candidate_orbit, n_items=len(q))
    point = orbit_recovery(q, orbit, divergence=divergence,
                           min_target_information=min_target_information)
    if n_boot <= 0 or not point.get("valid"):
        return {"point": point, "bootstrap": None}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(q), size=(n_boot, len(q)))
    draws = bootstrap_orbit_values(q, orbit, samples, divergence=divergence,
                                   min_target_information=min_target_information)
    return {
        "point": point,
        "bootstrap": {
            "n_requested": int(n_boot),
            "n_valid": int(len(draws[SCORE_KEY])),
            "confidence": float(confidence),
            "CI": {key: _percentile_interval(value, confidence)
                   for key, value in draws.items()},
            "scope": "iid item bootstrap; fixed target and candidate forms",
            "rank_bootstrap": "paired resampling of held-out midrank scores",
        },
    }


def _difference_report(point: float, draws: np.ndarray, confidence: float) -> dict:
    return {"point": float(point), "CI": _percentile_interval(draws, confidence)}


def paired_substitution_test(target: Sequence[float], *,
                             small_sparse_orbit: Mapping[str, Sequence[float]],
                             big_sparse_orbit: Mapping[str, Sequence[float]],
                             articulated_orbit: Mapping[str, Sequence[float]],
                             control_orbit: Mapping[str, Sequence[float]] | None = None,
                             divergence: str = "tvd", min_target_information: float = 1e-6,
                             gap_delta: float = 0.02, equivalence_delta: float = 0.02,
                             min_signature_rho: float = 0.5,
                             signature_equivalence_delta: float = 0.05,
                             n_boot: int = 1000, seed: int = 0,
                             confidence: float = 0.95) -> dict:
    """Strong baseline-gap-gated substitution decision on one held-out item set."""
    q = _vector(target, "target")
    arms = {
        "small_sparse": validate_orbit(small_sparse_orbit, n_items=len(q),
                                       name="small_sparse_orbit"),
        "big_sparse": validate_orbit(big_sparse_orbit, n_items=len(q),
                                     name="big_sparse_orbit"),
        "articulated": validate_orbit(articulated_orbit, n_items=len(q),
                                      name="articulated_orbit"),
    }
    if control_orbit is not None:
        arms["control"] = validate_orbit(control_orbit, n_items=len(q),
                                         name="control_orbit")
    points = {name: orbit_recovery(q, orbit, divergence=divergence,
                                   min_target_information=min_target_information)
              for name, orbit in arms.items()}
    if not all(row.get("valid") for row in points.values()):
        return {"valid": False, "error": "invalid_arm", "arms": points}

    rng = np.random.default_rng(seed)
    samples = (rng.integers(0, len(q), size=(n_boot, len(q))) if n_boot > 0
               else np.empty((0, len(q)), int))
    draws = {name: bootstrap_orbit_values(q, orbit, samples, divergence=divergence,
                                          min_target_information=min_target_information)
             for name, orbit in arms.items()}
    n_valid = min((len(row[SCORE_KEY]) for row in draws.values()), default=0)
    if n_valid:
        # Invalid resamples are extraordinarily rare after the target-information gate. Truncate
        # jointly so every reported difference remains paired.
        draws = {name: {key: value[:n_valid] for key, value in row.items()}
                 for name, row in draws.items()}

    direct_points = {
        "small_sparse_to_big_sparse": direct_orbit_fidelity(
            arms["big_sparse"], arms["small_sparse"]),
        "articulated_to_big_sparse": direct_orbit_fidelity(
            arms["big_sparse"], arms["articulated"]),
    }
    direct_draws = {
        "small_sparse_to_big_sparse": bootstrap_direct_orbit_fidelity(
            arms["big_sparse"], arms["small_sparse"], samples),
        "articulated_to_big_sparse": bootstrap_direct_orbit_fidelity(
            arms["big_sparse"], arms["articulated"], samples),
    }

    robust = {name: row["robust"] for name, row in points.items()}
    score = {name: row[SCORE_KEY] for name, row in robust.items()}
    rho = {name: row["spearman"] for name, row in robust.items()}

    def diff(left: str, right: str, key: str, point_values: dict[str, float | None]) -> dict:
        left_point, right_point = point_values[left], point_values[right]
        if left_point is None or right_point is None:
            return {"point": None, "CI": None}
        d = (draws[left][key] - draws[right][key]) if n_valid else np.asarray([])
        return _difference_report(float(left_point - right_point), d, confidence)

    baseline_gap = diff("big_sparse", "small_sparse", SCORE_KEY, score)
    improvement = diff("articulated", "small_sparse", SCORE_KEY, score)
    match_big = diff("articulated", "big_sparse", SCORE_KEY, score)
    rho_improvement = diff("articulated", "small_sparse", "spearman", rho)
    rho_match_big = diff("articulated", "big_sparse", "spearman", rho)
    specificity = (diff("articulated", "control", SCORE_KEY, score)
                   if "control" in arms else None)
    direct_articulated_rho = direct_points["articulated_to_big_sparse"]["robust"]["spearman"]
    direct_small_rho = direct_points["small_sparse_to_big_sparse"]["robust"]["spearman"]
    direct_articulated_rho_ci = _percentile_interval(
        direct_draws["articulated_to_big_sparse"]["spearman"], confidence)
    direct_gain_draws = (direct_draws["articulated_to_big_sparse"]["spearman"]
                         - direct_draws["small_sparse_to_big_sparse"]["spearman"])
    direct_signature_gain = {
        "point": (None if direct_articulated_rho is None or direct_small_rho is None else
                  float(direct_articulated_rho - direct_small_rho)),
        "CI": _percentile_interval(direct_gain_draws, confidence),
    }

    gap_ci, gain_ci, match_ci = (baseline_gap["CI"], improvement["CI"], match_big["CI"])
    rho_ci = (_percentile_interval(draws["articulated"]["spearman"], confidence)
              if n_valid else None)
    rho_gain_ci, rho_match_ci = rho_improvement["CI"], rho_match_big["CI"]
    baseline_confirmed = bool(gap_ci and gap_ci[0] > gap_delta)
    improvement_confirmed = bool(gain_ci and gain_ci[0] > 0.0)
    noninferior = bool(match_ci and match_ci[0] >= -equivalence_delta)
    equivalent = bool(match_ci and match_ci[0] >= -equivalence_delta
                      and match_ci[1] <= equivalence_delta)
    polarity_gate = bool(robust["articulated"]["all_positive_polarity"])
    signature_gate = bool(rho_ci and rho_ci[0] >= min_signature_rho)
    signature_improved = bool(rho_gain_ci and rho_gain_ci[0] > 0.0)
    signature_noninferior = bool(rho_match_ci and
                                 rho_match_ci[0] >= -signature_equivalence_delta)
    direct_signature_gate = bool(direct_articulated_rho_ci
                                 and direct_articulated_rho_ci[0] >= min_signature_rho)
    direct_signature_improved = bool(direct_signature_gain["CI"]
                                     and direct_signature_gain["CI"][0] > 0.0)
    specificity_confirmed = bool(specificity and specificity["CI"]
                                 and specificity["CI"][0] > 0.0)
    methodological = bool(baseline_confirmed and improvement_confirmed and noninferior
                          and polarity_gate and signature_gate and direct_signature_gate
                          and direct_signature_improved)
    return {
        "valid": True,
        "schema": "target_articulation_substitution_test/v2_direct_signature",
        "arms": points,
        "bootstrap": {"n_requested": int(n_boot), "n_paired_valid": int(n_valid),
                      "confidence": float(confidence),
                      "scope": "iid held-out item bootstrap; candidate frozen before evaluation",
                      "rank_bootstrap": "paired resampling of held-out midrank scores"},
        "baseline_gap_big_minus_small": baseline_gap,
        "articulation_gain_over_small": improvement,
        "articulated_minus_big": match_big,
        "signature_gain_over_small": rho_improvement,
        "signature_articulated_minus_big": rho_match_big,
        "direct_signature": direct_points,
        "direct_articulated_to_big_signature_CI": direct_articulated_rho_ci,
        "direct_signature_gain_over_small": direct_signature_gain,
        "articulation_minus_control": specificity,
        "articulated_signature_CI": rho_ci,
        "gates": {
            "baseline_gap_confirmed": baseline_confirmed,
            "articulation_improvement_confirmed": improvement_confirmed,
            "noninferior_to_big_sparse": noninferior,
            "equivalent_to_big_sparse": equivalent,
            "positive_polarity": polarity_gate,
            "signature_floor": signature_gate,
            "signature_improved": signature_improved,
            "signature_noninferior_to_big": signature_noninferior,
            "direct_signature_floor": direct_signature_gate,
            "direct_signature_improved": direct_signature_improved,
            "articulation_specificity": (specificity_confirmed
                                          if specificity is not None else None),
        },
        "methodological_substitution": methodological,
        "equivalent_methodological_substitution": bool(methodological and equivalent),
        "articulation_specific_substitution": (bool(methodological and specificity_confirmed)
                                                if specificity is not None else None),
        "paper_grade_substitution": False,
        "claim_note": ("Matched-control specificity was evaluated." if specificity is not None
                       else "A matched inert or wrong-construct control is required before "
                            "upgrading methodological substitution to articulation-specific "
                            "substitution."),
    }


def dose_record(candidate_id: str, channel: str, *, word_count: int | None = None,
                certified_unit_count: float | None = None,
                interaction_degree: int | None = None, scalar_cost: float | None = None,
                cost_basis: str = "unspecified") -> dict:
    manifest = load_manifest()
    if channel not in manifest["articulation_channels"]:
        raise ValueError(f"unknown articulation channel {channel!r}")
    for value, label in ((word_count, "word_count"),
                         (certified_unit_count, "certified_unit_count"),
                         (interaction_degree, "interaction_degree"),
                         (scalar_cost, "scalar_cost")):
        if value is not None and value < 0:
            raise ValueError(f"{label} cannot be negative")
    return {"candidate_id": str(candidate_id), "channel": channel,
            "word_count": word_count, "certified_unit_count": certified_unit_count,
            "interaction_degree": interaction_degree, "scalar_cost": scalar_cost,
            "cost_basis": cost_basis}


def monotone_frontier(candidates: Iterable[dict], *, score_key: str = SCORE_KEY) -> dict:
    """Free-disposal envelope over precomputed candidate records with declared scalar costs."""
    usable = []
    for row in candidates:
        dose = row.get("dose", {})
        recovery = row.get("recovery", {})
        robust = recovery.get("robust", {}) if recovery.get("valid") else {}
        cost, score = dose.get("scalar_cost"), robust.get(score_key)
        if cost is not None and score is not None and np.isfinite(cost) and np.isfinite(score):
            usable.append(row)
    usable.sort(key=lambda row: (float(row["dose"]["scalar_cost"]),
                                 str(row["dose"]["candidate_id"])))
    points, incumbent = [], None
    for row in usable:
        score = float(row["recovery"]["robust"][score_key])
        if incumbent is None or score > incumbent[0]:
            incumbent = (score, row)
        points.append({
            "cost": float(row["dose"]["scalar_cost"]),
            "observed_candidate": row["dose"]["candidate_id"],
            "observed_score": score,
            "frontier_candidate": incumbent[1]["dose"]["candidate_id"],
            "frontier_score": float(incumbent[0]),
        })
    bases = sorted({row["dose"].get("cost_basis") for row in usable})
    return {"schema": SCHEMA, "score": score_key, "cost_bases": bases,
            "points": points, "n_candidates": len(usable),
            "monotonicity": "free-disposal envelope; raw prompt performance may be non-monotone"}


def select_minimal_cost(candidates: Iterable[dict], *, target_score: float,
                        min_signature_rho: float = 0.5,
                        score_key: str = SCORE_KEY) -> dict:
    """Development-only selection; choose the cheapest candidate reaching the target."""
    usable = []
    for row in candidates:
        recovery = row.get("recovery", {})
        robust, dose = recovery.get("robust", {}), row.get("dose", {})
        cost, score, rho = dose.get("scalar_cost"), robust.get(score_key), robust.get("spearman")
        if (recovery.get("valid") and cost is not None and score is not None and rho is not None
                and robust.get("all_positive_polarity") and np.isfinite(cost)
                and np.isfinite(score) and np.isfinite(rho)):
            usable.append(row)
    if not usable:
        raise ValueError("no valid articulation candidates")
    attained = [row for row in usable
                if row["recovery"]["robust"][score_key] >= target_score
                and row["recovery"]["robust"]["spearman"] >= min_signature_rho]
    if attained:
        chosen = min(attained, key=lambda row: (float(row["dose"]["scalar_cost"]),
                                                str(row["dose"]["candidate_id"])))
    else:
        chosen = max(usable, key=lambda row: (float(row["recovery"]["robust"][score_key]),
                                              -float(row["dose"]["scalar_cost"])))
    return {"candidate_id": chosen["dose"]["candidate_id"],
            "target_attained": bool(attained), "target_score": float(target_score),
            "selected_score": float(chosen["recovery"]["robust"][score_key]),
            "selected_signature_rho": float(chosen["recovery"]["robust"]["spearman"]),
            "dose": chosen["dose"],
            "selection_rule": ("minimal scalar cost reaching target and signature floor" if attained
                               else "best oriented recovery; target not attained")}
