"""Finite panel-state aggregation and process-relative bounds for CR-3 v14."""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from .cr_audit import clopper_pearson_interval, clopper_pearson_upper


STATE_TABLE_SCHEMA = "cr3-v14-state-tables-v1"
CERTIFICATE_SCHEMA = "cr3-v14-value-certificate-v1"


def _finite_vector(values: Sequence[float], *, name: str, nonempty: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or (nonempty and len(array) == 0) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return array


def binary_entropy_bits(values: Sequence[int]) -> float:
    vector = np.asarray(values, dtype=np.uint8)
    if vector.ndim != 1 or len(vector) == 0 or np.any((vector != 0) & (vector != 1)):
        raise ValueError("binary entropy needs a nonempty binary vector")
    probability = float(np.mean(vector))
    if probability in (0.0, 1.0):
        return 0.0
    return float(-probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability))


def plugin_binary_mutual_information(target: Sequence[int], predicted: Sequence[int]) -> float:
    left = np.asarray(target, dtype=np.uint8)
    right = np.asarray(predicted, dtype=np.uint8)
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise ValueError("MI needs aligned nonempty vectors")
    if np.any((left != 0) & (left != 1)) or np.any((right != 0) & (right != 1)):
        raise ValueError("MI inputs must be binary")
    counts = np.zeros((2, 2), dtype=float)
    np.add.at(counts, (left, right), 1.0)
    joint = counts / float(len(left))
    product = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    keep = joint > 0.0
    return float(np.sum(joint[keep] * np.log2(joint[keep] / product[keep])))


def balanced_agreement(target: Sequence[int], predicted: Sequence[int]) -> float:
    left = np.asarray(target, dtype=np.uint8)
    right = np.asarray(predicted, dtype=np.uint8)
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise ValueError("balanced agreement needs aligned nonempty vectors")
    recalls = [float(np.mean(right[left == label] == label)) for label in (0, 1) if np.any(left == label)]
    return float(np.mean(recalls))


def enumerate_states(panel_size: int = 8) -> np.ndarray:
    if not 1 <= int(panel_size) <= 16:
        raise ValueError("panel_size must lie in [1, 16]")
    states = np.arange(1 << int(panel_size), dtype=np.uint16)
    shifts = np.arange(int(panel_size) - 1, -1, -1, dtype=np.uint16)
    return ((states[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


def signatures_to_states(signatures: np.ndarray, panels: Sequence[Sequence[int]]) -> np.ndarray:
    raw = np.asarray(signatures, dtype=float)
    if raw.ndim != 2 or raw.shape[0] == 0 or np.any(~np.isfinite(raw)):
        raise ValueError("signatures must be a finite prompt-by-probe matrix")
    panel_rows = [np.asarray(row, dtype=int) for row in panels]
    if not panel_rows or len({len(row) for row in panel_rows}) != 1:
        raise ValueError("panels must be a nonempty equal-width family")
    width = len(panel_rows[0])
    if width > 16 or any(len(set(map(int, row))) != width for row in panel_rows):
        raise ValueError("panel width is unsupported or contains duplicate probes")
    if any(np.any(row < 0) or np.any(row >= raw.shape[1]) for row in panel_rows):
        raise ValueError("panel index outside signature matrix")
    binary = (raw > 0.5).astype(np.uint16)
    weights = (1 << np.arange(width - 1, -1, -1, dtype=np.uint16)).astype(np.uint16)
    return np.column_stack([
        np.sum(binary[:, row] * weights[None, :], axis=1, dtype=np.uint32)
        for row in panel_rows
    ]).astype(np.uint16)


def validate_state_tables(raw_lift: np.ndarray, clipped_value: np.ndarray, *, panel_size: int = 8) -> None:
    raw = np.asarray(raw_lift, dtype=float)
    clipped = np.asarray(clipped_value, dtype=float)
    expected_states = 1 << int(panel_size)
    if (raw.ndim != 2 or raw.shape != clipped.shape or raw.shape[1] != expected_states
            or raw.shape[0] == 0 or np.any(~np.isfinite(raw)) or np.any(~np.isfinite(clipped))):
        raise ValueError("v14 state tables are incomplete, non-finite, or have the wrong shape")
    if np.any(clipped < 0.0) or not np.allclose(clipped, np.maximum(raw, 0.0), rtol=0.0, atol=1e-12):
        raise ValueError("certified state values must be raw lift clipped at zero")


def tie_class(values: Sequence[float], identifiers: Sequence[str], *, atol: float = 1e-12) -> dict:
    vector = _finite_vector(values, name="tie values")
    ids = [str(value) for value in identifiers]
    if len(ids) != len(vector) or len(set(ids)) != len(ids):
        raise ValueError("tie identifiers must be aligned and unique")
    maximum = float(np.max(vector))
    members = sorted(ids[index] for index in np.flatnonzero(np.isclose(vector, maximum, rtol=0.0, atol=atol)))
    return {
        "maximum": maximum,
        "size": len(members),
        "members": members,
        "canonical_representative": members[0],
        "is_unique": len(members) == 1,
    }


def aggregate_state_tables(
    *, raw_lift: np.ndarray, clipped_value: np.ndarray, prompt_signatures: np.ndarray,
    panels: Sequence[Sequence[int]], prompt_ids: Sequence[str],
    decoder_families: Sequence[str] | None = None,
) -> dict:
    """Evaluate every mined prompt by exact panel-state lookup."""
    validate_state_tables(raw_lift, clipped_value, panel_size=len(panels[0]))
    raw = np.asarray(raw_lift, dtype=float)
    clipped = np.asarray(clipped_value, dtype=float)
    codes = signatures_to_states(prompt_signatures, panels)
    ids = [str(value) for value in prompt_ids]
    if len(ids) != len(codes) or len(set(ids)) != len(ids):
        raise ValueError("prompt identifiers must be aligned and unique")
    panel_positions = np.arange(raw.shape[0], dtype=int)[None, :]
    prompt_raw = np.mean(raw[panel_positions, codes], axis=1)
    prompt_clipped = np.mean(clipped[panel_positions, codes], axis=1)
    panel_caps = np.max(clipped, axis=1)
    raw_panel_caps = np.max(raw, axis=1)
    cap = float(np.mean(panel_caps))
    achieved = float(np.max(prompt_clipped))
    if cap + 1e-12 < achieved:
        raise RuntimeError("exact free-recombination cap is below an observed value")
    result = {
        "schema": CERTIFICATE_SCHEMA,
        "n_prompts": len(ids),
        "n_panels": int(raw.shape[0]),
        "n_states_per_panel": int(raw.shape[1]),
        "prompt_ids": ids,
        "prompt_states": codes,
        "prompt_raw_lift": prompt_raw,
        "prompt_value": prompt_clipped,
        "panel_caps": panel_caps,
        "raw_panel_caps": raw_panel_caps,
        "free_recombination_cap": cap,
        "achieved_value": achieved,
        "structural_gap": float(cap - achieved),
        "recombination_slack": float(cap - achieved),
        "legibility_argmax": tie_class(prompt_clipped, ids),
        "raw_legibility_argmax": tie_class(prompt_raw, ids),
    }
    if decoder_families is not None:
        families = np.asarray(list(map(str, decoder_families)), dtype=object)
        if families.shape != (raw.shape[0],):
            raise ValueError("decoder_families must have one entry per panel")
        rows = {}
        for family in sorted(set(families.tolist())):
            mask = families == family
            family_prompt_values = np.mean(clipped[panel_positions[:, mask], codes[:, mask]], axis=1)
            rows[family] = {
                "n_panels": int(np.sum(mask)),
                "achieved_value": float(np.max(family_prompt_values)),
                "mean_panel_cap": float(np.mean(panel_caps[mask])),
                "mean_prompt_value": float(np.mean(family_prompt_values)),
            }
        result["decoder_family_rows"] = rows
        result["decoder_family_achieved_variance"] = float(np.var(
            [row["achieved_value"] for row in rows.values()], ddof=0
        ))
    return result


def fidelity_legibility_diagnostic(
    *, prompt_signatures_on_h: np.ndarray, target_on_h: Sequence[int],
    legibility_values: Sequence[float], prompt_ids: Sequence[str],
    minimum_rho_for_ranking: float = 0.50,
) -> dict:
    signatures = np.asarray(prompt_signatures_on_h, dtype=float)
    target = np.asarray(target_on_h, dtype=np.uint8)
    legibility = _finite_vector(legibility_values, name="legibility")
    ids = [str(value) for value in prompt_ids]
    if (signatures.ndim != 2 or signatures.shape[1:] != (len(target),)
            or signatures.shape[0] != len(legibility) or len(ids) != len(legibility)
            or np.any(~np.isfinite(signatures))):
        raise ValueError("fidelity inputs are not aligned")
    fidelity = np.asarray([
        plugin_binary_mutual_information(target, row > 0.5) for row in signatures
    ], dtype=float)
    if len(fidelity) < 2 or np.all(fidelity == fidelity[0]) or np.all(legibility == legibility[0]):
        rho = None
    else:
        observed = float(spearmanr(fidelity, legibility).statistic)
        rho = observed if np.isfinite(observed) else None
    ranking_allowed = rho is not None and rho >= float(minimum_rho_for_ranking)
    return {
        "fidelity": fidelity,
        "fidelity_argmax": tie_class(fidelity, ids),
        "legibility_argmax": tie_class(legibility, ids),
        "spearman_rho": rho,
        "minimum_rho_for_ranking": float(minimum_rho_for_ranking),
        "optimal_prompt_ranking_allowed": bool(ranking_allowed),
        "reporting_rule": (
            "rank claim allowed" if ranking_allowed
            else "report tie classes and canonical representatives; no optimal-prompt claim"
        ),
    }


def classify_status(
    *, achieved: float, cap: float, raw_panel_caps: Sequence[float],
    blind_value: float | None = None, annotated_canonical_value: float | None = None,
    best_annotated_value: float | None = None,
    future_gain_bound: float | None = None, resolved_tolerance: float = 0.01,
) -> dict:
    raw_caps = _finite_vector(raw_panel_caps, name="raw panel caps")
    annotated_for_liveness = (
        best_annotated_value
        if best_annotated_value is not None else annotated_canonical_value
    )
    control_inversion = (
        blind_value is not None and annotated_for_liveness is not None
        and float(blind_value) >= float(annotated_for_liveness)
    )
    if float(cap) <= 1e-12:
        label = "DEAD_INSTRUMENT" if control_inversion or np.max(raw_caps) < 0.0 else "ZERO_CAP"
    elif future_gain_bound is None:
        label = "UNRESOLVED"
    elif float(future_gain_bound) <= float(resolved_tolerance):
        label = "RESOLVED"
    elif float(achieved) + float(future_gain_bound) >= float(cap) - 1e-12:
        label = "RISING"
    else:
        label = "PLATEAUED"
    return {
        "status": label,
        "achieved_value": float(achieved),
        "cap": float(cap),
        "blind_value": None if blind_value is None else float(blind_value),
        "annotated_canonical_value": (
            None if annotated_canonical_value is None else float(annotated_canonical_value)
        ),
        "best_annotated_value": (
            None if best_annotated_value is None else float(best_annotated_value)
        ),
        "best_minus_blind": (
            None if blind_value is None or annotated_for_liveness is None
            else float(annotated_for_liveness) - float(blind_value)
        ),
        "control_inversion": bool(control_inversion),
        "future_gain_bound": None if future_gain_bound is None else float(future_gain_bound),
    }


def record_rank_gain_bound(*, n: int, m: int, achieved: float, cap: float) -> dict:
    if int(n) <= 0 or int(m) < 0:
        raise ValueError("record/rank needs n>0 and m>=0")
    if not np.isfinite([achieved, cap]).all() or float(cap) + 1e-12 < float(achieved):
        raise ValueError("record/rank needs finite cap >= achieved")
    probability = float(m) / float(n + m) if m else 0.0
    return {
        "method": "exchangeable_record_rank",
        "n_observed": int(n),
        "future_horizon": int(m),
        "improvement_probability_upper": probability,
        "gain_upper": float(probability * max(0.0, float(cap) - float(achieved))),
        "ties_make_bound_conservative": True,
    }


def split_sample_cp_gain_bound(
    *, discovery_achieved: float, audit_values: Sequence[float], cap: float,
    future_horizon: int, alpha: float = 0.05, current_achieved: float | None = None,
) -> dict:
    audit = _finite_vector(audit_values, name="audit values")
    if not 0.0 < float(alpha) < 1.0 or int(future_horizon) < 0:
        raise ValueError("invalid CP alpha or horizon")
    if float(cap) + 1e-12 < float(discovery_achieved):
        raise ValueError("cap is below the frozen discovery incumbent")
    hits = int(np.sum(audit > float(discovery_achieved) + 1e-12))
    probability_upper = clopper_pearson_upper(hits, len(audit), float(alpha))
    hit_probability = 1.0 - (1.0 - probability_upper) ** int(future_horizon)
    audit_achieved = float(max([discovery_achieved, *audit.tolist()]))
    current = audit_achieved if current_achieved is None else float(current_achieved)
    if current + 1e-12 < audit_achieved or current > float(cap) + 1e-12:
        raise ValueError("current_achieved must include the audit and not exceed the cap")
    best_possible_gain = max(0.0, float(cap) - current)
    interval = clopper_pearson_interval(hits, len(audit), float(alpha))
    return {
        "method": "split_sample_cp_fixed_discovery_threshold",
        "discovery_achieved": float(discovery_achieved),
        "n_audit": len(audit),
        "n_audit_improvements": hits,
        "audit_improvement_probability_interval": list(map(float, interval)),
        "audit_improvement_probability_upper": float(probability_upper),
        "future_horizon": int(future_horizon),
        "future_hit_probability_upper": float(hit_probability),
        "gain_upper": float(hit_probability * best_possible_gain),
        "audit_achieved_lower_bound": audit_achieved,
        "current_achieved_for_gain_pricing": current,
        "threshold_frozen_before_audit": True,
    }


def reject_adaptive_zero_hit_cp(*_args, **_kwargs):
    raise ValueError(
        "invalid bound: I={v>A} and A cannot be defined from the same sample used as "
        "zero-hit CP evidence; freeze A on a discovery population first"
    )


def dkw_expected_best_gain_bound(
    *, observed_values: Sequence[float], achieved: float, cap: float,
    future_horizon: int, alpha: float = 0.05,
) -> dict:
    values = np.sort(_finite_vector(observed_values, name="observed values"))
    if not 0.0 < float(alpha) < 1.0 or int(future_horizon) < 0:
        raise ValueError("invalid DKW alpha or horizon")
    if float(cap) + 1e-12 < max(float(achieved), float(values[-1])):
        raise ValueError("DKW cap is below an observed value")
    epsilon = math.sqrt(math.log(2.0 / float(alpha)) / (2.0 * len(values)))
    edges = np.unique(np.concatenate(([float(achieved)], values[values > achieved], [float(cap)])))
    gain = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        empirical_cdf = float(np.searchsorted(values, left, side="right")) / len(values)
        cdf_lower = max(0.0, empirical_cdf - epsilon)
        exceed = 1.0 - cdf_lower ** int(future_horizon) if future_horizon else 0.0
        gain += float(right - left) * exceed
    return {
        "method": "dkw_expected_future_best",
        "n_observed": len(values),
        "future_horizon": int(future_horizon),
        "alpha": float(alpha),
        "dkw_epsilon": float(epsilon),
        "gain_upper": float(min(max(0.0, cap - achieved), max(0.0, gain))),
    }


def process_gain_certificate(
    *, observed_values: Sequence[float], discovery_achieved: float,
    audit_values: Sequence[float], cap: float, future_horizon: int,
    alpha: float = 0.05, provenance_valid: bool = True,
) -> dict:
    audit = _finite_vector(audit_values, name="audit values")
    observed = _finite_vector(observed_values, name="observed values")
    achieved = float(max([discovery_achieved, *audit.tolist(), *observed.tolist()]))
    stochastic_alpha = float(alpha) / 2.0
    rows = {
        "split_cp": split_sample_cp_gain_bound(
            discovery_achieved=float(discovery_achieved), audit_values=audit, cap=float(cap),
            future_horizon=int(future_horizon), alpha=stochastic_alpha,
        ),
        "dkw": dkw_expected_best_gain_bound(
            observed_values=np.concatenate((observed, audit)), achieved=achieved, cap=float(cap),
            future_horizon=int(future_horizon), alpha=stochastic_alpha,
        ),
    }
    if provenance_valid:
        rows["record_rank"] = record_rank_gain_bound(
            n=len(observed) + len(audit), m=int(future_horizon), achieved=achieved, cap=float(cap)
        )
    valid = [float(row["gain_upper"]) for row in rows.values()]
    return {
        "future_horizon": int(future_horizon),
        "achieved_after_audit": achieved,
        "bounds": rows,
        "headline_gain_upper": float(min(valid)),
        "headline_rule": "minimum of all premise-valid declared bounds",
        "stochastic_bound_alpha": stochastic_alpha,
        "within_claim_alpha_split": "equal split across split-CP and DKW",
        "record_rank_available": bool(provenance_valid),
    }


def _row_hash(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest()


def novelty_collapse_curves(
    *, full_signatures: np.ndarray, joint_codes: np.ndarray, values: Sequence[float],
    frozen_incumbent: float, discovery_signature_hashes: Sequence[str] = (),
    discovery_code_hashes: Sequence[str] = (), families: Sequence[str] | None = None,
) -> list[dict]:
    signatures = np.asarray(full_signatures)
    codes = np.asarray(joint_codes)
    value = _finite_vector(values, name="novelty values")
    if (signatures.ndim != 2 or codes.ndim != 2 or signatures.shape[0] != len(value)
            or codes.shape[0] != len(value)):
        raise ValueError("novelty inputs must have aligned draw rows")
    family_rows = ["pooled"] * len(value) if families is None else list(map(str, families))
    if len(family_rows) != len(value):
        raise ValueError("families must align with draws")
    output = []
    for family in ["pooled", *sorted(set(family_rows))] if families is not None else ["pooled"]:
        positions = np.arange(len(value)) if family == "pooled" else np.flatnonzero(np.asarray(family_rows) == family)
        seen_behavior = set(map(str, discovery_signature_hashes))
        seen_code = set(map(str, discovery_code_hashes))
        behavior_contacts = code_contacts = improving_contacts = 0
        for prefix, position in enumerate(positions, start=1):
            behavior_hash = _row_hash(signatures[position])
            code_hash = _row_hash(codes[position])
            new_behavior = behavior_hash not in seen_behavior
            new_code = code_hash not in seen_code
            improving = new_code and float(value[position]) > float(frozen_incumbent) + 1e-12
            behavior_contacts += int(new_behavior)
            code_contacts += int(new_code)
            improving_contacts += int(improving)
            seen_behavior.add(behavior_hash)
            seen_code.add(code_hash)
            output.append({
                "family": family,
                "prefix": prefix,
                "draw_index": int(position),
                "new_behavior": bool(new_behavior),
                "new_code": bool(new_code),
                "new_value_improving_code": bool(improving),
                "cumulative_behavior_novelty_rate": behavior_contacts / prefix,
                "cumulative_code_novelty_rate": code_contacts / prefix,
                "cumulative_value_improving_novelty_rate": improving_contacts / prefix,
                "frozen_incumbent": float(frozen_incumbent),
            })
    return output
