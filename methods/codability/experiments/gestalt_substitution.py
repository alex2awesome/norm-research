#!/usr/bin/env python
"""Experiment G: locate and test configural/gestalt scale--articulation substitution.

This module joins the target-indexed frontier to the existing prompt-space combiner and composition
instruments.  It does not call every unexplained failure "gestalt."  Instead it keeps three
empirical locations separate:

* interaction within the declared unit span (lookup minus linear combiner);
* value carried by composed presentation beyond separately executed units (``Delta_comp``);
* target-aligned behavior of a holistic candidate outside the linear unit span.

All three are estimates inside the fixed-target DPI bracket.  The only all-prompt upper bound is
still the target ceiling returned by ``vinfo``.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from methods.codability.experiments.target_articulation_frontier import (
    orbit_recovery,
    paired_substitution_test,
    target_orbit_mean,
    validate_orbit,
    validate_target_spec,
)
from methods.metric_implementer.experiments.composition_gap import delta_comp
from methods.metric_implementer.experiments.prompt_space_bracket import (
    joint_combiner_ceiling,
    reliability_ceiling,
    span_residual,
)


SCHEMA = "gestalt_substitution/v1"


def _unit_matrix(unit_signals: Sequence[Sequence[float]], n_items: int) -> np.ndarray:
    signals = np.asarray(unit_signals, float)
    if signals.ndim != 2 or signals.shape[1] != n_items or signals.shape[0] < 1:
        raise ValueError("unit_signals must have shape (n_units, n_target_items)")
    if not np.isfinite(signals).all() or not ((0.0 <= signals).all()
                                               and (signals <= 1.0).all()):
        raise ValueError("unit_signals must contain finite probabilities in [0, 1]")
    return signals


def gestalt_diagnostics(*, target_orbit: Mapping[str, Sequence[float]],
                        unit_signals: Sequence[Sequence[float]],
                        candidate_orbits: Mapping[str, Mapping[str, Sequence[float]]],
                        unit_ids: Sequence[str] | None = None,
                        divergence: str = "tvd", min_target_information: float = 1e-6,
                        interaction_floor_bits: float = 0.02,
                        composition_floor_bits: float = 0.02,
                        seed: int = 0) -> dict:
    """Assemble the interaction/composition/span profile on one frozen probe set."""
    target_forms = validate_orbit(target_orbit, name="target_orbit")
    target = target_orbit_mean(target_forms)
    units = _unit_matrix(unit_signals, len(target))
    if unit_ids is not None and len(unit_ids) != units.shape[0]:
        raise ValueError("unit_ids must align with unit_signals rows")
    candidates = {str(candidate_id): validate_orbit(orbit, n_items=len(target),
                                                    name=f"candidate_orbits[{candidate_id!r}]")
                  for candidate_id, orbit in candidate_orbits.items()}
    if not candidates:
        raise ValueError("candidate_orbits must contain at least one holistic/composed candidate")

    target_panel = np.stack(list(target_forms.values()), axis=1)
    reliability = reliability_ceiling(target_panel)
    ladder = joint_combiner_ceiling(units, target)
    recovery = {candidate_id: orbit_recovery(
                    target, orbit, divergence=divergence,
                    min_target_information=min_target_information)
                for candidate_id, orbit in candidates.items()}
    candidate_means = {candidate_id: np.mean(np.stack(list(orbit.values())), axis=0)
                       for candidate_id, orbit in candidates.items()}
    spans = {candidate_id: span_residual(units, values, target=target)
             for candidate_id, values in candidate_means.items()}

    linear_bits = float(ladder["rungs_mean"].get("linear", 0.0))
    composed_ids = sorted(candidate_means)
    composition = delta_comp(
        np.stack([candidate_means[candidate_id] for candidate_id in composed_ids]),
        target, units.T, linear_bits, variant_ids=composed_ids,
        beyond_floor=composition_floor_bits, seed=seed)
    max_channel_gap = float(max((row.get("channel_gap_bits_mean", 0.0)
                                 for row in spans.values()), default=0.0))
    interaction_detected = bool(ladder["interaction_bits"] > interaction_floor_bits)
    composition_detected = bool(composition["composition_carries_value"])
    outside_span_detected = bool(max_channel_gap > composition_floor_bits)
    return {
        "schema": "gestalt_diagnostics/v1",
        "target": {"construction": "mean of target form orbit",
                   "n_forms": len(target_forms), "n_items": len(target)},
        "target_reliability": reliability,
        "unit_inventory": {"n_units": int(units.shape[0]),
                           "unit_ids": (list(unit_ids) if unit_ids is not None else None),
                           "certification_asserted": False},
        "joint_combiner_ladder": ladder,
        "candidate_recovery": recovery,
        "candidate_span": spans,
        "composition_gap": composition,
        "gestalt_profile": {
            "interaction_within_unit_span": interaction_detected,
            "interaction_bits": float(ladder["interaction_bits"]),
            "composed_presentation_beyond_units": composition_detected,
            "delta_comp_beyond_bits": float(composition["delta_comp_beyond"]),
            "candidate_alignment_outside_linear_span": outside_span_detected,
            "max_channel_gap_bits": max_channel_gap,
            "any_gestalt_signal": bool(interaction_detected or composition_detected
                                       or outside_span_detected),
        },
        "floors": {"interaction_bits": float(interaction_floor_bits),
                   "composition_or_span_bits": float(composition_floor_bits)},
        "certified": False,
        "scope_note": ("The combiner, composition, and span quantities locate achieved/configural "
                       "behavior inside the target DPI bracket; none is an all-prompt ceiling."),
    }


def gestalt_substitution_report(*, target_spec: Mapping,
                                target_orbit: Mapping[str, Sequence[float]],
                                unit_signals: Sequence[Sequence[float]],
                                candidate_orbits: Mapping[str, Mapping[str, Sequence[float]]],
                                selected_candidate_id: str,
                                small_sparse_orbit: Mapping[str, Sequence[float]],
                                big_gestalt_orbit: Mapping[str, Sequence[float]] | None = None,
                                control_orbit: Mapping[str, Sequence[float]] | None = None,
                                unit_ids: Sequence[str] | None = None,
                                divergence: str = "tvd", min_target_information: float = 1e-6,
                                gap_delta: float = 0.02, equivalence_delta: float = 0.02,
                                min_signature_rho: float = 0.5,
                                signature_equivalence_delta: float = 0.05,
                                n_boot: int = 1000, seed: int = 0) -> dict:
    """Join a frozen Experiment G substitution decision to its gestalt-location profile."""
    spec = validate_target_spec(target_spec)
    if spec["target_view"] != "gestalt":
        raise ValueError("Experiment G requires target_view='gestalt'")
    if selected_candidate_id not in candidate_orbits:
        raise ValueError("selected_candidate_id is absent from candidate_orbits")
    diagnostics = gestalt_diagnostics(
        target_orbit=target_orbit, unit_signals=unit_signals,
        candidate_orbits=candidate_orbits, unit_ids=unit_ids,
        divergence=divergence, min_target_information=min_target_information, seed=seed)
    target = target_orbit_mean(target_orbit)
    big_orbit = target_orbit if big_gestalt_orbit is None else big_gestalt_orbit
    substitution = paired_substitution_test(
        target,
        small_sparse_orbit=small_sparse_orbit,
        big_sparse_orbit=big_orbit,
        articulated_orbit=candidate_orbits[selected_candidate_id],
        control_orbit=control_orbit,
        divergence=divergence,
        min_target_information=min_target_information,
        gap_delta=gap_delta,
        equivalence_delta=equivalence_delta,
        min_signature_rho=min_signature_rho,
        signature_equivalence_delta=signature_equivalence_delta,
        n_boot=n_boot,
        seed=seed + 10_000,
    )
    return {
        "schema": SCHEMA,
        "experiment": "G_model_internal_gestalt",
        "target_spec": spec,
        "selected_candidate_id": selected_candidate_id,
        "diagnostics": diagnostics,
        "substitution": substitution,
        "paper_grade_claim_eligible": False,
        "paper_grade_note": ("Eligibility must be asserted by a preregistered driver after unit "
                             "certification, prompt provenance, probe alignment, and control gates."),
    }

