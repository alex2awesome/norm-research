"""Resampled-teaching-panel Reconstruction-MCQ value certificates (CR-3 v13).

The v12 value instrument froze ONE ordered eight-text teaching panel per metric and
reported an instrument-locked value. This driver instead resamples ``R`` frozen teaching
panels from the same design-time data, measures the Reconstruction-MCQ value of every
mined prompt under each panel, and reports a per-prompt mean value with a percentile
confidence interval ACROSS panels -- generalization over the panel choice rather than a
single-panel point. DKW expected-best bounds on the per-family distribution of the
per-prompt mean quantify the value-added of additional mining at each horizon.

Nothing here is anchor-, label-, outcome-, or human-aware. Panels are built by
``build_teaching_panel_library`` before any prompt search from frozen bootstrap behavior
only; each panel's value depends on a prompt exclusively through that prompt's eight-bit
executor verdict state on the panel's texts, which is what makes per-state caching exact.
All quantities are conditional on the frozen empirical probe panel and the frozen MCQ
codebook. The certified achieved value is the largest per-prompt CI lower bound; it is a
lower bound on articulation value, never a prompt-space upper bound.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..recon_channel import (
    mcq_logit_values_from_precomputed_behaviors,
    mcq_no_demo_choice_probabilities,
    mcq_option_order_design,
)
from .cr3_reconstruction_values import (
    _binary_state_rows,
    _bootstrap,
    _load_scored_rows,
    _payload_sha256,
    build_teaching_panel_library,
    validate_codebook_manifest,
)
from .cr_audit import (
    clopper_pearson_upper,
    dkw_expected_max_lower,
    dkw_expected_max_upper,
)
from .v13_value_cache import ValueCache, cache_key

SAMPLED_VALUE_SCHEMA = "cr3-sampled-v13"
PANEL_PLAN_SCHEMA = "cr3-sampled-panel-plan-v13"
PER_PROMPT_TABLE_SCHEMA = "cr3-sampled-per-prompt-values-v13"
RUN_MANIFEST_SCHEMA = "cr3-sampled-run-manifest-v13"

# New campaigns use these schemas.  The v13 names above remain readable so the
# historical 20-test instrument and already-written certificates do not change meaning.
VALUE_BOUND_RELEASE = "v13.1"
VALUE_BOUND_DESIGN_SCHEMA = "cr3-value-bound-design-v13.1"
VALUE_BOUND_STATE_SCHEMA = "cr3-value-bound-state-tables-v13.1"
VALUE_BOUND_CERTIFICATE_SCHEMA = "cr3-value-bound-certificate-v13.1"
VALUE_BOUND_RESULTS_SCHEMA = "cr3-value-bound-results-v13.1"
VALUE_BOUND_DESIGN_SALT = "cr3-v13.1-six-pool-design-20260713"
N_DESIGN_POOLS = 6
POOL_SIZE = 12
MCQ_PANELS_PER_POOL = 12
MCQ_PANEL_SIZE = 8
BEHAVIORAL_PANELS_PER_POOL = 4
BEHAVIORAL_PANEL_SIZE = 6

# Headline gate constants. A resampled value only earns a headline when the blind menu
# leaves genuine headroom AND the certified prompt's cross-panel interval is tight; both
# thresholds are recorded in every payload so a reader never has to guess them.
HEADLINE_MIN_BLIND_HEADROOM = 0.10
HEADLINE_MAX_ACHIEVED_VALUE_CI_WIDTH = 0.15

PRIMARY_CI_PERCENTILES = (2.5, 97.5)
SENSITIVITY_CI_PERCENTILES = (5.0, 95.0)

PLANTED_CONTROL_FAMILY = "planted_positive_control"
DEGENERATE_CONTROL_FAMILY = "degenerate_constant_control"
PLANTED_CONTROL_TEXT = "CR3_SAMPLED_PLANTED_POSITIVE_CONTROL_PROMPT"
DEGENERATE_CONTROL_TEXT = "CR3_SAMPLED_DEGENERATE_CONSTANT_CONTROL_PROMPT"

VALUE_STATUS_CERTIFIED_PRIMARY = "CERTIFIED_SAMPLED_VALUE"
VALUE_STATUS_SUGGESTIVE_SENSITIVITY = "SUGGESTIVE_SAMPLED_VALUE"
VALUE_STATUS_FORMAL_ONLY = "FORMAL_CERTIFICATE_ONLY"


def _percentile_interval(values: np.ndarray, percentiles: Sequence[float]) -> np.ndarray:
    """Per-row (prompt) low/high percentile interval across the panel axis."""
    matrix = np.asarray(values, dtype=float)
    lower = np.percentile(matrix, percentiles[0], axis=1)
    upper = np.percentile(matrix, percentiles[1], axis=1)
    return np.vstack([lower, upper])


def _headline_eligible(*, blind_headroom: float, achieved_value_ci_width: float) -> bool:
    """Headline requires both frozen gate constants: menu headroom and interval tightness."""
    return bool(
        float(blind_headroom) >= HEADLINE_MIN_BLIND_HEADROOM
        and float(achieved_value_ci_width) <= HEADLINE_MAX_ACHIEVED_VALUE_CI_WIDTH)


def _value_status(*, headline_eligible: bool, achieved_value: float, positive_label: str) -> str:
    """A positive headline needs an eligible instrument and a strictly positive lower bound."""
    if bool(headline_eligible) and float(achieved_value) > 1e-12:
        return positive_label
    return VALUE_STATUS_FORMAL_ONLY


def _allocate_horizon(sizes: Mapping[str, int], horizon: int) -> dict[str, int]:
    """Deterministic largest-remainder split of ``horizon`` across families by size."""
    families = sorted(sizes)
    total = sum(int(sizes[family]) for family in families)
    if total <= 0 or int(horizon) <= 0:
        return {family: 0 for family in families}
    raw = {family: int(horizon) * int(sizes[family]) / total for family in families}
    allocation = {family: int(math.floor(raw[family])) for family in families}
    remainder = int(horizon) - sum(allocation.values())
    order = sorted(families, key=lambda family: (-(raw[family] - allocation[family]), family))
    for family in order[:remainder]:
        allocation[family] += 1
    return allocation


def _hard_state_integers(score_rows: np.ndarray, panel_indices: np.ndarray) -> np.ndarray:
    """Per-prompt big-endian binary state over the panel's stored teaching order.

    Matches ``_binary_state_integer``/``_binary_state_rows``: the first teaching item is
    the most significant bit.
    """
    bits = (np.asarray(score_rows, dtype=float)[:, np.asarray(panel_indices, dtype=int)]
            > 0.5).astype(np.int64)
    weights = 1 << np.arange(bits.shape[1] - 1, -1, -1, dtype=np.int64)
    return bits @ weights


def _joint_tuple_species(panel_state_matrix: np.ndarray) -> list[str]:
    """Serialize each prompt's per-panel state vector into one deterministic species key."""
    matrix = np.asarray(panel_state_matrix, dtype=np.int64)
    return ["-".join(str(int(state)) for state in row) for row in matrix]


def _first_contact_novelty_counts(
    species: Sequence[str], families: Sequence[str],
) -> dict[str, dict]:
    """Per-family first-contact counts of joint tuples against the family's own prefix.

    A tuple new to the whole pool is necessarily new to its family, so the family-prefix
    first-contact rate upper-bounds pool-level tuple novelty.
    """
    if len(species) != len(families):
        raise ValueError("species and families must align one-to-one")
    seen_by_family: dict[str, set[str]] = {}
    counts: dict[str, dict] = {}
    for tuple_key, family in zip(species, families):
        family = str(family)
        seen = seen_by_family.setdefault(family, set())
        row = counts.setdefault(family, {
            "n_draws": 0, "n_first_contact_novel_tuples": 0, "n_distinct_tuples": 0})
        row["n_draws"] += 1
        if tuple_key not in seen:
            seen.add(tuple_key)
            row["n_first_contact_novel_tuples"] += 1
            row["n_distinct_tuples"] += 1
    return counts


def _unseen_tuple_probability_upper(u0_upper: float, horizon: int) -> float:
    """P(any of ``horizon`` future family draws lands a never-seen joint tuple) <= this."""
    if not 0.0 <= float(u0_upper) <= 1.0:
        raise ValueError("u0_upper must lie in [0, 1]")
    if int(horizon) < 0:
        raise ValueError("horizon must be nonnegative")
    return float(1.0 - (1.0 - float(u0_upper)) ** int(horizon))


def _state_capture_recapture(
    *,
    panel_state_matrix: np.ndarray,
    families: Sequence[str],
    mined_indices: Sequence[int],
    mean_value: np.ndarray,
    value_cap: float,
    alpha: float,
    horizons: Sequence[int],
    plan: Mapping[str, object],
    draw_order_source: str,
    state_values_by_panel: Sequence[np.ndarray] | None,
) -> dict:
    """The joint-tuple capture-recapture value-added block.

    Novelty statistics run over MINED rows only; synthetic calibration controls are not
    draws from the mining process. The free-recombination cap prices every enumerable
    per-panel state, seen or not, so it dominates every achievable resampled mean value.
    """
    mined = np.asarray(mined_indices, dtype=int)
    if len(mined) == 0:
        raise ValueError("capture-recapture requires at least one mined pool row")
    mined_species = _joint_tuple_species(np.asarray(panel_state_matrix)[mined])
    mined_families = [str(families[index]) for index in mined]
    counts = _first_contact_novelty_counts(mined_species, mined_families)
    family_names = sorted(counts)
    n_families = len(family_names)
    per_family = {}
    for family in family_names:
        row = counts[family]
        z = int(row["n_first_contact_novel_tuples"])
        n = int(row["n_draws"])
        per_family[family] = {
            **row,
            "missing_tuple_mass_upper_per_family_alpha": clopper_pearson_upper(
                z, n, float(alpha)),
            "missing_tuple_mass_upper_bonferroni_alpha": clopper_pearson_upper(
                z, n, float(alpha) / n_families),
        }

    observed_best_mean_value_mined = float(np.max(np.asarray(mean_value, float)[mined]))
    block: dict = {
        "joint_tuple_species_definition": (
            "hyphen-joined per-panel big-endian 8-bit state integers, in frozen panel-plan "
            "order"),
        "n_panels_R": int(np.asarray(panel_state_matrix).shape[1]),
        "draw_order_source": str(draw_order_source),
        "n_mined_draws": int(len(mined)),
        "n_distinct_joint_tuples": int(len(set(mined_species))),
        "observed_best_mean_value_mined": observed_best_mean_value_mined,
        "per_family": per_family,
        "premises": {
            "joint_tuple_species_is_the_value_relevant_quotient": True,
            "free_recombination_cap_is_conservative": True,
            "u0_from_clopper_pearson_under_iid_within_family_exchangeability": True,
            "family_prefix_first_contact_upper_bounds_pool_tuple_novelty": True,
            "no_smoothness_lipschitz_or_submodularity_assumptions": True,
            "controls_excluded_from_novelty_statistics": True,
        },
    }
    if state_values_by_panel is None:
        block["unseen_state_pricing"] = {
            "computed": False,
            "skip_reason": "unseen-state enumeration disabled by --skip-unseen-state-cap",
        }
        block["value_added_bounds"] = {
            "computed": False,
            "skip_reason": "the tuple-novelty gain formula requires the enumerated V-bar cap",
        }
        return block

    n_panels = int(np.asarray(panel_state_matrix).shape[1])
    if len(state_values_by_panel) != n_panels:
        raise ValueError("state value tables must cover every panel")
    per_panel = []
    all_state_maxima = []
    unseen_maxima = []
    panels = list(plan["panels"])
    for column in range(n_panels):
        state_values = np.asarray(state_values_by_panel[column], dtype=float)
        if state_values.shape != (256,) or np.any(~np.isfinite(state_values)):
            raise ValueError("each panel needs one finite value per enumerated 8-bit state")
        observed_states = set(
            int(state) for state in np.asarray(panel_state_matrix)[mined, column])
        unseen_states = [state for state in range(256) if state not in observed_states]
        all_state_max = float(np.max(state_values))
        unseen_state_max = (float(np.max(state_values[unseen_states]))
                            if unseen_states else 0.0)
        if unseen_state_max < 0.0:
            raise RuntimeError("unseen-state maximum value left the [0, value_cap] range")
        all_state_maxima.append(all_state_max)
        unseen_maxima.append(unseen_state_max)
        per_panel.append({
            "panel_index": column,
            "library_index": int(panels[column]["library_index"]),
            "n_states": 256,
            "n_pool_observed_states": len(observed_states),
            "n_unseen_states": len(unseen_states),
            "all_state_max_value": all_state_max,
            "unseen_state_max_value": unseen_state_max,
        })
    v_bar_cap = float(np.mean(all_state_maxima))
    if v_bar_cap < float(np.max(np.asarray(mean_value, float))) - 1e-9:
        raise RuntimeError(
            "free-recombination V-bar cap fell below an observed per-prompt mean value")
    free_recombination_headroom = float(max(
        0.0, v_bar_cap - observed_best_mean_value_mined))
    block["unseen_state_pricing"] = {
        "computed": True,
        "n_states_per_panel": 256,
        "state_encoding": "unsigned big-endian binary over the stored teaching order",
        "per_panel": per_panel,
        "mean_all_state_max_value_v_bar_cap": v_bar_cap,
        "mean_unseen_state_max_value": float(np.mean(unseen_maxima)),
        "value_cap": float(value_cap),
        "free_recombination_headroom": free_recombination_headroom,
    }

    sizes = {family: int(counts[family]["n_draws"]) for family in family_names}
    per_horizon = {}
    for horizon in horizons:
        family_rows = {}
        for family in family_names:
            u0_upper = per_family[family]["missing_tuple_mass_upper_per_family_alpha"]
            probability_any = _unseen_tuple_probability_upper(u0_upper, int(horizon))
            family_rows[family] = {
                "family_horizon": int(horizon),
                "component_alpha": float(alpha),
                "missing_tuple_mass_upper": float(u0_upper),
                "probability_any_unseen_tuple_upper": probability_any,
                "unseen_tuple_gain_upper": float(
                    probability_any * free_recombination_headroom),
            }
        allocation = _allocate_horizon(sizes, int(horizon))
        pooled_no_unseen = 1.0
        for family in family_names:
            u0_bonferroni = per_family[family]["missing_tuple_mass_upper_bonferroni_alpha"]
            pooled_no_unseen *= (1.0 - float(u0_bonferroni)) ** int(allocation[family])
        pooled_probability_any = float(1.0 - pooled_no_unseen)
        per_horizon[str(int(horizon))] = {
            "per_family": family_rows,
            "pooled": {
                "horizon_allocation": {
                    family: int(count) for family, count in allocation.items()},
                "component_alpha": float(alpha) / n_families,
                "probability_any_unseen_tuple_upper": pooled_probability_any,
                "unseen_tuple_gain_upper": float(
                    pooled_probability_any * free_recombination_headroom),
            },
        }
    block["value_added_bounds"] = {
        "computed": True,
        "gain_formula": (
            "probability_any_unseen_tuple_upper * max(0, v_bar_cap - "
            "observed_best_mean_value_mined)"),
        "per_horizon": per_horizon,
    }
    return block


def _codebook_menu(codebook_manifest: Mapping[str, object], target_metric_key: str) -> dict:
    """Resolve the frozen menu (target + stored distractors) without reselecting anything."""
    entries = codebook_manifest["entries"]
    if target_metric_key not in entries or not entries[target_metric_key]["valid"]:
        raise ValueError(f"target metric {target_metric_key!r} lacks a valid frozen codebook")
    entry = entries[target_metric_key]
    metric_meta = codebook_manifest["metrics"]
    target_bootstrap = _bootstrap(metric_meta[target_metric_key]["bootstrap_path"])
    if target_bootstrap["sha256"] != metric_meta[target_metric_key]["bootstrap_sha256"]:
        raise ValueError("target bootstrap changed after codebook freezing")
    distractors = []
    option_descriptions = [str(entry["target_description"])]
    for distractor_key in entry["distractor_metric_keys"]:
        metadata = metric_meta[distractor_key]
        view = _bootstrap(metadata["bootstrap_path"])
        if view["sha256"] != metadata["bootstrap_sha256"]:
            raise ValueError("distractor bootstrap changed after codebook freezing")
        distractors.append({
            "metric_id": distractor_key,
            "description": view["description"],
            "body": view["description"],
            "scores": view["target"],
        })
        option_descriptions.append(str(view["description"]))
    return {
        "entry": entry,
        "target_bootstrap": target_bootstrap,
        "distractors": distractors,
        "option_descriptions": option_descriptions,
    }


def _panel_index_plan(
    codebook_manifest: Mapping[str, object], target_metric_key: str, *, n_panels: int,
) -> dict:
    """Build the R deterministic frozen teaching panels for one metric."""
    library = build_teaching_panel_library(
        codebook_manifest, target_metric_key=target_metric_key, library_size=int(n_panels))
    if int(library["library_size"]) != int(n_panels):
        raise RuntimeError("teaching-panel library did not record the requested panel count")
    panels = []
    design_split = set(int(index) for index in codebook_manifest["design_indices"])
    for panel in library["panels"]:
        indices = [int(index) for index in panel["fixed_teaching_indices"]]
        if len(set(indices)) != len(indices) or not design_split.isdisjoint(indices):
            raise RuntimeError("teaching panel is not design-split-only and disjoint")
        panels.append({
            "library_index": int(panel["library_index"]),
            "role": str(panel["role"]),
            "fixed_teaching_indices": indices,
            "fixed_teaching_target_scores": [int(v) for v in panel["fixed_teaching_target_scores"]],
            "teaching_panel_sha256": str(panel["teaching_panel_sha256"]),
        })
    return {
        "schema": PANEL_PLAN_SCHEMA,
        "target_metric_key": str(target_metric_key),
        "n_panels_R": int(n_panels),
        "library_sha256": str(library["library_sha256"]),
        "panels": panels,
    }


def _blind_menu_prior(
    reconstructor, *, noun: str, option_descriptions: Sequence[str], n_perms: int,
) -> dict:
    """The blind, position-counterbalanced menu prior.

    The no-demonstration query renders only the unlabeled menu, so the prior is a function
    of the menu and the option-order schedule alone -- it is identical for every resampled
    panel. It is therefore computed once per metric and reused verbatim for every panel.
    """
    report = mcq_no_demo_choice_probabilities(
        reconstructor, noun=str(noun),
        option_descriptions=list(option_descriptions), n_draws=int(n_perms))
    canonical = np.asarray(report["canonical_choice_probabilities"], dtype=float)
    q0 = float(report["target_probability"])
    return {
        "canonical_choice_probabilities": canonical,
        "q0_target_probability": q0,
        "value_cap": float(1.0 - q0),
        "maximum_option_probability": float(report["maximum_option_probability"]),
        "normalized_entropy": float(report["normalized_entropy"]),
        "query_sha256": list(report["query_sha256"]),
    }


def _panel_prompt_values(
    reconstructor, *, noun: str, target_metric_key: str, target_description: str,
    distractors: Sequence[Mapping[str, object]], probe_texts: Sequence[str],
    prompt_texts: Sequence[str], score_rows: np.ndarray, panel_indices: Sequence[int],
    n_perms: int, max_chars: int, blind_canonical: np.ndarray, query_batch_size: int,
) -> np.ndarray:
    """Value every prompt under ONE frozen panel, caching by eight-bit verdict state.

    The value is a function of the prompt only through its ordered hard-verdict state on
    the panel's texts; identical states share one reconstruction call.
    """
    idx = np.asarray(panel_indices, dtype=int)
    rows = np.asarray(score_rows, dtype=float)
    hard_states = (rows[:, idx] > 0.5).astype(np.uint8)
    representative_by_state: dict[bytes, int] = {}
    for prompt_index in range(len(prompt_texts)):
        representative_by_state.setdefault(hard_states[prompt_index].tobytes(), prompt_index)
    representative_indices = list(representative_by_state.values())
    details = mcq_logit_values_from_precomputed_behaviors(
        reconstructor,
        noun=str(noun),
        candidate_prompt_texts=[prompt_texts[index] for index in representative_indices],
        target_metric_id=str(target_metric_key),
        target_description=str(target_description),
        target_score_rows=rows[representative_indices],
        probe_texts=list(probe_texts),
        distractors=list(distractors),
        design_indices=idx,
        codebook_frozen_before_prompt_search=True,
        n_examples=len(idx),
        n_reconstruction_draws=int(n_perms),
        max_chars=int(max_chars),
        query_batch_size=int(query_batch_size),
        fixed_no_demo_canonical_probabilities=np.asarray(blind_canonical, dtype=float),
        fixed_teaching_panel=True,
    )
    expected_order = idx.astype(int).tolist()
    value_by_state: dict[bytes, float] = {}
    for representative_position, prompt_index in enumerate(representative_indices):
        detail = details[representative_position]
        design = detail.get("design") or {}
        if (design.get("indices_in_prompt_order") != expected_order
                or not design.get("fixed_ordered_teaching_panel")):
            raise RuntimeError("panel value did not use the frozen ordered teaching panel")
        value_by_state[hard_states[prompt_index].tobytes()] = float(detail["value_mark"])
    return np.asarray(
        [value_by_state[hard_states[index].tobytes()] for index in range(len(prompt_texts))],
        dtype=float)


def _synthetic_control_rows(
    *, target_bootstrap: Mapping[str, object], planted_control: bool, degenerate_control: bool,
) -> dict:
    """Deterministic calibration prompts appended to the pool under explicit flags.

    The planted prompt's executor verdicts equal the target's frozen bootstrap verdicts, so
    every panel teaches the target exactly; the degenerate prompt's verdicts are constant
    and cannot teach any panel. Both ride the same instrument as an anchor pair.
    """
    target = np.asarray(target_bootstrap["target"], dtype=float)
    texts: list[str] = []
    rows: list[np.ndarray] = []
    families: list[str] = []
    if planted_control:
        texts.append(PLANTED_CONTROL_TEXT)
        rows.append((target > 0.5).astype(float))
        families.append(PLANTED_CONTROL_FAMILY)
    if degenerate_control:
        texts.append(DEGENERATE_CONTROL_TEXT)
        rows.append(np.zeros_like(target))
        families.append(DEGENERATE_CONTROL_FAMILY)
    return {"texts": texts, "rows": rows, "families": families}


def certify_sampled_value(
    reconstructor,
    *,
    codebook_manifest: Mapping[str, object],
    target_metric_key: str,
    scored_pool_path: str | Path,
    n_panels: int,
    n_perms: int,
    mcq_n_options: int,
    alpha: float,
    horizons: Sequence[int],
    reconstructor_model: str,
    reconstructor_revision: str,
    max_chars: int | None = None,
    query_batch_size: int = 512,
    planted_control: bool = False,
    degenerate_control: bool = False,
    with_unseen_state_cap: bool = True,
    codebook_provenance: Mapping[str, object] | None = None,
) -> dict:
    """Produce a v13 sampled-value certificate payload plus its per-prompt table arrays."""
    validate_codebook_manifest(codebook_manifest)
    target_metric_key = str(target_metric_key)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if int(mcq_n_options) < 2:
        raise ValueError("mcq_n_options must be at least 2")
    if int(n_perms) < int(mcq_n_options) or int(n_perms) % int(mcq_n_options) != 0:
        raise ValueError("n_perms must be a positive multiple of mcq_n_options")
    horizons = [int(value) for value in horizons]
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must be positive integers")

    noun = str(codebook_manifest["reconstruction_noun"])
    codebook_max_chars = int(codebook_manifest["reconstruction_max_chars"])
    if max_chars is None:
        max_chars = codebook_max_chars
    elif int(max_chars) != codebook_max_chars:
        raise ValueError("max_chars differs from the frozen codebook rendering contract")
    n_options = int(codebook_manifest["n_options"])
    if n_options != int(mcq_n_options):
        raise ValueError("mcq_n_options differs from the frozen codebook option count")

    menu = _codebook_menu(codebook_manifest, target_metric_key)
    entry = menu["entry"]
    target_bootstrap = menu["target_bootstrap"]
    distractors = menu["distractors"]
    option_descriptions = menu["option_descriptions"]
    probe_texts = target_bootstrap["probe_texts"]

    scored = _load_scored_rows(
        scored_pool_path, expected_probe_sha256=str(codebook_manifest["probe_sha256"]))
    if scored["signatures"].shape[1] != len(probe_texts):
        raise ValueError("scored prompt signatures are not aligned to bootstrap probes")
    prompt_texts = list(scored["texts"])
    score_rows = np.asarray(scored["signatures"], dtype=float)
    families = list(scored["families"]) if scored["families"] is not None else [
        "unlabeled"] * len(prompt_texts)
    if len(families) != len(prompt_texts):
        raise ValueError("scored pool family tags are not aligned to prompt rows")

    control_family_names = {PLANTED_CONTROL_FAMILY, DEGENERATE_CONTROL_FAMILY}
    if control_family_names.intersection(str(family) for family in families):
        raise ValueError("scored pool families collide with reserved control family names")
    n_mined = len(prompt_texts)
    controls = _synthetic_control_rows(
        target_bootstrap=target_bootstrap,
        planted_control=bool(planted_control), degenerate_control=bool(degenerate_control))
    if controls["texts"]:
        prompt_texts = prompt_texts + controls["texts"]
        score_rows = np.vstack([score_rows, np.asarray(controls["rows"], dtype=float)])
        families = families + controls["families"]
    n_prompts = len(prompt_texts)
    if n_prompts == 0 or n_mined == 0:
        raise ValueError("the value certificate needs at least one mined prompt row")
    with np.load(scored["path"], allow_pickle=True) as pool_npz:
        if "draw_order" in pool_npz.files:
            draw_order = np.asarray(pool_npz["draw_order"], dtype=int)
            if (draw_order.shape != (n_mined,)
                    or sorted(int(value) for value in draw_order) != list(range(n_mined))):
                raise ValueError("pool draw_order must be a permutation of its row indices")
            draw_order_source = "pool_draw_order_field"
        else:
            draw_order = np.arange(n_mined, dtype=int)
            draw_order_source = "pool_stored_order"

    plan = _panel_index_plan(
        codebook_manifest, target_metric_key, n_panels=int(n_panels))
    blind = _blind_menu_prior(
        reconstructor, noun=noun, option_descriptions=option_descriptions, n_perms=int(n_perms))
    value_cap = float(blind["value_cap"])

    value_matrix = np.empty((n_prompts, int(n_panels)), dtype=float)
    panel_state_matrix = np.empty((n_prompts, int(n_panels)), dtype=np.int64)
    state_values_by_panel: list[np.ndarray] | None = [] if with_unseen_state_cap else None
    for column, panel in enumerate(plan["panels"]):
        panel_indices = np.asarray(panel["fixed_teaching_indices"], dtype=int)
        column_values = _panel_prompt_values(
            reconstructor,
            noun=noun,
            target_metric_key=target_metric_key,
            target_description=str(entry["target_description"]),
            distractors=distractors,
            probe_texts=probe_texts,
            prompt_texts=prompt_texts,
            score_rows=score_rows,
            panel_indices=panel_indices,
            n_perms=int(n_perms),
            max_chars=int(max_chars),
            blind_canonical=blind["canonical_choice_probabilities"],
            query_batch_size=int(query_batch_size),
        )
        if column_values.shape != (n_prompts,) or np.any(~np.isfinite(column_values)):
            raise RuntimeError("panel value column did not cover every prompt row")
        if np.any(column_values < -1e-12) or np.any(column_values > value_cap + 1e-12):
            raise RuntimeError("panel value column left the frozen [0, value_cap] range")
        value_matrix[:, column] = np.clip(column_values, 0.0, value_cap)
        panel_state_matrix[:, column] = _hard_state_integers(score_rows, panel_indices)
        if state_values_by_panel is not None:
            # The exhaustive per-panel state population, priced through the SAME value
            # path as pool prompts; mirrors write_finite_state_scored_artifact's rows.
            state_bits = _binary_state_rows(len(panel_indices))
            synthetic_rows = np.zeros((len(state_bits), len(probe_texts)), dtype=float)
            synthetic_rows[:, panel_indices] = state_bits
            synthetic_texts = [
                f"finite-state transcript {state:0{len(panel_indices)}b}"
                for state in range(len(state_bits))
            ]
            state_values = _panel_prompt_values(
                reconstructor,
                noun=noun,
                target_metric_key=target_metric_key,
                target_description=str(entry["target_description"]),
                distractors=distractors,
                probe_texts=probe_texts,
                prompt_texts=synthetic_texts,
                score_rows=synthetic_rows,
                panel_indices=panel_indices,
                n_perms=int(n_perms),
                max_chars=int(max_chars),
                blind_canonical=blind["canonical_choice_probabilities"],
                query_batch_size=int(query_batch_size),
            )
            if (state_values.shape != (len(state_bits),)
                    or np.any(~np.isfinite(state_values))
                    or np.any(state_values < -1e-12)
                    or np.any(state_values > value_cap + 1e-12)):
                raise RuntimeError("state enumeration values left the frozen value range")
            state_values = np.clip(state_values, 0.0, value_cap)
            observed = value_matrix[:, column]
            expected = state_values[panel_state_matrix[:, column]]
            if not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    "pool prompt values disagree with their enumerated state values; the "
                    "state factorization invariant failed")
            state_values_by_panel.append(state_values)

    mean_value = value_matrix.mean(axis=1)
    primary_ci = _percentile_interval(value_matrix, PRIMARY_CI_PERCENTILES)
    sensitivity_ci = _percentile_interval(value_matrix, SENSITIVITY_CI_PERCENTILES)

    def _tier(ci: np.ndarray, positive_label: str) -> dict:
        lower, upper = ci[0], ci[1]
        best_index = int(np.argmax(lower))
        achieved_value = float(lower[best_index])
        ci_width = float(upper[best_index] - lower[best_index])
        headline_eligible = _headline_eligible(
            blind_headroom=blind["value_cap"], achieved_value_ci_width=ci_width)
        return {
            "achieved_value": achieved_value,
            "achieved_value_prompt_index": best_index,
            "achieved_value_prompt_text": prompt_texts[best_index],
            "achieved_value_prompt_family": families[best_index],
            "achieved_value_ci_lower": float(lower[best_index]),
            "achieved_value_ci_upper": float(upper[best_index]),
            "achieved_value_ci_width": ci_width,
            "headline_eligible": headline_eligible,
            "value_status": _value_status(
                headline_eligible=headline_eligible, achieved_value=achieved_value,
                positive_label=positive_label),
        }

    primary = _tier(primary_ci, VALUE_STATUS_CERTIFIED_PRIMARY)
    sensitivity = _tier(sensitivity_ci, VALUE_STATUS_SUGGESTIVE_SENSITIVITY)

    raw_best_index = int(np.argmax(mean_value))
    per_family_best: dict[str, dict] = {}
    family_indices: dict[str, list[int]] = {}
    for index, family in enumerate(families):
        family_indices.setdefault(str(family), []).append(index)
    for family, indices in family_indices.items():
        family_best_index = int(indices[int(np.argmax(mean_value[indices]))])
        per_family_best[family] = {
            "best_mean_value": float(mean_value[family_best_index]),
            "best_mean_value_prompt_index": family_best_index,
            "n_prompts": len(indices),
        }

    gain_bounds = _gain_bounds(
        mean_value=mean_value, families=families, family_indices=family_indices,
        value_cap=value_cap, alpha=float(alpha), horizons=horizons)

    state_capture_recapture = _state_capture_recapture(
        panel_state_matrix=panel_state_matrix,
        families=families,
        mined_indices=draw_order,
        mean_value=mean_value,
        value_cap=value_cap,
        alpha=float(alpha),
        horizons=horizons,
        plan=plan,
        draw_order_source=draw_order_source,
        state_values_by_panel=state_values_by_panel,
    )

    per_prompt_table = {
        "schema": PER_PROMPT_TABLE_SCHEMA,
        "target_metric_key": target_metric_key,
        "panel_plan_sha256": plan["library_sha256"],
        "prompt_texts": np.asarray(prompt_texts, dtype=object),
        "prompt_families": np.asarray(families, dtype=object),
        "panel_value_matrix": value_matrix,
        "mean_value_per_prompt": mean_value,
        "primary_ci_lower_per_prompt": primary_ci[0],
        "primary_ci_upper_per_prompt": primary_ci[1],
        "sensitivity_ci_lower_per_prompt": sensitivity_ci[0],
        "sensitivity_ci_upper_per_prompt": sensitivity_ci[1],
        "panel_library_indices": np.asarray(
            [panel["library_index"] for panel in plan["panels"]], dtype=int),
        "panel_fixed_teaching_indices": np.asarray(
            [panel["fixed_teaching_indices"] for panel in plan["panels"]], dtype=int),
        "panel_state_matrix": panel_state_matrix,
    }
    if state_values_by_panel is not None:
        per_prompt_table["state_values_by_panel"] = np.vstack(state_values_by_panel)

    certificate = {
        "schema": SAMPLED_VALUE_SCHEMA,
        "target": {
            "metric_key": target_metric_key,
            "description": str(entry["target_description"]),
            "bootstrap_sha256": str(target_bootstrap["sha256"]),
        },
        "executor": {
            "model": str(target_bootstrap["executor_model"]),
            "revision": str(target_bootstrap["executor_model_revision"]),
            "readout_id": str(target_bootstrap["readout_id"]),
        },
        "reconstructor": {
            "model": str(reconstructor_model),
            "revision": str(reconstructor_revision),
            "choice_readout_id": str(getattr(reconstructor, "choice_readout_id", "unverified")),
        },
        "n_panels_R": int(n_panels),
        "n_perms": int(n_perms),
        "mcq_n_options": int(n_options),
        "alpha": float(alpha),
        "reconstruction_noun": noun,
        "reconstruction_max_chars": int(max_chars),
        "panel_plan_sha256": plan["library_sha256"],
        "panel_plan": plan,
        "codebook_manifest_sha256": str(codebook_manifest["manifest_sha256"]),
        "codebook_provenance": (dict(codebook_provenance)
                                if codebook_provenance is not None else None),
        "source_scored_pool": {
            "path": str(scored["path"]),
            "sha256": str(scored["sha256"]),
            "n_scored_prompts": int(len(scored["texts"])),
            "n_prompts_including_controls": int(n_prompts),
            "draw_order_source": draw_order_source,
        },
        "achieved_value": {
            "primary_95": {
                "ci_percentiles": list(PRIMARY_CI_PERCENTILES),
                **primary,
            },
            "sensitivity_90": {
                "ci_percentiles": list(SENSITIVITY_CI_PERCENTILES),
                **sensitivity,
            },
            "raw_best_mean_value": float(mean_value[raw_best_index]),
            "raw_best_mean_value_prompt_index": raw_best_index,
            "per_family_best_mean_value": per_family_best,
        },
        "blind_prior_summary": {
            "menu_option_descriptions_sha256": _payload_sha256({
                "option_descriptions": list(option_descriptions)}),
            "q0_target_probability": float(blind["q0_target_probability"]),
            "blind_headroom_one_minus_q0": float(blind["value_cap"]),
            "value_cap": value_cap,
            "maximum_option_probability": float(blind["maximum_option_probability"]),
            "normalized_entropy": float(blind["normalized_entropy"]),
            "blind_prior_is_panel_invariant": True,
            "per_panel": [
                {
                    "panel_index": column,
                    "library_index": panel["library_index"],
                    "q0_target_probability": float(blind["q0_target_probability"]),
                    "blind_headroom_one_minus_q0": float(blind["value_cap"]),
                    "value_cap": value_cap,
                }
                for column, panel in enumerate(plan["panels"])
            ],
        },
        "gain_bounds": gain_bounds,
        "state_capture_recapture": state_capture_recapture,
        "headline_gates": {
            "min_blind_headroom": HEADLINE_MIN_BLIND_HEADROOM,
            "max_achieved_value_ci_width": HEADLINE_MAX_ACHIEVED_VALUE_CI_WIDTH,
            "observed_blind_headroom": float(blind["value_cap"]),
            "observed_achieved_value_ci_width_primary_95": primary["achieved_value_ci_width"],
            "observed_achieved_value_ci_width_sensitivity_90": (
                sensitivity["achieved_value_ci_width"]),
            "headline_eligible_primary_95": primary["headline_eligible"],
            "headline_eligible_sensitivity_90": sensitivity["headline_eligible"],
        },
        "reporting": {
            "primary_95": {
                "value_status": primary["value_status"],
                "is_primary_certificate": True,
                "alpha": float(alpha),
                "ci_percentiles": list(PRIMARY_CI_PERCENTILES),
            },
            "sensitivity_90": {
                "value_status": sensitivity["value_status"],
                "is_primary_certificate": False,
                "reporting_tier": "secondary_90_percent_sensitivity",
                "ci_percentiles": list(SENSITIVITY_CI_PERCENTILES),
            },
        },
        "calibration_controls": {
            "planted_control": bool(planted_control),
            "degenerate_control": bool(degenerate_control),
            "calibration_run": bool(planted_control or degenerate_control),
            "planted_control_family": PLANTED_CONTROL_FAMILY,
            "degenerate_control_family": DEGENERATE_CONTROL_FAMILY,
        },
        "premises": {
            "resampled_teaching_panels_from_frozen_design_data_only": True,
            "panels_built_before_prompt_search": True,
            "uses_external_labels": False,
            "uses_candidate_prompt_behavior_for_panel_selection": False,
            "value_is_function_of_fixed_binary_teaching_transcript_per_panel": True,
            "blind_prior_computed_once_menu_level_and_panel_invariant": True,
            "controls_ride_the_same_instrument_when_calibration_run": bool(
                planted_control or degenerate_control),
            "achieved_value_is_a_lower_bound_not_a_prompt_space_upper_bound": True,
        },
    }
    certificate["certificate_sha256"] = _payload_sha256(certificate)
    return {"certificate": certificate, "per_prompt_table": per_prompt_table}


def _gain_bounds(
    *, mean_value: np.ndarray, families: Sequence[str], family_indices: Mapping[str, list[int]],
    value_cap: float, alpha: float, horizons: Sequence[int],
) -> dict:
    """DKW expected-best value and gain bounds on the per-family mean-value distribution."""
    observed_best = float(np.max(mean_value)) if len(mean_value) else 0.0
    marks = {
        family: np.asarray(mean_value[indices], dtype=float)
        for family, indices in family_indices.items()
    }
    sizes = {family: len(indices) for family, indices in family_indices.items()}
    n_families = len(marks)
    pooled_component_alpha = float(alpha) / max(1, n_families)
    per_horizon: dict[str, dict] = {}
    for horizon in horizons:
        per_family: dict[str, dict] = {}
        for family, family_marks in marks.items():
            single_horizon = {family: int(horizon)}
            upper, upper_eps = dkw_expected_max_upper(
                {family: family_marks}, single_horizon, float(value_cap), float(alpha))
            lower, _ = dkw_expected_max_lower(
                {family: family_marks}, single_horizon, float(value_cap), float(alpha))
            per_family[family] = {
                "family_horizon": int(horizon),
                "component_alpha": float(alpha),
                "expected_best_value_upper": float(upper),
                "expected_best_value_lower": float(lower),
                "expected_best_gain_upper": float(max(0.0, upper - observed_best)),
                "expected_best_gain_lower": float(lower - observed_best),
                "dkw_epsilon": float(upper_eps[family]),
            }
        allocation = _allocate_horizon(sizes, int(horizon))
        pooled_upper, _ = dkw_expected_max_upper(
            marks, allocation, float(value_cap), pooled_component_alpha)
        pooled_lower, _ = dkw_expected_max_lower(
            marks, allocation, float(value_cap), pooled_component_alpha)
        per_horizon[str(int(horizon))] = {
            "per_family": per_family,
            "pooled": {
                "horizon_allocation": {family: int(count) for family, count in allocation.items()},
                "component_alpha": pooled_component_alpha,
                "expected_best_value_upper": float(pooled_upper),
                "expected_best_value_lower": float(pooled_lower),
                "expected_best_gain_upper": float(max(0.0, pooled_upper - observed_best)),
                "expected_best_gain_lower": float(pooled_lower - observed_best),
            },
        }
    return {
        "mark_name": "per_prompt_mean_reconstruction_mcq_value",
        "b_cap_value": float(value_cap),
        "observed_best_mean_value": observed_best,
        "per_horizon": per_horizon,
    }


def write_sampled_value_certificate(out_dir: str | Path, result: Mapping[str, object]) -> dict:
    """Immutably persist the certificate JSON and per-prompt NPZ (fail closed on overwrite)."""
    out_dir = Path(out_dir)
    certificate_path = out_dir / "certificate.json"
    table_path = out_dir / "per_prompt_values.npz"
    if certificate_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable certificate {certificate_path}")
    if table_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable value table {table_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = dict(result["per_prompt_table"])
    table_tmp = table_path.with_name(f".{table_path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(table_tmp, **table)
        with table_tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(table_tmp, table_path)
    finally:
        if table_tmp.exists():
            table_tmp.unlink()
    certificate = dict(result["certificate"])
    certificate["per_prompt_table_path"] = str(table_path)
    certificate["per_prompt_table_sha256"] = _file_sha256(table_path)
    certificate.pop("certificate_sha256", None)
    certificate["certificate_sha256"] = _payload_sha256(certificate)
    certificate_tmp = certificate_path.with_name(
        f".{certificate_path.name}.tmp-{os.getpid()}.json")
    try:
        certificate_tmp.write_text(json.dumps(certificate, indent=2), encoding="utf-8")
        with certificate_tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(certificate_tmp, certificate_path)
    finally:
        if certificate_tmp.exists():
            certificate_tmp.unlink()
    return certificate


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_metric_inputs(assets_root: Path, metric_argument: str, *, task: str) -> dict:
    """Resolve one ``--metrics`` entry to a codebook manifest and scored pool.

    Two layouts are auto-detected and both are fail-closed:

    - simple: ``<assets-root>/<metric>/codebook.json`` + ``<assets-root>/<metric>/pool.npz``;
    - production (mining-loop consolidation store): ``<assets-root>/mcq_codebooks/<task>.json``
      + ``<assets-root>/<metric>/historical/scored.npz``, with option bootstraps under
      ``<assets-root>/mcq_codebook_candidates/<key>/bootstrap/scored.npz``.

    A ``.npz`` argument binds its own pool: ``.../historical/scored.npz`` selects the
    production layout (metric key = grandparent directory name); any other pool selects
    the simple layout via its sibling ``codebook.json``. A bare metric key with BOTH
    layouts complete is ambiguous and refuses to guess.
    """
    candidate = Path(metric_argument)
    if candidate.suffix == ".npz":
        pool_path = (candidate if candidate.is_absolute() else assets_root / candidate).resolve()
        if not pool_path.exists():
            raise FileNotFoundError(
                f"metric {metric_argument!r} lacks a scored pool at {pool_path}")
        if pool_path.parent.name == "historical":
            metric_key = pool_path.parent.parent.name
            codebook_path = (assets_root / "mcq_codebooks" / f"{task}.json").resolve()
            layout = "production"
        else:
            metric_key = pool_path.parent.name
            codebook_path = pool_path.parent / "codebook.json"
            layout = "simple"
        if not codebook_path.exists():
            raise FileNotFoundError(
                f"metric {metric_argument!r} lacks a codebook at {codebook_path}")
        return {"metric_key": metric_key, "codebook_path": codebook_path,
                "pool_path": pool_path, "layout": layout}

    metric_directory = (assets_root / metric_argument).resolve()
    simple_codebook = metric_directory / "codebook.json"
    simple_pool = metric_directory / "pool.npz"
    production_codebook = (assets_root / "mcq_codebooks" / f"{task}.json").resolve()
    production_pool = metric_directory / "historical" / "scored.npz"
    simple_complete = simple_codebook.exists() and simple_pool.exists()
    production_complete = production_codebook.exists() and production_pool.exists()
    if simple_complete and production_complete:
        raise RuntimeError(
            f"metric {metric_argument!r} matches BOTH the simple and production layouts "
            f"under {assets_root}; refusing to guess")
    if simple_complete:
        return {"metric_key": metric_argument, "codebook_path": simple_codebook,
                "pool_path": simple_pool, "layout": "simple"}
    if production_complete:
        return {"metric_key": metric_argument, "codebook_path": production_codebook,
                "pool_path": production_pool, "layout": "production"}
    raise FileNotFoundError(
        f"metric {metric_argument!r} matches neither layout under {assets_root}: "
        f"simple needs {simple_codebook} + {simple_pool}; "
        f"production needs {production_codebook} + {production_pool}")


def _load_codebook(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_codebook_manifest(manifest)
    return manifest


def _load_production_codebook(path: str | Path, *, assets_root: Path) -> tuple[dict, dict]:
    """Load a mining-loop codebook manifest, remapping bootstrap paths fail-closed.

    Production manifests record absolute bootstrap paths from the machine that froze
    them. Every remapped local file must hash to the manifest's recorded
    ``bootstrap_sha256`` — content identity, not path identity, is the frozen contract.
    The stored manifest hash is verified BEFORE any remap; the hash recomputed after the
    remap covers an artifact whose only difference is verified-identical local content.
    """
    source = Path(path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    source_manifest_sha256 = str(manifest.get("manifest_sha256", ""))
    if source_manifest_sha256 != _payload_sha256(payload):
        raise ValueError(f"production codebook manifest {source} is invalid or mutated")
    n_remapped = 0
    for metric_key, metadata in payload.get("metrics", {}).items():
        recorded_sha256 = str(metadata.get("bootstrap_sha256", ""))
        candidates = [
            Path(str(metadata.get("bootstrap_path", ""))),
            assets_root / "mcq_codebook_candidates" / metric_key / "bootstrap" / "scored.npz",
            assets_root / metric_key / "bootstrap" / "scored.npz",
        ]
        resolved = None
        for candidate in candidates:
            if candidate.is_file() and _file_sha256(candidate) == recorded_sha256:
                resolved = candidate.resolve()
                break
        if resolved is None:
            raise FileNotFoundError(
                f"no local bootstrap for {metric_key!r} matches the frozen sha "
                f"{recorded_sha256[:12]}...; searched {[str(c) for c in candidates]}")
        if str(resolved) != str(metadata["bootstrap_path"]):
            metadata["bootstrap_path"] = str(resolved)
            n_remapped += 1
    remapped = {**payload, "manifest_sha256": _payload_sha256(payload)}
    validate_codebook_manifest(remapped)
    provenance = {
        "source_codebook_path": str(source),
        "source_codebook_manifest_sha256": source_manifest_sha256,
        "validated_codebook_manifest_sha256": remapped["manifest_sha256"],
        "n_bootstrap_paths_remapped": int(n_remapped),
        "bootstrap_content_sha256_verified": True,
    }
    return remapped, provenance


def _build_reconstructor(args) -> dict:
    from ..config import ImplementerConfig
    from ..vllm_backend import make_judge_backend, model_revision_id
    cfg = ImplementerConfig()
    if bool(args.fake_backends):
        cfg.vllm_fake = True
        revision = str(args.mcq_reconstructor)
    else:
        revision = model_revision_id(str(args.mcq_reconstructor))
    reconstructor = make_judge_backend(str(args.mcq_reconstructor), cfg, 0.0)
    return {"reconstructor": reconstructor, "revision": revision}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True,
                        help="consolidation store or run root with per-metric codebook.json/pool.npz")
    parser.add_argument("--task", required=True)
    parser.add_argument("--metrics", nargs="+", required=True,
                        help="metric keys or scored-pool .npz paths under --assets-root")
    parser.add_argument("--out-root", required=True,
                        help="output root; must not already exist (fail-closed)")
    parser.add_argument("--n-panels", type=int, default=12)
    parser.add_argument("--n-perms", type=int, default=8)
    parser.add_argument("--mcq-reconstructor", default="google/gemma-4-31b-it")
    parser.add_argument("--mcq-n-options", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--horizons", default="100,300")
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--fake-backends", action="store_true",
                        help="use the deterministic CPU FakeVLLM reconstructor (tests/dry-run)")
    parser.add_argument("--planted-control", action="store_true",
                        help="append a recoverable calibration prompt to every metric pool")
    parser.add_argument("--degenerate-control", action="store_true",
                        help="append a constant-verdict calibration prompt to every metric pool")
    parser.add_argument("--skip-unseen-state-cap", action="store_true",
                        help="skip the exhaustive 256-state pricing (on by default); the "
                             "capture-recapture gain bounds require it and are then omitted")
    return parser


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    horizons = [int(value) for value in str(args.horizons).split(",") if value.strip()]
    if not horizons:
        raise ValueError("--horizons must contain at least one positive integer")
    assets_root = Path(args.assets_root).resolve()
    if not assets_root.exists():
        raise FileNotFoundError(f"assets root {assets_root} does not exist")
    out_root = Path(args.out_root)
    if out_root.exists():
        raise FileExistsError(f"refusing to write into an existing out-root {out_root}")
    resolved = [_resolve_metric_inputs(assets_root, metric, task=str(args.task))
                for metric in args.metrics]
    keys = [item["metric_key"] for item in resolved]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate metric keys in --metrics")

    built = _build_reconstructor(args)
    reconstructor = built["reconstructor"]
    out_root.mkdir(parents=True, exist_ok=False)

    codebook_cache: dict[str, tuple[dict, dict | None]] = {}
    metric_records = []
    for item in resolved:
        cache_key = str(item["codebook_path"])
        if cache_key not in codebook_cache:
            if item["layout"] == "production":
                codebook_cache[cache_key] = _load_production_codebook(
                    item["codebook_path"], assets_root=assets_root)
            else:
                codebook_cache[cache_key] = (_load_codebook(item["codebook_path"]), None)
        codebook_manifest, provenance = codebook_cache[cache_key]
        result = certify_sampled_value(
            reconstructor,
            codebook_manifest=codebook_manifest,
            target_metric_key=item["metric_key"],
            scored_pool_path=item["pool_path"],
            n_panels=int(args.n_panels),
            n_perms=int(args.n_perms),
            mcq_n_options=int(args.mcq_n_options),
            alpha=float(args.alpha),
            horizons=horizons,
            reconstructor_model=str(args.mcq_reconstructor),
            reconstructor_revision=str(built["revision"]),
            query_batch_size=int(args.query_batch_size),
            planted_control=bool(args.planted_control),
            degenerate_control=bool(args.degenerate_control),
            with_unseen_state_cap=not bool(args.skip_unseen_state_cap),
            codebook_provenance=provenance,
        )
        certificate = write_sampled_value_certificate(out_root / item["metric_key"], result)
        metric_records.append({
            "metric_key": item["metric_key"],
            "assets_layout": item["layout"],
            "codebook_provenance": provenance,
            "panel_plan_sha256": certificate["panel_plan_sha256"],
            "certificate_sha256": certificate["certificate_sha256"],
            "primary_value_status": certificate["reporting"]["primary_95"]["value_status"],
        })
        print(f"  {item['metric_key']}: "
              f"{certificate['reporting']['primary_95']['value_status']} "
              f"achieved={certificate['achieved_value']['primary_95']['achieved_value']:.4f}",
              flush=True)

    run_manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "task": str(args.task),
        "assets_root": str(assets_root),
        "n_panels_R": int(args.n_panels),
        "n_perms": int(args.n_perms),
        "mcq_n_options": int(args.mcq_n_options),
        "alpha": float(args.alpha),
        "horizons": horizons,
        "reconstructor": {
            "model": str(args.mcq_reconstructor),
            "revision": str(built["revision"]),
            "choice_readout_id": str(getattr(reconstructor, "choice_readout_id", "unverified")),
            "fake_backends": bool(args.fake_backends),
        },
        "metrics": metric_records,
    }
    run_manifest["run_panel_plan_sha256"] = _payload_sha256({
        "n_panels_R": int(args.n_panels),
        "n_perms": int(args.n_perms),
        "reconstructor_model": str(args.mcq_reconstructor),
        "reconstructor_revision": str(built["revision"]),
        "per_metric_panel_plan_sha256": {
            record["metric_key"]: record["panel_plan_sha256"] for record in metric_records
        },
    })
    manifest_path = out_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return 0


# ======================================================================================
# v13.1 shared finite-pool engine
# ======================================================================================

def _stable_digest(*parts: object) -> str:
    return hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def _stable_diversity_order(
    indices: Sequence[int], behavior_columns: np.ndarray, *, salt: str,
) -> list[int]:
    """Interleave rare behavior patterns, with stable hashes for every tie.

    The behavior columns are canonical bootstrap/executor behaviors only.  Candidate
    prompt strings and candidate signatures never enter this design-time ordering.
    """
    values = np.asarray(behavior_columns, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != len(indices):
        raise ValueError("behavior columns must align with the proposed design indices")
    groups: dict[bytes, list[int]] = {}
    for position, index in enumerate(indices):
        pattern = values[:, position].tobytes()
        groups.setdefault(pattern, []).append(int(index))
    for pattern, members in groups.items():
        members.sort(key=lambda index: _stable_digest(salt, pattern.hex(), index))
    ordered_patterns = sorted(
        groups,
        key=lambda pattern: (
            len(groups[pattern]), _stable_digest(salt, "pattern", pattern.hex())
        ),
    )
    ordered: list[int] = []
    while ordered_patterns:
        next_patterns = []
        for pattern in ordered_patterns:
            members = groups[pattern]
            if members:
                ordered.append(members.pop(0))
            if members:
                next_patterns.append(pattern)
        ordered_patterns = next_patterns
    return ordered


def _pool_positive_quotas(n_positive: int, n_negative: int) -> list[int]:
    """Allocate the available target prevalence evenly without a balance gate."""
    total = int(n_positive) + int(n_negative)
    required = N_DESIGN_POOLS * POOL_SIZE
    if total < required:
        raise ValueError("the design split has fewer than 72 target verdicts")
    desired_positive = int(round(required * int(n_positive) / total))
    # Feasible prevalence-preserving sample: use every minority item when necessary,
    # but never reject a metric merely because its canonical executor is imbalanced.
    desired_positive = min(int(n_positive), max(required - int(n_negative), desired_positive))
    quotas = [desired_positive // N_DESIGN_POOLS] * N_DESIGN_POOLS
    for pool_index in range(desired_positive % N_DESIGN_POOLS):
        quotas[pool_index] += 1
    if any(value < 0 or value > POOL_SIZE for value in quotas):
        raise RuntimeError("pool target-stratification quota is infeasible")
    return quotas


def _ordered_panel(
    selected: Sequence[int], target_by_index: Mapping[int, int], *, salt: str,
) -> list[int]:
    by_label = {
        label: sorted(
            [int(index) for index in selected if int(target_by_index[int(index)]) == label],
            key=lambda index: _stable_digest(salt, label, index),
        )
        for label in (0, 1)
    }
    first = min((0, 1), key=lambda label: (-len(by_label[label]), label))
    ordered: list[int] = []
    while by_label[0] or by_label[1]:
        for label in (first, 1 - first):
            if by_label[label]:
                ordered.append(by_label[label].pop(0))
    return ordered


def _panel_family(
    pool_indices: Sequence[int], target_by_index: Mapping[int, int], *, channel: str,
    panel_size: int, n_panels: int, salt: str,
) -> list[dict]:
    combinations = []
    for selected in itertools.combinations([int(index) for index in pool_indices], panel_size):
        n_positive = sum(int(target_by_index[index]) for index in selected)
        balance_error = abs(n_positive - panel_size / 2.0)
        identity = _stable_digest(salt, channel, *selected)
        combinations.append((balance_error, identity, selected))
    combinations.sort(key=lambda row: (row[0], row[1]))
    if len(combinations) < n_panels:
        raise ValueError("pool is too small for the declared finite panel family")
    panels = []
    for panel_index, (_error, _identity, selected) in enumerate(combinations[:n_panels]):
        ordered = _ordered_panel(
            selected, target_by_index,
            salt=f"{salt}|{channel}|panel={panel_index}",
        )
        target_scores = [int(target_by_index[index]) for index in ordered]
        core = {
            "channel": str(channel),
            "panel_index_within_pool": int(panel_index),
            "fixed_teaching_indices": ordered,
            "fixed_teaching_target_scores": target_scores,
            "fixed_teaching_target_state": int("".join(map(str, target_scores)), 2),
            "selection_rule": (
                "minimum target-verdict imbalance then stable SHA; ordered by alternating "
                "target verdict with stable SHA ties"
            ),
        }
        panels.append({**core, "panel_sha256": _payload_sha256(core)})
    return panels


def build_value_bound_design_manifest(
    codebook_manifest: Mapping[str, object], *, target_metric_key: str,
    heldout_size: int = 60, salt: str = VALUE_BOUND_DESIGN_SALT,
) -> dict:
    """Freeze the shared six-pool MCQ/behavioral design from bootstrap data only."""
    validate_codebook_manifest(codebook_manifest)
    target_metric_key = str(target_metric_key)
    if heldout_size < 2:
        raise ValueError("heldout_size must be at least two")
    target = _bootstrap(codebook_manifest["metrics"][target_metric_key]["bootstrap_path"])
    probe_texts = [str(text) for text in target["probe_texts"]]
    n_probes = len(probe_texts)
    design_indices = [int(index) for index in codebook_manifest["design_indices"]]
    if len(design_indices) < N_DESIGN_POOLS * POOL_SIZE:
        raise ValueError("the frozen design split has fewer than the required 72 texts")
    if len(set(design_indices)) != len(design_indices):
        raise ValueError("the frozen design split contains duplicate indices")

    # Behavioral diversity is defined by the task-wide canonical metric bank.
    behavior_rows = []
    for metric_key in sorted(codebook_manifest["metrics"]):
        view = _bootstrap(codebook_manifest["metrics"][metric_key]["bootstrap_path"])
        if (view["sha256"]
                != codebook_manifest["metrics"][metric_key]["bootstrap_sha256"]):
            raise ValueError(f"bootstrap changed after freezing for {metric_key}")
        if len(view["target"]) != n_probes:
            raise ValueError("task metric bootstraps do not share the probe panel")
        behavior_rows.append((np.asarray(view["target"])[design_indices] > 0.5).astype(np.uint8))
    behavior_matrix = np.vstack(behavior_rows)
    target_bits = (np.asarray(target["target"], dtype=float) > 0.5).astype(np.uint8)
    target_by_index = {index: int(target_bits[index]) for index in design_indices}
    positive = [index for index in design_indices if target_by_index[index] == 1]
    negative = [index for index in design_indices if target_by_index[index] == 0]
    quotas = _pool_positive_quotas(len(positive), len(negative))
    design_position = {index: position for position, index in enumerate(design_indices)}

    def diverse(indices: Sequence[int], label: int) -> list[int]:
        positions = [design_position[int(index)] for index in indices]
        return _stable_diversity_order(
            indices, behavior_matrix[:, positions], salt=f"{salt}|label={label}"
        )

    positive_order = diverse(positive, 1)
    negative_order = diverse(negative, 0)
    pools = []
    pos_cursor = neg_cursor = 0
    for pool_index, n_positive in enumerate(quotas):
        n_negative = POOL_SIZE - n_positive
        members = [
            *positive_order[pos_cursor:pos_cursor + n_positive],
            *negative_order[neg_cursor:neg_cursor + n_negative],
        ]
        pos_cursor += n_positive
        neg_cursor += n_negative
        members = sorted(
            members, key=lambda index: _stable_digest(salt, "pool", pool_index, index)
        )
        pool_core = {
            "pool_index": int(pool_index),
            "pool_id": f"pool_{pool_index + 1}",
            "indices": members,
            "target_scores": [int(target_by_index[index]) for index in members],
            "probe_text_sha256": [
                hashlib.sha256(probe_texts[index].encode("utf-8")).hexdigest()
                for index in members
            ],
        }
        pools.append({
            **pool_core,
            "pool_sha256": _payload_sha256(pool_core),
            "mcq_panels": _panel_family(
                members, target_by_index, channel="mcq", panel_size=MCQ_PANEL_SIZE,
                n_panels=MCQ_PANELS_PER_POOL, salt=f"{salt}|pool={pool_index}",
            ),
            "behavioral_panels": _panel_family(
                members, target_by_index, channel="behavioral",
                panel_size=BEHAVIORAL_PANEL_SIZE,
                n_panels=BEHAVIORAL_PANELS_PER_POOL,
                salt=f"{salt}|pool={pool_index}",
            ),
        })
    flat_pool_indices = [index for pool in pools for index in pool["indices"]]
    if len(flat_pool_indices) != 72 or len(set(flat_pool_indices)) != 72:
        raise RuntimeError("the six frozen pools are not disjoint 12-text sets")

    design_set = set(design_indices)
    heldout_candidates = [index for index in range(n_probes) if index not in design_set]
    heldout_candidates.sort(
        key=lambda index: _stable_digest(salt, "heldout", index, probe_texts[index])
    )
    if len(heldout_candidates) < heldout_size:
        raise ValueError("the non-design split is too small for the frozen held-out set")
    heldout_indices = heldout_candidates[:int(heldout_size)]
    pool_membership = {
        int(index): str(pool["pool_id"]) for pool in pools for index in pool["indices"]
    }
    heldout_set = set(heldout_indices)
    split_membership = []
    for index in range(n_probes):
        if index in pool_membership:
            split_membership.append(pool_membership[index])
        elif index in design_set:
            split_membership.append("design_unused")
        elif index in heldout_set:
            split_membership.append("heldout_H")
        else:
            split_membership.append("evaluation_unused")
    permutation_design = mcq_option_order_design(
        int(codebook_manifest["n_options"]), 8
    )
    core = {
        "schema": VALUE_BOUND_DESIGN_SCHEMA,
        "release": VALUE_BOUND_RELEASE,
        "salt": str(salt),
        "target_metric_key": target_metric_key,
        "probe_sha256": str(codebook_manifest["probe_sha256"]),
        "n_probes": int(n_probes),
        "probe_text_sha256": [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in probe_texts
        ],
        "executor": {
            "model": str(target["executor_model"]),
            "revision": str(target["executor_model_revision"]),
            "readout_id": str(target["readout_id"]),
        },
        "codebook_manifest_sha256": str(codebook_manifest["manifest_sha256"]),
        "pools": pools,
        "heldout": {
            "name": "H",
            "indices": heldout_indices,
            "target_scores": target_bits[heldout_indices].astype(int).tolist(),
            "probe_text_sha256": [
                hashlib.sha256(probe_texts[index].encode("utf-8")).hexdigest()
                for index in heldout_indices
            ],
        },
        "split_membership": split_membership,
        "mcq_permutation_design": permutation_design,
        "tiers": {
            "A": {
                "active_pool_ids": [pool["pool_id"] for pool in pools],
                "mcq_panels_per_pool": MCQ_PANELS_PER_POOL,
                "behavioral_panels_per_pool": BEHAVIORAL_PANELS_PER_POOL,
            },
            "B": {
                "active_pool_ids": [pool["pool_id"] for pool in pools[:3]],
                "mcq_panels_per_pool": 8,
                "behavioral_panels_per_pool": BEHAVIORAL_PANELS_PER_POOL,
            },
        },
        "selection_contract": {
            "pools_built_from_design_split_only": True,
            "uses_canonical_executor_behaviors_for_stratification": True,
            "uses_candidate_prompt_text_or_behavior": False,
            "uses_external_labels": False,
            "pools_disjoint": True,
            "heldout_disjoint_from_all_pools": True,
            "target_verdict_balance_is_diagnostic_not_a_gate": True,
        },
        "pool_target_stratification": {
            "design_positive_count": int(len(positive)),
            "design_negative_count": int(len(negative)),
            "selected_positive_quotas": list(map(int, quotas)),
            "selection_rule": (
                "prevalence-proportional feasible total, distributed evenly across pools; "
                "no minimum per-verdict qualification"
            ),
        },
    }
    return {**core, "design_manifest_sha256": _payload_sha256(core)}


def active_panel_design(
    design_manifest: Mapping[str, object], *, channel: str, tier: str,
) -> dict:
    if design_manifest.get("schema") != VALUE_BOUND_DESIGN_SCHEMA:
        raise ValueError("unexpected v13.1 design schema")
    tier = str(tier).upper()
    if tier not in ("A", "B"):
        raise ValueError("tier must be A or B")
    if channel not in ("mcq", "behavioral"):
        raise ValueError("channel must be mcq or behavioral")
    tier_plan = design_manifest["tiers"][tier]
    active_ids = set(tier_plan["active_pool_ids"])
    panel_field = "mcq_panels" if channel == "mcq" else "behavioral_panels"
    count_field = (
        "mcq_panels_per_pool" if channel == "mcq" else "behavioral_panels_per_pool"
    )
    panels = []
    pools = []
    for pool in design_manifest["pools"]:
        if pool["pool_id"] not in active_ids:
            continue
        selected = list(pool[panel_field])[:int(tier_plan[count_field])]
        pool_position = len(pools)
        pools.append(pool)
        for panel in selected:
            panels.append({**panel, "pool_position": pool_position, "pool_id": pool["pool_id"]})
    return {"tier": tier, "channel": channel, "pools": pools, "panels": panels}


def _pool_pattern_integers(signatures: np.ndarray, pool_indices: Sequence[int]) -> np.ndarray:
    rows = np.asarray(signatures, dtype=float)
    bits = (rows[:, np.asarray(pool_indices, dtype=int)] > 0.5).astype(np.int64)
    weights = 1 << np.arange(POOL_SIZE - 1, -1, -1, dtype=np.int64)
    return bits @ weights


def enumerate_exact_pool_values(
    design_manifest: Mapping[str, object], *, channel: str, tier: str,
    state_values: np.ndarray, signatures: np.ndarray,
) -> dict:
    """Enumerate every 12-bit pattern and aggregate exact finite-family values."""
    active = active_panel_design(design_manifest, channel=channel, tier=tier)
    tables = np.asarray(state_values, dtype=float)
    n_states = 256 if channel == "mcq" else 64
    if tables.shape != (len(active["panels"]), n_states) or np.any(~np.isfinite(tables)):
        raise ValueError("state tables are incomplete, non-finite, or misaligned to the design")
    rows = np.asarray(signatures, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != int(design_manifest["n_probes"]):
        raise ValueError("candidate signatures do not align with the frozen probe panel")
    all_pool_bits = _binary_state_rows(POOL_SIZE).astype(np.int64)
    panel_size = MCQ_PANEL_SIZE if channel == "mcq" else BEHAVIORAL_PANEL_SIZE
    panel_weights = 1 << np.arange(panel_size - 1, -1, -1, dtype=np.int64)
    pool_pattern_values = []
    prompt_pool_values = []
    prompt_panel_values = np.empty((len(rows), len(active["panels"])), dtype=float)
    pool_caps = []
    pool_achieved = []
    panel_cursor = 0
    for pool_position, pool in enumerate(active["pools"]):
        pool_indices = [int(index) for index in pool["indices"]]
        local_by_global = {index: position for position, index in enumerate(pool_indices)}
        pool_panels = [
            (panel_index, panel) for panel_index, panel in enumerate(active["panels"])
            if int(panel["pool_position"]) == pool_position
        ]
        pattern_panel_values = []
        prompt_patterns = _pool_pattern_integers(rows, pool_indices)
        for panel_index, panel in pool_panels:
            local = np.asarray([
                local_by_global[int(index)] for index in panel["fixed_teaching_indices"]
            ], dtype=int)
            states = all_pool_bits[:, local] @ panel_weights
            values = tables[panel_index, states]
            pattern_panel_values.append(values)
            prompt_panel_values[:, panel_index] = tables[
                panel_index, states[prompt_patterns]
            ]
            panel_cursor += 1
        pool_values = np.mean(np.vstack(pattern_panel_values), axis=0)
        prompt_values = pool_values[prompt_patterns]
        pool_pattern_values.append(pool_values)
        prompt_pool_values.append(prompt_values)
        pool_caps.append(float(np.max(pool_values)))
        pool_achieved.append(float(np.max(prompt_values)))
    if panel_cursor != len(active["panels"]):
        raise RuntimeError("not every declared panel was aggregated exactly once")
    pool_pattern_matrix = np.vstack(pool_pattern_values)
    prompt_pool_matrix = np.vstack(prompt_pool_values).T
    mean_prompt_value = np.mean(prompt_pool_matrix, axis=1)
    achieved_index = int(np.argmax(mean_prompt_value))
    achieved = float(mean_prompt_value[achieved_index])
    exact_cap = float(np.mean(pool_caps))
    if exact_cap + 1e-12 < float(np.max(mean_prompt_value)):
        raise RuntimeError("exact structural cap fell below an observed prompt value")
    achieved_pool_values = prompt_pool_matrix[achieved_index]
    within_pool_variance = []
    for pool_position in range(len(active["pools"])):
        panel_columns = [
            index for index, panel in enumerate(active["panels"])
            if int(panel["pool_position"]) == pool_position
        ]
        within_pool_variance.append(float(np.var(
            prompt_panel_values[achieved_index, panel_columns], ddof=0
        )))
    return {
        "active_design": active,
        "pool_pattern_values": pool_pattern_matrix,
        "pool_caps": np.asarray(pool_caps, dtype=float),
        "prompt_panel_values": prompt_panel_values,
        "prompt_pool_values": prompt_pool_matrix,
        "mean_prompt_value": mean_prompt_value,
        "per_pool_achieved": np.asarray(pool_achieved, dtype=float),
        "achieved_value": achieved,
        "achieved_prompt_index": achieved_index,
        "exact_structural_cap": exact_cap,
        "exact_structural_gap": float(max(0.0, exact_cap - achieved)),
        "worst_pool_achieved_value": float(np.min(achieved_pool_values)),
        "achieved_pool_range": float(np.ptp(achieved_pool_values)),
        "pool_variance": float(np.var(achieved_pool_values, ddof=0)),
        "within_pool_panel_variance": np.asarray(within_pool_variance, dtype=float),
        "mean_within_pool_panel_variance": float(np.mean(within_pool_variance)),
        "recombination_slack": float(max(
            0.0, np.mean(pool_achieved) - achieved
        )),
    }


def _stream_value_marks(
    signatures: np.ndarray, active: Mapping[str, object],
    pool_pattern_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    patterns = np.column_stack([
        _pool_pattern_integers(signatures, pool["indices"])
        for pool in active["pools"]
    ])
    per_pool = np.column_stack([
        pool_pattern_values[pool_index, patterns[:, pool_index]]
        for pool_index in range(patterns.shape[1])
    ])
    return patterns, np.mean(per_pool, axis=1)


def fixed_prefix_capture_recapture(
    aggregation: Mapping[str, object], *, process_streams: Sequence[Mapping[str, object]] | None,
    alpha: float = 0.05, horizons: Sequence[int] = (100, 300),
) -> dict:
    """CP/DKW bounds from a fixed discovery prefix and never-absorbed audit suffix.

    Every audit draw is compared to the same frozen prefix.  The audit indicators are
    therefore ordinary Bernoulli observations conditional on that prefix; the prefix is
    never updated while walking the suffix.
    """
    if not process_streams:
        return {
            "available": False,
            "reason": (
                "legacy candidate bank has no frozen iid discovery-prefix / "
                "never-absorbed-audit provenance"
            ),
            "exact_all_prompt_cap_and_achieved_gap_remain_available": True,
        }
    streams = []
    n_probes = None
    for raw in process_streams:
        prefix = np.asarray(raw["discovery_prefix_signatures"], dtype=float)
        audit = np.asarray(raw["audit_suffix_signatures"], dtype=float)
        if (prefix.ndim != 2 or audit.ndim != 2 or prefix.shape[1] != audit.shape[1]
                or len(prefix) == 0 or len(audit) == 0
                or np.any(~np.isfinite(prefix)) or np.any(~np.isfinite(audit))):
            raise ValueError("capture/recapture streams need nonempty aligned finite matrices")
        if n_probes is None:
            n_probes = prefix.shape[1]
        if prefix.shape[1] != n_probes:
            raise ValueError("capture/recapture streams use different probe panels")
        streams.append({
            "family": str(raw["family"]),
            "prefix": prefix,
            "audit": audit,
            "provenance": dict(raw.get("provenance") or {}),
        })
    if len({stream["family"] for stream in streams}) != len(streams):
        raise ValueError("capture/recapture family names must be unique")
    active = aggregation["active_design"]
    pool_values = np.asarray(aggregation["pool_pattern_values"], dtype=float)
    n_pools = len(active["pools"])
    component_alpha = float(alpha) / (len(streams) * (n_pools + 1))
    family_rows = {}
    audit_marks = {}
    for stream in streams:
        prefix_patterns, _ = _stream_value_marks(stream["prefix"], active, pool_values)
        audit_patterns, marks = _stream_value_marks(stream["audit"], active, pool_values)
        audit_marks[stream["family"]] = marks
        per_pool = []
        for pool_index in range(n_pools):
            seen = set(map(int, prefix_patterns[:, pool_index]))
            novelty = np.asarray([
                int(pattern) not in seen for pattern in audit_patterns[:, pool_index]
            ], dtype=bool)
            z = int(np.sum(novelty))
            n = int(len(novelty))
            per_pool.append({
                "pool_id": active["pools"][pool_index]["pool_id"],
                "n_discovery_prefix": int(len(prefix_patterns)),
                "n_never_absorbed_audit": n,
                "n_audit_draws_unseen_relative_to_fixed_prefix": z,
                "missing_mass_upper": float(clopper_pearson_upper(
                    z, n, component_alpha
                )),
                "component_alpha": component_alpha,
            })
        prefix_joint = {tuple(map(int, row)) for row in prefix_patterns}
        joint_novelty = np.asarray([
            tuple(map(int, row)) not in prefix_joint for row in audit_patterns
        ], dtype=bool)
        z_joint = int(np.sum(joint_novelty))
        joint = {
            "n_discovery_prefix": int(len(prefix_patterns)),
            "n_never_absorbed_audit": int(len(joint_novelty)),
            "n_audit_draws_unseen_relative_to_fixed_prefix": z_joint,
            "missing_mass_upper": float(clopper_pearson_upper(
                z_joint, len(joint_novelty), component_alpha
            )),
            "component_alpha": component_alpha,
        }
        family_rows[stream["family"]] = {
            "per_pool": per_pool,
            "joint_pattern": joint,
            "provenance": stream["provenance"],
        }

    family_sizes = {
        stream["family"]: int(len(stream["prefix"])) for stream in streams
    }
    pool_caps = np.asarray(aggregation["pool_caps"], dtype=float)
    pool_achieved = np.asarray(aggregation["per_pool_achieved"], dtype=float)
    achieved = float(aggregation["achieved_value"])
    exact_cap = float(aggregation["exact_structural_cap"])
    slack = float(aggregation["recombination_slack"])
    horizon_rows = {}
    for horizon in map(int, horizons):
        allocation = _allocate_horizon(family_sizes, horizon)
        pool_gain_terms = []
        per_pool_horizon = []
        for pool_index in range(n_pools):
            probability_none = 1.0
            for family, future_n in allocation.items():
                u0 = family_rows[family]["per_pool"][pool_index]["missing_mass_upper"]
                probability_none *= (1.0 - float(u0)) ** int(future_n)
            probability_any = float(1.0 - probability_none)
            gain = probability_any * max(0.0, pool_caps[pool_index] - pool_achieved[pool_index])
            pool_gain_terms.append(gain)
            per_pool_horizon.append({
                "pool_id": active["pools"][pool_index]["pool_id"],
                "probability_any_prefix-unseen_pattern_upper": probability_any,
                "gain_upper": float(gain),
            })
        pool_decomposed_gain = float(np.mean(pool_gain_terms) + slack)
        joint_probability_none = 1.0
        for family, future_n in allocation.items():
            u0 = family_rows[family]["joint_pattern"]["missing_mass_upper"]
            joint_probability_none *= (1.0 - float(u0)) ** int(future_n)
        joint_probability_any = float(1.0 - joint_probability_none)
        joint_gain = float(joint_probability_any * max(0.0, exact_cap - achieved))
        dkw_upper, dkw_eps = dkw_expected_max_upper(
            audit_marks, allocation, exact_cap, float(alpha) / len(streams)
        )
        dkw_lower, _ = dkw_expected_max_lower(
            audit_marks, allocation, exact_cap, float(alpha) / len(streams)
        )
        dkw_gain_upper = float(max(0.0, dkw_upper - achieved))
        dkw_gain_lower = float(dkw_lower - achieved)
        horizon_rows[str(horizon)] = {
            "horizon_allocation": allocation,
            "pool_decomposed": {
                "per_pool": per_pool_horizon,
                "recombination_slack": slack,
                "gain_upper": pool_decomposed_gain,
            },
            "joint_pattern": {
                "probability_any_prefix-unseen_pattern_upper": joint_probability_any,
                "gain_upper": joint_gain,
            },
            "dkw_future_budget": {
                "expected_best_value_lower": float(dkw_lower),
                "expected_best_value_upper": float(dkw_upper),
                "expected_best_gain_lower": dkw_gain_lower,
                "expected_best_gain_upper": dkw_gain_upper,
                "epsilon_by_family": {
                    family: float(value) for family, value in dkw_eps.items()
                },
            },
            "headline_gain_upper_min_of_declared_bounds": float(min(
                pool_decomposed_gain, joint_gain, dkw_gain_upper
            )),
        }
    observed_audit_best = float(max(np.max(marks) for marks in audit_marks.values()))
    return {
        "available": True,
        "method": "fixed-prefix never-absorbed-audit CP plus DKW",
        "family_rows": family_rows,
        "horizons": horizon_rows,
        "observed_audit_best_value": observed_audit_best,
        "premises": {
            "audit_suffix_never_absorbed_into_discovery_prefix": True,
            "every_audit_indicator_uses_the_same_fixed_prefix": True,
            "sequential_first_contacts_not_treated_as_binomial": True,
            "iid_within_homogeneous_generator_family": True,
            "disjoint_pool_separability": True,
            "no_smoothness_submodularity_or_cross-family_independence_assumption": True,
        },
    }


def secondary_value_status(
    *, achieved: float, exact_cap: float, process_bounds: Mapping[str, object],
    epsilon: float,
) -> str:
    if exact_cap - achieved <= float(epsilon) + 1e-12:
        return "RESOLVED"
    if not process_bounds.get("available"):
        return "UNRESOLVED"
    horizon_rows = process_bounds.get("horizons") or {}
    if not horizon_rows:
        return "UNRESOLVED"
    largest_horizon = horizon_rows[sorted(horizon_rows, key=int)[-1]]
    if largest_horizon["headline_gain_upper_min_of_declared_bounds"] <= float(epsilon):
        return "PLATEAUED"
    if (float(process_bounds.get("observed_audit_best_value", achieved)) - achieved
            > float(epsilon)):
        return "RISING"
    return "UNRESOLVED"


def evaluate_mcq_state_tables_v13_1(
    reconstructor, *, codebook_manifest: Mapping[str, object],
    design_manifest: Mapping[str, object], target_metric_key: str, tier: str,
    constructor_revision: str, cache: ValueCache, query_batch_size: int = 512,
) -> dict:
    """Fill every declared MCQ state table with state-cell persistent caching."""
    menu = _codebook_menu(codebook_manifest, str(target_metric_key))
    option_descriptions = list(menu["option_descriptions"])
    option_ids = [str(target_metric_key), *menu["entry"]["distractor_metric_keys"]]
    menu_sha = _payload_sha256({
        "option_ids": option_ids, "option_descriptions": option_descriptions,
    })
    permutation_design = dict(design_manifest["mcq_permutation_design"])
    permutation_sha = _payload_sha256(permutation_design)
    blind_key = cache_key("mcq_blind", {
        "constructor_revision": str(constructor_revision),
        "menu_sha256": menu_sha,
        "permutation_design_sha256": permutation_sha,
    })
    blind_payload = cache.get(blind_key)
    if blind_payload is None:
        blind = _blind_menu_prior(
            reconstructor,
            noun=str(codebook_manifest["reconstruction_noun"]),
            option_descriptions=option_descriptions,
            n_perms=int(permutation_design["n_draws"]),
        )
        blind_payload = cache.put(blind_key, "mcq_blind", {
            "canonical_choice_probabilities": np.asarray(
                blind["canonical_choice_probabilities"], dtype=float
            ).tolist(),
            "q0_target_probability": float(blind["q0_target_probability"]),
            "value_cap": float(blind["value_cap"]),
            "maximum_option_probability": float(blind["maximum_option_probability"]),
            "normalized_entropy": float(blind["normalized_entropy"]),
        })
    blind_canonical = np.asarray(
        blind_payload["canonical_choice_probabilities"], dtype=float
    )
    active = active_panel_design(design_manifest, channel="mcq", tier=tier)
    target_bootstrap = menu["target_bootstrap"]
    probe_texts = list(target_bootstrap["probe_texts"])
    state_values = np.empty((len(active["panels"]), 256), dtype=float)
    raw_values = np.empty_like(state_values)
    shuffled_values = np.empty_like(state_values)
    annotation_accuracy = np.empty_like(state_values)
    cache_hits = cache_misses = 0
    for panel_position, panel in enumerate(active["panels"]):
        panel_indices = np.asarray(panel["fixed_teaching_indices"], dtype=int)
        keys = [cache_key("mcq_state", {
            "constructor_revision": str(constructor_revision),
            "menu_sha256": menu_sha,
            "panel_sha256": str(panel["panel_sha256"]),
            "state": state,
            "permutation_design_sha256": permutation_sha,
        }) for state in range(256)]
        payloads = [cache.get(key) for key in keys]
        missing = [state for state, payload in enumerate(payloads) if payload is None]
        cache_hits += 256 - len(missing)
        cache_misses += len(missing)
        if missing:
            bits = _binary_state_rows(MCQ_PANEL_SIZE)[missing]
            rows = np.zeros((len(missing), len(probe_texts)), dtype=float)
            rows[:, panel_indices] = bits
            details = mcq_logit_values_from_precomputed_behaviors(
                reconstructor,
                noun=str(codebook_manifest["reconstruction_noun"]),
                candidate_prompt_texts=[f"finite MCQ state {state:08b}" for state in missing],
                target_metric_id=str(target_metric_key),
                target_description=str(menu["entry"]["target_description"]),
                target_score_rows=rows,
                probe_texts=probe_texts,
                distractors=list(menu["distractors"]),
                design_indices=panel_indices,
                codebook_frozen_before_prompt_search=True,
                n_examples=MCQ_PANEL_SIZE,
                n_reconstruction_draws=int(permutation_design["n_draws"]),
                max_chars=int(codebook_manifest["reconstruction_max_chars"]),
                query_batch_size=int(query_batch_size),
                fixed_no_demo_canonical_probabilities=blind_canonical,
                fixed_teaching_panel=True,
            )
            for state, detail in zip(missing, details):
                expected_indices = panel_indices.astype(int).tolist()
                if detail["design"]["indices_in_prompt_order"] != expected_indices:
                    raise RuntimeError("MCQ query changed the frozen panel order")
                identification = detail["identification"]
                payloads[state] = cache.put(keys[state], "mcq_state", {
                    "value": float(detail["value_mark"]),
                    "raw_target_probability": float(detail["raw_target_option_probability"]),
                    "shuffled_target_probability": float(
                        identification["shuffled_label_score"]
                    ),
                    "annotation_accuracy": float(identification["identification_acc"]),
                    "panel_sha256": str(panel["panel_sha256"]),
                    "state": int(state),
                })
        for state, payload in enumerate(payloads):
            if payload is None:
                raise RuntimeError("MCQ cache failed to fill a declared state")
            state_values[panel_position, state] = float(payload["value"])
            raw_values[panel_position, state] = float(payload["raw_target_probability"])
            shuffled_values[panel_position, state] = float(
                payload["shuffled_target_probability"]
            )
            annotation_accuracy[panel_position, state] = float(
                payload["annotation_accuracy"]
            )
    if np.any(~np.isfinite(state_values)):
        raise RuntimeError("MCQ state tables contain non-finite cells")
    target_state_accuracy = [
        annotation_accuracy[index, int(panel["fixed_teaching_target_state"])]
        for index, panel in enumerate(active["panels"])
    ]
    return {
        "schema": VALUE_BOUND_STATE_SCHEMA,
        "channel": "mcq",
        "tier": str(tier).upper(),
        "active_design": active,
        "state_values": state_values,
        "raw_target_probabilities": raw_values,
        "shuffled_target_probabilities": shuffled_values,
        "annotation_accuracy": annotation_accuracy,
        "blind": blind_payload,
        "menu_sha256": menu_sha,
        "permutation_design_sha256": permutation_sha,
        "best_explanation_rate": float(np.mean(target_state_accuracy)),
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "non_disclosure": {
            "candidate_prompt_text_passed_to_query_builder": False,
            "queries_are_functions_only_of_panel_texts_labels_and_frozen_menu": True,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
