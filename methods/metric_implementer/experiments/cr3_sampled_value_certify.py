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
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..recon_channel import (
    mcq_logit_values_from_precomputed_behaviors,
    mcq_no_demo_choice_probabilities,
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

SAMPLED_VALUE_SCHEMA = "cr3-sampled-v13"
PANEL_PLAN_SCHEMA = "cr3-sampled-panel-plan-v13"
PER_PROMPT_TABLE_SCHEMA = "cr3-sampled-per-prompt-values-v13"
RUN_MANIFEST_SCHEMA = "cr3-sampled-run-manifest-v13"

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


if __name__ == "__main__":
    raise SystemExit(main())
