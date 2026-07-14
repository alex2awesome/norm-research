"""Qualification, freeze, and control-liveness contracts for CR-3 v14."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .v14_decoder_tuning import validate_shared_template
from .v14_panel_design import canonical_sha256


PREREG_SCHEMA = "cr3-v14-preregistration-v1"
QUALIFICATION_SCHEMA = "cr3-v14-decoder-qualification-v1"
SENTINEL_SCHEMA = "cr3-v14-sentinel-liveness-v1"
TEMPLATE_FREEZE_SCHEMA = "cr3-v14-template-freeze-v1"


def evaluate_decoder_qualification(
    rows: Sequence[Mapping[str, object]], *, minimum_positive: int = 4,
    shuffled_tolerance_bits: float = 0.01,
) -> dict:
    if len(rows) != 6:
        raise ValueError("decoder mini-qualification requires exactly six sentinel metrics")
    canonical = np.asarray([row["canonical_lift_bits"] for row in rows], dtype=float)
    shuffled = np.asarray([row["shuffled_lift_bits"] for row in rows], dtype=float)
    if np.any(~np.isfinite(canonical)) or np.any(~np.isfinite(shuffled)):
        raise ValueError("decoder mini-qualification contains non-finite values")
    positive = int(np.sum(canonical > 0.0))
    shuffled_mean = float(np.mean(np.abs(shuffled)))
    passed = positive >= int(minimum_positive) and shuffled_mean <= float(shuffled_tolerance_bits)
    return {
        "schema": QUALIFICATION_SCHEMA,
        "n_sentinels": 6,
        "n_positive_canonical": positive,
        "minimum_positive_required": int(minimum_positive),
        "mean_absolute_shuffled_lift_bits": shuffled_mean,
        "shuffled_tolerance_bits": float(shuffled_tolerance_bits),
        "passed": bool(passed),
        "rows": [dict(row) for row in rows],
    }


def choose_qualified_decoder(
    *, family: str, primary: Mapping[str, object], fallback: Mapping[str, object] | None,
) -> dict:
    """Apply the one-fallback-only policy without opening a model search loop."""
    if bool(primary.get("passed")):
        selected = dict(primary)
        role = "primary"
    elif fallback is not None and bool(fallback.get("passed")):
        selected = dict(fallback)
        role = "predeclared_same_lineage_fallback"
    else:
        raise RuntimeError(
            f"decoder family {family} failed its primary and one predeclared fallback"
        )
    return {
        "family": str(family), "selection_role": role,
        "model": str(selected["model"]), "revision": str(selected["revision"]),
        "qualification": selected,
        "additional_model_search_allowed": False,
    }


def evaluate_sentinel_liveness(rows: Sequence[Mapping[str, object]]) -> dict:
    """Block only structural failures and control-defined instrument death."""
    if not rows:
        raise ValueError("sentinel liveness needs at least one channel row")
    failures = []
    for row in rows:
        label = f"{row.get('metric_key')}:{row.get('channel')}:{row.get('arm')}"
        if not bool(row.get("structurally_valid", False)):
            failures.append({"row": label, "reason": "structural_failure"})
        planted = float(row.get("planted_positive_value", float("nan")))
        degenerate = float(row.get("degenerate_control_value", float("nan")))
        blind = float(row.get("blind_value", float("nan")))
        annotated = float(row.get("annotated_canonical_value", float("nan")))
        cap = float(row.get("cap", float("nan")))
        if not np.isfinite([planted, degenerate, blind, annotated, cap]).all():
            failures.append({"row": label, "reason": "non_finite_control"})
            continue
        if planted <= 0.0:
            failures.append({"row": label, "reason": "planted_positive_not_positive"})
        if degenerate > 1e-12:
            failures.append({"row": label, "reason": "degenerate_control_positive"})
        if blind >= annotated:
            failures.append({"row": label, "reason": "blind_annotated_control_inversion"})
    if all(float(row.get("cap", 0.0)) <= 1e-12 for row in rows):
        failures.append({"row": "all", "reason": "all_caps_zero"})
    return {
        "schema": SENTINEL_SCHEMA,
        "passed": not bool(failures),
        "fanout_blocked": bool(failures),
        "failures": failures,
        "scientific_weakness_is_not_a_gate": True,
        "rows": [dict(row) for row in rows],
    }


def build_production_freeze(
    *, design_index: Mapping[str, object], decoder_selections: Sequence[Mapping[str, object]],
    mcq_trace: Mapping[str, object], behavioral_unconstrained_trace: Mapping[str, object],
    behavioral_no_verbatim_trace: Mapping[str, object], forbidden_strings: Sequence[str],
    release_commit: str, out_path: str | Path, study_alpha: float = 0.05,
    n_certified_metrics: int = 35,
) -> dict:
    traces = {
        "mcq": dict(mcq_trace),
        "behavioral_unconstrained": dict(behavioral_unconstrained_trace),
        "behavioral_no_verbatim": dict(behavioral_no_verbatim_trace),
    }
    required = {
        "mcq": ("noun", "examples", "choices", "labels"),
        "behavioral_unconstrained": ("noun", "feature_table", "examples", "arm_instruction"),
        "behavioral_no_verbatim": ("noun", "feature_table", "examples", "arm_instruction"),
    }
    for name, trace in traces.items():
        validate_shared_template(
            str(trace["winner_template"]), forbidden_strings=forbidden_strings,
            required_fields=required[name],
        )
        if not bool(trace.get("shared_across_decoder_families", False)):
            raise ValueError(f"{name} trace is not a shared-template search")
    if len(decoder_selections) != 3 or {row["family"] for row in decoder_selections} != {
        "qwen", "llama", "mistral",
    }:
        raise ValueError("production freeze requires one qualified decoder per family")
    # Two horizons x three reported channel/arms x two instruments x all metrics.
    process_claims = int(n_certified_metrics) * 2 * 3 * 2
    cell_alpha = float(study_alpha) / process_claims
    untuned_mcq = str(mcq_trace.get("seed_template", ""))
    untuned_behavior = str(behavioral_unconstrained_trace.get("seed_template", ""))
    untuned_no_verbatim = str(behavioral_no_verbatim_trace.get("seed_template", ""))
    if not untuned_mcq or not untuned_behavior or not untuned_no_verbatim:
        raise ValueError("tuning traces must retain their untuned seed templates")
    core = {
        "schema": TEMPLATE_FREEZE_SCHEMA,
        "release": "v14.0",
        "release_commit": str(release_commit),
        "design_index_sha256": str(design_index["index_sha256"]),
        "decoder_panel": [dict(row) for row in decoder_selections],
        "instruments": {
            "untuned": {
                "mcq": untuned_mcq,
                "behavioral": {
                    "unconstrained": untuned_behavior,
                    "no_verbatim_examples": untuned_no_verbatim,
                },
            },
            "tuned": {
                "mcq": str(mcq_trace["winner_template"]),
                "behavioral": {
                    "unconstrained": str(behavioral_unconstrained_trace["winner_template"]),
                    "no_verbatim_examples": str(behavioral_no_verbatim_trace["winner_template"]),
                },
            },
        },
        "template_trace_sha256": {
            key: str(value["freeze_sha256"]) for key, value in traces.items()
        },
        "shared_across_metrics_and_decoder_families": True,
        "searched_per_family_variation": False,
        "mechanical_chat_formatting_only": True,
        "study_alpha": float(study_alpha),
        "process_bound_cell_alpha": cell_alpha,
        "alpha_scope": f"Bonferroni over {process_claims} metric/channel/instrument/horizon claims",
        "within_claim_alpha_split": "equal split between split-sample CP and DKW; record/rank is exact",
        "audit_budget_per_metric": 400,
        "future_horizons": [100, 300],
    }
    core["freeze_sha256"] = canonical_sha256(core)
    path = Path(out_path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(core, indent=2, ensure_ascii=False) + "\n")
    return core


def build_preregistration(
    *, template_freeze: Mapping[str, object], design_index: Mapping[str, object],
    metrics_manifest_sha256: str, out_path: str | Path,
) -> dict:
    core = {
        "schema": PREREG_SCHEMA,
        "release": "v14.0",
        "template_freeze_sha256": str(template_freeze["freeze_sha256"]),
        "design_index_sha256": str(design_index["index_sha256"]),
        "metrics_manifest_sha256": str(metrics_manifest_sha256),
        "executor": "meta-llama/Llama-3.1-8B-Instruct",
        "channels": ["mcq", "behavioral_unconstrained", "behavioral_no_verbatim"],
        "n_certified_metrics": 35,
        "audit_budget_per_metric": 400,
        "audit_families": ["phi4", "qwen14", "llama8"],
        "future_horizons": [100, 300],
        "valid_future_gain_bounds": ["record_rank", "split_sample_cp", "dkw"],
        "within_claim_alpha_split": "equal split between split-sample CP and DKW",
        "invalid_zero_hit_cp_retracted": True,
        "sk3_forbidden_physical_gpus": [1, 2, 3, 4],
        "post_freeze_scientific_iteration_allowed": False,
        "sentinel_blocks": ["structural_failure", "control_based_instrument_death"],
        "scientific_weakness_blocks": False,
    }
    core["preregistration_sha256"] = canonical_sha256(core)
    path = Path(out_path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(core, indent=2, ensure_ascii=False) + "\n")
    return core
