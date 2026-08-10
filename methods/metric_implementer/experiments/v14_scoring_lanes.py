"""Frozen FAST/CERT scoring-lane policies and promotion quarantine.

FAST rows are screening evidence.  CERT rows are a distinct remeasurement
population and are the only rows permitted in release artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .v14_panel_design import canonical_sha256
from .v14_value_bound import (
    plugin_binary_mutual_information, signatures_to_states, tie_class,
)


LANE_SCHEMA = "cr3-v14-scoring-lane-v1"
PROMOTION_SCHEMA = "cr3-v14-fast-to-cert-promotion-v1"
LANES = ("fast", "cert")

LANE_POLICIES = {
    "fast": {
        "claim_role": "screening_only",
        "panel_size": 6,
        "menu_permutations": 4,
        "state_scope": "observed_only",
        "reference": "frozen_executor_verdicts",
        "independent_reference": False,
        "exact_structural_cap": False,
        "cap_fields": ["target_entropy_level0", "record_rank"],
        "permutation_null_required": True,
        "preregistration_required": False,
        "release_eligible": False,
    },
    "cert": {
        "claim_role": "certification",
        "panel_size": "6_fanout_or_8_sentinel",
        "menu_permutations": "4_fanout_or_8_sentinel",
        "state_scope": "exhaustive_where_declared",
        "reference": "independent_majority_with_hidden_and_planted_anchors",
        "independent_reference": True,
        "exact_structural_cap": "where_enumerable_else_explicitly_unavailable",
        "cap_fields": ["exact_where_enumerable", "target_entropy_level0", "record_rank"],
        "permutation_null_required": True,
        "permutation_count": 10000,
        "miller_madow_required": True,
        "bootstrap_intervals_required": True,
        "preregistration_required": True,
        "release_eligible": True,
    },
}


def scoring_lane_policy(lane: str) -> dict:
    value = str(lane).lower()
    if value not in LANE_POLICIES:
        raise ValueError(f"unknown scoring lane {lane!r}; expected fast or cert")
    core = {"schema": LANE_SCHEMA, "lane": value, **LANE_POLICIES[value]}
    core["policy_sha256"] = canonical_sha256(core)
    return core


def validate_lane_policy(payload: Mapping[str, object]) -> None:
    expected = scoring_lane_policy(str(payload.get("lane", "")))
    if dict(payload) != expected:
        raise ValueError("scoring-lane policy differs from the frozen contract")


def assert_release_rows_are_cert(frame: pd.DataFrame) -> None:
    if "lane" not in frame.columns:
        raise ValueError("release results lack the scoring-lane column")
    lanes = set(frame["lane"].astype(str))
    if lanes != {"cert"}:
        raise ValueError(f"release artifacts may contain cert rows only; observed {sorted(lanes)}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_promotion_manifest(
    fast_results_path: str | Path, *, out_path: str | Path, run_sha: str,
    top_k_per_task: int, figure_metric_keys: Sequence[str] = (),
) -> dict:
    """Freeze deterministic FAST->CERT membership without copying scores.

    The source score is used to rank, but only selected identities and a hash of
    the complete screening table enter the manifest.  CERT measurements must be
    written beneath a different output root and are recomputed from scratch.
    """
    if int(top_k_per_task) < 0:
        raise ValueError("top_k_per_task must be nonnegative")
    source = Path(fast_results_path).resolve()
    frame = pd.read_parquet(source)
    required = {"lane", "task", "metric_key", "permutation_z_score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"FAST results lack columns {sorted(required - set(frame.columns))}")
    if set(frame["lane"].astype(str)) != {"fast"}:
        raise ValueError("promotion source must contain FAST rows only")
    if frame.duplicated(["metric_key"]).any():
        raise ValueError("promotion source must have one screening row per metric")
    if not np.all(np.isfinite(frame["permutation_z_score"].to_numpy(dtype=float))):
        raise ValueError("FAST promotion requires finite permutation z-scores")

    chosen: dict[str, set[str]] = {}
    for task, group in frame.groupby("task", sort=True):
        ordered = group.sort_values(
            ["permutation_z_score", "metric_key"], ascending=[False, True], kind="stable",
        )
        for key in ordered.head(int(top_k_per_task))["metric_key"].astype(str):
            chosen.setdefault(key, set()).add(f"top_{int(top_k_per_task)}:{task}")
    available = set(frame["metric_key"].astype(str))
    missing = sorted(set(map(str, figure_metric_keys)) - available)
    if missing:
        raise ValueError(f"planned-figure metrics absent from FAST population: {missing}")
    for key in map(str, figure_metric_keys):
        chosen.setdefault(key, set()).add("planned_figure")

    rows = []
    for key in sorted(chosen):
        source_row = frame.loc[frame.metric_key.astype(str) == key].iloc[0]
        rows.append({
            "metric_key": key, "task": str(source_row["task"]),
            "promotion_reasons": sorted(chosen[key]),
            "cert_measurement_mode": "fresh_from_scratch",
            "fast_values_may_not_be_copied": True,
        })
    manifest = {
        "schema": PROMOTION_SCHEMA,
        "run_sha": str(run_sha),
        "source_lane": "fast", "destination_lane": "cert",
        "source_results_path": str(source),
        "source_results_sha256": _file_sha256(source),
        "selection_rule": {
            "top_k_per_task_by_permutation_z": int(top_k_per_task),
            "include_planned_figure_metrics": True,
            "tie_break": "metric_key_ascending",
        },
        "selected": rows,
        "n_selected": len(rows),
        "remeasurement_from_scratch_required": True,
        "fast_artifact_reuse_forbidden": True,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_promotion_metric_keys(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    core = dict(payload)
    observed = str(core.pop("manifest_sha256", ""))
    if payload.get("schema") != PROMOTION_SCHEMA or observed != canonical_sha256(core):
        raise ValueError("invalid FAST->CERT promotion manifest")
    if not payload.get("remeasurement_from_scratch_required") or not payload.get(
        "fast_artifact_reuse_forbidden"
    ):
        raise ValueError("promotion manifest does not enforce population quarantine")
    return [str(row["metric_key"]) for row in payload["selected"]]


def _permutation_summary(observed: float, null: Sequence[float]) -> dict:
    values = np.asarray(null, dtype=float)
    if values.ndim != 1 or len(values) < 200 or np.any(~np.isfinite(values)):
        raise ValueError("FAST screening requires at least 200 finite null draws")
    sd = float(np.std(values, ddof=1))
    return {
        "permutation_count": len(values),
        "permutation_null_mean": float(np.mean(values)),
        "permutation_null_sd": sd,
        "permutation_z_score": (
            float((float(observed) - np.mean(values)) / sd) if sd > 0 else 0.0
        ),
        "permutation_percentile": float(
            (np.sum(values < float(observed)) + 0.5 * np.sum(values == float(observed)))
            / len(values)
        ),
        "permutation_p_greater_equal": float(
            (1 + np.sum(values >= float(observed))) / (len(values) + 1)
        ),
    }


def _observed_prompt_values(
    clipped_value: np.ndarray, prompt_codes: np.ndarray,
) -> np.ndarray:
    table = np.asarray(clipped_value, dtype=float)
    codes = np.asarray(prompt_codes, dtype=int)
    if table.ndim != 2 or codes.ndim != 2 or codes.shape[1] != table.shape[0]:
        raise ValueError("FAST state tables and prompt codes are not aligned")
    positions = np.arange(table.shape[0])[None, :]
    selected = table[positions, codes]
    if np.any(~np.isfinite(selected)):
        raise ValueError("FAST table is missing a state realized by a screened prompt")
    return np.mean(selected, axis=1)


def fast_mcq_code_permutation_null(
    clipped_value: np.ndarray, prompt_codes: np.ndarray, *,
    n_permutations: int = 200, seed: int = 0,
) -> np.ndarray:
    """Selection-preserving screening null for MCQ code/value association."""
    if int(n_permutations) < 200:
        raise ValueError("FAST permutation null requires B>=200")
    table = np.asarray(clipped_value, dtype=float)
    codes = np.asarray(prompt_codes, dtype=int)
    _observed_prompt_values(table, codes)
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(n_permutations), dtype=float)
    for draw in range(int(n_permutations)):
        values = np.zeros(codes.shape[0], dtype=float)
        for panel in range(codes.shape[1]):
            unique = np.unique(codes[:, panel])
            permuted = rng.permutation(table[panel, unique])
            lookup = dict(zip(unique.tolist(), permuted.tolist()))
            values += np.asarray([lookup[int(code)] for code in codes[:, panel]])
        output[draw] = float(np.max(values / codes.shape[1]))
    return output


def _binary_mi_rows(target: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Vectorized plug-in MI for many binary prediction rows."""
    y = np.asarray(target, dtype=np.uint8)
    p = np.asarray(predictions, dtype=np.uint8)
    if p.ndim != 2 or p.shape[1] != len(y):
        raise ValueError("prediction rows do not align with target")
    n = float(len(y))
    p64 = p.astype(np.int64)
    y64 = y.astype(np.int64)
    counts = np.empty((len(p), 2, 2), dtype=float)
    counts[:, 1, 1] = p64 @ y64
    counts[:, 0, 1] = np.sum(p64, axis=1) - counts[:, 1, 1]
    counts[:, 1, 0] = np.sum(y) - counts[:, 1, 1]
    counts[:, 0, 0] = n - np.sum(counts[:, [0, 1], [1, 0]], axis=1) - counts[:, 1, 1]
    joint = counts / n
    left = joint.sum(axis=2, keepdims=True)
    right = joint.sum(axis=1, keepdims=True)
    product = left * right
    terms = np.zeros_like(joint)
    keep = joint > 0
    terms[keep] = joint[keep] * np.log2(joint[keep] / product[keep])
    return np.sum(terms, axis=(1, 2))


def fast_behavioral_label_permutation_null(
    target: Sequence[int], hard_predictions: np.ndarray,
    blind_predictions: np.ndarray, prompt_codes: np.ndarray,
    shuffled_state_ids: np.ndarray, *, n_permutations: int = 200, seed: int = 0,
) -> np.ndarray:
    """Shuffle H labels while preserving panels, controls, and max selection."""
    if int(n_permutations) < 200:
        raise ValueError("FAST permutation null requires B>=200")
    y = np.asarray(target, dtype=np.uint8)
    predictions = np.asarray(hard_predictions, dtype=np.int8)
    blind = np.asarray(blind_predictions, dtype=np.uint8)
    codes = np.asarray(prompt_codes, dtype=int)
    shuffled = np.asarray(shuffled_state_ids, dtype=int)
    if predictions.ndim != 3 or predictions.shape[0] != codes.shape[1]:
        raise ValueError("FAST behavioral predictions do not align with prompt codes")
    if blind.shape != (predictions.shape[0], predictions.shape[2]):
        raise ValueError("FAST blind predictions do not align")
    for panel in range(codes.shape[1]):
        needed = np.unique(np.concatenate((codes[:, panel], shuffled[panel, codes[:, panel]])))
        if np.any(predictions[panel, needed] < 0):
            raise ValueError("FAST behavioral table lacks an observed/control prediction")
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(n_permutations), dtype=float)
    for draw in range(int(n_permutations)):
        permuted = rng.permutation(y)
        prompt_values = np.zeros(len(codes), dtype=float)
        for panel in range(codes.shape[1]):
            present = np.flatnonzero(np.all(predictions[panel] >= 0, axis=1))
            mi = np.full(predictions.shape[1], np.nan, dtype=float)
            mi[present] = _binary_mi_rows(permuted, predictions[panel, present])
            blind_mi = plugin_binary_mutual_information(permuted, blind[panel])
            state = codes[:, panel]
            control = shuffled[panel, state]
            prompt_values += np.maximum(0.0, mi[state] - np.maximum(blind_mi, mi[control]))
        output[draw] = float(np.max(prompt_values / codes.shape[1]))
    return output


def aggregate_fast_screening(
    *, raw_lift: np.ndarray, clipped_value: np.ndarray,
    prompt_signatures: np.ndarray, panels: Sequence[Sequence[int]],
    prompt_ids: Sequence[str], target_entropy_cap: float,
    permutation_null: Sequence[float], channel: str,
) -> dict:
    codes = signatures_to_states(prompt_signatures, panels)
    values = _observed_prompt_values(clipped_value, codes)
    raw_values = _observed_prompt_values(raw_lift, codes)
    ids = list(map(str, prompt_ids))
    achieved = float(np.max(values))
    cap = float(target_entropy_cap)
    if not np.isfinite(cap) or cap + 1e-12 < achieved:
        # MCQ lift is bounded by one rather than target entropy in bits.
        cap = 1.0 if str(channel) == "mcq" else cap
    if cap + 1e-12 < achieved:
        raise ValueError("level-0 FAST cap is below an observed value")
    return {
        "lane": "fast", "claim_role": "screening_only",
        "release_eligible": False, "channel": str(channel),
        "permutation_null_kind": (
            "mcq_observed_code_value_association_randomization"
            if str(channel) == "mcq" else
            "behavioral_heldout_label_shuffle_preserving_controls_and_max_selection"
        ),
        "n_prompts": len(ids), "n_panels": len(panels),
        "n_observed_joint_codes": int(len(np.unique(codes, axis=0))),
        "achieved_value": achieved,
        "exact_structural_cap": None,
        "exact_structural_cap_unavailable_reason": "FAST lane scores observed states only",
        "level0_cap": cap, "level0_gap": float(cap - achieved),
        "prompt_value": values, "prompt_raw_lift": raw_values,
        "legibility_argmax": tie_class(values, ids),
        **_permutation_summary(achieved, permutation_null),
    }
