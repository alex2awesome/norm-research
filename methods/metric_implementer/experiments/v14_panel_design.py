"""Frozen v14 probe splits, target-separating panels, and code diagnostics.

The module is deliberately decoder-free.  It consumes only executor-side binary
signatures and stable identifiers, so panel construction cannot adapt to either
reconstructor outputs or certification values.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "cr3-v14-panel-design-v1"
DEFAULT_SPLIT_SIZES = {"teaching": 120, "decoder_development": 30, "heldout": 150}
ELIGIBLE_FRACTIONS = (0.40, 0.45, 0.50, 0.55, 0.60)


def canonical_sha256(payload: object) -> str:
    packed = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _binary_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty metric-by-probe matrix")
    if np.any(~np.isfinite(raw.astype(float))):
        raise ValueError(f"{name} contains non-finite values")
    return (raw.astype(float) > 0.5).astype(np.uint8)


def freeze_probe_split(
    probe_ids: Sequence[str], *, run_sha: str, metric_key: str,
    split_sizes: Mapping[str, int] = DEFAULT_SPLIT_SIZES,
) -> dict:
    """Hash-rank probes into stable, disjoint S/D_dec/H splits."""
    ids = [str(value) for value in probe_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("probe_ids must be unique")
    sizes = {str(key): int(value) for key, value in split_sizes.items()}
    if set(sizes) != set(DEFAULT_SPLIT_SIZES) or any(value <= 0 for value in sizes.values()):
        raise ValueError("split_sizes must contain positive teaching/decoder_development/heldout")
    if sum(sizes.values()) != len(ids):
        raise ValueError(
            f"split sizes total {sum(sizes.values())}, but there are {len(ids)} probes"
        )
    ranked = sorted(
        range(len(ids)),
        key=lambda index: (
            canonical_sha256({
                "run_sha": str(run_sha), "metric": str(metric_key),
                "probe_id": ids[index], "purpose": "v14-probe-split",
            }),
            index,
        ),
    )
    output: dict[str, object] = {
        "schema": "cr3-v14-probe-split-v1",
        "run_sha": str(run_sha),
        "metric_key": str(metric_key),
        "probe_ids_sha256": canonical_sha256(ids),
        "n_probes": len(ids),
    }
    cursor = 0
    for name in ("teaching", "decoder_development", "heldout"):
        indices = sorted(ranked[cursor:cursor + sizes[name]])
        cursor += sizes[name]
        output[name] = {
            "indices": indices,
            "probe_ids": [ids[index] for index in indices],
            "sha256": canonical_sha256([ids[index] for index in indices]),
        }
    output["split_sha256"] = canonical_sha256(output)
    validate_probe_split(output)
    return output


def validate_probe_split(split: Mapping[str, object]) -> None:
    if split.get("schema") != "cr3-v14-probe-split-v1":
        raise ValueError("unsupported v14 probe split schema")
    groups = []
    for name in ("teaching", "decoder_development", "heldout"):
        row = split.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"missing split {name}")
        indices = [int(value) for value in row.get("indices", [])]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate indices inside {name}")
        groups.append(set(indices))
    if any(groups[left].intersection(groups[right]) for left in range(3) for right in range(left)):
        raise ValueError("v14 probe splits overlap")
    union = set().union(*groups)
    if union != set(range(int(split.get("n_probes", -1)))):
        raise ValueError("v14 probe splits do not cover the declared probes")


def code_entropy_bits(codes: np.ndarray) -> float:
    matrix = np.asarray(codes, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("codes must be a nonempty two-dimensional matrix")
    _, counts = np.unique(matrix, axis=0, return_counts=True)
    probabilities = counts.astype(float) / float(matrix.shape[0])
    return float(-np.sum(probabilities * np.log2(probabilities)))


def identification_diagnostic(
    codebook_signatures: np.ndarray, panel_indices: Sequence[int], *, target_index: int,
) -> dict:
    """Exact population I(z;M)=H(z), with target separation diagnostics."""
    signatures = _binary_matrix(codebook_signatures, name="codebook_signatures")
    panel = np.asarray(panel_indices, dtype=int)
    if panel.ndim != 1 or len(panel) == 0 or len(set(map(int, panel))) != len(panel):
        raise ValueError("panel_indices must be a nonempty unique vector")
    if np.any(panel < 0) or np.any(panel >= signatures.shape[1]):
        raise ValueError("panel index outside signature matrix")
    if not 0 <= int(target_index) < signatures.shape[0]:
        raise ValueError("target_index outside codebook")
    codes = signatures[:, panel]
    target = codes[int(target_index)]
    others = np.delete(codes, int(target_index), axis=0)
    distances = np.sum(others != target[None, :], axis=1)
    collisions = np.flatnonzero(np.all(others == target[None, :], axis=1))
    return {
        "identification_mi_bits": code_entropy_bits(codes),
        "code_entropy_bits": code_entropy_bits(codes),
        "n_codebook_metrics": int(signatures.shape[0]),
        "n_distinct_codes": int(len(np.unique(codes, axis=0))),
        "target_margin": int(np.min(distances)) if len(distances) else len(panel),
        "target_unique": not bool(len(collisions)),
        "n_target_collisions": int(len(collisions)),
        "ceilings": {
            "panel_bits": int(len(panel)),
            "codebook_bits": float(math.log2(signatures.shape[0])),
        },
        "scope": "identification_only_not_a_behavioral_value_ceiling",
    }


def _panel_score(
    signatures: np.ndarray, target_index: int, selected: Sequence[int],
) -> tuple[int, float]:
    if not selected:
        return 0, 0.0
    report = identification_diagnostic(signatures, selected, target_index=target_index)
    return int(report["target_margin"]), float(report["code_entropy_bits"])


def _completion_balance_feasible(
    target: np.ndarray, selected: Sequence[int], remaining: Sequence[int], panel_size: int,
) -> bool:
    slots = int(panel_size) - len(selected)
    if slots < 0:
        return False
    yes = int(np.sum(target[np.asarray(selected, dtype=int)])) if selected else 0
    remaining_values = target[np.asarray(remaining, dtype=int)] if remaining else np.empty(0)
    available_yes = int(np.sum(remaining_values))
    available_no = len(remaining_values) - available_yes
    min_yes = yes + max(0, slots - available_no)
    max_yes = yes + min(slots, available_yes)
    return max(min_yes, 3) <= min(max_yes, 5)


def _eligible_subset(
    teaching_indices: Sequence[int], *, run_sha: str, metric_key: str,
    trial: int, attempt: int, fraction: float, panel_size: int,
) -> list[int]:
    size = max(panel_size, int(round(len(teaching_indices) * float(fraction))))
    return sorted(
        teaching_indices,
        key=lambda index: (
            canonical_sha256({
                "run_sha": run_sha, "metric": metric_key, "trial": trial,
                "attempt": attempt, "probe": int(index), "purpose": "eligible-subset",
            }),
            int(index),
        ),
    )[:size]


def _greedy_panel(
    signatures: np.ndarray, target_index: int, eligible: Sequence[int], usage: np.ndarray,
    *, panel_size: int, usage_lambda: float, tie_salt: str,
) -> tuple[int, ...] | None:
    target = signatures[int(target_index)]
    selected: list[int] = []
    while len(selected) < panel_size:
        candidates = []
        current_margin, _ = _panel_score(signatures, target_index, selected)
        for index in eligible:
            if index in selected:
                continue
            trial_selected = selected + [int(index)]
            remaining = [value for value in eligible if value not in trial_selected]
            if not _completion_balance_feasible(target, trial_selected, remaining, panel_size):
                continue
            margin, entropy = _panel_score(signatures, target_index, trial_selected)
            primary = float(margin - current_margin) - float(usage_lambda) * float(usage[index])
            candidates.append((
                -primary, -entropy,
                canonical_sha256({"salt": tie_salt, "probe": int(index)}), int(index),
            ))
        if not candidates:
            return None
        selected.append(min(candidates)[-1])
    yes = int(np.sum(target[np.asarray(selected, dtype=int)]))
    if not 3 <= yes <= 5:
        return None
    return tuple(sorted(selected))


def _repair_coverage(
    signatures: np.ndarray, target_index: int, panels: list[tuple[int, ...]],
    teaching_indices: Sequence[int], *, min_coverage: int, max_swaps: int,
) -> tuple[list[tuple[int, ...]], int]:
    target = signatures[int(target_index)]
    universe = list(map(int, teaching_indices))
    swaps = 0
    while swaps < int(max_swaps):
        usage = {index: 0 for index in universe}
        for panel in panels:
            for index in panel:
                usage[index] += 1
        underused = sorted(
            (index for index in universe if usage[index] < min_coverage),
            key=lambda index: (usage[index], index),
        )
        if not underused:
            return panels, swaps
        incoming = underused[0]
        candidates = []
        for position, panel in enumerate(panels):
            if incoming in panel:
                continue
            old_margin, old_entropy = _panel_score(signatures, target_index, panel)
            old_unique = old_margin > 0
            for outgoing in panel:
                if usage[outgoing] <= min_coverage:
                    continue
                # Same-label swaps retain the hard 3--5 target balance exactly.
                if target[outgoing] != target[incoming]:
                    continue
                replacement = tuple(sorted((set(panel) - {outgoing}) | {incoming}))
                if replacement in panels:
                    continue
                new_margin, new_entropy = _panel_score(signatures, target_index, replacement)
                if old_unique and new_margin == 0:
                    continue
                candidates.append((
                    old_margin - new_margin,
                    old_entropy - new_entropy,
                    -usage[outgoing], position, outgoing, replacement,
                ))
        if not candidates:
            break
        _, _, _, position, _, replacement = min(candidates)
        panels[position] = replacement
        swaps += 1
    usage = {index: 0 for index in universe}
    for panel in panels:
        for index in panel:
            usage[index] += 1
    missing = {index: count for index, count in usage.items() if count < min_coverage}
    if missing:
        raise RuntimeError(f"coverage repair failed for {len(missing)} probes: {missing}")
    return panels, swaps


def _decoder_assignments(
    *, run_sha: str, metric_key: str, n_panels: int, decoder_families: Sequence[str],
) -> list[str]:
    families = [str(value) for value in decoder_families]
    if len(families) != 3 or len(set(families)) != 3:
        raise ValueError("v14 requires exactly three distinct decoder families")
    counts = [n_panels // 3 + int(index < n_panels % 3) for index in range(3)]
    slots = [family for family, count in zip(families, counts) for _ in range(count)]
    return [
        slots[index] for index in sorted(
            range(len(slots)),
            key=lambda index: canonical_sha256({
                "run_sha": run_sha, "metric": metric_key, "slot": index,
                "family": slots[index], "purpose": "decoder-panel-assignment",
            }),
        )
    ]


def build_panel_design(
    codebook_signatures: np.ndarray, *, target_index: int,
    teaching_indices: Sequence[int], run_sha: str, metric_key: str,
    probe_ids: Sequence[str] | None = None, n_panels: int = 50, panel_size: int = 8,
    usage_lambda: float = 0.25, max_attempts: int = 64, min_coverage: int = 2,
    max_repair_swaps: int = 120,
    decoder_families: Sequence[str] = ("qwen", "llama", "mistral"),
) -> dict:
    """Build the frozen diverse, target-separating v14 panel family."""
    signatures = _binary_matrix(codebook_signatures, name="codebook_signatures")
    teaching = sorted(map(int, teaching_indices))
    if len(teaching) < panel_size or len(teaching) != len(set(teaching)):
        raise ValueError("teaching_indices must contain at least panel_size unique probes")
    if min(teaching) < 0 or max(teaching) >= signatures.shape[1]:
        raise ValueError("teaching index outside signature matrix")
    target = signatures[int(target_index)]
    if np.unique(target[np.asarray(teaching, dtype=int)]).size != 2:
        raise ValueError("teaching split must contain both target labels")

    usage = np.zeros(signatures.shape[1], dtype=np.int64)
    panels: list[tuple[int, ...]] = []
    attempt_rows = []
    for trial in range(int(n_panels)):
        fraction = ELIGIBLE_FRACTIONS[trial % len(ELIGIBLE_FRACTIONS)]
        selected = None
        selected_attempt = None
        fallback = None
        for attempt in range(int(max_attempts)):
            eligible = _eligible_subset(
                teaching, run_sha=str(run_sha), metric_key=str(metric_key), trial=trial,
                attempt=attempt, fraction=fraction, panel_size=panel_size,
            )
            candidate = _greedy_panel(
                signatures, int(target_index), eligible, usage, panel_size=panel_size,
                usage_lambda=float(usage_lambda), tie_salt=f"{run_sha}:{metric_key}:{trial}:{attempt}",
            )
            if candidate is None or candidate in panels:
                continue
            report = identification_diagnostic(signatures, candidate, target_index=target_index)
            if fallback is None:
                fallback = (candidate, attempt, report)
            if report["target_unique"]:
                selected, selected_attempt = candidate, attempt
                break
        if selected is None:
            if fallback is None:
                raise RuntimeError(f"could not construct balanced unique panel {trial}")
            selected, selected_attempt, _ = fallback
        panels.append(selected)
        usage[np.asarray(selected, dtype=int)] += 1
        attempt_rows.append({
            "trial": trial,
            "eligible_fraction": fraction,
            "selected_attempt": int(selected_attempt),
        })

    panels, repair_swaps = _repair_coverage(
        signatures, int(target_index), panels, teaching,
        min_coverage=int(min_coverage), max_swaps=int(max_repair_swaps),
    )
    if len(set(panels)) != len(panels):
        raise RuntimeError("coverage repair created duplicate panels")
    assignments = _decoder_assignments(
        run_sha=str(run_sha), metric_key=str(metric_key), n_panels=len(panels),
        decoder_families=decoder_families,
    )
    final_usage = {index: 0 for index in teaching}
    rows = []
    for trial, (panel, family) in enumerate(zip(panels, assignments)):
        for index in panel:
            final_usage[index] += 1
        diagnostic = identification_diagnostic(signatures, panel, target_index=target_index)
        yes = int(np.sum(target[np.asarray(panel, dtype=int)]))
        panel_payload = {
            "trial": trial,
            "indices": list(panel),
            "probe_ids": ([str(probe_ids[index]) for index in panel] if probe_ids else None),
            "decoder_family": family,
            "target_yes": yes,
            "target_no": panel_size - yes,
            "target_state_bits": target[np.asarray(panel, dtype=int)].astype(int).tolist(),
            "target_uniqueness_exception": (
                None if diagnostic["target_unique"] else {
                    "reason": "no unique balanced panel found within 64 frozen eligible-subset attempts",
                    "n_target_collisions": diagnostic["n_target_collisions"],
                }
            ),
            **diagnostic,
            "construction": attempt_rows[trial],
        }
        panel_payload["panel_sha256"] = canonical_sha256(panel_payload)
        rows.append(panel_payload)

    manifest = {
        "schema": SCHEMA_VERSION,
        "run_sha": str(run_sha),
        "metric_key": str(metric_key),
        "target_index": int(target_index),
        "n_codebook_metrics": int(signatures.shape[0]),
        "n_probes": int(signatures.shape[1]),
        "teaching_indices": teaching,
        "teaching_sha256": canonical_sha256(teaching),
        "panel_size": int(panel_size),
        "n_panels": int(n_panels),
        "usage_lambda": float(usage_lambda),
        "eligible_fractions": list(ELIGIBLE_FRACTIONS),
        "max_attempts_per_panel": int(max_attempts),
        "minimum_probe_coverage": int(min_coverage),
        "coverage_repair_swaps": int(repair_swaps),
        "probe_usage": {str(index): int(final_usage[index]) for index in teaching},
        "panels": rows,
        "behavioral_ceiling_contract": (
            "identification_mi_is_diagnostic_only; behavioral ceilings are target entropy "
            "on H and exact per-panel state maxima"
        ),
    }
    manifest["design_sha256"] = canonical_sha256(manifest)
    validate_panel_design(manifest)
    return manifest


def validate_panel_design(manifest: Mapping[str, object]) -> None:
    if manifest.get("schema") != SCHEMA_VERSION:
        raise ValueError("unsupported v14 panel schema")
    panels = list(manifest.get("panels", []))
    if len(panels) != int(manifest.get("n_panels", -1)):
        raise ValueError("panel manifest is incomplete")
    panel_size = int(manifest.get("panel_size", -1))
    index_sets = []
    for row in panels:
        panel_core = dict(row)
        observed_panel_sha = str(panel_core.pop("panel_sha256", ""))
        if observed_panel_sha != canonical_sha256(panel_core):
            raise ValueError("panel checksum mismatch")
        indices = tuple(sorted(map(int, row["indices"])))
        if len(indices) != panel_size or len(set(indices)) != panel_size:
            raise ValueError("panel has wrong size or duplicate probes")
        if not 3 <= int(row["target_yes"]) <= 5:
            raise ValueError("panel violates target-label balance")
        index_sets.append(indices)
    if len(index_sets) != len(set(index_sets)):
        raise ValueError("duplicate v14 panels")
    family_counts = sorted(
        sum(str(row["decoder_family"]) == family for row in panels)
        for family in {str(row["decoder_family"]) for row in panels}
    )
    if len(family_counts) != 3 or family_counts[-1] - family_counts[0] > 1:
        raise ValueError("decoder-family panel assignment is not balanced 17/17/16")
    minimum = int(manifest.get("minimum_probe_coverage", -1))
    usage = {int(key): int(value) for key, value in manifest.get("probe_usage", {}).items()}
    teaching = set(map(int, manifest.get("teaching_indices", [])))
    if set(usage) != teaching or any(value < minimum for value in usage.values()):
        raise ValueError("v14 panels do not meet the hard coverage contract")
    core = dict(manifest)
    observed = str(core.pop("design_sha256", ""))
    if observed != canonical_sha256(core):
        raise ValueError("v14 panel manifest checksum mismatch")
