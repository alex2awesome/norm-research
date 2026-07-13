"""Unified two-channel CR-3 v13.1 value-bound campaign launcher.

The constructor phase fills all MCQ state cells and all behavioral induction cells while
one constructor is resident.  After constructors are released, one fixed Llama-3.1-8B
executor fills content-addressed behavioral executions.  SQLite state/rule caches make
both phases resumable without changing the frozen output schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import ImplementerConfig
from ..vllm_backend import (
    CR3_BINARY_READOUT_ID,
    make_judge_backend,
    model_revision_id,
    release_resident_engines,
)
from .behavioral_value_channel import (
    BEHAVIORAL_ARMS,
    evaluate_behavioral_state_tables,
)
from .cr3_reconstruction_values import _bootstrap, _payload_sha256, validate_codebook_manifest
from .cr3_sampled_value_certify import (
    VALUE_BOUND_CERTIFICATE_SCHEMA,
    VALUE_BOUND_DESIGN_SCHEMA,
    VALUE_BOUND_RELEASE,
    VALUE_BOUND_RESULTS_SCHEMA,
    VALUE_BOUND_STATE_SCHEMA,
    _file_sha256,
    _load_codebook,
    _load_production_codebook,
    build_value_bound_design_manifest,
    enumerate_exact_pool_values,
    evaluate_mcq_state_tables_v13_1,
    fixed_prefix_capture_recapture,
    secondary_value_status,
)
from .v13_value_cache import ValueCache


METRICS_MANIFEST_SCHEMA = "cr3-value-bound-metrics-v13.1"
CAMPAIGN_MANIFEST_SCHEMA = "cr3-value-bound-campaign-v13.1"
FIXED_EXECUTOR = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_CONSTRUCTORS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "microsoft/phi-4",
    "meta-llama/Llama-3.3-70B-Instruct",
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(value)).strip("_")


def _resolve_path(value: str | Path, base: Path) -> Path:
    raw = str(value)
    rewrites = json.loads(os.environ.get("V13_PATH_REWRITE_JSON", "{}"))
    for source, destination in sorted(
        rewrites.items(), key=lambda item: -len(str(item[0]))
    ):
        source = str(source).rstrip("/")
        if raw == source or raw.startswith(source + "/"):
            raw = str(destination).rstrip("/") + raw[len(source):]
            break
    path = Path(raw)
    return (path if path.is_absolute() else base / path).resolve()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **dict(arrays))
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_metrics_manifest(path: str | Path) -> tuple[dict, Path]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != METRICS_MANIFEST_SCHEMA:
        raise ValueError(f"unexpected metrics manifest schema in {source}")
    entries = list(payload.get("metrics") or [])
    if not entries:
        raise ValueError("metrics manifest contains no metric entries")
    required = {"task", "level", "metric", "metric_key", "codebook_path"}
    for index, entry in enumerate(entries):
        missing = sorted(required.difference(entry))
        if missing:
            raise ValueError(f"metric entry {index} lacks {missing}")
        if str(entry["task"]) == "grant-funding":
            raise ValueError("grant-writing metrics are excluded from v13.1")
    return payload, source.parent


def _load_codebook_for_entry(entry: Mapping[str, object], base: Path) -> dict:
    path = _resolve_path(entry["codebook_path"], base)
    layout = str(entry.get("codebook_layout") or "production")
    if layout == "production":
        assets_root = _resolve_path(entry.get("assets_root") or path.parent.parent, base)
        manifest, _provenance = _load_production_codebook(path, assets_root=assets_root)
    elif layout == "simple":
        manifest = _load_codebook(path)
    else:
        raise ValueError(f"unsupported codebook_layout {layout!r}")
    validate_codebook_manifest(manifest)
    if str(entry["metric_key"]) not in manifest["entries"]:
        raise ValueError(f"metric key {entry['metric_key']} is absent from {path}")
    return manifest


def _target_entropy_for_entry(entry: Mapping[str, object], base: Path) -> float:
    if entry.get("target_entropy_bits") is not None:
        return float(entry["target_entropy_bits"])
    codebook = _load_codebook_for_entry(entry, base)
    target = _bootstrap(codebook["metrics"][str(entry["metric_key"])]["bootstrap_path"])
    bits = (np.asarray(target["target"], dtype=float) > 0.5).astype(float)
    p = float(np.mean(bits))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def select_metric_entries(manifest: Mapping[str, object], base: Path) -> list[dict]:
    entries = [dict(entry) for entry in manifest["metrics"]]
    source_sets = dict(manifest.get("candidate_source_sets") or {})
    for entry in entries:
        source_set = entry.get("candidate_source_set")
        if source_set is not None:
            if entry.get("candidate_sources") or str(source_set) not in source_sets:
                raise ValueError(f"invalid candidate_source_set {source_set!r}")
            entry["candidate_sources"] = list(source_sets[str(source_set)])
    selection = dict(manifest.get("selection") or {})
    if not selection:
        return entries
    if selection.get("mode") != "target_entropy_quintiles":
        raise ValueError("only target_entropy_quintiles selection is supported")
    per_task = int(selection.get("per_task", 5))
    if per_task != 5:
        raise ValueError("v13.1 Tier B freezes exactly five metrics per task")
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        entry["target_entropy_bits"] = _target_entropy_for_entry(entry, base)
        grouped.setdefault(str(entry["task"]), []).append(entry)
    selected = []
    for task in sorted(grouped):
        rows = sorted(grouped[task], key=lambda row: (
            float(row["target_entropy_bits"]),
            hashlib.sha256(str(row["metric_key"]).encode("utf-8")).hexdigest(),
        ))
        if len(rows) < per_task:
            raise ValueError(f"task {task} has fewer than five Tier-B candidates")
        positions = [int(round(q * (len(rows) - 1) / (per_task - 1))) for q in range(per_task)]
        used = set()
        for quintile, position in enumerate(positions):
            if position in used:
                position = next(index for index in range(len(rows)) if index not in used)
            used.add(position)
            row = dict(rows[position])
            row["target_entropy_quintile"] = int(quintile + 1)
            selected.append(row)
    expected_tasks = set(selection.get("tasks") or grouped)
    if set(grouped) != expected_tasks:
        raise ValueError("metrics manifest tasks differ from the frozen Tier-B task list")
    if len(selected) != 5 * len(expected_tasks):
        raise RuntimeError("entropy-quintile selection returned the wrong batch size")
    return selected


def _materialize_entry_paths(entry: Mapping[str, object], base: Path) -> dict:
    """Make a selected entry relocatable before writing an upgrade manifest."""
    materialized = dict(entry)
    for field in ("codebook_path", "assets_root", "candidate_bank_path"):
        if materialized.get(field) is not None:
            materialized[field] = str(_resolve_path(materialized[field], base))
    sources = []
    for source in materialized.get("candidate_sources") or []:
        resolved = dict(source)
        resolved["path"] = str(_resolve_path(resolved["path"], base))
        sources.append(resolved)
    if sources:
        materialized["candidate_sources"] = sources
    materialized.pop("candidate_source_set", None)
    return materialized


def select_tier_a_upgrades(
    results: Sequence[Mapping[str, object]], entries: Sequence[Mapping[str, object]],
    *, n_gap: int = 5, n_disagreement: int = 5,
) -> tuple[list[dict], dict]:
    """Freeze Tier-A upgrades without mixing the two channels' value units.

    Structural-gap ranking uses only the primary behavioral channel (bits).  The
    disagreement ranking uses the already-declared cross-channel Spearman statistic,
    never an arithmetic combination of MCQ probability lift and behavioral MI.
    """
    frame = pd.DataFrame(list(results))
    required = {
        "task", "level", "metric_key", "constructor", "channel",
        "exact_structural_gap", "cross_channel_spearman",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Tier-A upgrade selection lacks complete Tier-B result columns")
    behavioral = frame.loc[frame["channel"] == "behavioral"].copy()
    if behavioral.empty:
        raise ValueError("Tier-A upgrades require the primary behavioral channel")
    identity = ["task", "level", "metric_key"]
    gap_rows = (
        behavioral.groupby(identity, as_index=False, dropna=False)["exact_structural_gap"]
        .mean()
        .sort_values(
            ["exact_structural_gap", "task", "level", "metric_key"],
            ascending=[False, True, True, True], kind="mergesort",
        )
    )
    gap_rows = gap_rows.head(min(int(n_gap), len(gap_rows)))
    gap_keys = [tuple(row[field] for field in identity) for _, row in gap_rows.iterrows()]

    disagreement_rows = (
        behavioral.groupby(identity, as_index=False, dropna=False)["cross_channel_spearman"]
        .mean()
    )
    disagreement_rows["disagreement_sort"] = disagreement_rows[
        "cross_channel_spearman"
    ].fillna(-np.inf)
    disagreement_rows = disagreement_rows.loc[
        ~disagreement_rows.apply(
            lambda row: tuple(row[field] for field in identity) in set(gap_keys), axis=1
        )
    ].sort_values(
        ["disagreement_sort", "task", "level", "metric_key"],
        ascending=[True, True, True, True], kind="mergesort",
    )
    disagreement_rows = disagreement_rows.head(
        min(int(n_disagreement), len(disagreement_rows))
    )
    disagreement_keys = [
        tuple(row[field] for field in identity)
        for _, row in disagreement_rows.iterrows()
    ]
    chosen_keys = gap_keys + disagreement_keys
    by_key = {
        (str(entry["task"]), str(entry["level"]), str(entry["metric_key"])): dict(entry)
        for entry in entries
    }
    chosen = [by_key[tuple(map(str, key))] for key in chosen_keys]
    selection = {
        "schema": "cr3-value-bound-tier-a-upgrades-v13.1",
        "structural_gap_basis": "mean behavioral unconstrained exact gap in bits",
        "cross_channel_disagreement_basis": (
            "ascending mean MCQ-versus-behavioral prompt-value Spearman; "
            "undefined correlations rank as maximal disagreement"
        ),
        "channels_numerically_combined": False,
        "largest_structural_gaps": [
            {
                **{field: str(row[field]) for field in identity},
                "mean_behavioral_gap_bits": float(row["exact_structural_gap"]),
            }
            for _, row in gap_rows.iterrows()
        ],
        "largest_cross_channel_disagreements": [
            {
                **{field: str(row[field]) for field in identity},
                "mean_cross_channel_spearman": (
                    None if pd.isna(row["cross_channel_spearman"])
                    else float(row["cross_channel_spearman"])
                ),
            }
            for _, row in disagreement_rows.iterrows()
        ],
        "n_selected": len(chosen),
    }
    return chosen, selection


def _load_npz_bank(path: Path, *, expected_n_probes: int, expected_probe_sha256: str) -> dict:
    with np.load(path, allow_pickle=True) as artifact:
        if "sigs" not in artifact.files:
            raise ValueError(f"candidate bank {path} lacks sigs")
        signatures = np.asarray(artifact["sigs"], dtype=float)
        if "texts" in artifact.files:
            texts = [str(value) for value in artifact["texts"]]
        elif "prompts" in artifact.files:
            texts = [str(value) for value in artifact["prompts"]]
        else:
            raise ValueError(f"candidate bank {path} lacks texts/prompts")
        if (signatures.ndim != 2 or signatures.shape != (len(texts), expected_n_probes)
                or np.any(~np.isfinite(signatures))):
            raise ValueError(f"candidate bank {path} has invalid signatures")
        if "probe_sha256" in artifact.files:
            observed = str(artifact["probe_sha256"])
            if observed != expected_probe_sha256:
                raise ValueError(f"candidate bank {path} uses a different probe panel")
        family_field = "families" if "families" in artifact.files else (
            "tags" if "tags" in artifact.files else None
        )
        families = (
            [str(value) for value in artifact[family_field]]
            if family_field else ["legacy_candidate"] * len(texts)
        )
        metadata = {}
        for field in (
            "generator_config_sha256", "models", "model_revisions", "temperatures",
            "seeds", "attempt_idx", "prompt_sha256",
        ):
            if field in artifact.files:
                values = np.asarray(artifact[field])
                if values.shape == (len(texts),):
                    metadata[field] = values.copy()
    return {
        "path": str(path), "sha256": _file_sha256(path), "signatures": signatures,
        "texts": texts, "families": families, "metadata": metadata,
    }


def _homogeneous_group_keys(bank: Mapping[str, object]) -> list[str]:
    metadata = bank["metadata"]
    if "generator_config_sha256" in metadata:
        return [str(value) for value in metadata["generator_config_sha256"]]
    fields = [field for field in ("models", "model_revisions", "temperatures") if field in metadata]
    if not fields:
        raise ValueError("provenance-valid split lacks immutable generator configuration fields")
    return [
        _payload_sha256({field: str(metadata[field][index]) for field in fields})
        for index in range(len(bank["texts"]))
    ]


def load_candidate_population(
    entry: Mapping[str, object], base: Path, *, n_probes: int, probe_sha256: str,
) -> dict:
    sources = list(entry.get("candidate_sources") or [])
    if not sources and entry.get("candidate_bank_path"):
        sources = [{"path": entry["candidate_bank_path"], "process_provenance": "legacy"}]
    if not sources:
        raise ValueError(f"metric {entry['metric_key']} has no candidate bank")
    discovery_signatures = []
    discovery_texts = []
    discovery_families = []
    process_streams = []
    source_provenance = []
    all_valid = True
    for source_index, source in enumerate(sources):
        path = _resolve_path(source["path"], base)
        bank = _load_npz_bank(
            path, expected_n_probes=n_probes, expected_probe_sha256=probe_sha256
        )
        source_name = str(source.get("source_name") or path.stem)
        process_mode = str(source.get("process_provenance") or "legacy")
        if process_mode == "fixed_prefix_suffix":
            fraction = float(source.get("discovery_prefix_fraction", 0.80))
            if not 0.0 < fraction < 1.0:
                raise ValueError("discovery_prefix_fraction must lie in (0,1)")
            group_keys = _homogeneous_group_keys(bank)
            groups: dict[str, list[int]] = {}
            for row_index, group_key in enumerate(group_keys):
                groups.setdefault(group_key, []).append(row_index)
            for group_key in sorted(groups):
                indices = groups[group_key]
                if len(indices) < 5:
                    raise ValueError("fixed prefix/audit generator family has fewer than five rows")
                n_prefix = min(len(indices) - 1, max(1, int(np.floor(fraction * len(indices)))))
                prefix_indices = indices[:n_prefix]
                audit_indices = indices[n_prefix:]
                family = f"{source_name}|{group_key[:16]}"
                discovery_signatures.append(bank["signatures"][prefix_indices])
                discovery_texts.extend(bank["texts"][index] for index in prefix_indices)
                discovery_families.extend([family] * len(prefix_indices))
                process_streams.append({
                    "family": family,
                    "discovery_prefix_signatures": bank["signatures"][prefix_indices],
                    "audit_suffix_signatures": bank["signatures"][audit_indices],
                    "provenance": {
                        "source_path": str(path), "source_sha256": bank["sha256"],
                        "generator_config_sha256": group_key,
                        "source_row_indices_discovery_prefix": prefix_indices,
                        "source_row_indices_never_absorbed_audit_suffix": audit_indices,
                        "prefix_fraction_rule": fraction,
                    },
                })
        elif process_mode == "legacy":
            all_valid = False
            discovery_signatures.append(bank["signatures"])
            discovery_texts.extend(bank["texts"])
            discovery_families.extend([
                f"{source_name}|{family}" for family in bank["families"]
            ])
        else:
            raise ValueError(f"unsupported process_provenance mode {process_mode!r}")
        source_provenance.append({
            "path": str(path), "sha256": bank["sha256"], "n_rows": len(bank["texts"]),
            "process_provenance": process_mode,
        })
    signatures = np.vstack(discovery_signatures)
    if len(discovery_texts) != len(signatures):
        raise RuntimeError("candidate population assembly lost row alignment")
    return {
        "signatures": signatures,
        "texts": discovery_texts,
        "families": discovery_families,
        "process_streams": process_streams if all_valid else None,
        "source_provenance": source_provenance,
        "n_discovery_prompts": int(len(signatures)),
        "n_never_absorbed_audit_prompts": int(sum(
            len(stream["audit_suffix_signatures"]) for stream in process_streams
        )) if all_valid else 0,
    }


def _make_backend(model: str, *, fake: bool):
    cfg = ImplementerConfig()
    cfg.vllm_fake = bool(fake)
    cfg.vllm_gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90"))
    default_tp = int(os.environ.get("VLLM_TP_SIZE", "1"))
    cfg.vllm_tp_size = int(
        os.environ.get("VLLM_EXECUTOR_TP_SIZE", str(default_tp))
        if str(model) == FIXED_EXECUTOR else default_tp
    )
    cfg.vllm_max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
    if os.environ.get("METRIC_IMPLEMENTER_LFS_HOME"):
        cfg.vllm_lfs_home = os.environ["METRIC_IMPLEMENTER_LFS_HOME"]
    overrides = json.loads(os.environ.get("V13_MODEL_PATH_OVERRIDES_JSON", "{}"))
    runtime_model = str(overrides.get(str(model), str(model)))
    backend = make_judge_backend(runtime_model, cfg, 0.0)
    revision = str(model) if fake else model_revision_id(runtime_model)
    return backend, revision


class _CacheOnlyConstructor:
    def generate_batch(self, *_args, **_kwargs):
        raise RuntimeError("behavioral induction cache is incomplete before executor phase")


def _cross_channel_spearman(mcq_values: np.ndarray, behavioral_values: np.ndarray) -> float | None:
    left = np.asarray(mcq_values, dtype=float)
    right = np.asarray(behavioral_values, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("cross-channel Spearman needs aligned prompt values")
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    rho = float(spearmanr(left, right).statistic)
    return rho if np.isfinite(rho) else None


def _prompt_frame(
    context: Mapping[str, object], aggregation: Mapping[str, object], *, constructor: str,
    channel: str, prompt_arm: str, cross_spearman: float | None,
) -> pd.DataFrame:
    population = context["population"]
    rows = []
    for index, (text, family, value) in enumerate(zip(
        population["texts"], population["families"], aggregation["mean_prompt_value"]
    )):
        rows.append({
            "task": context["entry"]["task"], "level": context["entry"]["level"],
            "metric": str(context["entry"]["metric"]),
            "metric_key": str(context["entry"]["metric_key"]),
            "executor": FIXED_EXECUTOR, "constructor": str(constructor),
            "channel": str(channel), "prompt_arm": str(prompt_arm),
            "prompt_index": int(index),
            "prompt_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
            "prompt_text": str(text), "prompt_family": str(family),
            "value": float(value),
            "pool_values": np.asarray(aggregation["prompt_pool_values"])[index].tolist(),
            "across_pool_variance": float(np.var(
                np.asarray(aggregation["prompt_pool_values"])[index], ddof=0
            )),
            "cross_channel_spearman": cross_spearman,
        })
    return pd.DataFrame(rows)


def _certificate_common(
    context: Mapping[str, object], *, constructor: str, constructor_revision: str,
    channel: str, tier: str, aggregation: Mapping[str, object],
    process: Mapping[str, object], cross_spearman: float | None, epsilon: float,
) -> dict:
    status = secondary_value_status(
        achieved=float(aggregation["achieved_value"]),
        exact_cap=float(aggregation["exact_structural_cap"]),
        process_bounds=process, epsilon=float(epsilon),
    )
    return {
        "schema": VALUE_BOUND_CERTIFICATE_SCHEMA,
        "release": VALUE_BOUND_RELEASE,
        "task": str(context["entry"]["task"]),
        "level": str(context["entry"]["level"]),
        "metric": str(context["entry"]["metric"]),
        "metric_key": str(context["entry"]["metric_key"]),
        "tier": str(tier).upper(), "channel": str(channel),
        "executor": {
            "model": FIXED_EXECUTOR,
            "revision": context["design"]["executor"]["revision"],
            "readout_id": context["design"]["executor"]["readout_id"],
        },
        "constructor": {"model": str(constructor), "revision": str(constructor_revision)},
        "design_manifest_sha256": context["design"]["design_manifest_sha256"],
        "achieved_value": float(aggregation["achieved_value"]),
        "achieved_prompt_index": int(aggregation["achieved_prompt_index"]),
        "exact_structural_cap": float(aggregation["exact_structural_cap"]),
        "exact_structural_gap": float(aggregation["exact_structural_gap"]),
        "worst_pool_achieved_value": float(aggregation["worst_pool_achieved_value"]),
        "pool_variance": float(aggregation["pool_variance"]),
        "pool_range": float(aggregation["achieved_pool_range"]),
        "mean_within_pool_panel_variance": float(
            aggregation["mean_within_pool_panel_variance"]
        ),
        "per_pool_exact_caps": np.asarray(aggregation["pool_caps"]).tolist(),
        "per_pool_achieved_values": np.asarray(
            aggregation["per_pool_achieved"]
        ).tolist(),
        "process_relative_gain_bounds": process,
        "cross_channel_spearman": cross_spearman,
        "secondary_summary": status,
        "secondary_summary_epsilon": float(epsilon),
        "finite_family_reporting": {
            "panel_mean_is_exact": True,
            "percentile_confidence_intervals_reported": False,
            "pool_range_and_variance_are_descriptive_sensitivity": True,
        },
        "source_population": {
            "n_discovery_prompts": context["population"]["n_discovery_prompts"],
            "n_never_absorbed_audit_prompts": context["population"][
                "n_never_absorbed_audit_prompts"
            ],
            "sources": context["population"]["source_provenance"],
        },
    }


def _result_row(certificate: Mapping[str, object], **extra) -> dict:
    process = certificate["process_relative_gain_bounds"]
    horizons = process.get("horizons") if process.get("available") else {}
    return {
        "schema": VALUE_BOUND_RESULTS_SCHEMA,
        "task": certificate["task"], "level": certificate["level"],
        "metric": certificate["metric"], "metric_key": certificate["metric_key"],
        "tier": certificate["tier"], "executor": certificate["executor"]["model"],
        "constructor": certificate["constructor"]["model"],
        "channel": certificate["channel"],
        "achieved_value": certificate["achieved_value"],
        "exact_structural_cap": certificate["exact_structural_cap"],
        "exact_structural_gap": certificate["exact_structural_gap"],
        "worst_pool_achieved_value": certificate["worst_pool_achieved_value"],
        "pool_variance": certificate["pool_variance"],
        "process_relative_gain_bound_h100": (
            horizons.get("100", {}).get("headline_gain_upper_min_of_declared_bounds")
        ),
        "process_relative_gain_bound_h300": (
            horizons.get("300", {}).get("headline_gain_upper_min_of_declared_bounds")
        ),
        "process_relative_bounds_available": bool(process.get("available")),
        "cross_channel_spearman": certificate["cross_channel_spearman"],
        "secondary_summary": certificate["secondary_summary"],
        **extra,
    }


def _channel_dir(out_root: Path, tier: str, entry: Mapping[str, object], constructor: str,
                 channel: str) -> Path:
    return (
        out_root / f"tier_{str(tier).upper()}" /
        _safe_name(f"{entry['task']}__{entry['level']}__{entry['metric_key']}") /
        _safe_name(constructor) / channel
    )


def _write_design(path: Path, context: Mapping[str, object]) -> None:
    design = dict(context["design"])
    design["campaign_identity"] = {
        "task": context["entry"]["task"], "level": context["entry"]["level"],
        "metric": str(context["entry"]["metric"]),
        "metric_key": context["entry"]["metric_key"],
    }
    _atomic_json(path, design)


def write_mcq_output(
    out_root: Path, context: Mapping[str, object], *, constructor: str,
    constructor_revision: str, tier: str, state_result: Mapping[str, object],
    aggregation: Mapping[str, object], cross_spearman: float | None,
) -> dict:
    directory = _channel_dir(out_root, tier, context["entry"], constructor, "mcq")
    process = fixed_prefix_capture_recapture(
        aggregation, process_streams=context["population"]["process_streams"]
    )
    certificate = _certificate_common(
        context, constructor=constructor, constructor_revision=constructor_revision,
        channel="mcq", tier=tier, aggregation=aggregation, process=process,
        cross_spearman=cross_spearman, epsilon=float(context["entry"].get("mcq_epsilon", 0.01)),
    )
    certificate.update({
        "value_name": "annotation-attributable Reconstruction-MCQ target-choice lift",
        "value_unit": "probability",
        "diagnostics": {
            "prior_balance": dict(state_result["blind"]),
            "gold_fidelity": context["entry"].get("gold_fidelity"),
            "gold_fidelity_available": context["entry"].get("gold_fidelity") is not None,
            "best_explanation_rate": float(state_result["best_explanation_rate"]),
            "diagnostics_never_gate_execution_or_reporting": True,
        },
        "non_disclosure": dict(state_result["non_disclosure"]),
    })
    certificate["certificate_sha256"] = _payload_sha256(certificate)
    frame = _prompt_frame(
        context, aggregation, constructor=constructor, channel="mcq",
        prompt_arm="target_choice_lift", cross_spearman=cross_spearman,
    )
    _write_design(directory / "design_manifest.json", context)
    _atomic_npz(directory / "state_tables.npz", {
        "schema": np.asarray(VALUE_BOUND_STATE_SCHEMA), "channel": np.asarray("mcq"),
        "tier": np.asarray(str(tier).upper()),
        "panel_sha256": np.asarray([
            panel["panel_sha256"] for panel in state_result["active_design"]["panels"]
        ], dtype=object),
        "panel_indices": np.asarray([
            panel["fixed_teaching_indices"]
            for panel in state_result["active_design"]["panels"]
        ], dtype=int),
        "state_values": np.asarray(state_result["state_values"], dtype=float),
        "raw_target_probabilities": np.asarray(
            state_result["raw_target_probabilities"], dtype=float
        ),
        "shuffled_target_probabilities": np.asarray(
            state_result["shuffled_target_probabilities"], dtype=float
        ),
        "pool_pattern_values": np.asarray(aggregation["pool_pattern_values"], dtype=float),
    })
    _atomic_parquet(directory / "prompt_values.parquet", frame)
    _atomic_json(directory / "certificate.json", certificate)
    return _result_row(certificate)


def write_behavioral_output(
    out_root: Path, context: Mapping[str, object], *, constructor: str,
    constructor_revision: str, tier: str, state_result: Mapping[str, object],
    aggregations: Mapping[str, Mapping[str, object]], cross_spearman: float | None,
) -> dict:
    directory = _channel_dir(out_root, tier, context["entry"], constructor, "behavioral")
    process_by_arm = {
        arm: fixed_prefix_capture_recapture(
            aggregations[arm], process_streams=context["population"]["process_streams"]
        ) for arm in BEHAVIORAL_ARMS
    }
    headline = aggregations["unconstrained"]
    certificate = _certificate_common(
        context, constructor=constructor, constructor_revision=constructor_revision,
        channel="behavioral", tier=tier, aggregation=headline,
        process=process_by_arm["unconstrained"], cross_spearman=cross_spearman,
        epsilon=float(context["entry"].get("behavioral_epsilon_bits", 0.01)),
    )
    certificate.update({
        "value_name": "held-out plug-in mutual-information lift over blind/shuffled controls",
        "value_unit": "bits", "headline_prompt_arm": "unconstrained",
        "target_entropy_bits": float(state_result["target_entropy_bits"]),
        "prompt_arms": {
            arm: {
                "achieved_value": float(aggregations[arm]["achieved_value"]),
                "exact_structural_cap": float(aggregations[arm]["exact_structural_cap"]),
                "exact_structural_gap": float(aggregations[arm]["exact_structural_gap"]),
                "worst_pool_achieved_value": float(
                    aggregations[arm]["worst_pool_achieved_value"]
                ),
                "pool_variance": float(aggregations[arm]["pool_variance"]),
                "process_relative_gain_bounds": process_by_arm[arm],
                "blind_mutual_information_bits": float(
                    state_result["arms"][arm]["blind_mutual_information_bits"]
                ),
                "n_distinct_rules": int(state_result["arms"][arm]["n_distinct_rules"]),
                "execution_degeneracy": state_result["arms"][arm]["execution_degeneracy"],
            } for arm in BEHAVIORAL_ARMS
        },
        "non_disclosure": dict(state_result["non_disclosure"]),
        "execution_degeneracy_is_reported_not_gated": True,
    })
    certificate["certificate_sha256"] = _payload_sha256(certificate)
    frames = [
        _prompt_frame(
            context, aggregations[arm], constructor=constructor, channel="behavioral",
            prompt_arm=arm, cross_spearman=cross_spearman,
        ) for arm in BEHAVIORAL_ARMS
    ]
    _write_design(directory / "design_manifest.json", context)
    arrays = {
        "schema": np.asarray(VALUE_BOUND_STATE_SCHEMA),
        "channel": np.asarray("behavioral"), "tier": np.asarray(str(tier).upper()),
        "panel_sha256": np.asarray([
            panel["panel_sha256"] for panel in state_result["active_design"]["panels"]
        ], dtype=object),
        "panel_indices": np.asarray([
            panel["fixed_teaching_indices"]
            for panel in state_result["active_design"]["panels"]
        ], dtype=int),
        "target_entropy_bits": np.asarray(state_result["target_entropy_bits"]),
    }
    for arm in BEHAVIORAL_ARMS:
        prefix = arm.replace("_examples", "")
        arrays[f"{prefix}__state_values"] = np.asarray(
            state_result["arms"][arm]["state_values"], dtype=float
        )
        arrays[f"{prefix}__raw_mutual_information_bits"] = np.asarray(
            state_result["arms"][arm]["raw_mutual_information_bits"], dtype=float
        )
        arrays[f"{prefix}__shuffled_mutual_information_bits"] = np.asarray(
            state_result["arms"][arm]["shuffled_mutual_information_bits"], dtype=float
        )
        arrays[f"{prefix}__balanced_agreement"] = np.asarray(
            state_result["arms"][arm]["balanced_agreement"], dtype=float
        )
        arrays[f"{prefix}__rule_sha256"] = np.asarray(
            state_result["arms"][arm]["rule_sha256"], dtype=object
        )
        arrays[f"{prefix}__pool_pattern_values"] = np.asarray(
            aggregations[arm]["pool_pattern_values"], dtype=float
        )
    _atomic_npz(directory / "state_tables.npz", arrays)
    _atomic_parquet(directory / "prompt_values.parquet", pd.concat(frames, ignore_index=True))
    _atomic_json(directory / "certificate.json", certificate)
    no_verbatim = aggregations["no_verbatim_examples"]
    return _result_row(
        certificate,
        no_verbatim_achieved_value=float(no_verbatim["achieved_value"]),
        no_verbatim_exact_structural_cap=float(no_verbatim["exact_structural_cap"]),
        no_verbatim_exact_structural_gap=float(no_verbatim["exact_structural_gap"]),
        exemplar_carrying_value_difference=float(
            headline["achieved_value"] - no_verbatim["achieved_value"]
        ),
    )


def _prepare_contexts(
    entries: Sequence[Mapping[str, object]], base: Path, *, preflight_one_panel: bool = False,
) -> list[dict]:
    contexts = []
    for entry in entries:
        codebook = _load_codebook_for_entry(entry, base)
        design = build_value_bound_design_manifest(
            codebook, target_metric_key=str(entry["metric_key"]),
            heldout_size=int(entry.get("heldout_size", 60)),
        )
        if preflight_one_panel:
            core = dict(design)
            core.pop("design_manifest_sha256", None)
            core["tiers"] = dict(core["tiers"])
            core["tiers"]["B"] = {
                "active_pool_ids": [core["pools"][0]["pool_id"]],
                "mcq_panels_per_pool": 1,
                "behavioral_panels_per_pool": 1,
            }
            core["preflight_design_limit"] = {
                "one_pool_and_one_panel_per_channel": True,
                "production_certificate": False,
            }
            design = {**core, "design_manifest_sha256": _payload_sha256(core)}
        if design["executor"]["model"] != FIXED_EXECUTOR:
            raise ValueError(
                f"metric {entry['metric_key']} uses executor {design['executor']['model']}, "
                f"not fixed {FIXED_EXECUTOR}"
            )
        population = load_candidate_population(
            entry, base, n_probes=int(design["n_probes"]),
            probe_sha256=str(design["probe_sha256"]),
        )
        contexts.append({
            "entry": dict(entry), "codebook": codebook,
            "design": design, "population": population,
        })
    return contexts


def run_campaign(
    *, channels: Sequence[str], constructor_models: Sequence[str],
    metrics_manifest_path: str | Path, tier: str, out_root: str | Path,
    fake_backends: bool = False, query_batch_size: int = 2048,
    cache_path: str | Path | None = None, allow_auto_upgrade: bool = True,
    preflight_one_panel: bool = False, metric_keys: Sequence[str] | None = None,
) -> list[dict]:
    channels = list(dict.fromkeys(map(str, channels)))
    if not channels or any(channel not in ("mcq", "behavioral") for channel in channels):
        raise ValueError("--channels must contain mcq and/or behavioral")
    tier = str(tier).upper()
    if tier not in ("A", "B"):
        raise ValueError("--tier must be A or B")
    metrics_manifest, base = load_metrics_manifest(metrics_manifest_path)
    entries = select_metric_entries(metrics_manifest, base)
    if metric_keys is not None:
        requested = set(map(str, metric_keys))
        entries = [entry for entry in entries if str(entry["metric_key"]) in requested]
        observed = {str(entry["metric_key"]) for entry in entries}
        if observed != requested:
            raise ValueError(f"requested metric keys are absent from the manifest: {requested-observed}")
    if preflight_one_panel and (tier != "B" or len(entries) != 1
                                or len(constructor_models) != 1):
        raise ValueError("--preflight-one-panel requires Tier B, one metric, and one model")
    contexts = _prepare_contexts(
        entries, base, preflight_one_panel=bool(preflight_one_panel)
    )
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cache_path = (
        Path(cache_path).resolve()
        if cache_path is not None else out_root / "cache" / "value_cells.sqlite"
    )
    staged: dict[tuple[str, str], dict] = {}
    revisions = {}
    with ValueCache(cache_path) as cache:
        # Constructor-resident phase: all MCQ tables and all behavioral inductions.
        for constructor in constructor_models:
            backend, revision = _make_backend(str(constructor), fake=fake_backends)
            revisions[str(constructor)] = revision
            for context in contexts:
                metric_key = str(context["entry"]["metric_key"])
                cell = staged.setdefault((str(constructor), metric_key), {})
                if "mcq" in channels:
                    state = evaluate_mcq_state_tables_v13_1(
                        backend, codebook_manifest=context["codebook"],
                        design_manifest=context["design"], target_metric_key=metric_key,
                        tier=tier, constructor_revision=revision, cache=cache,
                        query_batch_size=int(query_batch_size),
                    )
                    cell["mcq_state"] = state
                    cell["mcq_aggregation"] = enumerate_exact_pool_values(
                        context["design"], channel="mcq", tier=tier,
                        state_values=state["state_values"],
                        signatures=context["population"]["signatures"],
                    )
                if "behavioral" in channels:
                    evaluate_behavioral_state_tables(
                        backend, None, codebook_manifest=context["codebook"],
                        design_manifest=context["design"], target_metric_key=metric_key,
                        tier=tier, constructor_revision=revision,
                        executor_revision=str(context["design"]["executor"]["revision"]),
                        executor_readout_id=str(context["design"]["executor"]["readout_id"]),
                        cache=cache, query_batch_size=int(query_batch_size),
                        induction_only=True,
                    )
            release_resident_engines()

        # One fixed executor phase across every constructor/metric rule bank.
        executor = None
        if "behavioral" in channels:
            executor, resolved_executor_revision = _make_backend(
                FIXED_EXECUTOR, fake=fake_backends
            )
            if not fake_backends:
                declared_revisions = {
                    str(context["design"]["executor"]["revision"]) for context in contexts
                }
                if declared_revisions != {resolved_executor_revision}:
                    raise RuntimeError(
                        "resolved executor revision differs from the frozen metric designs"
                    )
            for constructor in constructor_models:
                for context in contexts:
                    metric_key = str(context["entry"]["metric_key"])
                    cell = staged[(str(constructor), metric_key)]
                    state = evaluate_behavioral_state_tables(
                        _CacheOnlyConstructor(), executor,
                        codebook_manifest=context["codebook"],
                        design_manifest=context["design"], target_metric_key=metric_key,
                        tier=tier, constructor_revision=revisions[str(constructor)],
                        executor_revision=str(context["design"]["executor"]["revision"]),
                        executor_readout_id=str(context["design"]["executor"]["readout_id"]),
                        cache=cache, query_batch_size=int(query_batch_size),
                    )
                    cell["behavioral_state"] = state
                    cell["behavioral_aggregations"] = {
                        arm: enumerate_exact_pool_values(
                            context["design"], channel="behavioral", tier=tier,
                            state_values=state["arms"][arm]["state_values"],
                            signatures=context["population"]["signatures"],
                        ) for arm in BEHAVIORAL_ARMS
                    }
            release_resident_engines()

        results = []
        for constructor in constructor_models:
            for context in contexts:
                metric_key = str(context["entry"]["metric_key"])
                cell = staged[(str(constructor), metric_key)]
                cross_spearman = None
                if "mcq" in channels and "behavioral" in channels:
                    cross_spearman = _cross_channel_spearman(
                        cell["mcq_aggregation"]["mean_prompt_value"],
                        cell["behavioral_aggregations"]["unconstrained"]["mean_prompt_value"],
                    )
                if "mcq" in channels:
                    results.append(write_mcq_output(
                        out_root, context, constructor=str(constructor),
                        constructor_revision=revisions[str(constructor)], tier=tier,
                        state_result=cell["mcq_state"],
                        aggregation=cell["mcq_aggregation"], cross_spearman=cross_spearman,
                    ))
                if "behavioral" in channels:
                    results.append(write_behavioral_output(
                        out_root, context, constructor=str(constructor),
                        constructor_revision=revisions[str(constructor)], tier=tier,
                        state_result=cell["behavioral_state"],
                        aggregations=cell["behavioral_aggregations"],
                        cross_spearman=cross_spearman,
                    ))
    base_result_count = len(results)
    upgrade_details = None
    if (
        tier == "B" and allow_auto_upgrade and not preflight_one_panel
        and bool(metrics_manifest.get("auto_upgrade_tier_a", False))
    ):
        if set(channels) != {"mcq", "behavioral"}:
            raise ValueError("automatic Tier-A upgrades require both declared channels")
        chosen, upgrade_selection = select_tier_a_upgrades(results, entries)
        if len(chosen) != 10:
            raise RuntimeError("automatic Tier-A upgrade selection did not yield ten metrics")
        upgrade_root = out_root / "tier_A_upgrades"
        upgrade_manifest_path = upgrade_root / "metrics_manifest.json"
        upgrade_manifest = {
            "schema": METRICS_MANIFEST_SCHEMA,
            "release": VALUE_BOUND_RELEASE,
            "auto_upgrade_tier_a": False,
            "selection_provenance": upgrade_selection,
            "metrics": [_materialize_entry_paths(entry, base) for entry in chosen],
        }
        _atomic_json(upgrade_manifest_path, upgrade_manifest)
        _atomic_json(out_root / "tier_A_upgrade_selection.json", upgrade_selection)
        upgrade_results = run_campaign(
            channels=channels, constructor_models=constructor_models,
            metrics_manifest_path=upgrade_manifest_path, tier="A",
            out_root=upgrade_root, fake_backends=fake_backends,
            query_batch_size=query_batch_size, cache_path=cache_path,
            allow_auto_upgrade=False, preflight_one_panel=False,
        )
        results.extend(upgrade_results)
        upgrade_details = {
            "selection_path": str(out_root / "tier_A_upgrade_selection.json"),
            "metrics_manifest_path": str(upgrade_manifest_path),
            "artifact_root": str(upgrade_root),
            "n_metrics": len(chosen),
            "n_results": len(upgrade_results),
            "selection": upgrade_selection,
        }
    results_frame = pd.DataFrame(results)
    _atomic_parquet(out_root / "results.parquet", results_frame)
    campaign = {
        "schema": CAMPAIGN_MANIFEST_SCHEMA, "release": VALUE_BOUND_RELEASE,
        "tier": tier, "channels": channels,
        "constructor_models": list(map(str, constructor_models)),
        "executor": FIXED_EXECUTOR,
        "metrics_manifest_path": str(Path(metrics_manifest_path).resolve()),
        "metrics_manifest_sha256": _file_sha256(metrics_manifest_path),
        "n_metrics": len(contexts), "n_results": len(results),
        "n_tier_results": base_result_count,
        "expected_n_tier_results": len(contexts) * len(constructor_models) * len(channels),
        "expected_n_results": (
            len(contexts) * len(constructor_models) * len(channels)
            + (0 if upgrade_details is None else int(upgrade_details["n_results"]))
        ),
        "cache_path": str(cache_path),
        "results_path": str(out_root / "results.parquet"),
        "selected_metrics": [context["entry"] for context in contexts],
        "scientific_or_model_qualification_gate_applied": False,
        "preflight_one_panel": bool(preflight_one_panel),
    }
    if upgrade_details is not None:
        campaign["automatic_tier_a_upgrades"] = upgrade_details
    if campaign["n_tier_results"] != campaign["expected_n_tier_results"]:
        raise RuntimeError("campaign did not produce every declared tier cell")
    if campaign["n_results"] != campaign["expected_n_results"]:
        raise RuntimeError("campaign did not produce every declared model/metric/channel cell")
    campaign["campaign_sha256"] = _payload_sha256(campaign)
    _atomic_json(out_root / "campaign_manifest.json", campaign)
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", nargs="+", choices=["mcq", "behavioral"], required=True)
    parser.add_argument("--constructor-models", nargs="+", required=True)
    parser.add_argument("--metrics-manifest", required=True)
    parser.add_argument("--tier", choices=["A", "B"], required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--cache-path")
    parser.add_argument("--metric-keys", nargs="+")
    parser.add_argument(
        "--disable-auto-upgrade", action="store_true",
        help=(
            "Run only the declared tier. Use this for independently scheduled model "
            "lanes, then select the shared Tier-A upgrades from consolidated results."
        ),
    )
    parser.add_argument("--preflight-one-panel", action="store_true")
    parser.add_argument("--fake-backends", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_campaign(
        channels=args.channels, constructor_models=args.constructor_models,
        metrics_manifest_path=args.metrics_manifest, tier=args.tier,
        out_root=args.out_root, fake_backends=bool(args.fake_backends),
        query_batch_size=int(args.query_batch_size), cache_path=args.cache_path,
        allow_auto_upgrade=not bool(args.disable_auto_upgrade),
        preflight_one_panel=bool(args.preflight_one_panel),
        metric_keys=args.metric_keys,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
