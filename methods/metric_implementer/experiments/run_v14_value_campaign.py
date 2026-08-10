"""Resumable v14 decoder-instrument campaign launcher.

GPU stages are deliberately split by resident model: ``constructor`` fills MCQ cells
and behavioral inductions for one decoder family, ``executor`` fills rule/probe cells
with the frozen 8B executor, and ``aggregate`` is CPU-only.  The split makes physical
GPU authorization explicit and preserves completed cells across process restarts.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import ImplementerConfig
from ..backends import LLMBackend
from ..vllm_backend import (
    CR3_BINARY_READOUT_ID,
    make_judge_backend,
    model_revision_id,
    release_resident_engines,
)
from .cr3_evidence_store import EvidenceCellStore, file_sha256
from .cr3_reconstruction_values import _bootstrap
from .cr3_sampled_value_certify import _codebook_menu
from .run_v13_value_campaign import (
    FIXED_EXECUTOR,
    _load_codebook_for_entry,
    _resolve_path,
    load_candidate_population,
    load_metrics_manifest,
    select_metric_entries,
    _target_entropy_for_entry,
)
from .v14_behavioral_channel import (
    BEHAVIORAL_ARMS,
    DEFAULT_TEMPLATE as DEFAULT_BEHAVIORAL_TEMPLATE,
    evaluate_behavioral_state_tables_v14,
    shuffled_state,
)
from .v14_audit import (
    AUDIT_FAMILIES,
    propose_family_audit,
    score_audit_ledger,
)
from .v14_mcq_channel import DEFAULT_MCQ_TEMPLATE, evaluate_mcq_state_tables_v14
from .v14_decoder_tuning import (
    select_dev_metrics,
    template_sha256,
    validate_shared_template,
)
from .v14_panel_design import (
    build_panel_design,
    canonical_sha256,
    ensure_teaching_label_balance,
    freeze_probe_split,
    validate_panel_design,
    validate_probe_split,
)
from .v14_value_bound import (
    aggregate_state_tables,
    classify_status,
    dkw_expected_best_gain_bound,
    fidelity_legibility_diagnostic,
    novelty_collapse_curves,
    record_rank_gain_bound,
    signatures_to_states,
    split_sample_cp_gain_bound,
    validate_state_tables,
    plugin_binary_mutual_information,
)
from .v14_tuning_evaluator import (
    aggregate_template_fitness,
    cached_probe_embeddings,
    freeze_reference_set,
    induce_behavioral_reference_templates,
    score_behavioral_reference_templates,
    score_mcq_reference_templates,
)
from .v14_preregistration import (
    build_preregistration,
    build_production_freeze,
    choose_qualified_decoder,
    evaluate_decoder_qualification,
    evaluate_sentinel_liveness,
)
from .v14_release_report import audit_release, write_release_outputs
from .v14_liveness_controls import (
    finish_liveness_executor_controls,
    run_liveness_constructor_controls,
)
from .v14_probe_extension import (
    append_extension_to_split, load_extension, load_task_candidates,
    score_extension_codebook, select_extension_texts, write_extension,
)
from .v14_scoring_lanes import (
    aggregate_fast_screening,
    build_promotion_manifest,
    fast_behavioral_label_permutation_null,
    fast_mcq_code_permutation_null,
    load_promotion_metric_keys,
    scoring_lane_policy,
    validate_lane_policy,
)


CAMPAIGN_SCHEMA = "cr3-v14-campaign-v1"
DESIGN_SCHEMA = "cr3-v14-metric-design-v1"
TEMPLATE_FREEZE_SCHEMA = "cr3-v14-template-freeze-v1"
FORBIDDEN_SK3_GPUS = {1, 2, 3, 4}
DEFAULT_DECODER_MODELS = {
    "qwen": "Qwen/Qwen2.5-14B-Instruct",
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
}
FALLBACK_DECODER_MODELS = {
    "qwen": "Qwen/Qwen2.5-32B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
DEFAULT_SENTINEL_METRIC_KEYS = (
    "humor_R3_metric0", "humor_R3_metric10", "humor_R3_metric11",
    "humor_R3_metric12", "humor_R3_metric34", "humor_R3_metric50",
)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(value)).strip("_")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def parse_physical_gpu_ids(values: Sequence[str] | str | None) -> tuple[int, ...]:
    if values is None:
        raw = os.environ.get("V14_PHYSICAL_GPUS", "")
        parts = raw.split(",") if raw else []
    elif isinstance(values, str):
        parts = values.split(",")
    else:
        parts = [part for value in values for part in str(value).split(",")]
    try:
        ids = tuple(int(part.strip()) for part in parts if part.strip())
    except ValueError as exc:
        raise ValueError("physical GPU IDs must be comma-separated integers") from exc
    if len(ids) != len(set(ids)) or any(value < 0 for value in ids):
        raise ValueError("physical GPU IDs must be unique nonnegative integers")
    return ids


def assert_gpu_authorized(
    physical_gpu_ids: Sequence[int], *, hostname: str | None = None, fake_backends: bool = False,
) -> None:
    if fake_backends:
        return
    host = str(hostname or socket.gethostname()).lower()
    ids = set(map(int, physical_gpu_ids))
    if not ids:
        raise RuntimeError("GPU phases require explicit physical IDs via --physical-gpus")
    if (host.startswith("sk3") or host.startswith("skampere3")) and ids.intersection(
        FORBIDDEN_SK3_GPUS
    ):
        raise RuntimeError(
            "hard safety stop: sk3 physical GPUs 1,2,3,4 are permanently forbidden for v14"
        )
    visible = tuple(
        int(value) for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip().isdigit()
    )
    if visible and tuple(map(int, physical_gpu_ids)) != visible:
        raise RuntimeError(
            f"declared physical GPUs {tuple(physical_gpu_ids)} do not match "
            f"CUDA_VISIBLE_DEVICES={visible}"
        )


def _backend(model: str, *, fake: bool):
    cfg = ImplementerConfig()
    cfg.vllm_fake = bool(fake)
    cfg.vllm_gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90"))
    cfg.vllm_tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
    cfg.vllm_max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
    if os.environ.get("METRIC_IMPLEMENTER_LFS_HOME"):
        cfg.vllm_lfs_home = os.environ["METRIC_IMPLEMENTER_LFS_HOME"]
    overrides = json.loads(os.environ.get("V14_MODEL_PATH_OVERRIDES_JSON", "{}"))
    runtime_model = str(overrides.get(str(model), str(model)))
    result = make_judge_backend(runtime_model, cfg, 0.0)
    revision = str(model) if fake else model_revision_id(runtime_model)
    return result, revision


def _probe_ids(probe_texts: Sequence[str]) -> list[str]:
    return [
        hashlib.sha256(f"{index}\x1f{text}".encode("utf-8")).hexdigest()
        for index, text in enumerate(probe_texts)
    ]


def _certification_probe_embeddings(
    probe_texts: Sequence[str], *, cache_path: str | Path, fake_backends: bool,
) -> np.ndarray:
    if not fake_backends:
        return cached_probe_embeddings(probe_texts, cache_path=cache_path)
    rows = []
    for text in map(str, probe_texts):
        digest = hashlib.sha256(("fake-bge\x1f" + text).encode("utf-8")).digest()
        vector = np.frombuffer(digest, dtype=np.uint8).astype(float) - 127.5
        rows.append(vector / np.linalg.norm(vector))
    return np.vstack(rows)


def _codebook_signatures(codebook: Mapping[str, object]) -> tuple[list[str], np.ndarray]:
    keys = sorted(map(str, codebook["metrics"]))
    rows = []
    for key in keys:
        bootstrap = _bootstrap(codebook["metrics"][key]["bootstrap_path"])
        rows.append(np.asarray(bootstrap["target"], dtype=float))
    matrix = np.vstack(rows)
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)):
        raise RuntimeError("codebook bootstrap targets are invalid")
    return keys, matrix


def _design_path(out_root: Path, metric_key: str) -> Path:
    return out_root / "designs" / _safe(metric_key) / "design_manifest.json"


def build_probe_extensions(
    *, metrics_manifest_path: str | Path, corpus_manifest_path: str | Path,
    out_root: str | Path, run_sha: str, fake_backends: bool,
    physical_gpu_ids: Sequence[int], query_batch_size: int = 2048,
) -> dict:
    """Select and score one shared 90-text append-only extension per task."""
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    manifest, base = load_metrics_manifest(metrics_manifest_path)
    entries = select_metric_entries({**manifest, "selection": {}}, base)
    corpus_path = Path(corpus_manifest_path).resolve()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    task_specs = corpus.get("tasks", corpus)
    by_task = {}
    for entry in entries:
        by_task.setdefault(str(entry["task"]), entry)
    executor, revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
    destination = Path(out_root).resolve()
    rows = []
    try:
        for task, entry in sorted(by_task.items()):
            output = destination / f"{task}.npz"
            if output.is_file():
                extension = load_extension(output)
                rows.append({"task": task, "path": str(output), "sha256": extension["sha256"]})
                continue
            codebook = _load_codebook_for_entry(entry, base)
            target_key = str(entry["metric_key"])
            existing = list(map(str, _bootstrap(
                codebook["metrics"][target_key]["bootstrap_path"]
            )["probe_texts"]))
            candidates = load_task_candidates(task_specs[task], base=corpus_path.parent)
            texts = select_extension_texts(
                candidates, existing_texts=existing, task=task, run_sha=run_sha,
            )
            payload = score_extension_codebook(
                executor, codebook=codebook, extension_texts=texts,
                executor_revision=revision, readout_id=CR3_BINARY_READOUT_ID,
                query_batch_size=query_batch_size,
            )
            write_extension(output, payload)
            extension = load_extension(output)
            rows.append({"task": task, "path": str(output), "sha256": extension["sha256"]})
    finally:
        release_resident_engines()
    result = {
        "schema": "cr3-v14-probe-extension-index-v1", "run_sha": str(run_sha),
        "base_n": 300, "extension_n": 90, "combined_n": 390,
        "executor": FIXED_EXECUTOR, "executor_revision": revision, "tasks": rows,
    }
    result["sha256"] = canonical_sha256(result)
    _atomic_json(destination / "index.json", result)
    return result


def _panel_balance_candidates(
    entries: Sequence[Mapping[str, object]], *, base: Path, run_sha: str,
) -> tuple[list[dict], list[dict]]:
    """Filter metrics whose frozen teaching split cannot meet the hard 3--5 balance."""
    codebooks: dict[tuple[str, str, str], dict] = {}
    eligible = []
    excluded = []
    for source in entries:
        row = dict(source)
        codebook_key = (
            str(row.get("codebook_path")), str(row.get("assets_root")),
            str(row.get("codebook_layout") or "production"),
        )
        if codebook_key not in codebooks:
            codebooks[codebook_key] = _load_codebook_for_entry(row, base)
        codebook = codebooks[codebook_key]
        metric_key = str(row["metric_key"])
        target = _bootstrap(codebook["metrics"][metric_key]["bootstrap_path"])
        target_bits = (np.asarray(target["target"], dtype=float) > 0.5).astype(np.uint8)
        probe_texts = list(map(str, target["probe_texts"]))
        split = freeze_probe_split(
            _probe_ids(probe_texts), run_sha=str(run_sha), metric_key=metric_key,
        )
        global_yes = int(np.sum(target_bits))
        global_no = int(len(target_bits) - global_yes)
        if min(global_yes, global_no) >= 3:
            split = ensure_teaching_label_balance(split, target_bits)
        teaching = np.asarray(split["teaching"]["indices"], dtype=int)
        yes = int(np.sum(target_bits[teaching]))
        no = int(len(teaching) - yes)
        report = {
            "global_yes": global_yes, "global_no": global_no,
            "teaching_yes": yes, "teaching_no": no,
            "minimum_per_class_required": 3,
            "eligible": min(global_yes, global_no, yes, no) >= 3,
        }
        row["target_entropy_bits"] = _binary_entropy(target_bits)
        row["v14_panel_balance"] = report
        (eligible if report["eligible"] else excluded).append(row)
    return eligible, excluded


def _select_v14_certification_metrics(
    candidates: Sequence[Mapping[str, object]], *, total: int = 35,
    required_metric_keys: Sequence[str] = DEFAULT_SENTINEL_METRIC_KEYS,
) -> tuple[list[dict], dict[str, int]]:
    """Select certification metrics while reserving one feasible dev metric per task."""
    grouped: dict[str, list[dict]] = {}
    for source in candidates:
        grouped.setdefault(str(source["task"]), []).append(dict(source))
    if len(grouped) != 7:
        raise ValueError("v14 certification requires all seven task families")
    required = set(map(str, required_metric_keys))
    by_key = {str(row["metric_key"]): row for row in candidates}
    if not required.issubset(by_key):
        raise ValueError("a required sentinel is not panel-feasible for certification")
    required_counts = {
        task: sum(str(row["metric_key"]) in required for row in rows)
        for task, rows in grouped.items()
    }
    # Metric-level holdout is load-bearing for decoder tuning.  Reserve one
    # panel-feasible metric from every task before allocating certification
    # quotas so the later seven-task development population is constructible.
    capacities = {task: len(rows) - 1 for task, rows in grouped.items()}
    if any(
        capacities[task] < max(1, required_counts[task])
        for task in grouped
    ):
        raise ValueError(
            "certification cannot reserve one panel-feasible development metric per task"
        )
    quotas = {
        task: max(min(5, capacities[task]), required_counts[task])
        for task, rows in grouped.items()
    }
    while sum(quotas.values()) < int(total):
        choices = [
            task for task in grouped if quotas[task] < capacities[task]
        ]
        if not choices:
            raise ValueError("fewer than 35 panel-feasible certification metrics remain")
        task = min(choices, key=lambda name: (
            -(capacities[name] - quotas[name]), name,
        ))
        quotas[task] += 1
    while sum(quotas.values()) > int(total):
        choices = [
            task for task in grouped if quotas[task] > max(1, required_counts[task])
        ]
        if not choices:
            raise ValueError("required sentinels leave no valid 35-metric quota allocation")
        task = min(choices, key=lambda name: (
            len(grouped[name]) - quotas[name], name,
        ))
        quotas[task] -= 1
    if sum(quotas.values()) != int(total):
        raise RuntimeError("v14 certification quota allocation is invalid")

    selected = []
    for task in sorted(grouped):
        rows = sorted(grouped[task], key=lambda row: (
            float(row["target_entropy_bits"]),
            hashlib.sha256(str(row["metric_key"]).encode("utf-8")).hexdigest(),
        ))
        quota = quotas[task]
        required_rows = [row for row in rows if str(row["metric_key"]) in required]
        remaining = [row for row in rows if str(row["metric_key"]) not in required]
        needed = quota - len(required_rows)
        positions = (
            [] if needed == 0 else [len(remaining) // 2] if needed == 1 else
            [int(round(rank * (len(remaining) - 1) / (needed - 1))) for rank in range(needed)]
        )
        task_rows = [*required_rows, *(remaining[position] for position in positions)]
        for rank, source in enumerate(task_rows):
            row = dict(source)
            row["target_entropy_quintile"] = int(
                round(rank * 4 / max(1, quota - 1)) + 1
            )
            row["v14_certification_task_quota"] = int(quota)
            selected.append(row)
    return selected, quotas


def build_designs(
    *, metrics_manifest_path: str | Path, out_root: str | Path, run_sha: str,
    metric_keys: Sequence[str] | None = None, probe_extension_root: str | Path | None = None,
    scoring_lane: str = "cert", promotion_manifest_path: str | Path | None = None,
) -> list[dict]:
    """CPU-only v14 split/panel freeze over append-only 390-probe assets."""
    lane_policy = scoring_lane_policy(scoring_lane)
    if promotion_manifest_path is not None:
        if scoring_lane != "cert" or metric_keys is not None:
            raise ValueError("a promotion manifest exclusively defines the CERT metric population")
        metric_keys = load_promotion_metric_keys(promotion_manifest_path)
    manifest, base = load_metrics_manifest(metrics_manifest_path)
    requested = None if metric_keys is None else set(map(str, metric_keys))
    selection_report = None
    if requested is not None:
        unselected_manifest = {**manifest, "selection": {}}
        entries = select_metric_entries(unselected_manifest, base)
        entries = [entry for entry in entries if str(entry["metric_key"]) in requested]
        if {str(entry["metric_key"]) for entry in entries} != requested:
            raise ValueError("requested metric key is absent from the selected manifest")
    elif scoring_lane == "fast":
        entries = select_metric_entries({**manifest, "selection": {}}, base)
        selection_report = {
            "mode": "fast_lane_all_manifest_metrics",
            "n_selected": len(entries),
            "claim_role": "screening_only",
        }
    elif (manifest.get("selection") or {}).get("mode") == "target_entropy_quintiles":
        feasible, excluded = _panel_balance_candidates(
            manifest["metrics"], base=base, run_sha=str(run_sha),
        )
        entries, quotas = _select_v14_certification_metrics(feasible, total=35)
        selection_report = {
            "mode": "target_entropy_quintiles_after_hard_panel_balance_filter",
            "n_eligible": len(feasible), "n_excluded": len(excluded),
            "n_selected": len(entries), "task_quotas": quotas,
            "required_sentinel_metric_keys": list(DEFAULT_SENTINEL_METRIC_KEYS),
            "quota_reallocation": (
                "reserve one panel-feasible development metric per task; start at min(5, "
                "the remaining feasible task population), then assign deficits to the task "
                "with the largest remaining feasible population; stable task-name tie-break"
            ),
            "excluded": [{
                "metric_key": row["metric_key"], **row["v14_panel_balance"],
            } for row in excluded],
        }
    else:
        entries = select_metric_entries(manifest, base)
    by_task_for_caps: dict[str, list[str]] = {}
    for entry in entries:
        by_task_for_caps.setdefault(str(entry["task"]), []).append(str(entry["metric_key"]))
    cap_sentinels = set(DEFAULT_SENTINEL_METRIC_KEYS)
    for task, keys in by_task_for_caps.items():
        cap_sentinels.add(min(
            keys, key=lambda key: hashlib.sha256(
                f"{run_sha}\x1f{task}\x1fcap-sentinel\x1f{key}".encode()
            ).hexdigest(),
        ))
    out = Path(out_root).resolve()
    results = []
    fast_exclusions = []
    for entry in entries:
        metric_key = str(entry["metric_key"])
        destination = _design_path(out, metric_key)
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            validate_metric_design(existing)
            if str(existing.get("scoring_lane", {}).get("lane", "cert")) != scoring_lane:
                raise RuntimeError("FAST and CERT designs require separate output roots")
            results.append(existing)
            continue
        codebook = _load_codebook_for_entry(entry, base)
        target_bootstrap = _bootstrap(codebook["metrics"][metric_key]["bootstrap_path"])
        base_probe_texts = list(map(str, target_bootstrap["probe_texts"]))
        if len(base_probe_texts) != 300:
            raise ValueError(f"v14 requires the 300-probe bank for {metric_key}")
        if probe_extension_root is None:
            raise ValueError("v14.1 design requires --probe-extension-root with 90 appended probes")
        extension_path = Path(probe_extension_root) / f"{entry['task']}.npz"
        extension = load_extension(extension_path)
        extension_keys = list(map(str, extension["metric_keys"]))
        if metric_key not in extension_keys:
            raise ValueError(f"probe extension lacks {metric_key}")
        probe_texts = [*base_probe_texts, *map(str, extension["texts"])]
        ids = _probe_ids(probe_texts)
        base_scores = np.asarray(target_bootstrap["target"], dtype=float)
        target_scores = np.concatenate([
            base_scores,
            np.asarray(extension["scores"], dtype=float)[extension_keys.index(metric_key)],
        ])
        target_bits = (target_scores > 0.5).astype(np.uint8)
        base_ids = ids[:300]
        try:
            split = ensure_teaching_label_balance(
                freeze_probe_split(
                    base_ids, run_sha=run_sha, metric_key=metric_key,
                    split_sizes={"teaching": 120, "decoder_development": 30, "heldout": 150},
                ),
                target_bits[:300],
            )
        except ValueError as exc:
            if scoring_lane != "fast":
                raise
            fast_exclusions.append({
                "metric_key": metric_key, "task": str(entry["task"]),
                "reason": "teaching_label_balance_infeasible", "detail": str(exc),
            })
            continue
        split = append_extension_to_split(split, ids)
        validate_probe_split(split)
        codebook_keys, base_signatures = _codebook_signatures(codebook)
        if codebook_keys != extension_keys:
            raise ValueError(f"probe extension codebook identity mismatch for {entry['task']}")
        signatures = np.concatenate([
            base_signatures, np.asarray(extension["scores"], dtype=float)
        ], axis=1)
        target_index = codebook_keys.index(metric_key)
        is_cap_sentinel = scoring_lane == "cert" and metric_key in cap_sentinels
        panel_size = 8 if is_cap_sentinel else 6
        try:
            panel = build_panel_design(
                signatures, target_index=target_index,
                teaching_indices=split["teaching"]["indices"], run_sha=run_sha,
                metric_key=metric_key, probe_ids=ids,
                decoder_families=("qwen", "llama", "mistral"),
                panel_size=panel_size,
            )
        except (ValueError, RuntimeError) as exc:
            if scoring_lane != "fast":
                raise
            fast_exclusions.append({
                "metric_key": metric_key, "task": str(entry["task"]),
                "reason": "panel_infeasible", "detail": str(exc),
            })
            continue
        materialized_entry = dict(entry)
        for field in ("codebook_path", "assets_root", "candidate_bank_path"):
            if materialized_entry.get(field):
                materialized_entry[field] = str(_resolve_path(materialized_entry[field], base))
        if materialized_entry.get("candidate_sources"):
            materialized_entry["candidate_sources"] = [
                {
                    **dict(source),
                    "path": str(_resolve_path(source["path"], base)),
                }
                for source in materialized_entry["candidate_sources"]
            ]
        materialized_entry.pop("candidate_source_set", None)
        core = {
            "schema": DESIGN_SCHEMA,
            "run_sha": str(run_sha),
            "metric_key": metric_key,
            "task": str(entry["task"]),
            "level": str(entry["level"]),
            "metric": str(entry["metric"]),
            "source_metrics_manifest": str(Path(metrics_manifest_path).resolve()),
            "source_metrics_manifest_sha256": file_sha256(metrics_manifest_path),
            "entry": materialized_entry,
            "scoring_lane": lane_policy,
            "promotion_manifest": (
                None if promotion_manifest_path is None else {
                    "path": str(Path(promotion_manifest_path).resolve()),
                    "sha256": file_sha256(promotion_manifest_path),
                    "fresh_remeasurement": True,
                }
            ),
            "probe_split": split,
            "panel_design": panel,
            "codebook_metric_keys": codebook_keys,
            "codebook_signatures_sha256": hashlib.sha256(
                np.ascontiguousarray((signatures > 0.5).astype(np.uint8)).tobytes()
            ).hexdigest(),
            "target_entropy_on_h_bits": _binary_entropy(
                target_scores[split["heldout"]["indices"]] > 0.5
            ),
            "probe_extension": {
                "path": str(extension_path.resolve()), "sha256": extension["sha256"],
                "append_only": True, "base_n": 300, "extension_n": 90,
            },
            "throughput_design": {
                "cap_sentinel": is_cap_sentinel,
                "panel_size": panel_size,
                "menu_permutations": 8 if is_cap_sentinel else 4,
                "state_scope": lane_policy["state_scope"],
                "headline_cap": (
                    "exact_enumerated" if is_cap_sentinel else "target_entropy"
                ),
            },
            "executor": {
                "model": str(target_bootstrap.get("executor_model", FIXED_EXECUTOR)),
                "revision": str(target_bootstrap.get("executor_model_revision", "unknown")),
                "readout_id": str(target_bootstrap.get("readout_id", CR3_BINARY_READOUT_ID)),
            },
        }
        core["design_manifest_sha256"] = canonical_sha256(core)
        _atomic_json(destination, core)
        results.append(core)
    index = {
        "schema": "cr3-v14-design-index-v1",
        "run_sha": str(run_sha), "scoring_lane": lane_policy,
        "metrics": [{
            "metric_key": row["metric_key"],
            "path": str(_design_path(out, row["metric_key"])),
            "sha256": row["design_manifest_sha256"],
        } for row in results],
    }
    if selection_report is not None:
        if scoring_lane == "fast":
            selection_report["n_selected"] = len(results)
            selection_report["n_excluded"] = len(fast_exclusions)
            selection_report["excluded_metrics_path"] = str(out / "fast_design_exclusions.json")
        index["selection"] = selection_report
    index["index_sha256"] = canonical_sha256(index)
    _atomic_json(out / "design_index.json", index)
    if scoring_lane == "fast":
        _atomic_json(out / "fast_design_exclusions.json", {
            "schema": "cr3-v14-fast-design-exclusions-v1",
            "lane": "fast", "rows": fast_exclusions,
            "campaign_continues_on_metric_failure": True,
        })
    return results


def _binary_entropy(bits: Sequence[int]) -> float:
    vector = np.asarray(bits, dtype=np.uint8)
    probability = float(np.mean(vector))
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(-probability * np.log2(probability) - (1.0 - probability) * np.log2(1.0 - probability))


def validate_metric_design(payload: Mapping[str, object]) -> None:
    if payload.get("schema") != DESIGN_SCHEMA:
        raise ValueError("unsupported v14 metric design")
    validate_probe_split(payload["probe_split"])
    validate_panel_design(payload["panel_design"])
    if "scoring_lane" in payload:
        validate_lane_policy(payload["scoring_lane"])
    core = dict(payload)
    observed = str(core.pop("design_manifest_sha256", ""))
    if observed != canonical_sha256(core):
        raise ValueError("v14 metric design checksum mismatch")
    teaching = set(payload["probe_split"]["teaching"]["indices"])
    if teaching != set(payload["panel_design"]["teaching_indices"]):
        raise ValueError("panel design does not use the frozen teaching split")


def load_designs(out_root: str | Path, metric_keys: Sequence[str] | None = None) -> list[dict]:
    root = Path(out_root).resolve()
    index = json.loads((root / "design_index.json").read_text(encoding="utf-8"))
    requested = None if metric_keys is None else set(map(str, metric_keys))
    rows = []
    for entry in index["metrics"]:
        if requested is not None and str(entry["metric_key"]) not in requested:
            continue
        payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        validate_metric_design(payload)
        rows.append(payload)
    if requested is not None and {row["metric_key"] for row in rows} != requested:
        raise ValueError("requested design is missing")
    return rows


def prepare_development_population(
    *, certified_out_root: str | Path, dev_metrics_manifest_path: str | Path,
    run_sha: str, probe_extension_root: str | Path | None = None,
    dev_min_tasks: int = 7,
) -> dict:
    """Freeze eight metric-held-out development metrics and sparse GEPA references."""
    certified = load_designs(certified_out_root)
    certified_keys = [str(row["metric_key"]) for row in certified]
    manifest, base = load_metrics_manifest(dev_metrics_manifest_path)
    candidates, panel_ineligible = _panel_balance_candidates(
        manifest["metrics"], base=base, run_sha=f"{run_sha}:dev",
    )
    by_task = {}
    for row in candidates:
        by_task.setdefault(str(row["task"]), []).append(row)
    for task_rows in by_task.values():
        ranked = sorted(task_rows, key=lambda row: (
            float(row["target_entropy_bits"]), str(row["metric_key"]),
        ))
        denominator = max(1, len(ranked) - 1)
        for position, row in enumerate(ranked):
            row["target_entropy_quintile"] = min(4, int(5 * position / (denominator + 1)))
    # Overselect a candidate pool, build it failure-tolerantly (fast lane records
    # per-metric infeasibility instead of aborting: dev panels use the ":dev" salt,
    # so feasibility under the main salt does not imply feasibility here), then
    # keep the first eight built designs with maximum task spread.
    # Feasibility under the ":dev" salt is unknowable before building, so build the
    # ENTIRE candidate pool (the caller bounds cost via the manifest size) and pick
    # the final eight from what actually builds.
    selected = select_dev_metrics(
        candidates,
        certified_metric_keys=[*certified_keys, *DEFAULT_SENTINEL_METRIC_KEYS],
        run_sha=run_sha, n_dev=None, min_tasks=int(dev_min_tasks),
    )
    for row in selected:
        for field in ("codebook_path", "assets_root", "candidate_bank_path"):
            if row.get(field):
                row[field] = str(_resolve_path(row[field], base))
        if row.get("candidate_sources"):
            for source in row["candidate_sources"]:
                source["path"] = str(_resolve_path(source["path"], base))
    dev_root = Path(certified_out_root).resolve() / "development"
    selected_manifest = {
        "schema": "cr3-value-bound-metrics-v13.1",
        "release": "v14-development-only-no-claims",
        "selection_provenance": (
            "overselected candidate pool; final eight chosen from successful builds "
            "with deterministic task-spread; disjoint from certified keys"
        ),
        "metrics": selected,
    }
    selected_path = dev_root / "dev_metrics.json"
    _atomic_json(selected_path, selected_manifest)
    built = build_designs(
        metrics_manifest_path=selected_path, out_root=dev_root, run_sha=f"{run_sha}:dev",
        probe_extension_root=probe_extension_root, scoring_lane="fast",
    )
    by_task: dict[str, list[dict]] = {}
    for design in built:
        by_task.setdefault(str(design["task"]), []).append(design)
    for rows in by_task.values():
        rows.sort(key=lambda row: hashlib.sha256(
            f"{run_sha}\x1fdev-final\x1f{row['metric_key']}".encode()).hexdigest())
    final: list[dict] = []
    task_cycle = sorted(by_task, key=lambda task: hashlib.sha256(
        f"{run_sha}\x1fdev-task\x1f{task}".encode()).hexdigest())
    depth = 0
    while len(final) < 8 and depth < 16:
        for task in task_cycle:
            rows = by_task[task]
            if depth < len(rows) and len(final) < 8:
                final.append(rows[depth])
        depth += 1
    if len(final) < 8:
        raise RuntimeError(
            f"only {len(final)} of the overselected dev candidates built successfully"
        )
    if len({str(row['task']) for row in final}) < int(dev_min_tasks):
        raise RuntimeError("built development designs span fewer tasks than required")
    final_keys = {str(row["metric_key"]) for row in final}
    # Rewrite the development design index to exactly the final eight designs so
    # downstream loaders (tuning contexts) see the frozen dev population only.
    dev_index_path = dev_root / "design_index.json"
    dev_index = json.loads(dev_index_path.read_text(encoding="utf-8"))
    dev_index["metrics"] = [
        row for row in dev_index["metrics"] if str(row["metric_key"]) in final_keys
    ]
    selection_note = dict(dev_index.get("selection") or {})
    selection_note["final_development_metric_keys"] = sorted(final_keys)
    dev_index["selection"] = selection_note
    dev_index.pop("index_sha256", None)
    dev_index["index_sha256"] = canonical_sha256(dev_index)
    _atomic_json(dev_index_path, dev_index)
    designs = final
    reference_rows = []
    for design in designs:
        context = _metric_context(design)
        ddec = list(design["probe_split"]["decoder_development"]["indices"])
        target = (np.asarray(context["target"]["target"], dtype=float)[ddec] > 0.5).astype(int)
        signatures = np.asarray(context["population"]["signatures"], dtype=float)
        reference_values = np.asarray([
            plugin_binary_mutual_information(target, row[ddec] > 0.5)
            for row in signatures
        ], dtype=float)
        # Near/far transfer needs only teaching examples and D_dec.  Do not even
        # embed certification H during adaptive development: H is first touched
        # after the template freeze.
        teaching = list(map(int, design["probe_split"]["teaching"]["indices"]))
        embedding_indices = sorted(set(teaching).union(ddec))
        embedding_texts = [context["probe_texts"][index] for index in embedding_indices]
        embedding_path = dev_root / "embeddings" / f"{_safe(design['metric_key'])}.npz"
        subset_embeddings = cached_probe_embeddings(
            embedding_texts, cache_path=embedding_path,
        )
        # freeze_reference_set aligns embeddings against signatures.shape[1] (the
        # codebook probe axis), which can differ from len(probe_texts) when the
        # 90-probe extension is appended; teaching+D_dec indices live in both.
        embeddings = np.zeros(
            (signatures.shape[1], subset_embeddings.shape[1]), dtype=float,
        )
        embeddings[np.asarray(embedding_indices, dtype=int)] = subset_embeddings
        reference = freeze_reference_set(
            metric_key=str(design["metric_key"]),
            panel_design=design["panel_design"],
            candidate_signatures=signatures,
            target_signature=context["target"]["target"],
            decoder_development_indices=ddec,
            candidate_reference_values=reference_values,
            probe_embeddings=embeddings,
        )
        path = dev_root / "references" / f"{_safe(design['metric_key'])}.json"
        _atomic_json(path, reference)
        reference_rows.append({
            "metric_key": str(design["metric_key"]), "path": str(path),
            "sha256": canonical_sha256(reference),
        })
    index = {
        "schema": "cr3-v14-development-reference-index-v1",
        "certified_metric_keys": certified_keys,
        "development_metric_keys": [str(row["metric_key"]) for row in designs],
        "sentinel_metric_keys": list(DEFAULT_SENTINEL_METRIC_KEYS),
        "metric_level_disjoint": not bool(
            set(certified_keys).intersection(row["metric_key"] for row in designs)
        ),
        "sentinel_disjoint": not bool(
            set(DEFAULT_SENTINEL_METRIC_KEYS).intersection(
                row["metric_key"] for row in designs
            )
        ),
        "selected_manifest": str(selected_path),
        "selected_manifest_sha256": file_sha256(selected_path),
        "references": reference_rows,
        "panel_balance_exclusions": [{
            "metric_key": row["metric_key"], **row["v14_panel_balance"],
        } for row in panel_ineligible],
    }
    if not index["metric_level_disjoint"] or not index["sentinel_disjoint"]:
        raise RuntimeError("development overlaps the certification or sentinel population")
    index["index_sha256"] = canonical_sha256(index)
    _atomic_json(dev_root / "reference_index.json", index)
    return index


def _development_contexts(out_root: str | Path) -> list[dict]:
    root = Path(out_root).resolve() / "development"
    designs = load_designs(root)
    contexts = []
    for design in designs:
        context = _metric_context(design)
        reference = json.loads(
            (root / "references" / f"{_safe(design['metric_key'])}.json").read_text()
        )
        menu = context["menu"]
        contexts.append({
            "metric_key": str(design["metric_key"]),
            "noun": str(context["codebook"]["reconstruction_noun"]),
            "probe_texts": context["probe_texts"],
            "target_signature": np.asarray(context["target"]["target"], dtype=float),
            "target_description": str(menu["entry"]["target_description"]),
            "distractors": menu["distractors"],
            "reference_set": reference,
            "executor_revision": str(design["executor"]["revision"]),
            "readout_id": str(design["executor"]["readout_id"]),
        })
    return contexts


def run_decoder_tuning(
    *, out_root: str | Path, channel: str, arm: str,
    decoder_models: Mapping[str, str], proposer_model: str,
    fake_backends: bool, physical_gpu_ids: Sequence[int], query_batch_size: int = 2048,
    max_metric_calls: int = 240,
) -> dict:
    """Run one shared-template search with OFFICIAL GEPA (github gepa 0.1.4).

    The in-house bounded loop was deprecated 2026-07-19 (verbatim copy in
    ``archive/inhouse_gepa_deprecated.py``). This generalizes ``official_gepa_decoder_tune.py``
    to all three decoder families and the behavioral channel. Selection signal = SEARCH-split
    normalized fitness ONLY (per dev-metric instance, so the Pareto frontier is meaningful);
    held-out transfer and gate flags are computed POST-HOC over every distinct valid template
    for the frozen report (same discipline as the in-house run it replaces).
    """
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    if channel not in {"mcq", "behavioral"}:
        raise ValueError("tuning channel must be mcq or behavioral")
    if channel == "behavioral" and arm not in BEHAVIORAL_ARMS:
        raise ValueError("behavioral tuning requires one declared arm")
    contexts = _development_contexts(out_root)
    if len(contexts) != 8:
        raise RuntimeError("v14 tuning requires exactly eight development metrics")

    import gepa
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    store_path = Path(out_root) / "development" / "tuning_cells.sqlite"
    seed_template = DEFAULT_MCQ_TEMPLATE if channel == "mcq" else DEFAULT_BEHAVIORAL_TEMPLATE
    required_fields = (
        ("noun", "examples", "choices", "labels") if channel == "mcq"
        else ("noun", "feature_table", "examples", "arm_instruction")
    )
    forbidden = []
    for context in contexts:
        forbidden.extend([context["metric_key"], context["target_description"]])
        forbidden.extend(
            str(item.get("description", "") if isinstance(item, Mapping) else item.description)
            for item in context["distractors"]
        )
    name = "mcq" if channel == "mcq" else f"behavioral__{arm}"
    run_dir = Path(out_root) / "development" / "tuning" / f"{name}_official_gepa"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "proposals.jsonl"
    placeholders = ", ".join("{" + field + "}" for field in required_fields)
    constraint = (
        "HARD CONSTRAINTS for any rewritten template (violations score -1): preserve EVERY "
        f"format placeholder from the current template EXACTLY — {placeholders} — keep them as "
        "literal curly-brace fields; do NOT mention any metric name, description, or example "
        "content; do NOT add exemplars; return only the template text."
    )

    def score_templates(templates, batch_contexts):
        # Mirror the retired evaluate_batch structure exactly: per-family decoder scoring over
        # the shared EvidenceCellStore; behavioral additionally executes the induced rules with
        # the FIXED_EXECUTOR after all families have been induced.
        rows = []
        behavioral_inductions = {}
        with EvidenceCellStore(store_path) as store:
            for family, model in decoder_models.items():
                decoder, decoder_revision = _backend(model, fake=fake_backends)
                try:
                    if channel == "mcq":
                        rows.extend(score_mcq_reference_templates(
                            decoder, templates=templates, contexts=batch_contexts,
                            decoder_family=family, constructor_revision=decoder_revision,
                            store=store, query_batch_size=query_batch_size,
                        ))
                    else:
                        behavioral_inductions[family] = induce_behavioral_reference_templates(
                            decoder, templates=templates, arm=arm, contexts=batch_contexts,
                            decoder_family=family, decoder_revision=decoder_revision,
                            store=store,
                        )
                finally:
                    release_resident_engines()
            if channel == "behavioral":
                executor, executor_revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
                try:
                    for family in decoder_models:
                        rows.extend(score_behavioral_reference_templates(
                            executor, templates=templates, arm=arm, contexts=batch_contexts,
                            induction_rows=behavioral_inductions[family],
                            executor_revision=executor_revision,
                            readout_id=CR3_BINARY_READOUT_ID, store=store,
                            query_batch_size=query_batch_size, decoder_family=family,
                        ))
                finally:
                    release_resident_engines()
        return rows

    def per_context_search_fitness(rows, metric_key):
        vals = [
            float(row["normalized_fitness"]) for row in rows
            if row["reference_split"] == "search" and str(row["metric_key"]) == str(metric_key)
        ]
        return float(np.mean(vals)) if vals else float("-inf")

    def contrast_feedback(rows, metric_key):
        search = [
            row for row in rows if row["reference_split"] == "search"
            and str(row["metric_key"]) == str(metric_key)
        ]
        if not search:
            return "no search rows for this development metric"
        best = max(search, key=lambda row: float(row["normalized_fitness"]))
        worst = min(search, key=lambda row: float(row["normalized_fitness"]))
        mean = float(np.mean([float(row["normalized_fitness"]) for row in search]))
        return (
            f"mean search fitness {mean:+.3f}; best state {best['state']} "
            f"fitness {float(best['normalized_fitness']):+.3f}; worst state {worst['state']} "
            f"fitness {float(worst['normalized_fitness']):+.3f}. Fitness is target lift over the "
            "strongest control, normalized; improve evidence-routed contrastive decoding "
            "without naming any metric content."
        )

    def log_proposal(template, batch_contexts, scores, invalid=None):
        with open(log_path, "a") as handle:
            handle.write(json.dumps({
                "ts": time.time(), "template_sha256": template_sha256(template),
                "template": template,
                "metrics": [str(item["metric_key"]) for item in batch_contexts],
                "scores": [None if not np.isfinite(value) else float(value) for value in scores],
                "invalid": invalid,
            }) + "\n")

    class V14DecoderAdapter(GEPAAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            template = str(next(iter(candidate.values())))
            try:
                validate_shared_template(
                    template, forbidden_strings=forbidden, required_fields=required_fields,
                )
            except Exception as exc:
                scores = [-1.0] * len(batch)
                trajs = ([{"data": item, "full_assistant_response": "",
                           "feedback": f"INVALID TEMPLATE (rejected before scoring): {exc}"}
                          for item in batch] if capture_traces else None)
                log_proposal(template, batch, scores, invalid=str(exc))
                return EvaluationBatch(outputs=[{}] * len(batch), scores=scores,
                                       trajectories=trajs)
            rows = None
            for attempt in range(3):
                try:
                    rows = score_templates([template], list(batch))
                    break
                except Exception as exc:
                    # engine re-init is flaky under churn (zombie EngineCore GPU-mem);
                    # release + wait + retry, and NEVER let one failure eat a gepa iteration.
                    print(f"[run_decoder_tuning] scorer attempt {attempt} failed: {exc}",
                          flush=True)
                    try:
                        release_resident_engines()
                    except Exception:
                        pass
                    if attempt < 2:
                        time.sleep(45)
            if rows is None:
                scores = [-1.0] * len(batch)
                trajs = ([{"data": item, "full_assistant_response": "",
                           "feedback": "evaluator transient failure (engine init); "
                                       "not a property of this template"}
                          for item in batch] if capture_traces else None)
                log_proposal(template, batch, scores, invalid="scorer-transient-failure")
                return EvaluationBatch(outputs=[{}] * len(batch), scores=scores,
                                       trajectories=trajs)
            scores = [per_context_search_fitness(rows, item["metric_key"]) for item in batch]
            trajs = ([{"data": item, "full_assistant_response": "",
                       "feedback": contrast_feedback(rows, item["metric_key"])}
                      for item in batch] if capture_traces else None)
            log_proposal(template, batch, scores)
            return EvaluationBatch(outputs=[{}] * len(batch), scores=scores,
                                   trajectories=trajs)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            component = components_to_update[0]
            items = [{
                "Inputs": "dev metric (content withheld — the shared template must stay "
                          "metric-agnostic)",
                "Generated Outputs": f"(constrained {channel} decode over panel states)",
                "Feedback": f"{trajectory['feedback']} {constraint}",
            } for trajectory in (eval_batch.trajectories or [])]
            return {component: items}

    # Proposer spec "<backend>:<model>" (e.g. "zai_anthropic:glm-5.2") routes GEPA's reflective
    # mutation through the HTTP API (stronger proposer + zero GPU); a bare model name loads
    # locally. Constructed exactly as the retired propose_fn built its proposer backend.
    if ":" in str(proposer_model) and not fake_backends:
        backend_name, model_name = str(proposer_model).split(":", 1)
        cfg = dataclasses.replace(ImplementerConfig(), backend=backend_name)
        reflection_backend = LLMBackend(model_name, "generator", cfg, temperature=1.0)
    else:
        reflection_backend, _revision = _backend(proposer_model, fake=fake_backends)

    def reflection_lm(prompt):
        text = prompt if isinstance(prompt, str) else json.dumps(prompt)
        return str(reflection_backend.generate_batch(
            [text], system=None, max_tokens=1200, temperature=1.0,
        )[0])

    component = "mcq_template" if channel == "mcq" else "behavioral_template"
    print(
        f"[run_decoder_tuning] official-gepa channel={channel} arm={arm} "
        f"families={sorted(decoder_models)} budget={max_metric_calls}", flush=True,
    )
    result = gepa.optimize(
        seed_candidate={component: seed_template},
        trainset=list(contexts), valset=list(contexts),
        adapter=V14DecoderAdapter(), reflection_lm=reflection_lm,
        max_metric_calls=int(max_metric_calls), run_dir=str(run_dir / "gepa_state"),
        seed=0, display_progress_bar=False, raise_on_exception=False,
    )
    winner_template = str(next(iter(result.best_candidate.values())))

    # POST-HOC full report (both splits, gate flags) over the seed, the winner, and every
    # distinct valid template gepa proposed; the evidence cache makes the rescore cheap.
    distinct, seen = [], set()
    for template in (seed_template, winner_template):
        digest = template_sha256(template)
        if digest not in seen:
            seen.add(digest)
            distinct.append(template)
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("invalid"):
                continue
            if record["template_sha256"] not in seen:
                seen.add(record["template_sha256"])
                distinct.append(record["template"])
    # A validator-passing non-winner proposal can still crash the scorer on dev metrics GEPA
    # never touched (reflection minibatches cover ~3 of 8 contexts); after a completed search
    # that must cost us the one template, never the whole run. Seed and winner are load-bearing
    # for the report, so they get the same release+retry treatment as the search path and still
    # raise if unscorable; every other template is dropped with a warning.
    aggregate_rows = []
    for template in distinct:
        digest = template_sha256(template)
        load_bearing = template in (seed_template, winner_template)
        rows, last_exc = None, None
        for attempt in range(3 if load_bearing else 1):
            try:
                rows = score_templates([template], contexts)
                break
            except Exception as exc:
                last_exc = exc
                print(f"[run_decoder_tuning] post-hoc rescore attempt {attempt} failed for "
                      f"{digest[:12]}: {exc}", flush=True)
                try:
                    release_resident_engines()
                except Exception:
                    pass
                if load_bearing and attempt < 2:
                    time.sleep(45)
        if rows is not None:
            aggregate_rows.extend(rows)
        elif load_bearing:
            raise RuntimeError(
                f"post-hoc rescore failed for the {'seed' if template == seed_template else 'winner'} "
                f"template ({digest[:12]}); report would be invalid without it"
            ) from last_exc
    reports = aggregate_template_fitness(aggregate_rows)
    seed_sha = template_sha256(seed_template)
    winner_sha = template_sha256(winner_template)
    report_keys = (
        "pooled_fitness", "heldout_prompt_fitness", "heldout_prompt_transfer_ok",
        "far_near_transfer_ok", "dev_identification_residual_bits", "per_family_fitness",
        "n_search_cells", "n_heldout_prompt_cells",
    )

    def finite_json(value):
        # aggregate reports can hold -inf/NaN for empty-cell templates; _atomic_json uses
        # allow_nan=False, so coerce every non-finite float to null before serializing.
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, Mapping):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(item) for item in value]
        return value

    payload = {
        "schema": "v14-tune-official-gepa-v1",
        "optimizer": "official-gepa",
        "channel": str(channel), "arm": str(arm),
        "shared_across_decoder_families": True,
        "mechanical_model_specific_chat_formatting_allowed": True,
        "searched_per_family_variation_allowed": False,
        "decoder_models": {str(family): str(model) for family, model in decoder_models.items()},
        "proposer_model": str(proposer_model),
        "budget_metric_calls": int(max_metric_calls),
        "n_distinct_templates": len(distinct),
        "seed_template": seed_template,
        "seed_template_sha256": seed_sha,
        "winner_template": winner_template,
        "winner_template_sha256": winner_sha,
        "winner_report": finite_json(reports.get(winner_sha, {})),
        "seed_report": finite_json(reports.get(seed_sha, {})),
        "reports": {
            sha: finite_json({key: report.get(key) for key in report_keys})
            for sha, report in reports.items()
        },
    }
    # Identity hash over the frozen template selection only (deterministic, NaN-free); used by
    # build_production_freeze as template_trace_sha256.
    payload["freeze_sha256"] = hashlib.sha256(json.dumps({
        "schema": payload["schema"], "channel": str(channel), "arm": str(arm),
        "seed_template_sha256": seed_sha, "winner_template_sha256": winner_sha,
        "winner_template": winner_template,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    _atomic_json(Path(out_root) / "development" / "tuning" / f"{name}.json", payload)
    return payload


def _qualification_panel(design: Mapping[str, object], family: str) -> dict:
    source = min(
        design["panel_design"]["panels"], key=lambda row: str(row["panel_sha256"])
    )
    row = dict(source)
    row["original_trial"] = int(source["trial"])
    row["decoder_family"] = str(family)
    return {
        **dict(design["panel_design"]),
        "panels": [row], "n_panels": 1,
        "design_sha256": canonical_sha256({
            "purpose": "decoder-mini-qualification", "family": family,
            "parent": design["panel_design"]["design_sha256"],
            "panel": source["panel_sha256"],
        }),
    }


def prepare_sentinel_population(
    *, out_root: str | Path, sentinel_metrics_manifest_path: str | Path,
    run_sha: str, sentinel_metric_keys: Sequence[str],
) -> list[dict]:
    if len(sentinel_metric_keys) != 6:
        raise ValueError("v14 sentinel population must contain exactly six metrics")
    return build_designs(
        metrics_manifest_path=sentinel_metrics_manifest_path,
        out_root=Path(out_root) / "sentinel", run_sha=f"{run_sha}:sentinel",
        metric_keys=sentinel_metric_keys,
    )


def run_qualification_constructor(
    *, out_root: str | Path, decoder_family: str, decoder_model: str,
    fake_backends: bool, physical_gpu_ids: Sequence[int], query_batch_size: int = 2048,
) -> dict:
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    designs = load_designs(Path(out_root) / "sentinel")
    if len(designs) != 6:
        raise RuntimeError("decoder mini-qualification requires six sentinel designs")
    backend, revision = _backend(decoder_model, fake=fake_backends)
    model_key = _safe(decoder_model)
    root = Path(out_root) / "sentinel" / "qualification" / decoder_family / model_key
    rows = []
    try:
        with EvidenceCellStore(Path(out_root) / "sentinel" / "qualification_cells.sqlite") as store:
            for design in designs:
                context = _metric_context(design)
                subset = _qualification_panel(design, decoder_family)
                marker = root / _safe(design["metric_key"]) / "induction.json"
                if marker.exists():
                    row = json.loads(marker.read_text())
                else:
                    row = evaluate_behavioral_state_tables_v14(
                        backend, None, design_manifest=subset,
                        probe_texts=context["probe_texts"],
                        heldout_indices=design["probe_split"]["heldout"]["indices"],
                        heldout_target=(np.asarray(context["target"]["target"])[
                            design["probe_split"]["heldout"]["indices"]
                        ] > 0.5).astype(int),
                        noun=str(context["codebook"]["reconstruction_noun"]),
                        decoder_revision=revision,
                        executor_revision=str(design["executor"]["revision"]),
                        readout_id=str(design["executor"]["readout_id"]), store=store,
                        templates={arm: DEFAULT_BEHAVIORAL_TEMPLATE for arm in BEHAVIORAL_ARMS},
                        query_batch_size=query_batch_size, induction_only=True,
                    )
                    _atomic_json(marker, row)
                rows.append({"metric_key": design["metric_key"], **row})
    finally:
        release_resident_engines()
    payload = {
        "schema": "cr3-v14-qualification-constructor-v1", "family": decoder_family,
        "model": decoder_model, "revision": revision, "rows": rows,
    }
    _atomic_json(root / "constructor.json", payload)
    return payload


def run_qualification_executor(
    *, out_root: str | Path, decoder_family: str, decoder_model: str,
    fake_backends: bool, physical_gpu_ids: Sequence[int], query_batch_size: int = 4096,
) -> dict:
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    designs = load_designs(Path(out_root) / "sentinel")
    model_key = _safe(decoder_model)
    root = Path(out_root) / "sentinel" / "qualification" / decoder_family / model_key
    constructor = json.loads((root / "constructor.json").read_text())
    executor, revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
    rows = []
    try:
        with EvidenceCellStore(Path(out_root) / "sentinel" / "qualification_cells.sqlite") as store:
            for design in designs:
                if not fake_backends and revision != str(design["executor"]["revision"]):
                    raise RuntimeError(
                        "qualification executor revision differs from the frozen design"
                    )
                context = _metric_context(design)
                subset = _qualification_panel(design, decoder_family)
                result = evaluate_behavioral_state_tables_v14(
                    _CacheOnlyConstructor(), executor, design_manifest=subset,
                    probe_texts=context["probe_texts"],
                    heldout_indices=design["probe_split"]["heldout"]["indices"],
                    heldout_target=(np.asarray(context["target"]["target"])[
                        design["probe_split"]["heldout"]["indices"]
                    ] > 0.5).astype(int),
                    noun=str(context["codebook"]["reconstruction_noun"]),
                    decoder_revision=str(constructor["revision"]),
                    executor_revision=str(design["executor"]["revision"]),
                    readout_id=str(design["executor"]["readout_id"]), store=store,
                    templates={arm: DEFAULT_BEHAVIORAL_TEMPLATE for arm in BEHAVIORAL_ARMS},
                    query_batch_size=query_batch_size,
                )
                panel = subset["panels"][0]
                state = int("".join(map(str, panel["target_state_bits"])), 2)
                arm_row = result["arms"]["unconstrained"]
                shuffled_lift = max(
                    0.0, float(arm_row["shuffled_mi"][0, state]) - float(arm_row["blind_mi"])
                )
                rows.append({
                    "metric_key": str(design["metric_key"]),
                    "canonical_lift_bits": float(arm_row["clipped_value"][0, state]),
                    "canonical_raw_lift_bits": float(arm_row["raw_lift"][0, state]),
                    "shuffled_lift_bits": shuffled_lift,
                })
    finally:
        release_resident_engines()
    qualification = evaluate_decoder_qualification(rows)
    qualification.update({
        "family": decoder_family, "model": decoder_model,
        "revision": str(constructor["revision"]), "executor_revision": revision,
    })
    _atomic_json(root / "qualification.json", qualification)
    return qualification


def freeze_production_instrument(
    *, out_root: str | Path, qualification_specs: Sequence[str], release_commit: str,
) -> dict:
    root = Path(out_root).resolve()
    selections = []
    for spec in qualification_specs:
        # family=primary_path[,fallback_path]
        if "=" not in spec:
            raise ValueError("qualification spec must be family=primary[,fallback]")
        family, paths = spec.split("=", 1)
        candidates = paths.split(",")
        if family not in DEFAULT_DECODER_MODELS or not 1 <= len(candidates) <= 2:
            raise ValueError("qualification spec has an undeclared family or model count")
        primary = json.loads(Path(candidates[0]).read_text())
        fallback = json.loads(Path(candidates[1]).read_text()) if len(candidates) > 1 else None
        if str(primary.get("model")) != DEFAULT_DECODER_MODELS[family]:
            raise ValueError(f"{family} primary qualification uses an undeclared model")
        if fallback is not None and str(fallback.get("model")) != FALLBACK_DECODER_MODELS[family]:
            raise ValueError(f"{family} fallback qualification uses an undeclared model")
        selections.append(choose_qualified_decoder(
            family=family, primary=primary, fallback=fallback,
        ))
    tuning_root = root / "development" / "tuning"
    traces = {
        "mcq": json.loads((tuning_root / "mcq.json").read_text()),
        "behavioral_unconstrained": json.loads(
            (tuning_root / "behavioral__unconstrained.json").read_text()
        ),
        "behavioral_no_verbatim": json.loads(
            (tuning_root / "behavioral__no_verbatim_examples.json").read_text()
        ),
    }
    design_index = json.loads((root / "design_index.json").read_text())
    forbidden = []
    for design in [*load_designs(root), *load_designs(root / "development")]:
        context = _metric_context(design)
        menu = context["menu"]
        forbidden.extend([
            str(design["metric_key"]), str(menu["entry"]["target_description"]),
        ])
        forbidden.extend(
            str(item.get("description", "") if isinstance(item, Mapping) else item.description)
            for item in menu["distractors"]
        )
    freeze = build_production_freeze(
        design_index=design_index, decoder_selections=selections,
        mcq_trace=traces["mcq"],
        behavioral_unconstrained_trace=traces["behavioral_unconstrained"],
        behavioral_no_verbatim_trace=traces["behavioral_no_verbatim"],
        forbidden_strings=forbidden, release_commit=release_commit,
        out_path=root / "template_freeze.json",
    )
    first_design = load_designs(root)[0]
    prereg = build_preregistration(
        template_freeze=freeze, design_index=design_index,
        metrics_manifest_sha256=str(first_design["source_metrics_manifest_sha256"]),
        out_path=root / "preregistration.json",
    )
    return {"template_freeze": freeze, "preregistration": prereg}


def run_liveness_constructor_stage(
    *, out_root: str | Path, decoder_family: str, fake_backends: bool,
    physical_gpu_ids: Sequence[int], query_batch_size: int = 1024,
) -> dict:
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    root = Path(out_root).resolve()
    freeze = load_template_freeze(root / "template_freeze.json")
    selection = next(
        row for row in freeze["decoder_panel"] if str(row["family"]) == decoder_family
    )
    backend, revision = _backend(str(selection["model"]), fake=fake_backends)
    if not fake_backends and revision != str(selection["revision"]):
        raise RuntimeError("liveness decoder revision differs from the production freeze")
    rows = {}
    try:
        with EvidenceCellStore(root / "sentinel" / "liveness_cells.sqlite") as store:
            for instrument, templates in freeze["instruments"].items():
                rows[instrument] = run_liveness_constructor_controls(
                    backend, decoder_family=decoder_family, decoder_revision=revision,
                    templates=templates, store=store, query_batch_size=query_batch_size,
                )
    finally:
        release_resident_engines()
    payload = {
        "schema": "cr3-v14-liveness-constructor-stage-v1",
        "decoder_family": decoder_family, "decoder_model": selection["model"],
        "decoder_revision": revision, "instruments": rows,
    }
    _atomic_json(root / "sentinel" / f"liveness_constructor_{decoder_family}.json", payload)
    return payload


def run_liveness_executor_stage(
    *, out_root: str | Path, fake_backends: bool,
    physical_gpu_ids: Sequence[int], query_batch_size: int = 2048,
) -> dict:
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    root = Path(out_root).resolve()
    sentinel_results = root / "sentinel" / "results.parquet"
    if not sentinel_results.is_file():
        raise RuntimeError("six-metric frozen-instrument sentinel evaluation is incomplete")
    sentinel_frame = pd.read_parquet(sentinel_results)
    if len(sentinel_frame) != 6 * 2 * 3 or sentinel_frame["metric_key"].nunique() != 6:
        raise RuntimeError("six-metric sentinel result artifact is structurally incomplete")
    executor, revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
    rows = []
    try:
        with EvidenceCellStore(root / "sentinel" / "liveness_cells.sqlite") as store:
            for family in DEFAULT_DECODER_MODELS:
                constructor = json.loads(
                    (root / "sentinel" / f"liveness_constructor_{family}.json").read_text()
                )
                for instrument, result in constructor["instruments"].items():
                    observed = finish_liveness_executor_controls(
                        executor, constructor_result=result, executor_revision=revision,
                        readout_id=CR3_BINARY_READOUT_ID, store=store,
                        query_batch_size=query_batch_size,
                    )
                    rows.extend({**row, "instrument": instrument} for row in observed)
    finally:
        release_resident_engines()
    payload = {
        "schema": "cr3-v14-liveness-control-rows-v1",
        "executor_revision": revision, "rows": rows,
    }
    control_path = root / "sentinel" / "liveness_control_rows.json"
    _atomic_json(control_path, payload)
    gate = apply_sentinel_gate(control_rows_path=control_path, out_root=root)
    return {"controls": payload, "gate": gate}


def load_template_freeze(path: str | Path) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != TEMPLATE_FREEZE_SCHEMA:
        raise ValueError("unsupported v14 template freeze")
    core = dict(payload)
    observed = str(core.pop("freeze_sha256", ""))
    if observed != canonical_sha256(core):
        raise ValueError("template freeze checksum mismatch")
    return payload


def _require_live_sentinel(path: str | Path | None, *, fake_backends: bool) -> None:
    if fake_backends:
        return
    if path is None:
        raise RuntimeError("production fan-out requires --sentinel-report")
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "cr3-v14-sentinel-liveness-v1" or not bool(
        payload.get("passed", False)
    ):
        raise RuntimeError("sentinel liveness did not pass; production fan-out is blocked")


def apply_sentinel_gate(*, control_rows_path: str | Path, out_root: str | Path) -> dict:
    payload = json.loads(Path(control_rows_path).read_text())
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("sentinel control artifact must contain a rows list")
    result = evaluate_sentinel_liveness(rows)
    _atomic_json(Path(out_root) / "sentinel_report.json", result)
    if not result["passed"]:
        raise RuntimeError("sentinel detected structural failure or control-defined instrument death")
    return result


def write_seed_template_freeze(path: str | Path) -> dict:
    """Write the untuned instrument only; tuned entries must later come from GEPA."""
    core = {
        "schema": TEMPLATE_FREEZE_SCHEMA,
        "instruments": {
            "untuned": {
                "mcq": DEFAULT_MCQ_TEMPLATE,
                "behavioral": {arm: DEFAULT_BEHAVIORAL_TEMPLATE for arm in BEHAVIORAL_ARMS},
            },
        },
        "shared_across_metrics_and_decoder_families": True,
        "searched_per_family_variation": False,
        "study_alpha": 0.05,
        "process_bound_cell_alpha": 0.05,
        "alpha_scope": "seed-only smoke freeze; production preregistration must replace this",
    }
    core["freeze_sha256"] = canonical_sha256(core)
    _atomic_json(Path(path), core)
    return core


def _subset_panel_design(design: Mapping[str, object], family: str) -> dict:
    panel = dict(design["panel_design"])
    selected = [
        dict(row) for row in panel["panels"] if str(row["decoder_family"]) == str(family)
    ]
    if not selected:
        raise ValueError(f"design has no panels assigned to decoder family {family}")
    for row in selected:
        row["original_trial"] = int(row["trial"])
    panel["panels"] = selected
    panel["n_panels"] = len(selected)
    panel["design_sha256"] = canonical_sha256({
        "parent": design["panel_design"]["design_sha256"],
        "family": family,
        "trials": [row["original_trial"] for row in selected],
    })
    return panel


def _lane_state_indices(
    design: Mapping[str, object], context: Mapping[str, object],
    panel_design: Mapping[str, object],
) -> list[list[int]] | None:
    lane = str(design.get("scoring_lane", {}).get("lane", "cert"))
    if lane == "cert":
        return None
    if lane != "fast":
        raise ValueError(f"unsupported scoring lane {lane!r}")
    signatures = np.asarray(context["population"]["signatures"], dtype=float)
    panels = [list(map(int, row["indices"])) for row in panel_design["panels"]]
    codes = signatures_to_states(signatures, panels)
    output = []
    for position, panel in enumerate(panel_design["panels"]):
        canonical = int("".join(map(str, panel["target_state_bits"])), 2)
        output.append(sorted({canonical, *map(int, codes[:, position])}))
    return output


class _CacheOnlyConstructor:
    def generate_batch(self, *_args, **_kwargs):
        raise RuntimeError("behavioral induction cache is incomplete")


def _metric_context(design: Mapping[str, object]) -> dict:
    entry = design["entry"]
    codebook = _load_codebook_for_entry(entry, Path("/"))
    metric_key = str(design["metric_key"])
    target = dict(_bootstrap(codebook["metrics"][metric_key]["bootstrap_path"]))
    base_probe_texts = list(map(str, target["probe_texts"]))
    probe_texts = base_probe_texts
    extension_row = design.get("probe_extension")
    if extension_row:
        extension = load_extension(extension_row["path"])
        keys = list(map(str, extension["metric_keys"]))
        if metric_key not in keys:
            raise RuntimeError(f"probe extension lacks target {metric_key}")
        probe_texts = [*base_probe_texts, *map(str, extension["texts"])]
        target["target"] = np.concatenate([
            np.asarray(target["target"], dtype=float),
            np.asarray(extension["scores"], dtype=float)[keys.index(metric_key)],
        ])
        target["probe_texts"] = np.asarray(probe_texts, dtype=object)
    population = load_candidate_population(
        entry, Path("/"), n_probes=len(base_probe_texts), probe_sha256=str(target["probe_sha256"]),
    )
    menu = _codebook_menu(codebook, metric_key)
    if extension_row:
        extension = load_extension(extension_row["path"])
        extension_keys = list(map(str, extension["metric_keys"]))
        for distractor in menu["distractors"]:
            key = str(distractor["metric_id"])
            if key not in extension_keys:
                raise RuntimeError(f"probe extension lacks distractor {key}")
            distractor["scores"] = np.concatenate([
                np.asarray(distractor["scores"], dtype=float),
                np.asarray(extension["scores"], dtype=float)[extension_keys.index(key)],
            ])
    return {
        "codebook": codebook, "target": target, "probe_texts": probe_texts,
        "population": population, "menu": menu,
    }


def run_constructor_stage(
    *, out_root: str | Path, template_freeze_path: str | Path,
    decoder_family: str, decoder_model: str, metric_keys: Sequence[str] | None,
    fake_backends: bool, physical_gpu_ids: Sequence[int], channels: Sequence[str],
    query_batch_size: int, sentinel_report_path: str | Path | None = None,
    require_sentinel: bool = True,
) -> list[dict]:
    designs = load_designs(out_root, metric_keys)
    lane = str(designs[0].get("scoring_lane", {}).get("lane", "cert")) if designs else "cert"
    if require_sentinel and lane == "cert":
        _require_live_sentinel(sentinel_report_path, fake_backends=fake_backends)
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    freeze = load_template_freeze(template_freeze_path)
    selections = [
        row for row in freeze.get("decoder_panel", [])
        if str(row["family"]) == str(decoder_family)
    ]
    if selections:
        if len(selections) != 1:
            raise RuntimeError(f"freeze has an invalid {decoder_family} decoder selection")
        frozen_model = str(selections[0]["model"])
        if str(decoder_model) != frozen_model:
            raise RuntimeError(
                f"constructor model {decoder_model!r} differs from frozen "
                f"{decoder_family} model {frozen_model!r}"
            )
    backend, revision = _backend(decoder_model, fake=fake_backends)
    if selections and not fake_backends and revision != str(selections[0]["revision"]):
        raise RuntimeError("constructor revision differs from the production freeze")
    root = Path(out_root).resolve()
    store_path = root / "evidence_cells.sqlite"
    reports = []
    try:
        with EvidenceCellStore(store_path) as store:
            for design in designs:
                metric_key = str(design["metric_key"])
                context = _metric_context(design)
                subset = _subset_panel_design(design, decoder_family)
                lane_states = _lane_state_indices(design, context, subset)
                metric_root = root / "state_chunks" / _safe(metric_key) / decoder_family
                for instrument, templates in freeze["instruments"].items():
                    instrument_root = metric_root / str(instrument)
                    if "mcq" in channels:
                        path = instrument_root / "mcq_state_tables.npz"
                        if not path.exists():
                            menu = context["menu"]
                            result = evaluate_mcq_state_tables_v14(
                                backend, design_manifest=subset,
                                noun=str(context["codebook"]["reconstruction_noun"]),
                                target_metric_id=metric_key,
                                target_description=str(menu["entry"]["target_description"]),
                                distractors=menu["distractors"],
                                probe_texts=context["probe_texts"],
                                constructor_revision=revision, store=store,
                                template=str(templates["mcq"]),
                                max_chars=int(context["codebook"]["reconstruction_max_chars"]),
                                n_reconstruction_draws=int(
                                    design.get("throughput_design", {}).get(
                                        "menu_permutations", 8
                                    )
                                ),
                                query_batch_size=int(query_batch_size),
                                state_indices_by_panel=lane_states,
                            )
                            _atomic_npz(
                                path, raw_lift=result["raw_lift"],
                                clipped_value=result["clipped_value"],
                                normalized_lift=result["normalized_lift"],
                                raw_target_probability=result["raw_target_probability"],
                                shuffled_target_probability=result["shuffled_target_probability"],
                                annotation_accuracy=result["annotation_accuracy"],
                                observed_state_mask=result["observed_state_mask"],
                            )
                            _atomic_json(instrument_root / "mcq_metadata.json", {
                                key: value for key, value in result.items()
                                if not isinstance(value, np.ndarray)
                            })
                    if "behavioral" in channels:
                        marker = instrument_root / "behavioral_induction.json"
                        if not marker.exists():
                            result = evaluate_behavioral_state_tables_v14(
                                backend, None, design_manifest=subset,
                                probe_texts=context["probe_texts"],
                                heldout_indices=design["probe_split"]["heldout"]["indices"],
                                heldout_target=(np.asarray(context["target"]["target"])[
                                    design["probe_split"]["heldout"]["indices"]
                                ] > 0.5).astype(int),
                                noun=str(context["codebook"]["reconstruction_noun"]),
                                decoder_revision=revision,
                                executor_revision=str(design["executor"]["revision"]),
                                readout_id=str(design["executor"]["readout_id"]),
                                store=store, templates=templates["behavioral"],
                                max_chars=int(context["codebook"]["reconstruction_max_chars"]),
                                query_batch_size=int(query_batch_size), induction_only=True,
                                state_indices_by_panel=lane_states,
                            )
                            _atomic_json(marker, result)
                reports.append({
                    "metric_key": metric_key, "decoder_family": decoder_family,
                    "decoder_model": decoder_model, "decoder_revision": revision,
                    "n_panels": len(subset["panels"]),
                })
    finally:
        release_resident_engines()
    _atomic_json(root / "constructor_stage" / f"{decoder_family}.json", {
        "schema": "cr3-v14-constructor-stage-v1", "rows": reports,
    })
    return reports


def run_executor_stage(
    *, out_root: str | Path, template_freeze_path: str | Path,
    metric_keys: Sequence[str] | None, fake_backends: bool,
    physical_gpu_ids: Sequence[int], query_batch_size: int,
    sentinel_report_path: str | Path | None = None,
    require_sentinel: bool = True,
) -> list[dict]:
    designs = load_designs(out_root, metric_keys)
    lane = str(designs[0].get("scoring_lane", {}).get("lane", "cert")) if designs else "cert"
    if require_sentinel and lane == "cert":
        _require_live_sentinel(sentinel_report_path, fake_backends=fake_backends)
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    freeze = load_template_freeze(template_freeze_path)
    backend, revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
    root = Path(out_root).resolve()
    reports = []
    try:
        with EvidenceCellStore(root / "evidence_cells.sqlite") as store:
            for design in designs:
                context = _metric_context(design)
                metric_key = str(design["metric_key"])
                if not fake_backends and revision != str(design["executor"]["revision"]):
                    raise RuntimeError("fixed executor revision differs from the frozen design")
                probe_embeddings = None
                if str(design.get("scoring_lane", {}).get("lane", "cert")) == "cert":
                    probe_embeddings = _certification_probe_embeddings(
                        context["probe_texts"],
                        cache_path=(
                            root / "certification_embeddings" / f"{_safe(metric_key)}.npz"
                        ),
                        fake_backends=fake_backends,
                    )
                for family in DEFAULT_DECODER_MODELS:
                    constructor_report_path = root / "constructor_stage" / f"{family}.json"
                    if not constructor_report_path.exists():
                        raise RuntimeError(f"constructor stage is incomplete for {family}")
                    subset = _subset_panel_design(design, family)
                    lane_states = _lane_state_indices(design, context, subset)
                    family_root = root / "state_chunks" / _safe(metric_key) / family
                    for instrument, templates in freeze["instruments"].items():
                        instrument_root = family_root / str(instrument)
                        induction = json.loads(
                            (instrument_root / "behavioral_induction.json").read_text()
                        )
                        path = instrument_root / "behavioral_state_tables.npz"
                        if path.exists():
                            continue
                        result = evaluate_behavioral_state_tables_v14(
                            _CacheOnlyConstructor(), backend, design_manifest=subset,
                            probe_texts=context["probe_texts"],
                            heldout_indices=design["probe_split"]["heldout"]["indices"],
                            heldout_target=(np.asarray(context["target"]["target"])[
                                design["probe_split"]["heldout"]["indices"]
                            ] > 0.5).astype(int),
                            noun=str(context["codebook"]["reconstruction_noun"]),
                            decoder_revision=str(induction["decoder_revision"]),
                            executor_revision=str(design["executor"]["revision"]),
                            readout_id=str(design["executor"]["readout_id"]),
                            store=store, templates=templates["behavioral"],
                            max_chars=int(context["codebook"]["reconstruction_max_chars"]),
                            query_batch_size=int(query_batch_size),
                            probe_embeddings=probe_embeddings,
                            state_indices_by_panel=lane_states,
                        )
                        arrays = {}
                        metadata = {
                            key: value for key, value in result.items()
                            if key != "arms"
                        }
                        metadata["arms"] = {}
                        for arm, row in result["arms"].items():
                            for field, value in row.items():
                                if isinstance(value, np.ndarray):
                                    arrays[f"{arm}__{field}"] = value
                                else:
                                    metadata["arms"].setdefault(arm, {})[field] = value
                        _atomic_npz(path, **arrays)
                        _atomic_json(instrument_root / "behavioral_metadata.json", metadata)
                        reports.append({
                            "metric_key": metric_key, "decoder_family": family,
                            "instrument": instrument,
                            "n_panels": len(subset["panels"]),
                        })
    finally:
        release_resident_engines()
    _atomic_json(root / "executor_stage.json", {
        "schema": "cr3-v14-executor-stage-v1", "executor_revision": revision,
        "rows": reports,
    })
    return reports


def run_audit_proposer_stage(
    *, out_root: str | Path, audit_family: str, metric_keys: Sequence[str] | None,
    fake_backends: bool, physical_gpu_ids: Sequence[int], total_budget: int = 400,
    sentinel_report_path: str | Path | None = None,
) -> list[dict]:
    _require_live_sentinel(sentinel_report_path, fake_backends=fake_backends)
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    if audit_family not in AUDIT_FAMILIES:
        raise ValueError(f"unknown frozen audit family {audit_family!r}")
    designs = load_designs(out_root, metric_keys)
    model = AUDIT_FAMILIES[audit_family]
    backend, revision = _backend(model, fake=fake_backends)
    reports = []
    try:
        for design in designs:
            context = _metric_context(design)
            entry = context["codebook"]["entries"][str(design["metric_key"])]
            reports.append(propose_family_audit(
                backend, out_root=out_root, metric_key=str(design["metric_key"]),
                metric_name=str(entry.get("target_name") or design["metric_key"]),
                metric_description=str(entry["target_description"]),
                family=audit_family, model=model, model_revision=revision,
                total_budget=int(total_budget),
            ))
    finally:
        release_resident_engines()
    _atomic_json(Path(out_root) / "audit" / f"proposer_stage_{audit_family}.json", {
        "schema": "cr3-v14-audit-proposer-stage-v1", "family": audit_family,
        "model": model, "model_revision": revision, "rows": reports,
    })
    return reports


def run_audit_score_stage(
    *, out_root: str | Path, metric_keys: Sequence[str] | None,
    fake_backends: bool, physical_gpu_ids: Sequence[int], total_budget: int = 400,
    query_batch_size: int = 4096, sentinel_report_path: str | Path | None = None,
) -> list[dict]:
    _require_live_sentinel(sentinel_report_path, fake_backends=fake_backends)
    assert_gpu_authorized(physical_gpu_ids, fake_backends=fake_backends)
    designs = load_designs(out_root, metric_keys)
    executor, revision = _backend(FIXED_EXECUTOR, fake=fake_backends)
    reports = []
    try:
        with EvidenceCellStore(Path(out_root) / "evidence_cells.sqlite") as store:
            for design in designs:
                context = _metric_context(design)
                if not fake_backends and revision != str(design["executor"]["revision"]):
                    raise RuntimeError("audit executor revision differs from the frozen design")
                reports.append(score_audit_ledger(
                    executor, out_root=out_root, metric_key=str(design["metric_key"]),
                    probe_texts=context["probe_texts"], executor_revision=revision,
                    readout_id=str(design["executor"]["readout_id"]), store=store,
                    total_budget=int(total_budget),
                    max_chars=int(context["codebook"]["reconstruction_max_chars"]),
                    query_batch_size=int(query_batch_size),
                ))
    finally:
        release_resident_engines()
    _atomic_json(Path(out_root) / "audit" / "score_stage.json", {
        "schema": "cr3-v14-audit-score-stage-v1", "executor_revision": revision,
        "rows": reports,
    })
    return reports


def _combine_chunks(
    root: Path, design: Mapping[str, object], instrument: str, channel: str,
    arm: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_panels = len(design["panel_design"]["panels"])
    panel_size = int(design["panel_design"]["panel_size"])
    n_states = 1 << panel_size
    raw = np.full((n_panels, n_states), np.nan, dtype=float)
    clipped = np.full_like(raw, np.nan)
    filled = np.zeros(n_panels, dtype=bool)
    for family in DEFAULT_DECODER_MODELS:
        subset = _subset_panel_design(design, family)
        path = (
            root / "state_chunks" / _safe(design["metric_key"]) / family / instrument /
            ("mcq_state_tables.npz" if channel == "mcq" else "behavioral_state_tables.npz")
        )
        with np.load(path, allow_pickle=False) as artifact:
            if channel == "mcq":
                part_raw = np.asarray(artifact["raw_lift"], dtype=float)
                part_clipped = np.asarray(artifact["clipped_value"], dtype=float)
            else:
                part_raw = np.asarray(artifact[f"{arm}__raw_lift"], dtype=float)
                part_clipped = np.asarray(artifact[f"{arm}__clipped_value"], dtype=float)
        for position, panel in enumerate(subset["panels"]):
            trial = int(panel["original_trial"])
            raw[trial] = part_raw[position]
            clipped[trial] = part_clipped[position]
            filled[trial] = True
    if not np.all(filled):
        raise RuntimeError("combined v14 table lacks one or more frozen trials")
    lane = str(design.get("scoring_lane", {}).get("lane", "cert"))
    if lane == "cert":
        validate_state_tables(raw, clipped, panel_size=panel_size)
    elif not np.any(np.isfinite(raw)):
        raise RuntimeError("FAST state table contains no observed cells")
    return raw, clipped


def _combine_behavioral_predictions(
    root: Path, design: Mapping[str, object], instrument: str, arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_panels = len(design["panel_design"]["panels"])
    n_states = 1 << int(design["panel_design"]["panel_size"])
    heldout_n = len(design["probe_split"]["heldout"]["indices"])
    predictions = np.full((n_panels, n_states, heldout_n), -1, dtype=np.int8)
    blind = np.full((n_panels, heldout_n), -1, dtype=np.int8)
    filled = np.zeros(n_panels, dtype=bool)
    for family in DEFAULT_DECODER_MODELS:
        subset = _subset_panel_design(design, family)
        path = (
            root / "state_chunks" / _safe(design["metric_key"]) / family / instrument /
            "behavioral_state_tables.npz"
        )
        with np.load(path, allow_pickle=False) as artifact:
            part = np.asarray(artifact[f"{arm}__hard_predictions"], dtype=np.int8)
            part_blind = np.asarray(artifact[f"{arm}__blind_hard_prediction"], dtype=np.int8)
        for position, panel in enumerate(subset["panels"]):
            trial = int(panel["original_trial"])
            predictions[trial] = part[position]
            blind[trial] = part_blind
            filled[trial] = True
    if not np.all(filled) or np.any(blind < 0):
        raise RuntimeError("FAST behavioral predictions lack a frozen panel")
    return predictions, blind


def _combine_behavioral_transfer_chunks(
    root: Path, design: Mapping[str, object], instrument: str, arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fields = ("near_raw_lift", "near_clipped_value", "far_raw_lift", "far_clipped_value")
    n_panels = len(design["panel_design"]["panels"])
    panel_size = int(design["panel_design"]["panel_size"])
    n_states = 1 << panel_size
    combined = {field: np.empty((n_panels, n_states), dtype=float) for field in fields}
    filled = np.zeros(n_panels, dtype=bool)
    for family in DEFAULT_DECODER_MODELS:
        subset = _subset_panel_design(design, family)
        path = (
            root / "state_chunks" / _safe(design["metric_key"]) / family / instrument /
            "behavioral_state_tables.npz"
        )
        with np.load(path, allow_pickle=False) as artifact:
            parts = {
                field: np.asarray(artifact[f"{arm}__{field}"], dtype=float)
                for field in fields
            }
        for position, panel in enumerate(subset["panels"]):
            trial = int(panel["original_trial"])
            for field in fields:
                combined[field][trial] = parts[field][position]
            filled[trial] = True
    if not np.all(filled):
        raise RuntimeError("combined behavioral transfer table lacks frozen trials")
    validate_state_tables(
        combined["near_raw_lift"], combined["near_clipped_value"], panel_size=panel_size,
    )
    validate_state_tables(
        combined["far_raw_lift"], combined["far_clipped_value"], panel_size=panel_size,
    )
    return tuple(combined[field] for field in fields)


def _complete_historical_population(context: Mapping[str, object]) -> dict:
    """Restore provenance-valid frozen suffixes as pre-audit achieved evidence."""
    population = context["population"]
    signatures = [np.asarray(population["signatures"], dtype=float)]
    prompts = list(map(str, population["texts"]))
    families = list(map(str, population["families"]))
    streams = population.get("process_streams")
    if streams is not None:
        for stream_position, stream in enumerate(streams):
            provenance = stream["provenance"]
            source_path = Path(provenance["source_path"])
            indices = list(map(
                int, provenance["source_row_indices_never_absorbed_audit_suffix"],
            ))
            with np.load(source_path, allow_pickle=True) as artifact:
                field = "texts" if "texts" in artifact.files else "prompts"
                source_texts = [str(artifact[field][index]) for index in indices]
            suffix = np.asarray(stream["audit_suffix_signatures"], dtype=float)
            if suffix.shape != (len(source_texts), signatures[0].shape[1]):
                raise RuntimeError("historical audit suffix text/signature rows are misaligned")
            signatures.append(suffix)
            prompts.extend(source_texts)
            families.extend([str(stream["family"])] * len(source_texts))
    matrix = np.vstack(signatures)
    ids = [
        f"historical:{index}:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        for index, text in enumerate(prompts)
    ]
    return {
        "signatures": matrix, "prompts": prompts, "prompt_ids": ids,
        "families": families, "record_rank_provenance_valid": streams is not None,
        "n_harvested_draws": len(prompts),
    }


def _control_diagnostics(
    root: Path, design: Mapping[str, object], instrument: str, channel: str,
    arm: str | None,
) -> dict:
    blind = []
    best = []
    explanation = []
    for family in DEFAULT_DECODER_MODELS:
        subset = _subset_panel_design(design, family)
        source = (
            root / "state_chunks" / _safe(design["metric_key"]) / family / instrument
        )
        if channel == "mcq":
            metadata = json.loads((source / "mcq_metadata.json").read_text())
            with np.load(source / "mcq_state_tables.npz", allow_pickle=False) as artifact:
                signal = np.asarray(artifact["raw_target_probability"], dtype=float)
            blind.extend([float(metadata["blind"]["target_probability"])] * len(subset["panels"]))
            best.extend(np.nanmax(signal, axis=1).astype(float).tolist())
            if metadata.get("best_explanation_rate") is not None:
                explanation.append(float(metadata["best_explanation_rate"]))
        else:
            metadata = json.loads((source / "behavioral_metadata.json").read_text())
            with np.load(source / "behavioral_state_tables.npz", allow_pickle=False) as artifact:
                signal = np.asarray(artifact[f"{arm}__raw_mi"], dtype=float)
            blind.extend([
                float(metadata["arms"][str(arm)]["blind_mi"])
            ] * len(subset["panels"]))
            best.extend(np.max(signal, axis=1).astype(float).tolist())
    blind_mean = float(np.mean(blind))
    best_mean = float(np.mean(best))
    return {
        "blind_signal": blind_mean,
        "best_annotated_signal": best_mean,
        "best_minus_blind_signal": best_mean - blind_mean,
        "best_explanation_rate": (
            float(np.mean(explanation)) if explanation else None
        ),
    }


def aggregate_fast_campaign(
    *, out_root: str | Path, template_freeze_path: str | Path,
    metric_keys: Sequence[str] | None, channels: Sequence[str],
    n_permutations: int = 200,
) -> pd.DataFrame:
    """Aggregate quarantined observed-state screening rows; emit no certificates."""
    if int(n_permutations) < 200:
        raise ValueError("FAST aggregation requires at least 200 permutation draws")
    root = Path(out_root).resolve()
    designs = load_designs(root, metric_keys)
    if any(str(row.get("scoring_lane", {}).get("lane", "cert")) != "fast" for row in designs):
        raise ValueError("FAST aggregation accepts FAST designs only")
    freeze = load_template_freeze(template_freeze_path)
    rows = []
    null_arrays = {}
    arm_exclusions = []
    for design in designs:
        context = _metric_context(design)
        metric_key = str(design["metric_key"])
        signatures = np.asarray(context["population"]["signatures"], dtype=float)
        prompts = list(map(str, context["population"]["texts"]))
        prompt_ids = [
            f"fast:{index}:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            for index, text in enumerate(prompts)
        ]
        panels = [row["indices"] for row in design["panel_design"]["panels"]]
        codes = signatures_to_states(signatures, panels)
        heldout = np.asarray(design["probe_split"]["heldout"]["indices"], dtype=int)
        target = (np.asarray(context["target"]["target"])[heldout] > 0.5).astype(np.uint8)
        for instrument in freeze["instruments"]:
            channel_arms = []
            if "mcq" in channels:
                channel_arms.append(("mcq", None))
            if "behavioral" in channels:
                channel_arms.extend(("behavioral", arm) for arm in BEHAVIORAL_ARMS)
            for channel, arm in channel_arms:
                try:
                    raw, clipped = _combine_chunks(root, design, str(instrument), channel, arm)
                except ValueError as exc:
                    # A constrained arm (e.g. no_verbatim_examples) can have VOID
                    # induction cells for states the constraint cannot satisfy;
                    # the achieved statistic and the null both require complete
                    # tables, so this (channel, arm) is undefined for this metric.
                    # FAST is screening-only: record the arm-level exclusion and
                    # keep aggregating the defined arms instead of aborting.
                    arm_exclusions.append({
                        "metric_key": metric_key, "instrument": str(instrument),
                        "channel": str(channel), "arm": arm,
                        "stage": "combine_chunks", "detail": str(exc),
                    })
                    continue
                seed = int.from_bytes(hashlib.sha256(
                    f"{design['run_sha']}\x1f{metric_key}\x1f{instrument}\x1f{channel}\x1f{arm}".encode()
                ).digest()[:8], "big")
                if channel == "mcq":
                    null = fast_mcq_code_permutation_null(
                        clipped, codes, n_permutations=n_permutations, seed=seed,
                    )
                else:
                    predictions, blind = _combine_behavioral_predictions(
                        root, design, str(instrument), str(arm),
                    )
                    n_states = predictions.shape[1]
                    shuffled_ids = np.empty((len(panels), n_states), dtype=int)
                    for panel_position, panel in enumerate(design["panel_design"]["panels"]):
                        shuffled_ids[panel_position] = [
                            shuffled_state(
                                state, int(design["panel_design"]["panel_size"]),
                                str(panel["panel_sha256"]),
                            )
                            for state in range(n_states)
                        ]
                    try:
                        null = fast_behavioral_label_permutation_null(
                            target, predictions, blind, codes, shuffled_ids,
                            n_permutations=n_permutations, seed=seed,
                        )
                    except ValueError as exc:
                        # VOID induction cells leave holes the null cannot fill
                        # (see combine_chunks handler above) — same record-and-skip.
                        arm_exclusions.append({
                            "metric_key": metric_key, "instrument": str(instrument),
                            "channel": str(channel), "arm": arm,
                            "stage": "permutation_null", "detail": str(exc),
                        })
                        continue
                try:
                    result = aggregate_fast_screening(
                        raw_lift=raw, clipped_value=clipped,
                        prompt_signatures=signatures, panels=panels, prompt_ids=prompt_ids,
                        target_entropy_cap=float(design["target_entropy_on_h_bits"]),
                        permutation_null=null, channel=channel,
                    )
                except ValueError as exc:
                    arm_exclusions.append({
                        "metric_key": metric_key, "instrument": str(instrument),
                        "channel": str(channel), "arm": arm,
                        "stage": "aggregate_fast_screening", "detail": str(exc),
                    })
                    continue
                null_key = _safe(f"{metric_key}__{instrument}__{channel}__{arm or 'none'}")
                null_arrays[null_key] = null
                for horizon in (100, 300):
                    bound = record_rank_gain_bound(
                        n=len(prompt_ids), m=horizon, achieved=result["achieved_value"],
                        cap=result["level0_cap"],
                    )
                    result[f"record_rank_gain_upper_h{horizon}"] = bound["gain_upper"]
                rows.append({
                    "lane": "fast", "release_eligible": False,
                    "task": design["task"], "level": design["level"],
                    "metric": design["metric"], "metric_key": metric_key,
                    "instrument": instrument, "channel": channel, "arm": arm,
                    "executor": design["executor"]["model"],
                    "achieved_value": result["achieved_value"],
                    "permutation_z_score": result["permutation_z_score"],
                    "permutation_percentile": result["permutation_percentile"],
                    "permutation_p_greater_equal": result["permutation_p_greater_equal"],
                    "permutation_count": result["permutation_count"],
                    "permutation_null_kind": result["permutation_null_kind"],
                    "exact_structural_cap": None,
                    "exact_structural_cap_status": "UNAVAILABLE_FAST_OBSERVED_ONLY",
                    "level0_cap": result["level0_cap"],
                    "record_rank_gain_upper_h100": result["record_rank_gain_upper_h100"],
                    "record_rank_gain_upper_h300": result["record_rank_gain_upper_h300"],
                    "n_observed_joint_codes": result["n_observed_joint_codes"],
                    "screening_only_not_for_claims": True,
                })
    frame = pd.DataFrame(rows)
    _atomic_parquet(root / "results.parquet", frame)
    summary = (
        frame.sort_values(
            ["metric_key", "permutation_z_score"], ascending=[True, False], kind="stable"
        ).drop_duplicates("metric_key")[
            ["lane", "task", "metric_key", "permutation_z_score"]
        ].reset_index(drop=True)
    )
    _atomic_parquet(root / "screening_summary.parquet", summary)
    _atomic_npz(root / "fast_permutation_nulls.npz", **null_arrays)
    _atomic_json(root / "fast_arm_exclusions.json", {
        "schema": "cr3-v14-fast-arm-exclusions-v1",
        "n_excluded": len(arm_exclusions),
        "note": (
            "channel/arm combinations undefined for a metric because VOID "
            "induction cells (unsatisfiable constrained-arm states) leave holes "
            "the achieved statistic and permutation null both require"
        ),
        "exclusions": arm_exclusions,
    })
    _atomic_json(root / "campaign_manifest.json", {
        "schema": CAMPAIGN_SCHEMA, "lane": "fast", "release_eligible": False,
        "n_metrics": len(designs), "n_results": len(frame),
        "permutation_count": int(n_permutations),
        "reference": "frozen_executor_verdicts",
        "independent_reference_used": False,
        "exact_state_enumeration_used": False,
        "certificates_emitted": False,
        "results_path": str(root / "results.parquet"),
        "promotion_source_path": str(root / "screening_summary.parquet"),
    })
    return frame


def aggregate_campaign(
    *, out_root: str | Path, template_freeze_path: str | Path,
    metric_keys: Sequence[str] | None, channels: Sequence[str],
) -> pd.DataFrame:
    root = Path(out_root).resolve()
    designs = load_designs(root, metric_keys)
    if designs and str(designs[0].get("scoring_lane", {}).get("lane", "cert")) == "fast":
        return aggregate_fast_campaign(
            out_root=root, template_freeze_path=template_freeze_path,
            metric_keys=metric_keys, channels=channels,
        )
    freeze = load_template_freeze(template_freeze_path)
    result_rows = []
    value_tables = {}
    for design in designs:
        context = _metric_context(design)
        metric_key = str(design["metric_key"])
        historical = _complete_historical_population(context)
        prompts = historical["prompts"]
        prompt_ids = historical["prompt_ids"]
        signatures = historical["signatures"]
        panels = [row["indices"] for row in design["panel_design"]["panels"]]
        families = [row["decoder_family"] for row in design["panel_design"]["panels"]]
        heldout = design["probe_split"]["heldout"]["indices"]
        target_h = (np.asarray(context["target"]["target"])[heldout] > 0.5).astype(int)
        for instrument in freeze["instruments"]:
            channel_arms = []
            if "mcq" in channels:
                channel_arms.append(("mcq", None))
            if "behavioral" in channels:
                channel_arms.extend(("behavioral", arm) for arm in BEHAVIORAL_ARMS)
            for channel, arm in channel_arms:
                raw, clipped = _combine_chunks(root, design, instrument, channel, arm)
                control = _control_diagnostics(
                    root, design, str(instrument), channel, arm,
                )
                discovery = aggregate_state_tables(
                    raw_lift=raw, clipped_value=clipped,
                    prompt_signatures=signatures, panels=panels,
                    prompt_ids=prompt_ids, decoder_families=families,
                )
                audit_path = root / "audit" / "signatures" / f"{metric_key}.npz"
                audit = None
                process_bounds = None
                novelty_rows = None
                all_signatures = signatures
                all_ids = prompt_ids
                all_prompts = prompts
                if audit_path.is_file():
                    with np.load(audit_path, allow_pickle=True) as artifact:
                        audit_signatures = np.asarray(artifact["sigs"], dtype=float)
                        audit_prompts = list(map(str, artifact["prompts"]))
                        audit_families = list(map(str, artifact["families"]))
                    audit_ids = [
                        f"audit:{index}:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
                        for index, text in enumerate(audit_prompts)
                    ]
                    audit = aggregate_state_tables(
                        raw_lift=raw, clipped_value=clipped,
                        prompt_signatures=audit_signatures, panels=panels,
                        prompt_ids=audit_ids, decoder_families=families,
                    )
                    all_signatures = np.vstack((signatures, audit_signatures))
                    all_ids = [*prompt_ids, *audit_ids]
                    all_prompts = [*prompts, *audit_prompts]
                aggregate = aggregate_state_tables(
                    raw_lift=raw, clipped_value=clipped,
                    prompt_signatures=all_signatures, panels=panels,
                    prompt_ids=all_ids, decoder_families=families,
                )
                transfer = None
                transfer_tables = None
                if channel == "behavioral":
                    transfer_tables = _combine_behavioral_transfer_chunks(
                        root, design, str(instrument), str(arm),
                    )
                    near_aggregate = aggregate_state_tables(
                        raw_lift=transfer_tables[0], clipped_value=transfer_tables[1],
                        prompt_signatures=all_signatures, panels=panels,
                        prompt_ids=all_ids, decoder_families=families,
                    )
                    far_aggregate = aggregate_state_tables(
                        raw_lift=transfer_tables[2], clipped_value=transfer_tables[3],
                        prompt_signatures=all_signatures, panels=panels,
                        prompt_ids=all_ids, decoder_families=families,
                    )
                    representative = str(
                        aggregate["legibility_argmax"]["canonical_representative"]
                    )
                    representative_position = all_ids.index(representative)
                    near_value = float(near_aggregate["prompt_value"][representative_position])
                    far_value = float(far_aggregate["prompt_value"][representative_position])
                    transfer = {
                        "embedding_model": "BAAI/bge-large-en-v1.5",
                        "near_value_at_legibility_representative": near_value,
                        "far_value_at_legibility_representative": far_value,
                        "far_near_ratio": (
                            float(far_value / near_value) if near_value > 1e-12 else None
                        ),
                        "near_independent_cap": near_aggregate["free_recombination_cap"],
                        "far_independent_cap": far_aggregate["free_recombination_cap"],
                        "diagnostic_only_not_primary_value": True,
                    }
                value_tables[(metric_key, str(instrument), channel, arm)] = np.asarray(
                    aggregate["prompt_value"], dtype=float
                )
                if audit is not None:
                    cell_alpha = float(freeze.get("process_bound_cell_alpha", 0.05))
                    stochastic_bound_alpha = cell_alpha / 2.0
                    process_bounds = {}
                    for horizon in (100, 300):
                        rows = {
                            "record_rank": record_rank_gain_bound(
                                n=(
                                    len(audit["prompt_value"])
                                    + (
                                        len(discovery["prompt_value"])
                                        if historical["record_rank_provenance_valid"] else 0
                                    )
                                ), m=horizon,
                                achieved=aggregate["achieved_value"],
                                cap=aggregate["free_recombination_cap"],
                            ),
                            "split_cp": split_sample_cp_gain_bound(
                                discovery_achieved=discovery["achieved_value"],
                                audit_values=audit["prompt_value"],
                                current_achieved=aggregate["achieved_value"],
                                cap=aggregate["free_recombination_cap"],
                                future_horizon=horizon, alpha=stochastic_bound_alpha,
                            ),
                            "dkw": dkw_expected_best_gain_bound(
                                observed_values=audit["prompt_value"],
                                achieved=aggregate["achieved_value"],
                                cap=aggregate["free_recombination_cap"],
                                future_horizon=horizon, alpha=stochastic_bound_alpha,
                            ),
                        }
                        per_family = {}
                        for audit_family in sorted(set(audit_families)):
                            mask = np.asarray(audit_families) == audit_family
                            family_values = np.asarray(audit["prompt_value"])[mask]
                            family_rows = {
                                "record_rank": record_rank_gain_bound(
                                    n=len(family_values), m=horizon,
                                    achieved=aggregate["achieved_value"],
                                    cap=aggregate["free_recombination_cap"],
                                ),
                                "split_cp": split_sample_cp_gain_bound(
                                    discovery_achieved=discovery["achieved_value"],
                                    audit_values=family_values,
                                    current_achieved=aggregate["achieved_value"],
                                    cap=aggregate["free_recombination_cap"],
                                    future_horizon=horizon, alpha=stochastic_bound_alpha,
                                ),
                                "dkw": dkw_expected_best_gain_bound(
                                    observed_values=family_values,
                                    achieved=aggregate["achieved_value"],
                                    cap=aggregate["free_recombination_cap"],
                                    future_horizon=horizon, alpha=stochastic_bound_alpha,
                                ),
                            }
                            per_family[audit_family] = {
                                "diagnostic_only": True, "bounds": family_rows,
                                "minimum_gain_upper": min(
                                    float(row["gain_upper"]) for row in family_rows.values()
                                ),
                            }
                        process_bounds[str(horizon)] = {
                            "bounds": rows,
                            "headline_gain_upper": min(
                                float(row["gain_upper"]) for row in rows.values()
                            ),
                            "headline_rule": "minimum of all premise-valid declared bounds",
                            "stochastic_bound_alpha": stochastic_bound_alpha,
                            "within_claim_alpha_split": "equal split across split-CP and DKW",
                            "per_family_diagnostics": per_family,
                        }
                    discovery_behavior_hashes = [
                        hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest()
                        for row in signatures
                    ]
                    discovery_codes = signatures_to_states(signatures, panels)
                    discovery_code_hashes = [
                        hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest()
                        for row in discovery_codes
                    ]
                    novelty_rows = novelty_collapse_curves(
                        full_signatures=audit_signatures,
                        joint_codes=signatures_to_states(audit_signatures, panels),
                        values=audit["prompt_value"],
                        frozen_incumbent=discovery["achieved_value"],
                        discovery_signature_hashes=discovery_behavior_hashes,
                        discovery_code_hashes=discovery_code_hashes,
                        families=audit_families,
                    )
                distortion = fidelity_legibility_diagnostic(
                    prompt_signatures_on_h=all_signatures[:, heldout], target_on_h=target_h,
                    legibility_values=aggregate["prompt_value"], prompt_ids=all_ids,
                )
                status = classify_status(
                    achieved=aggregate["achieved_value"],
                    cap=aggregate["free_recombination_cap"],
                    raw_panel_caps=aggregate["raw_panel_caps"],
                    future_gain_bound=(
                        None if process_bounds is None
                        else process_bounds["100"]["headline_gain_upper"]
                    ),
                    blind_value=control["blind_signal"],
                    best_annotated_value=control["best_annotated_signal"],
                )
                canonical_states = [
                    int("".join(map(str, panel["target_state_bits"])), 2)
                    for panel in design["panel_design"]["panels"]
                ]
                canonical_raw_lift = float(np.mean([
                    raw[position, state] for position, state in enumerate(canonical_states)
                ]))
                canonical_value = float(np.mean([
                    clipped[position, state] for position, state in enumerate(canonical_states)
                ]))
                artifact_root = (
                    root / "certificates" / _safe(metric_key) / str(instrument) /
                    (channel if arm is None else f"{channel}__{arm}")
                )
                state_arrays = {"raw_lift": raw, "clipped_value": clipped}
                if transfer_tables is not None:
                    state_arrays.update(dict(zip((
                        "near_raw_lift", "near_clipped_value",
                        "far_raw_lift", "far_clipped_value",
                    ), transfer_tables)))
                _atomic_npz(artifact_root / "state_tables.npz", **state_arrays)
                _atomic_json(artifact_root / "design_manifest.json", design)
                frame = pd.DataFrame({
                    "prompt_id": all_ids, "prompt_text": all_prompts,
                    "raw_lift": aggregate["prompt_raw_lift"],
                    "value": aggregate["prompt_value"],
                    "fidelity_bits": distortion["fidelity"],
                })
                _atomic_parquet(artifact_root / "prompt_values.parquet", frame)
                if novelty_rows is not None:
                    _atomic_parquet(
                        artifact_root / "novelty_curves.parquet", pd.DataFrame(novelty_rows)
                    )
                certificate = {
                    "schema": "cr3-v14-certificate-v1",
                    "lane": str(design.get("scoring_lane", {}).get("lane", "cert")),
                    "task": design["task"], "level": design["level"],
                    "metric": design["metric"], "metric_key": metric_key,
                    "instrument": instrument, "channel": channel, "arm": arm,
                    "executor": design["executor"],
                    "design_manifest_sha256": design["design_manifest_sha256"],
                    "achieved_value": aggregate["achieved_value"],
                    "free_recombination_cap": aggregate["free_recombination_cap"],
                    "structural_gap": aggregate["structural_gap"],
                    "annotated_canonical_raw_lift": canonical_raw_lift,
                    "annotated_canonical_value": canonical_value,
                    "control_diagnostics": control,
                    "near_far_transfer_diagnostic": transfer,
                    "panel_caps": aggregate["panel_caps"].tolist(),
                    "decoder_family_rows": aggregate.get("decoder_family_rows"),
                    "decoder_family_achieved_variance": aggregate.get(
                        "decoder_family_achieved_variance"
                    ),
                    "legibility_argmax": aggregate["legibility_argmax"],
                    "fidelity_argmax": distortion["fidelity_argmax"],
                    "oracle_omega_star": {
                        "definition": "decoder_free_argmax_of_behavioral_fidelity_on_H",
                        "tie_class": distortion["fidelity_argmax"],
                    },
                    "decoder_specific_optimum_instance": {
                        "definition": "reader_relative_argmax_of_reconstruction_legibility",
                        "tie_class": aggregate["legibility_argmax"],
                        "ranking_claim_allowed": distortion[
                            "optimal_prompt_ranking_allowed"
                        ],
                        "reporting_label": (
                            "decoder_specific_optimum_instance"
                            if distortion["optimal_prompt_ranking_allowed"]
                            else "tie_class_with_canonical_representative_no_optimal_claim"
                        ),
                    },
                    "distortion": {
                        key: value for key, value in distortion.items()
                        if key != "fidelity"
                    },
                    "status": status,
                    "behavioral_ceiling_contract": (
                        None if channel == "mcq" else {
                            "target_entropy_on_H_bits": design["target_entropy_on_h_bits"],
                            "identification_mi_is_not_a_behavioral_ceiling": True,
                        }
                    ),
                    "identification_diagnostic": {
                        "mean_I_z_M_bits": float(np.mean([
                            row["identification_mi_bits"]
                            for row in design["panel_design"]["panels"]
                        ])),
                        "mean_target_margin": float(np.mean([
                            row["target_margin"] for row in design["panel_design"]["panels"]
                        ])),
                        "n_target_uniqueness_exceptions": int(sum(
                            not row["target_unique"]
                            for row in design["panel_design"]["panels"]
                        )),
                        "scope": "identification_ceiling_and_panel_quality_diagnostic_only",
                        "behavioral_ceiling": False,
                    },
                    "ceiling_table": {
                        "instrument_free_recombination_cap": {
                            "value": aggregate["free_recombination_cap"],
                            "moves_with_decoder_tuning": True,
                            "role": "descriptive_ruler_not_headline",
                        },
                        "identification_I_z_M_bits": {
                            "value": float(np.mean([
                                row["identification_mi_bits"]
                                for row in design["panel_design"]["panels"]
                            ])),
                            "moves_with_decoder_tuning": False,
                            "role": "identification_only_not_behavioral_ceiling",
                        },
                        "target_entropy_on_H_bits": {
                            "value": design["target_entropy_on_h_bits"],
                            "moves_with_decoder_tuning": False,
                            "role": "behavioral_total_information_ceiling",
                        },
                    },
                    "gold_fidelity": design["entry"].get("gold_fidelity"),
                    "gold_fidelity_available": design["entry"].get("gold_fidelity") is not None,
                    "record_rank_harvested_provenance_valid": historical[
                        "record_rank_provenance_valid"
                    ],
                    "n_harvested_draws": historical["n_harvested_draws"],
                    "discovery_achieved_value": discovery["achieved_value"],
                    "audit_achieved_value": None if audit is None else audit["achieved_value"],
                    "process_relative_bounds": (
                        "pending_fresh_audit_stage" if process_bounds is None else process_bounds
                    ),
                }
                certificate["certificate_sha256"] = canonical_sha256(certificate)
                _atomic_json(artifact_root / "certificate.json", certificate)
                result_rows.append({
                    "lane": str(design.get("scoring_lane", {}).get("lane", "cert")),
                    "task": design["task"], "level": design["level"],
                    "metric": design["metric"], "metric_key": metric_key,
                    "instrument": instrument, "channel": channel, "arm": arm,
                    "executor": design["executor"]["model"],
                    "decoder_panel": "qwen14+llama70+mistral24",
                    "achieved_value": aggregate["achieved_value"],
                    "structural_cap": aggregate["free_recombination_cap"],
                    "structural_gap": aggregate["structural_gap"],
                    "decoder_family_variance": aggregate.get("decoder_family_achieved_variance"),
                    "gold_fidelity": design["entry"].get("gold_fidelity"),
                    "blind_to_best_gap": control["best_minus_blind_signal"],
                    "far_near_transfer_ratio": (
                        None if transfer is None else transfer["far_near_ratio"]
                    ),
                    "fidelity_legibility_spearman": distortion["spearman_rho"],
                    "ranking_allowed": distortion["optimal_prompt_ranking_allowed"],
                    "status": status["status"],
                    "certificate_path": str(artifact_root / "certificate.json"),
                })
    for row in result_rows:
        key = (str(row["metric_key"]), str(row["instrument"]))
        mcq = value_tables.get((*key, "mcq", None))
        behavioral = value_tables.get((*key, "behavioral", "unconstrained"))
        if mcq is None or behavioral is None or len(mcq) < 2:
            correlation = None
        elif np.all(mcq == mcq[0]) or np.all(behavioral == behavioral[0]):
            correlation = None
        else:
            observed = float(spearmanr(mcq, behavioral).statistic)
            correlation = observed if np.isfinite(observed) else None
        row["mcq_behavioral_spearman"] = correlation
        behavioral_rows = {
            str(candidate["arm"]): float(candidate["achieved_value"])
            for candidate in result_rows
            if str(candidate["metric_key"]) == key[0]
            and str(candidate["instrument"]) == key[1]
            and candidate["channel"] == "behavioral"
        }
        row["exemplar_vs_rule_achieved_gap_bits"] = (
            behavioral_rows.get("unconstrained", 0.0)
            - behavioral_rows.get("no_verbatim_examples", 0.0)
            if set(behavioral_rows) == set(BEHAVIORAL_ARMS) else None
        )
    frame = pd.DataFrame(result_rows)
    _atomic_parquet(root / "results.parquet", frame)
    _atomic_json(root / "campaign_manifest.json", {
        "schema": CAMPAIGN_SCHEMA,
        "lane": str(designs[0].get("scoring_lane", {}).get("lane", "cert")) if designs else None,
        "n_metrics": len(designs), "n_results": len(frame),
        "channels": list(channels), "template_freeze": str(Path(template_freeze_path).resolve()),
        "results_path": str(root / "results.parquet"),
        "gpu_restriction": "sk3 physical GPUs 1,2,3,4 forbidden",
        "scientific_gate_applied": False,
        "control_liveness_gate_applied": (root / "sentinel_report.json").is_file(),
        "sentinel_report": (
            str(root / "sentinel_report.json")
            if (root / "sentinel_report.json").is_file() else None
        ),
        "fresh_audit_complete": all(
            (root / "audit" / "signatures" / f"{design['metric_key']}.npz").is_file()
            for design in designs
        ),
    })
    return frame


def _decoder_models(arguments: Sequence[str] | None) -> dict[str, str]:
    if not arguments:
        return dict(DEFAULT_DECODER_MODELS)
    output = {}
    for value in arguments:
        if "=" not in value:
            raise ValueError("--decoder-model entries must be family=model")
        family, model = value.split("=", 1)
        output[family] = model
    if set(output) != set(DEFAULT_DECODER_MODELS):
        raise ValueError("decoder models must define qwen, llama, and mistral")
    return output


def _selected_constructor_model(
    template_freeze_path: str | Path, decoder_family: str,
    arguments: Sequence[str] | None,
) -> str:
    freeze = load_template_freeze(template_freeze_path)
    frozen = [
        row for row in freeze.get("decoder_panel", [])
        if str(row["family"]) == str(decoder_family)
    ]
    if not frozen:
        return _decoder_models(arguments)[decoder_family]
    if len(frozen) != 1:
        raise RuntimeError(f"freeze has an invalid {decoder_family} decoder selection")
    model = str(frozen[0]["model"])
    if arguments:
        requested = _decoder_models(arguments)[decoder_family]
        if requested != model:
            raise RuntimeError(
                f"requested constructor {requested!r} differs from frozen model {model!r}"
            )
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=[
        "extend-probes", "design", "promote", "prepare-dev", "prepare-sentinel", "qualify-constructor",
        "qualify-executor", "tune", "freeze", "seed-freeze", "constructor", "executor",
        "sentinel-constructor", "sentinel-executor", "sentinel-aggregate",
        "liveness-constructor", "liveness-executor", "sentinel-gate",
        "audit-proposer", "audit-score", "aggregate", "report",
    ])
    parser.add_argument("--metrics-manifest")
    parser.add_argument("--probe-extension-root")
    parser.add_argument("--probe-corpus-manifest")
    parser.add_argument("--scoring-lane", choices=["fast", "cert"], default="cert")
    parser.add_argument("--promotion-manifest")
    parser.add_argument("--fast-results")
    parser.add_argument("--top-k-per-task", type=int, default=3)
    parser.add_argument("--figure-metric-keys", nargs="*", default=[])
    parser.add_argument("--dev-metrics-manifest")
    parser.add_argument("--dev-min-tasks", type=int, default=7, help=(
        "minimum task span for the development pool; lowering below 7 is a "
        "declared deviation and must be recorded in the run artifacts"))
    parser.add_argument("--sentinel-metrics-manifest")
    parser.add_argument(
        "--sentinel-metric-keys", nargs="+", default=list(DEFAULT_SENTINEL_METRIC_KEYS),
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--run-sha", default="v14.0")
    parser.add_argument("--metric-keys", nargs="+")
    parser.add_argument("--template-freeze")
    parser.add_argument("--decoder-family", choices=sorted(DEFAULT_DECODER_MODELS))
    parser.add_argument("--audit-family", choices=sorted(AUDIT_FAMILIES))
    parser.add_argument("--decoder-model", action="append")
    parser.add_argument("--tuning-channel", choices=["mcq", "behavioral"])
    parser.add_argument("--tuning-arm", default="unconstrained")
    parser.add_argument("--proposer-model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--max-metric-calls", type=int, default=240, help=(
        "official-gepa budget for --phase tune (metric evaluations)"))
    parser.add_argument("--qualification", action="append")
    parser.add_argument("--release-commit")
    parser.add_argument("--sentinel-controls")
    parser.add_argument("--sentinel-report")
    parser.add_argument("--channels", nargs="+", choices=["mcq", "behavioral"],
                        default=["mcq", "behavioral"])
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--audit-budget", type=int, default=400)
    parser.add_argument("--physical-gpus", nargs="*")
    parser.add_argument("--fake-backends", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.out_root).resolve()
    if args.phase == "extend-probes":
        if not args.metrics_manifest or not args.probe_corpus_manifest:
            raise ValueError("extend-probes requires --metrics-manifest and --probe-corpus-manifest")
        build_probe_extensions(
            metrics_manifest_path=args.metrics_manifest,
            corpus_manifest_path=args.probe_corpus_manifest,
            out_root=args.probe_extension_root or root / "probe_extensions",
            run_sha=args.run_sha, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
        )
    elif args.phase == "design":
        if not args.metrics_manifest:
            raise ValueError("design phase requires --metrics-manifest")
        build_designs(
            metrics_manifest_path=args.metrics_manifest, out_root=root,
            run_sha=args.run_sha, metric_keys=args.metric_keys,
            probe_extension_root=args.probe_extension_root,
            scoring_lane=args.scoring_lane,
            promotion_manifest_path=args.promotion_manifest,
        )
    elif args.phase == "promote":
        if not args.fast_results:
            raise ValueError("promote requires --fast-results")
        build_promotion_manifest(
            args.fast_results,
            out_path=args.promotion_manifest or root / "promotion_manifest.json",
            run_sha=args.run_sha, top_k_per_task=args.top_k_per_task,
            figure_metric_keys=args.figure_metric_keys,
        )
    elif args.phase == "prepare-dev":
        if not args.dev_metrics_manifest:
            raise ValueError("prepare-dev phase requires --dev-metrics-manifest")
        prepare_development_population(
            certified_out_root=root,
            dev_metrics_manifest_path=args.dev_metrics_manifest,
            run_sha=args.run_sha,
            probe_extension_root=args.probe_extension_root,
            dev_min_tasks=args.dev_min_tasks,
        )
    elif args.phase == "prepare-sentinel":
        if not args.sentinel_metrics_manifest:
            raise ValueError("prepare-sentinel requires --sentinel-metrics-manifest")
        prepare_sentinel_population(
            out_root=root,
            sentinel_metrics_manifest_path=args.sentinel_metrics_manifest,
            run_sha=args.run_sha, sentinel_metric_keys=args.sentinel_metric_keys,
        )
    elif args.phase == "qualify-constructor":
        if not args.decoder_family:
            raise ValueError("qualify-constructor requires --decoder-family")
        model = _decoder_models(args.decoder_model)[args.decoder_family]
        run_qualification_constructor(
            out_root=root, decoder_family=args.decoder_family, decoder_model=model,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
        )
    elif args.phase == "qualify-executor":
        if not args.decoder_family:
            raise ValueError("qualify-executor requires --decoder-family")
        model = _decoder_models(args.decoder_model)[args.decoder_family]
        run_qualification_executor(
            out_root=root, decoder_family=args.decoder_family, decoder_model=model,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
        )
    elif args.phase == "tune":
        if not args.tuning_channel:
            raise ValueError("tune phase requires --tuning-channel")
        run_decoder_tuning(
            out_root=root, channel=args.tuning_channel, arm=args.tuning_arm,
            decoder_models=_decoder_models(args.decoder_model),
            proposer_model=args.proposer_model, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
            max_metric_calls=args.max_metric_calls,
        )
    elif args.phase == "freeze":
        if not args.qualification or not args.release_commit:
            raise ValueError("freeze requires three --qualification specs and --release-commit")
        freeze_production_instrument(
            out_root=root, qualification_specs=args.qualification,
            release_commit=args.release_commit,
        )
    elif args.phase == "sentinel-gate":
        if not args.sentinel_controls:
            raise ValueError("sentinel-gate requires --sentinel-controls")
        apply_sentinel_gate(
            control_rows_path=args.sentinel_controls, out_root=root,
        )
    elif args.phase == "liveness-constructor":
        if not args.decoder_family:
            raise ValueError("liveness-constructor requires --decoder-family")
        run_liveness_constructor_stage(
            out_root=root, decoder_family=args.decoder_family,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
        )
    elif args.phase == "liveness-executor":
        run_liveness_executor_stage(
            out_root=root, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
        )
    elif args.phase == "seed-freeze":
        destination = Path(args.template_freeze or root / "template_freeze.json")
        if destination.exists():
            raise FileExistsError(destination)
        write_seed_template_freeze(destination)
    elif args.phase == "constructor":
        if not args.template_freeze or not args.decoder_family:
            raise ValueError("constructor phase requires --template-freeze and --decoder-family")
        model = _selected_constructor_model(
            args.template_freeze, args.decoder_family, args.decoder_model,
        )
        run_constructor_stage(
            out_root=root, template_freeze_path=args.template_freeze,
            decoder_family=args.decoder_family,
            decoder_model=model, metric_keys=args.metric_keys,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            channels=args.channels, query_batch_size=args.query_batch_size,
            sentinel_report_path=args.sentinel_report,
        )
    elif args.phase == "sentinel-constructor":
        if not args.template_freeze or not args.decoder_family:
            raise ValueError(
                "sentinel-constructor requires --template-freeze and --decoder-family"
            )
        model = _selected_constructor_model(
            args.template_freeze, args.decoder_family, args.decoder_model,
        )
        run_constructor_stage(
            out_root=root / "sentinel", template_freeze_path=args.template_freeze,
            decoder_family=args.decoder_family, decoder_model=model,
            metric_keys=args.metric_keys, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            channels=args.channels, query_batch_size=args.query_batch_size,
            require_sentinel=False,
        )
    elif args.phase == "executor":
        if not args.template_freeze:
            raise ValueError("executor phase requires --template-freeze")
        run_executor_stage(
            out_root=root, template_freeze_path=args.template_freeze,
            metric_keys=args.metric_keys, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size,
            sentinel_report_path=args.sentinel_report,
        )
    elif args.phase == "sentinel-executor":
        if not args.template_freeze:
            raise ValueError("sentinel-executor requires --template-freeze")
        run_executor_stage(
            out_root=root / "sentinel", template_freeze_path=args.template_freeze,
            metric_keys=args.metric_keys, fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            query_batch_size=args.query_batch_size, require_sentinel=False,
        )
    elif args.phase == "sentinel-aggregate":
        if not args.template_freeze:
            raise ValueError("sentinel-aggregate requires --template-freeze")
        aggregate_campaign(
            out_root=root / "sentinel", template_freeze_path=args.template_freeze,
            metric_keys=args.metric_keys, channels=args.channels,
        )
    elif args.phase == "audit-proposer":
        if not args.audit_family:
            raise ValueError("audit-proposer phase requires --audit-family")
        run_audit_proposer_stage(
            out_root=root, audit_family=args.audit_family, metric_keys=args.metric_keys,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            total_budget=args.audit_budget,
            sentinel_report_path=args.sentinel_report,
        )
    elif args.phase == "audit-score":
        run_audit_score_stage(
            out_root=root, metric_keys=args.metric_keys,
            fake_backends=args.fake_backends,
            physical_gpu_ids=parse_physical_gpu_ids(args.physical_gpus),
            total_budget=args.audit_budget, query_batch_size=args.query_batch_size,
            sentinel_report_path=args.sentinel_report,
        )
    elif args.phase == "aggregate":
        if not args.template_freeze:
            raise ValueError("aggregate phase requires --template-freeze")
        aggregate_campaign(
            out_root=root, template_freeze_path=args.template_freeze,
            metric_keys=args.metric_keys, channels=args.channels,
        )
    elif args.phase == "report":
        report = audit_release(root)
        write_release_outputs(root, report)
        if not report["complete"]:
            raise RuntimeError("v14 release completion audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
