#!/usr/bin/env python3
"""Adaptive mining for an executor-indexed best-single-prompt ceiling.

Monitoring batches may decide when to stop, but are absorbed only through an ordered,
append-only ledger.  Once stopping fires, a separately seeded confirmation audit is
scored against the final frozen pool and never absorbed.  The confirmation calls
``prompt_articulation_certificate``; it simultaneously bounds behavioral missing
mass, exact-pattern missing mass, and expected best-prompt recovery at a declared
future draw horizon.

The run is resumable without directory scanning.  Completed but uncommitted batches
are reused, committed batches are reconstructed in ledger order, and confirmation
artifacts occupy a namespace the pool loader never reads.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from methods.metric_implementer.experiments.cr_audit import (  # noqa: E402
    SCHEMA_VERSION,
    all_finite_prompt_dpi_certificate,
    classify_prompt_evolution,
    prompt_articulation_certificate,
)
from methods.metric_implementer.experiments.cr3_reconstruction_values import (  # noqa: E402
    build_codebook_panel_plan,
    build_frozen_codebook_manifest,
    load_value_artifact,
    select_prior_balanced_panels,
    validate_codebook_manifest,
)
from methods.metric_implementer.experiments.cr3_evidence_store import (  # noqa: E402
    install_evidence_store,
    load_evidence_manifest,
)
from methods.metric_implementer.experiments.mine_clusters import (  # noqa: E402
    r1_groups,
    r2_groups,
    r3_groups,
)
from methods.metric_implementer.recon_channel import mcq_identity_channel  # noqa: E402

DEFAULT_WORKER = REPO / "scripts" / "tools" / "cr3_mining_worker.py"
DEFAULT_PYTHON = Path("/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python")
DEFAULT_WORKER_HOME = Path("/lfs/skampere3/0/alexspan")
LEDGER_SCHEMA = "cr3-ledger-v3"
MANIFEST_SCHEMA = "cr3-run-v11"
BOOTSTRAP_SCHEMA = "cr3-bootstrap-v2"
CODEBOOK_BOOTSTRAP_SCHEMA = "cr3-codebook-bootstrap-v1"
AUDIT_SIG_SCHEMA = "cr3-audit-signatures-v2"
SUPPORTED_TASK_PREFIXES = (
    "creative-writing",
    "humor",
    "news-homepages",
    "press-releases",
    "code-review",
    "math-stackexchange",
    "grant-funding",
    "peer-review",
    "legal-outcome-prediction",
)


def stable_seed(*parts: object) -> int:
    raw = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8]
    return int.from_bytes(raw, "big") & ((1 << 63) - 1)


def checkpoint_iterations(args) -> tuple[int, ...]:
    raw = str(getattr(args, "checkpoint_iters", "") or "")
    if not raw.strip():
        return ()
    try:
        values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    except ValueError as exc:
        raise ValueError("--checkpoint-iters must be comma-separated integers") from exc
    if any(value < 0 or value >= int(args.max_iter) for value in values):
        raise ValueError("checkpoint iterations must lie in [0, max_iter)")
    return values


def confirmation_alpha(args, *, n_metrics: int) -> tuple[float, dict]:
    """Alpha per immutable checkpoint/final cell and its declared simultaneity scope."""
    n_slots = len(checkpoint_iterations(args)) + 1  # checkpoints plus final confirmation
    study_alpha = getattr(args, "study_alpha", None)
    if study_alpha is None:
        cell_alpha = float(args.alpha) / n_slots
        scope = "simultaneous over all checkpoint/final claims for one metric"
        overall_alpha = float(args.alpha)
    else:
        cell_alpha = float(study_alpha) / (int(n_metrics) * n_slots)
        scope = "familywise simultaneous over all declared metrics and checkpoint/final claims"
        overall_alpha = float(study_alpha)
    return cell_alpha, {
        "scope": scope,
        "overall_alpha": overall_alpha,
        "cell_alpha": cell_alpha,
        "n_metrics": int(n_metrics),
        "n_slots_per_metric": int(n_slots),
        "checkpoint_iterations": list(checkpoint_iterations(args)),
    }


def target_value_gap(args) -> float:
    configured = getattr(args, "target_value_gap", None)
    return float(args.target_gap_bits if configured is None else configured)


def mcq_instrument_quality(state: dict, args) -> dict:
    fixed = np.asarray(state["fixed_no_demo_canonical_choice_probabilities"], float)
    if (fixed.ndim != 2 or fixed.shape[1] < 2
            or np.any(~np.isfinite(fixed)) or np.any(fixed < 0.0)):
        raise RuntimeError("cannot diagnose an invalid frozen no-demo choice channel")
    prior = fixed.mean(axis=0)
    prior = prior / prior.sum()
    positive = prior > 0.0
    entropy = float(-np.sum(prior[positive] * np.log2(prior[positive])))
    normalized_entropy = entropy / float(np.log2(len(prior)))
    entry = state["mcq_codebook_entry"]
    selected = list(entry["distractor_design_statistics"])
    min_kappa = float(min(row["kappa"] for row in selected))
    prior_calibration = entry.get("prior_calibration") or {}
    calibrated_prior = ((prior_calibration.get("prior") or {}).get(
        "canonical_mean_prior"))
    if calibrated_prior is not None and not np.allclose(
            np.asarray(calibrated_prior, float), prior, rtol=0.0, atol=1e-12):
        raise RuntimeError("frozen codebook prior calibration differs from the value channel")
    reasons = []
    if state["value_cap"] < args.mcq_min_headline_value_cap:
        reasons.append("frozen no-demo target prior leaves too little value headroom")
    if min_kappa < args.mcq_min_headline_distractor_kappa:
        reasons.append("selected distractor panel is behaviorally too easy")
    if not prior_calibration.get("passes_prior_balance", False):
        reasons.append("no candidate menu passed the predeclared blind no-demo prior-balance gate")
    return {
        "status": "HEADLINE_ELIGIBLE" if not reasons else "FORMAL_CERTIFICATE_ONLY",
        "headline_eligible": not reasons,
        "formal_all_prompt_bound_valid": True,
        "reasons": reasons,
        "thresholds": {
            "minimum_value_cap": float(args.mcq_min_headline_value_cap),
            "minimum_selected_distractor_kappa": float(
                args.mcq_min_headline_distractor_kappa),
        },
        "no_demo_canonical_option_prior": prior.tolist(),
        "no_demo_target_probability": float(prior[0]),
        "no_demo_prior_entropy_bits": entropy,
        "no_demo_prior_normalized_entropy": normalized_entropy,
        "no_demo_prior_total_variation_from_uniform": float(
            0.5 * np.abs(prior - 1.0 / len(prior)).sum()),
        "global_value_cap": float(state["value_cap"]),
        "target_design_yes_rate": float(entry["target_design_yes_rate"]),
        "selected_distractor_kappa_min": min_kappa,
        "selected_distractor_kappa_mean": float(np.mean(
            [row["kappa"] for row in selected])),
        "selected_distractor_disagreements_min": int(min(
            row["n_disagree"] for row in selected)),
        "selected_distractors": selected,
        "prior_calibration": prior_calibration,
        "scope": (
            "descriptive gate for scientific headline use; failure does not invalidate the "
            "fixed-instrument all-prompt inequality"),
    }


def _worker_environment(args) -> dict[str, str]:
    """Pin writable runtime/cache roots before the worker imports vLLM or Triton."""
    environment = dict(os.environ)
    home = str(Path(args.worker_home).resolve())
    environment.update({
        "HOME": home,
        "XDG_CACHE_HOME": str(Path(home) / ".cache"),
        "TRITON_CACHE_DIR": str(Path(home) / ".triton" / "cache"),
        "VLLM_CONFIG_ROOT": str(Path(home) / ".config" / "vllm"),
        "VLLM_NO_USAGE_STATS": "1",
    })
    return environment


def _retryable_worker_failure(output: str) -> bool:
    return "Error in memory profiling" in output and "current free memory" in output


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fin:
        for block in iter(lambda: fin.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: str | Path, payload: object, *, immutable: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if immutable and target.read_text() == encoded:
            return
        raise FileExistsError(f"refusing to overwrite {target}")
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("x", encoding="utf-8") as fout:
            fout.write(encoded)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, target)
        _fsync_directory(target.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _append_jsonl(path: str | Path, row: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, sort_keys=True) + "\n")
        fout.flush()
        os.fsync(fout.fileno())
    _fsync_directory(target.parent)


def _read_jsonl(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


@contextlib.contextmanager
def run_lock(root: Path):
    path = root / ".run.lock"
    lock = path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(f"another CR-3 process holds {path}") from exc
    lock.seek(0)
    lock.truncate()
    lock.write(json.dumps({"pid": os.getpid(), "host": os.uname().nodename, "time": time.time()}))
    lock.flush()
    try:
        yield
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _metric_task(key: str) -> str:
    for task in SUPPORTED_TASK_PREFIXES:
        if key.startswith(f"{task}_"):
            return task
    raise ValueError(f"cannot infer task from metric key {key!r}")


def _metric_level(key: str) -> str:
    match = re.search(r"_R([123])_metric", key, flags=re.IGNORECASE)
    return f"R{match.group(1)}" if match else ""


def _normalized_name(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _metric_identity(path: str, bucket: str) -> dict:
    source = Path(path).resolve()
    z = np.load(source, allow_pickle=True)
    key = source.name.replace("_sigs.npz", "")
    task = _metric_task(key)
    name = str(z["name"]) if "name" in z else key
    description = ""
    for field in ("metric_description", "description", "merged_description"):
        if field in z and str(z[field]).strip():
            description = str(z[field]).strip()
            break
    level = _metric_level(key)
    group_index = int(z["r2_idx"]) if "r2_idx" in z else None
    if not description:
        if not level or group_index is None:
            raise RuntimeError(
                f"{source} lacks a metric description and its hierarchy identity cannot be inferred")
        groups = ({"R1": r1_groups, "R2": r2_groups, "R3": r3_groups}[level]
                  (task, bucket) if level != "R1" else r1_groups(task))
        if not 0 <= group_index < len(groups):
            raise RuntimeError(f"hierarchy index {group_index} is out of range for {task}/{bucket}/{level}")
        group = groups[group_index]
        hierarchy_name = str(group.get("merged_name") or group.get("name") or "")
        if hierarchy_name and _normalized_name(hierarchy_name) != _normalized_name(name):
            raise RuntimeError(
                f"checkpoint/hierarchy name mismatch for {source}: {name!r} != {hierarchy_name!r}")
        description = str(group.get("merged_description") or group.get("description") or "").strip()
    if not description:
        raise RuntimeError(f"no substantive metric description found for {source}")
    if "prompts" not in z or "tags" not in z or "M_i" not in z:
        raise RuntimeError(f"{source} is missing prompts, tags, or M_i")
    return {
        "key": key,
        "task": task,
        "level": level or None,
        "group_index": group_index,
        "name": name,
        "description": description,
        "description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        "target_orbit_forms": max(1, int(z["orbit_forms"]) if "orbit_forms" in z else 1),
        "n_source_prompts": int(len(z["prompts"])),
    }


def _resolved_unique_paths(paths: Iterable[str]) -> list[str]:
    resolved = []
    seen = set()
    for path in paths:
        value = str(Path(path).resolve())
        if value not in seen:
            seen.add(value)
            resolved.append(value)
    return resolved


def _mcq_codebook_metric_paths(args) -> list[str]:
    if args.mcq_codebook_metrics is None:
        return _resolved_unique_paths(args.metrics)
    # Targets must always be members of the frozen candidate bank, even when the
    # caller lists only the additional distractor candidates.
    return _resolved_unique_paths([*args.metrics, *args.mcq_codebook_metrics])


def _manifest_payload(args) -> dict:
    metrics = _resolved_unique_paths(args.metrics)
    codebook_metrics = _mcq_codebook_metric_paths(args)
    worker = str(Path(args.worker).resolve())
    core = {
        "schema": MANIFEST_SCHEMA,
        "metrics": metrics,
        "metric_sha256": {p: file_sha256(p) for p in metrics},
        "metric_identity": {p: _metric_identity(p, args.r2_bucket) for p in metrics},
        "families": list(args.families),
        "family_tags": list(args.family_tags),
        "family_modes": list(args.family_modes),
        "executor": args.executor,
        "value_mode": args.value_mode,
        "mcq_reconstructor": args.mcq_reconstructor,
        "mcq_n_options": args.mcq_n_options,
        "mcq_design_size": args.mcq_design_size,
        "mcq_min_design_disagreements": args.mcq_min_design_disagreements,
        "mcq_n_examples": args.mcq_n_examples,
        "mcq_reconstruction_draws": args.mcq_reconstruction_draws,
        "mcq_choice_readout": args.mcq_choice_readout,
        "mcq_value_query_batch_size": args.mcq_value_query_batch_size,
        "mcq_min_headline_value_cap": args.mcq_min_headline_value_cap,
        "mcq_min_headline_distractor_kappa": args.mcq_min_headline_distractor_kappa,
        "mcq_prior_candidate_pool_size": args.mcq_prior_candidate_pool_size,
        "mcq_prior_max_panels_per_target": args.mcq_prior_max_panels_per_target,
        "mcq_prior_max_option_probability": args.mcq_prior_max_option_probability,
        "mcq_prior_target_probability_tolerance": args.mcq_prior_target_probability_tolerance,
        "mcq_prior_min_normalized_entropy": args.mcq_prior_min_normalized_entropy,
        "mcq_codebook_metrics": codebook_metrics,
        "mcq_codebook_metric_sha256": {
            path: file_sha256(path) for path in codebook_metrics
        },
        "mcq_codebook_metric_identity": {
            path: _metric_identity(path, args.r2_bucket) for path in codebook_metrics
        },
        "mcq_choice_probability_cache_schema": "cr3-choice-probability-cache-v1",
        "temperature": args.temp,
        "batch_per_family": args.batch_per_family,
        "confirm_per_family": args.confirm_per_family,
        "checkpoint_per_family": args.checkpoint_per_family,
        "checkpoint_iterations": list(checkpoint_iterations(args)),
        "ceiling_horizon_per_family": args.ceiling_horizon_per_family,
        "target_u0": args.target_u0,
        "target_gap_bits": args.target_gap_bits,
        "target_value_gap": target_value_gap(args),
        "max_iter": args.max_iter,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "min_delta_bits": args.min_delta_bits,
        "alpha": args.alpha,
        "study_alpha": args.study_alpha,
        "tau": args.tau,
        "tau_strict": args.tau_strict,
        "p_min": args.p_min,
        "value_p_min": args.value_p_min,
        "reuse_bootstrap_root": (
            str(Path(args.reuse_bootstrap_root).resolve())
            if args.reuse_bootstrap_root else None),
        "reuse_bootstrap_manifest_sha256": (
            file_sha256(Path(args.reuse_bootstrap_root).resolve() / "run_manifest.json")
            if args.reuse_bootstrap_root else None),
        "reuse_mcq_codebook_root": (
            str(Path(args.reuse_mcq_codebook_root).resolve())
            if args.reuse_mcq_codebook_root else None),
        "reuse_mcq_codebook_manifest_sha256": (
            file_sha256(Path(args.reuse_mcq_codebook_root).resolve() / "run_manifest.json")
            if args.reuse_mcq_codebook_root else None),
        "reuse_evidence_root": (
            str(Path(args.reuse_evidence_root).resolve())
            if args.reuse_evidence_root else None),
        "reuse_evidence_manifest_sha256": (
            load_evidence_manifest(args.reuse_evidence_root)["manifest_sha256"]
            if args.reuse_evidence_root else None),
        "reuse_evidence_manifest_file_sha256": (
            file_sha256(Path(args.reuse_evidence_root).resolve() / "evidence_manifest.json")
            if args.reuse_evidence_root else None),
        "r2_bucket": args.r2_bucket,
        "worker": worker,
        "worker_environment": {
            key: _worker_environment(args)[key]
            for key in (
                "HOME", "XDG_CACHE_HOME", "TRITON_CACHE_DIR",
                "VLLM_CONFIG_ROOT", "VLLM_NO_USAGE_STATS",
            )
        },
        "worker_max_attempts": args.worker_max_attempts,
        "worker_retry_delay_seconds": args.worker_retry_delay_seconds,
        "code_sha256": {
            "worker": file_sha256(worker),
            "orchestrator": file_sha256(__file__),
            "certificate": file_sha256(Path(__file__).with_name("cr_audit.py")),
            "reconstruction_values": file_sha256(
                Path(__file__).with_name("cr3_reconstruction_values.py")),
            "evidence_store": file_sha256(
                Path(__file__).with_name("cr3_evidence_store.py")),
            "vllm_backend": file_sha256(REPO / "methods" / "metric_implementer" / "vllm_backend.py"),
            "recon_channel": file_sha256(REPO / "methods" / "metric_implementer" / "recon_channel.py"),
            "alpha_probe": file_sha256(Path(__file__).with_name("alpha_probe.py")),
        },
        "dry_run": bool(args.dry_run),
    }
    core["run_id"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return core


def prepare_manifest(root: Path, args) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_manifest.json"
    expected = _manifest_payload(args)
    if path.exists():
        observed = json.loads(path.read_text())
        if observed != expected:
            raise RuntimeError("run arguments do not match the immutable run manifest")
        return observed
    non_lock_entries = [p for p in root.iterdir() if p.name != ".run.lock"]
    if non_lock_entries:
        raise RuntimeError(
            f"refusing to use nonempty output root without a v2 manifest: {root}; choose a new root"
        )
    _atomic_json(path, expected)
    return expected


def _write_jobs(path: Path, items: list[dict]) -> None:
    _atomic_json(path, items)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        with tmp.open("rb") as fin:
            os.fsync(fin.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _dry_propose(items: list[dict], family: str, model: str, temperature: float) -> None:
    for job in items:
        proposal_mode = str(job.get("proposal_mode", "atomic"))
        rows = []
        attempts = []
        for i in range(int(job["n"])):
            seed = stable_seed(job["base_seed"], i)
            text = f"Does synthetic criterion {seed} hold?"
            row = {
                "text": text,
                "family": family,
                "model": model,
                "model_revision": model,
                "temperature": temperature,
                "proposal_mode": proposal_mode,
                "seed": seed,
                "attempt_idx": i,
                "accepted_idx": i,
                "prompt_sha256": "dry-prompt",
                "prompt_template_id": f"dry-{proposal_mode}",
                "validator_id": f"dry-{proposal_mode}",
                "generator_config_sha256": "dry-config",
            }
            rows.append(row)
            attempts.append({**row, "valid": True, "raw_text": text, "raw_sha256": "dry"})
        _atomic_jsonl(job["out"], rows)


def _atomic_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("x", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, sort_keys=True) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, target)
        _fsync_directory(target.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _dry_score(items: list[dict], worlds: dict[str, dict], executor: str) -> None:
    for job in items:
        if job.get("mode") in {"bootstrap", "codebook_bootstrap"}:
            z = np.load(job["orig_npz"], allow_pickle=True)
            target = np.asarray(z["M_i"], float)
            n_forms = max(1, int(job.get("target_orbit_forms", 1)))
            codebook_only = job.get("mode") == "codebook_bootstrap"
            sigs = target[None, :] if codebook_only else np.asarray(z["sigs"], float)
            texts = ([job["metric_description"]] if codebook_only
                     else [str(x) for x in z["prompts"]])
            source_tags = (["canonical_codebook_target"] if codebook_only
                           else [str(x) for x in z["tags"]])
            probe_texts = [f"dry probe {i}" for i in range(sigs.shape[1])]
            probe_sha = hashlib.sha256(
                json.dumps(probe_texts, separators=(",", ":")).encode()).hexdigest()
            _atomic_npz(
                Path(job["out"]),
                schema=np.asarray(
                    CODEBOOK_BOOTSTRAP_SCHEMA if codebook_only else BOOTSTRAP_SCHEMA),
                sigs=sigs,
                texts=np.asarray(texts, object),
                source_tags=np.asarray(source_tags, object),
                target=target,
                target_forms=np.tile(target[None, :], (n_forms, 1)),
                target_form_names=np.asarray(
                    ["canonical"] + [f"dry_form_{i}" for i in range(1, n_forms)], object),
                target_form_texts=np.asarray([job["metric_description"]] * n_forms, object),
                metric_description=np.asarray(job["metric_description"]),
                probe_texts=np.asarray(probe_texts, object),
                probe_sha256=np.asarray(probe_sha),
                executor_model=np.asarray(executor),
                executor_model_revision=np.asarray(executor),
                executor_temperature=np.asarray(0.0),
                readout_id=np.asarray("dry"),
                cache_namespace_sha256=np.asarray("dry-cache"),
                source_checkpoint=np.asarray(job["orig_npz"]),
                source_checkpoint_sha256=np.asarray(file_sha256(job["orig_npz"])),
                metric_key=np.asarray(str(job.get("metric_key", ""))),
                legacy_alignment_json=np.asarray("{}"),
            )
            continue
        rows = []
        for path in job["criteria"]:
            rows.extend(_read_jsonl(path))
        world = worlds[job["out_key"]]
        sigs = []
        for row in rows:
            rng = np.random.default_rng(int(row["seed"]))
            species = int(rng.choice(len(world["p"]), p=world["p"]))
            sigs.append(world["cols"][species] * 0.98 + 0.01)
        _atomic_npz(
            Path(job["out"]),
            schema=np.asarray(AUDIT_SIG_SCHEMA),
            sigs=np.asarray(sigs, float),
            texts=np.asarray([r["text"] for r in rows], object),
            families=np.asarray([r["family"] for r in rows], object),
            models=np.asarray([r["model"] for r in rows], object),
            model_revisions=np.asarray([r["model_revision"] for r in rows], object),
            temperatures=np.asarray([r["temperature"] for r in rows], float),
            seeds=np.asarray([r["seed"] for r in rows], np.int64),
            attempt_idx=np.asarray([r["attempt_idx"] for r in rows], np.int64),
            accepted_idx=np.asarray([r["accepted_idx"] for r in rows], np.int64),
            prompt_sha256=np.asarray([r["prompt_sha256"] for r in rows], object),
            generator_config_sha256=np.asarray([r["generator_config_sha256"] for r in rows], object),
            probe_sha256=np.asarray(job.get("expected_probe_sha256", "dry-probes")),
            executor_model=np.asarray(executor),
            executor_model_revision=np.asarray(
                job.get("expected_executor_model_revision", executor)),
            executor_temperature=np.asarray(0.0),
            readout_id=np.asarray("dry"),
            cache_namespace_sha256=np.asarray("dry-cache"),
            source_criteria=np.asarray(job["criteria"], object),
        )


def run_stage(*, stage: str, items: list[dict], jobs_file: Path, model: str,
              family: str, temperature: float, args, worlds: dict[str, dict]) -> None:
    if not items:
        return
    digest = hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    jobs_file = jobs_file.with_name(f"{jobs_file.stem}-{digest}{jobs_file.suffix}")
    _write_jobs(jobs_file, items)
    if args.dry_run:
        if stage == "propose":
            _dry_propose(items, family, model, temperature)
        elif stage == "score":
            _dry_score(items, worlds, model)
        elif stage == "value":
            from types import SimpleNamespace
            from scripts.tools.cr3_mining_worker import stage_value
            stage_value(SimpleNamespace(jobs=str(jobs_file), model=model, fake=True))
        else:
            from types import SimpleNamespace
            from scripts.tools.cr3_mining_worker import stage_codebook_prior
            stage_codebook_prior(SimpleNamespace(jobs=str(jobs_file), model=model, fake=True))
        return
    cmd = [args.worker_python, args.worker, "--stage", stage, "--jobs", str(jobs_file),
           "--model", model]
    if stage == "propose":
        cmd += ["--family", family, "--temp", str(temperature)]
    attempt = 1
    while jobs_file.with_suffix(f".attempt-{attempt:02d}.log").exists():
        attempt += 1
    max_attempts = int(getattr(args, "worker_max_attempts", 1))
    retry_delay = float(getattr(args, "worker_retry_delay_seconds", 10.0))
    for retry_index in range(max_attempts):
        log = jobs_file.with_suffix(f".attempt-{attempt:02d}.log")
        with log.open("x", encoding="utf-8") as fout:
            result = subprocess.run(
                cmd,
                stdout=fout,
                stderr=subprocess.STDOUT,
                text=True,
                env=_worker_environment(args),
            )
        if result.returncode == 0:
            return
        output = log.read_text(errors="replace")
        tail = "\n".join(output.splitlines()[-40:])
        may_retry = (
            retry_index + 1 < max_attempts
            and _retryable_worker_failure(output)
            and not any(Path(item["out"]).exists() for item in items)
        )
        if not may_retry:
            raise RuntimeError(f"worker {stage} failed rc={result.returncode}\n{tail}")
        time.sleep(retry_delay)
        attempt += 1


def _validate_proposal(path: Path, family: str, n: int,
                       proposal_mode: str | None = None) -> None:
    if not path.exists():
        raise RuntimeError(f"missing proposal transaction: {path}")
    rows = _read_jsonl(path)
    if len(rows) != n or any(str(row.get("family")) != family for row in rows):
        raise RuntimeError(f"invalid proposal quota in {path}")
    if proposal_mode is not None and any(
            str(row.get("proposal_mode", "atomic")) != proposal_mode for row in rows):
        raise RuntimeError(f"proposal mode changed in {path}")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != n:
        raise RuntimeError(f"non-unique per-draw seeds in {path}")


def load_scored(path: Path, family_names: list[str], expected_per_family: int) -> tuple[np.ndarray, list[str], dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    if str(z["schema"]) != AUDIT_SIG_SCHEMA:
        raise RuntimeError(f"unexpected audit-signature schema in {path}")
    sigs = np.asarray(z["sigs"], float)
    families = [str(x) for x in z["families"]]
    if sigs.ndim != 2 or np.any(~np.isfinite(sigs)):
        raise RuntimeError(f"invalid signatures in {path}")
    counts = {f: families.count(f) for f in family_names}
    if set(families) != set(family_names) or counts != {f: expected_per_family for f in family_names}:
        raise RuntimeError(f"scored family quota mismatch in {path}: {counts}")
    seeds = np.asarray(z["seeds"], np.int64)
    for family in family_names:
        fam_seeds = seeds[np.asarray(families) == family]
        if len(set(map(int, fam_seeds))) != expected_per_family:
            raise RuntimeError(f"non-unique seeds for {family} in {path}")
    texts = [str(x) for x in z["texts"]]
    by_text: dict[str, bytes] = {}
    binary = (sigs > 0.5).astype(np.uint8)
    for text, col in zip(texts, binary):
        previous = by_text.setdefault(text, col.tobytes())
        if previous != col.tobytes():
            raise RuntimeError(f"duplicate text has inconsistent cached signatures in {path}")
    meta = {
        "probe_sha256": str(z["probe_sha256"]),
        "executor_model": str(z["executor_model"]),
        "executor_model_revision": str(z["executor_model_revision"]),
        "readout_id": str(z["readout_id"]),
        "cache_namespace_sha256": str(z["cache_namespace_sha256"]),
        "artifact_sha256": file_sha256(path),
    }
    return binary, families, meta


def load_bootstrap(path: Path, identity: dict, source_sha256: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    if str(z["schema"]) != BOOTSTRAP_SCHEMA:
        raise RuntimeError(f"unexpected bootstrap schema in {path}")
    if str(z["source_checkpoint_sha256"]) != source_sha256:
        raise RuntimeError(f"bootstrap source hash mismatch in {path}")
    if str(z["metric_description"]) != identity["description"]:
        raise RuntimeError(f"bootstrap metric description mismatch in {path}")
    sigs = np.asarray(z["sigs"], float)
    target = np.asarray(z["target"], float)
    target_forms = np.asarray(z["target_forms"], float)
    target_form_names = [str(x) for x in z["target_form_names"]]
    target_form_texts = [str(x) for x in z["target_form_texts"]]
    texts = [str(x) for x in z["texts"]]
    probes = [str(x) for x in z["probe_texts"]]
    if (sigs.ndim != 2 or len(sigs) != identity["n_source_prompts"]
            or sigs.shape[1] != len(target) or len(probes) != len(target)
            or len(texts) != len(sigs)
            or target_forms.shape != (identity["target_orbit_forms"], len(target))
            or len(target_form_names) != identity["target_orbit_forms"]
            or len(target_form_texts) != identity["target_orbit_forms"]):
        raise RuntimeError(f"invalid bootstrap dimensions in {path}")
    if np.any(~np.isfinite(sigs)) or np.any(~np.isfinite(target)):
        raise RuntimeError(f"non-finite bootstrap scores in {path}")
    if np.any(~np.isfinite(target_forms)) or not np.allclose(target, target_forms.mean(axis=0)):
        raise RuntimeError(f"bootstrap target does not match its frozen form average in {path}")
    binary = (sigs > 0.5).astype(np.uint8)
    by_text: dict[str, bytes] = {}
    for text, row in zip(texts, binary):
        previous = by_text.setdefault(text, row.tobytes())
        if previous != row.tobytes():
            raise RuntimeError(f"bootstrap cache assigned two behaviors to prompt {text!r}")
    meta = {
        "probe_sha256": str(z["probe_sha256"]),
        "executor_model": str(z["executor_model"]),
        "executor_model_revision": str(z["executor_model_revision"]),
        "readout_id": str(z["readout_id"]),
        "cache_namespace_sha256": str(z["cache_namespace_sha256"]),
        "artifact_sha256": file_sha256(path),
        "legacy_alignment": json.loads(str(z["legacy_alignment_json"])),
        "target_forms": (target_forms > 0.5).astype(np.uint8),
        "target_form_names": target_form_names,
        "target_form_texts": target_form_texts,
    }
    return binary, (target > 0.5).astype(np.uint8), meta


def load_codebook_bootstrap(path: Path, identity: dict, source_sha256: str) -> dict:
    """Validate a lightweight canonical-behavior artifact used only for MCQ design."""
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    if str(z["schema"]) != CODEBOOK_BOOTSTRAP_SCHEMA:
        raise RuntimeError(f"unexpected MCQ codebook bootstrap schema in {path}")
    if str(z["source_checkpoint_sha256"]) != source_sha256:
        raise RuntimeError(f"MCQ codebook bootstrap source hash mismatch in {path}")
    if str(z["metric_key"]) != identity["key"]:
        raise RuntimeError(f"MCQ codebook bootstrap metric key mismatch in {path}")
    if str(z["metric_description"]) != identity["description"]:
        raise RuntimeError(f"MCQ codebook bootstrap description mismatch in {path}")
    target = np.asarray(z["target"], float)
    target_forms = np.asarray(z["target_forms"], float)
    sigs = np.asarray(z["sigs"], float)
    texts = [str(value) for value in z["texts"]]
    probes = [str(value) for value in z["probe_texts"]]
    if (target.ndim != 1 or len(target) != len(probes)
            or target_forms.shape != (identity["target_orbit_forms"], len(target))
            or sigs.shape != (1, len(target)) or texts != [identity["description"]]
            or np.any(~np.isfinite(target)) or np.any(~np.isfinite(target_forms))
            or np.any(~np.isfinite(sigs))
            or not np.allclose(target, target_forms.mean(axis=0), rtol=0.0, atol=1e-12)
            or not np.allclose(sigs[0], target, rtol=0.0, atol=1e-12)):
        raise RuntimeError(f"invalid MCQ codebook bootstrap dimensions or values in {path}")
    required_text = {
        "probe_sha256", "executor_model", "executor_model_revision", "readout_id",
        "cache_namespace_sha256",
    }
    if not required_text.issubset(z.files) or any(not str(z[name]) for name in required_text):
        raise RuntimeError(f"MCQ codebook bootstrap lacks executor provenance in {path}")
    return {
        "path": path,
        "artifact_sha256": file_sha256(path),
        "probe_sha256": str(z["probe_sha256"]),
        "executor_model": str(z["executor_model"]),
        "executor_model_revision": str(z["executor_model_revision"]),
        "readout_id": str(z["readout_id"]),
        "cache_namespace_sha256": str(z["cache_namespace_sha256"]),
    }


def prepare_bootstraps(root: Path, manifest: dict, args) -> None:
    if args.reuse_bootstrap_root:
        source_root = Path(args.reuse_bootstrap_root).resolve()
        if source_root == root:
            raise RuntimeError("--reuse-bootstrap-root must differ from --out-root")
        for source_text in manifest["metrics"]:
            identity = manifest["metric_identity"][source_text]
            destination = root / identity["key"] / "bootstrap" / "scored.npz"
            if destination.exists():
                continue
            source = source_root / identity["key"] / "bootstrap" / "scored.npz"
            if not source.exists():
                raise RuntimeError(f"reusable bootstrap is missing: {source}")
            load_bootstrap(source, identity, manifest["metric_sha256"][source_text])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            _fsync_directory(destination.parent)
    items = []
    for source_text in manifest["metrics"]:
        source = Path(source_text)
        identity = manifest["metric_identity"][source_text]
        out = root / identity["key"] / "bootstrap" / "scored.npz"
        if out.exists():
            load_bootstrap(out, identity, manifest["metric_sha256"][source_text])
            continue
        items.append({
            "mode": "bootstrap",
            "task": identity["task"],
            "orig_npz": source_text,
            "metric_description": identity["description"],
            "target_orbit_forms": identity["target_orbit_forms"],
            "signature_cache_root": str(root / "signature_cache"),
            "out": str(out),
        })
    run_stage(
        stage="score",
        items=items,
        jobs_file=root / "jobs" / "bootstrap.json",
        model=args.executor,
        family="",
        temperature=0.0,
        args=args,
        worlds={},
    )
    for source_text in manifest["metrics"]:
        identity = manifest["metric_identity"][source_text]
        load_bootstrap(
            root / identity["key"] / "bootstrap" / "scored.npz",
            identity,
            manifest["metric_sha256"][source_text],
        )


def prepare_mcq_codebook_bootstraps(root: Path, manifest: dict, args) -> dict[str, list[Path]]:
    """Score one frozen canonical target behavior for every codebook candidate metric."""
    if args.value_mode != "reconstruction_mcq":
        return {}
    reuse_root = (Path(args.reuse_mcq_codebook_root).resolve()
                  if args.reuse_mcq_codebook_root else None)
    if reuse_root == root:
        raise RuntimeError("--reuse-mcq-codebook-root must differ from --out-root")
    items = []
    paths_by_task: dict[str, list[Path]] = {}
    seen_keys: dict[str, str] = {}
    for source_text in manifest["mcq_codebook_metrics"]:
        identity = manifest["mcq_codebook_metric_identity"][source_text]
        previous = seen_keys.setdefault(identity["key"], source_text)
        if previous != source_text:
            raise RuntimeError(
                f"MCQ codebook metric key collision: {identity['key']} maps to two checkpoints")
        out = root / "mcq_codebook_candidates" / identity["key"] / "bootstrap" / "scored.npz"
        paths_by_task.setdefault(identity["task"], []).append(out)
        source_sha = manifest["mcq_codebook_metric_sha256"][source_text]
        if out.exists():
            load_codebook_bootstrap(out, identity, source_sha)
            continue
        if reuse_root is not None:
            reusable = (reuse_root / "mcq_codebook_candidates" / identity["key"]
                        / "bootstrap" / "scored.npz")
            if not reusable.exists():
                raise RuntimeError(f"reusable MCQ codebook bootstrap is missing: {reusable}")
            load_codebook_bootstrap(reusable, identity, source_sha)
            out.parent.mkdir(parents=True, exist_ok=True)
            os.link(reusable, out)
            _fsync_directory(out.parent)
            continue
        items.append({
            "mode": "codebook_bootstrap",
            "task": identity["task"],
            "orig_npz": source_text,
            "metric_key": identity["key"],
            "metric_description": identity["description"],
            "target_orbit_forms": identity["target_orbit_forms"],
            "signature_cache_root": str(root / "signature_cache"),
            "out": str(out),
        })
    run_stage(
        stage="score",
        items=items,
        jobs_file=root / "jobs" / "mcq_codebook_bootstrap.json",
        model=args.executor,
        family="",
        temperature=0.0,
        args=args,
        worlds={},
    )
    for source_text in manifest["mcq_codebook_metrics"]:
        identity = manifest["mcq_codebook_metric_identity"][source_text]
        load_codebook_bootstrap(
            root / "mcq_codebook_candidates" / identity["key"] / "bootstrap" / "scored.npz",
            identity,
            manifest["mcq_codebook_metric_sha256"][source_text],
        )
    return paths_by_task


def prepare_mcq_codebooks(root: Path, states: dict[str, dict], args,
                          candidate_paths_by_task: dict[str, list[Path]]) -> None:
    if args.value_mode != "reconstruction_mcq":
        return
    from methods.metric_implementer.config import ImplementerConfig, apply_task_preset

    by_task: dict[str, list[dict]] = {}
    for state in states.values():
        by_task.setdefault(state["task"], []).append(state)
    planned = {}
    calibration_jobs = []
    for task, task_states in by_task.items():
        paths = sorted(candidate_paths_by_task.get(task, []), key=lambda path: str(path))
        if not paths:
            raise RuntimeError(f"no frozen MCQ codebook candidates were prepared for task {task}")
        plan = build_codebook_panel_plan(
            paths,
            target_metric_keys=[state["key"] for state in task_states],
            n_options=args.mcq_n_options,
            design_size=args.mcq_design_size,
            min_design_disagreements=args.mcq_min_design_disagreements,
            seed=stable_seed("mcq-codebook", task) % (2 ** 32),
            candidate_pool_size=args.mcq_prior_candidate_pool_size,
            max_panels_per_target=args.mcq_prior_max_panels_per_target,
        )
        plan_path = root / "mcq_codebooks" / f"{task}.panel_plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise RuntimeError(f"frozen MCQ panel plan changed for task {task}")
        else:
            _atomic_json(plan_path, plan)
        calibration_path = root / "mcq_codebooks" / f"{task}.prior_calibration.json"
        if not calibration_path.exists():
            cfg = ImplementerConfig()
            apply_task_preset(cfg, task)
            calibration_jobs.append({
                "panel_plan": str(plan_path),
                "noun": str(getattr(cfg, "item_noun", task)),
                "n_draws": args.mcq_reconstruction_draws,
                "query_batch_size": args.mcq_value_query_batch_size,
                "choice_probability_cache": str(
                    root / "mcq_query_cache" / "choice_probabilities.sqlite"),
                "out": str(calibration_path),
            })
        planned[task] = (task_states, paths, plan, calibration_path)

    run_stage(
        stage="codebook_prior",
        items=calibration_jobs,
        jobs_file=root / "jobs" / "codebook_prior.json",
        model=args.mcq_reconstructor,
        family="",
        temperature=0.0,
        args=args,
        worlds={},
    )

    for task, (task_states, paths, plan, calibration_path) in planned.items():
        if not calibration_path.exists():
            raise RuntimeError(f"missing MCQ prior calibration for task {task}")
        calibration = json.loads(calibration_path.read_text())
        selections = select_prior_balanced_panels(
            plan,
            calibration,
            maximum_option_probability=args.mcq_prior_max_option_probability,
            target_probability_tolerance=args.mcq_prior_target_probability_tolerance,
            minimum_normalized_entropy=args.mcq_prior_min_normalized_entropy,
        )
        expected = build_frozen_codebook_manifest(
            paths,
            n_options=args.mcq_n_options,
            design_size=args.mcq_design_size,
            min_design_disagreements=args.mcq_min_design_disagreements,
            seed=stable_seed("mcq-codebook", task) % (2 ** 32),
            panel_selections=selections,
        )
        path = root / "mcq_codebooks" / f"{task}.json"
        if path.exists():
            observed = json.loads(path.read_text())
            validate_codebook_manifest(observed)
            if observed["manifest_sha256"] != expected["manifest_sha256"]:
                raise RuntimeError(f"frozen MCQ codebook changed for task {task}")
        else:
            _atomic_json(path, expected)
        for state in task_states:
            entry = expected["entries"].get(state["key"])
            if not entry or not entry["valid"]:
                raise RuntimeError(
                    f"metric {state['key']} lacks a valid frozen MCQ codebook: "
                    f"{(entry or {}).get('failure')}")
            state["mcq_codebook_path"] = path
            state["mcq_codebook_sha256"] = expected["manifest_sha256"]
            state["mcq_codebook_entry"] = entry


def value_all(keys: list[str], states: dict[str, dict], scored: dict[str, Path], *,
              phase: str, iteration: int, args, worlds: dict[str, dict]) -> dict[str, dict]:
    if args.value_mode != "reconstruction_mcq":
        return {}
    outputs: dict[str, Path] = {}
    missing = []
    for key in keys:
        state = states[key]
        output = scored[key].with_name("values.npz")
        outputs[key] = output
        if output.exists():
            continue
        from methods.metric_implementer.config import ImplementerConfig, apply_task_preset
        cfg = ImplementerConfig()
        apply_task_preset(cfg, state["task"])
        item = {
            "codebook_manifest": str(state["mcq_codebook_path"]),
            "target_metric_key": key,
            "scored": str(scored[key]),
            "noun": str(getattr(cfg, "item_noun", state["task"])),
            "n_examples": args.mcq_n_examples,
            "n_reconstruction_draws": args.mcq_reconstruction_draws,
            "max_chars": args.mcq_max_chars,
            "choice_readout": args.mcq_choice_readout,
            "query_batch_size": args.mcq_value_query_batch_size,
            "choice_probability_cache": str(
                Path(args.out_root) / "mcq_query_cache" / "choice_probabilities.sqlite"),
            "out": str(output),
        }
        if state.get("fixed_no_demo_canonical_choice_probabilities") is not None:
            item["fixed_no_demo_canonical_choice_probabilities"] = np.asarray(
                state["fixed_no_demo_canonical_choice_probabilities"], float).tolist()
        missing.append(item)
    run_stage(
        stage="value",
        items=missing,
        jobs_file=Path(args.out_root) / "jobs" / f"value_{phase}_{iteration:03d}.json",
        model=args.mcq_reconstructor,
        family="",
        temperature=0.0,
        args=args,
        worlds=worlds,
    )
    loaded = {}
    for key, output in outputs.items():
        payload = load_value_artifact(
            output,
            expected_source_scored_sha256=file_sha256(scored[key]),
            expected_codebook_manifest_sha256=states[key]["mcq_codebook_sha256"],
        )
        if payload["target_metric_key"] != key:
            raise RuntimeError(f"value artifact target mismatch in {output}")
        loaded[key] = payload
    return loaded


def attach_mcq_pool_values(root: Path, states: dict[str, dict], args,
                           worlds: dict[str, dict]) -> None:
    if args.value_mode != "reconstruction_mcq":
        return
    scored = {key: state["bootstrap_path"] for key, state in states.items()}
    bootstrap_values = value_all(
        list(states), states, scored, phase="bootstrap", iteration=0, args=args, worlds=worlds)
    for key, state in states.items():
        payload = bootstrap_values[key]
        state["value_name"] = payload["value_name"]
        state["value_unit"] = payload["value_unit"]
        state["value_cap"] = float(payload["value_cap"])
        state["no_demonstration_target_probability"] = float(
            payload["no_demonstration_target_probability"])
        state["fixed_no_demo_canonical_choice_probabilities"] = np.asarray(
            payload["fixed_no_demo_canonical_choice_probabilities"], float)

    historical_scored = {
        key: state["historical_scored_path"]
        for key, state in states.items() if "historical_scored_path" in state
    }
    historical_values = value_all(
        list(historical_scored), states, historical_scored,
        phase="historical", iteration=0, args=args, worlds=worlds,
    ) if historical_scored else {}

    for key, state in states.items():
        payload = bootstrap_values[key]
        values = [payload["values"]]
        details = list(payload["details"])
        value_cap = state["value_cap"]
        no_demo = state["no_demonstration_target_probability"]
        fixed_no_demo = state["fixed_no_demo_canonical_choice_probabilities"]
        value_determined_by_exact_behavior = bool(
            payload["premises"].get("value_determined_by_exact_behavior"))
        if key in historical_values:
            historical = historical_values[key]
            if (not np.isclose(historical["value_cap"], value_cap, rtol=0.0, atol=1e-12)
                    or not np.array_equal(
                        historical["fixed_no_demo_canonical_choice_probabilities"],
                        fixed_no_demo)):
                raise RuntimeError(f"frozen MCQ control changed in historical values for {key}")
            values.append(historical["values"])
            details.extend(historical["details"])
            value_determined_by_exact_behavior = (
                value_determined_by_exact_behavior
                and bool(historical["premises"].get("value_determined_by_exact_behavior")))
        for row in _read_jsonl(state["dir"] / "absorption_ledger.jsonl"):
            value_path_text = row.get("value_path")
            if not value_path_text:
                raise RuntimeError(f"absorbed MCQ ledger row lacks value artifact for {key}")
            value_path = root / value_path_text
            absorbed = load_value_artifact(
                value_path,
                expected_source_scored_sha256=row["scored_sha256"],
                expected_codebook_manifest_sha256=state["mcq_codebook_sha256"],
            )
            if file_sha256(value_path) != row.get("value_sha256"):
                raise RuntimeError(f"absorbed value artifact hash mismatch: {value_path}")
            if (not np.isclose(absorbed["value_cap"], value_cap, rtol=0.0, atol=1e-12)
                    or not np.isclose(
                        absorbed["no_demonstration_target_probability"], no_demo,
                        rtol=0.0, atol=1e-12)
                    or not np.array_equal(
                        absorbed["fixed_no_demo_canonical_choice_probabilities"],
                        fixed_no_demo)):
                raise RuntimeError(f"frozen MCQ control changed in absorbed values for {key}")
            value_determined_by_exact_behavior = (
                value_determined_by_exact_behavior
                and bool(absorbed["premises"].get("value_determined_by_exact_behavior")))
            values.append(absorbed["values"])
            details.extend(absorbed["details"])
        state["pool_values"] = np.concatenate(values)
        if len(state["pool_values"]) != len(state["pool"]):
            raise RuntimeError(f"prompt/value pool length mismatch for {key}")
        if len(details) != len(state["pool"]):
            raise RuntimeError(f"prompt/value-detail pool length mismatch for {key}")
        state["pool_value_details"] = details
        state["pool_value_species"] = [
            detail["design"]["teaching_transcript_sha256"] for detail in details
        ]
        state["value_determined_by_exact_behavior"] = value_determined_by_exact_behavior
        confirmation_path = state["dir"] / "confirmation" / "certificate.json"
        if confirmation_path.exists():
            certificate = json.loads(confirmation_path.read_text())
            run = certificate.get("run") or {}
            value_path_text = run.get("confirmation_value_path")
            if not value_path_text:
                raise RuntimeError(f"MCQ confirmation lacks its value artifact for {key}")
            value_path = root / value_path_text
            loaded = load_value_artifact(
                value_path,
                expected_source_scored_sha256=run["confirmation_scored_sha256"],
                expected_codebook_manifest_sha256=state["mcq_codebook_sha256"],
            )
            if loaded["sha256"] != run.get("confirmation_value_sha256"):
                raise RuntimeError(f"MCQ confirmation value hash mismatch for {key}")


def _load_historical_import(root: Path, state: dict, manifest: dict) -> np.ndarray | None:
    """Load candidate-only historical prompts before replaying adaptive absorptions."""
    path = state["dir"] / "historical" / "import.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record.get("schema") != "cr3-historical-import-v1"
            or record.get("evidence_manifest_sha256")
            != manifest.get("reuse_evidence_manifest_sha256")
            or record.get("metric_key") != state["key"]
            or record.get("evidence_role") != "candidate_only"
            or record.get("eligible_as_fresh_audit") is not False):
        raise RuntimeError(f"invalid historical import contract for {state['key']}")
    candidate_path = root / record["candidate_path"]
    scored_path = root / record["scored_path"]
    if (file_sha256(candidate_path) != record["candidate_sha256"]
            or file_sha256(scored_path) != record["scored_sha256"]):
        raise RuntimeError(f"historical evidence changed for {state['key']}")
    batch, _, meta = load_scored(
        scored_path, ["historical_candidate"], int(record["n_candidates"]))
    for field in ("probe_sha256", "executor_model_revision", "readout_id",
                  "cache_namespace_sha256"):
        if meta[field] != state[field]:
            raise RuntimeError(f"historical {field} mismatch for {state['key']}")
    state["historical_import"] = record
    state["historical_scored_path"] = scored_path
    return batch


def prepare_historical_candidates(
    root: Path,
    states: dict[str, dict],
    manifest: dict,
    args,
    worlds: dict[str, dict],
) -> None:
    """Re-score reusable prompts as frozen-pool candidates, never as audit draws."""
    if not args.reuse_evidence_root:
        return
    evidence_root = Path(args.reuse_evidence_root).resolve()
    evidence = load_evidence_manifest(evidence_root)
    if evidence["manifest_sha256"] != manifest["reuse_evidence_manifest_sha256"]:
        raise RuntimeError("the configured evidence store changed after manifest freeze")
    jobs = []
    pending: dict[str, tuple[Path, Path, Path, int]] = {}
    for key, state in states.items():
        entry = evidence["metrics"].get(key)
        if entry is None:
            continue
        destination = state["dir"] / "historical" / "candidates.jsonl"
        source = evidence_root / entry["candidate_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
            with destination.open("rb") as handle:
                os.fsync(handle.fileno())
            _fsync_directory(destination.parent)
        if file_sha256(destination) != entry["candidate_sha256"]:
            raise RuntimeError(f"historical candidate copy changed for {key}")
        rows = _read_jsonl(destination)
        n_candidates = int(entry["n_unique_candidates"])
        if (len(rows) != n_candidates or n_candidates < 1
                or len({int(row["seed"]) for row in rows}) != n_candidates
                or any(row.get("evidence_role") != "candidate_only"
                       or row.get("eligible_as_fresh_audit") is not False
                       or row.get("family") != "historical_candidate" for row in rows)):
            raise RuntimeError(f"invalid candidate-only evidence rows for {key}")
        scored = state["dir"] / "historical" / "scored.npz"
        import_path = state["dir"] / "historical" / "import.json"
        pending[key] = (destination, scored, import_path, n_candidates)
        if not scored.exists():
            jobs.append({
                "mode": "audit",
                "task": state["task"],
                "criteria": [str(destination)],
                "family_names": ["historical_candidate"],
                "expected_per_family": n_candidates,
                "signature_cache_root": str(root / "signature_cache"),
                "expected_probe_sha256": state["probe_sha256"],
                "expected_executor_model_revision": state["executor_model_revision"],
                "expected_readout_id": state["readout_id"],
                "expected_cache_namespace_sha256": state["cache_namespace_sha256"],
                "out": str(scored),
                "out_key": key,
            })
    run_stage(
        stage="score",
        items=jobs,
        jobs_file=root / "jobs" / "score_historical.json",
        model=args.executor,
        family="",
        temperature=0.0,
        args=args,
        worlds=worlds,
    )
    for key, (candidate, scored, import_path, n_candidates) in pending.items():
        state = states[key]
        _, _, meta = load_scored(scored, ["historical_candidate"], n_candidates)
        for field in ("probe_sha256", "executor_model_revision", "readout_id",
                      "cache_namespace_sha256"):
            if meta[field] != state[field]:
                raise RuntimeError(f"historical {field} mismatch for {key}")
        record = {
            "schema": "cr3-historical-import-v1",
            "metric_key": key,
            "evidence_manifest_sha256": evidence["manifest_sha256"],
            "candidate_path": str(candidate.relative_to(root)),
            "candidate_sha256": file_sha256(candidate),
            "scored_path": str(scored.relative_to(root)),
            "scored_sha256": file_sha256(scored),
            "n_candidates": n_candidates,
            "pool_position": "after bootstrap and before adaptive absorption ledger",
            "evidence_role": "candidate_only",
            "eligible_as_fresh_audit": False,
            "may_raise_achieved_lower_bound": True,
        }
        if import_path.exists():
            if json.loads(import_path.read_text()) != record:
                raise RuntimeError(f"historical import record changed for {key}")
        else:
            _atomic_json(import_path, record)
        if "historical_import" not in state:
            batch = _load_historical_import(root, state, manifest)
            state["pool"] = np.vstack([state["pool"], batch])


def _load_metric(path: str, root: Path, dry_run: bool, manifest: dict) -> tuple[dict, dict | None]:
    source = Path(path).resolve()
    source_text = str(source)
    identity = manifest["metric_identity"][source_text]
    key = identity["key"]
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True)
    bootstrap_path = directory / "bootstrap" / "scored.npz"
    pool, target, bootstrap_meta = load_bootstrap(
        bootstrap_path, identity, manifest["metric_sha256"][source_text])
    if bootstrap_meta["executor_model"] != manifest["executor"]:
        raise RuntimeError(f"bootstrap executor mismatch for {key}")
    state = {
        "key": key,
        "task": identity["task"],
        "name": identity["name"],
        "description": identity["description"],
        "orig": str(source),
        "pool": pool,
        "target": target,
        "bootstrap_path": bootstrap_path,
        "bootstrap_sha256": bootstrap_meta["artifact_sha256"],
        "probe_sha256": bootstrap_meta["probe_sha256"],
        "executor_model_revision": bootstrap_meta["executor_model_revision"],
        "readout_id": bootstrap_meta["readout_id"],
        "cache_namespace_sha256": bootstrap_meta["cache_namespace_sha256"],
        "legacy_alignment": bootstrap_meta["legacy_alignment"],
        "target_forms": bootstrap_meta["target_forms"],
        "target_form_names": bootstrap_meta["target_form_names"],
        "target_form_texts": bootstrap_meta["target_form_texts"],
        "dir": directory,
        "iteration": 0,
        "best_u0": 1.0,
        "best_gap": float("inf"),
        "stall": 0,
        "stopped": None,
        "confirmed": False,
    }
    historical = _load_historical_import(root, state, manifest)
    if historical is not None:
        state["pool"] = np.vstack([state["pool"], historical])
    for expected_iter, row in enumerate(_read_jsonl(directory / "absorption_ledger.jsonl")):
        if row.get("schema") != LEDGER_SCHEMA or row.get("event") != "absorb":
            raise RuntimeError(f"invalid ledger row for {key}: {row}")
        if int(row["iteration"]) != expected_iter:
            raise RuntimeError(f"non-contiguous ledger for {key}")
        if (row.get("bootstrap_sha256") != state["bootstrap_sha256"]
                or row.get("probe_sha256") != state["probe_sha256"]
                or row.get("cache_namespace_sha256") != state["cache_namespace_sha256"]):
            raise RuntimeError(f"ledger provenance mismatch for {key} at iteration {expected_iter}")
        scored = root / row["scored_path"]
        if file_sha256(scored) != row["scored_sha256"]:
            raise RuntimeError(f"scored artifact hash mismatch: {scored}")
        batch, _, meta = load_scored(scored, row["family_names"], int(row["expected_per_family"]))
        for field in ("probe_sha256", "executor_model_revision", "readout_id", "cache_namespace_sha256"):
            if meta[field] != state[field]:
                raise RuntimeError(f"{field} changed in absorbed artifact {scored}")
        if int(row["pool_n_before"]) != len(state["pool"]):
            raise RuntimeError(f"pool length mismatch before ledger row {expected_iter}")
        state["pool"] = np.vstack([state["pool"], batch])
        state["iteration"] = expected_iter + 1
        state["best_u0"] = min(state["best_u0"], float(row["monitor"]["behavioral_U0"]))
        state["best_gap"] = min(state["best_gap"], float(row["monitor"]["horizon_gain_UCB_bits"]))
        state["stall"] = int(row["stall"])
        state["stopped"] = row.get("stopped")
    certificate_path = directory / "confirmation" / "certificate.json"
    if certificate_path.exists():
        certificate = json.loads(certificate_path.read_text())
        run = certificate.get("run", {})
        if certificate.get("schema") != SCHEMA_VERSION or not run.get("never_absorbed"):
            raise RuntimeError(f"invalid confirmation certificate {certificate_path}")
        scored = root / run["confirmation_scored_path"]
        if file_sha256(scored) != run["confirmation_scored_sha256"]:
            raise RuntimeError(f"confirmation hash mismatch for {certificate_path}")
        if not state["stopped"]:
            raise RuntimeError(f"confirmation exists before a stopping ledger event for {key}")
        state["confirmed"] = True
    world = None
    if dry_run:
        rng = np.random.default_rng(stable_seed(manifest["run_id"], key, "world"))
        n_species = 400
        probs = 1.0 / np.arange(1, n_species + 1) ** 1.4
        probs /= probs.sum()
        world = {"p": probs, "cols": (rng.random((n_species, len(target))) < 0.5).astype(float)}
    return state, world


def _proposal_path(state: dict, phase: str, iteration: int, family: str) -> Path:
    base = state["dir"] / phase / f"iter_{iteration:03d}"
    return base / f"proposal_{family}.jsonl"


def propose_all(keys: list[str], states: dict[str, dict], *, phase: str, iteration: int,
                n_per_family: int, manifest: dict, args, worlds: dict[str, dict]) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {key: [] for key in keys}
    jobs_dir = Path(args.out_root) / "jobs"
    for family, model, proposal_mode in zip(
            args.family_tags, args.families, args.family_modes):
        missing = []
        for key in keys:
            state = states[key]
            out = _proposal_path(state, phase, iteration, family)
            if out.exists():
                _validate_proposal(out, family, n_per_family, proposal_mode)
            else:
                missing.append({
                    "metric_name": state["name"],
                    "metric_description": state["description"],
                    "n": n_per_family,
                    "proposal_mode": proposal_mode,
                    "base_seed": stable_seed(manifest["run_id"], key, phase, iteration, family),
                    "out": str(out),
                })
            outputs[key].append(str(out))
        run_stage(
            stage="propose",
            items=missing,
            jobs_file=jobs_dir / f"propose_{phase}_{iteration:03d}_{family}.json",
            model=model,
            family=family,
            temperature=args.temp,
            args=args,
            worlds=worlds,
        )
        for key in keys:
            out = _proposal_path(states[key], phase, iteration, family)
            _validate_proposal(out, family, n_per_family, proposal_mode)
    return outputs


def score_all(keys: list[str], states: dict[str, dict], proposals: dict[str, list[str]], *,
              phase: str, iteration: int, expected_per_family: int, args,
              worlds: dict[str, dict]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    missing = []
    for key in keys:
        state = states[key]
        out = state["dir"] / phase / f"iter_{iteration:03d}" / "scored.npz"
        outputs[key] = out
        if not out.exists():
            missing.append({
                "mode": "audit",
                "task": state["task"],
                "criteria": proposals[key],
                "family_names": list(args.family_tags),
                "expected_per_family": expected_per_family,
                "signature_cache_root": str(Path(args.out_root) / "signature_cache"),
                "expected_probe_sha256": state["probe_sha256"],
                "expected_executor_model_revision": state["executor_model_revision"],
                "expected_readout_id": state["readout_id"],
                "expected_cache_namespace_sha256": state["cache_namespace_sha256"],
                "out": str(out),
                "out_key": key,
            })
    run_stage(
        stage="score",
        items=missing,
        jobs_file=Path(args.out_root) / "jobs" / f"score_{phase}_{iteration:03d}.json",
        model=args.executor,
        family="",
        temperature=0.0,
        args=args,
        worlds=worlds,
    )
    for key, path in outputs.items():
        _, _, meta = load_scored(path, list(args.family_tags), expected_per_family)
        state = states[key]
        for field in ("probe_sha256", "executor_model_revision", "readout_id", "cache_namespace_sha256"):
            if meta[field] != state[field]:
                raise RuntimeError(f"{field} changed in {path}")
    return outputs


def _certificate(state: dict, scored: Path, expected_per_family: int, manifest: dict, args,
                 *, alpha_override: float | None = None,
                 certificate_role: str = "monitor",
                 value_payload: dict | None = None) -> tuple[dict, np.ndarray]:
    audit, families, meta = load_scored(scored, list(args.family_tags), expected_per_family)
    common = dict(
        family_names=args.family_tags,
        horizon_per_family=args.ceiling_horizon_per_family,
        tau=args.tau,
        tau_strict=args.tau_strict,
        alpha=(args.alpha if alpha_override is None else alpha_override),
        p_min=args.p_min,
        value_p_min=args.value_p_min,
        scope={
            "run_id": manifest["run_id"],
            "metric": state["name"],
            "metric_key": state["key"],
            "executor": args.executor,
            "executor_model_revision": meta["executor_model_revision"],
            "probe_sha256": meta["probe_sha256"],
            "readout_id": meta["readout_id"],
            "cache_namespace_sha256": meta["cache_namespace_sha256"],
            "bootstrap_artifact_sha256": state["bootstrap_sha256"],
            "proposer_models": dict(zip(args.family_tags, args.families)),
            "proposer_modes": dict(zip(args.family_tags, args.family_modes)),
            "proposal_temperature": args.temp,
            "proposal_template": "family-mode-specific; see proposer_modes and run manifest",
            "scored_artifact_sha256": meta["artifact_sha256"],
            "iid_provenance_established": not args.dry_run,
            "prompt_class": (
                "frozen discovery pool (bootstrap, candidate-only historical imports, and "
                "absorbed monitors) union declared proposer-mixture support"),
            "historical_evidence_manifest_sha256": manifest.get(
                "reuse_evidence_manifest_sha256"),
            "n_candidate_only_historical_prompts": int(
                (state.get("historical_import") or {}).get("n_candidates", 0)),
            "certificate_role": certificate_role,
        },
    )
    if args.value_mode == "reconstruction_mcq":
        if value_payload is None or len(value_payload["values"]) != len(audit):
            raise RuntimeError("Reconstruction-MCQ certificate requires one value mark per audit row")
        if (not np.isclose(value_payload["value_cap"], state["value_cap"],
                           rtol=0.0, atol=1e-12)
                or not np.isclose(
                    value_payload["no_demonstration_target_probability"],
                    state["no_demonstration_target_probability"],
                    rtol=0.0, atol=1e-12)
                or not np.array_equal(
                    value_payload["fixed_no_demo_canonical_choice_probabilities"],
                    state["fixed_no_demo_canonical_choice_probabilities"])):
            raise RuntimeError("Reconstruction-MCQ audit changed the frozen no-demo global cap")
        value_determined_by_exact_behavior = (
            bool(state["value_determined_by_exact_behavior"])
            and bool(value_payload["premises"].get("value_determined_by_exact_behavior")))
        cert = prompt_articulation_certificate(
            state["pool"], audit, None, families,
            pool_values=state["pool_values"],
            audit_values=value_payload["values"],
            value_cap=state["value_cap"],
            value_name=state["value_name"],
            value_unit=state["value_unit"],
            value_determined_by_exact_behavior=value_determined_by_exact_behavior,
            pool_value_species=state["pool_value_species"],
            audit_value_species=[
                detail["design"]["teaching_transcript_sha256"]
                for detail in value_payload["details"]
            ],
            **common,
        )
        pool_best = float(cert["certified"]["pool_best_prompt_value"])
        audit_best = float(np.max(value_payload["values"]))
        best = max(pool_best, audit_best)
        gap = float(max(0.0, state["value_cap"] - best))
        epsilon = target_value_gap(args)
        if gap <= 1e-12:
            global_status = "CERTIFIED_GLOBAL_OPTIMUM"
        elif gap <= epsilon:
            global_status = "CERTIFIED_EPSILON_GLOBAL_OPTIMUM"
        else:
            global_status = "CERTIFIED_GLOBAL_GAP_BOUND"
        cert["all_finite_prompt_certificate"] = {
            "schema": "reconstruction-mcq-all-prompts-cap-v3",
            "status": global_status,
            "prompt_class": "all finite prompts Sigma*; no prompt-length budget",
            "best_evaluated_lower_bound": best,
            "absorbed_pool_best_lower_bound": pool_best,
            "current_audit_best_lower_bound": audit_best,
            "current_audit_role": certificate_role,
            "anchor_free_global_upper_bound": state["value_cap"],
            "global_optimization_gap_UCB": gap,
            "epsilon": epsilon,
            "identified_interval": [best, state["value_cap"]],
            "no_demonstration_target_probability": state[
                "no_demonstration_target_probability"],
            "bound_identity": "V_ann(p) <= 1 - q_no_demo(b) for every finite prompt p",
            "note": (
                "the frozen no-demonstration control is independent of candidate prompts; "
                "without additional executor structure, finite black-box observations cannot "
                "lower this all-strings cap further. CR-3 separately tightens the declared "
                "proposer-process horizon/support scope"),
            "uses_external_labels": False,
            "instrument_quality": mcq_instrument_quality(state, args),
        }
        return cert, audit

    cert = prompt_articulation_certificate(
        state["pool"], audit, state["target"], families, **common)
    # The CR-3 pool is intentionally atomic/process-relative.  Evaluate the known
    # target-form prompts separately because the unrestricted all-finite-prompt
    # problem admits them even though the atomic proposer does not.
    global_candidates = np.vstack([state["pool"], state["target_forms"]])
    global_labels = ([f"mined_pool_{i}" for i in range(len(state["pool"]))]
                     + [f"target_form:{name}" for name in state["target_form_names"]])
    one_form_identity = len(state["target_forms"]) == 1
    cert["all_finite_prompt_certificate"] = all_finite_prompt_dpi_certificate(
        global_candidates,
        state["target"],
        candidate_labels=global_labels,
        identity_witness_index=(len(state["pool"]) if one_form_identity else None),
        identity_witness_is_target_definition=(one_form_identity and not args.dry_run),
        epsilon_bits=target_value_gap(args),
        scope={
            "metric": state["name"],
            "metric_key": state["key"],
            "executor": args.executor,
            "executor_model_revision": meta["executor_model_revision"],
            "probe_sha256": meta["probe_sha256"],
            "readout_id": meta["readout_id"],
            "target_orbit_forms": len(state["target_forms"]),
            "target_form_text_sha256": [hashlib.sha256(text.encode()).hexdigest()
                                        for text in state["target_form_texts"]],
        },
    )
    return cert, audit


def write_checkpoint_certificates(
    keys: list[str],
    states: dict[str, dict],
    *,
    iteration: int,
    manifest: dict,
    args,
    worlds: dict[str, dict],
    cell_alpha: float,
    alpha_scope: dict,
) -> None:
    """Create immutable, never-absorbed certificates at a predeclared pool size."""
    if not keys:
        return
    print(f"=== immutable checkpoint audit after {iteration} absorbed iterations: "
          f"{len(keys)} metrics ===", flush=True)
    proposals = propose_all(
        keys, states, phase="checkpoint", iteration=iteration,
        n_per_family=args.checkpoint_per_family, manifest=manifest, args=args, worlds=worlds)
    scored = score_all(
        keys, states, proposals, phase="checkpoint", iteration=iteration,
        expected_per_family=args.checkpoint_per_family, args=args, worlds=worlds)
    values = value_all(
        keys, states, scored, phase="checkpoint", iteration=iteration, args=args, worlds=worlds)
    for key in keys:
        state = states[key]
        cert, _ = _certificate(
            state, scored[key], args.checkpoint_per_family, manifest, args,
            alpha_override=cell_alpha, certificate_role="checkpoint",
            value_payload=values.get(key))
        cert["run"] = {
            "phase": "checkpoint",
            "iterations_absorbed": int(iteration),
            "pool_size": int(len(state["pool"])),
            "checkpoint_scored_path": str(scored[key].relative_to(Path(args.out_root))),
            "checkpoint_scored_sha256": file_sha256(scored[key]),
            "never_absorbed": True,
            "alpha_scope": alpha_scope,
        }
        if key in values:
            cert["run"]["checkpoint_value_path"] = str(
                Path(values[key]["path"]).relative_to(Path(args.out_root)))
            cert["run"]["checkpoint_value_sha256"] = values[key]["sha256"]
        cert["prompt_evolution_status"] = classify_prompt_evolution(
            cert,
            confirmation_is_never_absorbed=True,
            stopping_rule_frozen_before_confirmation=True,
            plateau_epsilon=target_value_gap(args),
            saturation_missing_mass=args.target_u0,
        )
        path = state["dir"] / "checkpoint" / f"iter_{iteration:03d}" / "certificate.json"
        _atomic_json(path, cert)
        status = cert["prompt_evolution_status"]
        print(f"  {key}: {status['headline_status']} "
              f"mass={status['evidence']['behavioral_missing_mass_interval']} "
              f"gain={status['evidence']['finite_horizon_expected_best_gain_interval_bits']}",
              flush=True)


def write_certified_trajectories(states: dict[str, dict], args, alpha_scope: dict) -> None:
    """Collect immutable checkpoint/final certificates into one report per metric."""
    for key, state in states.items():
        points = []
        for iteration in checkpoint_iterations(args):
            path = state["dir"] / "checkpoint" / f"iter_{iteration:03d}" / "certificate.json"
            if not path.exists():
                continue
            cert = json.loads(path.read_text())
            points.append({
                "phase": "checkpoint",
                "iterations_absorbed": iteration,
                "pool_size": cert["run"]["pool_size"],
                "status": cert["prompt_evolution_status"],
                "prompt_ceiling_UCB": cert["certified"][
                    "finite_horizon_expected_prompt_ceiling_UCB"],
            })
        final_path = state["dir"] / "confirmation" / "certificate.json"
        if final_path.exists():
            cert = json.loads(final_path.read_text())
            points.append({
                "phase": "final_confirmation",
                "iterations_absorbed": cert["run"]["iterations_absorbed"],
                "pool_size": cert["run"]["final_pool_size"],
                "status": cert["prompt_evolution_status"],
                "prompt_ceiling_UCB": cert["certified"][
                    "finite_horizon_expected_prompt_ceiling_UCB"],
            })
        payload = {
            "schema": "cr3-certified-tightening-trajectory-v1",
            "metric_key": key,
            "executor": args.executor,
            "axis": "prompt evolution at fixed executor",
            "value_unit": state.get("value_unit", "bits"),
            "alpha_scope": alpha_scope,
            "points": sorted(points, key=lambda point: point["iterations_absorbed"]),
            "monitor_rows_are_certificates": False,
            "does_not_cover": ["all finite prompts", "OSL executor scaling", "external validity"],
        }
        _atomic_json(state["dir"] / "certified_trajectory.json", payload)


def write_mcq_bank_identity_summary(root: Path, states: dict[str, dict], args) -> None:
    """Report achieved bank-level identity MI for the best pool prompt per metric."""
    if args.value_mode != "reconstruction_mcq":
        return
    by_task: dict[str, list[dict]] = {}
    selected = {}
    for key, state in states.items():
        best_index = int(np.argmax(state["pool_values"]))
        detail = state["pool_value_details"][best_index]
        by_task.setdefault(state["task"], []).append(detail)
        selected[key] = {
            "pool_index": best_index,
            "value": float(state["pool_values"][best_index]),
            "candidate_prompt_sha256": detail["candidate_prompt_sha256"],
            "teaching_transcript_sha256": detail["design"]["teaching_transcript_sha256"],
        }

    tasks = {}
    for task, rows in sorted(by_task.items()):
        channels = {
            condition: mcq_identity_channel(rows, condition=condition)
            for condition in ("annotations", "no_demonstrations", "shuffled_labels")
        }
        valid = [channels[name] for name in channels if channels[name].get("valid")]
        annotation = channels["annotations"]
        control_mi = max(
            (channels[name].get("mutual_information_bits", 0.0)
             for name in ("no_demonstrations", "shuffled_labels")),
            default=0.0,
        )
        tasks[task] = {
            "channels": channels,
            "annotation_attributable_identity_mi_lift_bits": (
                max(0.0, float(annotation["mutual_information_bits"]) - control_mi)
                if annotation.get("valid") else None),
            "target_entropy_cap_bits": (
                float(annotation["target_entropy_bits"])
                if annotation.get("valid") else None),
            "n_valid_channels": len(valid),
        }
    payload = {
        "schema": "cr3-reconstruction-bank-identity-v1",
        "selection": "independent per-metric argmax of anchor-free V_ann in the final absorbed pool",
        "tasks": tasks,
        "selected_prompts": selected,
        "uses_external_labels": False,
        "scope": (
            "achieved closed-codebook I(J;Jhat) on the frozen reconstruction instrument; "
            "not an upper bound, not external validity, and not an OSL result"),
    }
    _atomic_json(root / "mcq_identity_final.json", payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--families", nargs="+", default=[
        "microsoft/phi-4", "Qwen/Qwen2.5-14B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"])
    parser.add_argument("--family-tags", nargs="+", default=["phi4", "qwen14", "llama8"])
    parser.add_argument("--family-modes", nargs="+", default=None,
                        choices=["atomic", "holistic"])
    parser.add_argument("--executor", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--value-mode", default="legacy_fixed_target",
                        choices=["legacy_fixed_target", "reconstruction_mcq"])
    parser.add_argument("--mcq-reconstructor", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--mcq-n-options", type=int, default=4)
    parser.add_argument("--mcq-design-size", type=int, default=120)
    parser.add_argument("--mcq-min-design-disagreements", type=int, default=2)
    parser.add_argument("--mcq-n-examples", type=int, default=8)
    parser.add_argument("--mcq-reconstruction-draws", type=int, default=4)
    parser.add_argument("--mcq-choice-readout", default="auto",
                        choices=["auto", "logits", "sampled"])
    parser.add_argument("--mcq-value-query-batch-size", type=int, default=512)
    parser.add_argument("--mcq-max-chars", type=int, default=600)
    parser.add_argument("--mcq-min-headline-value-cap", type=float, default=0.10)
    parser.add_argument("--mcq-min-headline-distractor-kappa", type=float, default=0.50)
    parser.add_argument("--mcq-prior-candidate-pool-size", type=int, default=16)
    parser.add_argument("--mcq-prior-max-panels-per-target", type=int, default=256)
    parser.add_argument("--mcq-prior-max-option-probability", type=float, default=0.35)
    parser.add_argument("--mcq-prior-target-probability-tolerance", type=float, default=0.10)
    parser.add_argument("--mcq-prior-min-normalized-entropy", type=float, default=0.90)
    parser.add_argument(
        "--mcq-codebook-metrics", nargs="+", default=None,
        help=("frozen task-level candidate bank for hard MCQ distractors; targets are added "
              "automatically and only canonical candidate behaviors are scored"),
    )
    parser.add_argument("--batch-per-family", type=int, default=150)
    parser.add_argument("--confirm-per-family", type=int, default=100)
    parser.add_argument("--checkpoint-per-family", type=int, default=100)
    parser.add_argument("--checkpoint-iters", default="",
                        help="comma-separated absorbed-iteration counts for immutable trajectory audits")
    parser.add_argument("--ceiling-horizon-per-family", type=int, default=100)
    parser.add_argument("--target-u0", type=float, default=0.10)
    parser.add_argument("--target-gap-bits", type=float, default=0.02)
    parser.add_argument("--target-value-gap", type=float, default=None,
                        help="unit-aware plateau tolerance; defaults to --target-gap-bits")
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--min-delta-bits", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--study-alpha", type=float, default=None,
                        help="optional familywise alpha across every metric and checkpoint/final claim")
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--tau-strict", type=float, default=0.95)
    parser.add_argument("--p-min", type=float, default=None)
    parser.add_argument("--value-p-min", type=float, default=None)
    parser.add_argument("--reuse-bootstrap-root", default=None,
                        help="verified immutable CR3 root whose matching bootstraps may be hard-linked")
    parser.add_argument(
        "--reuse-mcq-codebook-root", default=None,
        help="verified immutable CR3 root whose canonical MCQ candidate bootstraps may be hard-linked",
    )
    parser.add_argument(
        "--reuse-evidence-root", default=None,
        help=("validated CR3 evidence store; imported prompts are candidate-only and "
              "never confirmation observations"),
    )
    parser.add_argument("--r2-bucket", default="general")
    parser.add_argument("--temp", type=float, default=0.90)
    parser.add_argument("--out-root", default="/lfs/skampere3/0/alexspan/outputs/cr3_mining_v2")
    parser.add_argument("--worker", default=str(DEFAULT_WORKER))
    parser.add_argument("--worker-python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--worker-home", default=str(DEFAULT_WORKER_HOME),
                        help="writable HOME/cache root inherited from GPU process startup")
    parser.add_argument("--worker-max-attempts", type=int, default=3)
    parser.add_argument("--worker-retry-delay-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.family_modes is None:
        args.family_modes = ["atomic"] * len(args.families)

    if (len(args.families) != len(args.family_tags)
            or len(args.families) != len(args.family_modes)
            or len(set(args.family_tags)) != len(args.family_tags)):
        parser.error("--families, --family-tags, and --family-modes must align; tags must be unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", tag) or tag in {".", ".."}
           for tag in args.family_tags):
        parser.error("--family-tags must be safe filename components")
    if min(args.batch_per_family, args.confirm_per_family, args.checkpoint_per_family) <= 1:
        parser.error("per-family batch sizes must exceed one")
    if args.ceiling_horizon_per_family < 0:
        parser.error("--ceiling-horizon-per-family must be nonnegative")
    if not 2 <= args.mcq_n_options or not 2 <= args.mcq_design_size <= 300:
        parser.error("MCQ requires n-options >=2 and design-size in [2,300]")
    if (args.mcq_min_design_disagreements < 1 or args.mcq_n_examples < 2
            or args.mcq_n_examples > args.mcq_design_size):
        parser.error("invalid MCQ disagreement/example design")
    if (args.mcq_reconstruction_draws < args.mcq_n_options
            or args.mcq_reconstruction_draws % args.mcq_n_options != 0):
        parser.error("--mcq-reconstruction-draws must be a positive multiple of --mcq-n-options")
    if args.mcq_max_chars <= 0 or args.mcq_value_query_batch_size <= 0:
        parser.error("MCQ character and value-query batch sizes must be positive")
    if (not 0.0 <= args.mcq_min_headline_value_cap <= 1.0
            or not -1.0 <= args.mcq_min_headline_distractor_kappa <= 1.0):
        parser.error("invalid MCQ headline-quality thresholds")
    if (args.mcq_prior_candidate_pool_size < args.mcq_n_options - 1
            or args.mcq_prior_max_panels_per_target < 1):
        parser.error("invalid MCQ prior-calibration search budget")
    if (not 0.0 < args.mcq_prior_max_option_probability <= 1.0
            or not 0.0 <= args.mcq_prior_target_probability_tolerance <= 1.0
            or not 0.0 <= args.mcq_prior_min_normalized_entropy <= 1.0):
        parser.error("invalid MCQ prior-balance thresholds")
    if args.max_iter <= 0 or args.patience <= 0:
        parser.error("--max-iter and --patience must be positive")
    if args.worker_max_attempts <= 0 or args.worker_retry_delay_seconds < 0.0:
        parser.error("worker retry count must be positive and delay nonnegative")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must lie in (0, 1)")
    if args.study_alpha is not None and not 0.0 < args.study_alpha < 1.0:
        parser.error("--study-alpha must lie in (0, 1)")
    if not 0.0 <= args.tau <= args.tau_strict <= 1.0:
        parser.error("require 0 <= --tau <= --tau-strict <= 1")
    if not 0.0 <= args.target_u0 <= 1.0 or target_value_gap(args) < 0.0:
        parser.error("targets must be nonnegative and --target-u0 at most one")
    if args.min_delta < 0.0 or args.min_delta_bits < 0.0:
        parser.error("minimum improvements must be nonnegative")
    if args.p_min is not None and not 0.0 < args.p_min <= 1.0:
        parser.error("--p-min must lie in (0, 1]")
    if args.value_p_min is not None and not 0.0 < args.value_p_min <= 1.0:
        parser.error("--value-p-min must lie in (0, 1]")
    if args.value_p_min is not None and args.value_mode != "reconstruction_mcq":
        parser.error("--value-p-min is defined only for reconstruction_mcq value states")
    if args.mcq_codebook_metrics is not None and args.value_mode != "reconstruction_mcq":
        parser.error("--mcq-codebook-metrics is defined only for reconstruction_mcq mode")
    if args.reuse_mcq_codebook_root is not None and args.value_mode != "reconstruction_mcq":
        parser.error("--reuse-mcq-codebook-root is defined only for reconstruction_mcq mode")
    if args.reuse_bootstrap_root:
        reuse_manifest = Path(args.reuse_bootstrap_root).resolve() / "run_manifest.json"
        if not reuse_manifest.is_file():
            parser.error("--reuse-bootstrap-root must contain run_manifest.json")
    if args.reuse_mcq_codebook_root:
        reuse_manifest = Path(args.reuse_mcq_codebook_root).resolve() / "run_manifest.json"
        if not reuse_manifest.is_file():
            parser.error("--reuse-mcq-codebook-root must contain run_manifest.json")
    if args.reuse_evidence_root:
        try:
            load_evidence_manifest(args.reuse_evidence_root)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(f"invalid --reuse-evidence-root: {exc}")
    if not args.dry_run:
        worker_home = Path(args.worker_home).resolve()
        if not worker_home.is_dir() or not os.access(worker_home, os.W_OK | os.X_OK):
            parser.error("--worker-home must be an existing writable directory")
    try:
        checkpoint_iterations(args)
    except ValueError as exc:
        parser.error(str(exc))

    root = Path(args.out_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        manifest = prepare_manifest(root, args)
        if args.reuse_evidence_root:
            install_evidence_store(args.reuse_evidence_root, root)
        prepare_bootstraps(root, manifest, args)
        codebook_candidate_paths = prepare_mcq_codebook_bootstraps(root, manifest, args)
        states: dict[str, dict] = {}
        worlds: dict[str, dict] = {}
        for path in args.metrics:
            state, world = _load_metric(path, root, args.dry_run, manifest)
            if state["key"] in states:
                raise RuntimeError(f"duplicate metric key {state['key']}")
            states[state["key"]] = state
            if world is not None:
                worlds[state["key"]] = world
        prepare_mcq_codebooks(root, states, args, codebook_candidate_paths)
        prepare_historical_candidates(root, states, manifest, args, worlds)
        attach_mcq_pool_values(root, states, args, worlds)
        cell_alpha, alpha_scope = confirmation_alpha(args, n_metrics=len(states))

        while True:
            active = [key for key, state in states.items()
                      if state["stopped"] is None and state["iteration"] < args.max_iter]
            if not active:
                break
            iteration = min(states[key]["iteration"] for key in active)
            batch_keys = [key for key in active if states[key]["iteration"] == iteration]
            if iteration in checkpoint_iterations(args):
                write_checkpoint_certificates(
                    batch_keys,
                    states,
                    iteration=iteration,
                    manifest=manifest,
                    args=args,
                    worlds=worlds,
                    cell_alpha=cell_alpha,
                    alpha_scope=alpha_scope,
                )
            print(f"=== monitor iteration {iteration}: {len(batch_keys)} metrics ===", flush=True)
            proposals = propose_all(
                batch_keys, states, phase="monitor", iteration=iteration,
                n_per_family=args.batch_per_family, manifest=manifest, args=args, worlds=worlds)
            scored = score_all(
                batch_keys, states, proposals, phase="monitor", iteration=iteration,
                expected_per_family=args.batch_per_family, args=args, worlds=worlds)
            values = value_all(
                batch_keys, states, scored, phase="monitor", iteration=iteration,
                args=args, worlds=worlds)
            for key in batch_keys:
                state = states[key]
                cert, batch = _certificate(
                    state, scored[key], args.batch_per_family, manifest, args,
                    value_payload=values.get(key))
                primary = cert["certified"]
                u0 = float(primary["behavioral_missing_mass_U0"])
                gap = float(primary["finite_horizon_expected_best_gain_UCB_bits"])
                improved = ((state["best_u0"] - u0) >= args.min_delta
                            or (state["best_gap"] - gap) >= args.min_delta_bits)
                state["stall"] = 0 if improved else state["stall"] + 1
                state["best_u0"] = min(state["best_u0"], u0)
                state["best_gap"] = min(state["best_gap"], gap)
                stopped = None
                if u0 <= args.target_u0 and gap <= target_value_gap(args):
                    stopped = "target"
                elif iteration + 1 >= args.max_iter:
                    stopped = "max_iter"
                elif state["stall"] >= args.patience:
                    stopped = "patience"
                row = {
                    "schema": LEDGER_SCHEMA,
                    "event": "absorb",
                    "iteration": iteration,
                    "scored_path": str(scored[key].relative_to(root)),
                    "scored_sha256": file_sha256(scored[key]),
                    "bootstrap_sha256": state["bootstrap_sha256"],
                    "probe_sha256": state["probe_sha256"],
                    "cache_namespace_sha256": state["cache_namespace_sha256"],
                    "family_names": list(args.family_tags),
                    "expected_per_family": args.batch_per_family,
                    "pool_n_before": len(state["pool"]),
                    "pool_n_after": len(state["pool"]) + len(batch),
                    "monitor": {
                        "behavioral_U0": u0,
                        "exact_U0": primary["exact_pattern_missing_mass_U0"],
                        "pool_best_bits": primary["pool_best_prompt_recovery_bits"],
                        "horizon_gain_UCB_bits": gap,
                        "horizon_ceiling_UCB_bits": primary["finite_horizon_expected_prompt_ceiling_UCB_bits"],
                    },
                    "stall": state["stall"],
                    "stopped": stopped,
                    "time": time.time(),
                }
                if key in values:
                    row["value_path"] = str(Path(values[key]["path"]).relative_to(root))
                    row["value_sha256"] = values[key]["sha256"]
                _append_jsonl(state["dir"] / "absorption_ledger.jsonl", row)
                state["pool"] = np.vstack([state["pool"], batch])
                if key in values:
                    state["pool_values"] = np.concatenate([
                        state["pool_values"], values[key]["values"]])
                    state["pool_value_details"].extend(values[key]["details"])
                    state["pool_value_species"].extend(
                        detail["design"]["teaching_transcript_sha256"]
                        for detail in values[key]["details"])
                    state["value_determined_by_exact_behavior"] = (
                        bool(state["value_determined_by_exact_behavior"])
                        and bool(values[key]["premises"].get(
                            "value_determined_by_exact_behavior")))
                state["iteration"] += 1
                state["stopped"] = stopped
                print(
                    f"  {key}: U0={u0:.3f} horizon_gain<={gap:.4f} "
                    f"ceiling<={primary['finite_horizon_expected_prompt_ceiling_UCB']:.4f} "
                    f"{state.get('value_unit', 'bits')} "
                    f"stall={state['stall']} {('STOP:' + stopped) if stopped else ''}",
                    flush=True,
                )

        pending = [key for key, state in states.items() if state["stopped"] and not state["confirmed"]]
        if pending:
            print(f"=== immutable confirmation audits: {len(pending)} metrics ===", flush=True)
            proposals = propose_all(
                pending, states, phase="confirmation", iteration=0,
                n_per_family=args.confirm_per_family, manifest=manifest, args=args, worlds=worlds)
            scored = score_all(
                pending, states, proposals, phase="confirmation", iteration=0,
                expected_per_family=args.confirm_per_family, args=args, worlds=worlds)
            values = value_all(
                pending, states, scored, phase="confirmation", iteration=0,
                args=args, worlds=worlds)
            for key in pending:
                state = states[key]
                cert, _ = _certificate(
                    state, scored[key], args.confirm_per_family, manifest, args,
                    alpha_override=cell_alpha, certificate_role="final_confirmation",
                    value_payload=values.get(key))
                cert["run"] = {
                    "stopped": state["stopped"],
                    "iterations_absorbed": state["iteration"],
                    "final_pool_size": len(state["pool"]),
                    "confirmation_scored_path": str(scored[key].relative_to(root)),
                    "confirmation_scored_sha256": file_sha256(scored[key]),
                    "never_absorbed": True,
                    "alpha_scope": alpha_scope,
                }
                if key in values:
                    cert["run"]["confirmation_value_path"] = str(
                        Path(values[key]["path"]).relative_to(root))
                    cert["run"]["confirmation_value_sha256"] = values[key]["sha256"]
                cert["prompt_evolution_status"] = classify_prompt_evolution(
                    cert,
                    confirmation_is_never_absorbed=True,
                    stopping_rule_frozen_before_confirmation=True,
                    plateau_epsilon=target_value_gap(args),
                    saturation_missing_mass=args.target_u0,
                )
                path = state["dir"] / "confirmation" / "certificate.json"
                _atomic_json(path, cert)
                state["confirmed"] = True
                c = cert["certified"]
                print(
                    f"  {key}: CONFIRMED U0<={c['behavioral_missing_mass_U0']:.3f} "
                    f"horizon prompt ceiling<={c['finite_horizon_expected_prompt_ceiling_UCB']:.4f} "
                    f"{state.get('value_unit', 'bits')}",
                    flush=True,
                )
        write_certified_trajectories(states, args, alpha_scope)
        write_mcq_bank_identity_summary(root, states, args)
        print("CR3 PROMPT-CEILING LOOP DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
