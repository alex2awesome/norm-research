"""Prepare fixed-executor Tier-B codebooks while reusing legacy prompt banks.

This does not generate candidate prompts.  It re-scores each canonical R3 metric
description on the task's frozen 300-text probe panel with the v13.1 fixed executor,
then freezes task codebooks and a 35-metric entropy-quintile campaign manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..batch_scoring import _YESNO_TEMPLATE
from ..config import ImplementerConfig, apply_task_preset
from ..vllm_backend import (
    CR3_BINARY_READOUT_ID,
    make_judge_backend,
    model_revision_id,
    release_resident_engines,
)
from .cr3_reconstruction_values import build_frozen_codebook_manifest
from . import alpha_probe as aprobe
from .cr3_sampled_value_certify import _file_sha256
from .run_v13_value_campaign import (
    FIXED_EXECUTOR,
    METRICS_MANIFEST_SCHEMA,
    _atomic_json,
)


TASK_SPECS = {
    "humor": {
        "run": "runs/cr3_mcq_v12_humor_R3_n30_e601833/run_manifest.json",
        "input": "inputs/r3_humor/llama8b_glm",
        "noun": "joke",
    },
    "creative-writing": {
        "run": (
            "consolidation_v1/sk1_runs/"
            "cr3_mcq_v12_creative-writing_R3_n30_e601833/run_manifest.json"
        ),
        "input": "inputs/r3_cw/llama8b_glm",
        "noun": "story",
    },
    "code-review": {
        "run": "runs/cr3_mcq_v12_code-review_R3_n30_e601833/run_manifest.json",
        "input": "inputs/r3_cr/llama8b_glm",
        "noun": "pull request",
    },
    "news-homepages": {
        "run": "runs/cr3_mcq_v12_news-homepages_R3_n25_e601833/run_manifest.json",
        "input": "inputs/r3_news/llama8b_glm",
        "noun": "news-homepage excerpt",
    },
    "peer-review": {
        "run": "runs/cr3_mcq_v12_peer-review_R3_n13_e601833/run_manifest.json",
        "input": "inputs/r3_peer/llama8b_glm",
        "noun": "peer-review excerpt",
    },
    "legal-outcome-prediction": {
        "run": (
            "runs/cr3_mcq_v12_legal-outcome-prediction_R3_n12_e601833/"
            "run_manifest.json"
        ),
        "input": "inputs/r3_legal/llama8b_glm",
        "noun": "legal fact-section excerpt",
    },
    "math-stackexchange": {
        "run": "runs/cr3_mcq_v12_math-stackexchange_R3_n11_e601833/run_manifest.json",
        "input": "inputs/r3_math/llama8b_glm",
        "noun": "mathematics answer",
    },
}


def _sha_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _probe_sha256(probes: Sequence[str]) -> str:
    return _sha_text(json.dumps(list(probes), ensure_ascii=False, separators=(",", ":")))


def _score_seed(namespace: str, criterion: str, index: int) -> int:
    packed = f"{CR3_BINARY_READOUT_ID}\x1f{namespace}\x1f{criterion}\x1f{index}"
    return int.from_bytes(hashlib.sha256(packed.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def _atomic_signature(path: Path, *, criterion: str, namespace: str,
                      signature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary, criterion=np.asarray(criterion), namespace_sha256=np.asarray(namespace),
        signature=np.asarray(signature, dtype=float),
    )
    os.replace(temporary, path)


def _cached_signature(
    backend, *, criterion: str, probes: Sequence[str], max_chars: int,
    namespace: str, cache_root: Path,
) -> np.ndarray:
    path = cache_root / namespace / f"{_sha_text(criterion)}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as artifact:
            if (str(artifact["criterion"]) != criterion
                    or str(artifact["namespace_sha256"]) != namespace):
                raise RuntimeError(f"signature cache collision at {path}")
            signature = np.asarray(artifact["signature"], dtype=float)
    else:
        prompts = [
            _YESNO_TEMPLATE.format(rubric=criterion, text=str(text)[:max_chars])
            for text in probes
        ]
        seeds = [_score_seed(namespace, criterion, index) for index in range(len(probes))]
        signature = np.asarray(backend.score_binary_constrained(
            prompts, pos="YES", neg="NO", seed=seeds,
        ), dtype=float)
        _atomic_signature(
            path, criterion=criterion, namespace=namespace, signature=signature
        )
    if signature.shape != (len(probes),) or np.any(~np.isfinite(signature)):
        raise RuntimeError(f"invalid fixed-executor signature for {criterion!r}")
    return signature


def _reference_bootstrap(source_root: Path, task: str) -> Path:
    candidates = sorted(
        source_root.glob(f"consolidation_v1/**/*{task}*R3*/**/bootstrap/scored.npz")
    )
    if not candidates:
        raise FileNotFoundError(f"no reference bootstrap found for {task}")
    expected_revision = model_revision_id(FIXED_EXECUTOR)
    for path in candidates:
        with np.load(path, allow_pickle=True) as artifact:
            if (str(artifact.get("executor_model", "")) == FIXED_EXECUTOR
                    and str(artifact.get("executor_model_revision", "")) == expected_revision
                    and len(artifact["probe_texts"]) == 300):
                return path
    raise FileNotFoundError(f"no fixed-executor 300-probe bootstrap found for {task}")


def _identities(source_root: Path, task: str, spec: Mapping[str, str]) -> list[dict]:
    run = json.loads((source_root / spec["run"]).read_text(encoding="utf-8"))
    rows = []
    for identity in run["mcq_codebook_metric_identity"].values():
        if str(identity["task"]) != task:
            continue
        key = str(identity["key"])
        checkpoint = source_root / spec["input"] / f"{key}_sigs.npz"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        rows.append({**dict(identity), "checkpoint": str(checkpoint)})
    by_key = {str(row["key"]): row for row in rows}
    if len(by_key) != len(rows) or len(rows) < 5:
        raise RuntimeError(f"{task} lacks five unique R3 metric identities")
    return [by_key[key] for key in sorted(by_key)]


def prepare_assets(*, source_root: str | Path, out_root: str | Path) -> dict:
    source_root = Path(source_root).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = ImplementerConfig()
    cfg.vllm_gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.82"))
    cfg.vllm_max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
    cfg.vllm_tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
    if os.environ.get("METRIC_IMPLEMENTER_LFS_HOME"):
        cfg.vllm_lfs_home = os.environ["METRIC_IMPLEMENTER_LFS_HOME"]
    backend = make_judge_backend(FIXED_EXECUTOR, cfg, 0.0)
    revision = model_revision_id(FIXED_EXECUTOR)
    manifest_entries = []
    task_reports = []
    try:
        for task, spec in TASK_SPECS.items():
            identities = _identities(source_root, task, spec)
            reference = _reference_bootstrap(source_root, task)
            with np.load(reference, allow_pickle=True) as artifact:
                probes = [str(value) for value in artifact["probe_texts"]]
                probe_sha = str(artifact["probe_sha256"])
            if _probe_sha256(probes) != probe_sha:
                raise RuntimeError(f"reference probe hash mismatch for {task}")
            task_cfg = ImplementerConfig()
            apply_task_preset(task_cfg, task)
            max_chars = int(task_cfg.max_text_chars)
            namespace_payload = {
                "task": task, "probe_sha256": probe_sha,
                "executor_model": FIXED_EXECUTOR,
                "executor_model_revision": revision,
                "readout_id": CR3_BINARY_READOUT_ID,
                "yesno_template_sha256": _sha_text(_YESNO_TEMPLATE),
                "max_text_chars": max_chars,
            }
            namespace = _sha_text(json.dumps(
                namespace_payload, sort_keys=True, separators=(",", ":")
            ))
            bootstrap_paths = []
            for identity in identities:
                description = str(identity["description"]).strip()
                n_forms = max(1, int(identity.get("target_orbit_forms", 1)))
                forms = [("canonical", description)]
                if n_forms > 1:
                    forms.extend(list(aprobe._reformulations(description))[:n_forms - 1])
                form_signatures = np.vstack([
                    _cached_signature(
                        backend, criterion=form_text, probes=probes,
                        max_chars=max_chars, namespace=namespace,
                        cache_root=out_root / "signature_cache",
                    ) for _form_name, form_text in forms
                ])
                target = np.mean(form_signatures, axis=0)
                key = str(identity["key"])
                destination = (
                    out_root / task / "mcq_codebook_candidates" / key /
                    "bootstrap" / "scored.npz"
                )
                if not destination.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(
                        f".{destination.name}.tmp-{os.getpid()}.npz"
                    )
                    np.savez_compressed(
                        temporary,
                        sigs=target[None, :], texts=np.asarray([description], dtype=object),
                        target=target, metric_description=np.asarray(description),
                        probe_texts=np.asarray(probes, dtype=object),
                        probe_sha256=np.asarray(probe_sha),
                        executor_model=np.asarray(FIXED_EXECUTOR),
                        executor_model_revision=np.asarray(revision),
                        readout_id=np.asarray(CR3_BINARY_READOUT_ID),
                        target_forms=form_signatures,
                        target_form_names=np.asarray([name for name, _ in forms], dtype=object),
                        target_form_texts=np.asarray([text for _, text in forms], dtype=object),
                        source_checkpoint=np.asarray(identity["checkpoint"]),
                        source_checkpoint_sha256=np.asarray(_file_sha256(identity["checkpoint"])),
                        metric_key=np.asarray(key),
                    )
                    os.replace(temporary, destination)
                bootstrap_paths.append(destination)
            codebook = build_frozen_codebook_manifest(
                bootstrap_paths, n_options=4, design_size=120,
                min_design_disagreements=2, seed=0,
                reconstruction_noun=str(spec["noun"]), reconstruction_max_chars=600,
            )
            codebook_path = out_root / task / "mcq_codebooks" / f"{task}.json"
            _atomic_json(codebook_path, codebook)
            valid = 0
            identity_by_key = {str(row["key"]): row for row in identities}
            for key, entry in codebook["entries"].items():
                if not entry["valid"]:
                    continue
                valid += 1
                identity = identity_by_key[str(key)]
                manifest_entries.append({
                    "task": task, "level": str(identity["level"]),
                    "metric": str(identity["group_index"]), "metric_key": str(key),
                    "codebook_path": str(codebook_path), "codebook_layout": "production",
                    "assets_root": str(out_root / task),
                    "candidate_bank_path": str(identity["checkpoint"]),
                })
            if valid < 5:
                raise RuntimeError(f"{task} produced only {valid} valid codebook metrics")
            task_reports.append({
                "task": task, "n_codebook_metrics": len(identities),
                "n_valid_entries": valid, "probe_sha256": probe_sha,
                "codebook_path": str(codebook_path),
            })
    finally:
        release_resident_engines()
    metrics_manifest = {
        "schema": METRICS_MANIFEST_SCHEMA,
        "release": "v13.1",
        "description": (
            "Tier-B R3 breadth: fixed 8B target codebooks and legacy prompt/behavior banks"
        ),
        "auto_upgrade_tier_a": True,
        "selection": {
            "mode": "target_entropy_quintiles", "per_task": 5,
            "tasks": list(TASK_SPECS),
        },
        "metrics": manifest_entries,
    }
    manifest_path = out_root / "tier_b_metrics.json"
    _atomic_json(manifest_path, metrics_manifest)
    report = {
        "schema": "cr3-value-bound-tier-b-assets-v13.1",
        "executor": FIXED_EXECUTOR, "executor_revision": revision,
        "readout_id": CR3_BINARY_READOUT_ID,
        "candidate_prompt_regeneration_performed": False,
        "legacy_candidate_banks_process_relative_bounds_available": False,
        "tasks": task_reports, "metrics_manifest_path": str(manifest_path),
    }
    _atomic_json(out_root / "asset_report.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    prepare_assets(source_root=args.source_root, out_root=args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
