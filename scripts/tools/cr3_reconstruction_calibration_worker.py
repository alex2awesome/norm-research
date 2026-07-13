#!/usr/bin/env python3
"""Dedicated GPU worker for frozen Reconstruction-MCQ design calibration.

This entrypoint is intentionally separate from ``cr3_mining_worker.py``. The latter's
content hash defines the reusable executor-bootstrap namespace; adding a design-only
stage there would invalidate already-scored prompt signatures.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from methods.metric_implementer.config import ImplementerConfig  # noqa: E402
from methods.metric_implementer.experiments.cr3_reconstruction_values import (  # noqa: E402
    CENTRALNESS_REFERENCE_DRAWS,
    CachedChoiceReconstructor,
    score_task_centralness_reference,
)
from methods.metric_implementer.vllm_backend import (  # noqa: E402
    make_judge_backend,
    model_revision_id,
)


def _one_expected_job_value(jobs: list[dict], field: str):
    values = {str(job.get(field, "")) for job in jobs}
    if len(values) != 1 or not next(iter(values), ""):
        raise RuntimeError(f"centralness jobs must bind one nonempty {field}")
    return next(iter(values))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != data:
            raise RuntimeError(f"immutable centralness artifact changed: {target}")
        return
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(jobs_path: str | Path, *, model: str, fake: bool = False) -> None:
    with open(jobs_path, encoding="utf-8") as source:
        jobs = json.load(source)
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError("centralness worker requires at least one job")
    expected_model = _one_expected_job_value(jobs, "expected_reconstructor_model")
    expected_revision = _one_expected_job_value(jobs, "expected_reconstructor_revision")
    expected_readout = _one_expected_job_value(jobs, "expected_choice_readout_id")
    if model != expected_model:
        raise RuntimeError("centralness model differs from the frozen run manifest")

    config = ImplementerConfig()
    if fake:
        config.vllm_fake = True
    if getattr(config, "vllm_lfs_home", None):
        os.environ["HOME"] = str(config.vllm_lfs_home)
    revision = model_revision_id(model)
    if revision != expected_revision:
        raise RuntimeError("centralness model revision differs from the frozen run manifest")
    backend = make_judge_backend(model, config, 0.0)
    cache_paths = {
        str(job.get("choice_probability_cache"))
        for job in jobs if job.get("choice_probability_cache")
    }
    if len(cache_paths) > 1:
        raise RuntimeError("one centralness worker cannot mix choice-probability caches")
    if cache_paths:
        backend = CachedChoiceReconstructor(
            backend, next(iter(cache_paths)), model=model, revision=revision)
    if backend.choice_readout_id != expected_readout:
        raise RuntimeError("centralness backend does not implement the declared choice readout")

    for job in jobs:
        if int(job.get("n_draws", -1)) != CENTRALNESS_REFERENCE_DRAWS:
            raise RuntimeError("centralness job does not bind the frozen four-order design")
        with open(job["centralness_plan"], encoding="utf-8") as source:
            plan = json.load(source)
        payload = score_task_centralness_reference(
            backend,
            reference_plan=plan,
            noun=str(job["noun"]),
            query_batch_size=int(job.get("query_batch_size", 512)),
            reconstructor_model=model,
            reconstructor_revision=revision,
        )
        _atomic_json(job["out"], payload)
        print(
            f"[codebook_centralness] metrics={len(payload['centralness'])} -> {job['out']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    run(args.jobs, model=args.model, fake=args.fake)


if __name__ == "__main__":
    main()
