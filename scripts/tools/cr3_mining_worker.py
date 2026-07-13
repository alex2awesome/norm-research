#!/usr/bin/env python3
"""GPU worker for certified CR-3 prompt mining.

Proposal draws use one independently seeded vLLM request per sampling occasion.
Validation is deterministic rejection sampling, so retained rows are draws from the
frozen generator distribution conditional on the declared validity predicate.  The
worker fails unless every requested family quota is filled exactly.

Scoring defines one deterministic empirical behavior per unique prompt text.  A
run-persistent, content-addressed cache is keyed by the ordered probe panel, executor
revision, readout protocol, and prompt text, so recaptures across worker processes reuse
the exact same signature.  The initial pool and target are bootstrapped through that same
cache before mining.  Probe, executor, generator, and per-draw provenance travel with
every scored artifact.  Final files are written atomically and never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from methods.metric_implementer.config import ImplementerConfig, apply_task_preset  # noqa: E402
from methods.metric_implementer.experiments import alpha_probe as aprobe  # noqa: E402
from methods.metric_implementer.experiments.cr3_reconstruction_values import (  # noqa: E402
    CachedChoiceReconstructor,
    evaluate_scored_prompt_values,
    score_codebook_panel_priors,
    write_value_artifact,
)
from methods.metric_implementer.recon_channel import _YESNO_TEMPLATE  # noqa: E402
from methods.metric_implementer.experiments.run_real_test import _load_texts  # noqa: E402
from methods.metric_implementer.vllm_backend import (  # noqa: E402
    CR3_BINARY_READOUT_ID,
    make_judge_backend,
    model_revision_id,
)

READOUT_ID = CR3_BINARY_READOUT_ID
ATOMIC_PROPOSE_INSTRUCTION = (
    'You are sampling candidate articulations of an evaluation metric.\n'
    'Metric name: "{name}"\n'
    'Metric definition: "{description}"\n'
    "Propose ONE concrete, checkable yes/no criterion that a careful reader could verify "
    "about a text under this metric. Cover substantive content, including edge cases; avoid "
    "generic quality words.\n"
    "Output ONLY the criterion as one sentence ending in '?'."
)
HOLISTIC_PROPOSE_INSTRUCTION = (
    'You are sampling complete prompt articulations of an evaluation metric.\n'
    'Metric name: "{name}"\n'
    'Metric definition: "{description}"\n'
    "Write one standalone, comprehensive rubric that instructs a binary evaluator how to decide "
    "whether a text satisfies the full metric. Preserve the metric's distinct concepts, important "
    "tradeoffs, edge cases, and exclusions. Be concrete enough to apply consistently; do not reduce "
    "the metric to one example or one narrow subcriterion. Multiple sentences and structured clauses "
    "are allowed.\nOutput ONLY the rubric text."
)
PROPOSAL_MODES = {
    "atomic": {
        "prompt_template_id": "criterion-v2-description",
        "validator_id": "single-question-15-240-v1",
        "instruction": ATOMIC_PROPOSE_INSTRUCTION,
        "max_tokens": 80,
    },
    "holistic": {
        "prompt_template_id": "holistic-rubric-v1-description",
        "validator_id": "holistic-rubric-80-8000-v1",
        "instruction": HOLISTIC_PROPOSE_INSTRUCTION,
        "max_tokens": 2048,
    },
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_revision(model: str) -> str:
    return model_revision_id(model)


def _stable_seed(base_seed: int, attempt_idx: int) -> int:
    raw = hashlib.sha256(f"{int(base_seed)}:{int(attempt_idx)}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") & ((1 << 63) - 1)


def _valid(text: str, proposal_mode: str = "atomic") -> bool:
    s = text.strip()
    if proposal_mode == "atomic":
        return 15 <= len(s) <= 240 and "\n" not in s and s.endswith("?")
    if proposal_mode == "holistic":
        return 80 <= len(s) <= 8000
    raise ValueError(f"unknown proposal mode {proposal_mode!r}")


def _normalize_output(text: str | None) -> str:
    return (text or "").strip().strip('"').strip()


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_jsonl(path: str, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact {target}")
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


def _atomic_json(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("x", encoding="utf-8") as fout:
            json.dump(payload, fout, sort_keys=True, indent=2)
            fout.write("\n")
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, target)
        _fsync_directory(target.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_npz(path: str, **arrays) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact {target}")
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        with tmp.open("rb") as fin:
            os.fsync(fin.fileno())
        os.replace(tmp, target)
        _fsync_directory(target.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def draw_valid_rows(backend, prompt: str, *, n: int, base_seed: int, family: str,
                    model: str, temperature: float, max_attempt_mult: int = 10,
                    max_tokens: int | None = None,
                    proposal_mode: str = "atomic") -> tuple[list[dict], list[dict]]:
    """Draw exactly ``n`` accepted rows using a distinct seed for every attempt."""
    if n <= 0:
        raise ValueError("n must be positive")
    if proposal_mode not in PROPOSAL_MODES:
        raise ValueError(f"unknown proposal mode {proposal_mode!r}")
    mode = PROPOSAL_MODES[proposal_mode]
    max_tokens = int(mode["max_tokens"] if max_tokens is None else max_tokens)
    prompt_sha = _sha256_text(prompt)
    config = {
        "model": model,
        "model_revision": _model_revision(model),
        "temperature": float(temperature),
        "prompt_sha256": prompt_sha,
        "proposal_mode": proposal_mode,
        "prompt_template_id": mode["prompt_template_id"],
        "validator_id": mode["validator_id"],
        "max_tokens": int(max_tokens),
    }
    config_sha = _sha256_text(json.dumps(config, sort_keys=True, separators=(",", ":")))
    accepted: list[dict] = []
    attempts: list[dict] = []
    attempt_idx = 0
    max_attempts = max(n + 100, n * int(max_attempt_mult))
    while len(accepted) < n and attempt_idx < max_attempts:
        remaining = n - len(accepted)
        batch_n = min(max(16, remaining * 2), 256, max_attempts - attempt_idx)
        indices = list(range(attempt_idx, attempt_idx + batch_n))
        seeds = [_stable_seed(base_seed, i) for i in indices]
        outputs = backend.generate_batch(
            [prompt] * batch_n,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seeds,
        )
        if len(outputs) != batch_n:
            raise RuntimeError(f"backend returned {len(outputs)} outputs for {batch_n} requests")
        for idx, seed, raw in zip(indices, seeds, outputs):
            text = _normalize_output(raw)
            valid = _valid(text, proposal_mode)
            attempts.append({
                "attempt_idx": idx,
                "seed": seed,
                "valid": valid,
                "raw_text": text,
                "raw_sha256": _sha256_text(text),
                "family": family,
                "proposal_mode": proposal_mode,
                "generator_config_sha256": config_sha,
            })
            if valid and len(accepted) < n:
                accepted.append({
                    "text": text,
                    "family": family,
                    "model": model,
                    "model_revision": config["model_revision"],
                    "temperature": float(temperature),
                    "seed": seed,
                    "attempt_idx": idx,
                    "accepted_idx": len(accepted),
                    "prompt_sha256": prompt_sha,
                    "proposal_mode": proposal_mode,
                    "prompt_template_id": mode["prompt_template_id"],
                    "validator_id": mode["validator_id"],
                    "generator_config_sha256": config_sha,
                })
        attempt_idx += batch_n
    if len(accepted) != n:
        raise RuntimeError(
            f"family {family!r} filled only {len(accepted)}/{n} valid draws after {attempt_idx} attempts"
        )
    if len({row["seed"] for row in accepted}) != n:
        raise RuntimeError("accepted proposal seeds are not unique")
    return accepted, attempts


def stage_propose(args) -> None:
    with open(args.jobs, encoding="utf-8") as fin:
        jobs = json.load(fin)
    cfg = ImplementerConfig()
    if getattr(cfg, "vllm_lfs_home", None):
        os.environ["HOME"] = str(cfg.vllm_lfs_home)
    backend = make_judge_backend(args.model, cfg, args.temp)
    for job in jobs:
        description = str(job.get("metric_description") or job["metric_name"])
        proposal_mode = str(job.get("proposal_mode", "atomic"))
        if proposal_mode not in PROPOSAL_MODES:
            raise RuntimeError(f"unknown proposal mode {proposal_mode!r}")
        mode = PROPOSAL_MODES[proposal_mode]
        prompt = mode["instruction"].format(name=job["metric_name"], description=description)
        rows, attempts = draw_valid_rows(
            backend,
            prompt,
            n=int(job["n"]),
            base_seed=int(job["base_seed"]),
            family=args.family,
            model=args.model,
            temperature=args.temp,
            max_attempt_mult=args.max_attempt_mult,
            proposal_mode=proposal_mode,
        )
        # Accepted rows contain the seed and attempt index needed to reproduce the
        # deterministic rejection-sampling path.  Keeping one atomic stream avoids a
        # two-file transaction whose halves could diverge after a crash.
        _atomic_jsonl(job["out"], rows)
        print(
            f"[propose {args.family}/{proposal_mode}] {job['metric_name'][:40]}: "
            f"accepted {len(rows)}/{len(attempts)} attempts, unique_text={len({r['text'] for r in rows})}",
            flush=True,
        )


def _probe_sha256(probes: list[str]) -> str:
    packed = json.dumps(list(probes), ensure_ascii=False, separators=(",", ":"))
    return _sha256_text(packed)


def _score_seed(namespace_sha256: str, criterion: str, probe_index: int) -> int:
    packed = f"{READOUT_ID}\x1f{namespace_sha256}\x1f{criterion}\x1f{probe_index}"
    return int.from_bytes(hashlib.sha256(packed.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def _checked_signature(executor, criterion: str, probes: list[str], max_chars: int,
                       namespace_sha256: str) -> np.ndarray:
    prompts = [_YESNO_TEMPLATE.format(rubric=criterion, text=text[:max_chars]) for text in probes]
    seeds = [_score_seed(namespace_sha256, criterion, i) for i in range(len(probes))]
    raw = np.asarray(
        executor.score_binary_constrained(prompts, pos="YES", neg="NO", seed=seeds),
        float,
    )
    if raw.shape != (len(probes),):
        raise RuntimeError(f"signature shape {raw.shape} != ({len(probes)},)")
    if np.any(~np.isfinite(raw)):
        raise RuntimeError("executor returned non-finite signature scores")
    return raw


def _cache_namespace(task: str, probes: list[str], model: str, model_revision: str,
                     max_chars: int) -> tuple[str, dict]:
    payload = {
        "task": task,
        "probe_sha256": _probe_sha256(probes),
        "executor_model": model,
        "executor_model_revision": model_revision,
        "readout_id": READOUT_ID,
        "yesno_template_sha256": _sha256_text(_YESNO_TEMPLATE),
        "max_text_chars": int(max_chars),
    }
    digest = _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return digest, payload


def _content_cached_signature(cache_root: str, namespace_sha256: str, criterion: str,
                              n_probes: int, score_fn) -> tuple[np.ndarray, bool]:
    criterion_sha = _sha256_text(criterion)
    path = Path(cache_root) / namespace_sha256 / f"{criterion_sha}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["namespace_sha256"]) != namespace_sha256:
            raise RuntimeError(f"cache namespace mismatch in {path}")
        if str(z["criterion_sha256"]) != criterion_sha or str(z["criterion"]) != criterion:
            raise RuntimeError(f"cache criterion mismatch or SHA-256 collision in {path}")
        signature = np.asarray(z["signature"], float)
        if signature.shape != (n_probes,) or np.any(~np.isfinite(signature)):
            raise RuntimeError(f"invalid cached signature in {path}")
        return signature, False
    signature = np.asarray(score_fn(criterion), float)
    if signature.shape != (n_probes,) or np.any(~np.isfinite(signature)):
        raise RuntimeError("cannot cache an invalid signature")
    _atomic_npz(
        str(path),
        signature=signature,
        namespace_sha256=np.asarray(namespace_sha256),
        criterion=np.asarray(criterion),
        criterion_sha256=np.asarray(criterion_sha),
    )
    return signature, True


def _read_criteria(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fin:
            rows.extend(json.loads(line) for line in fin if line.strip())
    return rows


def _validate_quotas(rows: list[dict], family_names: list[str], expected_per_family: int) -> None:
    counts = {f: 0 for f in family_names}
    for row in rows:
        family = str(row.get("family"))
        if family not in counts:
            raise RuntimeError(f"unexpected family {family!r}")
        counts[family] += 1
    expected = {f: int(expected_per_family) for f in family_names}
    if counts != expected:
        raise RuntimeError(f"family quota mismatch: observed={counts}, expected={expected}")


def score_unique_texts(criteria: list[dict], *, cache_namespace: str, score_fn,
                       cache: dict[tuple[str, str], np.ndarray]) -> tuple[np.ndarray, int]:
    """Score each unique text once and reuse its exact signature for recaptures."""
    signatures: list[np.ndarray] = []
    n_new = 0
    for row in criteria:
        text = str(row["text"])
        key = (cache_namespace, text)
        if key not in cache:
            cache[key] = np.asarray(score_fn(text), float)
            n_new += 1
        signatures.append(cache[key])
    return np.asarray(signatures, float), n_new


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fin:
        for block in iter(lambda: fin.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _target_forms(description: str, n_forms: int) -> list[tuple[str, str]]:
    forms = [("canonical", description)]
    if n_forms > 1:
        forms.extend(list(aprobe._reformulations(description))[:n_forms - 1])
    return forms


def stage_score(args) -> None:
    with open(args.jobs, encoding="utf-8") as fin:
        jobs = json.load(fin)
    cfg_cache: dict[str, tuple[ImplementerConfig, list[str]]] = {}
    executor = None
    signature_cache: dict[tuple[str, str], np.ndarray] = {}
    for job in jobs:
        task = str(job["task"])
        if task not in cfg_cache:
            cfg = ImplementerConfig()
            apply_task_preset(cfg, task)
            cfg.n_oracle_items = 0
            if getattr(cfg, "vllm_lfs_home", None):
                os.environ["HOME"] = str(cfg.vllm_lfs_home)
            texts, _ = _load_texts(task, 60 + 300, cfg)
            probes = list(texts[60:360])
            if len(probes) != 300:
                raise RuntimeError(f"loaded {len(probes)} probes for {task}, expected 300")
            cfg_cache[task] = (cfg, probes)
        cfg, probes = cfg_cache[task]
        if executor is None:
            executor = make_judge_backend(args.model, cfg, 0.0)
        max_chars = int(getattr(cfg, "max_text_chars", 4000))
        model_revision = _model_revision(args.model)
        namespace_sha, namespace = _cache_namespace(
            task, probes, args.model, model_revision, max_chars)
        if job.get("expected_probe_sha256") not in (None, namespace["probe_sha256"]):
            raise RuntimeError("ordered probe panel changed after bootstrap")
        if job.get("expected_executor_model_revision") not in (None, model_revision):
            raise RuntimeError("executor model revision changed after bootstrap")
        if job.get("expected_readout_id") not in (None, READOUT_ID):
            raise RuntimeError("executor readout protocol changed after bootstrap")
        if job.get("expected_cache_namespace_sha256") not in (None, namespace_sha):
            raise RuntimeError("executor cache namespace changed after bootstrap")
        cache_root = str(job["signature_cache_root"])
        cache_created = 0

        def cached_score(text: str) -> np.ndarray:
            nonlocal cache_created
            signature, created = _content_cached_signature(
                cache_root,
                namespace_sha,
                text,
                len(probes),
                lambda criterion: _checked_signature(
                    executor, criterion, probes, max_chars, namespace_sha),
            )
            cache_created += int(created)
            return signature

        mode = str(job.get("mode", "audit"))
        if mode in {"bootstrap", "codebook_bootstrap"}:
            z = np.load(job["orig_npz"], allow_pickle=True)
            description = str(job["metric_description"]).strip()
            if not description:
                raise RuntimeError("bootstrap requires a nonempty metric description")
            n_forms = max(1, int(job.get("target_orbit_forms", 1)))
            forms = _target_forms(description, n_forms)
            form_sigs = np.vstack([cached_score(text) for _, text in forms])
            target = np.mean(form_sigs, axis=0)
            legacy_target = np.asarray(z["M_i"], float)
            if mode == "bootstrap":
                prompts = [str(x) for x in z["prompts"]]
                source_tags = [str(x) for x in z["tags"]]
                if not prompts or len(prompts) != len(source_tags):
                    raise RuntimeError("bootstrap checkpoint has invalid prompts/tags")
                criteria = [{"text": text} for text in prompts]
                sigs, unique_resolved = score_unique_texts(
                    criteria,
                    cache_namespace=namespace_sha,
                    score_fn=cached_score,
                    cache=signature_cache,
                )
                legacy = np.asarray(z["sigs"], float)
                legacy_agreement = {
                    "n_rows": len(sigs),
                    "mean_row_binary_agreement": float(np.mean(
                        (sigs > 0.5) == (legacy > 0.5))),
                    "exact_row_fraction": float(np.mean(np.all(
                        (sigs > 0.5) == (legacy > 0.5), axis=1))),
                    "target_binary_agreement": float(np.mean(
                        (target > 0.5) == (legacy_target > 0.5))),
                    "validity_role": (
                        "diagnostic only; bootstrap signatures define the v2 empirical executor"),
                }
                schema = "cr3-bootstrap-v2"
            else:
                # Distractor-bank candidates need only their canonical executor behavior.
                # Historical prompt pools are deliberately not rescored or admitted into search.
                prompts = [description]
                source_tags = ["canonical_codebook_target"]
                sigs = target[None, :]
                unique_resolved = len(forms)
                legacy_agreement = {
                    "n_rows": 1,
                    "target_binary_agreement": float(np.mean(
                        (target > 0.5) == (legacy_target > 0.5))),
                    "validity_role": (
                        "diagnostic only; canonical behavior defines the frozen MCQ candidate"),
                }
                schema = "cr3-codebook-bootstrap-v1"
            _atomic_npz(
                job["out"],
                schema=np.asarray(schema),
                sigs=sigs,
                texts=np.asarray(prompts, object),
                source_tags=np.asarray(source_tags, object),
                target=target,
                target_forms=form_sigs,
                target_form_names=np.asarray([name for name, _ in forms], object),
                target_form_texts=np.asarray([text for _, text in forms], object),
                metric_description=np.asarray(description),
                probe_texts=np.asarray(probes, object),
                probe_sha256=np.asarray(namespace["probe_sha256"]),
                executor_model=np.asarray(args.model),
                executor_model_revision=np.asarray(model_revision),
                executor_temperature=np.asarray(0.0),
                readout_id=np.asarray(READOUT_ID),
                cache_namespace_sha256=np.asarray(namespace_sha),
                source_checkpoint=np.asarray(job["orig_npz"]),
                source_checkpoint_sha256=np.asarray(_file_sha256(job["orig_npz"])),
                metric_key=np.asarray(str(job.get("metric_key", ""))),
                legacy_alignment_json=np.asarray(json.dumps(legacy_agreement, sort_keys=True)),
            )
            print(
                f"[{mode} {task}] wrote {job['out']} rows={len(prompts)} "
                f"unique_resolved={unique_resolved} cache_created={cache_created}",
                flush=True,
            )
            continue
        if mode != "audit":
            raise RuntimeError(f"unknown score mode {mode!r}")

        criteria = _read_criteria(list(job["criteria"]))
        family_names = [str(f) for f in job["family_names"]]
        _validate_quotas(criteria, family_names, int(job["expected_per_family"]))
        sigs, unique_resolved = score_unique_texts(
            criteria,
            cache_namespace=namespace_sha,
            score_fn=cached_score,
            cache=signature_cache,
        )
        _atomic_npz(
            job["out"],
            schema=np.asarray("cr3-audit-signatures-v2"),
            sigs=sigs,
            texts=np.asarray([r["text"] for r in criteria], object),
            families=np.asarray([r["family"] for r in criteria], object),
            models=np.asarray([r["model"] for r in criteria], object),
            model_revisions=np.asarray([r["model_revision"] for r in criteria], object),
            temperatures=np.asarray([r["temperature"] for r in criteria], float),
            seeds=np.asarray([r["seed"] for r in criteria], np.int64),
            attempt_idx=np.asarray([r["attempt_idx"] for r in criteria], np.int64),
            accepted_idx=np.asarray([r["accepted_idx"] for r in criteria], np.int64),
            prompt_sha256=np.asarray([r["prompt_sha256"] for r in criteria], object),
            generator_config_sha256=np.asarray([r["generator_config_sha256"] for r in criteria], object),
            probe_sha256=np.asarray(namespace["probe_sha256"]),
            executor_model=np.asarray(args.model),
            executor_model_revision=np.asarray(model_revision),
            executor_temperature=np.asarray(0.0),
            readout_id=np.asarray(READOUT_ID),
            cache_namespace_sha256=np.asarray(namespace_sha),
            source_criteria=np.asarray(list(job["criteria"]), object),
        )
        print(
            f"[score] wrote {job['out']} rows={len(criteria)} "
            f"unique_resolved={unique_resolved} cache_created={cache_created}",
            flush=True,
        )


def stage_value(args) -> None:
    """Assign anchor-free Reconstruction-MCQ values to every pre-scored prompt row."""
    with open(args.jobs, encoding="utf-8") as source:
        jobs = json.load(source)
    cfg = ImplementerConfig()
    if args.fake:
        cfg.vllm_fake = True
    if getattr(cfg, "vllm_lfs_home", None):
        os.environ["HOME"] = str(cfg.vllm_lfs_home)
    reconstructor = make_judge_backend(args.model, cfg, 0.0)
    revision = _model_revision(args.model)
    cache_paths = {str(job.get("choice_probability_cache")) for job in jobs
                   if job.get("choice_probability_cache")}
    if len(cache_paths) > 1:
        raise RuntimeError("one value worker cannot mix choice-probability caches")
    if cache_paths:
        reconstructor = CachedChoiceReconstructor(
            reconstructor,
            next(iter(cache_paths)),
            model=args.model,
            revision=revision,
        )
    expected_readouts = {str(job.get("expected_choice_readout_id", "")) for job in jobs}
    if len(expected_readouts) != 1 or "" in expected_readouts:
        raise RuntimeError("value jobs must declare one expected choice readout id")
    if reconstructor.choice_readout_id not in expected_readouts:
        raise RuntimeError("value backend does not implement the declared choice readout")
    for job in jobs:
        with open(job["codebook_manifest"], encoding="utf-8") as source:
            codebook = json.load(source)
        payload = evaluate_scored_prompt_values(
            reconstructor,
            codebook_manifest=codebook,
            target_metric_key=str(job["target_metric_key"]),
            scored_path=job["scored"],
            noun=str(job["noun"]),
            n_examples=int(job.get("n_examples", 8)),
            n_reconstruction_draws=int(job.get("n_reconstruction_draws", 4)),
            max_chars=int(job.get("max_chars", 600)),
            choice_readout=str(job.get("choice_readout", "auto")),
            query_batch_size=int(job.get("query_batch_size", 512)),
            fixed_no_demo_canonical_probabilities=(
                np.asarray(job["fixed_no_demo_canonical_choice_probabilities"], float)
                if job.get("fixed_no_demo_canonical_choice_probabilities") is not None
                else None),
            choice_probabilities_content_cached=bool(cache_paths),
        )
        write_value_artifact(
            job["out"], payload,
            reconstructor_model=args.model,
            reconstructor_revision=revision,
        )
        print(
            f"[value] {job['target_metric_key']}: rows={payload['n_rows']} "
            f"mean={float(np.mean(payload['values'])):.4f} -> {job['out']}",
            flush=True,
        )


def stage_codebook_prior(args) -> None:
    """Calibrate blind no-demo menu priors before any prompt-value search."""
    with open(args.jobs, encoding="utf-8") as source:
        jobs = json.load(source)
    cfg = ImplementerConfig()
    if args.fake:
        cfg.vllm_fake = True
    if getattr(cfg, "vllm_lfs_home", None):
        os.environ["HOME"] = str(cfg.vllm_lfs_home)
    reconstructor = make_judge_backend(args.model, cfg, 0.0)
    revision = _model_revision(args.model)
    cache_paths = {str(job.get("choice_probability_cache")) for job in jobs
                   if job.get("choice_probability_cache")}
    if len(cache_paths) > 1:
        raise RuntimeError("one prior worker cannot mix choice-probability caches")
    if cache_paths:
        reconstructor = CachedChoiceReconstructor(
            reconstructor,
            next(iter(cache_paths)),
            model=args.model,
            revision=revision,
        )
    expected_readouts = {str(job.get("expected_choice_readout_id", "")) for job in jobs}
    if len(expected_readouts) != 1 or "" in expected_readouts:
        raise RuntimeError("prior jobs must declare one expected choice readout id")
    if reconstructor.choice_readout_id not in expected_readouts:
        raise RuntimeError("prior backend does not implement the declared choice readout")
    for job in jobs:
        with open(job["panel_plan"], encoding="utf-8") as source:
            plan = json.load(source)
        payload = score_codebook_panel_priors(
            reconstructor,
            panel_plan=plan,
            noun=str(job["noun"]),
            n_draws=int(job.get("n_draws", 4)),
            query_batch_size=int(job.get("query_batch_size", 512)),
            reconstructor_model=args.model,
            reconstructor_revision=revision,
        )
        _atomic_json(job["out"], payload)
        print(
            f"[codebook_prior] targets={len(payload['rows'])} -> {job['out']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=["propose", "score", "value", "codebook_prior"],
    )
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", default="")
    parser.add_argument("--temp", type=float, default=0.9)
    parser.add_argument("--max-attempt-mult", type=int, default=10)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    {
        "propose": stage_propose,
        "score": stage_score,
        "value": stage_value,
        "codebook_prior": stage_codebook_prior,
    }[args.stage](args)


if __name__ == "__main__":
    main()
