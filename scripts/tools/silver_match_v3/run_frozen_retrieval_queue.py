#!/usr/bin/env python3
"""Validate or explicitly run a frozen retrieval queue.

Validation is the default.  GPU execution requires ``--run`` and is guarded by
the queue prerequisite plus an immediate genuine-idle target check.  Every step
is idempotent at the artifact level: valid audited outputs are skipped,
partial retrievals resume, and inconsistent sealed outputs fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .audit_candidate_outputs import audit_candidates
from .common import sha256_file
from .gpu_host_policy import validate_gpu_indices_for_host, validate_launch_gpus


def _verify_artifact(value: dict[str, Any]) -> None:
    path = Path(value["path"])
    if not path.exists() or sha256_file(path) != value["sha256"]:
        raise ValueError(f"frozen artifact hash mismatch: {path}")


def validate_plan(plan: dict[str, Any]) -> None:
    if (
        plan.get("schema_version") != "silver-match-v3-retrieval-command-queue-v1"
        or plan.get("status") != "FROZEN_NOT_LAUNCHED"
        or plan.get("release_ready") is not False
    ):
        raise ValueError("unsupported/unfrozen retrieval queue")
    for key in ("spec", "manifest", "selection"):
        _verify_artifact(plan[key])
    _verify_artifact(plan["bank"])
    for value in plan["implementations"].values():
        _verify_artifact(value)
    for corpus in plan["corpora"].values():
        _verify_artifact(corpus["canonical"])
    existing_sources: dict[str, tuple[str, int]] = {}
    for lane in plan.get("existing_lanes") or []:
        expected_k = int(lane["expected_k"])
        if set(lane["candidates"]) != set(plan["corpora"]):
            raise ValueError("existing lane does not cover every corpus")
        for corpus, frozen in lane["candidates"].items():
            for identity in frozen.values():
                _verify_artifact(identity)
            candidate = str(frozen["candidate"]["path"])
            if candidate in existing_sources:
                raise ValueError("existing candidate source is reused across lanes")
            existing_sources[candidate] = (str(corpus), expected_k)
    coverage = plan.get("coverage_contract") or {}
    if (
        coverage.get("scope") != "all-manifest-corpora-for-task"
        or int(coverage.get("corpus_count", -1)) != len(plan["corpora"])
        or int(coverage.get("norm_count", -1))
        != sum(int(value["count"]) for value in plan["corpora"].values())
        or coverage.get("one_exact_candidate_row_per_norm_required") is not True
        or coverage.get("diagnostic_subset_reuse_forbidden") is not True
    ):
        raise ValueError("queue lacks exact all-corpus production coverage")
    for system in plan["systems"]:
        _verify_artifact(system["fusion"])
        encoder = system["encoder"]
        root = Path(encoder["path"])
        if root.name != encoder["snapshot_revision"]:
            raise ValueError(f"encoder revision path mismatch: {root}")
        for relative, identity in encoder["identity_files"].items():
            path = root / relative
            if (
                not path.exists()
                or path.stat().st_size != int(identity["bytes"])
                or sha256_file(path) != identity["sha256"]
            ):
                raise ValueError(f"encoder identity mismatch: {path}")
        adapter = system.get("adapter")
        if adapter:
            adapter_root = Path(adapter["path"])
            for relative, identity in adapter["identity_files"].items():
                path = adapter_root / relative
                if (
                    not path.exists()
                    or path.stat().st_size != int(identity["bytes"])
                    or sha256_file(path) != identity["sha256"]
                ):
                    raise ValueError(f"adapter identity mismatch: {path}")
            content_sha256 = hashlib.sha256(
                json.dumps(
                    adapter["identity_files"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            if content_sha256 != adapter.get("content_sha256"):
                raise ValueError(f"adapter content identity mismatch: {adapter_root}")
    if not plan.get("steps"):
        raise ValueError("queue has no steps")
    execution = plan.get("execution") or {}
    if "gpu_index" not in execution:
        raise ValueError("queue lacks a target GPU")
    allowed = {"retrieve", "project", "union", "audit"}
    produced = {
        str(step["candidate"]): (str(step["corpus"]), int(step["expected_k"]))
        for step in plan["steps"]
        if step["kind"] == "retrieve"
    }
    if set(produced) & set(existing_sources):
        raise ValueError("generated and existing lane paths overlap")
    available_sources = {**produced, **existing_sources}
    union_steps = []
    for step in plan["steps"]:
        if step.get("kind") not in allowed or not step.get("command"):
            raise ValueError("invalid retrieval queue step")
        if step.get("corpus") not in plan["corpora"]:
            raise ValueError("step corpus is outside the frozen task")
        if step["kind"] == "union":
            sources = step.get("source_candidates") or []
            if len(sources) < 2 or not set(map(str, sources)) <= set(available_sources):
                raise ValueError("union sources are not frozen retrieval outputs")
            expected = step.get("source_expected_k") or {}
            if set(map(str, sources)) != set(expected):
                raise ValueError("union source depth contract is incomplete")
            if any(
                available_sources[str(source)]
                != (str(step["corpus"]), int(expected[str(source)]))
                for source in sources
            ):
                raise ValueError("union source routing/depth differs from its corpus")
            union_steps.append(step)
    union = plan.get("union")
    if union is not None:
        lane_weights = union.get("lane_weights")
        algorithm = union.get("algorithm")
        all_lane_names = {str(system["name"]) for system in plan["systems"]} | {
            str(lane["name"]) for lane in plan.get("existing_lanes") or []
        }
        if (
            not isinstance(lane_weights, dict)
            or algorithm
            not in {
                "weighted-complete-bank-rrf-v1",
                "coverage-preserving-component-prefix-rrf-v1",
            }
            or int(union.get("output_k", -1)) < 1
            or float(union.get("rank_constant", 0)) <= 0
            or set(lane_weights) != all_lane_names
            or any(float(value) <= 0 for value in lane_weights.values())
            or len(union_steps) != len(plan["corpora"])
            or {str(step["corpus"]) for step in union_steps} != set(plan["corpora"])
            or any(
                int(step["expected_k"]) != int(union["output_k"])
                for step in union_steps
            )
        ):
            raise ValueError("invalid or incomplete frozen union plan")
        preserve_components = union.get("preserve_components") or {}
        preserve_k = union.get("preserve_k")
        if algorithm == "coverage-preserving-component-prefix-rrf-v1":
            if (
                not isinstance(preserve_components, dict)
                or set(preserve_components) != set(lane_weights)
                or not all(preserve_components.values())
                or int(preserve_k or 0) < 1
                or int(union["output_k"]) != int(plan["full_k"])
            ):
                raise ValueError("invalid component-prefix preservation policy")
        elif preserve_components or preserve_k is not None:
            raise ValueError("ordinary union unexpectedly contains prefix preservation")
    elif union_steps:
        raise ValueError("union steps exist without a frozen union policy")


def _candidate_valid(
    *, manifest: Path, corpus: str, candidate: Path, expected_k: int
) -> bool:
    meta = candidate.with_suffix(candidate.suffix + ".meta.json")
    if not candidate.exists() or not meta.exists():
        return False
    try:
        audit_candidates(
            manifest_path=manifest,
            corpus=corpus,
            candidate_paths=[candidate],
            expected_k=expected_k,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    return True


def _stored_audit_valid(path: Path, *, candidate: Path, corpus: str, expected_k: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        inputs = payload.get("candidate_inputs") or {}
        frozen_input = inputs.get(str(candidate.resolve())) or {}
        return (
            payload.get("complete") is True
            and payload.get("corpus") == corpus
            and int(payload.get("expected_k", -1)) == expected_k
            and frozen_input.get("sha256") == sha256_file(candidate)
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _prerequisite_ready(plan: dict[str, Any]) -> bool:
    prerequisite = plan["execution"].get("prerequisite")
    if not prerequisite:
        return True
    return _candidate_valid(
        manifest=Path(plan["manifest"]["path"]),
        corpus=str(prerequisite["corpus"]),
        candidate=Path(prerequisite["candidate"]),
        expected_k=int(prerequisite["expected_k"]),
    )


def _wait_for_gpu(plan: dict[str, Any]) -> None:
    execution = plan["execution"]
    gpu = int(execution["gpu_index"])
    validate_gpu_indices_for_host([gpu])
    poll = min(60, max(5, int(execution.get("poll_seconds", 30))))
    while True:
        prerequisite_ready = _prerequisite_ready(plan)
        launch_guard: dict[str, Any] | None = None
        launch_error: str | None = None
        if prerequisite_ready:
            try:
                launch_guard = validate_launch_gpus([gpu])
            except RuntimeError as exc:
                launch_error = str(exc)
        if launch_guard is not None:
            return
        print(
            json.dumps(
                {
                    "status": "WAITING",
                    "prerequisite_ready": prerequisite_ready,
                    "target_gpu": gpu,
                    "target_idle_check": launch_error,
                    "gpu_count_gate_applied": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(poll)


def _run_command(plan: dict[str, Any], command: list[str], *, gpu: bool) -> None:
    environment = os.environ.copy()
    cache_root = plan["execution"].get("cache_root")
    if cache_root:
        root = Path(cache_root)
        environment.setdefault("HF_HOME", str(root / "huggingface"))
        environment.setdefault("HF_MODULES_CACHE", str(root / "huggingface_modules"))
        environment.setdefault("XDG_CACHE_HOME", str(root / "xdg"))
        environment.setdefault("TRANSFORMERS_CACHE", str(root / "transformers"))
    if gpu:
        environment["CUDA_VISIBLE_DEVICES"] = str(plan["execution"]["gpu_index"])
    subprocess.run(
        command,
        cwd=plan["execution"]["repo_root"],
        env=environment,
        check=True,
    )


def run_queue(
    plan: dict[str, Any],
    *,
    only_corpus: str | None = None,
    only_system: str | None = None,
    retrieval_only: bool = False,
) -> None:
    validate_gpu_indices_for_host([int(plan["execution"]["gpu_index"])])
    manifest = Path(plan["manifest"]["path"])
    selected_retrievals = {
        str(step["candidate"])
        for step in plan["steps"]
        if step["kind"] == "retrieve"
        and (only_corpus is None or step["corpus"] == only_corpus)
        and (only_system is None or step["system"] == only_system)
    }
    if (only_corpus is not None or only_system is not None) and not selected_retrievals:
        raise ValueError("lane filter selects no frozen retrieval step")
    for number, step in enumerate(plan["steps"], 1):
        if only_corpus is not None and step["corpus"] != only_corpus:
            continue
        if only_system is not None and step["system"] != only_system:
            continue
        if retrieval_only and (
            step["kind"] not in {"retrieve", "audit"}
            or str(step["candidate"]) not in selected_retrievals
        ):
            continue
        candidate = Path(step["candidate"])
        audit = Path(step["audit"])
        corpus = str(step["corpus"])
        expected_k = int(step["expected_k"])
        valid = _candidate_valid(
            manifest=manifest,
            corpus=corpus,
            candidate=candidate,
            expected_k=expected_k,
        )
        print(
            json.dumps(
                {
                    "step": number,
                    "kind": step["kind"],
                    "corpus": corpus,
                    "system": step["system"],
                    "already_valid": valid,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if step["kind"] == "retrieve":
            if valid:
                continue
            if audit.exists():
                raise ValueError(f"sealed retrieval is inconsistent: {audit}")
            candidate.parent.mkdir(parents=True, exist_ok=True)
            _wait_for_gpu(plan)
            _run_command(plan, step["command"], gpu=True)
            if not _candidate_valid(
                manifest=manifest,
                corpus=corpus,
                candidate=candidate,
                expected_k=expected_k,
            ):
                raise ValueError(f"retrieval did not produce a valid artifact: {candidate}")
        elif step["kind"] in {"project", "union"}:
            if valid:
                continue
            meta = candidate.with_suffix(candidate.suffix + ".meta.json")
            if candidate.exists() or meta.exists() or audit.exists():
                raise ValueError(
                    f"partial/inconsistent CPU projection fails closed: {candidate}"
                )
            if step["kind"] == "union":
                expected = step.get("source_expected_k") or {}
                for source in step.get("source_candidates") or []:
                    if not _candidate_valid(
                        manifest=manifest,
                        corpus=corpus,
                        candidate=Path(source),
                        expected_k=int(expected[str(source)]),
                    ):
                        raise ValueError(f"union source is not audited complete-bank: {source}")
            candidate.parent.mkdir(parents=True, exist_ok=True)
            _run_command(plan, step["command"], gpu=False)
            if not _candidate_valid(
                manifest=manifest,
                corpus=corpus,
                candidate=candidate,
                expected_k=expected_k,
            ):
                raise ValueError(f"projection did not produce a valid artifact: {candidate}")
        else:
            if not valid:
                raise ValueError(f"cannot audit an invalid candidate artifact: {candidate}")
            if _stored_audit_valid(
                audit, candidate=candidate, corpus=corpus, expected_k=expected_k
            ):
                continue
            if audit.exists():
                raise ValueError(f"existing candidate audit is inconsistent: {audit}")
            audit.parent.mkdir(parents=True, exist_ok=True)
            _run_command(plan, step["command"], gpu=False)
            if not _stored_audit_valid(
                audit, candidate=candidate, corpus=corpus, expected_k=expected_k
            ):
                raise ValueError(f"candidate audit did not seal: {audit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly authorize the frozen queue; default is validation only",
    )
    parser.add_argument("--only-corpus")
    parser.add_argument("--only-system")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="run only the selected complete-bank retrieval lane and its audit",
    )
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_NOT_LAUNCHED",
                    "plan": str(plan_path),
                    "plan_sha256": sha256_file(plan_path),
                    "steps": len(plan["steps"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if args.retrieval_only and not (args.only_corpus and args.only_system):
        parser.error("--retrieval-only requires --only-corpus and --only-system")
    run_queue(
        plan,
        only_corpus=args.only_corpus,
        only_system=args.only_system,
        retrieval_only=args.retrieval_only,
    )


if __name__ == "__main__":
    main()
