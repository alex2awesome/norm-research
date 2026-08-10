#!/usr/bin/env python3
"""Materialize an audited, launchable task retrieval queue on a runtime host.

The queue is derived from the all-task rollout, a path-only manifest mirror,
the selected primary retriever, an independent BGE lane, and optional audited
legacy prefixes.  It creates no candidate outputs and launches no GPU work.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from .common import sha256_file
from .freeze_retrieval_queue import adapter_identity, freeze
from .run_frozen_retrieval_queue import validate_plan


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _task_rollout(rollout: dict[str, Any], task: str) -> dict[str, Any]:
    rows = [row for row in rollout.get("tasks") or [] if row.get("task") == task]
    if len(rows) != 1:
        raise ValueError("rollout does not contain exactly one task row")
    return rows[0]


def _fusion(path: Path, task: str, bank_count: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    components = (payload.get("selected") or {}).get("component_weights")
    if (
        payload.get("task") != task
        or payload.get("selection_split") != "dev"
        or int(payload.get("bank_size", -1)) != bank_count
        or not isinstance(components, dict)
        or len(components) != 6
    ):
        raise ValueError(f"invalid task/dev/full-bank fusion: {path}")


def _parse_prefix(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not parts[0]:
        raise ValueError("prefix must be CORPUS=CANDIDATE=AUDIT")
    return parts[0], Path(parts[1]).resolve(), Path(parts[2]).resolve()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rollout_path = Path(args.rollout).resolve()
    canonical_path = Path(args.canonical_manifest).resolve()
    runtime_path = Path(args.runtime_manifest).resolve()
    attestation_path = Path(args.manifest_attestation).resolve()
    selection_source_path = Path(args.source_selection).resolve()
    primary_fusion = Path(args.primary_fusion).resolve()
    bge_fusion = Path(args.bge_fusion).resolve()
    selection_output = Path(args.selection_output).resolve()
    spec_output = Path(args.spec_output).resolve()
    plan_output = Path(args.plan_output).resolve()
    for path in (
        rollout_path,
        canonical_path,
        runtime_path,
        attestation_path,
        selection_source_path,
        primary_fusion,
        bge_fusion,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(path.exists() for path in (selection_output, spec_output, plan_output)):
        raise FileExistsError("one or more queue outputs already exist")

    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    canonical_sha = sha256_file(canonical_path)
    runtime_sha = sha256_file(runtime_path)
    if (
        rollout.get("schema_version") != "silver-match-v3-diverse-retrieval-rollout-v1"
        or rollout.get("canonical_manifest", {}).get("sha256") != canonical_sha
    ):
        raise ValueError("rollout is not bound to the supplied canonical manifest")
    task_row = _task_rollout(rollout, args.task)
    selected_rollout = task_row.get("selected_retriever") or {}
    if selected_rollout.get("sha256") != sha256_file(selection_source_path):
        raise ValueError("selected retriever differs from the all-task rollout")

    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if (
        attestation.get("status") != "FROZEN_PATH_ONLY_RUNTIME_MANIFEST"
        or attestation.get("canonical_manifest", {}).get("sha256") != canonical_sha
        or attestation.get("runtime_manifest", {}).get("sha256") != runtime_sha
        or attestation.get("all_artifact_hashes_equal") is not True
    ):
        raise ValueError("runtime manifest lacks a valid path-only attestation")

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    task_corpora = {
        name
        for name, meta in (runtime.get("corpora") or {}).items()
        if meta.get("task") == args.task
    }
    if task_corpora != set(task_row["corpora"]):
        raise ValueError("runtime task corpus inventory differs from rollout")
    bank_count = int(task_row["bank"]["count"])
    runtime_bank = runtime["banks"][args.task]
    if (
        int(runtime_bank["count"]) != bank_count
        or runtime_bank["source_sha256"] != task_row["bank"]["source_sha256"]
    ):
        raise ValueError("runtime bank differs from rollout")

    selection = json.loads(selection_source_path.read_text(encoding="utf-8"))
    chosen = selection.get("chosen") or {}
    expected_kind = selected_rollout.get("chosen_kind")
    if (
        selection.get("task") != args.task
        or selection.get("selection_split") != "external_dev_only"
        or selection.get("frozen_test_consumed") is not False
        or chosen.get("kind") != expected_kind
        or chosen.get("fusion_report_sha256") != sha256_file(primary_fusion)
        or selected_rollout.get("fusion_sha256") != sha256_file(primary_fusion)
    ):
        raise ValueError("primary selection/fusion differs from rollout")
    _fusion(primary_fusion, args.task, bank_count)
    _fusion(bge_fusion, args.task, bank_count)

    normalized = copy.deepcopy(selection)
    normalized["chosen"]["fusion_report"] = str(primary_fusion)
    normalized["chosen"]["fusion_report_sha256"] = sha256_file(primary_fusion)
    adapter_path = Path(args.primary_adapter).resolve() if args.primary_adapter else None
    if chosen.get("kind") == "adapter":
        if adapter_path is None or not args.adapter_evidence_meta:
            raise ValueError("selected adapter requires adapter path and production evidence")
        evidence_path = Path(args.adapter_evidence_meta).resolve()
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        expected_hashes = evidence.get("adapter_hashes") or {}
        actual_hashes = {
            path.name: sha256_file(path)
            for path in sorted(adapter_path.iterdir())
            if path.is_file()
        }
        if not expected_hashes or actual_hashes != expected_hashes:
            raise ValueError("runtime adapter differs from selected production evidence")
        identity = adapter_identity(adapter_path)
        normalized["chosen"]["adapter_path"] = str(adapter_path)
        normalized["chosen"]["adapter_content_sha256"] = identity["content_sha256"]
        adapter_evidence = _artifact(evidence_path)
    elif adapter_path is not None:
        raise ValueError("base retriever must not receive an adapter")
    else:
        adapter_evidence = None
    normalized["runtime_relocation"] = {
        "source_selection": _artifact(selection_source_path),
        "primary_fusion_bytes_changed": False,
        "primary_fusion_runtime_path": str(primary_fusion),
        "adapter_evidence": adapter_evidence,
        "selection_metrics_or_decision_changed": False,
    }

    prefixes = [_parse_prefix(value) for value in args.prefix]
    if prefixes and {corpus for corpus, _, _ in prefixes} != task_corpora:
        raise ValueError("legacy prefix lane must cover every task corpus")
    existing_lanes: list[dict[str, Any]] = []
    if prefixes:
        candidates = {}
        audits = {}
        for corpus, candidate, audit in prefixes:
            meta_path = candidate.with_suffix(candidate.suffix + ".meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("manifest_sha256") != runtime_sha
                or meta.get("output_sha256") != sha256_file(candidate)
                or int(meta.get("output_k", -1)) != 50
                or not audit.is_file()
            ):
                raise ValueError(f"invalid runtime-bound legacy prefix: {corpus}")
            candidates[corpus] = str(candidate)
            audits[corpus] = str(audit)
        existing_lanes.append(
            {
                "name": "legacy-selected-k50",
                "expected_k": 50,
                "candidates": candidates,
                "audits": audits,
            }
        )

    primary_name = str(chosen["name"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", primary_name):
        raise ValueError("selected primary name is unsafe")
    bge_name = "bge-large-en-v1.5"
    lane_names = {primary_name, bge_name} | (
        {"legacy-selected-k50"} if existing_lanes else set()
    )
    primary_k = min(200, bank_count)
    spec = {
        "task": args.task,
        "manifest": str(runtime_path),
        "selection": str(selection_output),
        "output_root": str(Path(args.output_root).resolve()),
        "cache_root": str(Path(args.cache_root).resolve()),
        "repo_root": str(Path(args.repo_root).resolve()),
        # Preserve an environment's python symlink.  Resolving it to the base
        # interpreter silently drops the environment's site-packages.
        "python": str(Path(args.python).absolute()),
        "gpu_index": args.gpu_index,
        "full_k": bank_count,
        "primary_k": primary_k,
        "systems": [
            {
                "name": primary_name,
                "selection_name": primary_name,
                "role": "primary",
                "encoder": str(Path(args.primary_encoder).resolve()),
                "query_format": "nemotron",
                "fusion": str(primary_fusion),
                **({"adapter": str(adapter_path)} if adapter_path else {}),
            },
            {
                "name": bge_name,
                "role": "diverse",
                "encoder": str(Path(args.bge_encoder).resolve()),
                "query_format": "raw",
                "fusion": str(bge_fusion),
                "query_batch_size": 512,
                "encoder_batch_size": 256,
            },
        ],
        "existing_lanes": existing_lanes,
        "union": {
            "name": "selected-bge-legacy-rrf-k200-v1",
            "output_k": primary_k,
            "rank_constant": 60.0,
            "lane_weights": {name: 1.0 for name in sorted(lane_names)},
        },
        "canonical_provenance": {
            "rollout": _artifact(rollout_path),
            "canonical_manifest": _artifact(canonical_path),
            "runtime_manifest_attestation": _artifact(attestation_path),
        },
    }

    written: list[Path] = []
    try:
        _write_new(selection_output, normalized)
        written.append(selection_output)
        _write_new(spec_output, spec)
        written.append(spec_output)
        plan = freeze(spec_output)
        validate_plan(plan)
        _write_new(plan_output, plan)
        written.append(plan_output)
    except Exception:
        for path in reversed(written):
            path.unlink(missing_ok=True)
        raise
    return {
        "schema_version": "silver-match-v3-runtime-task-retrieval-preparation-v1",
        "status": "FROZEN_LAUNCHABLE_NOT_LAUNCHED",
        "task": args.task,
        "corpus_count": len(task_corpora),
        "norm_count": int(task_row["norm_count"]),
        "full_k": bank_count,
        "primary_k": primary_k,
        "generated_complete_bank_lanes_per_corpus": 2,
        "exact_full_bank_rescue_available_after_retrieval": True,
        "selection": _artifact(selection_output),
        "spec": _artifact(spec_output),
        "plan": _artifact(plan_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--manifest-attestation", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-selection", required=True)
    parser.add_argument("--primary-fusion", required=True)
    parser.add_argument("--bge-fusion", required=True)
    parser.add_argument("--primary-encoder", required=True)
    parser.add_argument("--bge-encoder", required=True)
    parser.add_argument("--primary-adapter")
    parser.add_argument("--adapter-evidence-meta")
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--spec-output", required=True)
    parser.add_argument("--plan-output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
