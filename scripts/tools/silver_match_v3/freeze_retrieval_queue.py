#!/usr/bin/env python3
"""Freeze a hash-pinned, resume-safe full-bank retrieval command queue.

The input spec names one dev-selected primary retriever and one or more diverse
retrievers.  The resulting plan retrieves each system at exact full-bank depth,
    projects the primary full-bank ranking when no union is requested, optionally
    builds a deterministic multi-lane union, and audits every materialized artifact.  This module only
freezes commands; it never launches a GPU job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .common import sha256_file


COMPONENTS = {
    "dense_rank",
    "dense_statement_rank",
    "word_rank",
    "word_statement_rank",
    "char_rank",
    "char_statement_rank",
}
PRESERVABLE_COMPONENTS = COMPONENTS | {"rank"}


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def encoder_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    revision = path.name
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError(f"encoder is not pinned to an immutable snapshot revision: {path}")
    identity_files: dict[str, dict[str, Any]] = {}
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        relative = str(candidate.relative_to(path))
        identity_files[relative] = {
            "sha256": sha256_file(candidate),
            "bytes": candidate.stat().st_size,
        }
    if not identity_files:
        raise ValueError(f"encoder snapshot has no hashable identity files: {path}")
    return {
        "path": str(path),
        "snapshot_revision": revision,
        "identity_files": identity_files,
    }


def adapter_identity(path: Path) -> dict[str, Any]:
    """Bind every file in a small task-local adapter, including its weights."""
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    identity_files = {
        str(candidate.relative_to(path)): {
            "sha256": sha256_file(candidate),
            "bytes": candidate.stat().st_size,
        }
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
    }
    if not identity_files:
        raise ValueError(f"adapter has no files: {path}")
    content_sha256 = hashlib.sha256(
        json.dumps(identity_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "path": str(path),
        "identity_files": identity_files,
        "content_sha256": content_sha256,
    }


def _fusion(path: Path, *, task: str, bank_size: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = (payload.get("selected") or {}).get("component_weights")
    if (
        payload.get("task") != task
        or payload.get("selection_split") != "dev"
        or int(payload.get("bank_size", -1)) != bank_size
        or not isinstance(weights, dict)
        or set(weights) != COMPONENTS
        or not any(float(value) > 0 for value in weights.values())
    ):
        raise ValueError(f"invalid task/dev/full-bank fusion: {path}")
    return payload


def _retrieve_command(
    *,
    python: str,
    corpus: str,
    manifest: str,
    output: str,
    system: dict[str, Any],
    k: int,
    cache_root: str,
) -> list[str]:
    command = [
        python,
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.retrieve",
        corpus,
        "--manifest",
        manifest,
        "--output",
        output,
        "--encoder",
        system["encoder"]["path"],
        "--query-format",
        system["query_format"],
        "--device",
        "cuda",
        "--attention",
        system.get("attention", "eager"),
        "--fusion-weights",
        system["fusion"]["path"],
        "--no-reranker",
        "--cache-dir",
        str(Path(cache_root) / "huggingface"),
        "--dense-k",
        str(k),
        "--word-k",
        str(k),
        "--char-k",
        str(k),
        "--pre-rerank-k",
        str(k),
        "--output-k",
        str(k),
        "--query-batch-size",
        str(system.get("query_batch_size", 256)),
        "--encoder-batch-size",
        str(system.get("encoder_batch_size", 128)),
        "--metric-batch-size",
        str(system.get("metric_batch_size", min(k, 128))),
        "--resume",
    ]
    if system.get("adapter"):
        command.extend(["--adapter", system["adapter"]["path"]])
    return command


def _union_command(
    *,
    python: str,
    manifest: str,
    corpus: str,
    lanes: list[tuple[str, str, float]],
    output: str,
    output_k: int,
    rank_constant: float,
    preserve_components: dict[str, list[str]] | None = None,
    preserve_k: int | None = None,
    prefix_lanes: list[str] | None = None,
) -> list[str]:
    command = [
        python,
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.materialize_retrieval_lane_union",
        "--manifest",
        manifest,
        "--corpus",
        corpus,
        "--output",
        output,
        "--output-k",
        str(output_k),
        "--rank-constant",
        str(rank_constant),
    ]
    for name, path, weight in lanes:
        command.extend(["--lane", f"{name}={path}={weight}"])
    for name, components in sorted((preserve_components or {}).items()):
        for component in components:
            command.extend(["--preserve-component", f"{name}={component}"])
    if preserve_k is not None:
        command.extend(["--preserve-k", str(preserve_k)])
    for name in sorted(prefix_lanes or []):
        command.extend(["--prefix-lane", name])
    return command


def _audit_command(
    *, python: str, manifest: str, corpus: str, candidate: str, k: int, output: str
) -> list[str]:
    return [
        python,
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.audit_candidate_outputs",
        "--manifest",
        manifest,
        "--corpus",
        corpus,
        "--candidates",
        candidate,
        "--expected-k",
        str(k),
        "--output",
        output,
    ]


def freeze(spec_path: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest_path = _resolve(spec["manifest"], spec_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = str(spec["task"])
    if task not in manifest.get("banks", {}):
        raise KeyError(task)
    bank_meta = manifest["banks"][task]
    bank_size = int(bank_meta["count"])
    bank_path = _resolve(bank_meta["path"], manifest_path)
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_ids = [str(row.get("metric_id") or "") for row in bank_payload.get("metrics") or []]
    if (
        len(bank_ids) != bank_size
        or "" in bank_ids
        or len(set(bank_ids)) != bank_size
    ):
        raise ValueError("manifest bank count/metric universe mismatch")
    if int(spec["full_k"]) != bank_size:
        raise ValueError("full_k must equal the exact task bank size")
    primary_k = int(spec["primary_k"])
    if not 1 <= primary_k <= bank_size:
        raise ValueError("primary_k is outside the bank")

    selection_path = _resolve(spec["selection"], spec_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("chosen") or {}
    if (
        selection.get("task") != task
        or selection.get("selection_split") != "external_dev_only"
        or selection.get("frozen_test_consumed") is not False
        or chosen.get("kind") not in {"nemotron_base", "adapter"}
    ):
        raise ValueError("selection is not an unconsumed external-dev retriever choice")

    systems = spec.get("systems") or []
    if len(systems) < 2 or sum(value.get("role") == "primary" for value in systems) != 1:
        raise ValueError("exactly one primary and at least one diverse system are required")
    if any(value.get("role") not in {"primary", "diverse"} for value in systems):
        raise ValueError("retrieval system roles must be primary or diverse")
    frozen_systems = []
    names: set[str] = set()
    encoder_identities: dict[Path, dict[str, Any]] = {}
    adapter_identities: dict[Path, dict[str, Any]] = {}
    for raw in systems:
        name = str(raw["name"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names:
            raise ValueError(f"missing/duplicate system name: {name!r}")
        names.add(name)
        if raw.get("query_format") not in {"auto", "raw", "nemotron"}:
            raise ValueError(f"invalid query format for system: {name}")
        fusion_path = _resolve(raw["fusion"], spec_path)
        _fusion(fusion_path, task=task, bank_size=bank_size)
        query_batch_size = int(raw.get("query_batch_size", 256))
        encoder_batch_size = int(raw.get("encoder_batch_size", 128))
        metric_batch_size = int(raw.get("metric_batch_size", min(bank_size, 128)))
        if min(query_batch_size, encoder_batch_size, metric_batch_size) < 1:
            raise ValueError(f"batch sizes must be positive: {name}")
        encoder_path = _resolve(raw["encoder"], spec_path)
        if encoder_path not in encoder_identities:
            encoder_identities[encoder_path] = encoder_identity(encoder_path)
        adapter_path = _resolve(raw["adapter"], spec_path) if raw.get("adapter") else None
        if adapter_path is not None and adapter_path not in adapter_identities:
            adapter_identities[adapter_path] = adapter_identity(adapter_path)
        frozen_systems.append(
            {
                "name": name,
                "selection_name": raw.get("selection_name", name),
                "role": raw["role"],
                "query_format": raw["query_format"],
                "attention": raw.get("attention", "eager"),
                "query_batch_size": query_batch_size,
                "encoder_batch_size": encoder_batch_size,
                "metric_batch_size": metric_batch_size,
                "encoder": encoder_identities[encoder_path],
                "adapter": adapter_identities.get(adapter_path),
                "fusion": artifact(fusion_path),
            }
        )
    primary = next(value for value in frozen_systems if value["role"] == "primary")
    if (
        Path(str(chosen.get("fusion_report") or "")).resolve()
        != Path(primary["fusion"]["path"])
        or chosen.get("fusion_report_sha256") != primary["fusion"]["sha256"]
        or chosen.get("name") != primary["selection_name"]
    ):
        raise ValueError("primary system differs from the frozen retriever selection")
    if (chosen.get("kind") == "adapter") != bool(primary.get("adapter")):
        raise ValueError("primary adapter presence differs from frozen selection kind")
    if primary.get("adapter") and (
        Path(str(chosen.get("adapter_path") or "")).resolve()
        != Path(primary["adapter"]["path"])
        or chosen.get("adapter_content_sha256")
        != primary["adapter"]["content_sha256"]
    ):
        raise ValueError("primary adapter differs from the frozen retriever selection")
    primary_signature = (
        primary["encoder"]["snapshot_revision"],
        primary["fusion"]["sha256"],
        json.dumps(primary.get("adapter"), sort_keys=True),
    )
    if any(
        (
            value["encoder"]["snapshot_revision"],
            value["fusion"]["sha256"],
            json.dumps(value.get("adapter"), sort_keys=True),
        )
        == primary_signature
        for value in frozen_systems
        if value["role"] != "primary"
    ):
        raise ValueError("a diverse system duplicates the primary retrieval geometry")

    repo_root = _resolve(spec["repo_root"], spec_path)
    implementations = {
        name: artifact(repo_root / "scripts/tools/silver_match_v3" / name)
        for name in (
            "retrieve.py",
            "audit_candidate_outputs.py",
            "truncate_candidate_depth.py",
            "materialize_retrieval_lane_union.py",
            "run_frozen_retrieval_queue.py",
        )
    }
    python_value = Path(spec["python"])
    python_path = (
        python_value.absolute()
        if python_value.is_absolute()
        else (spec_path.parent / python_value).absolute()
    )
    python = str(python_path)
    if not python_path.exists():
        raise FileNotFoundError(python_path)
    output_root = _resolve(spec["output_root"], spec_path)
    cache_root = _resolve(spec.get("cache_root", output_root.parent / "cache"), spec_path)
    corpora = sorted(
        corpus for corpus, meta in manifest["corpora"].items() if meta["task"] == task
    )
    if not corpora:
        raise ValueError(f"task has no corpora: {task}")
    corpus_inputs = {}
    for corpus in corpora:
        meta = manifest["corpora"][corpus]
        canonical = _resolve(meta["path"], manifest_path)
        corpus_inputs[corpus] = {
            "count": int(meta["count"]),
            "canonical": artifact(canonical),
        }

    frozen_existing_lanes: list[dict[str, Any]] = []
    for raw in spec.get("existing_lanes") or []:
        name = str(raw.get("name") or "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names:
            raise ValueError(f"invalid/duplicate existing lane name: {name!r}")
        expected_k = int(raw.get("expected_k", -1))
        if not 1 <= expected_k < bank_size:
            raise ValueError("existing lane must be a proper bank prefix")
        candidates = raw.get("candidates") or {}
        if not isinstance(candidates, dict) or set(candidates) != set(corpora):
            raise ValueError("existing lane must cover every task corpus exactly")
        frozen_candidates = {}
        for corpus in corpora:
            candidate = _resolve(candidates[corpus], spec_path)
            meta = candidate.with_suffix(candidate.suffix + ".meta.json")
            audit_path = Path(str(candidate) + ".audit.json")
            if not audit_path.exists() and raw.get("audits"):
                audit_path = _resolve(raw["audits"][corpus], spec_path)
            frozen_candidates[corpus] = {
                "candidate": artifact(candidate),
                "meta": artifact(meta),
                "audit": artifact(audit_path),
            }
        frozen_existing_lanes.append(
            {
                "name": name,
                "expected_k": expected_k,
                "candidates": frozen_candidates,
            }
        )
        names.add(name)

    union_raw = spec.get("union")
    frozen_union: dict[str, Any] | None = None
    if union_raw is not None:
        if not isinstance(union_raw, dict):
            raise ValueError("union must be an object")
        union_name = str(union_raw.get("name") or "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", union_name):
            raise ValueError("union name is missing or unsafe")
        union_k = int(union_raw.get("output_k", primary_k))
        if not 1 <= union_k <= bank_size:
            raise ValueError("union output_k is outside the bank")
        rank_constant = float(union_raw.get("rank_constant", 60.0))
        if rank_constant <= 0:
            raise ValueError("union rank_constant must be positive")
        raw_weights = union_raw.get("lane_weights")
        if not isinstance(raw_weights, dict) or set(raw_weights) != names:
            raise ValueError("union lane_weights must exactly name every frozen system")
        lane_weights = {name: float(raw_weights[name]) for name in sorted(names)}
        if any(value <= 0 for value in lane_weights.values()):
            raise ValueError("union lane weights must be positive")
        raw_preserve = union_raw.get("preserve_components") or {}
        if not isinstance(raw_preserve, dict):
            raise ValueError("union preserve_components must be an object")
        preserve_components = {
            str(name): [str(component) for component in components]
            for name, components in raw_preserve.items()
            if isinstance(components, list)
        }
        if len(preserve_components) != len(raw_preserve):
            raise ValueError("union preserve component lists are invalid")
        preserve_k = union_raw.get("preserve_k")
        if preserve_components:
            if preserve_k is None:
                raise ValueError("union preserve_k is required")
            preserve_k = int(preserve_k)
            if (
                set(preserve_components) != names
                or not 1 <= preserve_k <= bank_size
                or union_k != bank_size
                or any(
                    not components
                    or len(set(components)) != len(components)
                    or not set(components) <= PRESERVABLE_COMPONENTS
                    for components in preserve_components.values()
                )
            ):
                raise ValueError(
                    "component-prefix preservation requires every system and full-bank output"
                )
            algorithm = "coverage-preserving-component-prefix-rrf-v1"
        elif preserve_k is not None:
            raise ValueError("union preserve_k requires preserve_components")
        else:
            algorithm = "weighted-complete-bank-rrf-v1"
        frozen_union = {
            "name": union_name,
            "output_k": union_k,
            "rank_constant": rank_constant,
            "lane_weights": lane_weights,
            "algorithm": algorithm,
            "preserve_components": preserve_components,
            "preserve_k": preserve_k,
        }

    steps = []
    full_by_corpus: dict[str, dict[str, Path]] = {corpus: {} for corpus in corpora}
    # Primary first for every corpus, so production K50 is available before
    # spending time on capture/recapture diversity lanes.
    ordered_systems = sorted(frozen_systems, key=lambda value: value["role"] != "primary")
    for system in ordered_systems:
        for corpus in corpora:
            full = output_root / "full_bank" / f"{corpus}.full{bank_size}.{system['name']}.jsonl"
            full_by_corpus[corpus][system["name"]] = full
            full_audit = Path(str(full) + ".audit.json")
            steps.append(
                {
                    "kind": "retrieve",
                    "system": system["name"],
                    "corpus": corpus,
                    "expected_k": bank_size,
                    "candidate": str(full),
                    "audit": str(full_audit),
                    "command": _retrieve_command(
                        python=python,
                        corpus=corpus,
                        manifest=str(manifest_path),
                        output=str(full),
                        system=system,
                        k=bank_size,
                        cache_root=str(cache_root),
                    ),
                }
            )
            steps.append(
                {
                    "kind": "audit",
                    "system": system["name"],
                    "corpus": corpus,
                    "expected_k": bank_size,
                    "candidate": str(full),
                    "audit": str(full_audit),
                    "command": _audit_command(
                        python=python,
                        manifest=str(manifest_path),
                        corpus=corpus,
                        candidate=str(full),
                        k=bank_size,
                        output=str(full_audit),
                    ),
                }
            )
            # A multi-lane union is itself the production primary-K artifact.
            # Materializing an additional direct-primary projection would copy
            # tens of gigabytes without adding retrieval coverage.
            if system["role"] == "primary" and frozen_union is None:
                topk = output_root / "candidates" / f"{corpus}.primary.{system['name']}.jsonl"
                topk_audit = Path(str(topk) + ".audit.json")
                steps.append(
                    {
                        "kind": "project",
                        "system": system["name"],
                        "corpus": corpus,
                        "expected_k": primary_k,
                        "candidate": str(topk),
                        "audit": str(topk_audit),
                        "source_candidate": str(full),
                        "command": [
                            python,
                            "-u",
                            "-m",
                            "scripts.tools.silver_match_v3.truncate_candidate_depth",
                            "--input",
                            str(full),
                            "--output",
                            str(topk),
                            "--output-k",
                            str(primary_k),
                        ],
                    }
                )
                steps.append(
                    {
                        "kind": "audit",
                        "system": system["name"],
                        "corpus": corpus,
                        "expected_k": primary_k,
                        "candidate": str(topk),
                        "audit": str(topk_audit),
                        "command": _audit_command(
                            python=python,
                            manifest=str(manifest_path),
                            corpus=corpus,
                            candidate=str(topk),
                            k=primary_k,
                            output=str(topk_audit),
                        ),
                    }
                )

    if frozen_union is not None:
        for corpus in corpora:
            union_output = (
                output_root
                / "candidates"
                / f"{corpus}.union.{frozen_union['name']}.jsonl"
            )
            union_audit = Path(str(union_output) + ".audit.json")
            lanes = [
                (
                    system["name"],
                    str(full_by_corpus[corpus][system["name"]]),
                    frozen_union["lane_weights"][system["name"]],
                )
                for system in ordered_systems
            ]
            lanes.extend(
                (
                    lane["name"],
                    lane["candidates"][corpus]["candidate"]["path"],
                    frozen_union["lane_weights"][lane["name"]],
                )
                for lane in frozen_existing_lanes
            )
            source_expected_k = {
                path: (
                    next(
                        (
                            lane["expected_k"]
                            for lane in frozen_existing_lanes
                            if lane["name"] == name
                        ),
                        bank_size,
                    )
                )
                for name, path, _ in lanes
            }
            steps.append(
                {
                    "kind": "union",
                    "system": frozen_union["name"],
                    "corpus": corpus,
                    "expected_k": frozen_union["output_k"],
                    "candidate": str(union_output),
                    "audit": str(union_audit),
                    "source_candidates": [path for _, path, _ in lanes],
                    "source_expected_k": source_expected_k,
                    "command": _union_command(
                        python=python,
                        manifest=str(manifest_path),
                        corpus=corpus,
                        lanes=lanes,
                        output=str(union_output),
                        output_k=frozen_union["output_k"],
                        rank_constant=frozen_union["rank_constant"],
                        preserve_components=frozen_union["preserve_components"],
                        preserve_k=frozen_union["preserve_k"],
                        prefix_lanes=[lane["name"] for lane in frozen_existing_lanes],
                    ),
                }
            )
            steps.append(
                {
                    "kind": "audit",
                    "system": frozen_union["name"],
                    "corpus": corpus,
                    "expected_k": frozen_union["output_k"],
                    "candidate": str(union_output),
                    "audit": str(union_audit),
                    "command": _audit_command(
                        python=python,
                        manifest=str(manifest_path),
                        corpus=corpus,
                        candidate=str(union_output),
                        k=frozen_union["output_k"],
                        output=str(union_audit),
                    ),
                }
            )
    return {
        "schema_version": "silver-match-v3-retrieval-command-queue-v1",
        "status": "FROZEN_NOT_LAUNCHED",
        "release_ready": False,
        "task": task,
        "spec": artifact(spec_path),
        "manifest": artifact(manifest_path),
        "bank": {
            **artifact(bank_path),
            "source_sha256": bank_meta["source_sha256"],
            "count": bank_size,
        },
        "selection": artifact(selection_path),
        "primary_k": primary_k,
        "full_k": bank_size,
        "union": frozen_union,
        "corpora": corpus_inputs,
        "coverage_contract": {
            "scope": "all-manifest-corpora-for-task",
            "corpus_count": len(corpora),
            "norm_count": sum(value["count"] for value in corpus_inputs.values()),
            "one_exact_candidate_row_per_norm_required": True,
            "diagnostic_subset_reuse_forbidden": True,
        },
        "systems": frozen_systems,
        "existing_lanes": frozen_existing_lanes,
        "implementations": implementations,
        "execution": {
            "repo_root": str(repo_root),
            "python": python,
            "gpu_index": int(spec["gpu_index"]),
            "gpu_count_gate_applied": False,
            "projected_owner_count_check_applied": False,
            "uses_batched_encoder_inference": True,
            "uses_openai_server": False,
            "poll_seconds": int(spec.get("poll_seconds", 30)),
            "prerequisite": spec.get("prerequisite"),
            "cache_root": str(cache_root),
        },
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = freeze(Path(args.spec))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": artifact(output), "task": plan["task"], "steps": len(plan["steps"])}))


if __name__ == "__main__":
    main()
