#!/usr/bin/env python3
"""Freeze a leakage-safe, spend-bounded task-local GEPA API experiment.

This is a planning/freeze step, not an inference runner.  It projects a
previously frozen train-only human panel and retriever candidates into
source-group-disjoint prompt-train/prompt-dev cells, seals all permanent
exclusions and prompt variants, and writes machine-readable bounded commands
for Gemma adjudication and verification.  No test, blind-audit, production, MI,
or outcome row can enter the plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl, sha256_file, write_jsonl
from .config import GEMMA4
from .make_calibration import split_for, split_group_for


ROLE_TO_PANEL = {"train": "prompt_train", "dev": "prompt_dev"}
ORDERS = ("original", "hashed")
VERIFIER_ORDERS = ("original", "hashed", "reverse")


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return indexed


def _load_task_norms(manifest: dict[str, Any], task: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for _, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != task:
            continue
        path = Path(meta["path"]).resolve()
        for uid, row in _index(path).items():
            if uid in output:
                raise ValueError(f"duplicate canonical task norm UID: {uid}")
            output[uid] = row
    if not output:
        raise ValueError(f"manifest contains no canonical norms for task {task}")
    return output


def _parse_variants(specs: Iterable[str], *, kind: str) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    names: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{kind} variant must be NAME=PATH[,PATH...]: {spec}")
        name, raw_paths = spec.split("=", 1)
        name = name.strip()
        if not name or name in names or not name.replace("-", "_").isalnum():
            raise ValueError(f"invalid or duplicate {kind} variant name: {name!r}")
        paths = [Path(value).resolve() for value in raw_paths.split(",") if value]
        if not paths or any(not path.is_file() for path in paths):
            raise ValueError(f"{kind} variant has a missing prompt component: {spec}")
        text = "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in paths) + "\n"
        variants.append(
            {
                "name": name,
                "components": [
                    {"path": str(path), "sha256": sha256_file(path)} for path in paths
                ],
                "combined_prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
        names.add(name)
    if not variants:
        raise ValueError(f"at least one {kind} prompt variant is required")
    return variants


def _command(module: str, *args: object) -> dict[str, Any]:
    return {"module": module, "argv": [str(value) for value in args]}


def _variant_prompt_args(variant: dict[str, Any]) -> list[str]:
    components = [row["path"] for row in variant["components"]]
    output = ["--prompt", components[0]]
    for path in components[1:]:
        output.extend(["--prompt-addon", path])
    return output


def _validate_truth_row(row: dict[str, Any], bank_ids: set[str]) -> None:
    decision = str(row.get("decision") or "")
    metric_id = row.get("metric_id")
    if decision == "MATCH":
        if str(metric_id) not in bank_ids:
            raise ValueError(
                f"MATCH truth uses metric outside frozen bank: {row.get('norm_uid')}/{metric_id}"
            )
    elif metric_id is not None:
        raise ValueError(
            f"non-MATCH truth must have metric_id null: {row.get('norm_uid')}"
        )


def _validate_predeclaration(
    *,
    path: Path,
    args: argparse.Namespace,
    adjudicator_variants: list[dict[str, Any]],
    verifier_variants: list[dict[str, Any]],
    panel: dict[str, dict[str, Any]],
    exclusion_paths: list[Path],
    excluded_uids: set[str],
) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "silver-match-v3-task-local-gepa-predeclaration-v1":
        raise ValueError("unsupported task-local GEPA predeclaration schema")
    task_lock = (lock.get("tasks") or {}).get(args.task)
    if not task_lock:
        raise ValueError(f"predeclaration does not contain task {args.task}")
    split = lock["split"]
    gate = lock["selection_gate"]
    api = lock["api"]
    direct = lock["direct_batch"]
    exact = {
        "candidate_k": (args.candidate_k, lock["candidate_k"]),
        "minimum_train": (args.minimum_train, split["minimum_prompt_train_rows"]),
        "minimum_dev": (args.minimum_dev, split["minimum_prompt_dev_rows"]),
        "minimum_point_precision": (
            args.minimum_point_precision,
            gate["minimum_point_precision"],
        ),
        "minimum_wilson_lower": (
            args.minimum_wilson_lower,
            gate["minimum_wilson_95_lower"],
        ),
        "minimum_retained": (args.minimum_retained, gate["minimum_retained_support"]),
        "api_base_url": (args.api_base_url, api["base_url"]),
        "model": (args.model, api["model"]),
        "direct_model": (args.direct_model, direct["model"]),
        "direct_batch_size": (args.direct_batch_size, direct["batch_size"]),
        "gpu_memory_utilization": (
            args.gpu_memory_utilization,
            direct["gpu_memory_utilization"],
        ),
    }
    mismatches = {
        key: {"requested": left, "predeclared": right}
        for key, (left, right) in exact.items()
        if left != right
    }
    if mismatches:
        raise ValueError(f"run differs from predeclaration: {mismatches}")
    if args.max_total_api_requests > int(api["maximum_total_logical_requests_per_task"]):
        raise ValueError("requested API cap exceeds the predeclared task ceiling")
    if int(api.get("implicit_transport_retries", -1)) != 0:
        raise ValueError("predeclaration must disable implicit transport retries")

    expected_seed = int(split["gepa_seed"])
    expected_dev = int(split["gepa_dev_percent"])
    for uid, row in panel.items():
        if int(row.get("gepa_split_seed", -1)) != expected_seed or int(
            row.get("gepa_dev_percent", -1)
        ) != expected_dev:
            raise ValueError(f"panel split parameters differ from predeclaration: {uid}")

    for kind, actual in (
        ("adjudicator_variants", adjudicator_variants),
        ("verifier_variants", verifier_variants),
    ):
        expected = [
            (str(row["name"]), str(row["combined_prompt_sha256"]))
            for row in task_lock[kind]
        ]
        observed = [
            (str(row["name"]), str(row["combined_prompt_sha256"])) for row in actual
        ]
        if observed != expected:
            raise ValueError(
                f"{kind} differ from frozen predeclaration: "
                f"observed={observed} expected={expected}"
            )

    # A task may name mandatory historical universes whose complete UID sets
    # must be represented by the supplied exclusion union. This makes Math's
    # failed r5 experiment impossible to accidentally recycle.
    root = path.parents[4]
    mandatory = (
        (task_lock.get("sealed_failed_experiment") or {}).get(
            "mandatory_exclusion_universes"
        )
        or []
    )
    mandatory_details = []
    supplied_paths = {item.resolve() for item in exclusion_paths}
    for row in mandatory:
        source = (root / str(row["path"])).resolve()
        if not source.is_file() or sha256_file(source) != str(row["sha256"]):
            raise ValueError(f"mandatory sealed exclusion artifact is missing or changed: {source}")
        indexed = _index(source)
        if int(row["count"]) != len(indexed):
            raise ValueError(f"mandatory sealed exclusion count changed: {source}")
        missing = sorted(set(indexed) - excluded_uids)
        if missing:
            raise ValueError(
                f"supplied exclusion union omits mandatory sealed UIDs: {source}: {missing[:3]}"
            )
        mandatory_details.append(
            {
                "path": str(source),
                "sha256": sha256_file(source),
                "uid_count": len(indexed),
                "supplied_directly": source in supplied_paths,
                "missing_from_exclusion_union": 0,
            }
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": lock.get("status"),
        "mandatory_sealed_exclusions": mandatory_details,
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    if args.minimum_train < 1 or args.minimum_dev < 1:
        raise ValueError("minimum train/dev support must be positive")
    if not 0.0 < args.minimum_point_precision <= 1.0:
        raise ValueError("minimum point precision must be in (0, 1]")
    if not 0.0 < args.minimum_wilson_lower <= 1.0:
        raise ValueError("minimum Wilson lower bound must be in (0, 1]")
    if args.minimum_retained < 1 or args.max_total_api_requests < 1:
        raise ValueError("retained support and API request budget must be positive")

    manifest_path = Path(args.manifest).resolve()
    panel_path = Path(args.panel).resolve()
    candidates_path = Path(args.candidates).resolve()
    exclusion_paths = [Path(value).resolve() for value in args.exclude_reference]
    output_root = Path(args.output_root).resolve()
    if not exclusion_paths:
        raise ValueError("at least one permanent --exclude-reference is required")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite GEPA freeze: {output_root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    norms = _load_task_norms(manifest, args.task)
    panel = _index(panel_path)
    candidates = _index(candidates_path)
    bank_meta = (manifest.get("banks") or {}).get(args.task)
    if not bank_meta:
        raise ValueError(f"manifest lacks frozen bank for task {args.task}")
    bank_path = Path(bank_meta["path"]).resolve()
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_rows = bank_payload.get("metrics") or bank_payload.get("bank") or []
    bank_ids = {str(row.get("metric_id") or "") for row in bank_rows}
    if not bank_ids or "" in bank_ids or len(bank_ids) != len(bank_rows):
        raise ValueError("frozen bank has missing/duplicate metric IDs")

    missing_norms = sorted(set(panel) - set(norms))
    if missing_norms:
        raise ValueError(f"panel UIDs absent from canonical task norms: {missing_norms[:3]}")
    missing_candidates = sorted(set(panel) - set(candidates))
    if missing_candidates:
        raise ValueError(f"retriever candidates do not cover panel: {missing_candidates[:3]}")

    excluded_uids: set[str] = set()
    exclusion_details: dict[str, Any] = {}
    for path in exclusion_paths:
        indexed = _index(path)
        missing = sorted(set(indexed) - set(norms))
        if missing:
            raise ValueError(f"exclusion UIDs absent from canonical task norms: {path}: {missing[:3]}")
        excluded_uids.update(indexed)
        exclusion_details[str(path)] = {
            "sha256": sha256_file(path),
            "uid_count": len(indexed),
        }
    excluded_groups = {split_group_for(norms[uid]) for uid in excluded_uids}

    rows_by_role: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    candidate_by_role: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    groups_by_role: dict[str, set[str]] = {"train": set(), "dev": set()}
    bank_hashes: set[str] = set()
    for uid, truth in sorted(panel.items()):
        if truth.get("task") != args.task:
            raise ValueError(f"panel row has wrong task: {uid}/{truth.get('task')}")
        if str(truth.get("predeclared_split")) != "train":
            raise ValueError(f"GEPA row is not upstream train-only: {uid}")
        role = str(truth.get("split") or "")
        if role not in rows_by_role:
            raise ValueError(f"GEPA row has non-train/dev local role: {uid}/{role}")
        group = split_group_for(norms[uid])
        if split_for(group) != "train":
            raise ValueError(f"canonical source group is not upstream train-only: {uid}")
        if group in excluded_groups:
            raise ValueError(f"GEPA panel overlaps permanently excluded source group: {uid}")
        _validate_truth_row(truth, bank_ids)
        candidate = candidates[uid]
        if candidate.get("task") != args.task:
            raise ValueError(f"candidate row has wrong task: {uid}")
        values = list(candidate.get("candidates") or [])
        ids = [str(value.get("metric_id") or "") for value in values]
        if len(values) < args.candidate_k or "" in ids or len(ids) != len(set(ids)):
            raise ValueError(f"candidate row is invalid or shorter than K={args.candidate_k}: {uid}")
        if not set(ids).issubset(bank_ids):
            raise ValueError(f"candidate row references a metric outside the frozen bank: {uid}")
        bank_hash = str(candidate.get("bank_source_sha256") or "")
        if not bank_hash:
            raise ValueError(f"candidate row lacks bank_source_sha256: {uid}")
        bank_hashes.add(bank_hash)
        groups_by_role[role].add(group)
        rows_by_role[role].append({**truth, "gepa_source_group": group})
        candidate_by_role[role].append(
            {**candidate, "candidates": values[: args.candidate_k], "gepa_source_group": group}
        )
    if groups_by_role["train"] & groups_by_role["dev"]:
        raise ValueError("prompt-train and prompt-dev overlap by canonical source group")
    if len(rows_by_role["train"]) < args.minimum_train or len(rows_by_role["dev"]) < args.minimum_dev:
        raise ValueError(
            "GEPA panel is underpowered for predeclared minimum support: "
            f"train={len(rows_by_role['train'])}/{args.minimum_train}, "
            f"dev={len(rows_by_role['dev'])}/{args.minimum_dev}"
        )
    if len(bank_hashes) != 1:
        raise ValueError(f"candidate rows do not share one frozen bank hash: {sorted(bank_hashes)}")

    adjudicator_variants = _parse_variants(
        args.adjudicator_variant, kind="adjudicator"
    )
    verifier_variants = _parse_variants(args.verifier_variant, kind="verifier")
    predeclaration = _validate_predeclaration(
        path=Path(args.predeclaration).resolve(),
        args=args,
        adjudicator_variants=adjudicator_variants,
        verifier_variants=verifier_variants,
        panel=panel,
        exclusion_paths=exclusion_paths,
        excluded_uids=excluded_uids,
    )

    # Each command permits at most one JSON-contract retry per row.  Verifier
    # proposal subsets can only be smaller, but use the full role size for a
    # conservative immutable upper bound.
    adjudicator_requests = (
        sum(2 * len(rows_by_role[role]) for role in ROLE_TO_PANEL for _ in ORDERS)
        * len(adjudicator_variants)
    )
    verifier_requests = (
        sum(2 * len(rows_by_role[role]) for role in ROLE_TO_PANEL for _ in VERIFIER_ORDERS)
        * len(verifier_variants)
        * len(adjudicator_variants)
    )
    maximum_requests = adjudicator_requests + verifier_requests
    if maximum_requests > args.max_total_api_requests:
        raise ValueError(
            f"predeclared API plan exceeds budget: {maximum_requests} > "
            f"{args.max_total_api_requests}"
        )

    output_root.mkdir(parents=True, exist_ok=False)
    role_paths: dict[str, dict[str, Path]] = {}
    for role in ROLE_TO_PANEL:
        role_dir = output_root / "panel" / role
        truth_path = role_dir / "truth.jsonl"
        candidate_path = role_dir / f"candidates.top{args.candidate_k}.jsonl"
        write_jsonl(truth_path, rows_by_role[role])
        write_jsonl(candidate_path, candidate_by_role[role])
        role_paths[role] = {"truth": truth_path, "candidates": candidate_path}

    write_jsonl(
        output_root / "exclusions.identity.jsonl",
        (
            {
                "norm_uid": uid,
                "source_group": split_group_for(norms[uid]),
                "permanently_excluded_from_prompt_selection": True,
                "permanently_excluded_from_gradients": True,
            }
            for uid in sorted(excluded_uids)
        ),
    )

    stages: list[dict[str, Any]] = []
    for variant in adjudicator_variants:
        for role, panel_role in ROLE_TO_PANEL.items():
            outputs: dict[str, str] = {}
            for order in ORDERS:
                output = output_root / "runs" / "adjudicator" / variant["name"] / role / f"{order}.jsonl"
                outputs[order] = str(output)
                cap = 2 * len(rows_by_role[role])
                stages.append(
                    {
                        "stage": "adjudicator",
                        "variant": variant["name"],
                        "role": panel_role,
                        "order": order,
                        "maximum_api_requests": cap,
                        "command": _command(
                            "scripts.tools.silver_match_v3.adjudicate_gemma_api",
                            "--manifest", manifest_path,
                            "--candidates", role_paths[role]["candidates"],
                            "--output", output,
                            "--split-role", role,
                            *_variant_prompt_args(variant),
                            "--api-base-url", args.api_base_url,
                            "--api-key-file", args.api_key_file,
                            "--max-api-requests", cap,
                            "--model", args.model,
                            "--max-candidates", args.candidate_k,
                            "--concurrency", args.concurrency,
                            "--transport-retries", 0,
                            "--order-mode", order,
                            "--resume",
                        ),
                        "direct_batch_command": _command(
                            "scripts.tools.silver_match_v3.adjudicate_gemma",
                            "--manifest", manifest_path,
                            "--candidates", role_paths[role]["candidates"],
                            "--output", output,
                            *_variant_prompt_args(variant),
                            "--model", args.direct_model,
                            "--max-candidates", args.candidate_k,
                            "--batch-size", args.direct_batch_size,
                            "--gpu-memory-utilization", args.gpu_memory_utilization,
                            "--order-mode", order,
                            "--resume",
                        ),
                    }
                )
            consensus = output_root / "runs" / "adjudicator" / variant["name"] / role / "two_order_consensus.jsonl"
            stages.append(
                {
                    "stage": "adjudicator_consensus",
                    "variant": variant["name"],
                    "role": panel_role,
                    "command": _command(
                        "scripts.tools.silver_match_v3.build_two_order_consensus_proposals",
                        "--original", outputs["original"],
                        "--hashed", outputs["hashed"],
                        "--task", args.task,
                        "--output", consensus,
                    ),
                }
            )
            stages.append(
                {
                    "stage": "adjudicator_score",
                    "variant": variant["name"],
                    "role": panel_role,
                    "command": _command(
                        "scripts.tools.silver_match_v3.score_two_order_gepa",
                        "--truth", role_paths[role]["truth"],
                        "--original", outputs["original"],
                        "--hashed", outputs["hashed"],
                        "--panel-role", panel_role,
                        "--output", output_root / "scores" / "adjudicator" / f"{variant['name']}.{role}.json",
                    ),
                }
            )

    # Verifier commands are frozen against every adjudicator variant rather
    # than being invented after errors are seen.  The consensus proposal is an
    # upstream reference; truth/candidates are projected to exactly those UIDs.
    for adj in adjudicator_variants:
        for verifier in verifier_variants:
            for role, panel_role in ROLE_TO_PANEL.items():
                base = output_root / "runs" / "adjudicator" / adj["name"] / role
                primary = base / "two_order_consensus.jsonl"
                truth_subset = output_root / "runs" / "verifier" / f"{adj['name']}__{verifier['name']}" / role / "truth.proposals.jsonl"
                candidate_subset = truth_subset.with_name(f"candidates.proposals.top{args.candidate_k}.jsonl")
                stages.extend(
                    [
                        {
                            "stage": "verifier_subset_truth",
                            "adjudicator_variant": adj["name"],
                            "verifier_variant": verifier["name"],
                            "role": panel_role,
                            "command": _command(
                                "scripts.tools.silver_match_v3.subset_jsonl_by_reference",
                                "--input", role_paths[role]["truth"],
                                "--reference", primary,
                                "--output", truth_subset,
                            ),
                        },
                        {
                            "stage": "verifier_subset_candidates",
                            "adjudicator_variant": adj["name"],
                            "verifier_variant": verifier["name"],
                            "role": panel_role,
                            "command": _command(
                                "scripts.tools.silver_match_v3.subset_jsonl_by_reference",
                                "--input", role_paths[role]["candidates"],
                                "--reference", primary,
                                "--output", candidate_subset,
                            ),
                        },
                    ]
                )
                verify_outputs: dict[str, str] = {}
                for order in VERIFIER_ORDERS:
                    output = truth_subset.with_name(f"{order}.jsonl")
                    verify_outputs[order] = str(output)
                    cap = 2 * len(rows_by_role[role])
                    stages.append(
                        {
                            "stage": "verifier",
                            "adjudicator_variant": adj["name"],
                            "verifier_variant": verifier["name"],
                            "role": panel_role,
                            "order": order,
                            "maximum_api_requests": cap,
                            "command": _command(
                                "scripts.tools.silver_match_v3.verify_gemma_api",
                                "--manifest", manifest_path,
                                "--candidates", candidate_subset,
                                "--primary", primary,
                                "--output", output,
                                "--split-role", role,
                                *_variant_prompt_args(verifier),
                                "--api-base-url", args.api_base_url,
                                "--api-key-file", args.api_key_file,
                                "--max-api-requests", cap,
                                "--model", args.model,
                                "--max-alternatives", args.candidate_k - 1,
                                "--concurrency", args.concurrency,
                                "--transport-retries", 0,
                                "--order-mode", order,
                                "--resume",
                            ),
                            "direct_batch_command": _command(
                                "scripts.tools.silver_match_v3.verify_gemma",
                                "--manifest", manifest_path,
                                "--candidates", candidate_subset,
                                "--primary", primary,
                                "--output", output,
                                *_variant_prompt_args(verifier),
                                "--model", args.direct_model,
                                "--max-alternatives", args.candidate_k - 1,
                                "--batch-size", args.direct_batch_size,
                                "--gpu-memory-utilization", args.gpu_memory_utilization,
                                "--order-mode", order,
                                "--resume",
                            ),
                        }
                    )
                stages.extend(
                    [
                        {
                            "stage": "verifier_score_two_order",
                            "adjudicator_variant": adj["name"],
                            "verifier_variant": verifier["name"],
                            "role": panel_role,
                            "command": _command(
                                "scripts.tools.silver_match_v3.score_two_order_verifier",
                                "--truth", truth_subset,
                                "--primary", primary,
                                "--original", verify_outputs["original"],
                                "--hashed", verify_outputs["hashed"],
                                "--output", output_root / "scores" / "verifier" / f"{adj['name']}__{verifier['name']}.{role}.two_order.json",
                            ),
                        },
                        {
                            "stage": "verifier_score_three_order",
                            "adjudicator_variant": adj["name"],
                            "verifier_variant": verifier["name"],
                            "role": panel_role,
                            "command": _command(
                                "scripts.tools.silver_match_v3.score_three_order_verifier",
                                "--truth", truth_subset,
                                "--primary", primary,
                                "--original", verify_outputs["original"],
                                "--hashed", verify_outputs["hashed"],
                                "--reverse", verify_outputs["reverse"],
                                "--selection-split", "dev" if role == "dev" else "optimize",
                                "--output", output_root / "scores" / "verifier" / f"{adj['name']}__{verifier['name']}.{role}.three_order.json",
                            ),
                        },
                    ]
                )

    plan = {
        "schema_version": "silver-match-v3-task-local-gepa-api-plan-v1",
        "status": "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE",
        "task": args.task,
        "scientific_scope": {
            "selection_universe": "canonical upstream train groups only",
            "prompt_mutation_role": "prompt_train",
            "prompt_selection_role": "prompt_dev",
            "test_or_blind_audit_consumed": False,
            "production_consumed": False,
            "outcomes_or_mi_used": False,
            "gradient_authorized": False,
            "exact_frozen_bank_leaf_is_primary": True,
            "family_credit_is_sensitivity_only": True,
        },
        "thresholds": {
            "minimum_point_precision": args.minimum_point_precision,
            "minimum_wilson_95_lower": args.minimum_wilson_lower,
            "minimum_retained": args.minimum_retained,
            "selection_rule": (
                "eligible only if all three thresholds pass on prompt_dev; among eligible "
                "policies prefer larger Wilson lower bound, then support, then point precision"
            ),
        },
        "api_budget": {
            "model": args.model,
            "base_url": args.api_base_url,
            "credential_file_path_only": str(Path(args.api_key_file).expanduser()),
            "credential_value_recorded": False,
            "maximum_adjudicator_requests": adjudicator_requests,
            "maximum_verifier_requests": verifier_requests,
            "maximum_total_requests": maximum_requests,
            "user_cap": args.max_total_api_requests,
            "retry_policy": "at most one JSON-contract retry per row and command",
            "transport_retry_policy": (
                "zero implicit HTTP retries; resume explicitly so every logical request "
                "remains inside the recorded cap"
            ),
        },
        "inference_backend_policy": {
            "preferred_when_quota_slot_exists": "direct_vllm_batch",
            "small_scale_fallback": "openrouter_exact_gemma4",
            "openai_server_backend_allowed": False,
            "run_exactly_one_backend_per_inference_cell": True,
            "do_not_mix_backends_within_an_order_pair": True,
            "direct_model": args.direct_model,
            "direct_batch_size": args.direct_batch_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "gpu_quota_is_external_to_this_plan": True,
        },
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "predeclaration": predeclaration,
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            "panel": {"path": str(panel_path), "sha256": sha256_file(panel_path)},
            "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            "exclusions": exclusion_details,
        },
        "exclusions": {
            "uid_count": len(excluded_uids),
            "source_group_count": len(excluded_groups),
            "selected_uid_overlap": 0,
            "selected_source_group_overlap": 0,
        },
        "roles": {
            role: {
                "panel_role": ROLE_TO_PANEL[role],
                "truth": {"path": str(paths["truth"]), "sha256": sha256_file(paths["truth"])},
                "candidates": {"path": str(paths["candidates"]), "sha256": sha256_file(paths["candidates"])},
                "count": len(rows_by_role[role]),
                "source_groups": len(groups_by_role[role]),
                "decision_counts": dict(sorted(Counter(str(row.get("decision")) for row in rows_by_role[role]).items())),
            }
            for role, paths in role_paths.items()
        },
        "candidate_k": args.candidate_k,
        "candidate_bank_source_sha256": next(iter(bank_hashes)),
        "adjudicator_variants": adjudicator_variants,
        "verifier_variants": verifier_variants,
        "commands": stages,
    }
    command_plan = output_root / "COMMAND_PLAN.json"
    command_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze_payload = {
        "schema_version": "silver-match-v3-task-local-gepa-freeze-v1",
        "task": args.task,
        "status": plan["status"],
        "command_plan": {"path": str(command_plan), "sha256": sha256_file(command_plan)},
        "panel_uid_sha256": {
            role: hashlib.sha256(
                "\n".join(sorted(str(row["norm_uid"]) for row in rows)).encode("utf-8")
            ).hexdigest()
            for role, rows in rows_by_role.items()
        },
        "source_group_overlap": 0,
        "exclusion_source_group_overlap": 0,
        "test_or_blind_audit_consumed": False,
        "production_consumed": False,
    }
    freeze_path = output_root / "FREEZE.json"
    freeze_path.write_text(
        json.dumps(freeze_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **freeze_payload,
        "freeze_sha256": sha256_file(freeze_path),
        "maximum_total_api_requests": maximum_requests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--predeclaration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--exclude-reference", action="append", required=True)
    parser.add_argument(
        "--adjudicator-variant",
        action="append",
        required=True,
        help="frozen NAME=PATH[,PATH...] prompt component sequence",
    )
    parser.add_argument(
        "--verifier-variant",
        action="append",
        required=True,
        help="frozen NAME=PATH[,PATH...] prompt component sequence",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--minimum-train", type=int, default=30)
    parser.add_argument("--minimum-dev", type=int, default=30)
    parser.add_argument("--minimum-point-precision", type=float, default=0.90)
    parser.add_argument("--minimum-wilson-lower", type=float, default=0.80)
    parser.add_argument("--minimum-retained", type=int, default=30)
    parser.add_argument("--max-total-api-requests", type=int, default=5000)
    parser.add_argument("--api-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-file", default="~/.openrouter-api-key.txt")
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--direct-model", default=GEMMA4)
    parser.add_argument("--direct-batch-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()
    if args.candidate_k < 2 or args.concurrency < 1 or args.direct_batch_size < 1:
        parser.error("candidate K must be at least two and batch/concurrency must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("GPU memory utilization must be in (0, 1]")
    return args


def main() -> None:
    print(json.dumps(freeze(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
