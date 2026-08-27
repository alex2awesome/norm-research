#!/usr/bin/env python3
"""Freeze a leakage-safe, spend-bounded explicit-role task-local GEPA experiment.

This is a planning/freeze step, not an inference runner.  It projects a
previously frozen, separately selected optimize/select human panels and their
retriever candidates into source-group-disjoint GEPA cells.  The planner binds
the pre-label role FREEZE and identity artifacts directly; it never recreates
the obsolete hash-split fields used by the legacy planner.  It also seals all
permanent exclusions and prompt variants and writes equivalent direct-batch and
bounded OpenRouter commands.  No test, blind-audit, production, MI, or outcome
row can enter the plan.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl, sha256_file, write_jsonl
from .config import GEMMA4
from .make_calibration import split_group_for
from .train_nemotron_lora import source_group_key, split_source_group


ROLE_TO_PANEL = {"optimize": "prompt_train", "select": "prompt_dev"}
ROLE_TO_RUNNER_SPLIT = {"optimize": "train", "select": "dev"}
ORDERS = ("original", "hashed")
VERIFIER_ORDERS = ("original", "hashed", "reverse")
TRUTH_RELEASE_SCHEMA = "silver-match-v3-clean-gepa-exact-truth-release-v2"
TRUTH_RELEASE_STATUS = "FROZEN_EXACT_TRUTH_RELEASE_AUDITED"
_HASH_CACHE: dict[Path, str] = {}


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return indexed


def _sha256(path: Path) -> str:
    path = path.resolve()
    if path not in _HASH_CACHE:
        _HASH_CACHE[path] = sha256_file(path)
    return _HASH_CACHE[path]


def _verify_ref(ref: dict[str, Any], *, label: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or not expected or _sha256(path) != expected:
        raise ValueError(f"{label} is missing or hash-drifted: {path}")
    return path


def _verify_candidate_lineage(
    *, source: Path, expected_sha256: str, manifest_sha256: str, candidate_k: int
) -> dict[str, Any]:
    """Recursively bind combined/filter metadata to retrieval query provenance."""
    visited: dict[Path, dict[str, Any]] = {}

    def walk(path: Path, expected: str) -> None:
        path = path.resolve()
        if path in visited:
            if _sha256(path) != expected:
                raise ValueError(
                    f"candidate lineage reuses a path with another hash: {path}"
                )
            return
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"candidate lineage data drift: {path}")
        meta_path = Path(str(path) + ".meta.json")
        if not meta_path.is_file():
            raise ValueError(f"candidate lineage metadata missing: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        output_hash = str(meta.get("output_sha256") or meta.get("sha256") or "")
        if output_hash != expected:
            raise ValueError(
                f"candidate lineage metadata does not bind output: {meta_path}"
            )
        node = {
            "path": str(path),
            "sha256": expected,
            "meta": {"path": str(meta_path), "sha256": _sha256(meta_path)},
            "kind": "",
        }
        visited[path] = node
        if meta.get("manifest_sha256") is not None:
            fusion = Path(str(meta.get("fusion_weights") or "")).resolve()
            adapter = Path(str(meta.get("adapter") or "")).resolve()
            if (
                str(meta.get("manifest_sha256")) != manifest_sha256
                or int(meta.get("output_k") or -1) < candidate_k
                or str(meta.get("query_format") or "") not in {"nemotron", "bge"}
                or not str(meta.get("query_views") or "")
                or not str(meta.get("encoder") or "")
                or not fusion.is_file()
                or _sha256(fusion) != str(meta.get("fusion_weights_sha256") or "")
            ):
                raise ValueError(
                    f"retrieval query/manifest/fusion provenance invalid: {meta_path}"
                )
            adapter_hashes = meta.get("adapter_hashes") or {}
            if adapter_hashes:
                if not adapter.is_dir():
                    raise ValueError(f"retrieval adapter is missing: {adapter}")
                for name, expected_adapter_sha in adapter_hashes.items():
                    if _sha256(adapter / str(name)) != str(expected_adapter_sha):
                        raise ValueError(
                            f"retrieval adapter hash drift: {adapter}/{name}"
                        )
            node["kind"] = "retrieval_leaf"
            node["query_provenance"] = {
                key: meta.get(key)
                for key in (
                    "manifest_sha256",
                    "query_format",
                    "query_views",
                    "dense_query_instruction",
                    "encoder",
                    "adapter",
                    "adapter_hashes",
                    "fusion_weights",
                    "fusion_weights_sha256",
                    "output_k",
                    "component_k",
                    "component_weights",
                    "rrf_constant",
                )
            }
            return
        inputs = meta.get("inputs")
        if isinstance(inputs, dict) and inputs:
            node["kind"] = "combined"
            for raw_path, child in inputs.items():
                child_sha = str((child or {}).get("sha256") or "")
                if not child_sha:
                    raise ValueError(
                        f"combined candidate lineage lacks child hash: {meta_path}"
                    )
                walk(Path(raw_path), child_sha)
            return
        if meta.get("input") and meta.get("input_sha256"):
            node["kind"] = "filtered"
            walk(Path(str(meta["input"])), str(meta["input_sha256"]))
            return
        raise ValueError(
            f"candidate lineage terminates without retrieval provenance: {meta_path}"
        )

    walk(source, expected_sha256)
    leaves = [node for node in visited.values() if node["kind"] == "retrieval_leaf"]
    if not leaves:
        raise ValueError("candidate lineage contains no retrieval-provenance leaf")
    return {
        "root": str(source.resolve()),
        "root_sha256": expected_sha256,
        "node_count": len(visited),
        "retrieval_leaf_count": len(leaves),
        "nodes": sorted(visited.values(), key=lambda row: row["path"]),
    }


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
        text = (
            "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in paths)
            + "\n"
        )
        variants.append(
            {
                "name": name,
                "components": [
                    {"path": str(path), "sha256": sha256_file(path)} for path in paths
                ],
                "combined_prompt_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
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


def _validate_truth_release(
    *,
    path: Path,
    task: str,
    role: str,
    truth_path: Path,
    freeze_path: Path,
    identities_path: Path,
    candidate_audit_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    release = json.loads(path.read_text(encoding="utf-8"))
    contract = release.get("scientific_contract") or {}
    replay = release.get("consensus_replay") or {}
    if (
        release.get("schema_version") != TRUTH_RELEASE_SCHEMA
        or release.get("status") != TRUTH_RELEASE_STATUS
        or release.get("task") != task
        or release.get("role") != role
        or int(release.get("count") or -1) != expected_count
        or (release.get("truth") or {}).get("sha256") != _sha256(truth_path)
        or (release.get("role_freeze") or {}).get("sha256") != _sha256(freeze_path)
        or (release.get("identities") or {}).get("sha256") != _sha256(identities_path)
        or (release.get("candidate_release") or {}).get("sha256")
        != _sha256(candidate_audit_path)
        or any(
            contract.get(key) is not True
            for key in (
                "exact_decision_and_leaf_consensus_complete",
                "consensus_recomputed_from_bound_pass_labels",
                "all_pass_labels_and_validations_hash_bound",
                "transcripts_hash_bound_and_leakage_audited",
                "strict_transcript_pass_required_for_every_consensus_pass",
                "cross_workspace_artifacts_hash_equivalent",
                "truth_may_be_used_only_for_declared_gepa_role",
            )
        )
        or contract.get("legacy_transcripts_allowed") is not False
        or int(replay.get("resolved_count") or -1) != expected_count
        or int(replay.get("unresolved_count", -1)) != 0
        or int(replay.get("round_count") or -1) < 2
        or replay.get("round_metadata_verified") is not True
        or replay.get("released_decision_metric_confidence_supporters_exact")
        is not True
    ):
        raise ValueError(
            f"{role} exact-truth release is missing, drifted, or incomplete"
        )
    consensus_path = _verify_ref(
        release.get("consensus_report") or {}, label=f"{role} consensus report"
    )
    consensus = json.loads(consensus_path.read_text(encoding="utf-8"))
    if (
        consensus.get("schema_version")
        != "silver-match-v3-exact-multi-pass-truth-report-v1"
        or consensus.get("complete") is not True
        or consensus.get("task") != task
        or consensus.get("gepa_role") != role
        or int(consensus.get("resolved_count") or -1) != expected_count
        or int(consensus.get("unresolved_count", -1)) != 0
        or ((consensus.get("outputs") or {}).get("resolved") or {}).get("sha256")
        != _sha256(truth_path)
    ):
        raise ValueError(f"{role} released consensus no longer binds exact truth")
    _verify_ref(
        release.get("independence_audit") or {}, label=f"{role} independence audit"
    )
    passes = release.get("passes") or []
    if len(passes) < 2:
        raise ValueError(f"{role} truth release has fewer than two independent passes")
    for pass_meta in passes:
        _verify_ref(pass_meta.get("labels") or {}, label=f"{role} released pass labels")
        _verify_ref(
            pass_meta.get("label_validation") or {},
            label=f"{role} released pass validation",
        )
        _verify_ref(
            pass_meta.get("pack_validation") or {},
            label=f"{role} released pass pack",
        )
        transcript = pass_meta.get("transcript_audit") or {}
        if (
            transcript.get("mode") != "strict_isolation_audit"
            or transcript.get("full_pack_artifact_binding") is not True
            or transcript.get("artifact_equivalence_verified") is not True
            or not transcript.get("guide_sha256")
        ):
            raise ValueError(
                f"{role} truth release requires a strict fully bound transcript audit"
            )
        _verify_ref(transcript, label=f"{role} strict transcript audit")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "status": release["status"],
        "consensus_report_sha256": _sha256(consensus_path),
        "pass_count": len(passes),
    }


def _validate_candidate_audit(
    *,
    path: Path,
    task: str,
    role: str,
    candidates_path: Path,
    freeze_path: Path,
    identities_path: Path,
    manifest_path: Path,
    bank_path: Path,
    bank_source_sha256: str,
    candidate_k: int,
    expected_count: int,
) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    inputs = audit.get("inputs") or {}
    outputs = audit.get("outputs") or {}
    source_ref = inputs.get("candidate_source") or {}
    if (
        audit.get("schema_version") != "silver-match-v3-clean-gepa-label-pack-v1"
        or audit.get("status") != "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING"
        or audit.get("truth_hidden") is not True
        or audit.get("task") != task
        or audit.get("gepa_role") != role
        or int(audit.get("count") or -1) != expected_count
        or int(audit.get("candidate_k") or -1) != candidate_k
        or audit.get("bank_source_sha256") != bank_source_sha256
        or (inputs.get("manifest") or {}).get("sha256") != _sha256(manifest_path)
        or (inputs.get("bank_source") or {}).get("sha256") != _sha256(bank_path)
        or (inputs.get("identities") or {}).get("sha256") != _sha256(identities_path)
        or (inputs.get("identity_freeze") or {}).get("sha256") != _sha256(freeze_path)
        or (outputs.get("candidates") or {}).get("sha256") != _sha256(candidates_path)
        or not str(source_ref.get("sha256") or "")
    ):
        raise ValueError(f"{role} candidate audit/release is missing or drifted")
    source_path = _verify_ref(source_ref, label=f"{role} frozen candidate source")
    upstream_path = _verify_ref(
        inputs.get("upstream_role_freeze") or {},
        label=f"{role} upstream train-role freeze",
    )
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    upstream_inputs = upstream.get("inputs") or {}
    upstream_output = upstream.get("output") or {}
    split_policy = upstream.get("split_policy") or {}
    if (
        upstream.get("schema_version")
        != "silver-match-v3-upstream-role-reference-freeze-v1"
        or upstream.get("status") != "FROZEN_AND_AUDIT_VERIFIED"
        or upstream.get("task") != task
        or upstream.get("bank_source_sha256") != bank_source_sha256
        or int(upstream.get("minimum_k") or -1) < candidate_k
        or (upstream_inputs.get("manifest") or {}).get("sha256")
        != _sha256(manifest_path)
        or (upstream_inputs.get("candidates") or {}).get("sha256")
        != str(source_ref["sha256"])
        or split_policy.get("function") != "train_nemotron_lora.split_source_group"
        or split_policy.get("group_function") != "train_nemotron_lora.source_group_key"
    ):
        raise ValueError(f"{role} upstream role/candidate provenance is invalid")
    try:
        split_seed = int(split_policy["split_seed"])
        train_percent = int(split_policy["train_percent"])
        dev_percent = int(split_policy["dev_percent"])
        test_percent = int(split_policy["test_percent"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} upstream split policy is incomplete") from exc
    if (
        train_percent <= 0
        or dev_percent <= 0
        or test_percent <= 0
        or train_percent + dev_percent + test_percent != 100
    ):
        raise ValueError(f"{role} upstream split policy is invalid")

    upstream_roles_path = _verify_ref(
        upstream_output, label=f"{role} upstream role reference"
    )
    upstream_roles = _index(upstream_roles_path)
    observed_role_counts = Counter(
        str(row.get("split") or "") for row in upstream_roles.values()
    )
    observed_group_counts = {
        split: len(
            {
                str(row.get("retriever_source_group") or "")
                for row in upstream_roles.values()
                if row.get("split") == split
            }
        )
        for split in ("train", "dev", "test")
    }
    if (
        any(
            row.get("schema_version") != "silver-match-v3-upstream-role-reference-v1"
            or row.get("task") != task
            or row.get("split") not in {"train", "dev", "test"}
            or not str(row.get("source_group") or "")
            or not str(row.get("retriever_source_group") or "")
            for row in upstream_roles.values()
        )
        or dict(observed_role_counts) != (upstream.get("roles") or {})
        or observed_group_counts != (upstream.get("role_source_groups") or {})
        or int(upstream.get("candidate_rows") or -1) != len(_index(source_path))
    ):
        raise ValueError(f"{role} upstream role reference is incomplete or drifted")

    role_audit = upstream.get("audit_verification") or {}
    audited_labels_path = _verify_ref(
        role_audit, label=f"{role} upstream role audit labels"
    )
    audited_labels = _index(audited_labels_path)
    overlap = set(audited_labels) & set(upstream_roles)
    mismatches = sum(
        str(audited_labels[uid].get("split") or "")
        != str(upstream_roles[uid].get("split") or "")
        for uid in overlap
    )
    if (
        int(role_audit.get("overlap", -1)) != len(overlap)
        or int(role_audit.get("mismatches", -1)) != mismatches
        or int(role_audit.get("exact_role_matches", -1)) != len(overlap) - mismatches
        or mismatches != 0
    ):
        raise ValueError(f"{role} upstream train-role audit does not verify exactly")
    run_config_path = _verify_ref(
        upstream_inputs.get("run_config") or {}, label=f"{role} retriever run config"
    )
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if (
        run_config.get("task") != task
        or int(run_config.get("selection_k") or -1) < candidate_k
        or not str(run_config.get("query_instruction") or "")
    ):
        raise ValueError(f"{role} retriever run configuration is invalid")
    lineage = _verify_candidate_lineage(
        source=source_path,
        expected_sha256=str(source_ref["sha256"]),
        manifest_sha256=_sha256(manifest_path),
        candidate_k=candidate_k,
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "status": audit["status"],
        "candidate_source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "upstream_role_freeze": {
            "path": str(upstream_path),
            "sha256": _sha256(upstream_path),
        },
        "upstream_role_reference": {
            "path": str(upstream_roles_path),
            "sha256": _sha256(upstream_roles_path),
            "row_count": len(upstream_roles),
            "field": "split",
        },
        "upstream_role_split_policy": {
            "function": split_policy["function"],
            "group_function": split_policy["group_function"],
            "split_seed": split_seed,
            "train_percent": train_percent,
            "dev_percent": dev_percent,
            "test_percent": test_percent,
        },
        "upstream_role_audit": {
            "path": str(audited_labels_path),
            "sha256": _sha256(audited_labels_path),
            "overlap": len(overlap),
            "mismatches": mismatches,
        },
        "retriever_run_config": {
            "path": str(run_config_path),
            "sha256": _sha256(run_config_path),
        },
        "query_lineage": lineage,
    }


def _validate_role_artifacts(
    *,
    task: str,
    role: str,
    truth_path: Path,
    candidates_path: Path,
    freeze_path: Path,
    identities_path: Path,
    truth_release_path: Path,
    candidate_audit_path: Path,
    norms: dict[str, dict[str, Any]],
    manifest_path: Path,
    bank_path: Path,
    bank_source_sha256: str,
    candidate_k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Fail closed unless a clean pre-label role freeze binds this exact universe."""
    truth = _index(truth_path)
    candidates = _index(candidates_path)
    identities = _index(identities_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    contract = freeze.get("content_contract") or {}
    frozen_identities = (freeze.get("outputs") or {}).get("identities") or {}
    if (
        freeze.get("schema_version") != "silver-match-v3-clean-gepa-panel-freeze-v1"
        or freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or freeze.get("task") != task
        or freeze.get("role") != role
        or freeze.get("required_upstream_split") != "train"
        or int(freeze.get("selected_count") or -1) != len(identities)
        or int(freeze.get("selected_source_groups") or -1)
        != len({str(row.get("source_group") or "") for row in identities.values()})
        or str(frozen_identities.get("sha256") or "") != sha256_file(identities_path)
        or contract.get("selection_uses_identity_and_source_group_only") is not True
        or any(
            contract.get(key) is not False
            for key in (
                "downstream_outcomes_read",
                "metric_ids_read",
                "model_prediction_fields_read",
                "truth_fields_read",
            )
        )
    ):
        raise ValueError(
            f"{role} role freeze is missing, changed, or not pre-label clean"
        )
    if set(truth) != set(identities) or set(candidates) != set(identities):
        raise ValueError(f"{role} truth/candidate UID universe differs from identities")
    candidate_audit = _validate_candidate_audit(
        path=candidate_audit_path,
        task=task,
        role=role,
        candidates_path=candidates_path,
        freeze_path=freeze_path,
        identities_path=identities_path,
        manifest_path=manifest_path,
        bank_path=bank_path,
        bank_source_sha256=bank_source_sha256,
        candidate_k=candidate_k,
        expected_count=len(identities),
    )
    frozen_upstream = (freeze.get("inputs") or {}).get("upstream_role_reference") or {}
    audited_upstream = candidate_audit["upstream_role_reference"]
    if (
        frozen_upstream.get("authoritative") is not True
        or frozen_upstream.get("field") != "split"
        or Path(str(frozen_upstream.get("path") or "")).resolve()
        != Path(audited_upstream["path"]).resolve()
        or str(frozen_upstream.get("sha256") or "") != audited_upstream["sha256"]
    ):
        raise ValueError(f"{role} role freeze does not bind the audited upstream roles")
    upstream_roles = _index(Path(audited_upstream["path"]))
    split_policy = candidate_audit["upstream_role_split_policy"]
    for uid, identity in identities.items():
        if uid not in norms:
            raise ValueError(f"{role} identity absent from canonical task norms: {uid}")
        norm = norms[uid]
        canonical_group = split_group_for(norm)
        retriever_group = source_group_key(norm)
        upstream_row = upstream_roles.get(uid) or {}
        upstream_split = split_source_group(
            retriever_group,
            seed=int(split_policy["split_seed"]),
            train_percent=int(split_policy["train_percent"]),
            dev_percent=int(split_policy["dev_percent"]),
        )
        if (
            identity.get("schema_version")
            != "silver-match-v3-clean-gepa-panel-identity-v1"
            or identity.get("task") != task
            or identity.get("gepa_role") != role
            or identity.get("upstream_split") != "train"
            or identity.get("source_group") != canonical_group
            or identity.get("permanently_excluded_from_mi_and_outcome_estimation")
            is not True
            or identity.get("permanently_excluded_from_retriever_gradients") is not True
            or upstream_row.get("schema_version")
            != "silver-match-v3-upstream-role-reference-v1"
            or upstream_row.get("task") != task
            or upstream_row.get("corpus") != norm.get("corpus")
            or upstream_row.get("source_group") != canonical_group
            or upstream_row.get("retriever_source_group") != retriever_group
            or upstream_row.get("split") != "train"
            or upstream_split != "train"
            or int(upstream_row.get("split_seed") or -1)
            != int(split_policy["split_seed"])
            or int(upstream_row.get("train_percent") or -1)
            != int(split_policy["train_percent"])
            or int(upstream_row.get("dev_percent") or -1)
            != int(split_policy["dev_percent"])
            or int(upstream_row.get("test_percent") or -1)
            != int(split_policy["test_percent"])
        ):
            raise ValueError(f"invalid or non-train {role} identity provenance: {uid}")
        row = truth[uid]
        if (
            row.get("task") != task
            or row.get("gepa_role") != role
            or row.get("split") != "train"
            or (row.get("source_group") or row.get("split_group")) != canonical_group
        ):
            raise ValueError(
                f"truth row differs from frozen explicit role identity: {uid}"
            )
    truth_release = _validate_truth_release(
        path=truth_release_path,
        task=task,
        role=role,
        truth_path=truth_path,
        freeze_path=freeze_path,
        identities_path=identities_path,
        candidate_audit_path=candidate_audit_path,
        expected_count=len(identities),
    )
    released_candidate_sha = str(
        (
            json.loads(truth_release_path.read_text(encoding="utf-8")).get(
                "candidate_release"
            )
            or {}
        ).get("candidate_sha256")
        or ""
    )
    if released_candidate_sha != _sha256(candidates_path):
        raise ValueError(f"{role} truth release is bound to another candidate slate")
    metadata = {
        "role": role,
        "truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
        "candidates": {
            "path": str(candidates_path),
            "sha256": sha256_file(candidates_path),
        },
        "freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
        "identities": {
            "path": str(identities_path),
            "sha256": sha256_file(identities_path),
        },
        "truth_release": truth_release,
        "candidate_audit": candidate_audit,
        "count": len(identities),
    }
    return truth, candidates, metadata


def _validate_predeclaration(
    *,
    path: Path,
    args: argparse.Namespace,
    adjudicator_variants: list[dict[str, Any]],
    verifier_variants: list[dict[str, Any]],
    exclusion_paths: list[Path],
    excluded_uids: set[str],
    norms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version")
        != "silver-match-v3-task-local-gepa-predeclaration-v1"
    ):
        raise ValueError("unsupported task-local GEPA predeclaration schema")
    if lock.get("status") != "FROZEN_AND_EXECUTION_AUTHORIZED":
        raise ValueError("task-local GEPA predeclaration is not execution-authorized")
    parent_ref = lock.get("parent_predeclaration") or {}
    if parent_ref.get("variants_gates_models_and_budgets_changed") is not False:
        raise ValueError("activation lock does not certify an unchanged parent design")
    parent_recorded = Path(str(parent_ref.get("path") or ""))
    parent_resolved = (
        parent_recorded.resolve()
        if parent_recorded.is_absolute()
        else (path.parents[4] / parent_recorded).resolve()
    )
    parent_path = _verify_ref(
        {**parent_ref, "path": str(parent_resolved)},
        label="parent task-local GEPA predeclaration",
    )
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("schema_version")
        != "silver-match-v3-task-local-gepa-predeclaration-v1"
        or parent.get("status")
        != "PREDECLARED_PENDING_CANONICAL_PACKS_AND_COMPLETE_EXCLUSIONS"
        or not str(lock.get("activation_scope") or "")
    ):
        raise ValueError("parent task-local GEPA predeclaration is invalid")

    # The activation record is append-only: it may date and describe the
    # activation, flip task authorization/blockers, and add execution evidence.
    # Every scientific choice (tasks, prompts, gates, models, budgets, and
    # forbidden inputs) must remain byte-for-byte equal as parsed JSON.
    normalized = copy.deepcopy(lock)
    normalized.pop("parent_predeclaration", None)
    normalized.pop("activation_scope", None)
    normalized["created_date"] = parent.get("created_date")
    normalized["status"] = parent.get("status")
    parent_tasks = parent.get("tasks") or {}
    activated_tasks = normalized.get("tasks") or {}
    if set(activated_tasks) != set(parent_tasks):
        raise ValueError("activation lock changed the predeclared task universe")
    for task_name, parent_task in parent_tasks.items():
        activated_task = activated_tasks[task_name]
        if activated_task.get("execution_authorized") is True:
            if (
                parent_task.get("execution_authorized") is not False
                or not parent_task.get("blocker")
                or activated_task.get("blocker") is not None
                or not isinstance(activated_task.get("execution_evidence"), dict)
            ):
                raise ValueError(f"invalid append-only task activation: {task_name}")
        activated_task["execution_authorized"] = parent_task.get("execution_authorized")
        activated_task["blocker"] = parent_task.get("blocker")
        if "execution_evidence" in parent_task:
            activated_task["execution_evidence"] = copy.deepcopy(
                parent_task["execution_evidence"]
            )
        else:
            activated_task.pop("execution_evidence", None)
    if normalized != parent:
        raise ValueError(
            "activation lock changed fields beyond authorization/blocker/evidence"
        )
    task_lock = (lock.get("tasks") or {}).get(args.task)
    if not task_lock:
        raise ValueError(f"predeclaration does not contain task {args.task}")
    if task_lock.get("execution_authorized") is not True or task_lock.get("blocker"):
        raise ValueError(f"predeclaration does not authorize task {args.task}")
    evidence = task_lock.get("execution_evidence") or {}
    supplied_exclusion_hashes = {_sha256(item) for item in exclusion_paths}
    if (
        evidence.get("optimize_truth_release_sha256")
        != _sha256(Path(args.optimize_truth_release))
        or evidence.get("select_truth_release_sha256")
        != _sha256(Path(args.select_truth_release))
        or evidence.get("complete_exclusion_union_sha256")
        not in supplied_exclusion_hashes
    ):
        raise ValueError(f"execution authorization evidence drifted for {args.task}")
    union_sha = str(evidence.get("complete_exclusion_union_sha256") or "")
    matching_unions = [path for path in exclusion_paths if _sha256(path) == union_sha]
    if len(matching_unions) != 1:
        raise ValueError(
            f"execution authorization does not identify one complete exclusion union: {args.task}"
        )
    complete_union_path = matching_unions[0]
    inventory_path = complete_union_path.parent / "EXCLUSION_INVENTORY.json"
    inventory_sha = str(evidence.get("exclusion_inventory_sha256") or "")
    if (
        not inventory_path.is_file()
        or not inventory_sha
        or _sha256(inventory_path) != inventory_sha
    ):
        raise ValueError(
            f"complete exclusion inventory is missing or hash-drifted: {args.task}"
        )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_contract = inventory.get("content_contract") or {}
    inventory_manifest = (inventory.get("inputs") or {}).get("manifest") or {}
    identity_union = inventory.get("identity_union") or {}
    inventory_sources = inventory.get("sources") or {}
    required_categories = set(inventory.get("required_categories") or [])
    observed_categories = set(inventory.get("observed_categories") or [])
    union_rows = _index(complete_union_path)
    union_groups = {
        str(row.get("source_group") or split_group_for(norms[uid]))
        for uid, row in union_rows.items()
    }
    union_corpora = Counter(
        str(row.get("corpus") or norms[uid].get("corpus") or "")
        for uid, row in union_rows.items()
    )
    union_splits = Counter(
        str(row.get("upstream_split") or "") for row in union_rows.values()
    )
    if (
        inventory.get("schema_version") != "silver-match-v3-gepa-exclusion-union-v1"
        or inventory.get("status")
        != "FROZEN_BEFORE_NEW_PANEL_SELECTION_PREDICTIONS_OR_LABELS"
        or inventory.get("task") != args.task
        or inventory.get("all_required_categories_present") is not True
        or not required_categories
        or observed_categories != required_categories
        or inventory_contract.get("parsed_sources_used_only_identity_fields")
        is not True
        or inventory_contract.get(
            "model_predictions_metric_ids_reasons_and_outcomes_used"
        )
        is not False
        or inventory_contract.get("sealed_test_or_outcome_structured_content_parsed")
        is not False
        or Path(str(inventory_manifest.get("path") or "")).resolve()
        != Path(args.manifest).resolve()
        or inventory_manifest.get("sha256") != _sha256(Path(args.manifest))
        or Path(str(identity_union.get("path") or "")).resolve()
        != complete_union_path.resolve()
        or str(identity_union.get("sha256") or "") != union_sha
        or int(identity_union.get("uids") or -1) != len(union_rows)
        or int(identity_union.get("source_groups") or -1) != len(union_groups)
        or (identity_union.get("by_corpus") or {})
        != dict(sorted(union_corpora.items()))
        or (identity_union.get("by_upstream_split") or {})
        != dict(sorted(union_splits.items()))
        or not inventory_sources
    ):
        raise ValueError(
            f"complete exclusion inventory contract is invalid: {args.task}"
        )
    if any(
        uid not in norms
        or row.get("task") != args.task
        or str(row.get("source_group") or "") != split_group_for(norms[uid])
        for uid, row in union_rows.items()
    ):
        raise ValueError(
            f"complete exclusion union has noncanonical identities: {args.task}"
        )
    source_uids: set[str] = set()
    source_groups: set[str] = set()
    source_categories: set[str] = set()
    source_details = []
    for recorded, meta in inventory_sources.items():
        source_path = Path(str(recorded)).resolve()
        if (
            not source_path.is_file()
            or _sha256(source_path) != str(meta.get("sha256") or "")
            or meta.get("format") != "jsonl"
            or meta.get("category") not in required_categories
            or set(meta.get("fields_used") or []) != {"norm_uid", "source_group"}
            or meta.get("canonical_source_group_recomputed") is not True
            or meta.get("structured_content_parsed") is not True
            or int(meta.get("supplied_source_group_mismatch_count", -1)) != 0
        ):
            raise ValueError(
                f"exclusion inventory source is missing or drifted: {source_path}"
            )
        indexed = _index(source_path)
        unknown = set(indexed) - set(norms)
        canonical_groups = {
            split_group_for(norms[uid]) for uid in indexed if uid in norms
        }
        if (
            unknown
            or int(meta.get("uids") or -1) != len(indexed)
            or int(meta.get("source_groups") or -1) != len(canonical_groups)
        ):
            raise ValueError(
                f"exclusion inventory source identity counts drifted: {source_path}"
            )
        source_uids.update(indexed)
        source_groups.update(canonical_groups)
        source_categories.add(str(meta["category"]))
        source_details.append(
            {
                "path": str(source_path),
                "sha256": _sha256(source_path),
                "category": meta["category"],
                "uid_count": len(indexed),
                "source_group_count": len(canonical_groups),
            }
        )
    if source_categories != required_categories:
        raise ValueError(
            "exclusion inventory source categories differ from required categories"
        )
    if not source_uids <= set(union_rows) or not source_groups <= union_groups:
        raise ValueError(
            "complete exclusion union omits inventoried source identities/groups"
        )
    if (
        args.task == "math-stackexchange"
        and evidence.get("sealed_r5_rows_remain_excluded") is not True
    ):
        raise ValueError(
            "Math execution authorization does not preserve sealed r5 exclusion"
        )
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
    if args.max_total_api_requests > int(
        api["maximum_total_logical_requests_per_task"]
    ):
        raise ValueError("requested API cap exceeds the predeclared task ceiling")
    if int(api.get("implicit_transport_retries", -1)) != 0:
        raise ValueError("predeclaration must disable implicit transport retries")

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
    mandatory = (task_lock.get("sealed_failed_experiment") or {}).get(
        "mandatory_exclusion_universes"
    ) or []
    mandatory_details = []
    supplied_paths = {item.resolve() for item in exclusion_paths}
    for row in mandatory:
        source = (root / str(row["path"])).resolve()
        if not source.is_file() or sha256_file(source) != str(row["sha256"]):
            raise ValueError(
                f"mandatory sealed exclusion artifact is missing or changed: {source}"
            )
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
        "parent_predeclaration": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "status": parent.get("status"),
            "protected_scientific_fields_unchanged": True,
        },
        "complete_exclusion_inventory": {
            "path": str(inventory_path),
            "sha256": _sha256(inventory_path),
            "identity_union": {
                "path": str(complete_union_path),
                "sha256": union_sha,
                "uid_count": len(union_rows),
                "source_group_count": len(union_groups),
            },
            "required_categories": sorted(required_categories),
            "sources": source_details,
        },
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
    input_paths = {
        "optimize": {
            "truth": Path(args.optimize_truth).resolve(),
            "candidates": Path(args.optimize_candidates).resolve(),
            "freeze": Path(args.optimize_freeze).resolve(),
            "identities": Path(args.optimize_identities).resolve(),
            "truth_release": Path(args.optimize_truth_release).resolve(),
            "candidate_audit": Path(args.optimize_candidate_audit).resolve(),
        },
        "select": {
            "truth": Path(args.select_truth).resolve(),
            "candidates": Path(args.select_candidates).resolve(),
            "freeze": Path(args.select_freeze).resolve(),
            "identities": Path(args.select_identities).resolve(),
            "truth_release": Path(args.select_truth_release).resolve(),
            "candidate_audit": Path(args.select_candidate_audit).resolve(),
        },
    }
    exclusion_paths = [Path(value).resolve() for value in args.exclude_reference]
    output_root = Path(args.output_root).resolve()
    if not exclusion_paths:
        raise ValueError("at least one permanent --exclude-reference is required")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite GEPA freeze: {output_root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    norms = _load_task_norms(manifest, args.task)
    bank_meta = (manifest.get("banks") or {}).get(args.task)
    if not bank_meta:
        raise ValueError(f"manifest lacks frozen bank for task {args.task}")
    bank_path = Path(bank_meta["path"]).resolve()
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_rows = bank_payload.get("metrics") or bank_payload.get("bank") or []
    bank_ids = {str(row.get("metric_id") or "") for row in bank_rows}
    if not bank_ids or "" in bank_ids or len(bank_ids) != len(bank_rows):
        raise ValueError("frozen bank has missing/duplicate metric IDs")
    bank_source_sha256 = str(bank_payload.get("source_sha256") or "")
    if (
        not bank_source_sha256
        or str(bank_meta.get("source_sha256") or "") != bank_source_sha256
    ):
        raise ValueError("manifest and bank payload do not bind one bank source hash")

    panel: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    uid_role: dict[str, str] = {}
    role_input_metadata: dict[str, dict[str, Any]] = {}
    for role, paths in input_paths.items():
        role_truth, role_candidates, metadata = _validate_role_artifacts(
            task=args.task,
            role=role,
            truth_path=paths["truth"],
            candidates_path=paths["candidates"],
            freeze_path=paths["freeze"],
            identities_path=paths["identities"],
            truth_release_path=paths["truth_release"],
            candidate_audit_path=paths["candidate_audit"],
            norms=norms,
            manifest_path=manifest_path,
            bank_path=bank_path,
            bank_source_sha256=bank_source_sha256,
            candidate_k=args.candidate_k,
        )
        overlap = sorted(set(panel) & set(role_truth))
        if overlap:
            raise ValueError(f"optimize/select UID overlap: {overlap[:3]}")
        panel.update(role_truth)
        candidates.update(role_candidates)
        uid_role.update({uid: role for uid in role_truth})
        role_input_metadata[role] = metadata

    excluded_uids: set[str] = set()
    exclusion_details: dict[str, Any] = {}
    for path in exclusion_paths:
        indexed = _index(path)
        missing = sorted(set(indexed) - set(norms))
        if missing:
            raise ValueError(
                f"exclusion UIDs absent from canonical task norms: {path}: {missing[:3]}"
            )
        excluded_uids.update(indexed)
        exclusion_details[str(path)] = {
            "sha256": sha256_file(path),
            "uid_count": len(indexed),
        }
    excluded_groups = {split_group_for(norms[uid]) for uid in excluded_uids}

    rows_by_role: dict[str, list[dict[str, Any]]] = {"optimize": [], "select": []}
    candidate_by_role: dict[str, list[dict[str, Any]]] = {"optimize": [], "select": []}
    groups_by_role: dict[str, set[str]] = {"optimize": set(), "select": set()}
    bank_hashes: set[str] = set()
    for uid, truth in sorted(panel.items()):
        if truth.get("task") != args.task:
            raise ValueError(f"panel row has wrong task: {uid}/{truth.get('task')}")
        role = uid_role[uid]
        if truth.get("gepa_role") != role or truth.get("split") != "train":
            raise ValueError(
                f"GEPA row does not preserve its frozen explicit role: {uid}"
            )
        group = split_group_for(norms[uid])
        if group in excluded_groups:
            raise ValueError(
                f"GEPA panel overlaps permanently excluded source group: {uid}"
            )
        _validate_truth_row(truth, bank_ids)
        candidate = candidates[uid]
        if candidate.get("task") != args.task:
            raise ValueError(f"candidate row has wrong task: {uid}")
        values = list(candidate.get("candidates") or [])
        ids = [str(value.get("metric_id") or "") for value in values]
        if len(values) < args.candidate_k or "" in ids or len(ids) != len(set(ids)):
            raise ValueError(
                f"candidate row is invalid or shorter than K={args.candidate_k}: {uid}"
            )
        if not set(ids).issubset(bank_ids):
            raise ValueError(
                f"candidate row references a metric outside the frozen bank: {uid}"
            )
        bank_hash = str(candidate.get("bank_source_sha256") or "")
        if bank_hash != bank_source_sha256:
            raise ValueError(
                f"candidate row bank source differs from manifest/bank: {uid}"
            )
        bank_hashes.add(bank_hash)
        groups_by_role[role].add(group)
        rows_by_role[role].append({**truth, "gepa_source_group": group})
        candidate_by_role[role].append(
            {
                **candidate,
                "candidates": values[: args.candidate_k],
                "gepa_source_group": group,
            }
        )
    if groups_by_role["optimize"] & groups_by_role["select"]:
        raise ValueError("optimize and select overlap by canonical source group")
    if (
        len(rows_by_role["optimize"]) < args.minimum_train
        or len(rows_by_role["select"]) < args.minimum_dev
    ):
        raise ValueError(
            "GEPA panel is underpowered for predeclared minimum support: "
            f"optimize={len(rows_by_role['optimize'])}/{args.minimum_train}, "
            f"select={len(rows_by_role['select'])}/{args.minimum_dev}"
        )
    if bank_hashes != {bank_source_sha256}:
        raise ValueError(
            f"candidate rows do not share one frozen bank hash: {sorted(bank_hashes)}"
        )

    adjudicator_variants = _parse_variants(args.adjudicator_variant, kind="adjudicator")
    verifier_variants = _parse_variants(args.verifier_variant, kind="verifier")
    predeclaration = _validate_predeclaration(
        path=Path(args.predeclaration).resolve(),
        args=args,
        adjudicator_variants=adjudicator_variants,
        verifier_variants=verifier_variants,
        exclusion_paths=exclusion_paths,
        excluded_uids=excluded_uids,
        norms=norms,
    )

    # Each command permits at most one JSON-contract retry per row.  Verifier
    # proposal subsets can only be smaller, but use the full role size for a
    # conservative immutable upper bound.
    adjudicator_requests = sum(
        2 * len(rows_by_role[role]) for role in ROLE_TO_PANEL for _ in ORDERS
    ) * len(adjudicator_variants)
    # Verifier cells are predeclared for every possible adjudicator winner, but
    # those branches are mutually exclusive.  Only the adjudicator variant
    # selected on the frozen select panel may be executed, so the spend bound
    # counts one branch rather than pretending every counterfactual is run.
    verifier_requests = sum(
        2 * len(rows_by_role[role]) for role in ROLE_TO_PANEL for _ in VERIFIER_ORDERS
    ) * len(verifier_variants)
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
                output = (
                    output_root
                    / "runs"
                    / "adjudicator"
                    / variant["name"]
                    / role
                    / f"{order}.jsonl"
                )
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
                            "--manifest",
                            manifest_path,
                            "--candidates",
                            role_paths[role]["candidates"],
                            "--output",
                            output,
                            "--split-role",
                            ROLE_TO_RUNNER_SPLIT[role],
                            *_variant_prompt_args(variant),
                            "--api-base-url",
                            args.api_base_url,
                            "--api-key-file",
                            args.api_key_file,
                            "--max-api-requests",
                            cap,
                            "--model",
                            args.model,
                            "--max-candidates",
                            args.candidate_k,
                            "--concurrency",
                            args.concurrency,
                            "--transport-retries",
                            0,
                            "--order-mode",
                            order,
                            "--resume",
                        ),
                        "direct_batch_command": _command(
                            "scripts.tools.silver_match_v3.adjudicate_gemma",
                            "--manifest",
                            manifest_path,
                            "--candidates",
                            role_paths[role]["candidates"],
                            "--output",
                            output,
                            *_variant_prompt_args(variant),
                            "--model",
                            args.direct_model,
                            "--max-candidates",
                            args.candidate_k,
                            "--batch-size",
                            args.direct_batch_size,
                            "--gpu-memory-utilization",
                            args.gpu_memory_utilization,
                            "--order-mode",
                            order,
                            "--resume",
                        ),
                    }
                )
            consensus = (
                output_root
                / "runs"
                / "adjudicator"
                / variant["name"]
                / role
                / "two_order_consensus.jsonl"
            )
            stages.append(
                {
                    "stage": "adjudicator_consensus",
                    "variant": variant["name"],
                    "role": panel_role,
                    "command": _command(
                        "scripts.tools.silver_match_v3.build_two_order_consensus_proposals",
                        "--original",
                        outputs["original"],
                        "--hashed",
                        outputs["hashed"],
                        "--task",
                        args.task,
                        "--output",
                        consensus,
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
                        "--truth",
                        role_paths[role]["truth"],
                        "--original",
                        outputs["original"],
                        "--hashed",
                        outputs["hashed"],
                        "--panel-role",
                        panel_role,
                        "--explicit-role",
                        role,
                        "--output",
                        output_root
                        / "scores"
                        / "adjudicator"
                        / f"{variant['name']}.{role}.json",
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
                truth_subset = (
                    output_root
                    / "runs"
                    / "verifier"
                    / f"{adj['name']}__{verifier['name']}"
                    / role
                    / "truth.proposals.jsonl"
                )
                candidate_subset = truth_subset.with_name(
                    f"candidates.proposals.top{args.candidate_k}.jsonl"
                )
                stages.extend(
                    [
                        {
                            "stage": "verifier_subset_truth",
                            "adjudicator_variant": adj["name"],
                            "verifier_variant": verifier["name"],
                            "conditional_on_selected_adjudicator_variant": adj["name"],
                            "role": panel_role,
                            "command": _command(
                                "scripts.tools.silver_match_v3.subset_jsonl_by_reference",
                                "--input",
                                role_paths[role]["truth"],
                                "--reference",
                                primary,
                                "--output",
                                truth_subset,
                            ),
                        },
                        {
                            "stage": "verifier_subset_candidates",
                            "adjudicator_variant": adj["name"],
                            "verifier_variant": verifier["name"],
                            "conditional_on_selected_adjudicator_variant": adj["name"],
                            "role": panel_role,
                            "command": _command(
                                "scripts.tools.silver_match_v3.subset_jsonl_by_reference",
                                "--input",
                                role_paths[role]["candidates"],
                                "--reference",
                                primary,
                                "--output",
                                candidate_subset,
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
                            "conditional_on_selected_adjudicator_variant": adj["name"],
                            "role": panel_role,
                            "order": order,
                            "maximum_api_requests": cap,
                            "command": _command(
                                "scripts.tools.silver_match_v3.verify_gemma_api",
                                "--manifest",
                                manifest_path,
                                "--candidates",
                                candidate_subset,
                                "--primary",
                                primary,
                                "--output",
                                output,
                                "--split-role",
                                ROLE_TO_RUNNER_SPLIT[role],
                                *_variant_prompt_args(verifier),
                                "--api-base-url",
                                args.api_base_url,
                                "--api-key-file",
                                args.api_key_file,
                                "--max-api-requests",
                                cap,
                                "--model",
                                args.model,
                                "--max-alternatives",
                                args.candidate_k - 1,
                                "--concurrency",
                                args.concurrency,
                                "--transport-retries",
                                0,
                                "--order-mode",
                                order,
                                "--resume",
                            ),
                            "direct_batch_command": _command(
                                "scripts.tools.silver_match_v3.verify_gemma",
                                "--manifest",
                                manifest_path,
                                "--candidates",
                                candidate_subset,
                                "--primary",
                                primary,
                                "--output",
                                output,
                                *_variant_prompt_args(verifier),
                                "--model",
                                args.direct_model,
                                "--max-alternatives",
                                args.candidate_k - 1,
                                "--batch-size",
                                args.direct_batch_size,
                                "--gpu-memory-utilization",
                                args.gpu_memory_utilization,
                                "--order-mode",
                                order,
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
                            "conditional_on_selected_adjudicator_variant": adj["name"],
                            "role": panel_role,
                            "command": _command(
                                "scripts.tools.silver_match_v3.score_two_order_verifier",
                                "--truth",
                                truth_subset,
                                "--primary",
                                primary,
                                "--original",
                                verify_outputs["original"],
                                "--hashed",
                                verify_outputs["hashed"],
                                "--explicit-role",
                                role,
                                "--output",
                                output_root
                                / "scores"
                                / "verifier"
                                / f"{adj['name']}__{verifier['name']}.{role}.two_order.json",
                            ),
                        },
                        {
                            "stage": "verifier_score_three_order",
                            "adjudicator_variant": adj["name"],
                            "verifier_variant": verifier["name"],
                            "conditional_on_selected_adjudicator_variant": adj["name"],
                            "role": panel_role,
                            "command": _command(
                                "scripts.tools.silver_match_v3.score_three_order_verifier",
                                "--truth",
                                truth_subset,
                                "--primary",
                                primary,
                                "--original",
                                verify_outputs["original"],
                                "--hashed",
                                verify_outputs["hashed"],
                                "--reverse",
                                verify_outputs["reverse"],
                                "--selection-split",
                                "dev" if role == "select" else "optimize",
                                "--explicit-role",
                                role,
                                "--output",
                                output_root
                                / "scores"
                                / "verifier"
                                / f"{adj['name']}__{verifier['name']}.{role}.three_order.json",
                            ),
                        },
                    ]
                )

    plan = {
        "schema_version": "silver-match-v3-explicit-role-task-local-gepa-plan-v1",
        "status": "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE",
        "task": args.task,
        "scientific_scope": {
            "selection_universe": "pre-label frozen explicit optimize/select roles from canonical upstream train groups only",
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
            "verifier_branch_policy": (
                "all possible adjudicator-winner branches are hash-frozen, but exactly one "
                "selected branch may execute; counterfactual branches consume zero requests"
            ),
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
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "predeclaration": predeclaration,
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            "explicit_roles": role_input_metadata,
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
                "truth": {
                    "path": str(paths["truth"]),
                    "sha256": sha256_file(paths["truth"]),
                },
                "candidates": {
                    "path": str(paths["candidates"]),
                    "sha256": sha256_file(paths["candidates"]),
                },
                "count": len(rows_by_role[role]),
                "source_groups": len(groups_by_role[role]),
                "decision_counts": dict(
                    sorted(
                        Counter(
                            str(row.get("decision")) for row in rows_by_role[role]
                        ).items()
                    )
                ),
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
    command_plan.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze_payload = {
        "schema_version": "silver-match-v3-explicit-role-task-local-gepa-freeze-v1",
        "task": args.task,
        "status": plan["status"],
        "command_plan": {
            "path": str(command_plan),
            "sha256": sha256_file(command_plan),
        },
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
    parser.add_argument("--optimize-truth", required=True)
    parser.add_argument("--optimize-candidates", required=True)
    parser.add_argument("--optimize-freeze", required=True)
    parser.add_argument("--optimize-identities", required=True)
    parser.add_argument("--optimize-truth-release", required=True)
    parser.add_argument("--optimize-candidate-audit", required=True)
    parser.add_argument("--select-truth", required=True)
    parser.add_argument("--select-candidates", required=True)
    parser.add_argument("--select-freeze", required=True)
    parser.add_argument("--select-identities", required=True)
    parser.add_argument("--select-truth-release", required=True)
    parser.add_argument("--select-candidate-audit", required=True)
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
    parser.add_argument(
        "--minimum-optimize", dest="minimum_train", type=int, default=30
    )
    parser.add_argument("--minimum-select", dest="minimum_dev", type=int, default=30)
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
        parser.error(
            "candidate K must be at least two and batch/concurrency must be positive"
        )
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("GPU memory utilization must be in (0, 1]")
    return args


def main() -> None:
    print(json.dumps(freeze(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
