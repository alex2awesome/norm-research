#!/usr/bin/env python3
"""Freeze a task-generic, four-role silver-match training handoff.

The task, bank, corpora, and canonical norm universe come only from the
caller-supplied canonical manifest.  Truth sources are separately hash-bound,
then joined through an authoritative caller-supplied role map.  Every frozen
candidate lane must cover the complete task norm universe, not merely the
truth subset.  The output contains training and held-out scoring queues only;
it can never claim production or release readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .build_gemma4_typed_dataset import build as build_gemma_dataset
from .build_nemotron_ce_pairs import DECISIONS, build as build_ce_pairs
from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .freeze_humor_final_stack_handoff import (
    ROLES,
    _index,
    _queue_payload,
    _ref,
    _relocate,
    _split_ce_rows,
    _truth_reason,
    _write_json,
    validate_pilot_recipe,
)
from .prepare_ce_eligible_truth import partition as partition_ce_truth
from .train_nemotron_lora import source_group_key


SCHEMA = "silver-match-v3-task-final-stack-handoff-v1"
TRUTH_ROW_SCHEMA = "silver-match-v3-task-final-stack-truth-v1"
TRUTH_MANIFEST_SCHEMA = "silver-match-v3-task-final-stack-truth-manifest-v1"
PROMPT_MANIFEST_SCHEMA = "silver-match-v3-task-composite-prompt-manifest-v1"
CANDIDATE_SCHEMA = "silver-match-v3-task-full-corpus-candidate-bundle-v1"
TRUTH_SOURCE_REPORT_SCHEMA = "silver-match-v3-task-truth-source-report-v1"
ROLE_MAP_SCHEMA = "silver-match-v3-task-truth-role-map-v1"
PROMPT_COMPONENT_SCHEMA = "silver-match-v3-task-prompt-components-v1"
PILOT_SELECTION_SCHEMA = "silver-match-v3-task-ce-pilot-selection-v1"


def _resolve(raw: Any, base: Path) -> Path:
    path = Path(str(raw or ""))
    if not str(path):
        raise ValueError("empty artifact path")
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _identity_sha(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_bindings(values: Sequence[str], *, label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = normalize_space(name)
        if not separator or not name or not raw_path or name in parsed:
            raise ValueError(f"invalid/duplicate {label} binding: {value!r}")
        parsed[name] = Path(raw_path).resolve()
    if not parsed:
        raise ValueError(f"at least one {label} binding is required")
    return parsed


def freeze_task_scope(
    manifest_path: Path, task: str
) -> tuple[dict[str, Any], Path, str, set[str], dict[str, dict[str, Any]]]:
    """Hash-lock one task's manifest-derived bank and complete corpus universe."""

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("canonical manifest must be an object")
    if manifest.get("source_mode") not in (None, "canonical"):
        raise ValueError("task handoff requires canonical source_mode")
    banks = manifest.get("banks")
    corpora = manifest.get("corpora")
    routing = manifest.get("routing") or {}
    if not isinstance(banks, Mapping) or not isinstance(corpora, Mapping):
        raise ValueError("canonical manifest lacks banks/corpora")
    bank_meta = banks.get(task)
    if not isinstance(bank_meta, Mapping):
        raise ValueError(f"task absent from manifest banks: {task}")
    bank_path = _resolve(bank_meta.get("path"), manifest_path.parent)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if not isinstance(bank, dict) or bank.get("task") != task:
        raise ValueError(f"manifest-derived bank task mismatch: {task}")
    metrics = bank.get("metrics")
    bank_hash = normalize_space(bank.get("source_sha256"))
    if (
        not isinstance(metrics, list)
        or not metrics
        or len(metrics) != int(bank_meta.get("count", -1))
        or bank_hash != normalize_space(bank_meta.get("source_sha256"))
    ):
        raise ValueError(f"manifest-derived bank count/provenance mismatch: {task}")
    metric_ids = [normalize_space(row.get("metric_id")) for row in metrics]
    if not all(metric_ids) or len(metric_ids) != len(set(metric_ids)):
        raise ValueError(f"manifest-derived bank has missing/duplicate metric IDs: {task}")
    if any(row.get("task") not in (None, task) for row in metrics):
        raise ValueError(f"foreign task metric in bank: {task}")

    corpus_names = sorted(
        corpus
        for corpus, meta in corpora.items()
        if isinstance(meta, Mapping) and meta.get("task") == task
    )
    if not corpus_names:
        raise ValueError(f"task has no manifest-derived corpora: {task}")
    if routing:
        for corpus in corpus_names:
            if routing.get(corpus) != task:
                raise ValueError(f"manifest routing differs for {task}/{corpus}")

    norms: dict[str, dict[str, Any]] = {}
    frozen_corpora: dict[str, dict[str, Any]] = {}
    for corpus in corpus_names:
        meta = corpora[corpus]
        if meta.get("coverage_complete") is not True:
            raise ValueError(f"canonical corpus coverage is incomplete: {task}/{corpus}")
        if meta.get("missing_optional_segments") not in (None, []):
            raise ValueError(f"canonical corpus has missing segments: {task}/{corpus}")
        path = _resolve(meta.get("path"), manifest_path.parent)
        ordered_uids: list[str] = []
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            if (
                not uid
                or uid in norms
                or row.get("task") != task
                or row.get("corpus") != corpus
                or not normalize_space(row.get("norm"))
            ):
                raise ValueError(f"foreign/duplicate/invalid canonical norm: {task}/{corpus}/{uid}")
            norms[uid] = row
            ordered_uids.append(uid)
        if len(ordered_uids) != int(meta.get("count", -1)):
            raise ValueError(f"canonical corpus count mismatch: {task}/{corpus}")
        frozen_corpora[corpus] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "norm_count": len(ordered_uids),
            "norm_uids_sha256": _identity_sha(ordered_uids),
        }

    scope = {
        "schema_version": "silver-match-v3-task-final-stack-scope-v1",
        "status": "FROZEN_MANIFEST_DERIVED_TASK_SCOPE",
        "task": task,
        "manifest": _ref(manifest_path),
        "bank": {
            **_ref(bank_path),
            "source_sha256": bank_hash,
            "metric_count": len(metrics),
            "metric_ids_sha256": _identity_sha(metric_ids),
        },
        "corpus_count": len(frozen_corpora),
        "norm_count": len(norms),
        "corpora": frozen_corpora,
        "release_ready": False,
    }
    return scope, bank_path, bank_hash, set(metric_ids), norms


def _validate_truth_report(
    *,
    report_path: Path,
    truth_path: Path,
    source_name: str,
    source_kind: str,
    task: str,
    bank_hash: str,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output = report.get("output")
    if (
        report.get("schema_version") != TRUTH_SOURCE_REPORT_SCHEMA
        or report.get("status") != "FROZEN_TRUSTED_TRUTH_SOURCE"
        or report.get("task") != task
        or report.get("source_name") != source_name
        or report.get("source_kind") != source_kind
        or normalize_space(report.get("bank_source_sha256")) != bank_hash
        or not isinstance(output, Mapping)
        or output.get("sha256") != sha256_file(truth_path)
    ):
        raise ValueError(f"truth source report contract failed: {source_kind}/{source_name}")
    count = sum(1 for _ in read_jsonl(truth_path))
    if int(output.get("count", -1)) != count:
        raise ValueError(f"truth source report count differs: {source_kind}/{source_name}")
    return report


def join_task_truth(
    *,
    task: str,
    bank_hash: str,
    norms: Mapping[str, Mapping[str, Any]],
    role_map_path: Path,
    existing_truth: Mapping[str, Path],
    existing_reports: Mapping[str, Path],
    new_truth: Mapping[str, Path],
    new_reports: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(existing_truth) != set(existing_reports) or set(new_truth) != set(new_reports):
        raise ValueError("truth source names differ from their report bindings")
    source_rows: dict[str, tuple[str, str, dict[str, Any]]] = {}
    source_refs: dict[str, dict[str, Any]] = {}
    for source_kind, paths, reports in (
        ("existing", existing_truth, existing_reports),
        ("new", new_truth, new_reports),
    ):
        for name, path in sorted(paths.items()):
            _validate_truth_report(
                report_path=reports[name],
                truth_path=path,
                source_name=name,
                source_kind=source_kind,
                task=task,
                bank_hash=bank_hash,
            )
            indexed = _index(path, f"{source_kind} truth {name}")
            for uid, row in indexed.items():
                if uid in source_rows:
                    prior = source_rows[uid]
                    raise ValueError(
                        f"truth UID conflict across sources: {uid}/{prior[0]}:{prior[1]}/{source_kind}:{name}"
                    )
                source_rows[uid] = (source_kind, name, row)
            source_refs[f"{source_kind}:{name}"] = {
                "truth": _ref(path),
                "report": _ref(reports[name]),
                "row_count": len(indexed),
            }

    role_rows = _index(role_map_path, "authoritative role map")
    if set(role_rows) != set(source_rows):
        raise ValueError(
            "role map and truth UID universes differ: "
            f"truth_only={len(set(source_rows)-set(role_rows))} "
            f"role_only={len(set(role_rows)-set(source_rows))}"
        )
    rows: list[dict[str, Any]] = []
    group_roles: dict[str, set[str]] = defaultdict(set)
    reason_provenance_only = 0
    for uid in sorted(source_rows):
        source_kind, source_name, source = source_rows[uid]
        role_row = role_rows[uid]
        role = normalize_space(role_row.get("role")).lower()
        corpus = normalize_space(role_row.get("corpus"))
        group = str(role_row.get("source_group") or "").strip()
        permanent_blind = role_row.get("permanent_blind")
        if (
            role_row.get("schema_version") not in (None, ROLE_MAP_SCHEMA)
            or role_row.get("task") != task
            or normalize_space(
                role_row.get("current_bank_source_sha256")
                or role_row.get("bank_source_sha256")
            )
            != bank_hash
            or role not in ROLES
            or (role == "blind") != (permanent_blind is True)
        ):
            raise ValueError(f"invalid role-map task/bank/role/permanent-blind contract: {uid}")
        norm = norms.get(uid)
        if norm is None or norm.get("corpus") != corpus or norm.get("task") != task:
            raise ValueError(f"role map contains foreign task/corpus norm: {uid}/{corpus}")
        canonical_group = source_group_key(dict(norm))
        if group != canonical_group:
            raise ValueError(f"role-map/canonical source_group mismatch: {uid}")
        if (
            source.get("task") != task
            or source.get("corpus") != corpus
            or normalize_space(
                source.get("current_bank_source_sha256")
                or source.get("bank_source_sha256")
            )
            != bank_hash
        ):
            raise ValueError(f"truth row has foreign task/corpus/bank: {uid}")
        source_group = str(source.get("source_group") or group).strip()
        if source_group != group:
            raise ValueError(f"truth/role-map source_group mismatch: {uid}")
        original_split = normalize_space(source.get("split")).lower()
        if original_split:
            allowed = {role} if role != "blind" else {"test", "blind"}
            if original_split not in allowed:
                raise ValueError(f"truth split differs from authoritative role: {uid}")
        decision = normalize_space(source.get("decision"))
        if decision not in DECISIONS:
            raise ValueError(f"truth decision is invalid: {uid}/{decision}")
        reason, provenance_only = _truth_reason(source)
        reason_provenance_only += int(provenance_only)
        group_roles[group].add(role)
        rows.append(
            {
                **source,
                "schema_version": TRUTH_ROW_SCHEMA,
                "task": task,
                "corpus": corpus,
                "source_group": group,
                "pre_handoff_frozen_split": original_split or None,
                "split": role,
                "handoff_role": role,
                "permanent_blind": role == "blind",
                "gradient_eligible": role == "train"
                and source.get("gradient_eligible") is not False,
                "dev_selection_eligible": role == "dev",
                "test_evaluation_only": role == "test",
                "blind_evaluation_only": role == "blind",
                "reason": reason,
                "reason_is_provenance_only": provenance_only,
                "handoff_truth_source_kind": source_kind,
                "handoff_truth_source_name": source_name,
                "current_bank_source_sha256": bank_hash,
            }
        )
    crossed = {group: roles for group, roles in group_roles.items() if len(roles) > 1}
    if crossed:
        raise ValueError(f"truth source groups cross authoritative roles: {len(crossed)}")
    role_counts = Counter(str(row["split"]) for row in rows)
    missing_roles = [role for role in ROLES if role_counts[role] == 0]
    if missing_roles:
        raise ValueError(f"truth role map does not preserve all four roles: {missing_roles}")
    corpus_counts = Counter(str(row["corpus"]) for row in rows)
    source_kind_counts = Counter(str(row["handoff_truth_source_kind"]) for row in rows)
    if source_kind_counts["existing"] == 0 or source_kind_counts["new"] == 0:
        raise ValueError("both existing and new truth are required")
    rows.sort(key=lambda row: str(row["norm_uid"]))
    return rows, {
        "schema_version": TRUTH_MANIFEST_SCHEMA,
        "status": "FROZEN_HASH_BOUND_SOURCE_DISJOINT_FOUR_ROLE_TRUTH",
        "task": task,
        "bank_source_sha256": bank_hash,
        "truth_rows": len(rows),
        "role_counts": {role: role_counts[role] for role in ROLES},
        "corpus_counts": dict(sorted(corpus_counts.items())),
        "truth_source_kind_counts": dict(sorted(source_kind_counts.items())),
        "truth_source_count": len(source_refs),
        "uid_conflicts_across_sources": 0,
        "source_groups_crossing_roles": 0,
        "test_or_blind_gradient_eligible": 0,
        "permanent_blind_rows": role_counts["blind"],
        "provenance_only_reason_rows": reason_provenance_only,
        "inputs": {
            "role_map": _ref(role_map_path),
            "truth_sources": source_refs,
        },
    }


def load_full_corpus_candidate_bundle(
    *,
    bundle_path: Path,
    task: str,
    bank_hash: str,
    bank_ids: set[str],
    scope: Mapping[str, Any],
    norms: Mapping[str, Mapping[str, Any]],
) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    lanes = payload.get("lanes")
    corpus_contract = payload.get("corpora")
    if (
        payload.get("schema_version") != CANDIDATE_SCHEMA
        or payload.get("status") != "FROZEN_FULL_CORPUS_DIVERSE_CANDIDATE_LANES"
        or payload.get("task") != task
        or normalize_space(payload.get("bank_source_sha256")) != bank_hash
        or payload.get("selection_split") != "dev"
        or payload.get("test_or_blind_labels_used_for_selection") is not False
        or not isinstance(lanes, Mapping)
        or len(lanes) < 2
        or not isinstance(corpus_contract, Mapping)
    ):
        raise ValueError("full-corpus candidate bundle contract failed")
    expected_corpora = scope["corpora"]
    if set(corpus_contract) != set(expected_corpora):
        raise ValueError("candidate bundle corpus universe differs from manifest task scope")
    for corpus, expected in expected_corpora.items():
        observed = corpus_contract[corpus]
        if (
            not isinstance(observed, Mapping)
            or int(observed.get("count", -1)) != int(expected["norm_count"])
            or normalize_space(observed.get("canonical_norm_sha256"))
            != normalize_space(expected["sha256"])
        ):
            raise ValueError(f"candidate bundle corpus count/hash drift: {task}/{corpus}")

    specs: list[tuple[str, Path]] = []
    lane_audits: dict[str, Any] = {}
    wanted = set(norms)
    for lane, binding in sorted(lanes.items()):
        if not isinstance(binding, Mapping):
            raise ValueError(f"candidate lane binding is invalid: {lane}")
        path = _resolve(binding.get("path"), bundle_path.parent)
        if sha256_file(path) != normalize_space(binding.get("sha256")):
            raise ValueError(f"candidate lane SHA drift: {lane}")
        seen: set[str] = set()
        counts: Counter[str] = Counter()
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            corpus = normalize_space(row.get("corpus"))
            if uid in seen or uid not in norms:
                raise ValueError(f"candidate lane has duplicate/foreign norm: {lane}/{uid}")
            if (
                row.get("task") != task
                or norms[uid].get("corpus") != corpus
                or normalize_space(
                    row.get("bank_source_sha256")
                    or row.get("current_bank_source_sha256")
                )
                != bank_hash
            ):
                raise ValueError(f"candidate lane has foreign task/corpus/bank: {lane}/{uid}")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"candidate lane has no candidates: {lane}/{uid}")
            ids = [normalize_space(candidate.get("metric_id")) for candidate in candidates]
            if not all(ids) or len(ids) != len(set(ids)) or not set(ids) <= bank_ids:
                raise ValueError(f"candidate lane has invalid bank metrics: {lane}/{uid}")
            seen.add(uid)
            counts[corpus] += 1
        if seen != wanted:
            raise ValueError(
                f"candidate lane is not full corpus: {lane}/missing={len(wanted-seen)}/foreign={len(seen-wanted)}"
            )
        expected_counts = {
            corpus: int(value["norm_count"]) for corpus, value in expected_corpora.items()
        }
        if dict(counts) != expected_counts:
            raise ValueError(f"candidate lane per-corpus counts differ: {lane}")
        specs.append((str(lane), path))
        lane_audits[str(lane)] = {
            **_ref(path),
            "norm_count": len(seen),
            "corpus_counts": dict(sorted(counts.items())),
        }
    return specs, {
        "schema_version": CANDIDATE_SCHEMA,
        "status": "VALIDATED_ALL_FROZEN_FULL_CORPUS_LANES",
        "task": task,
        "bundle": _ref(bundle_path),
        "bank_source_sha256": bank_hash,
        "candidate_lane_count": len(specs),
        "all_frozen_candidate_inputs_used": True,
        "every_lane_covers_every_manifest_norm": True,
        "expected_corpus_count": len(expected_corpora),
        "expected_norm_count": len(norms),
        "selection_split": "dev",
        "test_or_blind_labels_used_for_selection": False,
        "lanes": lane_audits,
    }


def freeze_task_composite_prompt(
    *,
    component_manifest_path: Path,
    task: str,
    output_path: Path,
    output_manifest_path: Path,
    published_output_path: Path,
) -> dict[str, Any]:
    payload = json.loads(component_manifest_path.read_text(encoding="utf-8"))
    guide = payload.get("guide")
    rules = payload.get("rules")
    audits = payload.get("train_only_judge_audits") or []
    if (
        payload.get("schema_version") != PROMPT_COMPONENT_SCHEMA
        or payload.get("status") != "FROZEN_TASK_LOCAL_RULE_COMPONENTS"
        or payload.get("task") != task
        or not isinstance(guide, Mapping)
        or not isinstance(rules, list)
        or not rules
        or not isinstance(audits, list)
    ):
        raise ValueError("task prompt component manifest contract failed")

    components: list[tuple[str, Path, str]] = []
    seen_names: set[str] = set()
    for value in [guide, *rules]:
        if not isinstance(value, Mapping):
            raise ValueError("prompt component is not an object")
        name = normalize_space(value.get("name"))
        path = _resolve(value.get("path"), component_manifest_path.parent)
        expected = normalize_space(value.get("sha256"))
        if not name or name in seen_names or sha256_file(path) != expected:
            raise ValueError(f"prompt component missing/duplicate/hash drift: {name}")
        seen_names.add(name)
        components.append((name, path, expected))

    audit_refs: list[dict[str, Any]] = []
    forbidden_uids: set[str] = set()
    component_hashes = {name: digest for name, _, digest in components}
    for value in audits:
        if not isinstance(value, Mapping):
            raise ValueError("train-only judge audit binding is not an object")
        name = normalize_space(value.get("name"))
        component = normalize_space(value.get("component"))
        path = _resolve(value.get("path"), component_manifest_path.parent)
        if sha256_file(path) != normalize_space(value.get("sha256")):
            raise ValueError(f"train-only judge audit hash drift: {name}")
        audit = json.loads(path.read_text(encoding="utf-8"))
        role = audit.get("role_contract") or {}
        blind_reads = role.get(
            "test_or_blind_rows_read_for_rule_authorship",
            role.get("blind_or_test_rows_read_for_rule_authorship"),
        )
        prompt = audit.get("prompt") or {}
        if (
            audit.get("schema_version") != "silver-match-v3-task-gepa-judge-audit-v1"
            or audit.get("status")
            != "FROZEN_TRAIN_ONLY_PROMPT_REFINEMENT_BEFORE_LABELING"
            or audit.get("task") != task
            or role.get("allowed_role") != "train"
            or int(role.get("dev_rows_read_for_rule_authorship", -1)) != 0
            or int(blind_reads if blind_reads is not None else -1) != 0
            or int(role.get("resolver_votes_or_outcomes_read", -1)) != 0
            or role.get("rule_authorship_completed_before_resolver_labels") is not True
            or component not in component_hashes
            or normalize_space(prompt.get("sha256")) != component_hashes[component]
        ):
            raise ValueError(f"train-only judge audit firewall failed: {name}")
        forbidden_uids.update(
            normalize_space(row.get("norm_uid"))
            for row in audit.get("judged_train_disagreements") or []
            if isinstance(row, Mapping) and normalize_space(row.get("norm_uid"))
        )
        audit_refs.append(
            {
                "name": name,
                "component": component,
                "artifact": _ref(path),
                "train_only_authorship_validated": True,
            }
        )

    sections = [
        f"# Frozen {task} typed-adjudicator rules",
        "",
        "Rules only: no source-pack items, truth labels, votes, outcomes, or example UIDs are included.",
    ]
    for name, path, _ in components:
        sections.extend(("", f"## {name}", "", path.read_text(encoding="utf-8").strip()))
    composite = "\n".join(sections).rstrip() + "\n"
    leaked = sorted(uid for uid in forbidden_uids if uid in composite)
    if leaked or "judged_train_disagreements" in composite or "preferred_key" in composite:
        raise ValueError("judge-audit truth example or UID leaked into composite prompt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(composite)
    output_ref = _ref(output_path)
    output_ref["path"] = str(published_output_path.resolve())
    manifest = {
        "schema_version": PROMPT_MANIFEST_SCHEMA,
        "status": "FROZEN_TASK_LOCAL_RULES_ONLY_COMPOSITE",
        "task": task,
        "input_manifest": _ref(component_manifest_path),
        "component_order": [name for name, _, _ in components],
        "components": {
            name: {**_ref(path), "expected_sha256": digest}
            for name, path, digest in components
        },
        "train_only_judge_audits": audit_refs,
        "truth_examples_included": False,
        "truth_labels_votes_or_outcomes_included": False,
        "example_uids_included": False,
        "output": output_ref,
    }
    _write_json(output_manifest_path, manifest)
    return manifest


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    task = normalize_space(args.task)
    if not task:
        raise ValueError("task is empty")
    final_root = Path(args.output_root).resolve()
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite handoff root: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    stage = final_root.parent / f".{final_root.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    try:
        manifest_path = Path(args.manifest).resolve()
        scope, bank_path, bank_hash, bank_ids, norms = freeze_task_scope(
            manifest_path, task
        )
        scope_path = stage / "TASK_SCOPE.json"
        _write_json(scope_path, scope)

        existing_truth = _parse_bindings(args.existing_truth, label="existing truth")
        existing_reports = _parse_bindings(
            args.existing_truth_report, label="existing truth report"
        )
        new_truth = _parse_bindings(args.new_truth, label="new truth")
        new_reports = _parse_bindings(args.new_truth_report, label="new truth report")
        joined, truth_report = join_task_truth(
            task=task,
            bank_hash=bank_hash,
            norms=norms,
            role_map_path=Path(args.role_map).resolve(),
            existing_truth=existing_truth,
            existing_reports=existing_reports,
            new_truth=new_truth,
            new_reports=new_reports,
        )
        truth_path = stage / "truth" / "truth.joined.all.jsonl"
        write_jsonl(truth_path, joined)
        truth_report["output"] = {
            "path": str(final_root / truth_path.relative_to(stage)),
            "sha256": sha256_file(truth_path),
            "count": len(joined),
        }
        truth_manifest = stage / "truth" / "MANIFEST.json"
        _write_json(truth_manifest, truth_report)

        eligible, typed_only, partition_report = partition_ce_truth(truth_path)
        ce_truth = stage / "truth" / "truth.ce-eligible.jsonl"
        typed_only_path = stage / "truth" / "truth.gemma-only.jsonl"
        write_jsonl(ce_truth, eligible)
        write_jsonl(typed_only_path, typed_only)
        partition_report["outputs"] = {
            "eligible": {
                "path": str(final_root / ce_truth.relative_to(stage)),
                "sha256": sha256_file(ce_truth),
                "count": len(eligible),
            },
            "typed_only": {
                "path": str(final_root / typed_only_path.relative_to(stage)),
                "sha256": sha256_file(typed_only_path),
                "count": len(typed_only),
            },
        }
        partition_report = _relocate(partition_report, stage, final_root)
        ce_partition_report = stage / "truth" / "CE_PARTITION.json"
        _write_json(ce_partition_report, partition_report)

        specs, candidate_audit = load_full_corpus_candidate_bundle(
            bundle_path=Path(args.candidate_bundle).resolve(),
            task=task,
            bank_hash=bank_hash,
            bank_ids=bank_ids,
            scope=scope,
            norms=norms,
        )
        candidate_args = [f"{lane}={path}" for lane, path in specs]

        prompt_path = stage / "prompts" / "TASK_GEMMA_COMPOSITE.txt"
        prompt_manifest = stage / "prompts" / "MANIFEST.json"
        freeze_task_composite_prompt(
            component_manifest_path=Path(args.prompt_components).resolve(),
            task=task,
            output_path=prompt_path,
            output_manifest_path=prompt_manifest,
            published_output_path=final_root / prompt_path.relative_to(stage),
        )

        ce_pair_path = stage / "ce" / "all.pairs.jsonl"
        ce_builder_report = stage / "ce" / "BUILDER_REPORT.json"
        ce_rows, ce_report = build_ce_pairs(
            argparse.Namespace(
                manifest=str(manifest_path),
                task=task,
                bank=str(bank_path),
                truth=[str(ce_truth)],
                split_assignments=None,
                candidates=candidate_args,
                hierarchy=str(Path(args.hierarchy).resolve()),
                maximum_pairs=args.maximum_pairs,
                global_negatives_per_norm=args.global_negatives_per_norm,
                context_chars=args.ce_context_chars,
                seed=args.pair_seed,
            )
        )
        write_jsonl(ce_pair_path, ce_rows)
        ce_report = _relocate(ce_report, stage, final_root)
        ce_report["output"] = {
            "path": str(final_root / ce_pair_path.relative_to(stage)),
            "sha256": sha256_file(ce_pair_path),
            "count": len(ce_rows),
        }
        _write_json(ce_builder_report, ce_report)

        ce_buckets, ce_split_audit = _split_ce_rows(ce_rows, eligible)
        ce_paths = {role: stage / "ce" / f"{role}.pairs.jsonl" for role in ROLES}
        for role, path in ce_paths.items():
            write_jsonl(path, ce_buckets[role])
        ce_split_report = stage / "ce" / "SPLIT_REPORT.json"
        _write_json(
            ce_split_report,
            {
                "schema_version": "silver-match-v3-task-final-ce-four-role-split-v1",
                "status": "FROZEN_SOURCE_DISJOINT_FOUR_ROLE_CE_INPUTS",
                "task": task,
                "audit": ce_split_audit,
                "input": {
                    "path": str(final_root / ce_pair_path.relative_to(stage)),
                    "sha256": sha256_file(ce_pair_path),
                },
                "outputs": {
                    role: {
                        "path": str(final_root / path.relative_to(stage)),
                        "sha256": sha256_file(path),
                        "count": len(ce_buckets[role]),
                        "training_access": (
                            "ALLOWED"
                            if role == "train"
                            else "SELECTION_ONLY"
                            if role == "dev"
                            else "FORBIDDEN"
                        ),
                    }
                    for role, path in ce_paths.items()
                },
            },
        )

        gemma_report = stage / "gemma" / "DATASET_REPORT.json"
        gemma_dir = stage / "gemma" / "dataset"
        gemma_buckets, gemma_dataset_report = build_gemma_dataset(
            argparse.Namespace(
                manifest=str(manifest_path),
                task=task,
                bank=str(bank_path),
                truth=[str(truth_path)],
                split_assignments=None,
                candidates=candidate_args,
                hierarchy=str(Path(args.hierarchy).resolve()),
                prompt=str(prompt_path),
                max_candidates=args.gemma_max_candidates,
                order_seed=args.gemma_order_seed,
                context_chars=args.gemma_context_chars,
                description_chars=args.gemma_description_chars,
                example_chars=args.gemma_example_chars,
                max_examples=args.gemma_max_examples,
            )
        )
        gemma_dir.mkdir(parents=True, exist_ok=False)
        gemma_paths = {role: gemma_dir / f"{role}.jsonl" for role in ROLES}
        for role, path in gemma_paths.items():
            if not gemma_buckets[role]:
                raise ValueError(f"Gemma dataset lost required role: {role}")
            write_jsonl(path, gemma_buckets[role])
        gemma_dataset_report = _relocate(gemma_dataset_report, stage, final_root)
        gemma_dataset_report["outputs"] = {
            role: {
                "path": str(final_root / path.relative_to(stage)),
                "sha256": sha256_file(path),
                "count": len(gemma_buckets[role]),
            }
            for role, path in gemma_paths.items()
        }
        _write_json(gemma_report, gemma_dataset_report)

        recipe, pilot_audit = validate_pilot_recipe(
            Path(args.pilot_selection).resolve(),
            ce_model=Path(args.ce_model).resolve(),
            task=task,
        )
        queue = _queue_payload(
            args=args,
            final_root=final_root,
            truth_manifest=truth_manifest,
            ce_partition_report=ce_partition_report,
            ce_builder_report=ce_builder_report,
            ce_split_report=ce_split_report,
            ce_paths=ce_paths,
            gemma_report=gemma_report,
            gemma_paths=gemma_paths,
            gemma_prompt=prompt_path,
            gemma_prompt_manifest=prompt_manifest,
            recipe=recipe,
            pilot_audit=pilot_audit,
            candidate_audit=candidate_audit,
            task=task,
        )
        queue = _relocate(queue, stage, final_root)
        queue_path = stage / "FINAL_STACK_QUEUE.json"
        _write_json(queue_path, queue)

        handoff = {
            "schema_version": SCHEMA,
            "status": "FROZEN_HANDOFF_NOT_PRODUCTION_OR_RELEASE_READY",
            "task": task,
            "scope": {
                "path": str(final_root / scope_path.relative_to(stage)),
                "sha256": sha256_file(scope_path),
                "expected_corpora": scope["corpus_count"],
                "expected_norms": scope["norm_count"],
            },
            "truth_manifest": {
                "path": str(final_root / truth_manifest.relative_to(stage)),
                "sha256": sha256_file(truth_manifest),
            },
            "candidate_bundle": candidate_audit,
            "prompt_manifest": {
                "path": str(final_root / prompt_manifest.relative_to(stage)),
                "sha256": sha256_file(prompt_manifest),
            },
            "ce_builder_report": {
                "path": str(final_root / ce_builder_report.relative_to(stage)),
                "sha256": sha256_file(ce_builder_report),
            },
            "ce_split_report": {
                "path": str(final_root / ce_split_report.relative_to(stage)),
                "sha256": sha256_file(ce_split_report),
            },
            "gemma_dataset_report": {
                "path": str(final_root / gemma_report.relative_to(stage)),
                "sha256": sha256_file(gemma_report),
            },
            "queue": {
                "path": str(final_root / queue_path.relative_to(stage)),
                "sha256": sha256_file(queue_path),
            },
            "readiness": queue["readiness"],
        }
        handoff_path = stage / "HANDOFF_MANIFEST.json"
        _write_json(handoff_path, handoff)
        os.rename(stage, final_root)
        return {
            **handoff,
            "manifest": str(final_root / "HANDOFF_MANIFEST.json"),
            "manifest_sha256": sha256_file(final_root / "HANDOFF_MANIFEST.json"),
        }
    except BaseException:
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "task",
        "manifest",
        "hierarchy",
        "role_map",
        "candidate_bundle",
        "prompt_components",
        "pilot_selection",
        "ce_model",
        "gemma_model",
        "python",
        "ce_trainer",
        "ce_scorer",
        "gemma_trainer",
        "runtime_root",
        "output_root",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--existing-truth", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--existing-truth-report", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument("--new-truth", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--new-truth-report", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument("--ce-seed", action="append", type=int, required=True)
    parser.add_argument("--gemma-seed", type=int, default=94137)
    parser.add_argument("--pair-seed", type=int, default=20260715)
    parser.add_argument("--maximum-pairs", type=int, default=400_000)
    parser.add_argument("--global-negatives-per-norm", type=int, default=4)
    parser.add_argument("--ce-context-chars", type=int, default=1600)
    parser.add_argument("--gemma-max-candidates", type=int, default=8)
    parser.add_argument("--gemma-order-seed", type=int, default=2026071501)
    parser.add_argument("--gemma-context-chars", type=int, default=1400)
    parser.add_argument("--gemma-description-chars", type=int, default=520)
    parser.add_argument("--gemma-example-chars", type=int, default=180)
    parser.add_argument("--gemma-max-examples", type=int, default=2)
    args = parser.parse_args(argv)
    if len(args.ce_seed) != 2 or len(set(args.ce_seed)) != 2:
        parser.error("provide exactly two distinct --ce-seed values")
    positive = (
        args.maximum_pairs,
        args.ce_context_chars,
        args.gemma_max_candidates,
        args.gemma_context_chars,
        args.gemma_description_chars,
        args.gemma_example_chars,
        args.gemma_max_examples,
    )
    if any(value <= 0 for value in positive) or args.global_negatives_per_norm < 0:
        parser.error("pair/prompt sizes must be positive and global negatives nonnegative")
    return args


def main() -> None:
    result = freeze(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
