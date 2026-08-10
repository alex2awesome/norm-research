#!/usr/bin/env python3
"""Build and freeze the post-consensus Humor final-stack handoff.

This is deliberately a *handoff*, not a release script.  It joins the older
trusted truth to the newly resolved exact-consensus pack, builds the two
task-local training datasets, and freezes commands for two final CE seeds and
one typed Gemma run.  It never opens test/blind labels for model selection and
it always records that production adjudication and release validation remain
outstanding.

All outputs are assembled in a private sibling directory and published with
one rename.  Existing output roots are never reused or overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .build_gemma4_typed_dataset import build as build_gemma_dataset
from .build_nemotron_ce_pairs import build as build_ce_pairs
from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .prepare_ce_eligible_truth import partition as partition_ce_truth


SCHEMA = "silver-match-v3-humor-final-stack-handoff-v1"
TRUTH_SCHEMA = "silver-match-v3-humor-final-stack-truth-v1"
TRUTH_MANIFEST_SCHEMA = "silver-match-v3-humor-final-stack-truth-manifest-v1"
QUEUE_SCHEMA = "silver-match-v3-humor-final-stack-queue-v1"
TASK = "humor"
ROLES = ("train", "dev", "test", "blind")
FINAL_EXPOSURES = (100_000, 200_000, 400_000)
FORBIDDEN_SELECTION_ROLES = {"test", "blind"}
EXPECTED_PROMPT_COMPONENT_SHA256 = {
    "GUIDE": "03e95ac5e072a9c79e2c88375753502fa82748d7152b1fad32ca0bffad4b19ad",
    "R1": "05e9ad7e8727e2e0811f7fa51979066dbdd75cb38fb05890ed377149a72df5e4",
    "R2": "e559e11e6bf2532be602eab7466d3503c829ee06819aad2f953a5cbbec8b7a62",
    "R3": "d0a4e8128e1c4818f24204be32ed0ae1638edf7c40a028afabca89f69cb7cb9f",
    "R4": "edf85f24879a5a72767a9aa631dc9ad10bf4a24bb39bc793ca16540782b703fb",
    "R5": "b2065ea4ce43c001b8f0b963f321b57d30ac2a11e96f06250a767a0c0297a756",
    "R6": "6fff8ea592b96cd48e347ffd9bdbde463ce4975a06475d339c548da22bfc4bf4",
    "R7": "691a690fdb8d81d5f82720a262e0a2d434d6c29cc0a5d75953bedf517d3dce01",
    "R8": "f39f7c26f2948011e3df13dca0facee3341e141fa87c80d50ce7acc1c0465837",
    "R9": "de4fe9597f28c8636d5e722908928c2afa2bf28543dc0959caa0047c1f4013d5",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ref(path: Path, **extra: Any) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _index(path: Path, label: str) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = normalize_space(row.get("norm_uid"))
        if not uid or uid in indexed:
            raise ValueError(f"{label} has missing/duplicate norm_uid: {uid!r}")
        indexed[uid] = row
    if not indexed:
        raise ValueError(f"{label} is empty")
    return indexed


def _bound_output(
    report_path: Path,
    artifact_path: Path,
    *,
    expected_schema: str,
    expected_status: str,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != expected_schema
        or report.get("status") != expected_status
        or report.get("task") != TASK
    ):
        raise ValueError(f"untrusted truth report contract: {report_path}")
    output = report.get("output")
    if output is None:
        output = (report.get("outputs") or {}).get("all")
    if not isinstance(output, Mapping):
        raise ValueError(f"truth report lacks a bound all/output artifact: {report_path}")
    if output.get("sha256") != sha256_file(artifact_path):
        raise ValueError(f"truth report output SHA differs: {artifact_path}")
    if int(output.get("count", -1)) != sum(1 for _ in read_jsonl(artifact_path)):
        raise ValueError(f"truth report output count differs: {artifact_path}")
    return report


def _bank_source_hash(bank_path: Path, *, task: str = TASK) -> str:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    value = normalize_space(bank.get("source_sha256")) if isinstance(bank, dict) else ""
    if bank.get("task") != task or not value:
        raise ValueError(f"bank is not the canonical {task} bank or lacks source_sha256")
    return value


def _truth_bank_hash(row: Mapping[str, Any]) -> str:
    return normalize_space(
        row.get("current_bank_source_sha256") or row.get("bank_source_sha256")
    )


def _truth_reason(row: Mapping[str, Any]) -> tuple[str, bool]:
    reason = normalize_space(row.get("reason") or row.get("rationale"))
    if reason:
        return reason, False
    # Some older trusted Sonnet rows retained only the audit reason for why the
    # label entered the trusted CE inventory.  Do not hallucinate a semantic
    # explanation; provide an explicit provenance-only target at the low
    # weighted reason field while preserving the high-weight decision/leaf.
    provenance = normalize_space(row.get("trusted_ce_reason"))
    if provenance:
        return f"Trusted existing label ({provenance}); no free-text rationale retained.", True
    raise ValueError(f"truth row has no rationale or trusted provenance: {row.get('norm_uid')}")


def _normalize_existing(
    row: Mapping[str, Any], *, bank_hash: str
) -> tuple[dict[str, Any], bool]:
    uid = normalize_space(row.get("norm_uid"))
    split = normalize_space(row.get("split")).lower()
    if split not in ROLES:
        raise ValueError(f"existing truth has invalid frozen split: {uid}/{split}")
    if row.get("task") != TASK or _truth_bank_hash(row) != bank_hash:
        raise ValueError(f"existing truth task/bank mismatch: {uid}")
    group = str(row.get("source_group") or "").strip()
    if not group:
        raise ValueError(f"existing truth lacks source_group: {uid}")
    reason, provenance_only = _truth_reason(row)
    gradient = row.get("gradient_eligible")
    if split in FORBIDDEN_SELECTION_ROLES and gradient is True:
        raise ValueError(f"held-out existing truth is gradient eligible: {uid}")
    rendered = {
        **row,
        "schema_version": TRUTH_SCHEMA,
        "split": split,
        "handoff_role": split,
        "source_group": group,
        "reason": reason,
        "reason_is_provenance_only": provenance_only,
        "gradient_eligible": bool(split == "train" and gradient is not False),
        "dev_selection_eligible": split == "dev",
        "test_evaluation_only": split == "test",
        "blind_evaluation_only": split == "blind",
        "handoff_truth_source": "existing_canonical_trusted_truth",
        "current_bank_source_sha256": bank_hash,
    }
    return rendered, provenance_only


def _normalize_consensus(
    row: Mapping[str, Any], *, bank_hash: str
) -> tuple[dict[str, Any], bool]:
    uid = normalize_space(row.get("norm_uid"))
    frozen_split = normalize_space(row.get("split")).lower()
    collection_role = normalize_space(row.get("collection_role")).lower()
    expected = {"train": "train", "dev": "dev", "blind": "test"}
    if collection_role not in expected or frozen_split != expected[collection_role]:
        raise ValueError(
            f"consensus frozen split/collection role mismatch: "
            f"{uid}/{frozen_split}/{collection_role}"
        )
    if row.get("task") != TASK or _truth_bank_hash(row) != bank_hash:
        raise ValueError(f"consensus truth task/bank mismatch: {uid}")
    group = str(row.get("source_group") or row.get("split_group") or "").strip()
    if not group:
        raise ValueError(f"consensus truth lacks source_group: {uid}")
    reason, provenance_only = _truth_reason(row)
    role = collection_role
    if role == "blind" and row.get("blind_evaluation_only") is not True:
        raise ValueError(f"consensus blind row lacks frozen blind-only flag: {uid}")
    if role == "train" and row.get("training_eligible") is not True:
        raise ValueError(f"consensus train row lacks frozen training flag: {uid}")
    if role == "dev" and row.get("dev_selection_eligible") is not True:
        raise ValueError(f"consensus dev row lacks frozen selection flag: {uid}")
    rendered = {
        **row,
        "schema_version": TRUTH_SCHEMA,
        "pre_handoff_frozen_split": frozen_split,
        # The collection pack encoded permanent blind as legacy split=test.
        # The downstream builders already support an explicit blind role, so
        # restore it here without losing the original value above.
        "split": role,
        "handoff_role": role,
        "source_group": group,
        "reason": reason,
        "reason_is_provenance_only": provenance_only,
        "gradient_eligible": role == "train",
        "dev_selection_eligible": role == "dev",
        "test_evaluation_only": role == "test",
        "blind_evaluation_only": role == "blind",
        "handoff_truth_source": "new_exact_multi_pass_consensus",
        "current_bank_source_sha256": bank_hash,
    }
    return rendered, provenance_only


def join_truth(
    *,
    existing_path: Path,
    existing_report_path: Path,
    consensus_path: Path,
    consensus_manifest_path: Path,
    bank_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join two hash-bound truth sources while preserving all four roles."""

    bank_hash = _bank_source_hash(bank_path)
    existing_report = _bound_output(
        existing_report_path,
        existing_path,
        expected_schema="silver-match-v3-humor-ce-existing-truth-report-v1",
        expected_status="CANONICAL_EXISTING_TRUTH_READY",
    )
    consensus_manifest = _bound_output(
        consensus_manifest_path,
        consensus_path,
        expected_schema="silver-match-v3-consensus-training-truth-manifest-v1",
        expected_status="COMPLETE_EXACT_CONSENSUS_WITH_FROZEN_SPLITS",
    )
    if (
        normalize_space(existing_report.get("bank_source_sha256")) != bank_hash
        or normalize_space(consensus_manifest.get("task")) != TASK
    ):
        raise ValueError("truth reports do not bind the current Humor bank/task")

    existing = _index(existing_path, "existing truth")
    consensus = _index(consensus_path, "consensus truth")
    overlap = sorted(set(existing) & set(consensus))
    if overlap:
        raise ValueError(f"truth UID conflict across sources: {overlap[:5]}")

    rows: list[dict[str, Any]] = []
    provenance_only = 0
    for uid in sorted(existing):
        rendered, weak_reason = _normalize_existing(existing[uid], bank_hash=bank_hash)
        rows.append(rendered)
        provenance_only += int(weak_reason)
    for uid in sorted(consensus):
        rendered, weak_reason = _normalize_consensus(consensus[uid], bank_hash=bank_hash)
        rows.append(rendered)
        provenance_only += int(weak_reason)

    group_roles: dict[str, set[str]] = defaultdict(set)
    uid_roles: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_roles[str(row["source_group"])].add(str(row["split"]))
        uid_roles[str(row["norm_uid"])].add(str(row["split"]))
    crossed_groups = {key: value for key, value in group_roles.items() if len(value) > 1}
    crossed_uids = {key: value for key, value in uid_roles.items() if len(value) > 1}
    if crossed_groups or crossed_uids:
        raise ValueError(
            "joined truth violates source-disjoint roles: "
            f"groups={len(crossed_groups)} uids={len(crossed_uids)}"
        )
    counts = Counter(str(row["split"]) for row in rows)
    missing_roles = [role for role in ROLES if not counts[role]]
    if missing_roles:
        raise ValueError(f"joined truth does not retain all four roles: {missing_roles}")
    if any(
        row["gradient_eligible"]
        for row in rows
        if row["split"] in FORBIDDEN_SELECTION_ROLES
    ):
        raise AssertionError("test/blind truth became gradient eligible")

    rows.sort(key=lambda row: str(row["norm_uid"]))
    report = {
        "schema_version": TRUTH_MANIFEST_SCHEMA,
        "status": "FROZEN_SOURCE_DISJOINT_FOUR_ROLE_TRUTH",
        "task": TASK,
        "bank_source_sha256": bank_hash,
        "truth_rows": len(rows),
        "source_counts": {
            "existing_canonical_trusted_truth": len(existing),
            "new_exact_multi_pass_consensus": len(consensus),
        },
        "role_counts": {role: counts[role] for role in ROLES},
        "decision_counts": dict(sorted(Counter(str(row["decision"]) for row in rows).items())),
        "source_group_count": len(group_roles),
        "source_groups_crossing_roles": 0,
        "norm_uids_crossing_roles": 0,
        "uid_conflicts_across_truth_sources": 0,
        "test_or_blind_gradient_eligible": 0,
        "provenance_only_reason_rows": provenance_only,
        "blind_role_restoration": {
            "source_encoding": "collection_role=blind, pre_handoff_frozen_split=test",
            "downstream_split": "blind",
            "original_split_retained_in": "pre_handoff_frozen_split",
        },
        "inputs": {
            "existing_truth": _ref(existing_path),
            "existing_truth_report": _ref(existing_report_path),
            "consensus_truth": _ref(consensus_path),
            "consensus_truth_manifest": _ref(consensus_manifest_path),
            "bank": _ref(bank_path),
        },
    }
    return rows, report


def load_full_candidate_bundle(
    path: Path, *, bank_hash: str
) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    """Resolve every hash-bound retriever input from a frozen capture artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "silver-match-v3-candidate-capture-sequence-v1":
        raise ValueError("candidate capture artifact has an unsupported schema")
    if (
        payload.get("selection_split") != "dev"
        or payload.get("test_labels_used_for_selection") is not False
    ):
        raise ValueError("candidate capture was not frozen with a dev-only firewall")
    inputs = payload.get("candidate_inputs")
    available = payload.get("available_lanes")
    sequence = payload.get("selected_sequence")
    if not isinstance(inputs, Mapping) or len(inputs) < 2:
        raise ValueError("full diverse bundle requires at least two retriever inputs")
    if not isinstance(available, list) or not available or not isinstance(sequence, list):
        raise ValueError("candidate capture lacks available/selected lane identities")
    roots = {str(value).split(":", 1)[0] for value in available}
    selected_roots = {str(value).split(":", 1)[0] for value in sequence}
    if roots != set(inputs) or not selected_roots <= roots:
        raise ValueError("candidate capture lane roots differ from frozen inputs")
    if len(roots) < 2 or len(available) < len(roots):
        raise ValueError("candidate capture does not demonstrate lane diversity")

    specs: list[tuple[str, Path]] = []
    refs: dict[str, dict[str, Any]] = {}
    for lane in sorted(inputs):
        value = inputs[lane]
        if not isinstance(value, Mapping):
            raise ValueError(f"candidate input is not a binding: {lane}")
        candidate = Path(str(value.get("path") or "")).resolve()
        observed = sha256_file(candidate)
        if observed != normalize_space(value.get("sha256")):
            raise ValueError(f"candidate input SHA mismatch: {lane}")
        specs.append((lane, candidate))
        refs[lane] = _ref(candidate)
    return specs, {
        "capture_freeze": _ref(path),
        "bank_source_sha256": bank_hash,
        "policy": "ALL_FROZEN_RETRIEVER_INPUTS_FULL_DIVERSE_UNION",
        "all_frozen_candidate_inputs_used": True,
        "candidate_input_count": len(specs),
        "available_lane_count": len(available),
        "selected_sequence_ignored_as_subset": True,
        "selection_split": "dev",
        "test_labels_used_for_selection": False,
        "inputs": refs,
    }


def _parse_named_paths(
    values: Sequence[str], *, expected: set[str], label: str
) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip().upper()
        if not separator or name not in expected or name in parsed or not raw_path:
            raise ValueError(f"invalid/duplicate {label} binding: {value!r}")
        parsed[name] = Path(raw_path).resolve()
    if set(parsed) != expected:
        raise ValueError(f"{label} bindings differ: expected={sorted(expected)} got={sorted(parsed)}")
    return parsed


def _validate_train_only_prompt_audit(
    audit_path: Path, *, round_name: str, prompt_path: Path
) -> tuple[dict[str, Any], set[str]]:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    role = payload.get("role_contract") or {}
    blind_reads = role.get(
        "test_or_blind_rows_read_for_rule_authorship",
        role.get("blind_or_test_rows_read_for_rule_authorship"),
    )
    if (
        payload.get("schema_version")
        != "silver-match-v3-humor-resolver-gepa-judge-audit-v1"
        or payload.get("status")
        != "FROZEN_TRAIN_ONLY_PROMPT_REFINEMENT_BEFORE_RESOLVER_LABELING"
        or payload.get("task") != TASK
        or role.get("allowed_role") != "train"
        or int(role.get("dev_rows_read_for_rule_authorship", -1)) != 0
        or int(blind_reads if blind_reads is not None else -1) != 0
        or int(role.get("resolver_votes_or_outcomes_read", -1)) != 0
        or role.get("rule_authorship_completed_before_resolver_labels") is not True
    ):
        raise ValueError(f"{round_name} audit does not prove train-only authorship")
    prompt = payload.get("prompt") or {}
    if (
        normalize_space(prompt.get("sha256")) != sha256_file(prompt_path)
        or Path(str(prompt.get("path") or "")).name != prompt_path.name
    ):
        raise ValueError(f"{round_name} audit does not bind its supplied rule file")
    uids = {
        normalize_space(row.get("norm_uid"))
        for row in payload.get("judged_train_disagreements") or []
        if isinstance(row, Mapping) and normalize_space(row.get("norm_uid"))
    }
    if not uids:
        raise ValueError(f"{round_name} audit has no judged train disagreements")
    source = payload.get("source_items") or {}
    source_path = Path(str(source.get("path") or "")).resolve()
    if (
        not source_path.is_file()
        or normalize_space(source.get("sha256")) != sha256_file(source_path)
    ):
        raise ValueError(f"{round_name} audit source-item hash drift")
    observed_roles = {
        normalize_space(row.get("norm_uid")): (
            normalize_space(row.get("collection_role")),
            normalize_space(row.get("split")),
        )
        for row in read_jsonl(source_path)
        if normalize_space(row.get("norm_uid")) in uids
    }
    if set(observed_roles) != uids or any(
        role_pair != ("train", "train") for role_pair in observed_roles.values()
    ):
        raise ValueError(f"{round_name} audit references a non-train or missing source item")
    return payload, uids


def freeze_composite_gemma_prompt(
    *,
    guide_path: Path,
    round_paths: Mapping[str, Path],
    train_only_audits: Mapping[str, Path],
    output_path: Path,
    manifest_path: Path,
    published_output_path: Path | None = None,
) -> dict[str, Any]:
    """Compose rule text only and bind every component to a hard-frozen hash."""

    components = {"GUIDE": guide_path.resolve(), **dict(round_paths)}
    if set(components) != set(EXPECTED_PROMPT_COMPONENT_SHA256):
        raise ValueError("composite prompt must contain GUIDE and exactly GEPA R1--R9")
    refs: dict[str, dict[str, Any]] = {}
    for name in ("GUIDE", *(f"R{value}" for value in range(1, 10))):
        path = components[name]
        observed = sha256_file(path)
        if observed != EXPECTED_PROMPT_COMPONENT_SHA256[name]:
            raise ValueError(f"frozen prompt component hash drift: {name}/{path}")
        refs[name] = _ref(path)

    audits: dict[str, dict[str, Any]] = {}
    forbidden_uids: set[str] = set()
    for name in ("R7", "R8", "R9"):
        payload, uids = _validate_train_only_prompt_audit(
            train_only_audits[name], round_name=name, prompt_path=components[name]
        )
        audits[name] = {
            "artifact": _ref(train_only_audits[name]),
            "status": payload["status"],
            "allowed_role": "train",
            "dev_rows_read_for_rule_authorship": 0,
            "test_or_blind_rows_read_for_rule_authorship": 0,
            "resolver_votes_or_outcomes_read": 0,
        }
        forbidden_uids.update(uids)

    sections = [
        "# Frozen Humor typed-adjudicator labeling rules",
        "",
        "This prompt contains rules only. It contains no source-pack item, truth label, vote, outcome, or example UID.",
    ]
    for name in ("GUIDE", *(f"R{value}" for value in range(1, 10))):
        title = "Independent labeling guide" if name == "GUIDE" else f"Humor GEPA {name} rules"
        sections.extend(("", f"## {title}", "", components[name].read_text(encoding="utf-8").strip()))
    composite = "\n".join(sections).rstrip() + "\n"
    leaked = sorted(uid for uid in forbidden_uids if uid and uid in composite)
    if leaked:
        raise ValueError(f"train-only audit UID leaked into composite prompt: {leaked[:3]}")
    if "judged_train_disagreements" in composite or "preferred_key" in composite:
        raise ValueError("train-only audit example structure leaked into composite prompt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(composite)
    output_ref = _ref(output_path)
    if published_output_path is not None:
        output_ref["path"] = str(published_output_path.resolve())
    manifest = {
        "schema_version": "silver-match-v3-humor-gemma-composite-prompt-manifest-v1",
        "status": "FROZEN_RULES_ONLY_GUIDE_PLUS_GEPA_R1_R9",
        "task": TASK,
        "component_order": ["GUIDE", *(f"R{value}" for value in range(1, 10))],
        "components": refs,
        "train_only_authorship_audits": audits,
        "truth_examples_included": False,
        "truth_labels_votes_or_outcomes_included": False,
        "example_uids_included": False,
        "output": {**output_ref, "component_count": len(components)},
    }
    _write_json(manifest_path, manifest)
    return manifest


def validate_pilot_recipe(
    selection_path: Path, *, ce_model: Path, task: str = TASK
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    schema = selection.get("schema_version")
    legacy_humor = schema == "silver-match-v3-humor-ce-pilot-selection-v1"
    generic = schema == "silver-match-v3-task-ce-pilot-selection-v1"
    if (
        not (legacy_humor and task == TASK or generic)
        or (generic and selection.get("task") != task)
        or (generic and selection.get("status") != "FROZEN_DEV_ONLY_SELECTION")
        or selection.get("selection_data_role") != "development_only"
        or selection.get("test_opened_before_selection") is not False
    ):
        raise ValueError("winning pilot was not frozen using development only")
    winner = selection.get("winner_record")
    if not isinstance(winner, Mapping):
        raise ValueError("pilot selection lacks winner_record")
    root = Path(str(winner.get("root") or "")).resolve()
    run_config_path = root / "run_config.json"
    if sha256_file(run_config_path) != normalize_space(winner.get("run_config_sha256")):
        raise ValueError("winning pilot run_config hash mismatch")
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if Path(str(config.get("model") or "")).resolve() != ce_model.resolve():
        raise ValueError("winning pilot recipe used another CE base model")
    audit = config.get("split_audit") or {}
    if (
        int(audit.get("source_group_overlap_count", -1)) != 0
        or not config.get("dev_pairs")
    ):
        raise ValueError("winning pilot recipe lacks a source-disjoint explicit dev split")
    lora = config.get("lora") or {}
    gate = config.get("dev_gate") or {}
    required = {
        "max_length": config.get("max_length"),
        "batch_size": config.get("batch_size_per_rank"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "lora_learning_rate": config.get("lora_learning_rate"),
        "head_learning_rate": config.get("head_learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "warmup_ratio": config.get("warmup_ratio"),
        "lora_rank": lora.get("rank"),
        "lora_alpha": lora.get("alpha"),
        "lora_dropout": lora.get("dropout"),
        "attention": config.get("attention"),
        "min_exact_precision": gate.get("minimum_exact_precision"),
        "min_wilson_lower": gate.get("minimum_wilson_lower"),
        "min_exact_predictions": gate.get("minimum_exact_predictions"),
    }
    if any(value is None for value in required.values()):
        raise ValueError("winning pilot run_config lacks required recipe fields")
    base_manifest = Path(str(selection.get("base_manifest") or "")).resolve()
    if sha256_file(base_manifest) != normalize_space(selection.get("base_manifest_sha256")):
        raise ValueError("pilot base-model manifest hash mismatch")
    return required, {
        "selection": _ref(selection_path),
        "winner": selection.get("winner"),
        "winner_run_config": _ref(run_config_path),
        "base_model_manifest": _ref(base_manifest),
        "selection_data_role": "development_only",
        "test_opened_before_selection": False,
        "recipe_fields_copied_without_search": sorted(required),
    }


def _split_ce_rows(
    rows: Sequence[dict[str, Any]], truth_rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    buckets = {role: [] for role in ROLES}
    pair_ids: set[tuple[str, str]] = set()
    group_roles: dict[str, set[str]] = defaultdict(set)
    truth_roles = {str(row["norm_uid"]): str(row["split"]) for row in truth_rows}
    for row in rows:
        uid = normalize_space(row.get("norm_uid"))
        metric = normalize_space(row.get("metric_id"))
        role = normalize_space(row.get("split"))
        group = str(row.get("source_group") or "").strip()
        identity = (uid, metric)
        if not all(identity) or identity in pair_ids or role not in buckets or not group:
            raise ValueError(f"invalid/duplicate CE pair: {identity}/{role}")
        if truth_roles.get(uid) != role:
            raise ValueError(f"CE pair role differs from truth: {uid}")
        if role != "train" and row.get("gradient_eligible") is not False:
            raise ValueError(f"held-out CE pair is gradient eligible: {uid}/{metric}")
        pair_ids.add(identity)
        group_roles[group].add(role)
        buckets[role].append(row)
    crossed = {group: roles for group, roles in group_roles.items() if len(roles) > 1}
    if crossed:
        raise ValueError(f"CE source groups cross roles: {len(crossed)}")
    missing = [role for role, values in buckets.items() if not values]
    if missing:
        raise ValueError(f"CE pair builder did not retain all four roles: {missing}")
    return buckets, {
        "pair_count": len(pair_ids),
        "role_pair_counts": {role: len(buckets[role]) for role in ROLES},
        "role_norm_counts": {
            role: len({str(row["norm_uid"]) for row in buckets[role]})
            for role in ROLES
        },
        "source_groups_crossing_roles": 0,
        "test_or_blind_gradient_eligible": 0,
    }


def _relocate(value: Any, stage: Path, final: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(stage), str(final))
    if isinstance(value, list):
        return [_relocate(item, stage, final) for item in value]
    if isinstance(value, dict):
        return {key: _relocate(item, stage, final) for key, item in value.items()}
    return value


def _ce_command(
    *,
    python: Path,
    trainer: Path,
    train_pairs: Path,
    dev_pairs: Path,
    model: Path,
    output: Path,
    seed: int,
    recipe: Mapping[str, Any],
) -> list[str]:
    command = [
        str(python),
        "-u",
        str(trainer),
        "--train-pairs",
        str(train_pairs),
        "--dev-pairs",
        str(dev_pairs),
        "--model",
        str(model),
        "--output",
        str(output),
    ]
    for exposure in FINAL_EXPOSURES:
        command.extend(("--exposure-budget", str(exposure)))
    flags = (
        ("max-length", "max_length"),
        ("batch-size", "batch_size"),
        ("gradient-accumulation-steps", "gradient_accumulation_steps"),
        ("lora-learning-rate", "lora_learning_rate"),
        ("head-learning-rate", "head_learning_rate"),
        ("weight-decay", "weight_decay"),
        ("warmup-ratio", "warmup_ratio"),
        ("lora-rank", "lora_rank"),
        ("lora-alpha", "lora_alpha"),
        ("lora-dropout", "lora_dropout"),
        ("attention", "attention"),
        ("min-exact-precision", "min_exact_precision"),
        ("min-wilson-lower", "min_wilson_lower"),
        ("min-exact-predictions", "min_exact_predictions"),
    )
    for flag, key in flags:
        command.extend((f"--{flag}", str(recipe[key])))
    command.extend(("--seed", str(seed)))
    return command


def _queue_payload(
    *,
    args: argparse.Namespace,
    final_root: Path,
    truth_manifest: Path,
    ce_partition_report: Path,
    ce_builder_report: Path,
    ce_split_report: Path,
    ce_paths: Mapping[str, Path],
    gemma_report: Path,
    gemma_paths: Mapping[str, Path],
    gemma_prompt: Path,
    gemma_prompt_manifest: Path,
    recipe: Mapping[str, Any],
    pilot_audit: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
    task: str = TASK,
) -> dict[str, Any]:
    python = Path(args.python).resolve()
    ce_trainer = Path(args.ce_trainer).resolve()
    ce_scorer = Path(args.ce_scorer).resolve()
    gemma_trainer = Path(args.gemma_trainer).resolve()
    ce_model = Path(args.ce_model).resolve()
    gemma_model = Path(args.gemma_model).resolve()
    for path in (python, ce_trainer, ce_scorer, gemma_trainer):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (ce_model, gemma_model):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if len(set(args.ce_seed)) != 2:
        raise ValueError("exactly two distinct final CE seeds are required")

    runtime = Path(args.runtime_root).resolve()
    ce_runs = []
    for seed in args.ce_seed:
        output = runtime / "ce" / f"seed-{seed}"
        ce_runs.append(
            {
                "seed": seed,
                "status": "FROZEN_AWAITING_EXECUTION",
                "selection_data_role": "dev_only",
                "test_or_blind_data_read_for_selection": False,
                "command": _ce_command(
                    python=python,
                    trainer=ce_trainer,
                    train_pairs=ce_paths["train"],
                    dev_pairs=ce_paths["dev"],
                    model=ce_model,
                    output=output,
                    seed=seed,
                    recipe=recipe,
                ),
                "output": str(output),
                "required_exposure_checkpoints": list(FINAL_EXPOSURES),
                "inference_after_dev_selection": [
                    {
                        "role": role,
                        "command_template": [
                            str(python),
                            "-u",
                            str(ce_scorer),
                            "score",
                            "--input-pairs",
                            str(ce_paths[role]),
                            "--output",
                            str(runtime / "ce" / f"seed-{seed}" / f"{role}.scores.jsonl"),
                            "--model",
                            str(ce_model),
                            "--base-manifest",
                            str((pilot_audit["base_model_manifest"] or {})["path"]),
                            "--base-manifest-sha256",
                            str((pilot_audit["base_model_manifest"] or {})["sha256"]),
                            "--checkpoint",
                            "{DEV_SELECTED_CHECKPOINT_PATH}",
                            "--training-report",
                            "{FINAL_TRAINING_REPORT_PATH}",
                            "--training-report-sha256",
                            "{FINAL_TRAINING_REPORT_SHA256}",
                            "--max-length",
                            str(recipe["max_length"]),
                        ],
                    }
                    for role in ("test", "blind")
                ],
            }
        )

    gemma_output = runtime / "gemma" / "typed-final"
    gemma_command = [
        str(python),
        "-u",
        str(gemma_trainer),
        "--dataset",
        str(gemma_paths["train"]),
        "--dev-dataset",
        str(gemma_paths["dev"]),
        "--model",
        str(gemma_model),
        "--report",
        str(gemma_output / "training_report.json"),
        "--output",
        str(gemma_output / "adapter"),
        "--max-length",
        "4096",
        "--batch-size",
        "2",
        "--gradient-accumulation-steps",
        "8",
        "--learning-rate",
        "2e-5",
        "--weight-decay",
        "0",
        "--lora-r",
        "8",
        "--lora-alpha",
        "16",
        "--lora-dropout",
        "0.05",
        "--exposure-checkpoints",
        ",".join(str(value) for value in FINAL_EXPOSURES),
        "--decision-loss-weight",
        "4",
        "--metric-id-loss-weight",
        "4",
        "--confidence-loss-weight",
        "1",
        "--reason-loss-weight",
        "0.25",
        "--structural-loss-weight",
        "0.25",
        "--seed",
        str(args.gemma_seed),
    ]

    bindings = {
        "truth_manifest": _ref(truth_manifest),
        "ce_partition_report": _ref(ce_partition_report),
        "ce_builder_report": _ref(ce_builder_report),
        "ce_split_report": _ref(ce_split_report),
        "gemma_dataset_report": _ref(gemma_report),
        "gemma_composite_prompt": _ref(gemma_prompt, training_access="ALLOWED"),
        "gemma_composite_prompt_manifest": _ref(
            gemma_prompt_manifest, training_access="ALLOWED"
        ),
        "ce_train": _ref(ce_paths["train"], training_access="ALLOWED"),
        "ce_dev": _ref(ce_paths["dev"], training_access="SELECTION_ONLY"),
        "ce_test": _ref(ce_paths["test"], training_access="FORBIDDEN"),
        "ce_blind": _ref(ce_paths["blind"], training_access="FORBIDDEN"),
        "gemma_train": _ref(gemma_paths["train"], training_access="ALLOWED"),
        "gemma_dev": _ref(gemma_paths["dev"], training_access="SELECTION_ONLY"),
        "gemma_test": _ref(gemma_paths["test"], training_access="FORBIDDEN"),
        "gemma_blind": _ref(gemma_paths["blind"], training_access="FORBIDDEN"),
        "ce_trainer": _ref(ce_trainer),
        "ce_scorer": _ref(ce_scorer),
        "gemma_trainer": _ref(gemma_trainer),
        "python": _ref(python),
    }
    return {
        "schema_version": QUEUE_SCHEMA,
        "status": "FROZEN_TRAINING_AND_HELDOUT_SCORING_QUEUE_NOT_RELEASE_READY",
        "created_at": _now(),
        "task": task,
        "output_root": str(final_root),
        "runtime_root": str(runtime),
        "bindings": bindings,
        "candidate_bundle": dict(candidate_audit),
        "pilot_recipe": {"recipe": dict(recipe), **dict(pilot_audit)},
        "ce": {
            "architecture": "two_independent_task_local_nemotron_cross_encoder_seeds",
            "runs": ce_runs,
            "automatic_match_policy": "same retained metric from both seeds only",
            "seed_disagreement_policy": "route to typed adjudicator or exhaustive rescue",
            "checkpoint_selection": "within each seed on dev only",
            "test_or_blind_selection": False,
        },
        "gemma": {
            "status": "FROZEN_AWAITING_EXECUTION",
            "architecture": "task_local_typed_gemma4_qkvo_lora",
            "command": gemma_command,
            "output": str(gemma_output),
            "checkpoint_selection": "dev only",
            "test_or_blind_selection": False,
            "field_loss_weights": {
                "decision": 4.0,
                "metric_id": 4.0,
                "confidence": 1.0,
                "reason": 0.25,
                "structure": 0.25,
            },
        },
        "selection_firewall": {
            "gradient_roles": ["train"],
            "selection_roles": ["dev"],
            "heldout_roles": ["test", "blind"],
            "test_or_blind_used_for_recipe_selection": False,
            "test_or_blind_used_for_checkpoint_selection": False,
            "test_or_blind_used_for_threshold_selection": False,
        },
        "readiness": {
            "training_queue_frozen": True,
            "production_adjudication_complete": False,
            "two_seed_consensus_complete": False,
            "typed_adjudicator_validation_complete": False,
            "exhaustive_rescue_complete": False,
            "final_risk_audit_complete": False,
            "production_ready": False,
            "release_ready": False,
            "silver_mi_correlations_ready": False,
        },
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    final_root = Path(args.output_root).resolve()
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite handoff root: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    stage = final_root.parent / f".{final_root.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    try:
        bank_path = Path(args.bank).resolve()
        bank_hash = _bank_source_hash(bank_path)
        joined, truth_report = join_truth(
            existing_path=Path(args.existing_truth).resolve(),
            existing_report_path=Path(args.existing_truth_report).resolve(),
            consensus_path=Path(args.consensus_truth).resolve(),
            consensus_manifest_path=Path(args.consensus_truth_manifest).resolve(),
            bank_path=bank_path,
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

        specs, candidate_audit = load_full_candidate_bundle(
            Path(args.candidate_capture_freeze).resolve(), bank_hash=bank_hash
        )
        candidate_args = [f"{lane}={path}" for lane, path in specs]

        round_paths = _parse_named_paths(
            args.gepa_rule,
            expected={f"R{value}" for value in range(1, 10)},
            label="GEPA rule",
        )
        prompt_audits = _parse_named_paths(
            args.gepa_train_only_audit,
            expected={"R7", "R8", "R9"},
            label="train-only GEPA audit",
        )
        gemma_prompt_path = stage / "prompts" / "HUMOR_GEMMA_COMPOSITE_R1_R9.txt"
        gemma_prompt_manifest = stage / "prompts" / "MANIFEST.json"
        freeze_composite_gemma_prompt(
            guide_path=Path(args.independent_labeling_guide).resolve(),
            round_paths=round_paths,
            train_only_audits=prompt_audits,
            output_path=gemma_prompt_path,
            manifest_path=gemma_prompt_manifest,
            published_output_path=(
                final_root / gemma_prompt_path.relative_to(stage)
            ),
        )

        ce_pair_path = stage / "ce" / "all.pairs.jsonl"
        ce_builder_report_path = stage / "ce" / "BUILDER_REPORT.json"
        ce_args = argparse.Namespace(
            manifest=str(Path(args.manifest).resolve()),
            task=TASK,
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
        ce_rows, ce_report = build_ce_pairs(ce_args)
        write_jsonl(ce_pair_path, ce_rows)
        ce_report = _relocate(ce_report, stage, final_root)
        ce_report["output"] = {
            "path": str(final_root / ce_pair_path.relative_to(stage)),
            "sha256": sha256_file(ce_pair_path),
            "count": len(ce_rows),
        }
        _write_json(ce_builder_report_path, ce_report)

        buckets, split_audit = _split_ce_rows(ce_rows, eligible)
        ce_paths = {role: stage / "ce" / f"{role}.pairs.jsonl" for role in ROLES}
        for role, path in ce_paths.items():
            write_jsonl(path, buckets[role])
        ce_split_report = stage / "ce" / "SPLIT_REPORT.json"
        _write_json(
            ce_split_report,
            {
                "schema_version": "silver-match-v3-humor-final-ce-four-role-split-v1",
                "status": "FROZEN_SOURCE_DISJOINT_FOUR_ROLE_CE_INPUTS",
                "task": TASK,
                "audit": split_audit,
                "input": {
                    "path": str(final_root / ce_pair_path.relative_to(stage)),
                    "sha256": sha256_file(ce_pair_path),
                },
                "builder_report": {
                    "path": str(final_root / ce_builder_report_path.relative_to(stage)),
                    "sha256": sha256_file(ce_builder_report_path),
                },
                "outputs": {
                    role: {
                        "path": str(final_root / path.relative_to(stage)),
                        "sha256": sha256_file(path),
                        "count": len(buckets[role]),
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

        gemma_dir = stage / "gemma" / "dataset"
        gemma_report_path = stage / "gemma" / "DATASET_REPORT.json"
        gemma_args = argparse.Namespace(
            manifest=str(Path(args.manifest).resolve()),
            task=TASK,
            bank=str(bank_path),
            truth=[str(truth_path)],
            split_assignments=None,
            candidates=candidate_args,
            hierarchy=str(Path(args.hierarchy).resolve()),
            prompt=str(gemma_prompt_path),
            max_candidates=args.gemma_max_candidates,
            order_seed=args.gemma_order_seed,
            context_chars=args.gemma_context_chars,
            description_chars=args.gemma_description_chars,
            example_chars=args.gemma_example_chars,
            max_examples=args.gemma_max_examples,
        )
        gemma_buckets, gemma_report = build_gemma_dataset(gemma_args)
        gemma_dir.mkdir(parents=True, exist_ok=False)
        gemma_paths = {role: gemma_dir / f"{role}.jsonl" for role in ROLES}
        for role, path in gemma_paths.items():
            if not gemma_buckets[role]:
                raise ValueError(f"Gemma dataset lost required role: {role}")
            write_jsonl(path, gemma_buckets[role])
        gemma_report = _relocate(gemma_report, stage, final_root)
        gemma_report["outputs"] = {
            role: {
                "path": str(final_root / path.relative_to(stage)),
                "sha256": sha256_file(path),
                "count": len(gemma_buckets[role]),
            }
            for role, path in gemma_paths.items()
        }
        _write_json(gemma_report_path, gemma_report)

        recipe, pilot_audit = validate_pilot_recipe(
            Path(args.pilot_selection).resolve(),
            ce_model=Path(args.ce_model).resolve(),
        )
        final_ce_paths = {
            role: final_root / path.relative_to(stage) for role, path in ce_paths.items()
        }
        final_gemma_paths = {
            role: final_root / path.relative_to(stage) for role, path in gemma_paths.items()
        }
        queue = _queue_payload(
            args=args,
            final_root=final_root,
            truth_manifest=truth_manifest,
            ce_partition_report=ce_partition_report,
            ce_builder_report=ce_builder_report_path,
            ce_split_report=ce_split_report,
            ce_paths=ce_paths,
            gemma_report=gemma_report_path,
            gemma_paths=gemma_paths,
            gemma_prompt=gemma_prompt_path,
            gemma_prompt_manifest=gemma_prompt_manifest,
            recipe=recipe,
            pilot_audit=pilot_audit,
            candidate_audit=candidate_audit,
        )
        # Queue refs were calculated while files live in staging.  Relocate all
        # paths, including commands, before the atomic publication rename.
        queue = _relocate(queue, stage, final_root)
        # Assert the relocated role paths are exactly the intended final files.
        if queue["bindings"]["ce_train"]["path"] != str(final_ce_paths["train"]):
            raise AssertionError("CE queue path relocation failed")
        if queue["bindings"]["gemma_train"]["path"] != str(final_gemma_paths["train"]):
            raise AssertionError("Gemma queue path relocation failed")
        queue_path = stage / "FINAL_STACK_QUEUE.json"
        _write_json(queue_path, queue)

        handoff = {
            "schema_version": SCHEMA,
            "status": "FROZEN_HANDOFF_NOT_PRODUCTION_OR_RELEASE_READY",
            "created_at": _now(),
            "task": TASK,
            "output_root": str(final_root),
            "truth_manifest": {
                "path": str(final_root / truth_manifest.relative_to(stage)),
                "sha256": sha256_file(truth_manifest),
            },
            "ce_partition_report": {
                "path": str(final_root / ce_partition_report.relative_to(stage)),
                "sha256": sha256_file(ce_partition_report),
            },
            "ce_builder_report": {
                "path": str(final_root / ce_builder_report_path.relative_to(stage)),
                "sha256": sha256_file(ce_builder_report_path),
            },
            "ce_split_report": {
                "path": str(final_root / ce_split_report.relative_to(stage)),
                "sha256": sha256_file(ce_split_report),
            },
            "gemma_dataset_report": {
                "path": str(final_root / gemma_report_path.relative_to(stage)),
                "sha256": sha256_file(gemma_report_path),
            },
            "gemma_composite_prompt_manifest": {
                "path": str(final_root / gemma_prompt_manifest.relative_to(stage)),
                "sha256": sha256_file(gemma_prompt_manifest),
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
        # Keep failed staging evidence for audit; never publish it as the
        # requested output and never erase artifacts that may explain failure.
        raise


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Install the one canonical CLI contract for this handoff freezer."""

    for name in (
        "manifest",
        "bank",
        "hierarchy",
        "existing_truth",
        "existing_truth_report",
        "consensus_truth",
        "consensus_truth_manifest",
        "candidate_capture_freeze",
        "pilot_selection",
        "ce_model",
        "gemma_model",
        "independent_labeling_guide",
        "python",
        "ce_trainer",
        "ce_scorer",
        "gemma_trainer",
        "runtime_root",
        "output_root",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--ce-seed", action="append", type=int, required=True)
    parser.add_argument(
        "--gepa-rule",
        action="append",
        required=True,
        metavar="R#=PATH",
        help="repeat exactly for frozen Humor GEPA R1 through R9",
    )
    parser.add_argument(
        "--gepa-train-only-audit",
        action="append",
        required=True,
        metavar="R#=PATH",
        help="repeat exactly for the R7, R8, and R9 train-only judge audits",
    )
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


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if len(args.ce_seed) != 2 or len(set(args.ce_seed)) != 2:
        parser.error("provide exactly two distinct --ce-seed values")
    positive = {
        "maximum_pairs": args.maximum_pairs,
        "ce_context_chars": args.ce_context_chars,
        "gemma_max_candidates": args.gemma_max_candidates,
        "gemma_context_chars": args.gemma_context_chars,
        "gemma_description_chars": args.gemma_description_chars,
        "gemma_example_chars": args.gemma_example_chars,
        "gemma_max_examples": args.gemma_max_examples,
    }
    if any(value <= 0 for value in positive.values()) or args.global_negatives_per_norm < 0:
        parser.error(f"invalid positive/count arguments: {positive}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return args


def main() -> None:
    result = freeze(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
