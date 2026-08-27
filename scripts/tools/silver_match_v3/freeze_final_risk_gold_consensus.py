#!/usr/bin/env python3
"""Freeze transcript-audited multi-pass gold for one final-risk sample."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .adjudicate_gemma import CONFIDENCES, DECISIONS
from .common import read_jsonl, sha256_file
from .finalize_exact_multi_pass_truth import _decision_key, _winner


SCHEMA = "silver-match-v3-final-risk-gold-consensus-release-v1"
STATUS = "FROZEN_COMPLETE_TRANSCRIPT_AUDITED_EXACT_CONSENSUS"
SAMPLE_SCHEMA = "silver-match-v3-final-decision-sample-v2"
CONSENSUS_SCHEMA = "silver-match-v3-exact-multi-pass-truth-report-v1"
LABEL_SCHEMA = "silver-match-v3-independent-label-validation-v1"
CODEX_TRANSCRIPT_SCHEMA = "silver-match-v3-isolated-labeler-transcript-audit-v1"
OPENROUTER_TRANSCRIPT_SCHEMA = (
    "silver-match-v3-openrouter-labeler-transcript-audit-v1"
)
COMPOSITE_TRANSCRIPT_SCHEMA = (
    "silver-match-v3-composite-openrouter-direct-vllm-transcript-audit-v1"
)
INDEPENDENCE_SCHEMA = "silver-match-v3-independent-pack-view-audit-v1"
SOURCE_PACK_SCHEMA = "silver-match-v3-final-risk-label-pack-v1"
PERMUTED_PACK_SCHEMA = "silver-match-v3-permuted-independent-teacher-pack-v1"
RESOLVER_PACK_SCHEMA = "silver-match-v3-exact-unresolved-resolver-pack-v1"


def _resolve(value: str | Path, anchor: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _bound(reference: dict[str, Any], anchor: Path, label: str) -> Path:
    if (
        not isinstance(reference, dict)
        or not reference.get("path")
        or not reference.get("sha256")
    ):
        raise ValueError(f"{label} lacks a path/hash binding")
    path = _resolve(str(reference["path"]), anchor)
    if not path.is_file() or sha256_file(path) != str(reference["sha256"]):
        raise ValueError(f"{label} artifact changed: {path}")
    return path


def _named_paths(values: list[str], flag: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in output:
            raise ValueError(f"{flag} must be unique NAME=PATH values")
        output[name] = Path(raw_path).resolve()
    return output


def _rows(path: Path, label: str) -> dict[str, dict[str, Any]]:
    values = list(read_jsonl(path))
    output = {str(row.get("norm_uid") or ""): row for row in values}
    if not values or "" in output or len(output) != len(values):
        raise ValueError(f"{label} has missing/duplicate norm UIDs: {path}")
    return output


def _verify_pack(
    validation_path: Path,
    *,
    task: str,
    bank_source_sha256: str,
) -> tuple[dict[str, Any], Path, Path, dict[str, dict[str, Any]]]:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("task") != task
        or validation.get("truth_hidden") is not True
        or validation.get("bank_source_sha256") != bank_source_sha256
    ):
        raise ValueError(f"pass pack task/bank/blinding drift: {validation_path}")
    outputs = validation.get("outputs") or {}
    items_path = _bound(outputs.get("items") or {}, validation_path, "pack items")
    bank_path = _bound(outputs.get("bank") or {}, validation_path, "pack bank")
    chunks = outputs.get("chunks") or {}
    for raw_path, expected_sha in chunks.items():
        path = _resolve(raw_path, validation_path)
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise ValueError(f"pack chunk changed: {path}")
    items = _rows(items_path, "pass items")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("task") != task or bank.get("source_sha256") != bank_source_sha256:
        raise ValueError("pass bank identity mismatch")
    return validation, items_path, bank_path, items


def _verify_transcript(
    audit_path: Path,
    *,
    pack_validation: dict[str, Any],
    pack_validation_path: Path,
    bank_path: Path,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    pack_root = pack_validation_path.parent
    expected_chunks = {
        _resolve(raw_path, pack_validation_path).stem: _resolve(
            raw_path, pack_validation_path
        )
        for raw_path in (pack_validation.get("outputs") or {}).get("chunks") or {}
    }
    rows = {str(row.get("chunk") or ""): row for row in audit.get("chunks") or []}
    transcript_schema = audit.get("schema_version")
    if (
        transcript_schema
        not in {
            CODEX_TRANSCRIPT_SCHEMA,
            OPENROUTER_TRANSCRIPT_SCHEMA,
            COMPOSITE_TRANSCRIPT_SCHEMA,
        }
        or audit.get("status") != "PASS"
        or audit.get("complete") is not True
        or audit.get("violations") != []
        or set(rows) != set(expected_chunks)
        or (audit.get("bank") or {}).get("sha256") != sha256_file(bank_path)
    ):
        raise ValueError(f"invalid strict transcript audit: {audit_path}")
    if transcript_schema == OPENROUTER_TRANSCRIPT_SCHEMA:
        contract = audit.get("contract") or {}
        if (
            audit.get("truth_hidden") is not True
            or (audit.get("pack_validation") or {}).get("sha256")
            != sha256_file(pack_validation_path)
            or contract.get(
                "exact_request_body_reconstructed_from_only_frozen_bank_chunk_and_guides"
            )
            is not True
            or contract.get(
                "sample_keys_predictions_proposals_mi_and_outcomes_absent"
            )
            is not True
            or contract.get("api_credentials_not_logged") is not True
        ):
            raise ValueError(f"invalid OpenRouter transcript-isolation contract: {audit_path}")
    elif transcript_schema == COMPOSITE_TRANSCRIPT_SCHEMA:
        contract = audit.get("contract") or {}
        if (
            audit.get("truth_hidden") is not True
            or (audit.get("pack_validation") or {}).get("sha256")
            != sha256_file(pack_validation_path)
            or contract.get("every_chunk_has_exactly_one_verified_backend") is not True
            or contract.get("openrouter_chunks_retain_exact_request_transcripts")
            is not True
            or contract.get("direct_vllm_rows_reparsed_from_raw_responses") is not True
            or contract.get("direct_vllm_item_prompts_independently_reconstructed")
            is not True
            or contract.get(
                "sample_keys_predictions_proposals_mi_and_outcomes_absent"
            )
            is not True
        ):
            raise ValueError(f"invalid composite transcript-isolation contract: {audit_path}")
    for chunk_id, chunk in expected_chunks.items():
        row = rows[chunk_id]
        raw = (
            Path(str(row.get("raw_label_path") or "")).resolve()
            if transcript_schema == COMPOSITE_TRANSCRIPT_SCHEMA
            else pack_root / "raw_labels" / f"{chunk_id}.json"
        )
        if transcript_schema == CODEX_TRANSCRIPT_SCHEMA:
            log = pack_root / "logs" / f"{chunk_id}.log"
            event_valid = (
                log.is_file()
                and row.get("log_sha256") == sha256_file(log)
                and int(row.get("command_count", 0)) >= 1
            )
        elif transcript_schema == OPENROUTER_TRANSCRIPT_SCHEMA:
            transcript = pack_root / "api_transcripts" / f"{chunk_id}.json"
            event_valid = (
                transcript.is_file()
                and row.get("transcript_sha256") == sha256_file(transcript)
                and int(row.get("request_count", 0)) >= 1
            )
        else:
            backend = str(row.get("backend") or "")
            artifacts = row.get("backend_artifacts") or {}
            if backend == "openrouter_api":
                required = ("api_transcript",)
            elif backend == "direct_vllm":
                required = (
                    "direct_output",
                    "direct_meta",
                    "candidates",
                    "runner",
                    "runner_log",
                    "frozen_plan",
                )
            else:
                required = ()
            artifact_valid = bool(required)
            for key in required:
                reference = artifacts.get(key) or {}
                path = Path(str(reference.get("path") or "")).resolve()
                artifact_valid = (
                    artifact_valid
                    and path.is_file()
                    and reference.get("sha256") == sha256_file(path)
                )
            event_valid = artifact_valid and int(row.get("event_count", 0)) >= 1
        if (
            not raw.is_file()
            or row.get("chunk_sha256") != sha256_file(chunk)
            or row.get("raw_label_sha256") != sha256_file(raw)
            or not event_valid
        ):
            raise ValueError(f"transcript artifact drift: {chunk_id}")
    return {
        "path": str(audit_path),
        "sha256": sha256_file(audit_path),
        "schema_version": transcript_schema,
        "model": audit.get("model"),
        "audited_chunks": len(rows),
        "event_count": int(
            audit.get("command_count")
            if transcript_schema == CODEX_TRANSCRIPT_SCHEMA
            else (
                audit.get("request_count")
                if transcript_schema == OPENROUTER_TRANSCRIPT_SCHEMA
                else audit.get("event_count")
            )
            or 0
        ),
    }


def _verify_label_pass(
    *,
    name: str,
    consensus_meta: dict[str, Any],
    label_validation_path: Path,
    transcript_audit_path: Path,
    report_path: Path,
    task: str,
    bank_source_sha256: str,
) -> dict[str, Any]:
    labels_path = _bound(
        consensus_meta.get("labels") or {}, report_path, f"{name} labels"
    )
    pack_validation_path = _bound(
        consensus_meta.get("pack_validation") or {}, report_path, f"{name} pack"
    )
    pack_validation, items_path, bank_path, items = _verify_pack(
        pack_validation_path,
        task=task,
        bank_source_sha256=bank_source_sha256,
    )
    validation = json.loads(label_validation_path.read_text(encoding="utf-8"))
    output_ref = validation.get("output") or {}
    pack_ref = validation.get("pack_validation") or {}
    transcript_ref = validation.get("transcript_audit") or {}
    if (
        validation.get("schema_version") != LABEL_SCHEMA
        or validation.get("task") != task
        or validation.get("complete") is not True
        or validation.get("bank_source_sha256") != bank_source_sha256
        or _resolve(str(output_ref.get("path") or ""), label_validation_path)
        != labels_path
        or output_ref.get("sha256") != sha256_file(labels_path)
        or _resolve(str(pack_ref.get("path") or ""), label_validation_path)
        != pack_validation_path
        or pack_ref.get("sha256") != sha256_file(pack_validation_path)
        or _resolve(str(transcript_ref.get("path") or ""), label_validation_path)
        != transcript_audit_path
        or transcript_ref.get("sha256") != sha256_file(transcript_audit_path)
        or transcript_ref.get("status") != "PASS"
    ):
        raise ValueError(f"label validation/transcript binding drift: {name}")
    transcript = _verify_transcript(
        transcript_audit_path,
        pack_validation=pack_validation,
        pack_validation_path=pack_validation_path,
        bank_path=bank_path,
    )
    labels = _rows(labels_path, f"{name} labels")
    if set(labels) != set(items):
        raise ValueError(f"{name} labels do not exactly cover its blind pack")
    bank_ids = {
        str(row["metric_id"])
        for row in json.loads(bank_path.read_text(encoding="utf-8")).get("metrics")
        or []
    }
    for uid, row in labels.items():
        decision, metric_id = _decision_key(row)
        if (
            row.get("task") != task
            or row.get("current_bank_source_sha256") != bank_source_sha256
            or decision not in DECISIONS
            or str(row.get("confidence") or "").lower() not in CONFIDENCES
            or (decision == "MATCH" and metric_id not in bank_ids)
            or (decision != "MATCH" and row.get("metric_id") is not None)
        ):
            raise ValueError(f"invalid validated label in {name}: {uid}")
    return {
        "name": name,
        "labels_path": labels_path,
        "labels": labels,
        "pack_validation_path": pack_validation_path,
        "pack_validation": pack_validation,
        "pack_items_path": items_path,
        "pack_items": items,
        "bank_path": bank_path,
        "label_validation": _artifact(label_validation_path),
        "transcript_audit": transcript,
    }


def evaluate_gold_consensus(
    *,
    sample_report_path: Path,
    scope: str,
    truth_path: Path,
    consensus_report_path: Path,
    independence_audit_path: Path,
    label_validation_paths: dict[str, Path],
    transcript_audit_paths: dict[str, Path],
) -> dict[str, Any]:
    sample_report_path = sample_report_path.resolve()
    truth_path = truth_path.resolve()
    consensus_report_path = consensus_report_path.resolve()
    independence_audit_path = independence_audit_path.resolve()
    sample = json.loads(sample_report_path.read_text(encoding="utf-8"))
    sample_meta = (sample.get("outputs") or {}).get(scope) or {}
    if (
        sample.get("schema_version") != SAMPLE_SCHEMA
        or sample.get("sample_kind") not in {"match", "abstention"}
        or not scope.startswith("task:")
    ):
        raise ValueError("invalid final-risk sample report/scope")
    task = scope.split(":", 1)[1]
    blind_path = _bound(
        sample_meta.get("blind") or {}, sample_report_path, "blind sample"
    )
    sample_uids = set(_rows(blind_path, "blind sample"))
    source_pack_ref = sample_meta.get("label_pack_validation") or {}
    source_pack_validation_path = _bound(
        source_pack_ref, sample_report_path, "source label pack"
    )
    source_pack, _, source_bank_path, source_items = _verify_pack(
        source_pack_validation_path,
        task=task,
        bank_source_sha256=sample["bank_outputs"][task]["source_sha256"],
    )
    if (
        source_pack.get("schema_version") != SOURCE_PACK_SCHEMA
        or set(source_items) != sample_uids
    ):
        raise ValueError("source label pack differs from sampled UIDs")
    bank_source_sha256 = str(source_pack["bank_source_sha256"])

    consensus = json.loads(consensus_report_path.read_text(encoding="utf-8"))
    truth = _rows(truth_path, "resolved consensus truth")
    report_passes = (consensus.get("inputs") or {}).get("passes") or {}
    rounds = consensus.get("rounds") or []
    names = [str(row.get("pass") or "") for row in rounds]
    if (
        consensus.get("schema_version") != CONSENSUS_SCHEMA
        or consensus.get("task") != task
        or consensus.get("complete") is not True
        or int(consensus.get("source_count", -1)) != len(sample_uids)
        or int(consensus.get("resolved_count", -1)) != len(sample_uids)
        or int(consensus.get("unresolved_count", -1)) != 0
        or set(truth) != sample_uids
        or _resolve(
            str(
                ((consensus.get("outputs") or {}).get("resolved") or {}).get("path")
                or ""
            ),
            consensus_report_path,
        )
        != truth_path
        or ((consensus.get("outputs") or {}).get("resolved") or {}).get("sha256")
        != sha256_file(truth_path)
        or _resolve(
            str(
                (
                    (consensus.get("inputs") or {}).get("source_pack_validation") or {}
                ).get("path")
                or ""
            ),
            consensus_report_path,
        )
        != source_pack_validation_path
        or ((consensus.get("inputs") or {}).get("source_pack_validation") or {}).get(
            "sha256"
        )
        != sha256_file(source_pack_validation_path)
        or not names
        or len(names) != len(set(names))
        or set(names) != set(report_passes)
        or set(label_validation_paths) != set(names)
        or set(transcript_audit_paths) != set(names)
    ):
        raise ValueError("consensus report is incomplete or not bound to the sample")
    if len(names) < 2:
        raise ValueError("final-risk gold requires at least two independent passes")

    passes: dict[str, dict[str, Any]] = {}
    for name in names:
        passes[name] = _verify_label_pass(
            name=name,
            consensus_meta=report_passes[name],
            label_validation_path=label_validation_paths[name].resolve(),
            transcript_audit_path=transcript_audit_paths[name].resolve(),
            report_path=consensus_report_path,
            task=task,
            bank_source_sha256=bank_source_sha256,
        )

    independence = json.loads(independence_audit_path.read_text(encoding="utf-8"))
    first, second = passes[names[0]], passes[names[1]]
    independence_passes = independence.get("passes") or {}
    independent_roots = {
        Path(str(row.get("root") or "")).resolve()
        for row in independence_passes.values()
    }
    if (
        independence.get("schema_version") != INDEPENDENCE_SCHEMA
        or independence.get("status")
        != "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING"
        or independence.get("task") != task
        or int(independence.get("count", -1)) != len(sample_uids)
        or independence.get("distinct_bank_order") is not True
        or independence.get("distinct_item_order") is not True
        or independence.get("distinct_seeds") is not True
        or independence.get("same_uid_set") is not True
        or independence.get("same_bank_leaf_set") is not True
        or independence.get("same_canonical_item_content_by_uid") is not True
        or independence.get("candidate_proposals_exposed_to_either_pass") is not False
        or independence.get("prior_truth_or_predictions_exposed_to_either_pass")
        is not False
        or independence.get("pass_predictions_mutually_visible") is not False
        or independent_roots
        != {first["pack_validation_path"].parent, second["pack_validation_path"].parent}
    ):
        raise ValueError("initial passes lack a valid pre-label independence freeze")
    expected_initial = {
        first["pack_validation_path"].parent: first,
        second["pack_validation_path"].parent: second,
    }
    for audit_pass in independence_passes.values():
        root = Path(str(audit_pass.get("root") or "")).resolve()
        observed = expected_initial.get(root)
        if observed is None or (
            audit_pass.get("validation_sha256")
            != sha256_file(observed["pack_validation_path"])
            or audit_pass.get("items_sha256")
            != sha256_file(observed["pack_items_path"])
            or audit_pass.get("bank_sha256") != sha256_file(observed["bank_path"])
            or audit_pass.get("seed") != observed["pack_validation"].get("seed")
        ):
            raise ValueError("independence audit pass binding changed")
    initial_seeds = []
    for initial in (first, second):
        validation = initial["pack_validation"]
        source_ref = validation.get("source_pack") or {}
        if (
            validation.get("schema_version") != PERMUTED_PACK_SCHEMA
            or _resolve(
                str(source_ref.get("path") or ""), initial["pack_validation_path"]
            )
            != source_pack_validation_path.parent
            or source_ref.get("validation_sha256")
            != sha256_file(source_pack_validation_path)
        ):
            raise ValueError("initial pass is not derived from the frozen sample pack")
        initial_seeds.append(validation.get("seed"))
    if len(initial_seeds) != len(set(initial_seeds)):
        raise ValueError("initial pass permutation seeds are not distinct")

    votes: dict[str, list[tuple[str, str | None]]] = {uid: [] for uid in sample_uids}
    unresolved = set(sample_uids)
    replay_rounds = []
    for ordinal, name in enumerate(names, 1):
        observed = set(passes[name]["labels"])
        expected = sample_uids if ordinal <= 2 else unresolved
        if observed != expected:
            raise ValueError(
                f"pass {name} does not cover the exact unresolved frontier"
            )
        if ordinal > 2:
            validation = passes[name]["pack_validation"]
            source_ref = (validation.get("inputs") or {}).get(
                "source_pack_validation"
            ) or {}
            unresolved_ref = (validation.get("inputs") or {}).get("unresolved") or {}
            unresolved_path = _bound(
                unresolved_ref,
                passes[name]["pack_validation_path"],
                f"{name} unresolved frontier",
            )
            if (
                validation.get("schema_version") != RESOLVER_PACK_SCHEMA
                or validation.get("truth_hidden") is not True
                or validation.get("prior_decisions_and_metric_ids_hidden") is not True
                or validation.get("selection_rule")
                != "all_and_only_current_exact_consensus_unresolved_uids"
                or int(validation.get("source_count", -1)) != len(sample_uids)
                or int(validation.get("count", -1)) != len(expected)
                or _resolve(
                    str(source_ref.get("path") or ""),
                    passes[name]["pack_validation_path"],
                )
                != source_pack_validation_path
                or source_ref.get("sha256") != sha256_file(source_pack_validation_path)
                or set(_rows(unresolved_path, f"{name} unresolved frontier"))
                != expected
            ):
                raise ValueError(f"invalid exact unresolved resolver lineage: {name}")
        before = len(unresolved)
        for uid, row in passes[name]["labels"].items():
            votes[uid].append(_decision_key(row))
        unresolved = {uid for uid, values in votes.items() if _winner(values) is None}
        replay_rounds.append(
            {
                "pass": name,
                "ordinal": ordinal,
                "labeled_count": len(observed),
                "unresolved_before": before,
                "newly_resolved": before - len(unresolved),
                "unresolved_after": len(unresolved),
            }
        )
    if unresolved:
        raise ValueError("final-risk gold consensus remains unresolved")
    report_rounds = [
        {
            key: row.get(key)
            for key in (
                "pass",
                "ordinal",
                "labeled_count",
                "unresolved_before",
                "newly_resolved",
                "unresolved_after",
            )
        }
        for row in rounds
    ]
    if report_rounds != replay_rounds:
        raise ValueError("consensus round ledger differs from replay")
    decision_counts: Counter[str] = Counter()
    for uid, truth_row in truth.items():
        winner = _winner(votes[uid])
        if winner != _decision_key(truth_row):
            raise ValueError(f"resolved truth differs from replayed winner: {uid}")
        decision_counts[winner[0]] += 1
    if dict(sorted(decision_counts.items())) != consensus.get(
        "resolved_decision_counts"
    ):
        raise ValueError("resolved decision counts differ from consensus replay")

    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "complete": True,
        "task": task,
        "sample_kind": sample["sample_kind"],
        "sample_scope": scope,
        "count": len(sample_uids),
        "bank_source_sha256": bank_source_sha256,
        "sample_report": _artifact(sample_report_path),
        "source_pack_validation": _artifact(source_pack_validation_path),
        "source_bank": _artifact(source_bank_path),
        "consensus_truth": _artifact(truth_path),
        "consensus_report": _artifact(consensus_report_path),
        "initial_independence_audit": _artifact(independence_audit_path),
        "pass_order": names,
        "passes": {
            name: {
                "labels": _artifact(passes[name]["labels_path"]),
                "pack_validation": _artifact(passes[name]["pack_validation_path"]),
                "label_validation": passes[name]["label_validation"],
                "transcript_audit": passes[name]["transcript_audit"],
            }
            for name in names
        },
        "replayed_rounds": replay_rounds,
        "decision_counts": dict(sorted(decision_counts.items())),
        "contract": {
            "at_least_two_complete_independently_permuted_passes": True,
            "all_resolvers_cover_only_prior_unresolved_frontier": True,
            "unique_exact_two_vote_consensus": True,
            "every_pass_strictly_transcript_audited": True,
            "no_unresolved_sample_rows_dropped": True,
            "permanently_excluded_from_gradients": True,
        },
    }


def verify_gold_consensus_release(
    release_path: Path,
    *,
    expected_sample_report: Path | None = None,
    expected_truth: Path | None = None,
) -> dict[str, Any]:
    release_path = release_path.resolve()
    frozen = json.loads(release_path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != SCHEMA:
        raise ValueError("unsupported final-risk gold consensus release")
    sample_report = _bound(
        frozen.get("sample_report") or {}, release_path, "sample report"
    )
    truth = _bound(frozen.get("consensus_truth") or {}, release_path, "consensus truth")
    if (
        expected_sample_report is not None
        and sample_report != expected_sample_report.resolve()
    ):
        raise ValueError("gold consensus release binds another sample")
    if expected_truth is not None and truth != expected_truth.resolve():
        raise ValueError("gold consensus release binds another truth file")
    consensus_report = _bound(
        frozen.get("consensus_report") or {}, release_path, "consensus report"
    )
    independence = _bound(
        frozen.get("initial_independence_audit") or {},
        release_path,
        "independence audit",
    )
    pass_order = frozen.get("pass_order") or []
    pass_refs = frozen.get("passes") or {}
    if (
        not isinstance(pass_order, list)
        or len(pass_order) != len(set(pass_order))
        or set(pass_order) != set(pass_refs)
    ):
        raise ValueError("gold consensus pass order is not canonical")
    recomputed = evaluate_gold_consensus(
        sample_report_path=sample_report,
        scope=str(frozen.get("sample_scope") or ""),
        truth_path=truth,
        consensus_report_path=consensus_report,
        independence_audit_path=independence,
        label_validation_paths={
            name: _bound(
                pass_refs[name].get("label_validation") or {},
                release_path,
                f"{name} label validation",
            )
            for name in pass_order
        },
        transcript_audit_paths={
            name: _bound(
                pass_refs[name].get("transcript_audit") or {},
                release_path,
                f"{name} transcript audit",
            )
            for name in pass_order
        },
    )
    for key in (
        "status",
        "complete",
        "task",
        "sample_kind",
        "sample_scope",
        "count",
        "bank_source_sha256",
        "pass_order",
        "replayed_rounds",
        "decision_counts",
        "contract",
    ):
        if frozen.get(key) != recomputed.get(key):
            raise ValueError(f"gold consensus release differs from replay: {key}")
    return recomputed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-report", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--consensus-report", required=True)
    parser.add_argument("--independence-audit", required=True)
    parser.add_argument("--label-validation", action="append", required=True)
    parser.add_argument("--transcript-audit", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = evaluate_gold_consensus(
        sample_report_path=Path(args.sample_report),
        scope=args.scope,
        truth_path=Path(args.truth),
        consensus_report_path=Path(args.consensus_report),
        independence_audit_path=Path(args.independence_audit),
        label_validation_paths=_named_paths(
            args.label_validation, "--label-validation"
        ),
        transcript_audit_paths=_named_paths(
            args.transcript_audit, "--transcript-audit"
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "task": result["task"],
                "count": result["count"],
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
