#!/usr/bin/env python3
"""Bridge fully consumed prior-generation select truth to CE-only train rows.

This is a narrow, append-only role transition.  Label decisions and metric IDs
are copied byte-for-value from completed exact-consensus truth.  The transition
is allowed only after a new select panel has been frozen and proven UID/source-
group disjoint from every consumed panel and every additional forbidden
reference (for example, the current optimize panel).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    groups = [str(row.get("source_group") or "") for row in rows]
    if not rows or "" in uids or "" in groups or len(uids) != len(set(uids)):
        raise ValueError(f"empty, missing, or duplicate identities: {path}")
    return rows


def _verify_ref(ref: dict[str, Any], label: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    if not path.is_file() or sha256_file(path) != str(ref.get("sha256") or ""):
        raise ValueError(f"{label} hash/path mismatch: {path}")
    return path


def _verify_truth_report(
    truth_path: Path, report_path: Path, task: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _rows(truth_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved_ref = ((report.get("outputs") or {}).get("resolved") or {})
    unresolved_ref = ((report.get("outputs") or {}).get("unresolved") or {})
    if (
        report.get("task") != task
        or report.get("complete") is not True
        or report.get("gepa_role") != "select"
        or int(report.get("resolved_count", -1)) != len(rows)
        or resolved_ref.get("sha256") != sha256_file(truth_path)
        or unresolved_ref.get("sha256") != EMPTY_SHA256
    ):
        raise ValueError(f"incomplete or mismatched exact truth report: {report_path}")
    _verify_ref(resolved_ref, "resolved truth")
    unresolved_path = _verify_ref(unresolved_ref, "unresolved truth")
    if unresolved_path.stat().st_size != 0:
        raise ValueError(f"completed truth has nonempty unresolved rows: {unresolved_path}")
    for name, value in sorted(((report.get("inputs") or {}).get("passes") or {}).items()):
        _verify_ref(value.get("labels") or {}, f"{name} labels")
        _verify_ref(value.get("pack_validation") or {}, f"{name} pack validation")
    _verify_ref(
        ((report.get("inputs") or {}).get("source_pack_validation") or {}),
        "source pack validation",
    )
    bank_hashes = {str(row.get("current_bank_source_sha256") or "") for row in rows}
    if len(bank_hashes) != 1 or "" in bank_hashes:
        raise ValueError(f"truth rows do not bind one current bank: {truth_path}")
    if any(
        row.get("task") != task
        or row.get("gepa_role") != "select"
        or row.get("split") != "dev"
        or row.get("evaluation_only") is not True
        for row in rows
    ):
        raise ValueError(f"source is not prior select/dev truth: {truth_path}")
    return rows, report


def _verify_transcript_evidence(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError("transcript-clean evidence is required")
    result = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        direct_pass = (
            payload.get("status") == "PASS"
            and payload.get("complete") is True
            and not payload.get("violations")
        )
        composite_pass = (
            payload.get("status") == "PASS_COMPOSITE_TRANSCRIPT_CLEAN_LABELS"
            and payload.get("complete") is True
            and ((payload.get("transcript_audit") or {}).get("status"))
            == "PASS_COMPOSITE_TRANSCRIPT_CLEAN"
        )
        if not (direct_pass or composite_pass):
            raise ValueError(f"transcript evidence is not clean: {path}")
        result.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "status": payload.get("status"),
            }
        )
    return result


def bridge(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_specs = []
    all_source_rows: list[tuple[str, Path, Path, list[dict[str, Any]]]] = []
    for raw in args.source:
        try:
            name, truth_raw, report_raw = raw.split("=", 1)[0], *raw.split("=", 1)[1].split(",", 1)
        except ValueError as exc:
            raise ValueError("--source must be NAME=TRUTH,REPORT") from exc
        truth_path = Path(truth_raw).resolve()
        report_path = Path(report_raw).resolve()
        rows, _ = _verify_truth_report(truth_path, report_path, args.task)
        source_specs.append(
            {
                "name": name,
                "truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
                "truth_report": {
                    "path": str(report_path),
                    "sha256": sha256_file(report_path),
                },
                "count": len(rows),
            }
        )
        all_source_rows.append((name, truth_path, report_path, rows))

    fresh_identities_path = Path(args.fresh_identities).resolve()
    fresh_freeze_path = Path(args.fresh_freeze).resolve()
    fresh = _rows(fresh_identities_path)
    freeze = json.loads(fresh_freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("task") != args.task
        or freeze.get("role") != "select"
        or freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or (((freeze.get("outputs") or {}).get("identities") or {}).get("sha256"))
        != sha256_file(fresh_identities_path)
    ):
        raise ValueError("new select identity freeze is invalid")

    reference_sets: list[tuple[str, set[str], set[str]]] = [
        (
            "fresh_select",
            {str(row["norm_uid"]) for row in fresh},
            {str(row["source_group"]) for row in fresh},
        )
    ]
    forbidden = []
    for raw in args.forbidden_reference:
        name, path_raw = raw.split("=", 1)
        path = Path(path_raw).resolve()
        rows = _rows(path)
        reference_sets.append(
            (
                name,
                {str(row["norm_uid"]) for row in rows},
                {str(row["source_group"]) for row in rows},
            )
        )
        forbidden.append(
            {"name": name, "path": str(path), "sha256": sha256_file(path), "count": len(rows)}
        )

    bridged: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_groups: set[str] = set()
    overlaps: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    policy_path = Path(args.policy).resolve()
    policy_sha = sha256_file(policy_path)
    for name, truth_path, report_path, rows in all_source_rows:
        uids = {str(row["norm_uid"]) for row in rows}
        groups = {str(row["source_group"]) for row in rows}
        if uids & seen_uids or groups & seen_groups:
            raise ValueError(f"historical source panels overlap: {name}")
        for ref_name, ref_uids, ref_groups in reference_sets:
            overlap = {
                "source": name,
                "reference": ref_name,
                "uid_overlap": len(uids & ref_uids),
                "source_group_overlap": len(groups & ref_groups),
            }
            overlaps.append(overlap)
            if overlap["uid_overlap"] or overlap["source_group_overlap"]:
                raise ValueError(f"historical source overlaps protected reference: {overlap}")
        seen_uids.update(uids)
        seen_groups.update(groups)
        truth_sha = sha256_file(truth_path)
        report_sha = sha256_file(report_path)
        for row in rows:
            if row["current_bank_source_sha256"] != args.bank_source_sha256:
                raise ValueError(f"stale historical bank hash: {row['norm_uid']}")
            decision_counts[str(row["decision"])] += 1
            bridged.append(
                {
                    **row,
                    "schema_version": "silver-match-v3-ce-historical-train-bridge-v1",
                    "original_gepa_role": row.get("gepa_role"),
                    "original_gepa_panel_role": row.get("gepa_panel_role"),
                    "original_split": row.get("split"),
                    "original_evaluation_only": row.get("evaluation_only"),
                    "gepa_role": "historical_train",
                    "gepa_panel_role": None,
                    "split": "train",
                    "evaluation_only": False,
                    "prompt_gradient_eligible": False,
                    "prompt_selection_eligible": False,
                    "ce_training_eligible": True,
                    "retriever_training_eligible": False,
                    "prior_generation_select_consumed": True,
                    "historical_train_source_name": name,
                    "historical_train_source_truth_sha256": truth_sha,
                    "historical_train_source_report_sha256": report_sha,
                    "ce_policy_path": str(policy_path),
                    "ce_policy_sha256": policy_sha,
                }
            )

    bridged.sort(key=lambda row: str(row["norm_uid"]))
    report = {
        "schema_version": "silver-match-v3-ce-historical-train-bridge-report-v1",
        "status": "FROZEN_HISTORICAL_TRAIN_AFTER_NEW_SELECT_FREEZE",
        "task": args.task,
        "count": len(bridged),
        "unique_uids": len(seen_uids),
        "unique_source_groups": len(seen_groups),
        "decision_counts": dict(sorted(decision_counts.items())),
        "label_decisions_or_metric_ids_changed": False,
        "prompt_or_retriever_training_authorized": False,
        "ce_training_authorized": True,
        "bank_source_sha256": args.bank_source_sha256,
        "sources": source_specs,
        "fresh_select": {
            "identities": {
                "path": str(fresh_identities_path),
                "sha256": sha256_file(fresh_identities_path),
                "count": len(fresh),
            },
            "freeze": {"path": str(fresh_freeze_path), "sha256": sha256_file(fresh_freeze_path)},
        },
        "forbidden_references": forbidden,
        "overlap_audit": overlaps,
        "transcript_clean_evidence": _verify_transcript_evidence(
            [Path(value).resolve() for value in args.transcript_evidence]
        ),
        "policy": {"path": str(policy_path), "sha256": policy_sha},
    }
    return bridged, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--bank-source-sha256", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--fresh-identities", required=True)
    parser.add_argument("--fresh-freeze", required=True)
    parser.add_argument("--forbidden-reference", action="append", default=[])
    parser.add_argument("--transcript-evidence", action="append", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite historical CE bridge")
    rows, report = bridge(args)
    write_jsonl(output, rows)
    report["output"] = {"path": str(output), "sha256": sha256_file(output)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
