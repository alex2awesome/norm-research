#!/usr/bin/env python3
"""Revise canonical dev truth from an immutable prediction-hidden tie-break."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate UIDs: {path}")
    return output


def decision_key(row: dict[str, Any]) -> tuple[str, str | None]:
    decision = str(row.get("decision") or "")
    return decision, str(row["metric_id"]) if decision == "MATCH" else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--scoring-truth", required=True)
    parser.add_argument("--tiebreak", required=True)
    parser.add_argument("--forbidden-items", required=True)
    parser.add_argument("--change-uid", required=True)
    parser.add_argument("--omit-uid", action="append", required=True)
    parser.add_argument("--preserve-uid", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--quarantine-output", required=True)
    parser.add_argument("--r3-exclusions-output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "v1": args.v1,
            "primary": args.primary,
            "scoring_truth": args.scoring_truth,
            "tiebreak": args.tiebreak,
            "forbidden_items": args.forbidden_items,
        }.items()
    }
    outputs = {
        name: Path(value).resolve()
        for name, value in {
            "v2": args.output,
            "quarantine": args.quarantine_output,
            "r3_exclusions": args.r3_exclusions_output,
            "report": args.report,
        }.items()
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite canonical revision outputs")
    v1, primary, truth, tiebreak, forbidden = (
        index(paths[name])
        for name in ("v1", "primary", "scoring_truth", "tiebreak", "forbidden_items")
    )
    tie_uids = set(tiebreak)
    expected = {args.change_uid, *args.omit_uid, *args.preserve_uid}
    if tie_uids != expected:
        raise ValueError("tiebreak UID set does not equal change/omit/preserve declaration")
    if not expected.issubset(v1) or not expected.issubset(primary) or not expected.issubset(truth):
        raise ValueError("declared tie-break UID absent from a canonical input")
    for uid in expected:
        if decision_key(truth[uid]) != decision_key(v1[uid]):
            raise ValueError(f"scoring truth does not reproduce immutable v1: {uid}")

    change_uid = args.change_uid
    if decision_key(primary[change_uid]) != decision_key(tiebreak[change_uid]):
        raise ValueError("change row lacks exact primary+tiebreak corroboration")
    if decision_key(v1[change_uid]) == decision_key(tiebreak[change_uid]):
        raise ValueError("change row does not change canonical decision/leaf")
    omit = set(args.omit_uid)
    for uid in omit:
        keys = {decision_key(v1[uid]), decision_key(primary[uid]), decision_key(tiebreak[uid])}
        if len(keys) != 3:
            raise ValueError(f"omit row is not an unresolved three-way conflict: {uid}")
    for uid in args.preserve_uid:
        if decision_key(v1[uid]) != decision_key(tiebreak[uid]):
            raise ValueError(f"declared preserve row is not tie-break confirmed: {uid}")

    tie_hash = sha256_file(paths["tiebreak"])
    primary_hash = sha256_file(paths["primary"])
    v2_rows = []
    quarantine = []
    preserved_exact = 0
    for uid, row in v1.items():
        if uid in omit:
            quarantine.append(
                {
                    **row,
                    "quarantine_reason": "primary, canonical-v1, and independent tie-break all disagree",
                    "primary_prediction": primary[uid],
                    "tiebreak_prediction": tiebreak[uid],
                    "tiebreak_sha256": tie_hash,
                }
            )
            continue
        if uid == change_uid:
            revised = {
                **row,
                "decision": tiebreak[uid]["decision"],
                "metric_id": tiebreak[uid]["metric_id"],
                "confidence": tiebreak[uid]["confidence"],
                "reason": tiebreak[uid]["reason"],
                "label_source": "semantic_resolved_truth_tiebreak_revision_v2",
                "agreement_sources": ["r2_primary_exact_consensus", "independent_tiebreak"],
                "dissenting_sources": ["canonical_v1"],
                "canonical_v1_prediction": {
                    "decision": row["decision"],
                    "metric_id": row.get("metric_id"),
                    "confidence": row.get("confidence"),
                    "reason": row.get("reason"),
                },
                "primary_prediction": primary[uid],
                "tiebreak_prediction": tiebreak[uid],
                "revision_input_sha256": {
                    "primary": primary_hash,
                    "tiebreak": tie_hash,
                    "v1": sha256_file(paths["v1"]),
                },
            }
            v2_rows.append(revised)
        else:
            v2_rows.append(row)
            preserved_exact += 1

    v2_by_uid = {str(row["norm_uid"]): row for row in v2_rows}
    if set(v2_by_uid) != set(v1) - omit or len(v2_rows) != len(v1) - len(omit):
        raise AssertionError("v2 UID/count diff is not exactly the declared omissions")
    for uid in set(v1) - omit - {change_uid}:
        if v2_by_uid[uid] != v1[uid]:
            raise AssertionError(f"undeclared canonical row changed: {uid}")

    r3_exclusions = []
    for uid in [change_uid, *args.omit_uid]:
        status = "primary_confirmed_not_false_retain" if uid == change_uid else "unresolved_three_way_conflict"
        r3_exclusions.append(
            {
                "norm_uid": uid,
                "task": v1[uid]["task"],
                "split": "dev",
                "status": status,
                "exclude_from_r3_false_retain_gradients": True,
                "canonical_v1_decision": v1[uid]["decision"],
                "canonical_v1_metric_id": v1[uid].get("metric_id"),
                "primary_decision": primary[uid]["decision"],
                "primary_metric_id": primary[uid].get("metric_id"),
                "tiebreak_decision": tiebreak[uid]["decision"],
                "tiebreak_metric_id": tiebreak[uid].get("metric_id"),
                "tiebreak_sha256": tie_hash,
            }
        )

    forbidden_uids = set(forbidden)
    forbidden_groups = {
        str(row.get("source_group") or row.get("split_group")) for row in forbidden.values()
    }
    v2_groups = {str(row.get("source_group") or row.get("split_group")) for row in v2_rows}
    exclusion_groups = {
        str(v1[row["norm_uid"]].get("source_group") or v1[row["norm_uid"]].get("split_group"))
        for row in r3_exclusions
    }
    blind_overlap = {
        "v2_uid": len(set(v2_by_uid) & forbidden_uids),
        "v2_source_group": len(v2_groups & forbidden_groups),
        "r3_exclusion_uid": len({row["norm_uid"] for row in r3_exclusions} & forbidden_uids),
        "r3_exclusion_source_group": len(exclusion_groups & forbidden_groups),
    }
    if any(blind_overlap.values()):
        raise ValueError(f"canonical revision overlaps permanent blind rows: {blind_overlap}")

    write_jsonl(outputs["v2"], v2_rows)
    write_jsonl(outputs["quarantine"], quarantine)
    write_jsonl(outputs["r3_exclusions"], r3_exclusions)
    report = {
        "schema_version": "silver-match-v3-resolved-truth-tiebreak-revision-v2",
        "task": next(iter(v1.values()))["task"],
        "v1_count": len(v1),
        "v2_count": len(v2_rows),
        "preserved_exact_row_count": preserved_exact,
        "changed_count": 1,
        "changed_uid": change_uid,
        "changed_from": {"decision": v1[change_uid]["decision"], "metric_id": v1[change_uid].get("metric_id")},
        "changed_to": {"decision": tiebreak[change_uid]["decision"], "metric_id": tiebreak[change_uid].get("metric_id")},
        "omitted_count": len(omit),
        "omitted_uids": sorted(omit),
        "preserve_tiebreak_confirmed_uids": sorted(args.preserve_uid),
        "r3_exclusion_count": len(r3_exclusions),
        "blind60_overlap": blind_overlap,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "report"
        },
    }
    outputs["report"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
