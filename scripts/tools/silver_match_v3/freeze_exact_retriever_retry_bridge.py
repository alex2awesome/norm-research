#!/usr/bin/env python3
"""Freeze a task-local retriever retry bridge from exact train-only truth."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in output or len(output) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return output


def _expect(name: str, observed: int, expected: int | None) -> None:
    if expected is not None and observed != expected:
        raise ValueError(f"{name} count changed: {observed} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--prior-labels", required=True)
    parser.add_argument("--upstream-roles", required=True)
    parser.add_argument("--optimize-truth", required=True)
    parser.add_argument("--select-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-prior-input", type=int)
    parser.add_argument("--expected-prior-train", type=int)
    parser.add_argument("--expected-optimize-input", type=int)
    parser.add_argument("--expected-optimize-matches", type=int)
    parser.add_argument("--expected-select-input", type=int)
    args = parser.parse_args()

    paths = {
        "prior": Path(args.prior_labels).resolve(),
        "roles": Path(args.upstream_roles).resolve(),
        "optimize": Path(args.optimize_truth).resolve(),
        "select": Path(args.select_truth).resolve(),
    }
    output, report_path = Path(args.output).resolve(), Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite immutable retriever retry bridge")
    prior, roles = _index(paths["prior"]), _index(paths["roles"])
    optimize, select = _index(paths["optimize"]), _index(paths["select"])
    _expect("prior input", len(prior), args.expected_prior_input)
    _expect("optimize input", len(optimize), args.expected_optimize_input)
    _expect("select input", len(select), args.expected_select_input)

    needed = set(prior) | set(optimize) | set(select)
    missing_roles = sorted(needed - set(roles))
    if missing_roles:
        raise ValueError(f"rows missing authoritative upstream roles: {missing_roles[:3]}")
    bank_hashes: set[str] = set()
    prior_role_counts: Counter[str] = Counter()
    selected: dict[str, dict[str, Any]] = {}
    provenance_counts: Counter[str] = Counter()

    def validate_common(uid: str, row: dict[str, Any]) -> dict[str, Any]:
        role = roles[uid]
        if str(row.get("task")) != args.task or str(role.get("task")) != args.task:
            raise ValueError(f"task mismatch: {uid}")
        if str(row.get("corpus")) != str(role.get("corpus")):
            raise ValueError(f"corpus mismatch: {uid}")
        supplied_group = str(row.get("split_group") or row.get("source_group") or "")
        allowed_groups = {
            str(role.get("source_group") or ""),
            str(role.get("retriever_source_group") or ""),
        }
        if supplied_group not in allowed_groups:
            raise ValueError(f"source-group mismatch: {uid}")
        bank_hash = str(row.get("current_bank_source_sha256") or "")
        if not bank_hash:
            raise ValueError(f"missing bank hash: {uid}")
        bank_hashes.add(bank_hash)
        return role

    for uid, row in prior.items():
        role = validate_common(uid, row)
        upstream_split = str(role.get("split") or "")
        prior_role_counts[upstream_split] += 1
        if str(row.get("decision")) != "MATCH" or not row.get("metric_id"):
            raise ValueError(f"prior strong-human row is not an exact MATCH: {uid}")
        if str(row.get("label_source")) != "independent_subagent":
            raise ValueError(f"prior row is not from the frozen strong-human source: {uid}")
        if upstream_split != "train":
            continue
        selected[uid] = {
            **row,
            "split": "train",
            "split_group": role["source_group"],
            "source_group": role["retriever_source_group"],
            "training_eligible": True,
            "training_role": "gradient_candidate",
            "retriever_retry_teacher_provenance": "prior_strong_human_authoritative_train",
            "authoritative_upstream_role_sha256": sha256_file(paths["roles"]),
        }
        provenance_counts["prior_strong_human_authoritative_train"] += 1
    _expect("prior authoritative train", len(selected), args.expected_prior_train)

    optimize_matches = 0
    for uid, row in optimize.items():
        role = validate_common(uid, row)
        if str(role.get("split")) != "train":
            raise ValueError(f"optimize truth is not authoritative upstream train: {uid}")
        if (
            str(row.get("gepa_role")) != "optimize"
            or row.get("prompt_gradient_eligible") is not True
            or row.get("prompt_selection_eligible") is not False
        ):
            raise ValueError(f"invalid optimize truth role flags: {uid}")
        if str(row.get("decision")) != "MATCH":
            continue
        optimize_matches += 1
        supporters = row.get("agreement_sources") or []
        if len(supporters) < 2 or not row.get("metric_id"):
            raise ValueError(f"optimize MATCH lacks two-source exact consensus: {uid}")
        if uid in selected:
            raise ValueError(f"new optimize truth overlaps prior teacher UID: {uid}")
        selected[uid] = {
            **row,
            "split": "train",
            "split_group": role["source_group"],
            "source_group": role["retriever_source_group"],
            "training_eligible": True,
            "training_role": "gradient_candidate",
            "retriever_retry_teacher_provenance": "new_exact_optimize_multi_pass_consensus",
            "training_authorization": "root_directive_2026-07-12_exact_optimize_match_bridge",
            "authoritative_upstream_role_sha256": sha256_file(paths["roles"]),
        }
        provenance_counts["new_exact_optimize_multi_pass_consensus"] += 1
    _expect("optimize exact MATCH", optimize_matches, args.expected_optimize_matches)

    select_uids, select_groups = set(select), set()
    for uid, row in select.items():
        role = validate_common(uid, row)
        if (
            str(role.get("split")) != "train"
            or str(row.get("gepa_role")) != "select"
            or row.get("prompt_selection_eligible") is not True
            or row.get("prompt_gradient_eligible") is not False
            or row.get("training_eligible") is not False
        ):
            raise ValueError(f"invalid immutable select truth role flags: {uid}")
        select_groups.add(str(role["retriever_source_group"]))

    selected_groups = [str(row["source_group"]) for row in selected.values()]
    if len(selected_groups) != len(set(selected_groups)):
        raise ValueError("retry teachers are not unique by retriever source group")
    if set(selected) & select_uids or set(selected_groups) & select_groups:
        raise ValueError("retry teachers overlap immutable select rows or groups")
    if len(bank_hashes) != 1:
        raise ValueError(f"inputs do not share one frozen bank hash: {sorted(bank_hashes)}")

    rows = [selected[uid] for uid in sorted(selected)]
    write_jsonl(output, rows)
    report = {
        "schema_version": "silver-match-v3-exact-retriever-retry-bridge-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "status": "FROZEN_TRAIN_ONLY_EXACT_TEACHERS",
        "bank_source_sha256": next(iter(bank_hashes)),
        "counts": {
            "prior_input": len(prior),
            "prior_authoritative_roles": dict(sorted(prior_role_counts.items())),
            "prior_selected_train": provenance_counts[
                "prior_strong_human_authoritative_train"
            ],
            "new_optimize_input": len(optimize),
            "new_optimize_exact_match_selected": provenance_counts[
                "new_exact_optimize_multi_pass_consensus"
            ],
            "select_rows_permanently_excluded": len(select),
            "final_teachers": len(rows),
            "final_source_groups": len(selected_groups),
            "final_metrics": len({str(row["metric_id"]) for row in rows}),
        },
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "audits": {
            "selected_uid_overlap_with_select": 0,
            "selected_source_group_overlap_with_select": 0,
            "all_selected_authoritative_upstream_train": True,
            "weak_or_forced_labels_selected": 0,
            "nonmatch_labels_selected": 0,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
