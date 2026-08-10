#!/usr/bin/env python3
"""Partition exact consensus truth into pairwise-CE-eligible and typed-only rows.

An unanchored ``MATCH_FAMILY_ONLY`` label is valid supervision for the
generative typed adjudicator, but it cannot truthfully label any individual
norm/metric pair as FAMILY.  Such rows are retained in an explicit exclusion
artifact rather than guessed or mislabeled.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_nemotron_ce_pairs import DECISIONS, _family_anchors
from .common import normalize_space, read_jsonl, sha256_file, write_jsonl


def partition(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path = path.resolve()
    rows = list(read_jsonl(path))
    uids = [normalize_space(row.get("norm_uid")) for row in rows]
    if not rows or "" in uids or len(uids) != len(set(uids)):
        raise ValueError("truth is empty or has missing/duplicate norm_uid values")
    tasks = {normalize_space(row.get("task")) for row in rows}
    bank_hashes = {normalize_space(row.get("current_bank_source_sha256")) for row in rows}
    if "" in tasks or len(tasks) != 1 or "" in bank_hashes or len(bank_hashes) != 1:
        raise ValueError("truth task or bank identity is missing/mixed")
    group_splits: dict[str, set[str]] = defaultdict(set)
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        uid = normalize_space(row["norm_uid"])
        decision = normalize_space(row.get("decision"))
        split = normalize_space(row.get("split"))
        group = normalize_space(row.get("source_group") or row.get("split_group"))
        if decision not in DECISIONS or split not in {"train", "dev", "test", "blind"} or not group:
            raise ValueError(f"invalid decision/split/source group: {uid}")
        group_splits[group].add(split)
        if decision == "MATCH_FAMILY_ONLY" and not _family_anchors(row):
            excluded.append({
                **row,
                "ce_exclusion_reason": "unanchored_family_only_has_no_truthful_pair_label",
                "gemma_typed_eligible": True,
                "ce_pair_eligible": False,
            })
        else:
            eligible.append({**row, "ce_pair_eligible": True})
    crossing = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if crossing:
        raise ValueError(f"source groups cross splits: {list(crossing)[:3]}")
    report = {
        "schema_version": "silver-match-v3-ce-eligible-truth-report-v1",
        "status": "PARTITIONED_WITHOUT_INFERRED_FAMILY_ANCHORS",
        "task": next(iter(tasks)),
        "bank_source_sha256": next(iter(bank_hashes)),
        "input": {"path": str(path), "sha256": sha256_file(path), "count": len(rows)},
        "ce_pair_eligible_count": len(eligible),
        "typed_only_excluded_count": len(excluded),
        "input_decision_counts": dict(sorted(Counter(row["decision"] for row in rows).items())),
        "eligible_split_counts": dict(sorted(Counter(row["split"] for row in eligible).items())),
        "excluded_split_counts": dict(sorted(Counter(row["split"] for row in excluded).items())),
        "source_groups_crossing_splits": 0,
        "policy": {
            "unanchored_match_family_only": "exclude_from_pairwise_ce_only",
            "retained_for_gemma_typed_adjudicator": True,
            "family_anchor_inference_from_free_text_reason": False,
        },
    }
    return eligible, excluded, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--excluded", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    outputs = [Path(args.output).resolve(), Path(args.excluded).resolve(), Path(args.report).resolve()]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite CE truth partition")
    eligible, excluded, report = partition(Path(args.truth))
    write_jsonl(outputs[0], eligible)
    write_jsonl(outputs[1], excluded)
    report["outputs"] = {
        "eligible": {"path": str(outputs[0]), "sha256": sha256_file(outputs[0]), "count": len(eligible)},
        "typed_only": {"path": str(outputs[1]), "sha256": sha256_file(outputs[1]), "count": len(excluded)},
    }
    outputs[2].parent.mkdir(parents=True, exist_ok=True)
    outputs[2].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
