#!/usr/bin/env python3
"""Freeze train/dev/test CE pair files without changing pair payloads."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl


REPORT_SCHEMA = "silver-match-v3-nemotron-ce-split-report-v1"
SPLITS = ("train", "dev", "test")


def _ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def split_rows(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen_pairs: set[tuple[str, str]] = set()
    uid_split: dict[str, str] = {}
    uid_decision: dict[str, str] = {}
    exact_uids: set[str] = set()
    reject_uids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    acceptable_as_reject = 0
    relation_id_mismatches = 0
    expected_relation_ids = {"REJECT": 0, "FAMILY": 1, "EXACT": 2}
    for row in read_jsonl(path):
        uid = normalize_space(row.get("norm_uid"))
        metric_id = normalize_space(row.get("metric_id"))
        split = normalize_space(row.get("split"))
        group = str(row.get("source_group") or "").strip()
        if not uid or not metric_id or split not in buckets or not group:
            raise ValueError(f"invalid CE pair identity/split: {uid}/{metric_id}")
        pair = (uid, metric_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate CE pair: {uid}/{metric_id}")
        seen_pairs.add(pair)
        prior_split = uid_split.setdefault(uid, split)
        if prior_split != split:
            raise ValueError(f"norm UID crosses pair splits: {uid}")
        decision = normalize_space(row.get("decision"))
        prior_decision = uid_decision.setdefault(uid, decision)
        if prior_decision != decision:
            raise ValueError(f"norm UID crosses truth decisions: {uid}")
        relation = normalize_space(row.get("relation"))
        if row.get("relation_id") != expected_relation_ids.get(relation):
            relation_id_mismatches += 1
        acceptable = {
            normalize_space(value)
            for value in (row.get("acceptable_metric_ids") or [])
        }
        acceptable.discard("")
        if relation == "REJECT" and metric_id in acceptable:
            acceptable_as_reject += 1
        if relation == "EXACT":
            exact_uids.add(uid)
        elif relation == "REJECT":
            reject_uids.add(uid)
        group_splits[group].add(split)
        buckets[split].append(row)
    if not seen_pairs:
        raise ValueError("CE pair input is empty")
    crossed = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if crossed:
        raise ValueError(f"source groups cross pair splits: {len(crossed)}")
    if acceptable_as_reject or relation_id_mismatches:
        raise ValueError(
            "CE relation integrity failed: "
            f"acceptable_as_reject={acceptable_as_reject}, "
            f"relation_id_mismatches={relation_id_mismatches}"
        )
    audit: dict[str, Any] = {
        "input_pair_count": len(seen_pairs),
        "unique_norm_count": len(uid_split),
        "unique_source_group_count": len(group_splits),
        "source_groups_crossing_splits": 0,
        "acceptable_metric_as_reject_count": 0,
        "relation_id_mismatch_count": 0,
        "splits": {},
    }
    for split, rows in buckets.items():
        if not rows:
            raise ValueError(f"CE split is empty: {split}")
        split_uids = {normalize_space(row["norm_uid"]) for row in rows}
        match_uids = {
            uid for uid in split_uids if uid_decision[uid] == "MATCH"
        }
        typed_nonmatch_uids = split_uids - match_uids
        exact_match_uids = match_uids & exact_uids
        reject_typed_uids = typed_nonmatch_uids & reject_uids
        audit["splits"][split] = {
            "pair_count": len(rows),
            "norm_count": len(split_uids),
            "source_group_count": len(
                {str(row["source_group"]).strip() for row in rows}
            ),
            "metric_count": len({normalize_space(row["metric_id"]) for row in rows}),
            "match_norm_count": len(match_uids),
            "match_norms_with_exact_pair": len(exact_match_uids),
            "match_exact_pair_coverage": (
                len(exact_match_uids) / len(match_uids) if match_uids else None
            ),
            "typed_nonmatch_norm_count": len(typed_nonmatch_uids),
            "typed_nonmatch_norms_with_reject_pair": len(reject_typed_uids),
            "typed_nonmatch_reject_pair_coverage": (
                len(reject_typed_uids) / len(typed_nonmatch_uids)
                if typed_nonmatch_uids
                else None
            ),
            "relation_counts": dict(
                sorted(Counter(str(row["relation"]) for row in rows).items())
            ),
            "decision_counts": dict(
                sorted(Counter(str(row["decision"]) for row in rows).items())
            ),
            "gradient_eligible_counts": dict(
                sorted(
                    Counter(str(bool(row.get("gradient_eligible"))) for row in rows).items()
                )
            ),
        }
    return buckets, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--builder-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="existing_truth.compact400k.v2")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    builder_report_path = Path(args.builder_report).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve()
    outputs = {
        split: output_dir / f"{args.prefix}.{split}.pairs.jsonl" for split in SPLITS
    }
    if report_path.exists() or any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite frozen CE split outputs/report")
    buckets, audit = split_rows(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_refs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        write_jsonl(outputs[split], buckets[split])
        output_refs[split] = {**_ref(outputs[split]), "count": len(buckets[split])}
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "SOURCE_DISJOINT_CE_SPLITS_READY",
        "input": _ref(input_path),
        "builder_report": _ref(builder_report_path),
        "audit": audit,
        "outputs": output_refs,
        "training_contract": {
            "train_pairs": output_refs["train"],
            "dev_pairs": output_refs["dev"],
            "held_out_test_pairs": output_refs["test"],
            "test_gradient_eligible": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
