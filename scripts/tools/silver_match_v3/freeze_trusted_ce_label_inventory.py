#!/usr/bin/env python3
"""Freeze a conservative trusted-label inventory for learned CE supervision."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .adjudicate_gemma import DECISIONS
from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .filter_strong_ce_supervision import is_weak_forced


def classify(row: Mapping[str, Any]) -> tuple[int, str] | None:
    source = normalize_space(row.get("label_source"))
    confidence = normalize_space(row.get("confidence"))
    if is_weak_forced(row):
        return None
    if source in {
        "strict_three_model_exact_high_consensus",
        "exact_multi_pass_consensus",
    } and confidence in {"high", "medium"}:
        return 5, "independent_exact_consensus"
    if source == "independent_subagent" and confidence in {"high", "medium"}:
        return 4, "independent_strong_label"
    if source == "sonnet_audit":
        anchor = (row.get("notes") or {}).get("anchor_gate") or {}
        if row.get("legacy_candidate_valid") is True and int(anchor.get("good_exact", 0)) >= 2:
            return 3, "sonnet_audit_validated_unique_bridge"
        return None
    if source in {"sonnet_full", "sonnet_pilot"}:
        if (
            confidence == "high"
            and row.get("legacy_candidate_valid") is True
            and normalize_space(row.get("bridge_method")) == "normalized_current_name_unique"
            and normalize_space(row.get("production_uid_bridge_method"))
            == "alias_unique_norm_exact"
            and int((((row.get("notes") or {}).get("anchor_gate") or {}).get("good_exact", 0)))
            >= 2
        ):
            return 2, "sonnet_high_unique_bridge_anchor_pass"
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs = [Path(value).resolve() for value in args.input]
    output = Path(args.output).resolve()
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta_path.exists():
        raise FileExistsError(output)

    selected: dict[str, tuple[int, dict[str, Any], str]] = {}
    counts: Counter[str] = Counter()
    exclusion: Counter[str] = Counter()
    conflict: Counter[str] = Counter()
    for path in inputs:
        for row in read_jsonl(path):
            counts["input_rows"] += 1
            row_task = normalize_space(row.get("task"))
            if row_task and row_task != args.task:
                exclusion["other_task"] += 1
                continue
            uid = normalize_space(row.get("norm_uid"))
            decision = normalize_space(row.get("decision"))
            metric_id = normalize_space(row.get("metric_id"))
            if not uid or decision not in DECISIONS:
                exclusion["missing_uid_or_decision"] += 1
                continue
            if decision == "MATCH" and not metric_id:
                exclusion["match_without_metric"] += 1
                continue
            if decision != "MATCH" and metric_id:
                exclusion["typed_nonmatch_with_metric"] += 1
                continue
            trusted = classify(row)
            if trusted is None:
                if is_weak_forced(row):
                    exclusion["weak_forced"] += 1
                elif normalize_space(row.get("confidence")) == "low":
                    exclusion["low_confidence"] += 1
                else:
                    exclusion[f"untrusted_source:{normalize_space(row.get('label_source')) or 'UNSPECIFIED'}"] += 1
                continue
            priority, reason = trusted
            rendered = dict(row)
            rendered.update(
                {
                    "task": args.task,
                    "trusted_ce_inventory": True,
                    "trusted_ce_reason": reason,
                    "trusted_ce_source_path": str(path),
                    "ce_weak_forced_positive": False,
                }
            )
            previous = selected.get(uid)
            if previous is None:
                selected[uid] = (priority, rendered, str(path))
                continue
            previous_priority, previous_row, _ = previous
            same = (
                normalize_space(previous_row.get("decision")) == decision
                and normalize_space(previous_row.get("metric_id")) == metric_id
            )
            if same:
                conflict["duplicate_identical"] += 1
                if priority > previous_priority:
                    selected[uid] = (priority, rendered, str(path))
                continue
            conflict["conflicting_label"] += 1
            if priority == previous_priority:
                raise ValueError(f"equal-priority trusted-label conflict for {uid}")
            if priority > previous_priority:
                conflict["higher_priority_replaced_lower"] += 1
                selected[uid] = (priority, rendered, str(path))
            else:
                conflict["lower_priority_dropped"] += 1

    rows = [selected[uid][1] for uid in sorted(selected)]
    if not rows:
        raise ValueError("trusted CE inventory is empty")
    write_jsonl(output, rows)
    label_sources = Counter(normalize_space(row.get("label_source")) or "UNSPECIFIED" for row in rows)
    decisions = Counter(normalize_space(row.get("decision")) for row in rows)
    reasons = Counter(normalize_space(row.get("trusted_ce_reason")) for row in rows)
    meta = {
        "schema_version": "silver-match-v3-trusted-ce-label-inventory-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "policy": {
            "sonnet": "high-confidence unique current-bank bridge with anchor gate; audited valid rows",
            "independent": "medium/high only",
            "exact_consensus": "medium/high only",
            "weak_forced_as_exact_positive": False,
            "low_confidence": False,
            "conflict_resolution": "higher provenance priority; equal-priority conflict is fatal",
        },
        "inputs": {str(path): sha256_file(path) for path in inputs},
        "audit": {
            "input_rows": counts["input_rows"],
            "output_rows": len(rows),
            "decision_counts": dict(sorted(decisions.items())),
            "label_source_counts": dict(sorted(label_sources.items())),
            "trust_reason_counts": dict(sorted(reasons.items())),
            "exclusions": dict(sorted(exclusion.items())),
            "deduplication_and_conflicts": dict(sorted(conflict.items())),
            "weak_forced_rows_used_as_exact_positives": 0,
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**meta, "meta_sha256": sha256_file(meta_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
