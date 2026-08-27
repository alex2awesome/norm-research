#!/usr/bin/env python3
"""Build balanced exact-match verifier GEPA slates from frozen truth.

Every source group appears once.  Existing wrong/typed proposals are negatives;
enough correct proposals are deterministically converted to their strongest
available sibling/confusion negative to balance the slate.  Remaining rows use
the exact gold metric as positive proposals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path) -> dict[str, dict[str, Any]]:
    values = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in values}
    if len(output) != len(values):
        raise ValueError(f"duplicate norm_uid in {path}")
    return output


def _hash(uid: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{uid}".encode()).hexdigest()


def _confusion_ids(truth: dict[str, Any]) -> list[str]:
    output: list[str] = []
    predictions = truth.get("source_predictions") or {}
    values = predictions.values() if isinstance(predictions, dict) else predictions
    for row in values:
        if not isinstance(row, dict):
            continue
        metric_id = row.get("metric_id")
        if metric_id is not None and str(metric_id) not in output:
            output.append(str(metric_id))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=1297)
    parser.add_argument("--exclude-reference", action="append", default=[])
    parser.add_argument(
        "--maximize-gold-positives",
        action="store_true",
        help="Use every in-slate exact gold MATCH as a positive-capable proposal.",
    )
    parser.add_argument(
        "--balance-by-downsampling",
        action="store_true",
        help="Deterministically downsample the larger pre-conversion class.",
    )
    args = parser.parse_args()
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in ("truth", "primary", "candidates")
    }
    rows = {name: _index(path) for name, path in paths.items()}
    uids = set(rows["truth"])
    if any(set(value) != uids for value in rows.values()):
        raise ValueError("truth, primary, and candidates lack exact UID coverage")
    exclusion_paths = [Path(value).resolve() for value in args.exclude_reference]
    excluded = {
        str(row["norm_uid"])
        for path in exclusion_paths
        for row in read_jsonl(path)
    }
    excluded_present = uids & excluded
    uids -= excluded
    if not uids:
        raise ValueError("all verifier GEPA rows were excluded")

    correct, forced_negative = [], []
    for uid in sorted(uids):
        truth, primary = rows["truth"][uid], rows["primary"][uid]
        candidate_ids = {
            str(row["metric_id"])
            for row in rows["candidates"][uid].get("candidates") or []
        }
        exact_gold_available = (
            truth.get("decision") == "MATCH"
            and str(truth.get("metric_id")) in candidate_ids
        )
        exact_primary = exact_gold_available and str(truth.get("metric_id")) == str(
            primary.get("metric_id")
        )
        positive_capable = (
            exact_gold_available if args.maximize_gold_positives else exact_primary
        )
        if positive_capable:
            correct.append(uid)
        else:
            forced_negative.append(uid)
    downsampled: set[str] = set()
    if args.balance_by_downsampling:
        target = min(len(correct), len(forced_negative))
        if target == 0:
            raise ValueError("cannot balance a verifier slate with an empty class")
        keep_correct = set(
            sorted(correct, key=lambda uid: (_hash(uid, args.seed), uid))[:target]
        )
        keep_negative = set(
            sorted(forced_negative, key=lambda uid: (_hash(uid, args.seed), uid))[:target]
        )
        downsampled = uids - keep_correct - keep_negative
        uids = keep_correct | keep_negative
        correct = [uid for uid in correct if uid in uids]
        forced_negative = [uid for uid in forced_negative if uid in uids]
        converted: set[str] = set()
    else:
        negative_target = len(uids) // 2
        convert_count = max(0, negative_target - len(forced_negative))
        if convert_count > len(correct):
            raise ValueError("cannot construct balanced verifier slate")
        converted = set(
            sorted(correct, key=lambda uid: (_hash(uid, args.seed), uid))[:convert_count]
        )

    output_primary = []
    report_rows = []
    for uid in sorted(uids):
        truth, original = rows["truth"][uid], rows["primary"][uid]
        candidate_ids = [
            str(row["metric_id"]) for row in rows["candidates"][uid].get("candidates") or []
        ]
        if uid in correct and uid not in converted:
            proposal_id, target = str(truth["metric_id"]), "CONFIRM_MATCH"
            source = "gold_positive"
        elif uid in converted:
            gold = str(truth["metric_id"])
            preferred = [
                metric_id
                for metric_id in _confusion_ids(truth)
                if metric_id != gold and metric_id in candidate_ids
            ]
            proposal_id = (preferred + [value for value in candidate_ids if value != gold])[0]
            target, source = "REJECT", "hard_sibling_negative"
        else:
            proposal_id = str(original["metric_id"])
            target, source = "REJECT", "observed_wrong_or_typed_negative"
        if proposal_id not in candidate_ids:
            raise ValueError(f"proposal absent from candidate slate for {uid}: {proposal_id}")
        output_primary.append(
            {
                **original,
                "decision": "MATCH",
                "metric_id": proposal_id,
                "verifier_gepa_target": target,
                "verifier_gepa_proposal_source": source,
            }
        )
        report_rows.append(
            {
                "norm_uid": uid,
                "source_group": truth.get("source_group"),
                "proposal_metric_id": proposal_id,
                "truth_decision": truth.get("decision"),
                "truth_metric_id": truth.get("metric_id"),
                "target": target,
                "proposal_source": source,
            }
        )

    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    outputs = {
        "truth": (output / "truth.jsonl", [rows["truth"][uid] for uid in sorted(uids)]),
        "primary": (output / "primary.jsonl", output_primary),
        "candidates": (output / "candidates.jsonl", [rows["candidates"][uid] for uid in sorted(uids)]),
        "targets": (output / "targets.jsonl", report_rows),
    }
    for path, values in outputs.values():
        write_jsonl(path, values)
    payload = {
        "schema_version": "silver-match-v3-balanced-verifier-gepa-train-v1",
        "seed": args.seed,
        "count": len(uids),
        "positive_count": sum(row["target"] == "CONFIRM_MATCH" for row in report_rows),
        "negative_count": sum(row["target"] == "REJECT" for row in report_rows),
        "proposal_source_counts": {
            source: sum(row["proposal_source"] == source for row in report_rows)
            for source in sorted({row["proposal_source"] for row in report_rows})
        },
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "exclusion_hashes": {str(path): sha256_file(path) for path in exclusion_paths},
        "excluded_present_count": len(excluded_present),
        "maximize_gold_positives": args.maximize_gold_positives,
        "balance_by_downsampling": args.balance_by_downsampling,
        "downsampled_count": len(downsampled),
        "output_hashes": {name: sha256_file(path) for name, (path, _) in outputs.items()},
    }
    report = output / "REPORT.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**payload, "report_sha256": sha256_file(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
