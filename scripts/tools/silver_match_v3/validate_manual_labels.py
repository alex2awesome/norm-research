#!/usr/bin/env python3
"""Validate and enrich independent manual/subagent matcher labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adjudicate_gemma import CONFIDENCES, DECISIONS
from .common import read_jsonl, sha256_file, write_jsonl
from .config import DEFAULT_OUTPUT_ROOT


def validate_labels(
    raw_labels: list[dict[str, Any]],
    items: dict[str, dict[str, Any]],
    bank_ids: dict[str, set[str]],
    *,
    annotator: str,
    candidate_ranks: dict[str, dict[str, int]] | None = None,
    bank_hashes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    output, seen = [], set()
    for index, label in enumerate(raw_labels):
        uid = str(label.get("norm_uid") or "")
        if uid not in items:
            raise ValueError(f"label {index}: unknown norm_uid {uid!r}")
        if uid in seen:
            raise ValueError(f"label {index}: duplicate norm_uid {uid}")
        seen.add(uid)
        decision = str(label.get("decision") or "").upper()
        if decision not in DECISIONS:
            raise ValueError(f"label {index}: invalid decision {decision!r}")
        confidence = str(label.get("confidence") or "").lower()
        if confidence not in CONFIDENCES:
            raise ValueError(f"label {index}: invalid confidence {confidence!r}")
        reason = str(label.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"label {index}: missing reason")
        metric_id = label.get("metric_id")
        metric_id = None if metric_id is None else str(metric_id)
        task = items[uid]["task"]
        if decision == "MATCH":
            if metric_id not in bank_ids[task]:
                raise ValueError(
                    f"label {index}: metric {metric_id!r} is not in the {task} bank"
                )
        elif metric_id is not None:
            raise ValueError(f"label {index}: abstention must have metric_id null")
        rank = None
        if metric_id and candidate_ranks is not None:
            rank = candidate_ranks.get(uid, {}).get(metric_id)
        item = items[uid]
        output.append(
            {
                "schema_version": item["schema_version"],
                "norm_uid": uid,
                "corpus": item["corpus"],
                "task": task,
                "row": item["row"],
                "split_group": item.get("split_group"),
                "split": item.get("split"),
                "decision": decision,
                "metric_id": metric_id,
                "current_bank_source_sha256": (
                    bank_hashes.get(task) if bank_hashes is not None else None
                ),
                "confidence": confidence,
                "reason": reason,
                "label_source": "independent_subagent",
                "annotator": annotator,
                "retrieved_rank": rank,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--items", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidates")
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    manifest_path, items_path, labels_path = map(
        Path, (args.manifest, args.items, args.labels)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = {row["norm_uid"]: row for row in read_jsonl(items_path)}
    bank_ids = {}
    bank_hashes = {}
    for task, meta in manifest["banks"].items():
        metrics = json.loads(Path(meta["path"]).read_text(encoding="utf-8"))["metrics"]
        bank_ids[task] = {row["metric_id"] for row in metrics}
        bank_hashes[task] = str(meta["source_sha256"])
    candidate_ranks = None
    if args.candidates:
        candidate_ranks = {
            row["norm_uid"]: {
                candidate["metric_id"]: int(candidate["rank"])
                for candidate in row.get("candidates") or []
            }
            for row in read_jsonl(Path(args.candidates))
        }
    raw = list(read_jsonl(labels_path))
    output = validate_labels(
        raw,
        items,
        bank_ids,
        annotator=args.annotator,
        candidate_ranks=candidate_ranks,
        bank_hashes=bank_hashes,
    )
    if args.require_complete and len(output) != len(items):
        missing = sorted(set(items) - {row["norm_uid"] for row in output})
        raise ValueError(f"labels incomplete: {len(missing)} missing; first={missing[:3]}")
    output_path = Path(args.output)
    write_jsonl(output_path, output)
    meta = {
        "manifest_sha256": sha256_file(manifest_path),
        "items_sha256": sha256_file(items_path),
        "labels_sha256": sha256_file(labels_path),
        "annotator": args.annotator,
        "count": len(output),
        "complete": len(output) == len(items),
        "match_count": sum(row["decision"] == "MATCH" for row in output),
        "retrieval_miss_count": sum(
            row["decision"] == "MATCH" and row["retrieved_rank"] is None for row in output
        ),
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
