#!/usr/bin/env python3
"""Validate truth-blind structured predictions for a human anchor pack."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .adjudicate_gemma import CONFIDENCES, DECISIONS
from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--raw-label-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--annotator", default="codex-gpt-5.6-sol-high-anchor")
    args = parser.parse_args()
    pack = Path(args.pack_root).resolve()
    raw_root = Path(args.raw_label_dir).resolve()
    output, report_path = Path(args.output).resolve(), Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite anchor validation")
    validation_path = pack / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    task = str(validation["task"])
    bank = json.loads((pack / "bank.json").read_text(encoding="utf-8"))
    bank_ids = [str(row["metric_id"]) for row in bank["metrics"]]
    if bank.get("source_sha256") != validation.get("bank_source_sha256"):
        raise ValueError("anchor bank identity mismatch")
    items = list(read_jsonl(pack / "items.jsonl"))
    by_uid = {str(row["norm_uid"]): row for row in items}
    labels = {}
    raw_hashes = {}
    for chunk_path in sorted((pack / "chunks").glob("part-*.jsonl")):
        chunk = chunk_path.stem
        expected = {str(row["norm_uid"]) for row in read_jsonl(chunk_path)}
        raw_path = raw_root / f"{chunk}.json"
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_hashes[str(raw_path)] = sha256_file(raw_path)
        values = payload.get("labels") or []
        observed = {str(row.get("norm_uid") or "") for row in values}
        if payload.get("task") != task or payload.get("chunk_id") != chunk:
            raise ValueError(f"anchor raw task/chunk mismatch: {raw_path}")
        if len(values) != len(expected) or observed != expected:
            raise ValueError(f"anchor raw UID coverage mismatch: {raw_path}")
        for value in values:
            uid = str(value["norm_uid"])
            if uid in labels:
                raise ValueError(f"duplicate anchor prediction: {uid}")
            decision = str(value.get("decision") or "").upper()
            confidence = str(value.get("confidence") or "").lower()
            metric_id = value.get("metric_id")
            metric_id = None if metric_id is None else str(metric_id)
            reason = str(value.get("reason") or "").strip()
            if decision not in DECISIONS or confidence not in CONFIDENCES or not reason:
                raise ValueError(f"invalid anchor prediction: {uid}")
            if decision == "MATCH":
                if metric_id not in bank_ids:
                    raise ValueError(f"anchor metric absent from bank: {uid}/{metric_id}")
            elif metric_id is not None:
                raise ValueError(f"anchor abstention carries metric: {uid}")
            item = by_uid[uid]
            labels[uid] = {
                "schema_version": item["schema_version"],
                "norm_uid": uid,
                "corpus": item["corpus"],
                "task": task,
                "row": item["row"],
                "decision": decision,
                "metric_id": metric_id,
                "confidence": confidence,
                "reason": reason,
                "candidate_ids": bank_ids,
                "candidate_bank_source_sha256": validation["bank_source_sha256"],
                "annotator": args.annotator,
                "label_source": "truth_blind_anchor_audit",
            }
    if set(labels) != set(by_uid):
        raise ValueError("anchor predictions are incomplete")
    ordered = [labels[str(row["norm_uid"])] for row in items]
    write_jsonl(output, ordered)
    report = {
        "schema_version": "silver-match-v3-labeler-anchor-validation-v1",
        "task": task,
        "complete": True,
        "count": len(ordered),
        "decision_counts": dict(sorted(Counter(row["decision"] for row in ordered).items())),
        "confidence_counts": dict(sorted(Counter(row["confidence"] for row in ordered).items())),
        "bank_source_sha256": validation["bank_source_sha256"],
        "pack_validation_sha256": sha256_file(validation_path),
        "truth_sha256": validation["truth"]["sha256"],
        "raw_chunks": raw_hashes,
        "output_sha256": sha256_file(output),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
