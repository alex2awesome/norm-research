#!/usr/bin/env python3
"""Join frozen typed-adapter dev predictions to gold and sample deterministic errors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .build_typed_lora_dev_error_packet import _category, _gold, _prompt_fields, _stable_key
from .common import read_jsonl, sha256_file


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dev_path = Path(args.dev_dataset).resolve()
    prediction_path = Path(args.predictions).resolve()
    meta_path = Path(args.inference_meta).resolve()
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("status") != "COMPLETE_DEV_ONLY_PAIRED_ORDER_INFERENCE"
        or meta.get("test_or_blind_rows_read") != 0
        or meta["outputs"]["original"]["sha256"] != sha256_file(prediction_path)
    ):
        raise ValueError("invalid or drifting dev-only inference metadata")
    dev = list(read_jsonl(dev_path))
    predictions = list(read_jsonl(prediction_path))
    by_uid = {str(row["norm_uid"]): row for row in predictions}
    if (
        len(by_uid) != len(predictions)
        or len(dev) != len(predictions)
        or {str(row["norm_uid"]) for row in dev} != set(by_uid)
        or any(row.get("split") != "dev" for row in dev)
    ):
        raise ValueError("dev/prediction UID universe differs")

    joined: list[dict[str, Any]] = []
    for source in dev:
        prediction_row = by_uid[str(source["norm_uid"])]
        prediction = {
            key: prediction_row.get(key)
            for key in ("decision", "metric_id", "confidence", "reason")
        }
        gold = _gold(source)
        fields = _prompt_fields(source["messages"][0]["content"])
        category = _category(gold, prediction)
        joined.append(
            {
                "norm_uid": source["norm_uid"],
                "source_group": source.get("source_group"),
                "view": source.get("view"),
                **fields,
                "candidate_metric_ids": source["candidate_metric_ids"],
                "gold": gold,
                "gold_metric_card": fields["candidate_cards"].get(gold.get("metric_id")),
                "prediction": prediction,
                "predicted_metric_card": fields["candidate_cards"].get(prediction.get("metric_id")),
                "raw_prediction": prediction_row.get("raw"),
                "prediction_provenance": {
                    "adapter_exposure": args.expected_exposure,
                    "order_mode": prediction_row.get("order_mode"),
                    "backend": "direct_batch_vllm_not_openai_server",
                },
                "audit_category": category,
            }
        )

    requested = {
        "correct_exact_match": args.correct,
        "false_match": args.false_match,
        "missed_gold_match": args.missed,
        "abstention_type_error": args.abstention_error,
    }
    available: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for category, count in requested.items():
        values = [row for row in joined if row["audit_category"] == category]
        values.sort(key=lambda row: _stable_key(row, category))
        available[category] = len(values)
        selected.extend(
            {"sample_category": category, "sample_rank": rank, **row}
            for rank, row in enumerate(values[:count], 1)
        )
    if any(available[key] < value for key, value in requested.items()):
        raise ValueError(f"insufficient requested examples: {available}")

    root.mkdir(parents=True, exist_ok=False)
    packet_path = root / "manual_audit_packet.c2.original_order.jsonl"
    with packet_path.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    gold_match = sum(row["gold"]["decision"] == "MATCH" for row in joined)
    predicted_match = sum(row["prediction"]["decision"] == "MATCH" for row in joined)
    correct = sum(row["audit_category"] == "correct_exact_match" for row in joined)
    report = {
        "schema_version": "silver-match-v3-humor-typed-c2-dev-manual-audit-v1",
        "status": "COMPLETE_DEV_ONLY_DETERMINISTIC_MANUAL_AUDIT_PACKET",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter_exposure": args.expected_exposure,
        "dev_dataset": _artifact(dev_path),
        "predictions": _artifact(prediction_path),
        "inference_meta": _artifact(meta_path),
        "selection": {
            "method": "ascending_sha256(humor-c2-dev-audit-v1\\0category\\0norm_uid)",
            "requested": requested,
            "available": available,
            "selected": dict(Counter(row["sample_category"] for row in selected)),
        },
        "original_order_counts": {
            "rows": len(joined),
            "gold_match": gold_match,
            "predicted_match": predicted_match,
            "correct_exact_match": correct,
            "exact_precision": correct / predicted_match if predicted_match else None,
            "exact_recall": correct / gold_match if gold_match else None,
            "decision_counts": dict(Counter(row["prediction"]["decision"] for row in joined)),
        },
        "artifact": _artifact(packet_path),
        "test_or_blind_rows_read": 0,
    }
    report_path = root / "REPORT.json"
    _write_json(report_path, report)
    return {**report, "report": _artifact(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--inference-meta", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-exposure", type=int, required=True)
    parser.add_argument("--correct", type=int, default=3)
    parser.add_argument("--false-match", type=int, default=4)
    parser.add_argument("--missed", type=int, default=3)
    parser.add_argument("--abstention-error", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
