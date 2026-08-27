#!/usr/bin/env python3
"""Audit a frozen pointwise CE queue's per-metric pair exposure without a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import train_cross_encoder as trainer
from .common import sha256_file
from .launch_frozen_cross_encoder_queues import validate_queue


class AuditComplete(RuntimeError):
    pass


def audit(queue_path: Path) -> dict[str, Any]:
    queue_path = queue_path.resolve()
    queue, verified = validate_queue(queue_path)
    commands = queue.get("commands") or []
    if len(commands) != 3:
        raise ValueError("pointwise queue must contain three frozen variants")
    command = commands[0]["command"]
    if command[2:4] != ["-m", "scripts.tools.silver_match_v3.train_cross_encoder"]:
        raise ValueError("queue does not invoke the pointwise v3 trainer")

    result: dict[str, Any] = {}
    original_builder = trainer.build_training_pairs
    original_argv = list(sys.argv)

    def wrapped(labels: Any, bank: list[dict[str, Any]], candidate_ids: Any, **kwargs: Any) -> Any:
        rows = original_builder(labels, bank, candidate_ids, **kwargs)
        positive = Counter(
            str(row["metric_id"])
            for row in rows
            if float(row["label"]) == 1.0
        )
        negative = Counter(
            str(row["metric_id"])
            for row in rows
            if float(row["label"]) == 0.0
        )
        exposure = []
        failures = []
        for metric in bank:
            metric_id = str(metric["metric_id"])
            p = int(positive[metric_id])
            n = int(negative[metric_id])
            fraction = p / (p + n) if p + n else 0.0
            reasons = []
            if p and n == 0:
                reasons.append("positive_metric_never_negative")
            if p and n < p:
                reasons.append("negative_to_positive_ratio_below_1")
            if fraction > 0.5:
                reasons.append("positive_pair_fraction_above_0_5")
            row = {
                "metric_id": metric_id,
                "positive_pairs": p,
                "negative_pairs": n,
                "positive_pair_fraction": fraction,
                "gate_failures": reasons,
            }
            exposure.append(row)
            if reasons:
                failures.append(row)
        result.update(
            {
                "schema_version": "silver-match-v3-pointwise-ce-exposure-audit-v1",
                "status": (
                    "PASS_POINTWISE_EXPOSURE_GATE"
                    if not failures
                    else "FAIL_POINTWISE_EXPOSURE_GATE"
                ),
                "task": queue["task"],
                "queue": {
                    "path": str(queue_path),
                    "sha256": sha256_file(queue_path),
                },
                "verified_queue_artifact_count": len(verified),
                "gates": {
                    "positive_metric_must_have_negative": True,
                    "minimum_negative_to_positive_pair_ratio": 1.0,
                    "maximum_positive_pair_fraction": 0.5,
                },
                "bank_metric_count": len(bank),
                "positive_metric_count": sum(value > 0 for value in positive.values()),
                "positive_pair_count": sum(positive.values()),
                "negative_pair_count": sum(negative.values()),
                "pair_kind_counts": dict(
                    sorted(Counter(row["kind"] for row in rows).items())
                ),
                "zero_negative_positive_metric_count": sum(
                    1 for metric_id, value in positive.items() if value and not negative[metric_id]
                ),
                "failed_metric_count": len(failures),
                "failed_metrics": failures,
                "per_metric_exposure": exposure,
            }
        )
        raise AuditComplete

    trainer.build_training_pairs = wrapped
    try:
        sys.argv = [command[3], *command[4:]]
        try:
            trainer.train(trainer.parse_args())
        except AuditComplete:
            pass
    finally:
        trainer.build_training_pairs = original_builder
        sys.argv = original_argv
    if not result:
        raise RuntimeError("pointwise exposure audit did not execute")
    result["auditor"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    result["gpu_consumed"] = False
    result["model_initialized"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(Path(args.queue))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "status": result["status"],
                "task": result["task"],
                "failed_metric_count": result["failed_metric_count"],
                "zero_negative_positive_metric_count": result[
                    "zero_negative_positive_metric_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
