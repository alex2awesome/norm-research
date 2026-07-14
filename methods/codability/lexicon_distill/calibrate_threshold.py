"""Calibrate a hierarchy SAME-edge threshold on frozen development predictions only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .evaluate_similarity_lora import metrics


_DEV_SPLITS = frozenset({"pair_dev", "cold_dev_any", "cold_dev_both", "frontier_dev"})


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    example_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = str(row.get("split") or "")
            if split not in _DEV_SPLITS:
                raise ValueError(f"{path}:{line_number}: calibration accepts development splits only")
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in example_ids:
                raise ValueError(f"{path}:{line_number}: missing or duplicate example_id")
            example_ids.add(example_id)
            probabilities = row.get("probabilities")
            if (
                not isinstance(probabilities, list)
                or len(probabilities) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                    for value in probabilities
                )
                or abs(sum(probabilities) - 1) > 1e-5
            ):
                raise ValueError(f"{path}:{line_number}: invalid probability vector")
            truth = row.get("truth")
            if type(truth) is not int or truth not in (0, 1, 2):
                raise ValueError(f"{path}:{line_number}: invalid truth label")
            result.append(row)
    if not result:
        raise ValueError("empty calibration predictions")
    return result


def _at_threshold(rows: Iterable[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        probabilities = [float(value) for value in row["probabilities"]]
        prediction = 2 if probabilities[2] >= threshold else max((0, 1), key=probabilities.__getitem__)
        result.append({**row, "prediction": prediction})
    return result


def calibrate(
    predictions_path: str | Path,
    report_path: str | Path,
    *,
    target_precision: float = 0.60,
    minimum_recall: float = 0.50,
    protocol_id: str | None = None,
    adapter_sha256: str | None = None,
    protocol_sha256: str | None = None,
    adapter_file: str | Path | None = None,
    protocols_path: str | Path | None = None,
) -> dict[str, Any]:
    """Maximize SAME recall subject to the predeclared precision and recall floors."""
    if not 0 <= target_precision <= 1 or not 0 <= minimum_recall <= 1:
        raise ValueError("precision and recall thresholds must be in [0, 1]")
    source = Path(predictions_path).expanduser().resolve()
    destination = Path(report_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if adapter_sha256 is not None:
        adapter_sha256 = _digest(adapter_sha256, "adapter_sha256")
    if protocol_sha256 is not None:
        protocol_sha256 = _digest(protocol_sha256, "protocol_sha256")
    rows = _rows(source)
    levels = {str(row.get("level") or "") for row in rows}
    if len(levels) != 1 or next(iter(levels)) not in {"R1", "R2", "R3"}:
        raise ValueError(f"calibration requires exactly one valid level, found {sorted(levels)}")
    protocols = {str(row.get("protocol_id") or "") for row in rows}
    if protocol_id is not None:
        rows = [row for row in rows if row.get("protocol_id") == protocol_id]
        if not rows:
            raise ValueError(f"no development predictions for protocol {protocol_id}")
        protocols = {protocol_id}
    if len(protocols) != 1 or not next(iter(protocols)):
        raise ValueError(
            "calibration requires exactly one protocol; select R2 legacy/v2/v2.1 explicitly")
    selected_protocol = next(iter(protocols))
    adapter_ref = None
    if adapter_file is not None:
        adapter_path = Path(adapter_file).expanduser().resolve()
        observed_adapter_sha = _sha256(adapter_path)
        if adapter_sha256 is not None and adapter_sha256 != observed_adapter_sha:
            raise ValueError("adapter_sha256 does not match adapter_file")
        adapter_sha256 = observed_adapter_sha
        adapter_ref = {"path": str(adapter_path), "sha256": observed_adapter_sha}
    protocols_ref = None
    if protocols_path is not None:
        bundle_path = Path(protocols_path).expanduser().resolve()
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        protocol_row = bundle.get(selected_protocol)
        if not isinstance(protocol_row, dict) or not isinstance(protocol_row.get("text"), str):
            raise ValueError(f"protocol bundle lacks {selected_protocol}")
        observed_protocol_sha = hashlib.sha256(protocol_row["text"].encode()).hexdigest()
        if protocol_row.get("sha256") != observed_protocol_sha:
            raise ValueError(f"protocol bundle text/hash mismatch for {selected_protocol}")
        if protocol_sha256 is not None and protocol_sha256 != observed_protocol_sha:
            raise ValueError("protocol_sha256 does not match protocols_path")
        protocol_sha256 = observed_protocol_sha
        protocols_ref = {"path": str(bundle_path), "sha256": _sha256(bundle_path)}
    thresholds = sorted({0.0, 1.0, *[float(row["probabilities"][2]) for row in rows]})
    candidates = []
    for threshold in thresholds:
        current = metrics(_at_threshold(rows, threshold))
        candidates.append({"threshold": threshold, "metrics": current})
    eligible = [
        row
        for row in candidates
        if row["metrics"]["same_precision"] >= target_precision
        and row["metrics"]["same_recall"] >= minimum_recall
    ]
    lineage_complete = adapter_sha256 is not None and protocol_sha256 is not None
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["metrics"]["same_recall"],
                row["metrics"]["cohen_kappa"],
                row["metrics"]["macro_f1"],
                row["threshold"],
            ),
        )
        certified = lineage_complete
        reason = (
            "maximized SAME recall subject to development precision and recall floors"
            if lineage_complete
            else "missing adapter/protocol hash lineage; threshold is diagnostic only"
        )
    else:
        # This operating point may be useful diagnostically, but its explicit uncertified status
        # prevents a hierarchy builder from silently treating a failed calibration as production.
        selected = max(
            candidates,
            key=lambda row: (
                min(row["metrics"]["same_precision"], row["metrics"]["same_recall"]),
                row["metrics"]["same_f1"],
                row["metrics"]["cohen_kappa"],
            ),
        )
        certified = False
        reason = "no development threshold cleared both floors"
    report = {
        "schema_version": "gemma4-similarity-threshold-calibration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "predictions": {"path": str(source), "sha256": _sha256(source)},
        "n": len(rows),
        "selection_split": sorted({str(row["split"]) for row in rows}),
        "level": next(iter(levels)),
        "protocol_id": selected_protocol,
        "adapter_sha256": adapter_sha256,
        "protocol_sha256": protocol_sha256,
        "adapter": adapter_ref,
        "protocols": protocols_ref,
        "tasks": sorted({str(row.get("task") or "") for row in rows}),
        "target_same_precision": target_precision,
        "minimum_same_recall": minimum_recall,
        "certified": certified,
        "reason": reason,
        "selected_same_threshold": selected["threshold"],
        "selected_related_weight": 0.0,
        "selected_metrics": selected["metrics"],
        "n_thresholds_evaluated": len(candidates),
        "semantic_truth": "persisted LLM teacher labels; threshold selection performs no string comparison",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-precision", type=float, default=0.60)
    parser.add_argument("--minimum-recall", type=float, default=0.50)
    parser.add_argument("--protocol-id")
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--adapter-file")
    parser.add_argument("--protocols")
    args = parser.parse_args()
    print(
        json.dumps(
            calibrate(
                args.predictions,
                args.report,
                target_precision=args.target_precision,
                minimum_recall=args.minimum_recall,
                protocol_id=args.protocol_id,
                adapter_sha256=args.adapter_sha256,
                protocol_sha256=args.protocol_sha256,
                adapter_file=args.adapter_file,
                protocols_path=args.protocols,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
