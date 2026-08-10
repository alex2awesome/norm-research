#!/usr/bin/env python3
"""Restore frozen train/dev/test roles onto exact multi-pass consensus truth."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path, label: str) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"{label} has missing or duplicate norm_uid values")
    return indexed


def _teacher_reason(truth: dict[str, Any]) -> tuple[str, str | None]:
    predictions = truth.get("source_predictions") or {}
    supporters = list(truth.get("agreement_sources") or [])
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    candidates = []
    for source in supporters:
        row = predictions.get(source)
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip()
        if reason:
            candidates.append(
                (confidence_order.get(str(row.get("confidence") or "").lower(), 3), source, reason)
            )
    if candidates:
        _, source, reason = min(candidates)
        return reason, source
    reason = str(truth.get("reason") or "").strip()
    if not reason:
        raise ValueError(f"resolved consensus lacks any rationale: {truth.get('norm_uid')}")
    return reason, None


def materialize(pack_root: Path, consensus_report: Path, output_root: Path) -> dict[str, Any]:
    pack_root = pack_root.resolve()
    consensus_report = consensus_report.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)

    validation_path = pack_root / "validation.json"
    items_path = pack_root / "items.jsonl"
    bank_path = pack_root / "bank.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("source-pack items hash mismatch")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source-pack bank hash mismatch")

    report = json.loads(consensus_report.read_text(encoding="utf-8"))
    if (
        report.get("schema_version")
        != "silver-match-v3-exact-multi-pass-truth-report-v1"
        or report.get("complete") is not True
        or int(report.get("unresolved_count", -1)) != 0
    ):
        raise ValueError("consensus report is not complete exact multi-pass truth")
    source_ref = (report.get("inputs") or {}).get("source_pack_validation") or {}
    if source_ref.get("sha256") != sha256_file(validation_path):
        raise ValueError("consensus is bound to another source pack")
    resolved_ref = (report.get("outputs") or {}).get("resolved") or {}
    resolved_path = Path(str(resolved_ref.get("path") or ""))
    if not resolved_path.is_file() or sha256_file(resolved_path) != resolved_ref.get("sha256"):
        raise ValueError("resolved consensus artifact hash mismatch")

    items = _index(items_path, "source pack")
    resolved = _index(resolved_path, "resolved consensus")
    if set(items) != set(resolved):
        raise ValueError("complete consensus does not exactly cover frozen source UIDs")

    group_splits: dict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, Any]] = []
    for uid, item in items.items():
        truth = resolved[uid]
        teacher_reason, teacher_reason_source = _teacher_reason(truth)
        split = str(item.get("split") or item.get("predeclared_split") or "")
        role = str(item.get("collection_role") or "")
        group = str(item.get("split_group") or item.get("source_group") or "")
        if split not in {"train", "dev", "test"} or role not in {"train", "dev", "blind"}:
            raise ValueError(f"invalid frozen role/split for {uid}: {role}/{split}")
        if (role == "blind") != (split == "test") or (role == "train") != (split == "train"):
            raise ValueError(f"frozen role/split contract mismatch for {uid}: {role}/{split}")
        if not group:
            raise ValueError(f"source group missing for {uid}")
        group_splits[group].add(split)
        rows.append(
            {
                **item,
                "schema_version": "silver-match-v3-consensus-training-truth-v1",
                "decision": truth["decision"],
                "metric_id": truth.get("metric_id"),
                "confidence": truth["confidence"],
                "reason": teacher_reason,
                "teacher_reason_source": teacher_reason_source,
                "consensus_resolution_reason": truth["reason"],
                "label_source": truth["label_source"],
                "agreement_sources": truth.get("agreement_sources") or [],
                "source_predictions": truth.get("source_predictions") or {},
                "current_bank_source_sha256": validation["bank_source_sha256"],
                "training_eligible": split == "train",
                "dev_selection_eligible": split == "dev",
                "blind_evaluation_only": split == "test",
                "consensus_report_sha256": sha256_file(consensus_report),
                "source_pack_validation_sha256": sha256_file(validation_path),
            }
        )
    crossings = {group: values for group, values in group_splits.items() if len(values) != 1}
    if crossings:
        raise ValueError(f"source groups cross frozen splits: {list(crossings)[:3]}")

    output_root.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {"all": output_root / "truth.all.jsonl"}
    for split in ("train", "dev", "test"):
        paths[split] = output_root / f"truth.{split}.jsonl"
    write_jsonl(paths["all"], rows)
    for split in ("train", "dev", "test"):
        write_jsonl(paths[split], [row for row in rows if row["split"] == split])

    split_counts = Counter(str(row["split"]) for row in rows)
    decision_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "dev", "test"):
        decision_counts[split] = dict(
            sorted(Counter(str(row["decision"]) for row in rows if row["split"] == split).items())
        )
    manifest = {
        "schema_version": "silver-match-v3-consensus-training-truth-manifest-v1",
        "status": "COMPLETE_EXACT_CONSENSUS_WITH_FROZEN_SPLITS",
        "task": validation["task"],
        "count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "decision_counts_by_split": decision_counts,
        "source_group_cross_split_count": 0,
        "blind_rows_training_eligible": sum(
            bool(row["training_eligible"]) for row in rows if row["split"] == "test"
        ),
        "inputs": {
            "pack_validation": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
            "pack_items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            "consensus_report": {
                "path": str(consensus_report),
                "sha256": sha256_file(consensus_report),
            },
            "resolved_consensus": {
                "path": str(resolved_path),
                "sha256": sha256_file(resolved_path),
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path), "count": (
                len(rows) if name == "all" else split_counts[name]
            )}
            for name, path in paths.items()
        },
    }
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--consensus-report", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    manifest = materialize(
        Path(args.pack_root), Path(args.consensus_report), Path(args.output_root)
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
