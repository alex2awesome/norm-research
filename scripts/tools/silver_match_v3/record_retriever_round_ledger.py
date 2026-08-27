#!/usr/bin/env python3
"""Seal dev-only retriever round claims while keeping frozen tests unread."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import sha256_file


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help=(
            "TASK:promote|reject:DEV_REPORT:FUSION_REPORT:SELECTION:TRAINING_REPORT:"
            "TEACHERS:ADAPTER_DIR"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    entries: dict[str, Any] = {}
    for spec in args.entry:
        parts = spec.split(":", 7)
        if len(parts) != 8:
            raise ValueError(f"invalid --entry: {spec!r}")
        task, decision, *raw_paths = parts
        if task in entries or decision not in {"promote", "reject"}:
            raise ValueError(f"duplicate/invalid entry: {task}:{decision}")
        dev_path, fusion_path, selection_path, training_path, teachers_path, adapter_path = map(
            lambda value: Path(value).resolve(), raw_paths
        )
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        training = json.loads(training_path.read_text(encoding="utf-8"))
        gate = dev.get("promotion_gate") or {}
        if any(value.get("task") != task for value in (dev, fusion, selection, training)):
            raise ValueError(f"task mismatch in ledger entry: {task}")
        if dev.get("split") != "dev" or selection.get("selection_split") != "external_dev_only":
            raise ValueError(f"entry did not use external dev only: {task}")
        if selection.get("frozen_test_consumed") is not False:
            raise ValueError(f"selection claims frozen-test consumption: {task}")
        chosen_kind = str((selection.get("chosen") or {}).get("kind") or "")
        if decision == "promote" and (gate.get("passed") is not True or chosen_kind != "adapter"):
            raise ValueError(f"promote entry lacks passed adapter gate: {task}")
        if decision == "reject" and (gate.get("passed") is not False or chosen_kind == "adapter"):
            raise ValueError(f"reject entry is inconsistent with dev selection: {task}")
        before, after = dev["before"]["exact"], dev["after"]["exact"]
        entries[task] = {
            "decision": (
                "PROMOTE_TASK_SPECIFIC_R4_DEV_SELECTED"
                if decision == "promote"
                else "REJECT_R4_KEEP_UNADAPTED_BASE"
            ),
            "primary_target": "exact_recall_at_50",
            "dense_external_dev": {
                "n_match_labels": dev.get("n_match_labels"),
                "base_recall_at_50": before.get("recall_at_50"),
                "adapter_recall_at_50": after.get("recall_at_50"),
                "delta_recall_at_50": dev.get("delta", {}).get("recall_at_50"),
                "base_recall_at_80": before.get("recall_at_80"),
                "adapter_recall_at_80": after.get("recall_at_80"),
                "delta_recall_at_80": dev.get("delta", {}).get("recall_at_80"),
                "promotion_gate": gate,
                "paired": dev.get("paired"),
            },
            "dev_selected_fusion": {
                "chosen_kind": chosen_kind,
                "chosen_name": (selection.get("chosen") or {}).get("name"),
                "metrics": fusion.get("metrics", {}).get("dev"),
                "component_weights": (fusion.get("selected") or {}).get(
                    "component_weights"
                ),
            },
            "artifacts": {
                "dev_report": artifact(dev_path),
                "fusion_report": artifact(fusion_path),
                "selection": artifact(selection_path),
                "training_report": artifact(training_path),
                "teachers": artifact(teachers_path),
                "adapter": {
                    "path": str(adapter_path),
                    "files": {
                        path.name: sha256_file(path)
                        for path in sorted(adapter_path.iterdir())
                        if path.is_file()
                    },
                },
            },
            "frozen_test": {
                "status": "SEALED_UNCONSUMED",
                "metrics_reported": False,
            },
        }
    manifest = Path(args.manifest).resolve()
    payload = {
        "schema_version": "silver-match-v3-retriever-round-ledger-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_basis": "external_dev_only",
        "manifest": artifact(manifest),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
