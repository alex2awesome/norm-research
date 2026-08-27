#!/usr/bin/env python3
"""Bridge exact optimize-role truth to CE-only gradient records.

The source truth remains immutable and ineligible for retriever gradients.
This bridge is allowed only because the frozen CE policy predeclares optimize
truth as adjudicator-model training data.  Select, test, blind, and evaluation
truth fail closed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .adjudicate_gemma import DECISIONS
from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_group_for


def bridge(
    *,
    manifest_path: Path,
    task: str,
    truth_path: Path,
    policy_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    truth_path = truth_path.resolve()
    policy_path = policy_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if task not in manifest.get("banks", {}) or task not in policy.get("scope", []):
        raise ValueError("task is outside manifest or CE policy scope")
    role_contract = policy.get("role_contract") or {}
    if "optimize" not in str(role_contract.get("train") or ""):
        raise ValueError("CE policy did not predeclare optimize truth for training")
    bank_meta = manifest["banks"][task]
    bank = json.loads(Path(bank_meta["path"]).read_text(encoding="utf-8"))["metrics"]
    bank_ids = {str(row["metric_id"]) for row in bank}
    truth = list(read_jsonl(truth_path))
    if not truth:
        raise ValueError("empty optimize truth")
    uids = [str(row.get("norm_uid") or "") for row in truth]
    if "" in uids or len(uids) != len(set(uids)):
        raise ValueError("optimize truth has missing/duplicate UIDs")
    norm_by_uid: dict[str, dict[str, Any]] = {}
    needed = set(uids)
    for corpus, meta in manifest["corpora"].items():
        if meta["task"] != task:
            continue
        for row in read_jsonl(Path(meta["path"])):
            uid = str(row["norm_uid"])
            if uid in needed:
                norm_by_uid[uid] = row
    if set(norm_by_uid) != needed:
        raise ValueError("optimize truth contains UIDs outside task manifest")
    bridged = []
    decision_counts: Counter[str] = Counter()
    for row in truth:
        uid = str(row["norm_uid"])
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        if (
            row.get("task") != task
            or row.get("gepa_role") != "optimize"
            or row.get("prompt_gradient_eligible") is not True
            or row.get("evaluation_only") is not False
            or row.get("split") != "train"
            or row.get("current_bank_source_sha256") != bank_meta["source_sha256"]
            or decision not in DECISIONS
        ):
            raise ValueError(f"truth row is not eligible optimize evidence: {uid}")
        if decision == "MATCH":
            if str(metric_id) not in bank_ids:
                raise ValueError(f"truth MATCH is outside current bank: {uid}")
        elif metric_id is not None:
            raise ValueError(f"truth abstention carries a metric ID: {uid}")
        canonical_group = split_group_for(norm_by_uid[uid])
        if str(row.get("source_group")) != canonical_group:
            raise ValueError(f"truth source-group provenance mismatch: {uid}")
        decision_counts[decision] += 1
        bridged.append(
            {
                **row,
                "schema_version": "silver-match-v3-ce-optimize-truth-bridge-v1",
                "label_source": "exact_multi_pass_optimize_truth_ce_bridge",
                "ce_training_eligible": True,
                "retriever_training_eligible": False,
                "ce_policy_path": str(policy_path),
                "ce_policy_sha256": sha256_file(policy_path),
                "source_truth_path": str(truth_path),
                "source_truth_sha256": sha256_file(truth_path),
            }
        )
    report = {
        "schema_version": "silver-match-v3-ce-optimize-truth-bridge-report-v1",
        "status": "COMPLETE",
        "task": task,
        "count": len(bridged),
        "unique_uids": len(bridged),
        "unique_source_groups": len({row["source_group"] for row in bridged}),
        "decision_counts": dict(sorted(decision_counts.items())),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "bank_source_sha256": bank_meta["source_sha256"],
        "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        "source_truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
        "retriever_training_authorized": False,
        "ce_training_authorized": True,
    }
    return bridged, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite CE truth bridge")
    rows, report = bridge(
        manifest_path=Path(args.manifest),
        task=args.task,
        truth_path=Path(args.truth),
        policy_path=Path(args.policy),
    )
    write_jsonl(output, rows)
    report["output"] = {"path": str(output), "sha256": sha256_file(output)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
