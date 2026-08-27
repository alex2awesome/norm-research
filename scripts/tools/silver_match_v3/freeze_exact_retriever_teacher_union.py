#!/usr/bin/env python3
"""Freeze exact prior-human plus optimize truth for a task-local retriever retry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .train_nemotron_lora import source_group_key


def unique(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in output or len(output) != len(rows):
        raise ValueError(f"missing/duplicate UID: {path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--prior-report", required=True)
    parser.add_argument("--optimize", required=True)
    parser.add_argument("--optimize-freeze", required=True)
    parser.add_argument("--select", required=True)
    parser.add_argument("--select-freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name.replace("-", "_"))).resolve() for name in (
        "manifest", "prior", "prior_report", "optimize", "optimize_freeze", "select", "select_freeze"
    )}
    output = Path(args.output).resolve()
    meta = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta.exists():
        raise FileExistsError(output)
    manifest = json.loads(paths["manifest"].read_text())
    bank_meta = manifest["banks"][args.task]
    bank_hash = str(bank_meta["source_sha256"])
    bank_path = Path(bank_meta["path"])
    bank_ids = {str(row["metric_id"]) for row in json.loads(bank_path.read_text())["metrics"]}
    norms: dict[str, dict[str, Any]] = {}
    for corpus, artifact in manifest["corpora"].items():
        if artifact.get("task") != args.task:
            continue
        for row in read_jsonl(Path(artifact["path"])):
            norms[str(row["norm_uid"])] = row
    prior, optimize, select = unique(paths["prior"]), unique(paths["optimize"]), unique(paths["select"])
    prior_report = json.loads(paths["prior_report"].read_text())
    optimize_freeze = json.loads(paths["optimize_freeze"].read_text())
    select_freeze = json.loads(paths["select_freeze"].read_text())
    if prior_report.get("source_group_overlap", {}).get("teacher_external") != 0:
        raise ValueError("prior teacher/external overlap is not frozen zero")
    for freeze, role, rows in ((optimize_freeze, "optimize", optimize), (select_freeze, "select", select)):
        if (
            freeze.get("status") != "FROZEN_COMPLETE_EXACT_TRUTH"
            or freeze.get("role") != role
            or freeze.get("contracts", {}).get("unresolved_count") != 0
            or freeze.get("count") != len(rows)
            or freeze.get("bank_source_sha256") != bank_hash
        ):
            raise ValueError(f"invalid {role} truth release")
        if freeze["artifacts"]["truth"]["sha256"] != sha256_file(paths[role]):
            raise ValueError(f"{role} truth differs from release")
    needed = set(prior) | set(optimize) | set(select)
    if not needed <= set(norms):
        raise ValueError("teacher truth contains noncanonical UIDs")
    select_groups = {source_group_key(norms[uid]) for uid in select}
    chosen: dict[str, dict[str, Any]] = {}
    provenance: Counter[str] = Counter()
    def common(uid: str, row: dict[str, Any]) -> str:
        if row.get("task") != args.task or row.get("current_bank_source_sha256") != bank_hash:
            raise ValueError(f"task/bank mismatch: {uid}")
        metric = str(row.get("metric_id") or "")
        if row.get("decision") == "MATCH" and metric not in bank_ids:
            raise ValueError(f"invalid exact leaf: {uid}")
        return source_group_key(norms[uid])
    for uid, row in prior.items():
        group = common(uid, row)
        if (
            row.get("decision") != "MATCH"
            or row.get("label_source") != "independent_subagent"
            or row.get("split") != "train"
        ):
            raise ValueError(f"prior row is not frozen exact human train: {uid}")
        chosen[uid] = {**row, "source_group": group, "gradient_eligible": True,
                       "retriever_retry_teacher_provenance": "prior_strong_human_train"}
        provenance["prior_strong_human_train"] += 1
    for uid, row in optimize.items():
        group = common(uid, row)
        if (
            row.get("gepa_role") != "optimize"
            or row.get("prompt_gradient_eligible") is not True
            or row.get("prompt_selection_eligible") is not False
        ):
            raise ValueError(f"invalid optimize role: {uid}")
        if row.get("decision") != "MATCH":
            continue
        if len(row.get("agreement_sources") or []) < 2:
            raise ValueError(f"optimize MATCH lacks independent agreement: {uid}")
        if uid in chosen:
            raise ValueError(f"prior/optimize UID overlap: {uid}")
        chosen[uid] = {**row, "source_group": group, "gradient_eligible": True,
                       "training_eligible": True,
                       "retriever_retry_teacher_provenance": "new_exact_optimize_consensus"}
        provenance["new_exact_optimize_consensus"] += 1
    groups = [source_group_key(norms[uid]) for uid in chosen]
    if len(groups) != len(set(groups)):
        raise ValueError("teacher groups are not unique")
    if set(chosen) & set(select) or set(groups) & select_groups:
        raise ValueError("teachers overlap immutable select UIDs/groups")
    rows = [chosen[uid] for uid in sorted(chosen)]
    write_jsonl(output, rows)
    report = {
        "schema_version": "silver-match-v3-exact-retriever-teacher-union-v1",
        "status": "FROZEN_TRAIN_ONLY_EXACT_TEACHERS",
        "task": args.task,
        "bank_source_sha256": bank_hash,
        "counts": {"prior_input": len(prior), "optimize_input": len(optimize),
                   "select_excluded": len(select), "final_teachers": len(rows),
                   "final_source_groups": len(groups),
                   "final_metrics": len({str(row["metric_id"]) for row in rows})},
        "provenance": dict(sorted(provenance.items())),
        "audits": {"select_uid_overlap": 0, "select_source_group_overlap": 0,
                   "weak_or_forced_selected": 0, "nonmatch_selected": 0,
                   "prior_external_source_group_overlap": 0},
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "output": {"path": str(output), "sha256": sha256_file(output)}
    }
    meta.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**report, "meta_sha256": sha256_file(meta)}, sort_keys=True))


if __name__ == "__main__":
    main()
