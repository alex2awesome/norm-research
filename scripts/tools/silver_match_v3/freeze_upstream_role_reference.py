#!/usr/bin/env python3
"""Freeze authoritative retriever split roles for a canonical K50 universe.

The retriever/LoRA split is defined by ``source_group_key`` plus the seed and
percentages in its immutable run config.  It is not interchangeable with the
older calibration hash split.  This command projects that role onto every UID
covered by a frozen candidate artifact and verifies all overlapping audited
teacher roles exactly before emitting an identity-only reference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_group_for
from .train_nemotron_lora import source_group_key, split_source_group


def _resolve(path: str, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--audit-reference")
    parser.add_argument("--minimum-k", type=int, default=50)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.minimum_k < 1:
        parser.error("--minimum-k must be positive")

    manifest_path = Path(args.manifest).resolve()
    candidates_path = Path(args.candidates).resolve()
    run_config_path = Path(args.run_config).resolve()
    audit_path = Path(args.audit_reference).resolve() if args.audit_reference else None
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if run_config.get("task") != args.task:
        raise ValueError("run config task mismatch")
    configured_manifest = Path(str(run_config.get("manifest") or "")).resolve()
    if configured_manifest != manifest_path:
        raise ValueError("run config points at a different manifest")
    split_seed = int(run_config["split_seed"])
    train_percent = int(run_config["train_percent"])
    dev_percent = int(run_config["dev_percent"])
    test_percent = int(run_config["test_percent"])
    if train_percent + dev_percent + test_percent != 100:
        raise ValueError("run config split percentages do not sum to 100")

    bank_meta = (manifest.get("banks") or {}).get(args.task) or {}
    bank_hash = str(bank_meta.get("source_sha256") or "")
    if not bank_hash:
        raise ValueError("manifest lacks task bank source hash")
    candidate_uids: set[str] = set()
    candidate_rows = 0
    for row in read_jsonl(candidates_path):
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in candidate_uids:
            raise ValueError(f"candidate artifact has missing/duplicate UID: {uid!r}")
        if row.get("task") != args.task:
            raise ValueError(f"candidate task mismatch: {uid}")
        if str(row.get("bank_source_sha256") or "") != bank_hash:
            raise ValueError(f"candidate bank identity mismatch: {uid}")
        candidates = list(row.get("candidates") or [])
        if len(candidates) < args.minimum_k:
            raise ValueError(f"candidate row shorter than K={args.minimum_k}: {uid}")
        metric_ids = [str(value.get("metric_id") or "") for value in candidates]
        if "" in metric_ids or len(metric_ids) != len(set(metric_ids)):
            raise ValueError(f"candidate row has missing/duplicate metric IDs: {uid}")
        candidate_uids.add(uid)
        candidate_rows += 1
    if not candidate_uids:
        raise ValueError("empty candidate artifact")

    canonical: dict[str, dict[str, Any]] = {}
    task_corpora: set[str] = set()
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != args.task:
            continue
        task_corpora.add(str(corpus))
        for row in read_jsonl(_resolve(str(meta["path"]), manifest_path)):
            uid = str(row.get("norm_uid") or "")
            if uid not in candidate_uids:
                continue
            if uid in canonical:
                raise ValueError(f"duplicate canonical candidate UID: {uid}")
            if row.get("task") != args.task or row.get("corpus") != corpus:
                raise ValueError(f"canonical task/corpus mismatch: {uid}")
            canonical[uid] = row
    missing = sorted(candidate_uids - set(canonical))
    if missing:
        raise ValueError(f"candidate UIDs absent from canonical manifest: {missing[:3]}")

    rows: list[dict[str, Any]] = []
    roles: dict[str, str] = {}
    for uid in sorted(candidate_uids):
        norm = canonical[uid]
        retriever_group = source_group_key(norm)
        role = split_source_group(
            retriever_group,
            split_seed,
            train_percent,
            dev_percent,
        )
        roles[uid] = role
        rows.append(
            {
                "schema_version": "silver-match-v3-upstream-role-reference-v1",
                "norm_uid": uid,
                "task": args.task,
                "corpus": str(norm["corpus"]),
                "source_group": split_group_for(norm),
                "retriever_source_group": retriever_group,
                "split": role,
                "split_seed": split_seed,
                "train_percent": train_percent,
                "dev_percent": dev_percent,
                "test_percent": test_percent,
            }
        )

    audit_matches = audit_mismatches = audit_covered = 0
    if audit_path is not None:
        seen_audit: set[str] = set()
        for row in read_jsonl(audit_path):
            if row.get("task") != args.task:
                raise ValueError("audit reference contains another task")
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in seen_audit:
                raise ValueError(f"audit reference has missing/duplicate UID: {uid!r}")
            seen_audit.add(uid)
            if uid not in roles:
                continue
            role = str(row.get("split") or "")
            if role not in {"train", "dev", "test"}:
                raise ValueError(f"audit reference has invalid role: {uid}/{role!r}")
            audit_covered += 1
            if roles[uid] == role:
                audit_matches += 1
            else:
                audit_mismatches += 1
        if audit_covered == 0:
            raise ValueError("audit reference has no overlap with candidate universe")
        if audit_mismatches:
            raise ValueError(
                f"derived roles disagree with {audit_mismatches}/{audit_covered} audited rows"
            )

    output_root.mkdir(parents=True, exist_ok=False)
    roles_path = output_root / "roles.jsonl"
    write_jsonl(roles_path, rows)
    role_groups: dict[str, set[str]] = {role: set() for role in ("train", "dev", "test")}
    for row in rows:
        role_groups[str(row["split"])].add(str(row["source_group"]))
    report = {
        "schema_version": "silver-match-v3-upstream-role-reference-freeze-v1",
        "status": "FROZEN_AND_AUDIT_VERIFIED",
        "task": args.task,
        "candidate_rows": candidate_rows,
        "minimum_k": args.minimum_k,
        "bank_source_sha256": bank_hash,
        "roles": dict(sorted(Counter(roles.values()).items())),
        "role_source_groups": {
            role: len(groups) for role, groups in sorted(role_groups.items())
        },
        "task_corpora": sorted(task_corpora),
        "split_policy": {
            "function": "train_nemotron_lora.split_source_group",
            "group_function": "train_nemotron_lora.source_group_key",
            "split_seed": split_seed,
            "train_percent": train_percent,
            "dev_percent": dev_percent,
            "test_percent": test_percent,
        },
        "audit_verification": {
            "path": str(audit_path) if audit_path is not None else None,
            "sha256": sha256_file(audit_path) if audit_path is not None else None,
            "overlap": audit_covered,
            "exact_role_matches": audit_matches,
            "mismatches": audit_mismatches,
        },
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "candidates": {
                "path": str(candidates_path),
                "sha256": sha256_file(candidates_path),
            },
            "run_config": {
                "path": str(run_config_path),
                "sha256": sha256_file(run_config_path),
            },
        },
        "output": {"path": str(roles_path), "sha256": sha256_file(roles_path)},
        "content_contract": {
            "candidate_metric_ids_used_only_to_verify_k_and_uniqueness": True,
            "candidate_scores_used": False,
            "teacher_decisions_metric_ids_reasons_used": False,
            "norm_text_predictions_outcomes_used": False,
        },
    }
    report_path = output_root / "FREEZE.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "freeze_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
