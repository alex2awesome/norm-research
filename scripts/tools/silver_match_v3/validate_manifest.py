#!/usr/bin/env python3
"""Independently validate and hash-lock a frozen silver-match manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import normalize_space, read_jsonl, sha256_file
from .config import DEFAULT_OUTPUT_ROOT


UID_RE = re.compile(r"^[0-9a-f]{64}$")


def resolve(path: str, manifest_path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else manifest_path.parent / value


def validate(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_lock, norm_lock = {}, {}
    bank_counts = {}
    bank_ids: dict[str, set[str]] = {}
    for task, meta in sorted(manifest["banks"].items()):
        path = resolve(meta["path"], manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task") != task:
            raise ValueError(f"bank task mismatch: {path}")
        metrics = payload.get("metrics") or []
        ids = [str(metric.get("metric_id") or "") for metric in metrics]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"empty/duplicate metric IDs: {path}")
        if len(metrics) != int(meta["count"]):
            raise ValueError(f"bank count mismatch: {path}")
        bank_counts[task] = len(metrics)
        bank_ids[task] = set(ids)
        bank_lock[task] = {"path": str(path), "sha256": sha256_file(path), "count": len(metrics)}

    seen_uids: set[str] = set()
    task_counts: Counter[str] = Counter()
    corpus_report = {}
    for corpus, meta in sorted(manifest["corpora"].items()):
        if not meta.get("coverage_complete"):
            raise ValueError(f"canonical corpus is incomplete: {corpus}")
        task = meta["task"]
        path = resolve(meta["path"], manifest_path)
        count = context = paper = explicit_judged = uid_fallback = 0
        source_groups = set()
        polarities: Counter[str] = Counter()
        kinds: Counter[str] = Counter()
        for row in read_jsonl(path):
            count += 1
            uid = str(row.get("norm_uid") or "")
            if not UID_RE.fullmatch(uid):
                raise ValueError(f"malformed norm_uid in {path}: {uid!r}")
            if uid in seen_uids:
                raise ValueError(f"duplicate global norm_uid: {uid}")
            seen_uids.add(uid)
            if row.get("corpus") != corpus or row.get("task") != task:
                raise ValueError(f"routing mismatch for {uid}")
            if not normalize_space(row.get("norm")):
                raise ValueError(f"empty norm for {uid}")
            context += bool(normalize_space(row.get("context")))
            paper += bool(normalize_space(row.get("paper_id")))
            explicit_judged += bool(row.get("extraction_faithful")) and bool(
                row.get("extraction_valid")
            )
            source_id = normalize_space(row.get("source_id"))
            group = normalize_space(row.get("paper_id")) or source_id
            if group:
                source_groups.add(group)
            else:
                uid_fallback += 1
            polarities[normalize_space(row.get("polarity")) or "MISSING"] += 1
            kinds[normalize_space(row.get("kind")) or "MISSING"] += 1
        if count != int(meta["count"]):
            raise ValueError(f"corpus count mismatch: {corpus}: {count} != {meta['count']}")
        task_counts[task] += count
        norm_lock[corpus] = {"path": str(path), "sha256": sha256_file(path), "count": count}
        corpus_report[corpus] = {
            "task": task,
            "count": count,
            "source_groups": len(source_groups),
            "source_group_uid_fallback": uid_fallback,
            "context_count": context,
            "context_rate": context / count if count else None,
            "paper_id_count": paper,
            "extraction_judged_count": explicit_judged,
            "polarities": dict(sorted(polarities.items())),
            "kinds": dict(sorted(kinds.items())),
        }
    if len(seen_uids) != int(manifest["total_norms"]):
        raise ValueError("manifest total_norms mismatch")
    if len(corpus_report) != int(manifest["total_corpora"]):
        raise ValueError("manifest total_corpora mismatch")
    if len(bank_counts) != int(manifest["total_tasks"]):
        raise ValueError("manifest total_tasks mismatch")
    for alias, target in (manifest.get("aliases") or {}).items():
        if alias in manifest["corpora"]:
            raise ValueError(f"alias duplicates canonical corpus: {alias}")
        if target not in manifest["corpora"]:
            raise ValueError(f"alias target missing: {alias} -> {target}")

    report = {
        "status": "VALID",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "total_norms": len(seen_uids),
        "total_corpora": len(corpus_report),
        "total_tasks": len(bank_counts),
        "task_norm_counts": dict(sorted(task_counts.items())),
        "bank_counts": bank_counts,
        "corpora": corpus_report,
    }
    lock = {
        "schema_version": manifest["schema_version"],
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "banks": bank_lock,
        "norms": norm_lock,
    }
    return report, lock


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--report")
    parser.add_argument("--lock")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    report, lock = validate(manifest_path)
    root = manifest_path.parent
    atomic_json(Path(args.report) if args.report else root / "validation_report.json", report)
    atomic_json(Path(args.lock) if args.lock else root / "artifact_lock.json", lock)
    print(json.dumps({k: report[k] for k in ("status", "total_norms", "total_corpora", "total_tasks")}), flush=True)


if __name__ == "__main__":
    main()
