#!/usr/bin/env python3
"""Fail-closed coverage/provenance audit for production retrieval outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl, sha256_file


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else anchor.parent / value


def audit_candidates(
    *,
    manifest_path: Path,
    corpus: str,
    candidate_paths: Iterable[Path],
    expected_k: int,
) -> dict[str, Any]:
    if expected_k < 1:
        raise ValueError("expected_k must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corpus not in manifest.get("corpora", {}):
        raise KeyError(f"unknown corpus: {corpus}")
    corpus_meta = manifest["corpora"][corpus]
    task = str(corpus_meta["task"])
    bank_meta = manifest["banks"][task]
    bank_path = _resolve(bank_meta["path"], manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
    bank_ids = {str(row["metric_id"]) for row in bank}
    if len(bank_ids) != len(bank):
        raise ValueError(f"duplicate metric IDs in bank: {task}")
    required_k = min(expected_k, len(bank_ids))
    canonical_path = _resolve(corpus_meta["path"], manifest_path)
    canonical = {
        str(row["norm_uid"]): int(row["row"]) for row in read_jsonl(canonical_path)
    }
    if len(canonical) != int(corpus_meta["count"]):
        raise ValueError(f"canonical UID/count mismatch for {corpus}")

    paths = [Path(path) for path in candidate_paths]
    if not paths:
        raise ValueError("at least one candidate path is required")
    seen: set[str] = set()
    file_reports = {}
    candidate_count_distribution: Counter[int] = Counter()
    fusion_hashes: set[str] = set()
    adapters: set[str] = set()
    retrieval_signatures: set[tuple[str, str, str, bool]] = set()
    manifest_hash_in_meta: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"candidate metadata missing: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recorded_manifest_sha = str(meta.get("manifest_sha256") or "")
        if recorded_manifest_sha and recorded_manifest_sha != sha256_file(manifest_path):
            raise ValueError(f"candidate metadata manifest hash mismatch: {path}")
        manifest_hash_in_meta[
            "pinned" if recorded_manifest_sha else "legacy_audit_pinned_only"
        ] += 1
        actual_hash = sha256_file(path)
        if meta.get("output_sha256") != actual_hash:
            raise ValueError(f"candidate output hash mismatch: {path}")
        if str(meta.get("corpus")) != corpus or str(meta.get("task")) != task:
            raise ValueError(f"candidate metadata routing mismatch: {path}")
        if str(meta.get("bank_source_sha256")) != str(bank_meta["source_sha256"]):
            raise ValueError(f"candidate metadata bank hash mismatch: {path}")
        if int(meta.get("output_k", -1)) != expected_k:
            raise ValueError(f"candidate metadata output_k mismatch: {path}")
        fusion_hash = str(meta.get("fusion_weights_sha256") or "")
        if fusion_hash:
            fusion_path = Path(str(meta.get("fusion_weights")))
            if not fusion_path.exists() or sha256_file(fusion_path) != fusion_hash:
                raise ValueError(f"frozen fusion provenance mismatch: {path}")
            fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
            if fusion.get("selection_split") != "dev" or fusion.get("task") != task:
                raise ValueError(f"fusion is not task-matched dev selection: {path}")
            fusion_hashes.add(fusion_hash)
        adapter = str(meta.get("adapter") or "")
        if adapter:
            adapter_path = Path(adapter)
            recorded = meta.get("adapter_hashes") or {}
            actual = {
                value.name: sha256_file(value)
                for value in sorted(adapter_path.iterdir())
                if value.is_file()
            }
            if actual != recorded:
                raise ValueError(f"adapter hash mismatch: {path}")
            adapters.add(adapter)
        encoder = str(meta.get("encoder") or "")
        query_format = str(meta.get("query_format") or "")
        query_views = str(meta.get("query_views") or "")
        dense_query_instruction = bool(meta.get("dense_query_instruction"))
        if not encoder or not query_format or not query_views:
            raise ValueError(f"candidate retrieval signature incomplete: {path}")
        retrieval_signatures.add(
            (encoder, query_format, query_views, dense_query_instruction)
        )

        file_count = 0
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in seen:
                raise ValueError(f"missing/duplicate candidate UID: {uid!r}")
            if uid not in canonical:
                raise ValueError(f"candidate UID outside canonical corpus: {uid}")
            if (
                str(row.get("corpus")) != corpus
                or str(row.get("task")) != task
                or int(row.get("row", -1)) != canonical[uid]
            ):
                raise ValueError(f"candidate row routing mismatch: {uid}")
            if str(row.get("bank_source_sha256")) != str(bank_meta["source_sha256"]):
                raise ValueError(f"candidate row bank hash mismatch: {uid}")
            candidates = row.get("candidates") or []
            candidate_count_distribution[len(candidates)] += 1
            if len(candidates) != required_k:
                raise ValueError(
                    f"candidate depth mismatch for {uid}: {len(candidates)} != {required_k}"
                )
            ids = [str(value.get("metric_id") or "") for value in candidates]
            if len(set(ids)) != len(ids) or not set(ids) <= bank_ids:
                raise ValueError(f"duplicate/out-of-bank candidate ID for {uid}")
            ranks = [int(value.get("rank", -1)) for value in candidates]
            if ranks != list(range(1, required_k + 1)):
                raise ValueError(f"candidate ranks are not contiguous for {uid}")
            seen.add(uid)
            file_count += 1
        file_reports[str(path)] = {
            "count": file_count,
            "sha256": actual_hash,
            "meta": str(meta_path),
            "meta_sha256": sha256_file(meta_path),
        }

    missing = set(canonical) - seen
    if missing or len(seen) != int(corpus_meta["count"]):
        raise ValueError(
            f"candidate coverage mismatch for {corpus}: expected={corpus_meta['count']} "
            f"observed={len(seen)} missing={len(missing)} sample={sorted(missing)[:3]}"
        )
    return {
        "schema_version": "silver-match-v3-production-candidate-audit-v1",
        "complete": True,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "corpus": corpus,
        "task": task,
        "expected_count": int(corpus_meta["count"]),
        "observed_count": len(seen),
        "bank_count": len(bank_ids),
        "bank_source_sha256": bank_meta["source_sha256"],
        "expected_k": expected_k,
        "materialized_k": required_k,
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_count_distribution.items())
        },
        "manifest_hash_provenance": dict(sorted(manifest_hash_in_meta.items())),
        "fusion_hashes": sorted(fusion_hashes),
        "adapters": sorted(adapters),
        "retrieval_signatures": [
            {
                "encoder": encoder,
                "query_format": query_format,
                "query_views": query_views,
                "dense_query_instruction": dense_query_instruction,
            }
            for encoder, query_format, query_views, dense_query_instruction in sorted(
                retrieval_signatures
            )
        ],
        "candidate_inputs": file_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--expected-k", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit_candidates(
        manifest_path=Path(args.manifest).resolve(),
        corpus=args.corpus,
        candidate_paths=[Path(path).resolve() for path in args.candidates],
        expected_k=args.expected_k,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
