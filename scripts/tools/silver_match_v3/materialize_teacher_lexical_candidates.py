#!/usr/bin/env python3
"""Materialize full-bank lexical rankings for frozen teacher/dev UID panels.

Held-out optimize/select identities are intentionally absent from production
retrieval outputs.  Cross-encoder training already falls back to the same
word+character TF-IDF ranking when a teacher UID lacks candidates; this command
makes that fallback explicit, complete, hash-bound, and auditable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import metric_card, norm_query, read_jsonl, sha256_file, write_jsonl
from .retrieve import build_vectorizers, top_indices
from .train_nemotron_lora import source_group_key


def _resolve(value: str, anchor: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--uid-source", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    source_paths = [Path(value).resolve() for value in args.uid_source]
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_meta = (manifest.get("banks") or {}).get(args.task)
    if not isinstance(bank_meta, dict):
        raise KeyError(f"task absent from manifest: {args.task}")
    bank_path = _resolve(str(bank_meta["path"]), manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_hash = str(bank_meta["source_sha256"])
    metrics = list(bank.get("metrics") or [])
    metric_ids = [str(row.get("metric_id") or "") for row in metrics]
    if (
        bank.get("task") != args.task
        or bank.get("source_sha256") != bank_hash
        or not metric_ids
        or "" in metric_ids
        or len(metric_ids) != len(set(metric_ids))
    ):
        raise ValueError("invalid current task bank")

    target: set[str] = set()
    source_counts: Counter[str] = Counter()
    duplicate_references = 0
    for path in source_paths:
        seen_here: set[str] = set()
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in seen_here:
                raise ValueError(f"missing/duplicate UID within source: {path}/{uid!r}")
            row_task = str(row.get("task") or "")
            if row_task and row_task != args.task:
                raise ValueError(f"task mismatch: {path}/{uid}")
            row_bank = str(
                row.get("current_bank_source_sha256")
                or row.get("bank_source_sha256")
                or ""
            )
            if row_bank and row_bank != bank_hash:
                raise ValueError(f"stale bank reference: {path}/{uid}")
            seen_here.add(uid)
            duplicate_references += uid in target
            target.add(uid)
        source_counts[str(path)] = len(seen_here)
    if not target:
        raise ValueError("UID sources contain no rows")

    norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != args.task:
            continue
        for row in read_jsonl(_resolve(str(meta["path"]), manifest_path)):
            uid = str(row.get("norm_uid") or "")
            if uid not in target:
                continue
            if uid in norms:
                raise ValueError(f"target UID duplicated across canonical corpora: {uid}")
            if row.get("task") != args.task or row.get("corpus") != corpus:
                raise ValueError(f"canonical task/corpus mismatch: {uid}")
            norms[uid] = row
    missing = sorted(target - set(norms))
    if missing:
        raise ValueError(f"UID source rows absent from manifest: {missing[:3]}")

    ordered_uids = sorted(target)
    cards = [metric_card(metric) for metric in metrics]
    queries = [norm_query(norms[uid]) for uid in ordered_uids]
    word, char, card_word, card_char = build_vectorizers(cards)
    scores = (word.transform(queries) @ card_word.T).toarray()
    scores += (char.transform(queries) @ card_char.T).toarray()
    rankings = top_indices(scores, len(metrics))
    rows = []
    for row_index, uid in enumerate(ordered_uids):
        norm = norms[uid]
        candidates = [
            {
                "metric_id": metric_ids[int(metric_index)],
                "rank": rank,
                "lexical_score": float(scores[row_index, int(metric_index)]),
            }
            for rank, metric_index in enumerate(rankings[row_index], 1)
        ]
        rows.append(
            {
                "schema_version": "silver-match-v3-ce-full-bank-lexical-candidates-v1",
                "task": args.task,
                "corpus": norm["corpus"],
                "row": norm.get("row"),
                "norm_uid": uid,
                "source_group": source_group_key(norm),
                "bank_source_sha256": bank_hash,
                "retrieval_system": "frozen_word_plus_char_tfidf_full_bank",
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )

    write_jsonl(output_path, rows)
    meta = {
        "schema_version": "silver-match-v3-ce-full-bank-lexical-candidates-report-v1",
        "status": "COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "count": len(rows),
        "unique_uids": len(rows),
        "unique_source_groups": len({row["source_group"] for row in rows}),
        "candidate_count_per_uid": len(metrics),
        "full_bank_coverage": True,
        "bank_source_sha256": bank_hash,
        "duplicate_uid_references_across_sources": duplicate_references,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            "uid_sources": {
                str(path): {"sha256": sha256_file(path), "count": source_counts[str(path)]}
                for path in source_paths
            },
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**meta, "meta_sha256": sha256_file(meta_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
