#!/usr/bin/env python3
"""Build truth-blind full-bank labeling packs for unresolved rescue decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


def _resolve(raw: str, anchor: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else anchor.parent / path


def _key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()


def prepare(
    *,
    manifest_path: Path,
    unresolved_path: Path,
    output_root: Path,
    chunk_size: int,
    seed: int,
) -> dict[str, Any]:
    if chunk_size < 1 or chunk_size > 25:
        raise ValueError("chunk_size must be in [1, 25]")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unresolved = list(read_jsonl(unresolved_path))
    by_uid = {str(row.get("norm_uid") or ""): row for row in unresolved}
    if "" in by_uid or len(by_uid) != len(unresolved):
        raise ValueError("unresolved input has missing/duplicate UIDs")
    needed_by_corpus: dict[str, set[str]] = defaultdict(set)
    for uid, row in by_uid.items():
        corpus = str(row.get("corpus") or "")
        task = str(row.get("task") or "")
        if corpus not in manifest["corpora"] or task != manifest["corpora"][corpus]["task"]:
            raise ValueError(f"unresolved routing mismatch: {uid}")
        needed_by_corpus[corpus].add(uid)
    canonical = {}
    for corpus, needed in needed_by_corpus.items():
        path = _resolve(manifest["corpora"][corpus]["path"], manifest_path)
        for row in read_jsonl(path):
            uid = str(row["norm_uid"])
            if uid in needed:
                canonical[uid] = row
    missing = set(by_uid) - set(canonical)
    if missing:
        raise ValueError(f"unresolved UIDs absent from canonical data: {sorted(missing)[:3]}")

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for task in sorted({str(row["task"]) for row in unresolved}):
        task_root = output_root / task
        task_root.mkdir(parents=True)
        task_rows = [row for row in unresolved if row["task"] == task]
        task_rows.sort(
            key=lambda row: (
                _key(seed, f"item:{task}", str(row["norm_uid"])),
                str(row["norm_uid"]),
            )
        )
        blind = []
        key_rows = []
        for unresolved_row in task_rows:
            uid = str(unresolved_row["norm_uid"])
            source = canonical[uid]
            blind.append(
                {
                    "norm_uid": uid,
                    "corpus": source["corpus"],
                    "task": source["task"],
                    "row": source["row"],
                    "human_statement": source.get("norm"),
                    "evidence_passage": source.get("context"),
                    "kind": source.get("kind"),
                    "polarity": source.get("polarity"),
                    "instruction": (
                        "Return one exact current-bank MATCH or a typed abstention; "
                        "do not force a sibling or thematic neighbor."
                    ),
                }
            )
            key_rows.append(
                {
                    "norm_uid": uid,
                    "source": unresolved_row.get("source"),
                    "unresolved_reason": unresolved_row.get("unresolved_reason"),
                }
            )
        bank_path = _resolve(manifest["banks"][task]["path"], manifest_path)
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        metrics = list(bank["metrics"])
        metrics.sort(
            key=lambda row: (
                _key(seed, f"bank:{task}", str(row["metric_id"])),
                str(row["metric_id"]),
            )
        )
        blind_path = task_root / "items.blind.jsonl"
        key_path = task_root / "items.key.jsonl"
        rendered_bank_path = task_root / "bank.blind.json"
        write_jsonl(blind_path, blind)
        write_jsonl(key_path, key_rows)
        rendered_bank_path.write_text(
            json.dumps(
                {
                    "task": task,
                    "source_sha256": manifest["banks"][task]["source_sha256"],
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        # Standard independent-label pack.  It intentionally excludes the
        # sibling items.key.jsonl so the labeling workspace can be copied on
        # its own without exposing system reasons.
        label_pack = task_root / "label_pack"
        label_pack.mkdir(parents=True)
        label_items = []
        for row in task_rows:
            uid = str(row["norm_uid"])
            source = canonical[uid]
            group = split_group_for(source)
            label_items.append(
                {
                    "schema_version": source.get("schema_version")
                    or manifest.get("schema_version")
                    or "silver-match-v3.0",
                    "norm_uid": uid,
                    "corpus": source["corpus"],
                    "task": source["task"],
                    "row": source["row"],
                    "norm": source.get("norm"),
                    "context": source.get("context"),
                    "aspect": source.get("aspect"),
                    "kind": source.get("kind"),
                    "polarity": source.get("polarity"),
                    "split_group": group,
                    "source_group": group,
                    "split": split_for(group),
                    "boundary_stratum": "strict_rescue_unresolved",
                    "permanently_excluded_from_gradients": True,
                }
            )
        label_items_path = label_pack / "items.jsonl"
        label_bank_path = label_pack / "bank.json"
        write_jsonl(label_items_path, label_items)
        label_bank_path.write_text(
            json.dumps(
                {
                    "schema_version": bank.get("schema_version")
                    or manifest.get("schema_version")
                    or "silver-match-v3.0",
                    "task": task,
                    "source_sha256": manifest["banks"][task]["source_sha256"],
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        label_chunks = {}
        for start in range(0, len(label_items), chunk_size):
            chunk = label_pack / "chunks" / f"part-{start // chunk_size:03d}.jsonl"
            write_jsonl(chunk, label_items[start : start + chunk_size])
            label_chunks[str(chunk)] = sha256_file(chunk)
        label_validation = {
            "schema_version": "silver-match-v3-unresolved-label-pack-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "count": len(label_items),
            "unique_source_groups": len(
                {str(row["source_group"]) for row in label_items}
            ),
            "chunk_size": chunk_size,
            "chunk_count": len(label_chunks),
            "bank_metric_count": len(metrics),
            "bank_source_sha256": manifest["banks"][task]["source_sha256"],
            "truth_hidden": True,
            "system_reasons_hidden": True,
            "system_key_excluded_from_label_pack": True,
            "permanently_excluded_from_gradients": True,
            "inputs": {
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "unresolved": {
                    "path": str(unresolved_path),
                    "sha256": sha256_file(unresolved_path),
                },
            },
            "outputs": {
                "items": {
                    "path": str(label_items_path),
                    "sha256": sha256_file(label_items_path),
                },
                "bank": {
                    "path": str(label_bank_path),
                    "sha256": sha256_file(label_bank_path),
                },
                "chunks": label_chunks,
            },
        }
        label_validation_path = label_pack / "validation.json"
        label_validation_path.write_text(
            json.dumps(label_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        chunks = {}
        for start in range(0, len(blind), chunk_size):
            chunk = task_root / "chunks" / f"part-{start // chunk_size:03d}.jsonl"
            write_jsonl(chunk, blind[start : start + chunk_size])
            chunks[str(chunk)] = {
                "count": min(chunk_size, len(blind) - start),
                "sha256": sha256_file(chunk),
            }
        outputs[task] = {
            "count": len(blind),
            "reason_counts": dict(
                sorted(Counter(row.get("unresolved_reason") for row in task_rows).items())
            ),
            "bank_source_sha256": manifest["banks"][task]["source_sha256"],
            "items": {"path": str(blind_path), "sha256": sha256_file(blind_path)},
            "key": {"path": str(key_path), "sha256": sha256_file(key_path)},
            "bank": {
                "path": str(rendered_bank_path),
                "sha256": sha256_file(rendered_bank_path),
            },
            "chunks": chunks,
            "label_pack_validation": {
                "path": str(label_validation_path),
                "sha256": sha256_file(label_validation_path),
            },
        }
    report = {
        "schema_version": "silver-match-v3-unresolved-blind-pack-v1",
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "unresolved": {
            "path": str(unresolved_path),
            "sha256": sha256_file(unresolved_path),
        },
        "count": len(unresolved),
        "chunk_size": chunk_size,
        "seed": seed,
        "system_reasons_hidden_from_items": True,
        "outputs": outputs,
    }
    report_path = output_root / "validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--unresolved", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=161803)
    args = parser.parse_args()
    report = prepare(
        manifest_path=Path(args.manifest).resolve(),
        unresolved_path=Path(args.unresolved).resolve(),
        output_root=Path(args.output_root).resolve(),
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
