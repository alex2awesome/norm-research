#!/usr/bin/env python3
"""Materialize a task-local manifest from disjoint frozen truth-hidden packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _parse_pack(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--pack must be NAME=ROOT")
    name, root = raw.split("=", 1)
    if not name:
        raise ValueError("empty pack name")
    return name, Path(root).resolve()


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    packs = [_parse_pack(raw) for raw in args.pack]
    if len(packs) < 2 or len({name for name, _ in packs}) != len(packs):
        raise ValueError("at least two uniquely named packs are required")

    corpora: dict[str, dict[str, Any]] = {}
    pack_records = []
    all_rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_groups: set[str] = set()
    canonical_bank: dict[str, Any] | None = None
    canonical_bank_path: Path | None = None
    bank_source_sha: str | None = None
    for name, root in packs:
        validation_path = root / "validation.json"
        items_path = root / "items.jsonl"
        bank_path = root / "bank.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        expected_items_sha = (((validation.get("outputs") or {}).get("items") or {}).get("sha256"))
        expected_bank_sha = (((validation.get("outputs") or {}).get("bank") or {}).get("sha256"))
        if (
            validation.get("truth_hidden") is not True
            or validation.get("task") != args.task
            or sha256_file(items_path) != expected_items_sha
            or sha256_file(bank_path) != expected_bank_sha
        ):
            raise ValueError(f"pack validation/hash mismatch: {root}")
        rows = list(read_jsonl(items_path))
        uids = {str(row.get("norm_uid") or "") for row in rows}
        groups = {str(row.get("source_group") or "") for row in rows}
        if (
            not rows
            or "" in uids
            or "" in groups
            or len(uids) != len(rows)
            or any(row.get("task") != args.task for row in rows)
        ):
            raise ValueError(f"invalid task pack identities: {root}")
        if uids & seen_uids or groups & seen_groups:
            raise ValueError(f"pack identities/source groups overlap: {name}")
        seen_uids.update(uids)
        seen_groups.update(groups)
        all_rows.extend(rows)

        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        current_source_sha = str(bank.get("source_sha256") or "")
        normalized_bank = {
            str(row["metric_id"]): row for row in bank.get("metrics") or []
        }
        if not current_source_sha or len(normalized_bank) != len(bank.get("metrics") or []):
            raise ValueError(f"invalid bank: {bank_path}")
        if canonical_bank is None:
            canonical_bank = normalized_bank
            canonical_bank_path = bank_path
            bank_source_sha = current_source_sha
        elif normalized_bank != canonical_bank or current_source_sha != bank_source_sha:
            raise ValueError(f"pack bank semantics differ: {bank_path}")

        corpora[name] = {
            "task": args.task,
            "path": str(items_path),
            "count": len(rows),
            "sha256": sha256_file(items_path),
            "source_pack_validation_path": str(validation_path),
            "source_pack_validation_sha256": sha256_file(validation_path),
        }
        pack_records.append(
            {
                "name": name,
                "root": str(root),
                "validation_sha256": sha256_file(validation_path),
                "items_sha256": sha256_file(items_path),
                "bank_sha256": sha256_file(bank_path),
                "count": len(rows),
            }
        )

    assert canonical_bank is not None and canonical_bank_path is not None and bank_source_sha
    merged_norms_raw = getattr(args, "merged_norms_output", None)
    if merged_norms_raw:
        merged_norms = Path(merged_norms_raw).resolve()
        if merged_norms.exists():
            raise FileExistsError(merged_norms)
        corpus_values = {str(row.get("corpus") or "") for row in all_rows}
        if len(corpus_values) != 1 or "" in corpus_values:
            raise ValueError(f"merged role packs do not share one corpus: {corpus_values}")
        corpus = next(iter(corpus_values))
        write_jsonl(merged_norms, sorted(all_rows, key=lambda row: str(row["norm_uid"])))
        corpora = {
            corpus: {
                "task": args.task,
                "path": str(merged_norms),
                "count": len(all_rows),
                "sha256": sha256_file(merged_norms),
                "source_role_pack_count": len(packs),
            }
        }

    payload = {
        "schema_version": "silver-match-v3-task-local-role-pack-manifest-v1",
        "status": "FROZEN_TRUTH_HIDDEN_ROLE_PACK_PATH_BINDING",
        "task": args.task,
        "truth_or_label_fields_in_manifest": False,
        "banks": {
            args.task: {
                "path": str(canonical_bank_path),
                "count": len(canonical_bank),
                "sha256": sha256_file(canonical_bank_path),
                "source_sha256": bank_source_sha,
            }
        },
        "corpora": corpora,
        "pack_count": len(packs),
        "total_norms": len(seen_uids),
        "unique_source_groups": len(seen_groups),
        "cross_pack_uid_overlap": 0,
        "cross_pack_source_group_overlap": 0,
        "packs": pack_records,
        "merged_norms": (
            {"path": str(merged_norms), "sha256": sha256_file(merged_norms)}
            if merged_norms_raw
            else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "task": args.task,
        "total_norms": len(seen_uids),
        "pack_count": len(packs),
        "bank_source_sha256": bank_source_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--pack", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--merged-norms-output",
        help=(
            "Optionally merge all disjoint pack items into one corpus artifact so "
            "trainers that enforce manifest-key/corpus equality can consume it."
        ),
    )
    args = parser.parse_args()
    print(json.dumps(materialize(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
