#!/usr/bin/env python3
"""Bind a truth-hidden full-bank pack to a minimal local inference manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file


def materialize(pack_root: Path, output: Path) -> dict:
    pack_root = pack_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    validation_path = pack_root / "validation.json"
    items_path = pack_root / "items.jsonl"
    bank_path = pack_root / "bank.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("truth_hidden") is not True:
        raise ValueError("task-local inference manifest requires a truth-hidden pack")
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("pack items hash mismatch")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("pack bank hash mismatch")
    rows = list(read_jsonl(items_path))
    task = str(validation["task"])
    corpora = sorted({str(row["corpus"]) for row in rows})
    if not rows or any(str(row.get("task")) != task for row in rows):
        raise ValueError("pack items are empty or cross-task")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("task") != task or bank.get("source_sha256") != validation.get(
        "bank_source_sha256"
    ):
        raise ValueError("pack bank identity mismatch")
    payload = {
        "schema_version": "silver-match-v3-task-local-inference-manifest-v1",
        "truth_or_label_fields_in_manifest": False,
        "source_pack": {
            "validation_path": str(validation_path),
            "validation_sha256": sha256_file(validation_path),
            "truth_hidden": True,
        },
        "corpora": {
            corpus: {
                "task": task,
                "path": str(items_path),
                "sha256": sha256_file(items_path),
            }
            for corpus in corpora
        },
        "banks": {
            task: {
                "path": str(bank_path),
                "sha256": sha256_file(bank_path),
                "source_sha256": validation["bank_source_sha256"],
            }
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "task": task,
        "corpora": corpora,
        "count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(Path(args.pack_root), Path(args.output)), sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
