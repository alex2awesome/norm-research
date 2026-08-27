#!/usr/bin/env python3
"""Hydrate a frozen human panel into a truth-hidden labeler anchor pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (anchor.parent / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    manifest_path, truth_path = Path(args.manifest).resolve(), Path(args.truth).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite anchor pack: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth = [row for row in read_jsonl(truth_path) if row.get("task") == args.task]
    if not truth:
        raise ValueError("anchor truth has no rows for task")
    truth_uids = [str(row["norm_uid"]) for row in truth]
    if len(truth_uids) != len(set(truth_uids)):
        raise ValueError("anchor truth contains duplicate UIDs")
    canonical = {}
    target = set(truth_uids)
    for meta in manifest["corpora"].values():
        if meta.get("task") != args.task:
            continue
        for row in read_jsonl(resolve(meta["path"], manifest_path)):
            uid = str(row["norm_uid"])
            if uid in target:
                canonical[uid] = row
    missing = sorted(target - set(canonical))
    if missing:
        raise KeyError(f"anchor UIDs absent from manifest: {missing[:3]}")
    items = [canonical[uid] for uid in sorted(truth_uids)]
    bank_meta = manifest["banks"][args.task]
    bank_path = resolve(bank_meta["path"], manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("source_sha256") != bank_meta.get("source_sha256"):
        raise ValueError("anchor bank identity mismatch")
    output.mkdir(parents=True, exist_ok=True)
    items_path, bank_output = output / "items.jsonl", output / "bank.json"
    write_jsonl(items_path, items)
    bank_output.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n")
    chunks = []
    for start in range(0, len(items), args.chunk_size):
        path = output / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, items[start : start + args.chunk_size])
        chunks.append(path)
    report = {
        "schema_version": "silver-match-v3-labeler-anchor-pack-v1",
        "task": args.task,
        "count": len(items),
        "chunk_size": args.chunk_size,
        "chunk_counts": {
            path.stem: sum(1 for _ in read_jsonl(path)) for path in chunks
        },
        "truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
        "manifest_sha256": sha256_file(manifest_path),
        "bank_source_sha256": str(bank_meta["source_sha256"]),
        "outputs": {
            "items": sha256_file(items_path),
            "bank": sha256_file(bank_output),
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
    }
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
