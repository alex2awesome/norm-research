#!/usr/bin/env python3
"""Project an audited retrieval artifact to a shallower candidate depth.

The projection is deliberately mechanical: preserve row order and every row-level
field, retain the first K candidates, and pin the complete source artifact and
metadata by hash.  This lets an exact full-bank retrieval serve as the source of
the production K50 without a second encoder run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl, sha256_file, write_jsonl


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _validate_candidates(
    rows: Iterable[dict[str, Any]], *, expected_depth: int
) -> Iterable[dict[str, Any]]:
    seen_uids: set[str] = set()
    for row in rows:
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in seen_uids:
            raise ValueError(f"missing/duplicate source norm_uid: {uid!r}")
        seen_uids.add(uid)
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != expected_depth:
            raise ValueError(
                f"source candidate depth mismatch for {uid}: "
                f"{len(candidates) if isinstance(candidates, list) else 'non-list'} "
                f"!= {expected_depth}"
            )
        metric_ids = [str(item.get("metric_id") or "") for item in candidates]
        if "" in metric_ids or len(set(metric_ids)) != len(metric_ids):
            raise ValueError(f"missing/duplicate source metric IDs for {uid}")
        ranks = [int(item.get("rank", -1)) for item in candidates]
        if ranks != list(range(1, expected_depth + 1)):
            raise ValueError(f"source candidate ranks are not contiguous for {uid}")
        yield row


def truncate_candidate_depth(
    *, input_path: Path, output_path: Path, output_k: int
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if output_k < 1:
        raise ValueError("output_k must be positive")
    if output_path.exists() or _meta_path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite projection: {output_path}")
    input_meta_path = _meta_path(input_path)
    if not input_path.exists() or not input_meta_path.exists():
        raise FileNotFoundError(f"source candidate artifact/metadata missing: {input_path}")

    input_meta = json.loads(input_meta_path.read_text(encoding="utf-8"))
    input_sha = sha256_file(input_path)
    if input_meta.get("output_sha256") != input_sha:
        raise ValueError("source candidate metadata hash mismatch")
    source_k = int(input_meta.get("output_k", -1))
    if output_k > source_k:
        raise ValueError(f"cannot expand source depth {source_k} to {output_k}")

    count = 0

    def projected_rows() -> Iterable[dict[str, Any]]:
        nonlocal count
        rows = _validate_candidates(read_jsonl(input_path), expected_depth=source_k)
        for source in rows:
            row = dict(source)
            row["candidates"] = [dict(item) for item in source["candidates"][:output_k]]
            count += 1
            yield row

    write_jsonl(output_path, projected_rows())
    expected_count = int(input_meta.get("input_count", count))
    if count != expected_count:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"source row count mismatch: {count} != {expected_count}")

    output_meta = dict(input_meta)
    output_meta.update(
        {
            "input_count": count,
            "new_count": count,
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "output_k": output_k,
            "projection": {
                "algorithm": "stable-prefix-v1",
                "source_candidates": str(input_path),
                "source_candidates_sha256": input_sha,
                "source_meta": str(input_meta_path),
                "source_meta_sha256": sha256_file(input_meta_path),
                "source_output_k": source_k,
            },
        }
    )
    meta_path = _meta_path(output_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(output_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(meta_path)
    return output_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-k", type=int, required=True)
    args = parser.parse_args()
    result = truncate_candidate_depth(
        input_path=Path(args.input),
        output_path=Path(args.output),
        output_k=args.output_k,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
