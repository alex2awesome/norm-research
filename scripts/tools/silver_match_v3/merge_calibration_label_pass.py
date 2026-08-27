#!/usr/bin/env python3
"""Merge train/dev outputs from one independently configured label pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--declared-seed", type=int, required=True)
    parser.add_argument("--model-snapshot", required=True)
    args = parser.parse_args()
    inputs = [Path(value).resolve() for value in args.input]
    output = Path(args.output).resolve()
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if len(inputs) < 2 or len(inputs) != len(set(inputs)):
        raise ValueError("provide at least two distinct pass shards")
    if output.exists() or meta_path.exists():
        raise FileExistsError(output)
    metas = [json.loads(path.with_suffix(path.suffix + ".meta.json").read_text()) for path in inputs]
    config_keys = (
        "model", "prompt_sha256", "prompt_component_sha256", "order_mode",
        "max_candidates", "prompt_rendering", "backend",
    )
    config = {key: metas[0].get(key) for key in config_keys}
    if any(any(meta.get(key) != value for key, value in config.items()) for meta in metas[1:]):
        raise ValueError("label pass shards differ in model/prompt/order/rendering")
    seen = set()

    def rows():
        for path, meta in zip(inputs, metas):
            if meta.get("output_sha256") != sha256_file(path):
                raise ValueError(f"input output/meta hash mismatch: {path}")
            for row in read_jsonl(path):
                uid = str(row.get("norm_uid") or "")
                if not uid or uid in seen:
                    raise ValueError(f"missing/duplicate UID: {uid}")
                if row.get("model") != config["model"] or row.get("prompt_sha256") != config["prompt_sha256"]:
                    raise ValueError(f"row configuration mismatch: {uid}")
                seen.add(uid)
                yield row

    count = write_jsonl(output, rows())
    report = {
        "schema_version": "silver-match-v3-merged-independent-label-pass-v1",
        **config,
        "seed": args.declared_seed,
        "model_snapshot": str(Path(args.model_snapshot).resolve()),
        "model_snapshot_revision": Path(args.model_snapshot).resolve().name,
        "api_base_urls": sorted({str(meta.get("api_base_url")) for meta in metas}),
        "count": count,
        "inputs": {str(path): {"sha256": sha256_file(path), "meta_sha256": sha256_file(path.with_suffix(path.suffix + ".meta.json"))} for path in inputs},
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "meta_sha256": sha256_file(meta_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
