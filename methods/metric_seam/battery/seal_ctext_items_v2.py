#!/usr/bin/env python3
"""Create a one-way, ctext-only input artifact for a blind compiler run.

The sealer is deliberately a separate preparation step. It may read a historical item file
that carries extra metadata, but emits only ``datapoint_id`` and ``ctext``. Downstream blind
preparation can therefore prove it never deserialized outcome-bearing source keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def seal(source: Path, output: Path, manifest_path: Path) -> dict[str, Any]:
    if output.exists() or manifest_path.exists():
        raise FileExistsError("sealed output and manifest must be created once at new paths")
    raw = json.loads(source.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("source must be a non-empty JSON list")
    rows = []
    source_keys: set[str] = set()
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"source row {index} is not an object")
        source_keys.update(str(key) for key in item)
        datapoint_id, ctext = item.get("datapoint_id"), item.get("ctext")
        if not isinstance(datapoint_id, str) or not datapoint_id:
            raise ValueError(f"source row {index} has invalid datapoint_id")
        if datapoint_id in seen:
            raise ValueError(f"duplicate datapoint_id {datapoint_id!r}")
        if not isinstance(ctext, str):
            raise ValueError(f"source row {index} has no string ctext")
        seen.add(datapoint_id)
        rows.append({"datapoint_id": datapoint_id, "ctext": ctext})
    rows.sort(key=lambda row: row["datapoint_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(rows))
    payload = {
        "schema_version": "metric-seam-ctext-only-seal-v1",
        "source": {"path": str(source.resolve()), "sha256": sha256(source)},
        "sealed": {
            "path": str(output.resolve()),
            "sha256": sha256(output),
            "n_items": len(rows),
            "allowed_keys": ["ctext", "datapoint_id"],
        },
        "projection": {
            "source_keys_observed": sorted(source_keys),
            "source_values_copied_for": ["ctext", "datapoint_id"],
            "all_other_source_values_discarded": True,
            "outcome_values_recorded_in_manifest": False,
        },
    }
    manifest_path.write_bytes(canonical_bytes(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = seal(args.source, args.out, args.manifest)
    print(json.dumps(payload["sealed"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
