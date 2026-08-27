#!/usr/bin/env python3
"""Attest that a runtime manifest changes only artifact paths.

Every bank and corpus mirror is hashed against the canonical manifest before
the report is sealed.  This keeps a host-local manifest usable without
weakening the canonical scientific identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


def _resolve(path: str, manifest: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (manifest.parent / value).resolve()


def _scientific(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in meta.items() if key != "path"}


def audit(
    source_path: Path,
    runtime_path: Path,
    source_inventory_path: Path | None = None,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    runtime_path = runtime_path.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    source_inventory = None
    inventory_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if source_inventory_path is not None:
        source_inventory_path = source_inventory_path.resolve()
        source_inventory = json.loads(source_inventory_path.read_text(encoding="utf-8"))
        if source_inventory.get("source_manifest_sha256") != sha256_file(source_path):
            raise ValueError("source inventory is bound to another canonical manifest")
        inventory_rows = {
            (str(row["section"]), str(row["name"])): row
            for row in source_inventory.get("artifacts") or []
        }
    if set(source.get("banks") or {}) != set(runtime.get("banks") or {}):
        raise ValueError("runtime bank inventory differs from canonical")
    if set(source.get("corpora") or {}) != set(runtime.get("corpora") or {}):
        raise ValueError("runtime corpus inventory differs from canonical")

    restored = json.loads(json.dumps(runtime))
    artifacts: list[dict[str, Any]] = []
    for section in ("banks", "corpora"):
        for name, source_meta in sorted(source[section].items()):
            runtime_meta = runtime[section][name]
            if _scientific(source_meta) != _scientific(runtime_meta):
                raise ValueError(f"scientific metadata differs: {section}.{name}")
            inventory_row = inventory_rows.get((section, name))
            source_sha = str(
                (inventory_row or {}).get("sha256") or source_meta.get("sha256") or ""
            )
            runtime_artifact = _resolve(str(runtime_meta["path"]), runtime_path)
            if not runtime_artifact.is_file():
                raise FileNotFoundError(runtime_artifact)
            observed_sha = sha256_file(runtime_artifact)
            if not source_sha or observed_sha != source_sha:
                raise ValueError(f"runtime artifact hash differs: {section}.{name}")
            restored[section][name]["path"] = source_meta["path"]
            artifacts.append(
                {
                    "section": section,
                    "name": name,
                    "canonical_path": source_meta["path"],
                    "runtime_path": str(runtime_artifact),
                    "sha256": observed_sha,
                    "bytes": runtime_artifact.stat().st_size,
                }
            )
    if restored != source:
        raise ValueError("runtime manifest differs outside bank/corpus paths")
    return {
        "schema_version": "silver-match-v3-runtime-manifest-attestation-v1",
        "status": "FROZEN_PATH_ONLY_RUNTIME_MANIFEST",
        "canonical_manifest": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "runtime_manifest": {
            "path": str(runtime_path),
            "sha256": sha256_file(runtime_path),
        },
        "only_bank_and_corpus_paths_changed": True,
        "all_scientific_metadata_equal": True,
        "all_artifact_hashes_equal": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "source_artifact_inventory": (
            {
                "path": str(source_inventory_path),
                "sha256": sha256_file(source_inventory_path),
                "artifact_count": len(inventory_rows),
            }
            if source_inventory_path is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--source-artifact-inventory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(
        Path(args.canonical_manifest),
        Path(args.runtime_manifest),
        Path(args.source_artifact_inventory) if args.source_artifact_inventory else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "artifact_count": report["artifact_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
