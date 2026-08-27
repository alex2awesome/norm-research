#!/usr/bin/env python3
"""Hash every runtime bank/corpus artifact referenced by a frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def build_inventory(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = []
    for section in ("banks", "corpora"):
        for name, metadata in sorted((manifest.get(section) or {}).items()):
            path = Path(str(metadata.get("path") or ""))
            if not path.is_file():
                raise FileNotFoundError(path)
            artifacts.append(
                {
                    "section": section,
                    "name": name,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "schema_version": "silver-match-v3-manifest-artifact-inventory-v1",
        "status": "FROZEN_ON_SOURCE_HOST",
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = build_inventory(Path(args.manifest).resolve())
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
