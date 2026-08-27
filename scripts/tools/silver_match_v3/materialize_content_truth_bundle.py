#!/usr/bin/env python3
"""Materialize a portable, hash-verified content truth release bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .common import sha256_file


NAMES = {
    "role_freeze": "role_freeze.json",
    "source_pack_validation": "source_pack.validation.json",
    "source_items": "source_items.jsonl",
    "source_bank": "source_bank.json",
    "resolution_report": "resolution_report.json",
    "truth": "truth.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-freeze", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    source_freeze = Path(args.release_freeze).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite truth bundle: {output}")
    release = json.loads(source_freeze.read_text(encoding="utf-8"))
    if (
        release.get("schema_version")
        != "silver-match-v3-content-truth-release-freeze-v1"
        or release.get("status") != "FROZEN_COMPLETE_EXACT_TRUTH"
        or release.get("contracts", {}).get("exact_source_coverage") is not True
        or int(release.get("contracts", {}).get("unresolved_count", -1)) != 0
    ):
        raise ValueError("source is not a complete frozen content truth release")
    output.mkdir(parents=True, exist_ok=True)
    copied = {}
    for key, name in NAMES.items():
        value = release["artifacts"][key]
        source = Path(value["path"])
        if sha256_file(source) != value["sha256"]:
            raise ValueError(f"release artifact hash mismatch: {key}")
        destination = output / name
        shutil.copyfile(source, destination)
        copied[key] = {"relative_path": name, "sha256": sha256_file(destination)}
    frozen_copy = output / "SOURCE_FREEZE.json"
    shutil.copyfile(source_freeze, frozen_copy)
    bundle = {
        "schema_version": "silver-match-v3-portable-content-truth-bundle-v1",
        "status": "MATERIALIZED_HASH_VERIFIED",
        "task": release["task"],
        "role": release["role"],
        "count": release["count"],
        "match_count": release["match_count"],
        "typed_nonmatch_count": release["typed_nonmatch_count"],
        "bank_source_sha256": release["bank_source_sha256"],
        "identity_sha256": release["identity"]["sha256"],
        "source_freeze": {
            "relative_path": frozen_copy.name,
            "sha256": sha256_file(frozen_copy),
        },
        "artifacts": copied,
        "contracts": release["contracts"],
    }
    bundle_path = output / "BUNDLE.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**bundle, "bundle_sha256": sha256_file(bundle_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
