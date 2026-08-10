#!/usr/bin/env python3
"""Freeze an append-only resume contract for an interrupted Gemma relocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-binding", required=True)
    parser.add_argument("--partial-model-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    binding_path = Path(args.source_binding).resolve()
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        binding.get("schema_version")
        != "silver-match-v3-pr-gemma-author-runtime-source-binding-v1"
        or binding.get("status") != "FROZEN_SOURCE_BINDING_BEFORE_RELOCATION"
    ):
        raise ValueError("invalid frozen source binding")
    root = Path(args.partial_model_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    source_rows = (binding.get("gemma_snapshot") or {}).get("manifest") or []
    expected = {str(row["name"]): row for row in source_rows}
    partial_rows: list[dict[str, Any]] = []
    seen_logical: set[str] = set()
    for path in sorted(value for value in root.iterdir() if value.is_file()):
        logical_name = path.name
        kind = "canonical_name"
        if logical_name not in expected:
            match = re.fullmatch(r"\.(.+)\.[A-Za-z0-9]{6}", path.name)
            if match and match.group(1) in expected:
                logical_name = match.group(1)
                kind = "interrupted_rsync_temporary_prefix"
            else:
                raise ValueError(f"unexpected file in partial model root: {path.name}")
        if logical_name in seen_logical:
            raise ValueError(f"duplicate partial prefix for frozen model file: {logical_name}")
        seen_logical.add(logical_name)
        if path.name not in expected and kind != "interrupted_rsync_temporary_prefix":
            raise ValueError(f"unexpected file in partial model root: {path.name}")
        size = path.stat().st_size
        if size > int(expected[logical_name]["size"]):
            raise ValueError(f"partial file exceeds frozen source size: {logical_name}")
        partial_rows.append(
            {
                "physical_name": path.name,
                "expected_name": logical_name,
                "kind": kind,
                "size": size,
                "sha256": sha256_file(path),
            }
        )
    usage = shutil.disk_usage(root)
    expected_bytes = sum(int(row["size"]) for row in source_rows)
    present_bytes = sum(int(row["size"]) for row in partial_rows)
    remaining_bytes = expected_bytes - present_bytes
    if remaining_bytes < 0 or usage.free < remaining_bytes + 20 * 1024**3:
        raise ValueError("target filesystem lacks frozen-model bytes plus 20 GiB safety")

    payload = {
        "schema_version": "silver-match-v3-pr-gemma-transfer-continuation-freeze-v1",
        "status": "FROZEN_APPEND_ONLY_RESUME_AFTER_INTERRUPTED_COPY",
        "source_binding": {
            "path": str(binding_path),
            "sha256": sha256_file(binding_path),
        },
        "partial_model_root": str(root),
        "interrupted_prefix": partial_rows,
        "expected_model_manifest_sha256": (binding.get("gemma_snapshot") or {}).get(
            "manifest_sha256"
        ),
        "storage_gate": {
            "filesystem_total_bytes": usage.total,
            "filesystem_free_bytes_pre_resume": usage.free,
            "expected_model_bytes": expected_bytes,
            "partial_bytes_present": present_bytes,
            "remaining_bytes_upper_bound": remaining_bytes,
            "minimum_safety_bytes": 20 * 1024**3,
            "passes": True,
        },
        "continuation_contract": {
            "resume_same_exact_source_snapshot": True,
            "rsync_partial_prefix_may_be_completed_in_place": True,
            "no_unrelated_file_eviction_or_deletion": True,
            "post_copy_full_byte_hash_validation_required": True,
            "inference_forbidden_until_post_copy_gate_passes": True,
            "scientific_settings_unchanged": True,
        },
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **payload}))


if __name__ == "__main__":
    main()
