#!/usr/bin/env python3
"""Run the frozen Humor selector with exact-hash artifact relocation.

This wrapper is only for the case where a queue-pinned implementation was
updated in the shared checkout after inference.  It verifies the untouched
queue and completion-marker hash, relocates only explicitly frozen artifact
records in an in-memory validation copy, and then invokes the original frozen
selector.  Model outputs, truth, variants, thresholds, and ordering rules are
unchanged.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import run_humor_fresh_select_gpu_queue as queue_runner
from . import select_humor_fresh_release as selector
from .common import sha256_file


SCHEMA = "silver-match-v3-frozen-artifact-relocation-v1"
_ORIGINAL_VALIDATE = queue_runner.validate_queue


def _argument(name: str) -> Path:
    positions = [i for i, value in enumerate(sys.argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(sys.argv):
        raise ValueError(f"expected one {name} argument")
    return Path(sys.argv[positions[0] + 1]).resolve()


def _replace_artifact(value: Any, relocation: dict[str, Any]) -> int:
    count = 0
    if isinstance(value, dict):
        if {"path", "sha256"} <= set(value):
            original = relocation["original"]
            if str(value["path"]) == str(original["path"]):
                if str(value["sha256"]) != str(original["sha256"]):
                    raise ValueError("relocation original hash differs from frozen queue")
                if "bytes" in value and int(value["bytes"]) != int(original["bytes"]):
                    raise ValueError("relocation original size differs from frozen queue")
                value["path"] = str(relocation["relocated"]["path"])
                count += 1
            return count
        return sum(_replace_artifact(child, relocation) for child in value.values())
    if isinstance(value, list):
        return sum(_replace_artifact(child, relocation) for child in value)
    return 0


def _load_manifest() -> tuple[Path, dict[str, Any]]:
    raw = os.environ.get("SILVER_MATCH_V3_ARTIFACT_RELOCATION_MANIFEST")
    if not raw:
        raise ValueError("SILVER_MATCH_V3_ARTIFACT_RELOCATION_MANIFEST is required")
    path = Path(raw).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    queue_path = _argument("--queue")
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("status") != "FROZEN_EXACT_HASH_RELOCATION_ONLY"
        or manifest.get("task") != "humor"
        or manifest.get("queue_sha256") != sha256_file(queue_path)
    ):
        raise ValueError("invalid or wrong-queue relocation manifest")
    return path, manifest


def _relocated_validate(queue: dict[str, Any]) -> None:
    _, manifest = _load_manifest()
    validation_copy = copy.deepcopy(queue)
    for relocation in manifest.get("relocations") or []:
        relocated = relocation.get("relocated") or {}
        path = Path(str(relocated.get("path") or ""))
        if (
            not path.is_file()
            or path.stat().st_size != int(relocated.get("bytes", -1))
            or sha256_file(path) != str(relocated.get("sha256") or "")
            or str(relocated.get("sha256"))
            != str((relocation.get("original") or {}).get("sha256"))
            or int(relocated.get("bytes", -1))
            != int((relocation.get("original") or {}).get("bytes", -2))
        ):
            raise ValueError("relocated artifact is not byte-identical to frozen identity")
        observed = _replace_artifact(validation_copy.get("inputs"), relocation)
        if observed != int(relocation.get("expected_queue_reference_count", 1)):
            raise ValueError("relocation did not replace the exact frozen reference count")
    _ORIGINAL_VALIDATE(validation_copy)


def main() -> None:
    manifest_path, manifest = _load_manifest()
    output = _argument("--output")
    if output.exists():
        raise FileExistsError(output)
    core = output.with_name(output.name + ".relocation-core.json")
    if core.exists():
        raise FileExistsError(core)
    output_position = sys.argv.index("--output") + 1
    sys.argv[output_position] = str(core)
    selector.validate_queue = _relocated_validate
    selector.main()
    report = json.loads(core.read_text(encoding="utf-8"))
    report["frozen_artifact_relocation"] = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "core_selection": {"path": str(core), "sha256": sha256_file(core)},
        "queue_sha256": manifest["queue_sha256"],
        "validation_rule": "relocated bytes must exactly equal each frozen queue artifact identity",
        "model_outputs_truth_variants_thresholds_or_order_rules_changed": False,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "output_sha256": sha256_file(output),
                "relocation_manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
