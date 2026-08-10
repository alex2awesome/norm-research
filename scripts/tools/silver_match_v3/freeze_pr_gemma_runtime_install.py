#!/usr/bin/env python3
"""Freeze an isolated, exact-version Gemma runtime install before mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-binding", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--uv", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    binding = Path(args.source_binding).resolve()
    python = Path(args.python).resolve()
    uv = Path(args.uv).resolve()
    environment = Path(args.environment).resolve()
    if environment.exists():
        raise FileExistsError(environment)
    if not binding.is_file() or not python.is_file() or not uv.is_file():
        raise FileNotFoundError("binding, Python, and uv must exist")
    usage = shutil.disk_usage(environment.parent)
    if usage.free < 30 * 1024**3:
        raise ValueError("less than 30 GiB free for isolated runtime")
    payload = {
        "schema_version": "silver-match-v3-pr-gemma-runtime-install-freeze-v1",
        "status": "FROZEN_BEFORE_ISOLATED_RUNTIME_INSTALL",
        "source_binding": {"path": str(binding), "sha256": sha256_file(binding)},
        "base_python": {"path": str(python), "sha256": sha256_file(python)},
        "installer": {"path": str(uv), "sha256": sha256_file(uv)},
        "target_environment": str(environment),
        "exact_requirements": {
            "vllm": "0.23.0",
            "torch": "2.11.0",
            "transformers": "5.12.1",
        },
        "storage_free_bytes_preinstall": usage.free,
        "contract": {
            "new_isolated_environment_only": True,
            "existing_environments_unchanged": True,
            "scientific_inputs_and_settings_unchanged": True,
            "inference_forbidden_until_versions_and_gemma4_registry_validate": True,
        },
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **payload}))


if __name__ == "__main__":
    main()
