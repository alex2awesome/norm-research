#!/usr/bin/env python3
"""Freeze and re-audit the exact Python dependency environment for GPU inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .common import sha256_file


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _environment_versions() -> list[dict[str, str]]:
    versions: dict[str, str] = {}
    for distribution in metadata.distributions():
        raw_name = str(distribution.metadata.get("Name") or "").strip()
        if not raw_name:
            continue
        name = _canonical(raw_name)
        version = str(distribution.version)
        prior = versions.get(name)
        if prior is not None and prior != version:
            raise ValueError(f"duplicate installed distribution versions for {name}")
        versions[name] = version
    return [{"name": name, "version": versions[name]} for name in sorted(versions)]


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _package_payload(raw_name: str) -> dict[str, Any]:
    name = _canonical(raw_name)
    distribution = metadata.distribution(name)
    declared = sorted(str(path) for path in (distribution.files or []))
    entries: list[dict[str, Any]] = []
    for relative in declared:
        relative_path = Path(relative)
        if "__pycache__" in relative_path.parts or relative.endswith((".pyc", ".pyo")):
            continue
        requested = Path(distribution.locate_file(relative_path)).absolute()
        if not requested.is_file():
            continue
        resolved = requested.resolve()
        entries.append(
            {
                "declared_relative_path": relative,
                "requested_path": str(requested),
                "resolved_path": str(resolved),
                "sha256": sha256_file(requested),
                "size_bytes": requested.stat().st_size,
            }
        )
    if not entries:
        raise ValueError(f"runtime distribution has no hashable files: {name}")
    return {
        "name": name,
        "version": str(distribution.version),
        "declared_paths_count": len(declared),
        "declared_paths_sha256": _json_digest(declared),
        "content_file_count": len(entries),
        "content_size_bytes": sum(int(row["size_bytes"]) for row in entries),
        "content_inventory_sha256": _json_digest(entries),
        "files": entries,
    }


def build_inventory(packages: list[str]) -> dict[str, Any]:
    requested = sorted({_canonical(name) for name in packages})
    if not requested:
        raise ValueError("at least one core runtime package is required")
    python = Path(sys.executable).resolve()
    all_versions = _environment_versions()
    core = [_package_payload(name) for name in requested]
    return {
        "schema_version": "silver-match-v3-python-runtime-dependency-inventory-v1",
        "status": "FROZEN_EXACT_PYTHON_RUNTIME_DEPENDENCIES",
        "python": {
            "path": str(python),
            "sha256": sha256_file(python),
            "size_bytes": python.stat().st_size,
            "version": sys.version,
        },
        "environment_distribution_count": len(all_versions),
        "environment_distribution_versions_sha256": _json_digest(all_versions),
        "environment_distribution_versions": all_versions,
        "core_packages": core,
        "core_package_count": len(core),
        "core_content_file_count": sum(int(row["content_file_count"]) for row in core),
        "core_content_size_bytes": sum(int(row["content_size_bytes"]) for row in core),
        "core_content_inventory_sha256": _json_digest(core),
    }


def assert_exact_runtime_dependencies(inventory_path: Path) -> dict[str, Any]:
    inventory_path = inventory_path.resolve()
    frozen = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        frozen.get("schema_version")
        != "silver-match-v3-python-runtime-dependency-inventory-v1"
        or frozen.get("status") != "FROZEN_EXACT_PYTHON_RUNTIME_DEPENDENCIES"
    ):
        raise ValueError("invalid Python runtime dependency inventory")
    python = Path(sys.executable).resolve()
    frozen_python = frozen.get("python") or {}
    if (
        python != Path(str(frozen_python.get("path") or "")).resolve()
        or sha256_file(python) != frozen_python.get("sha256")
        or python.stat().st_size != int(frozen_python.get("size_bytes", -1))
        or sys.version != frozen_python.get("version")
    ):
        raise ValueError("Python executable/version drifted from dependency inventory")
    current_versions = _environment_versions()
    if (
        len(current_versions) != int(frozen.get("environment_distribution_count", -1))
        or _json_digest(current_versions)
        != frozen.get("environment_distribution_versions_sha256")
        or current_versions != frozen.get("environment_distribution_versions")
    ):
        raise ValueError("installed Python distribution set or version drift")
    expected_core = list(frozen.get("core_packages") or [])
    current_core = [_package_payload(str(row.get("name") or "")) for row in expected_core]
    if (
        current_core != expected_core
        or _json_digest(current_core) != frozen.get("core_content_inventory_sha256")
        or sum(int(row["content_file_count"]) for row in current_core)
        != int(frozen.get("core_content_file_count", -1))
        or sum(int(row["content_size_bytes"]) for row in current_core)
        != int(frozen.get("core_content_size_bytes", -1))
    ):
        raise ValueError("core Python runtime dependency content drift")
    return {
        "schema_version": "silver-match-v3-python-runtime-dependency-audit-v1",
        "status": "EXACT_PYTHON_RUNTIME_DEPENDENCIES_PASS",
        "inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "python_sha256": frozen_python["sha256"],
        "environment_distribution_count": len(current_versions),
        "environment_distribution_versions_sha256": _json_digest(current_versions),
        "core_package_count": len(current_core),
        "core_content_file_count": sum(int(row["content_file_count"]) for row in current_core),
        "core_content_size_bytes": sum(int(row["content_size_bytes"]) for row in current_core),
        "core_content_inventory_sha256": _json_digest(current_core),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = build_inventory(args.package)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "output_sha256": sha256_file(output),
                "python_sha256": payload["python"]["sha256"],
                "environment_distribution_count": payload["environment_distribution_count"],
                "core_package_count": payload["core_package_count"],
                "core_content_file_count": payload["core_content_file_count"],
                "core_content_size_bytes": payload["core_content_size_bytes"],
                "core_content_inventory_sha256": payload["core_content_inventory_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
