#!/usr/bin/env python3
"""Run an independent detect-secrets scan and emit a counts-only receipt.

Subprocess output is parsed in memory and never echoed, persisted, or embedded
in exceptions.  The receipt records only tool/configuration versions, hashes,
an aggregate finding count, and pass/fail.  It never records detector findings,
line numbers, source identifiers, ctext excerpts, or matching values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

try:
    from .seal_ctext_items_v2 import canonical_bytes, sha256
except ImportError:  # pragma: no cover - direct-script compatibility
    from seal_ctext_items_v2 import canonical_bytes, sha256  # type: ignore[no-redef]


SCHEMA = "metric-seam.detect-secrets-counts-only-audit.v1"


def _run_hidden(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        # stdout/stderr may contain scanner findings or excerpts.  Never surface
        # either channel, even when the tool fails.
        raise RuntimeError("independent secret scanner failed with hidden output")
    return completed.stdout


def _configuration_sha256(payload: dict[str, Any]) -> str:
    configuration = {
        "filters": payload.get("filters", []),
        "plugins": payload.get("plugins", []),
        "version": payload.get("version"),
    }
    encoded = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_receipt(
    *, artifact_path: Path, scanner_command: str = "detect-secrets"
) -> dict[str, Any]:
    if not artifact_path.is_file():
        raise ValueError("scan artifact must be an existing regular file")
    executable = shutil.which(scanner_command)
    if executable is None:
        raise RuntimeError("independent secret scanner is unavailable")

    version = _run_hidden([executable, "--version"]).strip()
    if not version or len(version) > 100:
        raise RuntimeError("independent secret scanner returned an invalid hidden version")
    raw_scan = _run_hidden([executable, "scan", str(artifact_path.resolve())])
    try:
        payload = json.loads(raw_scan)
    except json.JSONDecodeError as exc:
        raise RuntimeError("independent secret scanner returned invalid hidden output") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise RuntimeError("independent secret scanner returned an invalid hidden schema")

    finding_count = 0
    for findings in payload["results"].values():
        if not isinstance(findings, list):
            raise RuntimeError("independent secret scanner returned an invalid hidden schema")
        finding_count += len(findings)

    return {
        "schema": SCHEMA,
        "artifact_basename": artifact_path.name,
        "artifact_sha256": sha256(artifact_path),
        "auditor_sha256": sha256(Path(__file__)),
        "scanner": "detect-secrets",
        "scanner_version": version,
        "scanner_configuration_sha256": _configuration_sha256(payload),
        "aggregate_finding_count": finding_count,
        "scan_passed": finding_count == 0,
        "counts_only": True,
        "finding_details_recorded": False,
        "detector_type_counts_recorded": False,
        "line_numbers_recorded": False,
        "matching_values_recorded": False,
        "ctext_excerpts_recorded": False,
        "subprocess_output_recorded": False,
        "model_calls": False,
        "gpu_used": False,
    }


def write_receipt(
    *,
    artifact_path: Path,
    receipt_path: Path,
    scanner_command: str = "detect-secrets",
) -> dict[str, Any]:
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt {receipt_path}")
    receipt = build_receipt(
        artifact_path=artifact_path,
        scanner_command=scanner_command,
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as handle:
        handle.write(canonical_bytes(receipt))
    receipt_path.chmod(0o444)
    return receipt

