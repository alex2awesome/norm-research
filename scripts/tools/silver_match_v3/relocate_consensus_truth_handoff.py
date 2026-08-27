#!/usr/bin/env python3
"""Relocate frozen consensus truth bytes without opening held-out outcomes.

Only manifest paths change.  Truth and CE-partition JSONL files are copied
byte-for-byte and re-hashed.  This permits a CPU-only downstream handoff on a
different host while retaining the original source-validation identity.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from .common import sha256_file


SCHEMA = "silver-match-v3-consensus-truth-path-relocation-v1"


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _published_ref(local_path: Path, published_path: Path) -> dict[str, Any]:
    ref = _ref(local_path)
    ref["path"] = str(published_path)
    return ref


def _source_path(ref: Mapping[str, Any], anchor: Path) -> Path:
    path = Path(str(ref.get("path") or ""))
    path = path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()
    if not path.is_file() or sha256_file(path) != ref.get("sha256"):
        raise ValueError(f"frozen source reference differs: {path}")
    return path


def _copy_bound(
    ref: Mapping[str, Any], *, source_anchor: Path, destination: Path, published: Path
) -> dict[str, Any]:
    source = _source_path(ref, source_anchor)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as read_handle, destination.open("xb") as write_handle:
        shutil.copyfileobj(read_handle, write_handle, length=8 * 1024 * 1024)
        write_handle.flush()
        os.fsync(write_handle.fileno())
    if sha256_file(destination) != ref.get("sha256"):
        raise ValueError("relocated consensus bytes changed")
    return {**dict(ref), "path": str(published)}


def relocate(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest_path = Path(args.manifest).resolve()
    source_ce_report_path = Path(args.ce_report).resolve()
    source_validation = Path(args.source_validation).resolve()
    output_root = Path(args.output_root).resolve()
    published_root = Path(args.published_output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    ce_report = json.loads(source_ce_report_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    ce_outputs = ce_report.get("outputs") or {}
    pack_validation = (manifest.get("inputs") or {}).get("pack_validation") or {}
    if (
        manifest.get("schema_version")
        != "silver-match-v3-consensus-training-truth-manifest-v1"
        or manifest.get("status")
        != "COMPLETE_EXACT_CONSENSUS_WITH_FROZEN_SPLITS"
        or manifest.get("task") != "humor"
        or set(outputs) != {"all", "train", "dev", "test"}
        or int(manifest.get("source_group_cross_split_count", -1)) != 0
        or int(manifest.get("blind_rows_training_eligible", -1)) != 0
        or pack_validation.get("sha256") != sha256_file(source_validation)
        or ce_report.get("schema_version")
        != "silver-match-v3-ce-eligible-truth-report-v1"
        or ce_report.get("status")
        != "PARTITIONED_WITHOUT_INFERRED_FAMILY_ANCHORS"
        or ce_report.get("task") != "humor"
        or int(ce_report.get("source_groups_crossing_splits", -1)) != 0
        or set(ce_outputs) != {"eligible", "typed_only"}
    ):
        raise ValueError("consensus/CE relocation source contract differs")

    output_root.mkdir(parents=True, exist_ok=False)
    relocated_outputs = {}
    for name in ("all", "train", "dev", "test"):
        filename = f"truth.{name}.jsonl"
        relocated_outputs[name] = _copy_bound(
            outputs[name],
            source_anchor=source_manifest_path,
            destination=output_root / filename,
            published=published_root / filename,
        )
    all_source = _source_path(outputs["all"], source_manifest_path)
    ce_input = ce_report.get("input") or {}
    if (
        Path(str(ce_input.get("path") or "")).resolve() != all_source
        or ce_input.get("sha256") != outputs["all"].get("sha256")
    ):
        raise ValueError("CE partition is not bound to consensus all-truth")
    relocated_ce_outputs = {}
    for name in ("eligible", "typed_only"):
        filename = f"truth.ce-{name.replace('_', '-')}.jsonl"
        relocated_ce_outputs[name] = _copy_bound(
            ce_outputs[name],
            source_anchor=source_ce_report_path,
            destination=output_root / filename,
            published=published_root / filename,
        )
    validation_destination = output_root / "source.validation.json"
    with source_validation.open("rb") as read_handle, validation_destination.open(
        "xb"
    ) as write_handle:
        shutil.copyfileobj(read_handle, write_handle)
        write_handle.flush()
        os.fsync(write_handle.fileno())
    if sha256_file(validation_destination) != sha256_file(source_validation):
        raise ValueError("source validation changed during relocation")

    relocated_manifest = dict(manifest)
    relocated_manifest["outputs"] = relocated_outputs
    relocated_inputs = dict(relocated_manifest.get("inputs") or {})
    relocated_inputs["pack_validation"] = {
        **dict(pack_validation),
        "path": str(published_root / "source.validation.json"),
    }
    relocated_manifest["inputs"] = relocated_inputs
    relocated_manifest_path = output_root / "MANIFEST.json"
    _write(relocated_manifest_path, relocated_manifest)

    relocated_ce = dict(ce_report)
    relocated_ce["input"] = {
        **dict(ce_input),
        "path": str(published_root / "truth.all.jsonl"),
    }
    relocated_ce["outputs"] = relocated_ce_outputs
    relocated_ce_path = output_root / "CE_REPORT.json"
    _write(relocated_ce_path, relocated_ce)
    report = {
        "schema_version": SCHEMA,
        "status": "BYTE_EXACT_TRUTH_AND_CE_OUTPUTS_PATHS_ONLY_RELOCATED",
        "task": "humor",
        "source": {
            "manifest": _ref(source_manifest_path),
            "ce_report": _ref(source_ce_report_path),
            "source_validation": _ref(source_validation),
        },
        "relocated": {
            "manifest": _published_ref(
                relocated_manifest_path, published_root / "MANIFEST.json"
            ),
            "ce_report": _published_ref(
                relocated_ce_path, published_root / "CE_REPORT.json"
            ),
            "source_validation": _published_ref(
                validation_destination, published_root / "source.validation.json"
            ),
            "published_root": str(published_root),
        },
        "truth_or_ce_output_bytes_changed": False,
        "truth_rows_parsed_by_relocator": 0,
        "test_or_blind_truth_rows_parsed": 0,
        "test_or_blind_outcomes_used": False,
        "gpu_processes_launched": 0,
    }
    _write(output_root / "RELOCATION_REPORT.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ce-report", required=True)
    parser.add_argument("--source-validation", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--published-output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(relocate(args), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
