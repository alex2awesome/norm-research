#!/usr/bin/env python3
"""Freeze audited path-only mirrors for a manifest and CE policy.

The scientific contents are immutable.  This utility permits only two runtime
relocations: canonical bank/corpus artifact paths in a manifest and the base
model path in a cross-encoder policy.  Every relocated artifact is hash-checked
before an append-only mirror and attestation are written.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relocate_path(value: str, old_prefix: Path, new_prefix: Path) -> Path:
    source = Path(value)
    try:
        suffix = source.relative_to(old_prefix)
    except ValueError as exc:
        raise ValueError(f"artifact path is outside frozen source prefix: {source}") from exc
    return new_prefix / suffix


def _file_equivalence(
    source: Path,
    mirror: Path,
    source_inventory: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not mirror.is_file():
        raise FileNotFoundError(mirror)
    inventory_row = (source_inventory or {}).get(str(source))
    if source.is_file():
        source_sha = sha256_file(source)
        source_size = source.stat().st_size
        source_evidence = "direct_file_hash"
    elif inventory_row:
        source_sha = str(inventory_row["sha256"])
        source_size = int(inventory_row["size_bytes"])
        source_evidence = "hash_bound_source_inventory"
    else:
        raise FileNotFoundError(source)
    mirror_sha = sha256_file(mirror)
    if source_sha != mirror_sha or source_size != mirror.stat().st_size:
        raise ValueError(f"relocated artifact hash mismatch: {source} != {mirror}")
    return {
        "source_path": str(source),
        "mirror_path": str(mirror),
        "source_sha256": source_sha,
        "mirror_sha256": mirror_sha,
        "size_bytes": source_size,
        "source_evidence": source_evidence,
    }


def relocate_manifest(
    source_path: Path,
    output_path: Path,
    old_prefix: Path,
    new_prefix: Path,
    source_inventory: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    mirror = copy.deepcopy(source)
    artifacts: list[dict[str, Any]] = []
    changed: list[str] = []
    for section in ("banks", "corpora"):
        for name, metadata in sorted((source.get(section) or {}).items()):
            if not isinstance(metadata, dict) or not metadata.get("path"):
                raise ValueError(f"manifest {section}.{name} lacks an artifact path")
            original_artifact = Path(str(metadata["path"]))
            relocated_artifact = _relocate_path(
                str(metadata["path"]), old_prefix, new_prefix
            )
            artifacts.append(
                _file_equivalence(
                    original_artifact, relocated_artifact, source_inventory
                )
            )
            mirror[section][name]["path"] = str(relocated_artifact)
            changed.append(f"{section}.{name}.path")

    restored = copy.deepcopy(mirror)
    for section in ("banks", "corpora"):
        for name, metadata in sorted((source.get(section) or {}).items()):
            restored[section][name]["path"] = metadata["path"]
    if restored != source:
        raise AssertionError("manifest relocation changed a non-path field")
    _write_new(output_path, mirror)
    return (
        {
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "mirror_path": str(output_path),
            "mirror_sha256": sha256_file(output_path),
            "only_changed_fields": changed,
            "artifact_count": len(artifacts),
            "all_artifact_hashes_equal": True,
        },
        artifacts,
    )


def relocate_policy(
    source_path: Path,
    output_path: Path,
    model_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    mirror = copy.deepcopy(source)
    old_model_path = str(source["base_model"]["path"])
    mirror["base_model"]["path"] = str(model_path)
    restored = copy.deepcopy(mirror)
    restored["base_model"]["path"] = old_model_path
    if restored != source:
        raise AssertionError("policy relocation changed a scientific field")

    model_artifacts = []
    for filename, expected_sha in sorted(source["base_model"]["file_sha256"].items()):
        artifact = model_path / filename
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        observed = sha256_file(artifact)
        if observed != expected_sha:
            raise ValueError(f"relocated model hash mismatch: {artifact}")
        model_artifacts.append(
            {
                "path": str(artifact),
                "sha256": observed,
                "expected_sha256": expected_sha,
                "size_bytes": artifact.stat().st_size,
            }
        )
    _write_new(output_path, mirror)

    source_eligibility = source_path.with_suffix(".ELIGIBILITY.json")
    mirror_eligibility = output_path.with_suffix(".ELIGIBILITY.json")
    eligibility_record: dict[str, Any] | None = None
    if source_eligibility.exists():
        eligibility = json.loads(source_eligibility.read_text(encoding="utf-8"))
        if eligibility.get("policy_sha256") != sha256_file(source_path):
            raise ValueError("source policy eligibility hash is stale")
        relocated_eligibility = copy.deepcopy(eligibility)
        relocated_eligibility["policy_sha256"] = sha256_file(output_path)
        restored_eligibility = copy.deepcopy(relocated_eligibility)
        restored_eligibility["policy_sha256"] = eligibility["policy_sha256"]
        if restored_eligibility != eligibility:
            raise AssertionError("eligibility relocation changed a non-hash field")
        _write_new(mirror_eligibility, relocated_eligibility)
        eligibility_record = {
            "source_path": str(source_eligibility),
            "source_sha256": sha256_file(source_eligibility),
            "mirror_path": str(mirror_eligibility),
            "mirror_sha256": sha256_file(mirror_eligibility),
            "only_changed_field": "policy_sha256",
        }

    return (
        {
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "mirror_path": str(output_path),
            "mirror_sha256": sha256_file(output_path),
            "only_changed_field": "base_model.path",
            "source_model_path": old_model_path,
            "mirror_model_path": str(model_path),
            "all_declared_model_hashes_match": True,
            "eligibility": eligibility_record,
        },
        model_artifacts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--manifest-old-prefix", required=True)
    parser.add_argument("--manifest-new-prefix", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument(
        "--source-artifact-inventory",
        help=(
            "hash/size inventory produced on the source host; required when source "
            "artifact paths are not mounted on the relocation host"
        ),
    )
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--mirror-model", required=True)
    parser.add_argument("--output-policy", required=True)
    parser.add_argument("--output-attestation", required=True)
    args = parser.parse_args()

    attestation_path = Path(args.output_attestation).resolve()
    if attestation_path.exists():
        raise FileExistsError(attestation_path)
    source_inventory_path = (
        Path(args.source_artifact_inventory).resolve()
        if args.source_artifact_inventory
        else None
    )
    source_inventory_payload = (
        json.loads(source_inventory_path.read_text(encoding="utf-8"))
        if source_inventory_path
        else None
    )
    source_inventory = (
        {
            str(row["path"]): row
            for row in source_inventory_payload.get("artifacts", [])
        }
        if source_inventory_payload
        else None
    )
    source_manifest_path = Path(args.source_manifest).resolve()
    if source_inventory_payload and source_inventory_payload.get(
        "source_manifest_sha256"
    ) != sha256_file(source_manifest_path):
        raise ValueError("source artifact inventory is bound to another manifest")
    manifest_record, manifest_artifacts = relocate_manifest(
        source_manifest_path,
        Path(args.output_manifest).resolve(),
        Path(args.manifest_old_prefix).resolve(),
        Path(args.manifest_new_prefix).resolve(),
        source_inventory,
    )
    policy_record, model_artifacts = relocate_policy(
        Path(args.source_policy).resolve(),
        Path(args.output_policy).resolve(),
        Path(args.mirror_model).resolve(),
    )
    attestation = {
        "schema_version": "silver-match-v3-runtime-relocation-attestation-v1",
        "status": "FROZEN_RUNTIME_RELOCATION_ONLY",
        "scientific_hyperparameters_changed": False,
        "labels_or_roles_changed": False,
        "manifest": manifest_record,
        "manifest_artifacts": manifest_artifacts,
        "source_artifact_inventory": (
            {
                "path": str(source_inventory_path),
                "sha256": sha256_file(source_inventory_path),
                "source_manifest_sha256": source_inventory_payload.get(
                    "source_manifest_sha256"
                ),
            }
            if source_inventory_path
            else None
        ),
        "cross_encoder_policy": policy_record,
        "model_artifacts": model_artifacts,
    }
    _write_new(attestation_path, attestation)
    print(
        json.dumps(
            {
                "output": str(attestation_path),
                "sha256": sha256_file(attestation_path),
                "manifest": manifest_record,
                "policy": policy_record,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
