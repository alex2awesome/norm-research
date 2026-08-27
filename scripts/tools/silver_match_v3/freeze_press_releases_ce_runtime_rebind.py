#!/usr/bin/env python3
"""Freeze an artifact-equivalent Press Releases CE policy for another host."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import sha256_file


def _read(path: str) -> tuple[Path, dict[str, Any]]:
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return value, json.loads(value.read_text(encoding="utf-8"))


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _require_hash(path: Path, expected: str, name: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} hash mismatch: expected={expected}, observed={observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--source-eligibility", required=True)
    parser.add_argument("--relocated-manifest", required=True)
    parser.add_argument("--prior-relocation-attestation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--optimize-candidates", required=True)
    parser.add_argument("--select-candidates", required=True)
    parser.add_argument("--optimize-identity", required=True)
    parser.add_argument("--select-identity", required=True)
    parser.add_argument("--exclusion-union", required=True)
    parser.add_argument("--output-policy", required=True)
    parser.add_argument("--output-eligibility", required=True)
    parser.add_argument("--output-attestation", required=True)
    args = parser.parse_args()

    source_path, source = _read(args.source_policy)
    eligibility_path, eligibility = _read(args.source_eligibility)
    manifest_path, manifest = _read(args.relocated_manifest)
    prior_path, prior = _read(args.prior_relocation_attestation)
    model = Path(args.model).resolve()
    optimize_candidates = Path(args.optimize_candidates).resolve()
    select_candidates = Path(args.select_candidates).resolve()
    optimize_identity = Path(args.optimize_identity).resolve()
    select_identity = Path(args.select_identity).resolve()
    exclusion_union = Path(args.exclusion_union).resolve()
    output_policy = Path(args.output_policy).resolve()
    output_eligibility = Path(args.output_eligibility).resolve()
    output_attestation = Path(args.output_attestation).resolve()
    for output in (output_policy, output_eligibility, output_attestation):
        if output.exists():
            raise FileExistsError(output)

    source_sha = sha256_file(source_path)
    if (
        source.get("schema_version")
        != "silver-match-v3-cross-encoder-press-releases-policy-v2"
        or source.get("scope") != ["press-releases"]
        or eligibility.get("policy_sha256") != source_sha
    ):
        raise ValueError("source policy/eligibility pair is not the frozen PR CE policy")
    if (
        prior.get("status") != "FROZEN_RUNTIME_RELOCATION_ONLY"
        or prior.get("labels_or_roles_changed") is not False
        or prior.get("scientific_hyperparameters_changed") is not False
        or prior.get("manifest", {}).get("mirror_sha256") != sha256_file(manifest_path)
        or prior.get("manifest", {}).get("all_artifact_hashes_equal") is not True
        or prior.get("cross_encoder_policy", {}).get("source_sha256") != source_sha
    ):
        raise ValueError("prior host relocation attestation does not bind these artifacts")
    if "press-releases" not in manifest.get("banks", {}):
        raise ValueError("relocated manifest lacks Press Releases")

    base = source["base_model"]
    if not model.is_dir():
        raise FileNotFoundError(model)
    for relative, expected in base["file_sha256"].items():
        _require_hash(model / relative, expected, f"base_model/{relative}")
    immutable = source["immutable_artifacts"]
    candidate_paths = [optimize_candidates, select_candidates]
    for path, expected in zip(candidate_paths, immutable["candidate_inputs"], strict=True):
        _require_hash(path, expected["sha256"], expected["role"] + " candidates")
    _require_hash(optimize_identity, immutable["optimize_identity"]["sha256"], "optimize identity")
    _require_hash(select_identity, immutable["select_identity"]["sha256"], "select identity")
    _require_hash(exclusion_union, immutable["exclusion_union"]["sha256"], "exclusion union")

    rebound = copy.deepcopy(source)
    rebound["base_model"]["path"] = str(model)
    rebound["immutable_artifacts"]["manifest"]["path"] = str(manifest_path)
    rebound["immutable_artifacts"]["manifest"]["sha256"] = sha256_file(manifest_path)
    for index, path in enumerate(candidate_paths):
        rebound["immutable_artifacts"]["candidate_inputs"][index]["path"] = str(path)
    rebound["immutable_artifacts"]["optimize_identity"]["path"] = str(optimize_identity)
    rebound["immutable_artifacts"]["select_identity"]["path"] = str(select_identity)
    rebound["immutable_artifacts"]["exclusion_union"]["path"] = str(exclusion_union)
    _write_new(output_policy, rebound)

    rebound_eligibility = copy.deepcopy(eligibility)
    rebound_eligibility["policy_sha256"] = sha256_file(output_policy)
    _write_new(output_eligibility, rebound_eligibility)

    restored = copy.deepcopy(rebound)
    restored["base_model"]["path"] = source["base_model"]["path"]
    restored["immutable_artifacts"]["manifest"] = copy.deepcopy(
        source["immutable_artifacts"]["manifest"]
    )
    for index in range(2):
        restored["immutable_artifacts"]["candidate_inputs"][index]["path"] = source[
            "immutable_artifacts"
        ]["candidate_inputs"][index]["path"]
    for key in ("optimize_identity", "select_identity", "exclusion_union"):
        restored["immutable_artifacts"][key]["path"] = source["immutable_artifacts"][key]["path"]
    if restored != source:
        raise AssertionError("runtime rebind changed a non-relocation field")
    restored_eligibility = copy.deepcopy(rebound_eligibility)
    restored_eligibility["policy_sha256"] = eligibility["policy_sha256"]
    if restored_eligibility != eligibility:
        raise AssertionError("runtime eligibility rebind changed a non-hash field")

    attestation = {
        "schema_version": "silver-match-v3-pr-ce-runtime-rebind-v1",
        "status": "FROZEN_RUNTIME_REBIND_ONLY",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scientific_hyperparameters_changed": False,
        "labels_or_roles_changed": False,
        "source_policy": _artifact(source_path),
        "source_eligibility": _artifact(eligibility_path),
        "rebound_policy": _artifact(output_policy),
        "rebound_eligibility": _artifact(output_eligibility),
        "prior_relocation_attestation": _artifact(prior_path),
        "relocated_manifest": {
            **_artifact(manifest_path),
            "bank_source_sha256": manifest["banks"]["press-releases"]["source_sha256"],
            "all_manifest_artifact_hashes_equal_to_source": True,
        },
        "changed_fields": [
            "base_model.path",
            "immutable_artifacts.manifest.path",
            "immutable_artifacts.manifest.sha256",
            "immutable_artifacts.candidate_inputs[0].path",
            "immutable_artifacts.candidate_inputs[1].path",
            "immutable_artifacts.optimize_identity.path",
            "immutable_artifacts.select_identity.path",
            "immutable_artifacts.exclusion_union.path",
            "eligibility.policy_sha256",
        ],
        "verified_runtime_artifacts": {
            "model": {"path": str(model), "file_sha256": base["file_sha256"]},
            "optimize_candidates": _artifact(optimize_candidates),
            "select_candidates": _artifact(select_candidates),
            "optimize_identity": _artifact(optimize_identity),
            "select_identity": _artifact(select_identity),
            "exclusion_union": _artifact(exclusion_union),
        },
    }
    _write_new(output_attestation, attestation)
    print(
        json.dumps(
            {
                "policy": _artifact(output_policy),
                "eligibility": _artifact(output_eligibility),
                "attestation": _artifact(output_attestation),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
