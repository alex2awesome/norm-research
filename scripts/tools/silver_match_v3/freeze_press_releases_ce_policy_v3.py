#!/usr/bin/env python3
"""Freeze the append-only press-releases CE v3 implementation rebind.

The v2 scientific contract was frozen before select truth was opened, but its
trainer hash points to an implementation artifact that is no longer available.
This utility preserves every scientific field byte-semantically, changes only
implementation/status/supersession metadata, and emits a machine-checkable
semantic-projection audit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


V2_SCHEMA = "silver-match-v3-cross-encoder-press-releases-policy-v2"
V2_POLICY_SHA256 = "27464f8906f740896dec5580126dbaa4136117de24744dde8f2d630ac16393bc"
V3_REVISION = "press-releases-cross-encoder-policy-v3-implementation-rebind"
TRAINER_PATH = Path("scripts/tools/silver_match_v3/train_cross_encoder.py")
SELECTOR_PATH = Path("scripts/tools/silver_match_v3/select_cross_encoder_variants.py")
CURRENT_TRAINER_SHA256 = "a014b80329c087c6be5687fbe89e92d1ef3cb0128033ce6272a5269bd3fd7523"
CURRENT_SELECTOR_SHA256 = "6e3488b6bdd350d66fce36c73568e68ceddb8648b8e50b63a4a608756024724a"
UNAVAILABLE_V2_TRAINER_SHA256 = "97714bd60dca42823a2ca9ee481eae46ccb7ae5cc4f655f92752047e33bbc8d2"
ALLOWED_TOP_LEVEL_DIFFS = {
    "implementation",
    "implementation_supersession",
    "policy_revision",
    "status",
}


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scientific_projection(policy: dict[str, Any]) -> dict[str, Any]:
    """Remove only non-scientific implementation/revision metadata."""

    return {
        key: copy.deepcopy(value)
        for key, value in policy.items()
        if key not in ALLOWED_TOP_LEVEL_DIFFS
    }


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze(
    *,
    v2_policy_path: Path,
    output_policy_path: Path,
    repo_root: Path,
    expected_v2_sha256: str = V2_POLICY_SHA256,
) -> dict[str, Any]:
    v2_policy_path = v2_policy_path.resolve()
    output_policy_path = output_policy_path.resolve()
    repo_root = repo_root.resolve()
    v2_sha = sha256_file(v2_policy_path)
    if v2_sha != expected_v2_sha256:
        raise ValueError(f"unexpected v2 policy hash: {v2_sha}")

    trainer = (repo_root / TRAINER_PATH).resolve()
    selector = (repo_root / SELECTOR_PATH).resolve()
    trainer_sha = sha256_file(trainer)
    selector_sha = sha256_file(selector)
    if trainer_sha != CURRENT_TRAINER_SHA256:
        raise ValueError(f"trainer changed before v3 freeze: {trainer_sha}")
    if selector_sha != CURRENT_SELECTOR_SHA256:
        raise ValueError(f"selector changed before v3 freeze: {selector_sha}")

    v2 = json.loads(v2_policy_path.read_text(encoding="utf-8"))
    if v2.get("schema_version") != V2_SCHEMA or v2.get("scope") != ["press-releases"]:
        raise ValueError("v2 source is not the exact press-releases-only policy")
    pinned = v2.get("implementation", {})
    if pinned.get("train_cross_encoder_sha256") != UNAVAILABLE_V2_TRAINER_SHA256:
        raise ValueError("v2 no longer carries the recorded unavailable trainer hash")
    if pinned.get("select_cross_encoder_variants_sha256") != CURRENT_SELECTOR_SHA256:
        raise ValueError("v2 selector binding changed")

    v3 = copy.deepcopy(v2)
    v3["status"] = (
        "FROZEN_IMPLEMENTATION_REBIND_BEFORE_PRESS_RELEASES_SELECT_TRUTH_SCORING"
    )
    v3["policy_revision"] = V3_REVISION
    v3["implementation"] = {
        "train_cross_encoder_path": str(TRAINER_PATH),
        "train_cross_encoder_sha256": trainer_sha,
        "select_cross_encoder_variants_path": str(SELECTOR_PATH),
        "select_cross_encoder_variants_sha256": selector_sha,
    }
    v3["implementation_supersession"] = {
        "relationship": "implementation-only append-only rebind of v2",
        "supersedes_policy_path": str(v2_policy_path),
        "supersedes_policy_sha256": v2_sha,
        "unavailable_v2_trainer_sha256": UNAVAILABLE_V2_TRAINER_SHA256,
        "replacement_trainer_sha256": trainer_sha,
        "selector_sha256_unchanged": selector_sha,
        "reason": (
            "the exact v2 trainer implementation artifact is unavailable; no "
            "compatibility attestation is made"
        ),
        "scientific_contract_changed": False,
        "select_truth_opened_by_rebind_agent": False,
        "select_scores_opened_by_rebind_agent": False,
        "rebind_agent": "/root/content_resume",
    }

    changed_keys = sorted(
        key for key in set(v2) | set(v3) if v2.get(key) != v3.get(key)
    )
    if set(changed_keys) != ALLOWED_TOP_LEVEL_DIFFS:
        raise AssertionError(f"unexpected v2-to-v3 changed fields: {changed_keys}")
    v2_projection = scientific_projection(v2)
    v3_projection = scientific_projection(v3)
    if v2_projection != v3_projection:
        raise AssertionError("scientific semantic projection changed")

    sibling = output_policy_path.with_suffix("")
    eligibility_path = sibling.with_name(sibling.name + ".ELIGIBILITY.json")
    supersession_path = sibling.with_name(sibling.name + ".SUPERSESSION.json")
    audit_path = sibling.with_name(sibling.name + ".SEMANTIC_PROJECTION_AUDIT.json")
    for path in (output_policy_path, eligibility_path, supersession_path, audit_path):
        if path.exists():
            raise FileExistsError(path)

    _write_new(output_policy_path, v3)
    v3_sha = sha256_file(output_policy_path)
    projection_sha = _canonical_sha(v2_projection)
    eligibility = {
        "schema_version": "silver-match-v3-policy-eligibility-v1",
        "policy_sha256": v3_sha,
        "policy_revision": V3_REVISION,
        "eligible_primary_tasks": ["press-releases"],
        "restricted_primary_tasks": {},
        "status": "PR_ONLY_POLICY_ELIGIBLE_PENDING_DEV_GATES_AND_REQUIRED_BLIND_AUDITS",
        "frozen_before_press_releases_select_truth_scoring": True,
        "scientific_policy_authoring_agent": v2["independence_and_recusal"]["authoring_agent"],
        "implementation_rebind_agent": "/root/content_resume",
    }
    supersession = {
        "schema_version": "silver-match-v3-policy-supersession-v1",
        "status": "FROZEN_APPEND_ONLY_IMPLEMENTATION_REBIND",
        "v2_policy": {"path": str(v2_policy_path), "sha256": v2_sha},
        "v3_policy": {"path": str(output_policy_path), "sha256": v3_sha},
        "unavailable_implementation_sha256": UNAVAILABLE_V2_TRAINER_SHA256,
        "replacement_implementation_sha256": trainer_sha,
        "selector_sha256_unchanged": selector_sha,
        "scientific_contract_changed": False,
        "compatibility_attestation_made": False,
        "select_truth_opened_by_rebind_agent": False,
    }
    audit = {
        "schema_version": "silver-match-v3-policy-semantic-projection-audit-v1",
        "status": "PASS",
        "v2_policy_sha256": v2_sha,
        "v3_policy_sha256": v3_sha,
        "allowed_changed_top_level_fields": sorted(ALLOWED_TOP_LEVEL_DIFFS),
        "observed_changed_top_level_fields": changed_keys,
        "v2_scientific_projection_sha256": projection_sha,
        "v3_scientific_projection_sha256": _canonical_sha(v3_projection),
        "scientific_projection_equal": True,
        "schema_version_unchanged": v3["schema_version"] == v2["schema_version"],
        "scope_unchanged": v3["scope"] == v2["scope"],
        "predeclared_variants_unchanged": (
            v3["predeclared_variants"] == v2["predeclared_variants"]
        ),
        "fixed_training_unchanged": v3["fixed_training"] == v2["fixed_training"],
        "dev_gate_unchanged": v3["dev_gate"] == v2["dev_gate"],
        "role_contract_unchanged": v3["role_contract"] == v2["role_contract"],
    }
    _write_new(eligibility_path, eligibility)
    _write_new(supersession_path, supersession)
    _write_new(audit_path, audit)
    return {
        "policy": {"path": str(output_policy_path), "sha256": v3_sha},
        "eligibility": {"path": str(eligibility_path), "sha256": sha256_file(eligibility_path)},
        "supersession": {"path": str(supersession_path), "sha256": sha256_file(supersession_path)},
        "semantic_projection_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "trainer_sha256": trainer_sha,
        "selector_sha256": selector_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2-policy",
        default="scripts/tools/silver_match_v3/policies/press_releases_cross_encoder_policy_v2.json",
    )
    parser.add_argument(
        "--output-policy",
        default="scripts/tools/silver_match_v3/policies/press_releases_cross_encoder_policy_v3.json",
    )
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = freeze(
        v2_policy_path=Path(args.v2_policy),
        output_policy_path=Path(args.output_policy),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
