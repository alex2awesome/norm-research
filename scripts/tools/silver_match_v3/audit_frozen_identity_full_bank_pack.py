#!/usr/bin/env python3
"""Fail closed on a truth-hidden full-bank identity pack and its exclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .make_calibration import split_for, split_group_for


FORBIDDEN_FIELDS = {
    "acceptable_metric_ids",
    "candidate_ids",
    "candidates",
    "decision",
    "label",
    "metric_id",
    "outcome",
    "prediction",
    "raw_response",
    "reason",
}


def _artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _bound(entry: dict[str, Any], anchor: Path) -> Path:
    path = Path(str(entry.get("path") or ""))
    path = path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"artifact missing or hash-mismatched: {path}")
    return path


def _unique(
    path: Path, *, require_unique_groups: bool = True
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    rows = list(read_jsonl(path))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    groups = [str(row.get("source_group") or row.get("split_group") or "") for row in rows]
    if (
        not rows
        or "" in uids
        or "" in groups
        or len(uids) != len(set(uids))
        or (require_unique_groups and len(groups) != len(set(groups)))
    ):
        raise ValueError(f"empty/duplicate/missing identity fields: {path}")
    return rows, set(uids), set(groups)


def audit_pack(
    *,
    manifest_path: Path,
    task: str,
    exclusion_inventory_path: Path,
    identities_path: Path,
    freeze_path: Path,
    binding_path: Path,
    overlap_audit_path: Path,
    pack_root: Path,
    expected_count: int,
    expected_role: str = "verifier_dev",
    binding_panel_key: str = "fresh_dev",
    upstream_role_reference_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path, exclusion_inventory_path, identities_path = map(
        Path.resolve, (manifest_path, exclusion_inventory_path, identities_path)
    )
    freeze_path, binding_path, overlap_audit_path, pack_root = map(
        Path.resolve, (freeze_path, binding_path, overlap_audit_path, pack_root)
    )
    upstream_role_reference_path = (
        upstream_role_reference_path.resolve()
        if upstream_role_reference_path is not None
        else None
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exclusion = json.loads(exclusion_inventory_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    overlap = json.loads(overlap_audit_path.read_text(encoding="utf-8"))
    validation_path = pack_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    if (
        exclusion.get("status") != "FROZEN_BEFORE_NEW_PANEL_SELECTION_PREDICTIONS_OR_LABELS"
        or exclusion.get("task") != task
        or exclusion.get("all_required_categories_present") is not True
        or (exclusion.get("content_contract") or {}).get(
            "model_predictions_metric_ids_reasons_and_outcomes_used"
        )
        is not False
    ):
        raise ValueError("invalid exclusion inventory")
    union_entry = exclusion.get("identity_union") or {}
    union_path = _bound(union_entry, exclusion_inventory_path)
    excluded_rows, excluded_uids, excluded_groups = _unique(
        union_path, require_unique_groups=False
    )
    if (
        len(excluded_rows) != int(union_entry.get("uids", -1))
        or len(excluded_groups) != int(union_entry.get("source_groups", -1))
    ):
        raise ValueError("exclusion-union count mismatch")

    identity_rows, identity_uids, identity_groups = _unique(identities_path)
    frozen_identity = ((freeze.get("outputs") or {}).get("identities") or {})
    freeze_contract = freeze.get("content_contract") or {}
    if (
        freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or freeze.get("task") != task
        or freeze.get("role") != expected_role
        or freeze.get("required_upstream_split") != "train"
        or int(freeze.get("selected_count", -1)) != expected_count
        or int(freeze.get("selected_source_groups", -1)) != expected_count
        or frozen_identity.get("sha256") != sha256_file(identities_path)
        or freeze_contract.get("selection_uses_identity_and_source_group_only") is not True
        or any(
            freeze_contract.get(field) is not False
            for field in (
                "truth_fields_read",
                "model_prediction_fields_read",
                "metric_ids_read",
                "downstream_outcomes_read",
            )
        )
    ):
        raise ValueError("invalid truth-hidden identity freeze")
    if identity_uids & excluded_uids or identity_groups & excluded_groups:
        raise ValueError("fresh identities overlap an exclusion")
    if any(FORBIDDEN_FIELDS & set(row) for row in identity_rows):
        raise ValueError("fresh identity contains forbidden label/proposal fields")

    if (
        overlap.get("schema_version") != "silver-match-v3-label-overlap-audit-v1"
        or ((overlap.get("left") or {}).get("sha256")) != sha256_file(identities_path)
        or ((overlap.get("right") or {}).get("sha256")) != sha256_file(union_path)
        or int(((overlap.get("overlap") or {}).get("uids", -1))) != 0
        or int(((overlap.get("overlap") or {}).get("source_groups", -1))) != 0
    ):
        raise ValueError("identity overlap audit is absent or nonzero")

    fresh = binding.get(binding_panel_key) or {}
    if (
        binding.get("schema_version") != "silver-match-v3-policy-identity-binding-v1"
        or binding.get("status") != "FROZEN_IDENTITIES_LABELS_UNMATERIALIZED_AND_UNOPENED"
        or binding.get("task") != task
        or fresh.get("identities_sha256") != sha256_file(identities_path)
        or fresh.get("freeze_sha256") != sha256_file(freeze_path)
        or int(fresh.get("count", -1)) != expected_count
        or int(fresh.get("source_groups", -1)) != expected_count
    ):
        raise ValueError("identity binding mismatch")
    policy_path = _bound(binding.get("policy") or {}, binding_path)

    if (
        validation.get("schema_version")
        != "silver-match-v3-frozen-identity-full-bank-source-pack-v1"
        or validation.get("status")
        != "FROZEN_CANDIDATE_AND_TRUTH_HIDDEN_BEFORE_LABELING"
        or validation.get("task") != task
        or validation.get("role") != expected_role
        or validation.get("required_split") != "train"
        or validation.get("truth_hidden") is not True
        or validation.get("candidate_proposals_hidden") is not True
        or validation.get("prior_labels_predictions_mi_and_outcomes_not_read") is not True
        or int(validation.get("count", -1)) != expected_count
        or int(validation.get("source_groups", -1)) != expected_count
    ):
        raise ValueError("invalid truth-hidden source-pack validation")
    inputs = validation.get("inputs") or {}
    expected_inputs = {
        "manifest": manifest_path,
        "identities": identities_path,
        "identity_freeze": freeze_path,
        "identity_binding": binding_path,
    }
    for name, path in expected_inputs.items():
        if (inputs.get(name) or {}).get("sha256") != sha256_file(path):
            raise ValueError(f"pack validation input drift: {name}")
    if upstream_role_reference_path is not None and (
        (inputs.get("upstream_role_reference") or {}).get("sha256")
        != sha256_file(upstream_role_reference_path)
    ):
        raise ValueError("pack validation input drift: upstream_role_reference")

    output_entries = validation.get("outputs") or {}
    items_path = _bound(output_entries.get("items") or {}, validation_path)
    bank_path = _bound(output_entries.get("bank") or {}, validation_path)
    item_rows, item_uids, item_groups = _unique(items_path)
    if [row["norm_uid"] for row in item_rows] != [row["norm_uid"] for row in identity_rows]:
        raise ValueError("pack item order/coverage differs from frozen identities")
    if item_uids != identity_uids or item_groups != identity_groups:
        raise ValueError("pack identities differ from frozen identities")
    if any(
        row.get("task") != task
        or row.get("truth_hidden") is not True
        or row.get("gepa_role") != expected_role
        or row.get("predeclared_split") != "train"
        or FORBIDDEN_FIELDS & set(row)
        for row in item_rows
    ):
        raise ValueError("pack item violates truth/candidate-hidden contract")

    canonical: dict[str, dict[str, Any]] = {}
    for corpus, meta in (manifest.get("corpora") or {}).items():
        if meta.get("task") != task:
            continue
        source = Path(str(meta["path"]))
        source = source.resolve() if source.is_absolute() else (manifest_path.parent / source).resolve()
        for row in read_jsonl(source):
            uid = str(row.get("norm_uid") or "")
            if uid in identity_uids:
                canonical[uid] = row
    if set(canonical) != identity_uids:
        raise ValueError("pack identities are not exactly canonical")
    authoritative_roles: dict[str, dict[str, Any]] | None = None
    if upstream_role_reference_path is not None:
        frozen_role_reference = (freeze.get("inputs") or {}).get(
            "upstream_role_reference"
        ) or {}
        if (
            not upstream_role_reference_path.is_file()
            or sha256_file(upstream_role_reference_path)
            != str(frozen_role_reference.get("sha256") or "")
            or frozen_role_reference.get("authoritative") is not True
        ):
            raise ValueError("authoritative role reference is absent or hash-drifted")
        authoritative_roles = {}
        for row in read_jsonl(upstream_role_reference_path):
            uid = str(row.get("norm_uid") or "")
            if (
                not uid
                or uid in authoritative_roles
                or row.get("schema_version")
                != "silver-match-v3-upstream-role-reference-v1"
                or row.get("task") != task
                or row.get("split") not in {"train", "dev", "test"}
                or not row.get("source_group")
            ):
                raise ValueError("invalid or duplicate authoritative role row")
            authoritative_roles[uid] = row
    for identity in identity_rows:
        uid = str(identity["norm_uid"])
        group = split_group_for(canonical[uid])
        if authoritative_roles is None:
            assigned_split = split_for(group)
        else:
            assignment = authoritative_roles.get(uid) or {}
            if (
                assignment.get("task") != task
                or assignment.get("corpus") != canonical[uid].get("corpus")
                or assignment.get("source_group") != group
            ):
                raise ValueError(f"authoritative role/identity mismatch: {uid}")
            assigned_split = str(assignment.get("split") or "")
        if group != identity["source_group"] or assigned_split != "train":
            raise ValueError(f"canonical group/split mismatch: {uid}")

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_meta = (manifest.get("banks") or {}).get(task) or {}
    if (
        bank.get("task") != task
        or bank.get("source_sha256") != bank_meta.get("source_sha256")
        or validation.get("bank_source_sha256") != bank_meta.get("source_sha256")
        or len(bank.get("metrics") or []) != int(validation.get("bank_metric_count", -1))
    ):
        raise ValueError("pack bank differs from canonical manifest bank")

    chunk_entries = output_entries.get("chunks") or {}
    chunks: list[Path] = []
    for raw, expected_sha in sorted(chunk_entries.items()):
        path = Path(raw).resolve()
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"chunk missing or hash-mismatched: {path}")
        chunks.append(path)
    chunk_rows = [row for path in chunks for row in read_jsonl(path)]
    if chunk_rows != item_rows:
        raise ValueError("chunk concatenation differs from source items")

    return {
        "schema_version": "silver-match-v3-frozen-identity-full-bank-pack-audit-v1",
        "status": "EXACT_TRUTH_AND_CANDIDATE_HIDDEN_PACK_PASS",
        "task": task,
        "count": expected_count,
        "source_groups": expected_count,
        "bank_metric_count": len(bank.get("metrics") or []),
        "uid_overlap_with_all_exclusions": 0,
        "source_group_overlap_with_all_exclusions": 0,
        "truth_hidden": True,
        "candidate_proposals_hidden": True,
        "labels_predictions_mi_and_outcomes_read": False,
        "artifacts": {
            "manifest": _artifact(manifest_path),
            "exclusion_inventory": _artifact(exclusion_inventory_path),
            "exclusion_identities": _artifact(union_path),
            "identities": _artifact(identities_path),
            "identity_freeze": _artifact(freeze_path),
            "identity_binding": _artifact(binding_path),
            "policy": _artifact(policy_path),
            "overlap_audit": _artifact(overlap_audit_path),
            "upstream_role_reference": (
                _artifact(upstream_role_reference_path)
                if upstream_role_reference_path is not None
                else None
            ),
            "pack_validation": _artifact(validation_path),
            "items": _artifact(items_path),
            "bank": _artifact(bank_path),
        },
        "chunks": {str(path): sha256_file(path) for path in chunks},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--exclusion-inventory", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--identity-freeze", required=True)
    parser.add_argument("--identity-binding", required=True)
    parser.add_argument("--overlap-audit", required=True)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-role", default="verifier_dev")
    parser.add_argument("--binding-panel-key", default="fresh_dev")
    parser.add_argument("--upstream-role-reference")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit_pack(
        manifest_path=Path(args.manifest),
        task=args.task,
        exclusion_inventory_path=Path(args.exclusion_inventory),
        identities_path=Path(args.identities),
        freeze_path=Path(args.identity_freeze),
        binding_path=Path(args.identity_binding),
        overlap_audit_path=Path(args.overlap_audit),
        pack_root=Path(args.pack_root),
        expected_count=args.expected_count,
        expected_role=args.expected_role,
        binding_panel_key=args.binding_panel_key,
        upstream_role_reference_path=(
            Path(args.upstream_role_reference)
            if args.upstream_role_reference
            else None
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**report, "output": str(output), "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
