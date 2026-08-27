#!/usr/bin/env python3
"""Hydrate a frozen identity panel into a candidate-hidden full-bank label pack.

This is the label-source analogue of ``prepare_clean_gepa_label_pack`` for a
panel selected directly from the canonical corpus rather than from an existing
retrieval-candidate universe.  It binds the pre-label identity freeze and the
separate policy/identity receipt, then copies only canonical norm content and
the current task bank.  No retrieval proposals, labels, predictions, MI, or
outcomes are read or emitted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


FORBIDDEN_FIELDS = {
    "acceptable_metric_ids",
    "candidate_ids",
    "decision",
    "label",
    "metric_id",
    "outcome",
    "prediction",
    "raw_response",
    "reason",
}


def _resolve(value: str, anchor: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--identity-freeze", required=True)
    parser.add_argument("--identity-binding", required=True)
    parser.add_argument(
        "--upstream-role-reference",
        help=(
            "authoritative pre-frozen UID/source-group/split map; required when "
            "the panel freeze used an upstream split assignment rather than "
            "make_calibration.split_for"
        ),
    )
    parser.add_argument(
        "--binding-panel-key",
        default="fresh_dev",
        help="identity panel entry in the frozen policy binding",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.chunk_size > 25:
        parser.error("--chunk-size must be in [1, 25]")

    manifest_path = Path(args.manifest).resolve()
    identities_path = Path(args.identities).resolve()
    freeze_path = Path(args.identity_freeze).resolve()
    binding_path = Path(args.identity_binding).resolve()
    role_reference_path = (
        Path(args.upstream_role_reference).resolve()
        if args.upstream_role_reference
        else None
    )
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty pack: {output_root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    identity_sha = sha256_file(identities_path)
    freeze_sha = sha256_file(freeze_path)
    manifest_sha = sha256_file(manifest_path)
    frozen_output = (freeze.get("outputs") or {}).get("identities") or {}
    content_contract = freeze.get("content_contract") or {}
    if (
        freeze.get("schema_version")
        != "silver-match-v3-clean-gepa-panel-freeze-v1"
        or freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or freeze.get("task") != args.task
        or str(frozen_output.get("sha256") or "") != identity_sha
        or str(((freeze.get("inputs") or {}).get("manifest") or {}).get("sha256") or "")
        != manifest_sha
        or content_contract.get("selection_uses_identity_and_source_group_only") is not True
        or any(
            content_contract.get(field) is not False
            for field in (
                "downstream_outcomes_read",
                "metric_ids_read",
                "model_prediction_fields_read",
                "truth_fields_read",
            )
        )
    ):
        raise ValueError("identity panel is not a valid truth-hidden pre-label freeze")
    bound_panel = binding.get(args.binding_panel_key) or {}
    if (
        binding.get("schema_version") != "silver-match-v3-policy-identity-binding-v1"
        or binding.get("status")
        != "FROZEN_IDENTITIES_LABELS_UNMATERIALIZED_AND_UNOPENED"
        or binding.get("task") != args.task
        or str(bound_panel.get("identities_sha256") or "") != identity_sha
        or str(bound_panel.get("freeze_sha256") or "") != freeze_sha
        or int(bound_panel.get("count", -1)) != int(freeze.get("selected_count", -2))
    ):
        raise ValueError("policy/identity binding does not match the requested panel")

    identities = list(read_jsonl(identities_path))
    identity_uids = [str(row.get("norm_uid") or "") for row in identities]
    identity_groups = [str(row.get("source_group") or "") for row in identities]
    if (
        not identities
        or "" in identity_uids
        or "" in identity_groups
        or len(identity_uids) != len(set(identity_uids))
        or len(identity_groups) != len(set(identity_groups))
        or len(identities) != int(freeze.get("selected_count", -1))
        or len(identity_groups) != int(freeze.get("selected_source_groups", -1))
    ):
        raise ValueError("identity panel is empty, duplicate, or count-inconsistent")
    required_split = str(freeze.get("required_upstream_split") or "")
    role = str(freeze.get("role") or "")
    if required_split not in {"train", "dev", "test"} or not role:
        raise ValueError("identity freeze lacks a valid role/split")
    authoritative_roles: dict[str, dict[str, Any]] | None = None
    if role_reference_path is not None:
        frozen_role_reference = (freeze.get("inputs") or {}).get(
            "upstream_role_reference"
        ) or {}
        if (
            not role_reference_path.is_file()
            or sha256_file(role_reference_path)
            != str(frozen_role_reference.get("sha256") or "")
            or frozen_role_reference.get("authoritative") is not True
        ):
            raise ValueError(
                "upstream role reference is absent, non-authoritative, or hash-drifted"
            )
        authoritative_roles = {}
        for row in read_jsonl(role_reference_path):
            uid = str(row.get("norm_uid") or "")
            if (
                not uid
                or uid in authoritative_roles
                or row.get("schema_version")
                != "silver-match-v3-upstream-role-reference-v1"
                or row.get("task") != args.task
                or row.get("split") not in {"train", "dev", "test"}
                or not row.get("source_group")
            ):
                raise ValueError("invalid or duplicate authoritative role row")
            authoritative_roles[uid] = row
    for row in identities:
        if (
            row.get("task") != args.task
            or row.get("upstream_split") != required_split
            or row.get("gepa_role") != role
            or FORBIDDEN_FIELDS & set(row)
        ):
            raise ValueError(f"invalid or label-bearing identity row: {row.get('norm_uid')}")

    target = set(identity_uids)
    canonical: dict[str, dict[str, Any]] = {}
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != args.task:
            continue
        for row in read_jsonl(_resolve(str(meta["path"]), manifest_path)):
            uid = str(row.get("norm_uid") or "")
            if uid not in target:
                continue
            if uid in canonical:
                raise ValueError(f"duplicate canonical target UID: {uid}")
            if row.get("task") != args.task or row.get("corpus") != corpus:
                raise ValueError(f"canonical task/corpus mismatch: {uid}")
            if FORBIDDEN_FIELDS & set(row):
                raise ValueError(f"canonical target contains forbidden label field: {uid}")
            canonical[uid] = row
    missing = sorted(target - set(canonical))
    if missing:
        raise ValueError(f"identity UIDs absent from canonical manifest: {missing[:3]}")

    identity_by_uid = {str(row["norm_uid"]): row for row in identities}
    items: list[dict[str, Any]] = []
    for uid in identity_uids:
        norm = canonical[uid]
        group = split_group_for(norm)
        identity = identity_by_uid[uid]
        if authoritative_roles is None:
            assigned_split = split_for(group)
        else:
            assignment = authoritative_roles.get(uid) or {}
            if (
                assignment.get("task") != args.task
                or assignment.get("corpus") != norm["corpus"]
                or assignment.get("source_group") != group
                or assignment.get("source_group") != identity["source_group"]
            ):
                raise ValueError(f"authoritative role/identity mismatch: {uid}")
            assigned_split = str(assignment.get("split") or "")
        if (
            group != identity["source_group"]
            or assigned_split != required_split
            or norm["corpus"] != identity["corpus"]
        ):
            raise ValueError(f"canonical identity/split mismatch: {uid}")
        items.append(
            {
                **norm,
                "source_group": group,
                "split_group": group,
                "split": required_split,
                "predeclared_split": required_split,
                "gepa_role": role,
                "truth_hidden": True,
                "permanently_excluded_from_retriever_gradients": True,
                "permanently_excluded_from_mi_and_outcome_estimation": True,
            }
        )

    bank_meta = (manifest.get("banks") or {}).get(args.task) or {}
    bank_path = _resolve(str(bank_meta.get("path") or ""), manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_hash = str(bank_meta.get("source_sha256") or "")
    metric_ids = [str(row.get("metric_id") or "") for row in bank.get("metrics") or []]
    if (
        bank.get("task") != args.task
        or bank.get("source_sha256") != bank_hash
        or not metric_ids
        or "" in metric_ids
        or len(metric_ids) != len(set(metric_ids))
    ):
        raise ValueError("manifest bank is invalid or identity-mismatched")

    output_root.mkdir(parents=True, exist_ok=True)
    items_path, bank_output = output_root / "items.jsonl", output_root / "bank.json"
    write_jsonl(items_path, items)
    bank_output.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    chunks: list[Path] = []
    for start in range(0, len(items), args.chunk_size):
        path = output_root / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, items[start : start + args.chunk_size])
        chunks.append(path)

    report = {
        "schema_version": "silver-match-v3-frozen-identity-full-bank-source-pack-v1",
        "status": "FROZEN_CANDIDATE_AND_TRUTH_HIDDEN_BEFORE_LABELING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "role": role,
        "binding_panel_key": args.binding_panel_key,
        "required_split": required_split,
        "count": len(items),
        "source_groups": len(identity_groups),
        "corpora": dict(sorted(Counter(str(row["corpus"]) for row in items).items())),
        "bank_metric_count": len(metric_ids),
        "bank_source_sha256": bank_hash,
        "truth_hidden": True,
        "candidate_proposals_hidden": True,
        "prior_labels_predictions_mi_and_outcomes_not_read": True,
        "inputs": {
            "manifest": _artifact(manifest_path),
            "identities": _artifact(identities_path),
            "identity_freeze": _artifact(freeze_path),
            "identity_binding": _artifact(binding_path),
            "upstream_role_reference": (
                _artifact(role_reference_path)
                if role_reference_path is not None
                else None
            ),
            "bank_source": _artifact(bank_path),
        },
        "outputs": {
            "items": _artifact(items_path),
            "bank": _artifact(bank_output),
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
        "usage_contract": {
            "may_train_or_select_retriever": False,
            "may_use_for_mi_or_outcome_estimation": False,
            "may_use_as_final_blind_audit": False,
            "may_use_for_ce_development_truth_after_independent_resolution": True,
        },
    }
    validation_path = output_root / "validation.json"
    validation_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report, "validation_sha256": sha256_file(validation_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
