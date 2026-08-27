#!/usr/bin/env python3
"""Hydrate a frozen clean GEPA identity panel into a truth-hidden label pack.

The pack contains canonical item text, a deterministically permuted full bank,
and the frozen K50 slate solely for later retrieval-recall accounting.  No
prior decision, proposal, prediction, or truth field is copied into the pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_group_for


FORBIDDEN_IDENTITY_FIELDS = {
    "decision",
    "metric_id",
    "acceptable_metric_ids",
    "reason",
    "label",
    "prediction",
    "outcome",
}


def _resolve(path: str, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def _order(seed: int, namespace: str, value: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\x1f{namespace}\x1f{value}".encode()).hexdigest()
    return digest, value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--identity-freeze", required=True)
    parser.add_argument("--upstream-role-freeze", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--bank-order-seed", type=int, default=20260712)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.candidate_k < 1 or args.chunk_size < 1:
        parser.error("--candidate-k and --chunk-size must be positive")

    manifest_path = Path(args.manifest).resolve()
    identities_path = Path(args.identities).resolve()
    identity_freeze_path = Path(args.identity_freeze).resolve()
    upstream_freeze_path = Path(args.upstream_role_freeze).resolve()
    candidates_path = Path(args.candidates).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity_freeze = json.loads(identity_freeze_path.read_text(encoding="utf-8"))
    upstream_freeze = json.loads(upstream_freeze_path.read_text(encoding="utf-8"))
    if identity_freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES":
        raise ValueError("identity panel is not a pre-label freeze")
    if identity_freeze.get("task") != args.task:
        raise ValueError("identity freeze task mismatch")
    if identity_freeze.get("required_upstream_split") != "train":
        raise ValueError("identity panel is not authoritative upstream-train")
    recorded_identity = ((identity_freeze.get("outputs") or {}).get("identities") or {})
    if recorded_identity.get("sha256") != sha256_file(identities_path):
        raise ValueError("identity freeze hash mismatch")
    if upstream_freeze.get("status") != "FROZEN_AND_AUDIT_VERIFIED":
        raise ValueError("upstream role reference is not audit-verified")
    if upstream_freeze.get("task") != args.task:
        raise ValueError("upstream role freeze task mismatch")
    recorded_candidates = ((upstream_freeze.get("inputs") or {}).get("candidates") or {})
    if Path(str(recorded_candidates.get("path") or "")).resolve() != candidates_path:
        raise ValueError("candidate artifact differs from upstream role freeze")
    candidate_sha256 = sha256_file(candidates_path)
    if recorded_candidates.get("sha256") != candidate_sha256:
        raise ValueError("candidate artifact hash differs from upstream role freeze")

    identities = list(read_jsonl(identities_path))
    identity_uids = [str(row.get("norm_uid") or "") for row in identities]
    if (
        not identities
        or "" in identity_uids
        or len(identity_uids) != len(set(identity_uids))
    ):
        raise ValueError("identity panel is empty or has missing/duplicate UIDs")
    if any(row.get("task") != args.task for row in identities):
        raise ValueError("identity panel contains another task")
    if any(FORBIDDEN_IDENTITY_FIELDS & set(row) for row in identities):
        raise ValueError("identity panel leaks a forbidden label/prediction field")
    if any(row.get("upstream_split") != "train" for row in identities):
        raise ValueError("identity panel contains a nontrain row")
    roles = {str(row.get("gepa_role") or "") for row in identities}
    if len(roles) != 1 or roles != {str(identity_freeze.get("role") or "")}:
        raise ValueError("identity role mismatch")
    role = next(iter(roles))
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
            canonical[uid] = row
    missing = sorted(target - set(canonical))
    if missing:
        raise ValueError(f"identity UIDs absent from canonical manifest: {missing[:3]}")

    identity_by_uid = {str(row["norm_uid"]): row for row in identities}
    items: list[dict[str, Any]] = []
    groups: set[str] = set()
    for uid in identity_uids:
        norm = canonical[uid]
        group = split_group_for(norm)
        if identity_by_uid[uid].get("source_group") != group:
            raise ValueError(f"identity/canonical source-group mismatch: {uid}")
        if group in groups:
            raise ValueError(f"identity panel repeats canonical source group: {group}")
        groups.add(group)
        items.append(
            {
                **norm,
                "split_group": group,
                "source_group": group,
                "split": "train",
                "predeclared_split": "train",
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
    if bank.get("task") != args.task or bank.get("source_sha256") != bank_hash:
        raise ValueError("manifest/bank identity mismatch")
    metrics = list(bank.get("metrics") or [])
    metric_ids = [str(row.get("metric_id") or "") for row in metrics]
    if not metrics or "" in metric_ids or len(metric_ids) != len(set(metric_ids)):
        raise ValueError("bank has missing/duplicate metric IDs")
    permuted_metrics = sorted(
        metrics,
        key=lambda row: _order(
            args.bank_order_seed,
            f"{args.task}:{role}:bank",
            str(row["metric_id"]),
        ),
    )

    selected_candidates: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(candidates_path):
        uid = str(row.get("norm_uid") or "")
        if uid not in target:
            continue
        if uid in selected_candidates:
            raise ValueError(f"candidate artifact repeats target UID: {uid}")
        if row.get("task") != args.task or row.get("bank_source_sha256") != bank_hash:
            raise ValueError(f"candidate task/bank mismatch: {uid}")
        values = list(row.get("candidates") or [])
        if len(values) < args.candidate_k:
            raise ValueError(f"candidate row shorter than requested K: {uid}")
        ids = [str(value.get("metric_id") or "") for value in values]
        if "" in ids or len(ids) != len(set(ids)):
            raise ValueError(f"candidate row has missing/duplicate metric IDs: {uid}")
        selected_candidates[uid] = {
            **row,
            "candidates": [
                {**value, "rank": rank}
                for rank, value in enumerate(values[: args.candidate_k], 1)
            ],
        }
    if set(selected_candidates) != target:
        missing_candidates = sorted(target - set(selected_candidates))
        raise ValueError(f"candidate artifact does not cover panel: {missing_candidates[:3]}")

    output_root.mkdir(parents=True, exist_ok=False)
    items_path = output_root / "items.jsonl"
    candidate_output = output_root / f"candidates.top{args.candidate_k}.jsonl"
    bank_output = output_root / "bank.json"
    write_jsonl(items_path, items)
    write_jsonl(candidate_output, [selected_candidates[uid] for uid in identity_uids])
    bank_output.write_text(
        json.dumps({**bank, "metrics": permuted_metrics}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    chunk_paths: list[Path] = []
    ordered_items = sorted(
        items,
        key=lambda row: _order(
            args.bank_order_seed,
            f"{args.task}:{role}:items",
            str(row["norm_uid"]),
        ),
    )
    for start in range(0, len(ordered_items), args.chunk_size):
        path = output_root / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, ordered_items[start : start + args.chunk_size])
        chunk_paths.append(path)

    report = {
        "schema_version": "silver-match-v3-clean-gepa-label-pack-v1",
        "status": "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING",
        "task": args.task,
        "gepa_role": role,
        "count": len(items),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunk_paths),
        "source_groups": len(groups),
        "corpora": dict(sorted(Counter(str(row["corpus"]) for row in items).items())),
        "upstream_train_count": len(items),
        "bank_metric_count": len(metrics),
        "bank_source_sha256": bank_hash,
        "candidate_k": args.candidate_k,
        "truth_hidden": True,
        "prior_decisions_proposals_predictions_and_outcomes_hidden": True,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "identities": {"path": str(identities_path), "sha256": sha256_file(identities_path)},
            "identity_freeze": {
                "path": str(identity_freeze_path),
                "sha256": sha256_file(identity_freeze_path),
            },
            "upstream_role_freeze": {
                "path": str(upstream_freeze_path),
                "sha256": sha256_file(upstream_freeze_path),
            },
            "candidate_source": {"path": str(candidates_path), "sha256": candidate_sha256},
            "bank_source": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
        },
        "outputs": {
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "candidates": {
                "path": str(candidate_output),
                "sha256": sha256_file(candidate_output),
            },
            "bank": {"path": str(bank_output), "sha256": sha256_file(bank_output)},
            "chunks": {str(path): sha256_file(path) for path in chunk_paths},
        },
        "usage_contract": {
            "optimize_may_mutate_prompts": role == "optimize",
            "select_may_choose_only_predeclared_variants": role == "select",
            "may_train_or_select_retriever": False,
            "may_use_for_mi_or_outcome_estimation": False,
            "may_use_as_test_or_blind_audit": False,
        },
    }
    report_path = output_root / "validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "validation_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
