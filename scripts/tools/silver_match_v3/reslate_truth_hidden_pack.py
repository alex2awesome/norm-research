#!/usr/bin/env python3
"""Create a deterministic, independently permuted view of a truth-hidden pack.

The source pack's identities, roles, source groups, evidence, and metric bank
are immutable.  Only presentation order and chunk membership change.  The
result is a distinct, append-only label slate that remains compatible with
``validate_independent_teacher_labels`` and
``finalize_exact_multi_pass_truth``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .common import read_jsonl, sha256_file, write_jsonl


FORBIDDEN_ITEM_FIELDS = {
    "acceptable_metric_ids",
    "candidate_ids",
    "candidates",
    "confidence",
    "decision",
    "label",
    "labels",
    "metric_id",
    "outcome",
    "outcomes",
    "prediction",
    "predictions",
    "proposal",
    "proposals",
    "raw_response",
    "reason",
}
PRESERVED_IDENTITY_FIELDS = (
    "task",
    "corpus",
    "source_group",
    "split_group",
    "split",
    "predeclared_split",
    "collection_role",
)


def _ref(path: Path, *, count: int | None = None) -> dict[str, Any]:
    path = path.resolve()
    value: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if count is not None:
        value["count"] = count
    return value


def _recorded_by_name(values: dict[str, str]) -> dict[str, str]:
    result = {Path(path).name: digest for path, digest in values.items()}
    if len(result) != len(values):
        raise ValueError("recorded artifact basenames are not unique")
    return result


def _assert_bound(path: Path, reference: dict[str, Any], label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(reference.get("sha256") or ""):
        raise ValueError(f"{label} hash mismatch: {path}")


def _uid_index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in result:
            raise ValueError(f"{label} has a missing or duplicate norm_uid: {uid!r}")
        result[uid] = row
    if not result:
        raise ValueError(f"{label} is empty")
    return result


def _permutation(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    namespace: str,
    identity: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    """Hash-shuffle rows, forcing a changed order for every nontrivial slate."""

    original_ids = [identity(row) for row in rows]
    if "" in original_ids or len(original_ids) != len(set(original_ids)):
        raise ValueError(f"{namespace} identities are missing or duplicate")

    def key(row: dict[str, Any]) -> tuple[str, str]:
        value = identity(row)
        digest = hashlib.sha256(
            f"{seed}\x1f{namespace}\x1f{value}".encode("utf-8")
        ).hexdigest()
        return digest, value

    shuffled = sorted(rows, key=key)
    if len(shuffled) > 1 and [identity(row) for row in shuffled] == original_ids:
        # Hash ordering can coincidentally preserve a tiny source order.  A
        # deterministic nonzero rotation keeps the distinct-slate contract.
        offset = 1 + seed % (len(shuffled) - 1)
        shuffled = shuffled[offset:] + shuffled[:offset]
    return shuffled


def _load_source(root: Path) -> dict[str, Any]:
    root = root.resolve()
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("truth_hidden") is not True:
        raise ValueError("source pack is not truth-hidden")
    if validation.get("prior_decisions_proposals_predictions_and_outcomes_hidden") is not True:
        raise ValueError("source pack does not attest that prior outcomes are hidden")

    outputs = validation.get("outputs") or {}
    items_path, bank_path = root / "items.jsonl", root / "bank.json"
    _assert_bound(items_path, outputs.get("items") or {}, "source items")
    _assert_bound(bank_path, outputs.get("bank") or {}, "source bank")
    items = list(read_jsonl(items_path))
    item_by_uid = _uid_index(items, "source items")
    task = str(validation.get("task") or "")
    for uid, row in item_by_uid.items():
        leaked = FORBIDDEN_ITEM_FIELDS & set(row)
        if leaked:
            raise ValueError(f"source labeler item leaks truth/proposals: {uid}/{sorted(leaked)}")
        if row.get("truth_hidden") is not True or str(row.get("task") or "") != task:
            raise ValueError(f"source item truth/task mismatch: {uid}")
        if not (row.get("source_group") or row.get("split_group")):
            raise ValueError(f"source item lacks source-group provenance: {uid}")
        if not (row.get("collection_role") and row.get("split")):
            raise ValueError(f"source item lacks frozen role/split: {uid}")

    chunk_paths = sorted((root / "chunks").glob("part-*.jsonl"))
    recorded_chunks = _recorded_by_name(outputs.get("chunks") or {})
    if len(chunk_paths) != len(recorded_chunks):
        raise ValueError("source chunk count differs from validation")
    chunk_uids: list[str] = []
    for path in chunk_paths:
        if sha256_file(path) != recorded_chunks.get(path.name):
            raise ValueError(f"source chunk hash mismatch: {path}")
        chunk_rows = list(read_jsonl(path))
        for row in chunk_rows:
            uid = str(row.get("norm_uid") or "")
            if uid not in item_by_uid or row != item_by_uid[uid]:
                raise ValueError(f"source chunk is not an exact item view: {path}/{uid}")
            chunk_uids.append(uid)
    if len(chunk_uids) != len(set(chunk_uids)) or set(chunk_uids) != set(item_by_uid):
        raise ValueError("source chunks do not cover items exactly once")

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = list(bank.get("metrics") or [])
    metric_ids = [str(row.get("metric_id") or "") for row in metrics]
    if not metrics or "" in metric_ids or len(metric_ids) != len(set(metric_ids)):
        raise ValueError("source bank has missing or duplicate metric IDs")
    if str(bank.get("task") or "") != task or str(bank.get("source_sha256") or "") != str(
        validation.get("bank_source_sha256") or ""
    ):
        raise ValueError("source bank task/source identity mismatch")

    identities: list[dict[str, Any]] | None = None
    identities_ref = outputs.get("identities")
    if identities_ref:
        identities_path = root / "identities.jsonl"
        _assert_bound(identities_path, identities_ref, "source identities")
        identities = list(read_jsonl(identities_path))
        identity_by_uid = _uid_index(identities, "source identities")
        if set(identity_by_uid) != set(item_by_uid):
            raise ValueError("source identities do not exactly cover items")
        for uid, identity_row in identity_by_uid.items():
            item = item_by_uid[uid]
            for field in PRESERVED_IDENTITY_FIELDS:
                if field in identity_row and field in item and identity_row[field] != item[field]:
                    raise ValueError(f"source identity/item {field} mismatch: {uid}")

    return {
        "root": root,
        "validation_path": validation_path,
        "validation": validation,
        "items_path": items_path,
        "items": items,
        "item_by_uid": item_by_uid,
        "bank_path": bank_path,
        "bank": bank,
        "metrics": metrics,
        "identities": identities,
        "chunk_size": int(validation.get("chunk_size") or max(len(items), 1)),
    }


def reslate(source_root: Path, output_root: Path, *, seed: int, chunk_size: int | None = None) -> dict[str, Any]:
    """Materialize one append-only truth-hidden reslate and return validation."""

    source = _load_source(source_root)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"append-only output root already exists: {output_root}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    size = int(chunk_size or source["chunk_size"])
    if size < 1 or size > 25:
        raise ValueError("chunk size must be between 1 and 25")

    items = _permutation(
        source["items"], seed=seed, namespace="items", identity=lambda row: str(row["norm_uid"])
    )
    metrics = _permutation(
        source["metrics"],
        seed=seed,
        namespace="bank",
        identity=lambda row: str(row["metric_id"]),
    )
    identities = (
        _permutation(
            source["identities"],
            seed=seed,
            namespace="identities",
            identity=lambda row: str(row["norm_uid"]),
        )
        if source["identities"] is not None
        else None
    )

    output_root.mkdir(parents=True, exist_ok=False)
    items_path, bank_path = output_root / "items.jsonl", output_root / "bank.json"
    write_jsonl(items_path, items)
    materialized_bank = {**source["bank"], "metrics": metrics}
    bank_path.write_text(
        json.dumps(materialized_bank, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    chunk_paths: list[Path] = []
    for start in range(0, len(items), size):
        path = output_root / "chunks" / f"part-{start // size:03d}.jsonl"
        write_jsonl(path, items[start : start + size])
        chunk_paths.append(path)

    outputs: dict[str, Any] = {
        "items": _ref(items_path, count=len(items)),
        "bank": _ref(bank_path, count=len(metrics)),
        "chunks": {str(path.resolve()): sha256_file(path) for path in chunk_paths},
    }
    if identities is not None:
        identities_path = output_root / "identities.jsonl"
        write_jsonl(identities_path, identities)
        outputs["identities"] = _ref(identities_path, count=len(identities))
        role_paths: dict[str, Any] = {}
        for role in sorted({str(row["collection_role"]) for row in identities}):
            role_path = output_root / "identities" / f"{role}.jsonl"
            role_rows = [row for row in identities if str(row["collection_role"]) == role]
            write_jsonl(role_path, role_rows)
            role_paths[role] = _ref(role_path, count=len(role_rows))
        outputs["identities_by_role"] = role_paths

    if len(items) > 1 and sha256_file(items_path) == sha256_file(source["items_path"]):
        raise AssertionError("reslate failed to change item order")
    if len(metrics) > 1 and sha256_file(bank_path) == sha256_file(source["bank_path"]):
        raise AssertionError("reslate failed to change bank order")

    source_item_map = source["item_by_uid"]
    output_item_map = _uid_index(list(read_jsonl(items_path)), "reslated items")
    if output_item_map != source_item_map:
        raise AssertionError("reslate changed item content, UIDs, roles, or source groups")
    for path in chunk_paths:
        for row in read_jsonl(path):
            if FORBIDDEN_ITEM_FIELDS & set(row):
                raise AssertionError(f"reslated chunk leaks truth/proposals: {path}")

    role_counts = Counter(str(row["collection_role"]) for row in items)
    split_counts = Counter(str(row["split"]) for row in items)
    freeze = {
        "schema_version": "silver-match-v3-truth-hidden-reslate-freeze-v1",
        "status": "FROZEN_RESLATE_BEFORE_ANY_LABELS_PREDICTIONS_OR_OUTCOMES",
        "task": source["validation"]["task"],
        "seed": seed,
        "chunk_size": size,
        "count": len(items),
        "bank_metric_count": len(metrics),
        "role_counts": dict(sorted(role_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_pack": {
            "root": str(source["root"]),
            "validation": _ref(source["validation_path"]),
            "items": _ref(source["items_path"], count=len(items)),
            "bank": _ref(source["bank_path"], count=len(metrics)),
        },
        "outputs": outputs,
        "contract": {
            "same_uid_set": True,
            "same_item_payload_by_uid": True,
            "same_roles_splits_and_source_groups": True,
            "same_metric_cards_by_metric_id": True,
            "items_and_bank_independently_permuted": True,
            "labels_predictions_proposals_mi_and_outcomes_read": False,
            "truth_or_candidate_fields_added_to_labeler_items": False,
        },
    }
    freeze_path = output_root / "FREEZE.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = dict(source["validation"])
    validation.update(
        {
            "schema_version": "silver-match-v3-truth-hidden-reslated-label-pack-v1",
            "status": "FROZEN_TRUTH_HIDDEN_RESLATE_BEFORE_LABELING",
            "count": len(items),
            "chunk_size": size,
            "chunk_count": len(chunk_paths),
            "truth_hidden": True,
            "prior_decisions_proposals_predictions_and_outcomes_hidden": True,
            "selection_freeze": _ref(freeze_path),
            "outputs": outputs,
            "reslate": {
                "seed": seed,
                "source_validation": _ref(source["validation_path"]),
                "same_uid_role_group_payload": True,
                "distinct_items_hash": sha256_file(items_path) != sha256_file(source["items_path"]),
                "distinct_bank_hash": sha256_file(bank_path) != sha256_file(source["bank_path"]),
            },
        }
    )
    validation_path = output_root / "validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if sha256_file(validation_path) == sha256_file(source["validation_path"]):
        raise AssertionError("reslate validation is not distinct from source")

    # Reopen and revalidate the exact emitted pack before returning it.
    _load_source(output_root)
    result = dict(validation)
    result["validation_sha256"] = sha256_file(validation_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--chunk-size", type=int)
    args = parser.parse_args()
    result = reslate(
        Path(args.source_pack),
        Path(args.output_root),
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
