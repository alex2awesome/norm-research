#!/usr/bin/env python3
"""Steward-side privacy and interface audit for a ctext compiler view.

Receipts contain counts and hashes only.  The auditor never records or prints a
matching credential surface, source identifier, or ctext excerpt.  A compiler
handoff is admissible only when every count is zero and the handoff envelope is
the single compiler-bundle file (not the steward manifest).
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from .seal_ctext_items_v2 import canonical_bytes, sha256
    from .seal_ctext_train_view_v3 import (
        CREDENTIAL_PATTERNS,
        SCHEMA_BUNDLE,
        credential_pattern_counts,
    )
except ImportError:  # pragma: no cover - direct-script compatibility
    from seal_ctext_items_v2 import canonical_bytes, sha256  # type: ignore[no-redef]
    from seal_ctext_train_view_v3 import (  # type: ignore[no-redef]
        CREDENTIAL_PATTERNS,
        SCHEMA_BUNDLE,
        credential_pattern_counts,
    )


SCHEMA_RECEIPT = "metric-seam.ctext-compiler-steward-audit.v1"
_BUNDLE_TOP_LEVEL_KEYS = {
    "construct",
    "criterion_id",
    "interface",
    "objective",
    "schema",
    "task",
    "train_items",
}
_OUTCOME_ITEM_KEYS = {
    "answer",
    "gold",
    "ground_truth",
    "judge",
    "judgement",
    "label",
    "reference",
    "residual",
    "result",
    "rho",
    "score",
    "target",
}


def _all_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _all_keys(child)


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def build_receipt(*, bundle_path: Path, manifest_path: Path, source_path: Path) -> dict:
    bundle_bytes = bundle_path.read_bytes()
    bundle_text = bundle_bytes.decode("utf-8")
    bundle = json.loads(bundle_text)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("source must be a JSON list")

    category_totals: Counter[str] = Counter()
    item_keys_ok = True
    structural_outcome_key_count = 0
    train_items = bundle.get("train_items", [])
    for item in train_items:
        if not isinstance(item, dict) or set(item) != {"ctext", "item_key"}:
            item_keys_ok = False
            continue
        ctext = item.get("ctext")
        if not isinstance(ctext, str):
            item_keys_ok = False
            continue
        category_totals.update(credential_pattern_counts(ctext))
        structural_outcome_key_count += sum(
            key.strip().lower() in _OUTCOME_ITEM_KEYS for key in _all_keys(item)
        )

    source_ids = {
        row.get("datapoint_id")
        for row in source
        if isinstance(row, dict) and isinstance(row.get("datapoint_id"), str)
    }
    bundle_strings = list(_all_strings(bundle))
    source_identifier_occurrence_count = sum(
        value.count(datapoint_id)
        for datapoint_id in source_ids
        for value in bundle_strings
    )

    source_path_embedded = str(source_path.resolve()) in bundle_text
    manifest_path_embedded = str(manifest_path.resolve()) in bundle_text
    top_level_allowlist_ok = set(bundle) == _BUNDLE_TOP_LEVEL_KEYS
    bundle_hash_matches_manifest = (
        manifest.get("artifacts", {})
        .get("compiler_bundle.json", {})
        .get("sha256")
        == sha256(bundle_path)
    )
    schema_ok = bundle.get("schema") == SCHEMA_BUNDLE
    credential_total = sum(category_totals.values())
    violation_count = sum(
        [
            credential_total,
            source_identifier_occurrence_count,
            structural_outcome_key_count,
            int(not item_keys_ok),
            int(source_path_embedded),
            int(manifest_path_embedded),
            int(not top_level_allowlist_ok),
            int(not bundle_hash_matches_manifest),
            int(not schema_ok),
        ]
    )

    return {
        "schema": SCHEMA_RECEIPT,
        "bundle_sha256": sha256(bundle_path),
        "source_sha256": sha256(source_path),
        "manifest_sha256": sha256(manifest_path),
        "counts_only": True,
        "matching_values_recorded": False,
        "source_identifiers_recorded": False,
        "ctext_excerpts_recorded": False,
        "credential_scan": {
            "n_train_items_scanned": len(train_items),
            "pattern_counts": {
                pattern.category: category_totals.get(pattern.category, 0)
                for pattern in CREDENTIAL_PATTERNS
            },
            "total_matches": credential_total,
        },
        "interface_audit": {
            "source_identifier_occurrence_count": source_identifier_occurrence_count,
            "structural_outcome_key_count": structural_outcome_key_count,
            "item_key_allowlist_ok": item_keys_ok,
            "top_level_key_allowlist_ok": top_level_allowlist_ok,
            "source_path_embedded": source_path_embedded,
            "manifest_path_embedded": manifest_path_embedded,
            "bundle_hash_matches_manifest": bundle_hash_matches_manifest,
            "bundle_schema_ok": schema_ok,
        },
        "compiler_handoff_envelope": {
            "artifact_count": 1,
            "contains_compiler_bundle": True,
            "contains_steward_manifest": False,
            "filesystem_isolation_enforced": False,
            "requires_bundle_only_handoff": True,
        },
        "violation_count": violation_count,
        "compiler_handoff_allowed": violation_count == 0,
    }


def write_receipt(
    *, bundle_path: Path, manifest_path: Path, source_path: Path, receipt_path: Path
) -> dict:
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt {receipt_path}")
    receipt = build_receipt(
        bundle_path=bundle_path, manifest_path=manifest_path, source_path=source_path
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as handle:
        handle.write(canonical_bytes(receipt))
    receipt_path.chmod(0o444)
    return receipt
