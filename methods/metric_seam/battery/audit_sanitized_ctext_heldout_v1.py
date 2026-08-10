#!/usr/bin/env python3
"""Counts-only steward privacy and replay audits for heldout ctext seals.

Both audits are steward-side operations.  They may deserialize the trusted
source solely to replay the ``datapoint_id``/``ctext`` projection, but their
receipts contain only counts, hashes, and booleans.  Matching values, source
identifiers, identifier maps, ctext excerpts, source paths, and scanner output
are never returned, persisted, or printed.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random
import stat
from typing import Any, Iterable

try:
    from .sanitized_ctext_projection_v1 import project_ctext
    from .seal_ctext_items_v2 import canonical_bytes, sha256
    from .seal_ctext_train_view_v3 import (
        CREDENTIAL_PATTERNS,
        credential_pattern_counts,
        sanitize_ctext,
    )
    from .seal_sanitized_ctext_heldout_v1 import (
        BUNDLE_SCHEMA,
        MANIFEST_SCHEMA,
    )
except ImportError:  # pragma: no cover - direct-script compatibility
    from sanitized_ctext_projection_v1 import project_ctext  # type: ignore[no-redef]
    from seal_ctext_items_v2 import canonical_bytes, sha256  # type: ignore[no-redef]
    from seal_ctext_train_view_v3 import (  # type: ignore[no-redef]
        CREDENTIAL_PATTERNS,
        credential_pattern_counts,
        sanitize_ctext,
    )
    from seal_sanitized_ctext_heldout_v1 import (  # type: ignore[no-redef]
        BUNDLE_SCHEMA,
        MANIFEST_SCHEMA,
    )


PRIVACY_SCHEMA = "metric-seam.sanitized-ctext-heldout-privacy-audit.v1"
REPLAY_SCHEMA = "metric-seam.sanitized-ctext-heldout-replay-audit.v1"
_TOP_LEVEL_KEYS = {"criterion_id", "heldout_items", "interface", "schema", "task"}
_ITEM_KEYS = {"ctext", "item_key"}
_OUTCOME_KEYS = {
    "answer",
    "correlation",
    "gold",
    "ground_truth",
    "judge",
    "judgement",
    "label",
    "reference",
    "reference_value",
    "residual",
    "result",
    "rho",
    "score",
    "target",
    "target_value",
}


def _write_exclusive_readonly(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def _all_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _is_readonly(path: Path) -> bool:
    mode = stat.S_IMODE(path.stat().st_mode)
    return mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0


def _load_source_projection(source_path: Path) -> tuple[dict[str, str], set[str]]:
    """Return only the two-value source projection and observed key names."""

    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("source must be a nonempty JSON list")
    by_id: dict[str, str] = {}
    source_keys: set[str] = set()
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("every source row must be an object")
        source_keys.update(str(key) for key in row.keys())
        if "datapoint_id" not in row or "ctext" not in row:
            raise ValueError("a source row lacks an allowlisted field")
        identifier = row["datapoint_id"]
        ctext = row["ctext"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("source identifiers must be nonempty strings")
        if identifier in by_id:
            raise ValueError("source identifiers must be unique")
        if not isinstance(ctext, str):
            raise ValueError("source ctext values must be strings")
        by_id[identifier] = ctext
    return by_id, source_keys


def build_privacy_receipt(
    *, source_path: Path, bundle_path: Path, manifest_path: Path
) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_by_id, _source_keys = _load_source_projection(source_path)
    items = bundle.get("heldout_items", []) if isinstance(bundle, dict) else []

    item_allowlist_ok = isinstance(items, list) and all(
        isinstance(row, dict)
        and set(row) == _ITEM_KEYS
        and isinstance(row.get("item_key"), str)
        and isinstance(row.get("ctext"), str)
        for row in items
    )
    aliases = [row.get("item_key") for row in items if isinstance(row, dict)]
    expected_aliases = [f"heldout_{index:04d}" for index in range(1, len(items) + 1)]
    aliases_canonical = aliases == expected_aliases

    category_totals: Counter[str] = Counter()
    if item_allowlist_ok:
        for row in items:
            category_totals.update(credential_pattern_counts(row["ctext"]))
    credential_total = sum(category_totals.values())

    bundle_strings = list(_all_strings(bundle))
    source_identifier_occurrence_count = sum(
        value.count(identifier)
        for identifier in source_by_id
        for value in bundle_strings
    )
    structural_outcome_key_count = sum(
        key.strip().casefold().replace("-", "_") in _OUTCOME_KEYS
        for row in items
        for key in _all_keys(row)
    ) if isinstance(items, list) else 1
    top_level_allowlist_ok = isinstance(bundle, dict) and set(bundle) == _TOP_LEVEL_KEYS
    schema_ok = (
        isinstance(bundle, dict)
        and bundle.get("schema") == BUNDLE_SCHEMA
        and manifest.get("schema") == MANIFEST_SCHEMA
    )
    bundle_hash_matches = manifest.get("artifact", {}).get("sha256") == sha256(
        bundle_path
    )
    manifest_forbids_ids = (
        manifest.get("projection", {}).get("source_identifiers_emitted") is False
        and manifest.get("projection", {}).get("source_identifier_map_emitted")
        is False
    )
    readonly_ok = _is_readonly(bundle_path) and _is_readonly(manifest_path)

    violation_count = sum(
        (
            credential_total,
            source_identifier_occurrence_count,
            structural_outcome_key_count,
            int(not item_allowlist_ok),
            int(not aliases_canonical),
            int(not top_level_allowlist_ok),
            int(not schema_ok),
            int(not bundle_hash_matches),
            int(not manifest_forbids_ids),
            int(not readonly_ok),
        )
    )
    return {
        "schema": PRIVACY_SCHEMA,
        "source_sha256": sha256(source_path),
        "bundle_sha256": sha256(bundle_path),
        "manifest_sha256": sha256(manifest_path),
        "counts_only": True,
        "matching_values_recorded": False,
        "source_identifiers_recorded": False,
        "source_identifier_map_recorded": False,
        "ctext_excerpts_recorded": False,
        "credential_scan": {
            "row_count": len(items) if isinstance(items, list) else 0,
            "pattern_counts": {
                pattern.category: category_totals.get(pattern.category, 0)
                for pattern in CREDENTIAL_PATTERNS
            },
            "total_matches": credential_total,
        },
        "interface_audit": {
            "source_identifier_occurrence_count": source_identifier_occurrence_count,
            "structural_outcome_key_count": structural_outcome_key_count,
            "item_key_allowlist_ok": item_allowlist_ok,
            "aliases_canonical": aliases_canonical,
            "top_level_key_allowlist_ok": top_level_allowlist_ok,
            "schema_ok": schema_ok,
            "bundle_hash_matches_manifest": bundle_hash_matches,
            "manifest_forbids_identifier_emission": manifest_forbids_ids,
            "bundle_and_manifest_readonly": readonly_ok,
        },
        "handoff": {
            "artifact_count": 1,
            "bundle_only": True,
            "steward_manifest_excluded": True,
            "authorship_isolation": "procedural_not_os_isolated",
        },
        "violation_count": violation_count,
        "audit_passed": violation_count == 0,
    }


def write_privacy_receipt(
    *,
    source_path: Path,
    bundle_path: Path,
    manifest_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    if receipt_path.exists():
        raise FileExistsError("refusing to overwrite a privacy receipt")
    receipt = build_privacy_receipt(
        source_path=source_path,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
    )
    _write_exclusive_readonly(receipt_path, receipt)
    return receipt


def _split_ids(
    identifiers: Iterable[str], *, train_count: int, split_seed: int
) -> tuple[set[str], list[str]]:
    shuffled = sorted(identifiers)
    random.Random(split_seed).shuffle(shuffled)
    train_ids = set(shuffled[:train_count])
    return train_ids, sorted(set(shuffled) - train_ids)


def _aggregate(
    identifiers: Iterable[str], counts_by_id: dict[str, dict[str, int]]
) -> dict[str, Any]:
    ids = list(identifiers)
    categories = [pattern.category for pattern in CREDENTIAL_PATTERNS]
    category_counts = {
        category: sum(counts_by_id[identifier][category] for identifier in ids)
        for category in categories
    }
    return {
        "row_count": len(ids),
        "changed_row_count": sum(
            any(counts_by_id[identifier].values()) for identifier in ids
        ),
        "category_counts": category_counts,
        "total_matches": sum(category_counts.values()),
    }


def build_replay_receipt(
    *, source_path: Path, bundle_path: Path, manifest_path: Path
) -> dict[str, Any]:
    source_by_id, source_keys = _load_source_projection(source_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = manifest.get("partition", {})
    train_count = partition.get("train_count")
    split_seed = partition.get("seed")
    if not isinstance(train_count, int) or isinstance(train_count, bool):
        raise ValueError("manifest train_count is invalid")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("manifest split seed is invalid")

    projected_by_id: dict[str, str] = {}
    counts_by_id: dict[str, dict[str, int]] = {}
    post_counts_by_id: dict[str, dict[str, int]] = {}
    for identifier, ctext in source_by_id.items():
        projected = project_ctext(ctext)
        independent, counts = sanitize_ctext(ctext)
        if independent != projected:
            raise AssertionError("canonical projection and replay sanitizer diverged")
        projected_by_id[identifier] = projected
        counts_by_id[identifier] = counts
        post_counts_by_id[identifier] = credential_pattern_counts(projected)

    train_ids, heldout_ids = _split_ids(
        projected_by_id, train_count=train_count, split_seed=split_seed
    )
    expected_items = [
        {
            "item_key": f"heldout_{index:04d}",
            "ctext": projected_by_id[identifier],
        }
        for index, identifier in enumerate(heldout_ids, 1)
    ]
    observed_items = bundle.get("heldout_items", [])
    replay_summary = {
        "full": _aggregate(projected_by_id, counts_by_id),
        "train": _aggregate(train_ids, counts_by_id),
        "heldout": _aggregate(heldout_ids, counts_by_id),
    }
    recorded_summary = manifest.get("credential_redaction", {})
    aggregate_counts_match = all(
        recorded_summary.get(split, {}).get(field)
        == replay_summary[split][field]
        for split in ("full", "train", "heldout")
        for field in ("row_count", "changed_row_count", "category_counts", "total_matches")
    )
    source_key_names_match = (
        manifest.get("projection", {}).get("source_keys_observed")
        == sorted(source_keys)
    )
    post_sanitization_total = sum(
        sum(counts.values()) for counts in post_counts_by_id.values()
    )
    rows_equal = observed_items == expected_items
    replay_passed = (
        rows_equal
        and aggregate_counts_match
        and source_key_names_match
        and post_sanitization_total == 0
        and len(heldout_ids) == partition.get("heldout_count")
    )
    return {
        "schema": REPLAY_SCHEMA,
        "source_sha256": sha256(source_path),
        "bundle_sha256": sha256(bundle_path),
        "manifest_sha256": sha256(manifest_path),
        "counts_only": True,
        "matching_values_recorded": False,
        "source_identifiers_recorded": False,
        "source_identifier_map_recorded": False,
        "ctext_excerpts_recorded": False,
        "source_key_name_count": len(source_keys),
        "replay": replay_summary,
        "bundle_rows_equal_replayed_sanitized_heldout": rows_equal,
        "manifest_aggregate_counts_match": aggregate_counts_match,
        "manifest_source_key_names_match": source_key_names_match,
        "post_sanitization_pattern_count": post_sanitization_total,
        "post_sanitization_full_corpus_clean": post_sanitization_total == 0,
        "replay_passed": replay_passed,
    }


def write_replay_receipt(
    *,
    source_path: Path,
    bundle_path: Path,
    manifest_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    if receipt_path.exists():
        raise FileExistsError("refusing to overwrite a replay receipt")
    receipt = build_replay_receipt(
        source_path=source_path,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
    )
    _write_exclusive_readonly(receipt_path, receipt)
    return receipt
