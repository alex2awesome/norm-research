#!/usr/bin/env python3
"""Seal a deterministic sanitized-ctext heldout view with opaque aliases.

This module is a trusted projection boundary.  It may deserialize a historical
item file, but it indexes values only for ``datapoint_id`` and ``ctext``.  Every
ctext is passed through the canonical projection hook before the deterministic
split is selected.  Original identifiers are used in memory for partitioning
and ordering only; neither the bundle nor its manifest contains an identifier
or an identifier map.

The emitted bundle is the complete downstream handoff.  Its rows contain only
``item_key`` and sanitized ``ctext``.  The separate steward manifest contains
hashes, observed source *key names*, and aggregate sanitizer counts, never
source values, matching surfaces, excerpts, or original identifiers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable

try:
    from .sanitized_ctext_projection_v1 import project_ctext
    from .seal_ctext_items_v2 import canonical_bytes, sha256
    from .seal_ctext_train_view_v3 import (
        CREDENTIAL_PATTERNS,
        SANITIZER_SCHEMA,
        sanitize_ctext,
    )
except ImportError:  # pragma: no cover - direct-script compatibility
    from sanitized_ctext_projection_v1 import project_ctext  # type: ignore[no-redef]
    from seal_ctext_items_v2 import canonical_bytes, sha256  # type: ignore[no-redef]
    from seal_ctext_train_view_v3 import (  # type: ignore[no-redef]
        CREDENTIAL_PATTERNS,
        SANITIZER_SCHEMA,
        sanitize_ctext,
    )


BUNDLE_SCHEMA = "metric-seam.sanitized-ctext-heldout-view.v1"
MANIFEST_SCHEMA = "metric-seam.sanitized-ctext-heldout-seal-manifest.v1"
DEFAULT_TRAIN_COUNT = 150
DEFAULT_HELDOUT_COUNT = 100
DEFAULT_SPLIT_SEED = 7


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_exclusive_readonly(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _summarize_counts(
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


def _trusted_projection(
    raw: Any,
    *,
    train_count: int,
    heldout_count: int,
    split_seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Project, sanitize, partition, and erase source identifiers.

    The body of this function is intentionally narrow.  After ``json.loads``
    has deserialized each source object, only the two allowlisted source values
    are indexed.  Iterating ``row.keys()`` records permitted key *names* only.
    """

    if not isinstance(raw, list) or not raw:
        raise ValueError("source must be a nonempty JSON list")
    if train_count <= 0 or heldout_count <= 0:
        raise ValueError("partition counts must be positive")
    if len(raw) != train_count + heldout_count:
        raise ValueError("source row count differs from the frozen partition")

    projected_by_id: dict[str, str] = {}
    counts_by_id: dict[str, dict[str, int]] = {}
    source_keys: set[str] = set()
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("every source row must be an object")
        source_keys.update(str(key) for key in row.keys())
        if "datapoint_id" not in row or "ctext" not in row:
            raise ValueError("a source row lacks an allowlisted field")
        datapoint_id = row["datapoint_id"]
        ctext = row["ctext"]
        if not isinstance(datapoint_id, str) or not datapoint_id:
            raise ValueError("source identifiers must be nonempty strings")
        if datapoint_id in projected_by_id:
            raise ValueError("source identifiers must be unique")
        if not isinstance(ctext, str):
            raise ValueError("source ctext values must be strings")

        # The canonical hook is the operative representation transform for
        # every row and runs before the partition is selected.
        projected = project_ctext(ctext)
        independently_sanitized, aggregate_counts = sanitize_ctext(ctext)
        if independently_sanitized != projected:
            raise AssertionError("canonical projection and sanitizer replay diverged")
        projected_by_id[datapoint_id] = projected
        counts_by_id[datapoint_id] = aggregate_counts

    shuffled = sorted(projected_by_id)
    random.Random(split_seed).shuffle(shuffled)
    train_ids = set(shuffled[:train_count])
    heldout_ids = sorted(set(projected_by_id) - train_ids)
    if len(heldout_ids) != heldout_count:
        raise AssertionError("heldout complement has the wrong row count")

    heldout_items = [
        {
            "item_key": f"heldout_{index:04d}",
            "ctext": projected_by_id[identifier],
        }
        for index, identifier in enumerate(heldout_ids, 1)
    ]
    expected_aliases = [
        f"heldout_{index:04d}" for index in range(1, heldout_count + 1)
    ]
    if [row["item_key"] for row in heldout_items] != expected_aliases:
        raise AssertionError("heldout aliases are not canonical")
    if any(set(row) != {"item_key", "ctext"} for row in heldout_items):
        raise AssertionError("heldout row exceeds the frozen allowlist")

    all_ids = set(projected_by_id)
    count_summary = {
        "schema": SANITIZER_SCHEMA,
        "applied_to_every_row_before_partition": True,
        "full": _summarize_counts(all_ids, counts_by_id),
        "train": _summarize_counts(train_ids, counts_by_id),
        "heldout": _summarize_counts(heldout_ids, counts_by_id),
        "matching_values_recorded": False,
        "source_identifiers_recorded": False,
        "ctext_excerpts_recorded": False,
    }
    projection = {
        "source_keys_observed": sorted(source_keys),
        "source_value_keys_indexed": ["ctext", "datapoint_id"],
        "ctext_projection": (
            "methods.metric_seam.battery.sanitized_ctext_projection_v1.project_ctext"
        ),
        "source_identifiers_used_only_for_partition_and_order": True,
        "source_identifiers_emitted": False,
        "source_identifier_map_emitted": False,
        "all_nonallowlisted_source_values_discarded": True,
        "historical_reference_values_indexed": False,
        "historical_reference_values_recorded": False,
    }
    return heldout_items, projection, count_summary


def seal_heldout_view(
    *,
    source_path: Path,
    bundle_path: Path,
    manifest_path: Path,
    task: str,
    criterion_id: str,
    train_count: int = DEFAULT_TRAIN_COUNT,
    heldout_count: int = DEFAULT_HELDOUT_COUNT,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[Path, Path]:
    """Create one immutable opaque heldout bundle and steward manifest."""

    if bundle_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite a heldout seal artifact")
    if not task or not criterion_id:
        raise ValueError("task and criterion_id are required")

    # json.loads necessarily deserializes whole objects.  `_trusted_projection`
    # is the sole consumer and indexes only the explicitly allowlisted values.
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    heldout_items, projection, count_summary = _trusted_projection(
        raw,
        train_count=train_count,
        heldout_count=heldout_count,
        split_seed=split_seed,
    )
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "task": task,
        "criterion_id": criterion_id,
        "interface": {
            "representation": "sanitized_ctext",
            "sanitizer_schema": SANITIZER_SCHEMA,
            "item_allowed_keys": ["ctext", "item_key"],
            "item_keys": "opaque heldout aliases",
            "source_identifiers_available": False,
            "source_identifier_map_available": False,
            "reference_values_available": False,
        },
        "heldout_items": heldout_items,
    }
    bundle_bytes = canonical_bytes(bundle)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "criterion_id": criterion_id,
        "source": {
            "sha256": sha256(source_path),
            "row_count": len(raw),
            "values_recorded": False,
        },
        "partition": {
            "algorithm": (
                "sort datapoint_id; random.Random(seed).shuffle; first train_count; "
                "sorted complement is heldout"
            ),
            "seed": split_seed,
            "train_count": train_count,
            "heldout_count": heldout_count,
            "source_identifiers_emitted": False,
            "source_identifier_map_emitted": False,
        },
        "projection": projection,
        "credential_redaction": count_summary,
        "artifact": {
            "basename": bundle_path.name,
            "sha256": _sha256_bytes(bundle_bytes),
            "row_count": heldout_count,
            "allowed_item_keys": ["ctext", "item_key"],
            "readonly": True,
        },
        "policy": {
            "trusted_preparer_deserialized_source": True,
            "downstream_receives_bundle_only": True,
            "historical_reference_accessed": False,
            "reference_values_recorded": False,
            "correlation_computed": False,
            "evaluation_performed": False,
            "model_calls": False,
            "gpu_used": False,
            "authorship_isolation": "procedural_not_os_isolated",
        },
    }
    manifest_bytes = canonical_bytes(manifest)
    try:
        _write_exclusive_readonly(bundle_path, bundle_bytes)
        _write_exclusive_readonly(manifest_path, manifest_bytes)
    except Exception:
        for path in (bundle_path, manifest_path):
            if path.exists():
                path.chmod(0o644)
                path.unlink()
        raise
    return bundle_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--criterion-id", required=True)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--heldout-count", type=int, default=DEFAULT_HELDOUT_COUNT)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()
    bundle, manifest = seal_heldout_view(
        source_path=args.source,
        bundle_path=args.bundle,
        manifest_path=args.manifest,
        task=args.task,
        criterion_id=args.criterion_id,
        train_count=args.train_count,
        heldout_count=args.heldout_count,
        split_seed=args.split_seed,
    )
    # Counts and hashes only: never print paths, ctext, identifiers, or values.
    print(json.dumps({
        "artifact_count": 2,
        "bundle_sha256": sha256(bundle),
        "heldout_count": args.heldout_count,
        "manifest_sha256": sha256(manifest),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
