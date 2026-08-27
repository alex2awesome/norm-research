#!/usr/bin/env python3
"""Independently replay the frozen sanitizer over the full corpus in memory.

The receipt contains counts, hashes, and equality booleans only.  It neither
materializes heldout text nor records source identifiers, matched values, or
ctext excerpts.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Iterable

try:
    from .seal_ctext_items_v2 import canonical_bytes, sha256
    from .seal_ctext_train_view_v3 import (
        CREDENTIAL_PATTERNS,
        SANITIZER_SCHEMA,
        credential_pattern_counts,
        sanitize_ctext,
    )
except ImportError:  # pragma: no cover - direct-script compatibility
    from seal_ctext_items_v2 import canonical_bytes, sha256  # type: ignore[no-redef]
    from seal_ctext_train_view_v3 import (  # type: ignore[no-redef]
        CREDENTIAL_PATTERNS,
        SANITIZER_SCHEMA,
        credential_pattern_counts,
        sanitize_ctext,
    )


SCHEMA = "metric-seam.full-corpus-sanitizer-replay.v1"


def _summarize(
    ids: Iterable[str],
    *,
    redactions: dict[str, dict[str, int]],
    post_scan: dict[str, dict[str, int]],
) -> dict[str, Any]:
    ids = list(ids)
    categories = [pattern.category for pattern in CREDENTIAL_PATTERNS]
    redaction_totals = {
        category: sum(redactions[datapoint_id][category] for datapoint_id in ids)
        for category in categories
    }
    post_totals = {
        category: sum(post_scan[datapoint_id][category] for datapoint_id in ids)
        for category in categories
    }
    return {
        "row_count": len(ids),
        "changed_row_count": sum(any(redactions[datapoint_id].values()) for datapoint_id in ids),
        "redaction_category_counts": redaction_totals,
        "redaction_total_matches": sum(redaction_totals.values()),
        "post_sanitization_pattern_counts": post_totals,
        "post_sanitization_total_matches": sum(post_totals.values()),
    }


def build_replay_receipt(
    *, source_path: Path, bundle_path: Path, manifest_path: Path
) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("source must be a JSON list")

    sanitized_by_id: dict[str, str] = {}
    redactions: dict[str, dict[str, int]] = {}
    post_scan: dict[str, dict[str, int]] = {}
    for index, row in enumerate(source):
        if not isinstance(row, dict):
            raise ValueError(f"source row {index} is not an object")
        datapoint_id, ctext = row.get("datapoint_id"), row.get("ctext")
        if not isinstance(datapoint_id, str) or datapoint_id in sanitized_by_id:
            raise ValueError("source identifiers must be unique strings")
        if not isinstance(ctext, str):
            raise ValueError("source ctext must be a string")
        sanitized, counts = sanitize_ctext(ctext)
        sanitized_by_id[datapoint_id] = sanitized
        redactions[datapoint_id] = counts
        post_scan[datapoint_id] = credential_pattern_counts(sanitized)

    partition = manifest["partition"]
    shuffled = sorted(sanitized_by_id)
    random.Random(partition["seed"]).shuffle(shuffled)
    selected = set(shuffled[: partition["train_count"]])
    all_ids = set(sanitized_by_id)
    replay = {
        "full": _summarize(all_ids, redactions=redactions, post_scan=post_scan),
        "train": _summarize(selected, redactions=redactions, post_scan=post_scan),
        "heldout": _summarize(
            all_ids - selected, redactions=redactions, post_scan=post_scan
        ),
    }
    expected_train = [sanitized_by_id[datapoint_id] for datapoint_id in sorted(selected)]
    observed_train = [row.get("ctext") for row in bundle.get("train_items", [])]

    recorded = manifest["credential_redaction"]
    recorded_counts_match = all(
        recorded[split]["row_count"] == replay[split]["row_count"]
        and recorded[split]["changed_row_count"] == replay[split]["changed_row_count"]
        and recorded[split]["category_counts"]
        == replay[split]["redaction_category_counts"]
        and recorded[split]["total_matches"] == replay[split]["redaction_total_matches"]
        for split in ("full", "train", "heldout")
    )
    post_total = sum(
        replay[split]["post_sanitization_total_matches"]
        for split in ("train", "heldout")
    )
    return {
        "schema": SCHEMA,
        "sanitizer_schema": SANITIZER_SCHEMA,
        "source_sha256": sha256(source_path),
        "bundle_sha256": sha256(bundle_path),
        "manifest_sha256": sha256(manifest_path),
        "counts_only": True,
        "matching_values_recorded": False,
        "source_identifiers_recorded": False,
        "ctext_excerpts_recorded": False,
        "replay": replay,
        "recorded_manifest_counts_match": recorded_counts_match,
        "compiler_train_rows_equal_replayed_sanitized_train": observed_train == expected_train,
        "post_sanitization_full_corpus_clean": post_total == 0,
        "replay_passed": (
            recorded_counts_match and observed_train == expected_train and post_total == 0
        ),
    }


def write_replay_receipt(
    *, source_path: Path, bundle_path: Path, manifest_path: Path, receipt_path: Path
) -> dict[str, Any]:
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite replay receipt {receipt_path}")
    receipt = build_replay_receipt(
        source_path=source_path,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as handle:
        handle.write(canonical_bytes(receipt))
    receipt_path.chmod(0o444)
    return receipt
