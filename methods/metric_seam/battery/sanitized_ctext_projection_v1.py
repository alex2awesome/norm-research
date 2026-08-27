"""Reusable representation hook for every sanitized-ctext metric channel.

Prompt, code, TRAIN, and held-out paths must call this module instead of
reimplementing credential handling.  The hook imports the exact frozen
sanitizer used by the trusted sealer, returns no matching values, and offers an
assertion for consumers that require an already-projected representation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

try:
    from .seal_ctext_train_view_v3 import (
        SANITIZER_SCHEMA,
        credential_pattern_counts,
        sanitize_ctext,
    )
except ImportError:  # pragma: no cover - direct-script compatibility
    from seal_ctext_train_view_v3 import (  # type: ignore[no-redef]
        SANITIZER_SCHEMA,
        credential_pattern_counts,
        sanitize_ctext,
    )


SCHEMA = "metric-seam.sanitized-ctext-projection-hook.v1"


def project_ctext(text: str) -> str:
    """Return the canonical sanitized representation without match surfaces."""
    if not isinstance(text, str):
        raise TypeError("ctext must be a string")
    sanitized, _counts = sanitize_ctext(text)
    if any(credential_pattern_counts(sanitized).values()):
        raise AssertionError("canonical sanitizer left a recognized credential pattern")
    return sanitized


def require_projected_ctext(text: str) -> str:
    """Reject a raw representation when a downstream channel requires sanitized text."""
    projected = project_ctext(text)
    if projected != text:
        raise ValueError("ctext has not passed the canonical sanitized representation hook")
    return text


def project_record(
    row: Mapping[str, Any],
    *,
    ctext_key: str = "ctext",
    passthrough_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Project one record through an explicit non-ctext key allowlist."""
    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping")
    if ctext_key not in row:
        raise ValueError(f"row has no {ctext_key!r} field")
    passthrough = tuple(passthrough_keys)
    if ctext_key in passthrough:
        raise ValueError("ctext_key cannot also be a passthrough key")
    missing = [key for key in passthrough if key not in row]
    if missing:
        raise ValueError("row is missing an explicitly allowed passthrough key")
    projected = {key: row[key] for key in passthrough}
    projected[ctext_key] = project_ctext(row[ctext_key])
    return projected


def project_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    ctext_key: str = "ctext",
    passthrough_keys: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Apply the identical projection to an ordered record stream."""
    passthrough = tuple(passthrough_keys)
    return [
        project_record(
            row,
            ctext_key=ctext_key,
            passthrough_keys=passthrough,
        )
        for row in rows
    ]


def representation_contract() -> dict[str, Any]:
    """Return non-sensitive machine-readable policy for downstream manifests."""
    return {
        "schema": SCHEMA,
        "sanitizer_schema": SANITIZER_SCHEMA,
        "same_hook_required_for": ["TRAIN", "code", "heldout", "prompt"],
        "matching_values_returned": False,
        "external_supervised_anchor": False,
    }

