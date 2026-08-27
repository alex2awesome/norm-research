"""Tests for the shared sanitized-ctext representation boundary."""

import pytest

from methods.metric_seam.battery.sanitized_ctext_projection_v1 import (
    project_ctext,
    project_record,
    project_records,
    representation_contract,
    require_projected_ctext,
)
from methods.metric_seam.battery.seal_ctext_train_view_v3 import REDACTION_TOKEN


def test_projection_is_deterministic_and_matches_frozen_sanitizer():
    synthetic = "synthetic-not-a-live-secret-123456"
    raw = f'client_secret = "{synthetic}"\ntimeout = 30'
    first = project_ctext(raw)
    assert first == project_ctext(raw)
    assert synthetic not in first
    assert REDACTION_TOKEN in first
    assert "timeout = 30" in first
    assert require_projected_ctext(first) == first
    with pytest.raises(ValueError, match="canonical sanitized representation"):
        require_projected_ctext(raw)


def test_record_projection_uses_explicit_passthrough_allowlist_only():
    raw = {
        "item_key": "opaque-1",
        "ctext": 'password = "synthetic-value-long-enough-123"',
        "judgement": 9,
        "repo": "not-forwarded",
    }
    projected = project_record(raw, passthrough_keys=("item_key",))
    assert set(projected) == {"ctext", "item_key"}
    assert "judgement" not in projected
    assert "repo" not in projected
    assert REDACTION_TOKEN in projected["ctext"]


def test_record_stream_preserves_order_and_rejects_implicit_keys():
    rows = [
        {"item_key": "one", "ctext": "first", "label": 1},
        {"item_key": "two", "ctext": "second", "label": 0},
    ]
    projected = project_records(rows, passthrough_keys=("item_key",))
    assert [row["item_key"] for row in projected] == ["one", "two"]
    assert all(set(row) == {"ctext", "item_key"} for row in projected)
    with pytest.raises(ValueError, match="missing"):
        project_record(rows[0], passthrough_keys=("missing_key",))


def test_contract_keeps_axes_and_channels_separate():
    contract = representation_contract()
    assert contract["same_hook_required_for"] == ["TRAIN", "code", "heldout", "prompt"]
    assert contract["matching_values_returned"] is False
    assert contract["external_supervised_anchor"] is False

