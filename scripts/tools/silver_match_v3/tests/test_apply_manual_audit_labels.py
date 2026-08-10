import pytest

from scripts.tools.silver_match_v3.apply_manual_audit_labels import apply_labels


def test_applies_complete_manual_labels():
    packet = [{"norm_uid": "a", "manual_decision": None}]
    labels = [{
        "norm_uid": "a", "manual_decision": "EXACT_MATCH",
        "manual_metric_id": "a1", "manual_reason": "same leaf", "auditor": "agent-1",
    }]
    rows = apply_labels(packet, labels)
    assert rows[0]["manual_decision"] == "EXACT_MATCH"


def test_rejects_missing_packet_label():
    with pytest.raises(ValueError, match="UID mismatch"):
        apply_labels([{"norm_uid": "a"}], [])
