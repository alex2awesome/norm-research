from scripts.tools.silver_match_v3.prepare_teacher_verification import (
    compact_candidates,
    select_proposals,
)


def test_selects_only_high_sonnet_and_injects_missing_primary():
    rows = [
        {"norm_uid": "u", "task": "humor", "decision": "MATCH", "metric_id": "a9", "confidence": "high", "label_source": "sonnet_full"},
        {"norm_uid": "v", "task": "humor", "decision": "MATCH", "metric_id": "a1", "confidence": "low", "label_source": "sonnet_full"},
    ]
    proposals = select_proposals(rows, "humor", {"high"})
    assert [row["norm_uid"] for row in proposals] == ["u"]
    candidates = [{"norm_uid": "u", "candidates": [{"metric_id": f"a{i}"} for i in range(4)]}]
    compact = compact_candidates(proposals, candidates, limit=3)
    assert [row["metric_id"] for row in compact[0]["candidates"]] == ["a9", "a0", "a1"]
    assert compact[0]["candidates"][0]["injected_primary"] is True
    assert compact[0]["primary_was_injected"] is True
