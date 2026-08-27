from scripts.tools.silver_match_v3.build_teacher_verification_audit import (
    blind_packet,
    build_packet,
)


def test_packet_stratifies_injected_and_retained():
    proposals = [
        {"norm_uid": "a", "metric_id": "a1", "task": "t", "corpus": "c"},
        {"norm_uid": "b", "metric_id": "a2", "task": "t", "corpus": "c"},
    ]
    candidates = [
        {"norm_uid": "a", "candidates": [{"metric_id": "a1"}]},
        {"norm_uid": "b", "candidates": [{"metric_id": "a2", "injected_primary": True}]},
    ]
    first = [
        {"norm_uid": "a", "decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "high"},
        {"norm_uid": "b", "decision": "BETTER_CANDIDATE", "metric_id": "a3", "confidence": "high"},
    ]
    second = [
        {"norm_uid": "a", "decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "medium"},
        {"norm_uid": "b", "decision": "BETTER_CANDIDATE", "metric_id": "a3", "confidence": "high"},
    ]
    norms = {"c": {
        "a": {"norm_uid": "a", "norm": "n1", "context": "c1", "source_id": "x__1"},
        "b": {"norm_uid": "b", "norm": "n2", "context": "c2", "source_id": "y__1"},
    }}
    banks = {"t": {
        "a1": {"metric_id": "a1", "name": "one"},
        "a2": {"metric_id": "a2", "name": "two"},
        "a3": {"metric_id": "a3", "name": "three"},
    }}
    rows, report = build_packet(
        proposals, candidates, first, second, norms, banks,
        per_stratum=2, rare_max_count=1, seed=1,
    )
    assert len(rows) == 2
    by_uid = {row["norm_uid"]: row for row in rows}
    assert by_uid["a"]["gemma_outcome"] == "retained"
    assert by_uid["a"]["audit_design_weight"] == 1.0
    assert by_uid["b"]["proposal_retrieval_status"] == "injected_for_verification"
    assert by_uid["b"]["correction_metrics"][0]["metric_id"] == "a3"
    assert by_uid["a"]["retrieved_alternative_metrics"] == []
    assert report["selected_source_buckets"] == {"x": 1, "y": 1}
    blinded = blind_packet(rows, 1)
    assert "gemma_outcome" not in blinded[0]
    assert "audit_stratum" not in blinded[0]
