from scripts.tools.silver_match_v3.select_independent_gradient_labels import select


def test_select_requires_match_confidence_and_nonanchor():
    rows = [
        {"norm_uid": "a", "decision": "MATCH", "confidence": "high"},
        {"norm_uid": "b", "decision": "MATCH", "confidence": "medium"},
        {"norm_uid": "c", "decision": "NOISE", "confidence": "high"},
        {"norm_uid": "d", "decision": "MATCH", "confidence": "high"},
    ]
    selected, audit = select(rows, {"d"}, {"high"})
    assert selected == [rows[0]]
    assert audit["selected"] == 1
    assert audit["excluded_hidden_anchor"] == 1
    assert audit["excluded_confidence:medium"] == 1
    assert audit["excluded_decision:NOISE"] == 1
