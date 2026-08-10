from scripts.tools.silver_match_v3.build_verifier_calibration import build, compact_slate


def test_required_metrics_survive_compaction():
    row = {
        "norm_uid": "u",
        "candidates": [{"metric_id": f"a{i}"} for i in range(20)],
    }
    result = compact_slate(row, ("a19", "a18"), 5)
    assert [value["metric_id"] for value in result["candidates"]] == [
        "a19", "a18", "a0", "a1", "a2"
    ]


def test_truth_hidden_calibration_preserves_retriever_order():
    row = {
        "norm_uid": "u",
        "candidates": [{"metric_id": f"a{i}"} for i in range(20)],
    }
    result = compact_slate(row, ("a3",), 5, reorder_required=False)
    assert [value["metric_id"] for value in result["candidates"]] == [
        "a0", "a1", "a2", "a3", "a4"
    ]


def test_typed_truth_is_kept_to_measure_false_match_rejection():
    proposals = [
        {"task": "t", "norm_uid": "match", "decision": "MATCH", "metric_id": "a0"},
        {"task": "t", "norm_uid": "typed", "decision": "MATCH", "metric_id": "a1"},
    ]
    human = [
        {
            "task": "t",
            "norm_uid": "match",
            "split": "dev",
            "decision": "MATCH",
            "metric_id": "a0",
        },
        {
            "task": "t",
            "norm_uid": "typed",
            "split": "dev",
            "decision": "NO_EXPLICIT_CRITERION",
            "metric_id": None,
        },
    ]
    candidates = {
        uid: {
            "task": "t",
            "norm_uid": uid,
            "candidates": [{"metric_id": "a0"}, {"metric_id": "a1"}],
        }
        for uid in ("match", "typed")
    }
    _, primary, truth, report = build(
        task="t",
        proposal_rows=proposals,
        human_rows=human,
        candidates=candidates,
        split="dev",
        candidate_limit=2,
    )
    assert len(primary) == len(truth) == 2
    assert report["agreement"]["typed_truth:NO_EXPLICIT_CRITERION"] == 1
    assert report["truth_decisions"] == {"MATCH": 1, "NO_EXPLICIT_CRITERION": 1}
