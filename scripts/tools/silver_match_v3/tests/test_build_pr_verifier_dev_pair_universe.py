from scripts.tools.silver_match_v3.build_pr_verifier_dev_pair_universe import (
    round_robin_metrics,
)


def test_round_robin_metrics_cycles_leaves_deterministically():
    rows = [
        {"norm_uid": f"a{i}", "proposal_metric_id": "a"} for i in range(4)
    ] + [
        {"norm_uid": "b0", "proposal_metric_id": "b"},
        {"norm_uid": "c0", "proposal_metric_id": "c"},
    ]
    first = round_robin_metrics(rows, limit=4, seed=7, target="REJECT")
    second = round_robin_metrics(list(reversed(rows)), limit=4, seed=7, target="REJECT")
    assert first == second
    assert {row["proposal_metric_id"] for row in first[:3]} == {"a", "b", "c"}
    assert len({row["norm_uid"] for row in first}) == 4
