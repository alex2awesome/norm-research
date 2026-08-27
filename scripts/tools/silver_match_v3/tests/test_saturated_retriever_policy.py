import json
from pathlib import Path


def test_saturated_retriever_policy_is_predeclared_and_fail_closed() -> None:
    path = (
        Path(__file__).parents[1]
        / "policies"
        / "peer_legal_saturated_retriever_policy_v1.json"
    )
    policy = json.loads(path.read_text())
    assert policy["policy_name"] == "saturated_r50_noninferiority_depth_gain"
    assert policy["scope"]["tasks"] == [
        "peer-review",
        "legal-outcome-prediction",
    ]
    gate = policy["external_dev_gate"]
    assert gate["noninferiority_margin"] == 0.05
    assert gate["one_sided_alpha"] == 0.05
    assert gate["bootstrap_repetitions"] == 20000
    assert gate["bootstrap_seed"] == 947311
    assert gate["failure_action"] == "retain_frozen_pretrained_base_fusion"
    assert policy["scope"]["frozen_test_must_remain_untouched_until_promotion_seals"]
    assert all(
        values["frozen_provisional_base_fusion"]["recall_at_50"] == 1.0
        for values in policy["task_inputs"].values()
    )
