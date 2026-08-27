from scripts.tools.silver_match_v3.select_retrieval_variant import (
    choose_variant,
    paired_bootstrap_evidence,
)


def variant(name, kind, r50, r80, macro50=None, macro80=None, mrr=0.1):
    return {
        "name": name,
        "kind": kind,
        "dev_metrics": {
            "recall_at_50": r50,
            "macro_recall_at_50": r50 if macro50 is None else macro50,
            "recall_at_80": r80,
            "macro_recall_at_80": r80 if macro80 is None else macro80,
            "mrr": mrr,
        },
    }


def test_rejects_adapter_without_clear_dev_gain():
    result = choose_variant(
        [
            variant("bge", "bge_base", 0.80, 0.90),
            variant("nemotron", "nemotron_base", 0.70, 0.85),
            variant("lora", "adapter", 0.72, 0.90),
        ],
        min_adapter_gain=0.03,
    )
    assert result["chosen_name"] == "bge"
    lora = next(row for row in result["decisions"] if row["name"] == "lora")
    assert not lora["eligible"]


def test_adapter_can_win_after_gate_and_dev_objective():
    result = choose_variant(
        [
            variant("bge", "bge_base", 0.80, 0.90),
            variant("nemotron", "nemotron_base", 0.70, 0.85),
            variant("lora", "adapter", 0.84, 0.91),
        ]
    )
    assert result["chosen_name"] == "lora"


def paired_items(ranks):
    return [
        {
            "norm_uid": f"u-{index}",
            "metric_id": f"a{index % 5}",
            "exact_rank": rank,
        }
        for index, rank in enumerate(ranks)
    ]


def test_saturated_policy_can_select_supported_depth_gain_without_r50_loss():
    base_ranks = [20] * 50 + [40] * 50
    adapter_ranks = [10] * 50 + [25] * 50
    base = {
        **variant("nemotron", "nemotron_base", 1.0, 1.0, mrr=0.0375),
        "items": paired_items(base_ranks),
    }
    adapter = {
        **variant("lora", "adapter", 1.0, 1.0, mrr=0.07),
        "items": paired_items(adapter_ranks),
    }
    result = choose_variant(
        [base, adapter],
        adapter_policy="saturated_r50_noninferiority_depth_gain",
        bootstrap_repetitions=2_000,
        bootstrap_seed=11,
    )
    decision = next(row for row in result["decisions"] if row["name"] == "lora")
    assert decision["eligible"]
    assert decision["paired_external_dev_evidence"]["checks"][
        "mrr_supported_positive_gain"
    ]
    assert result["chosen_name"] == "lora"


def test_saturated_policy_rejects_observed_r50_loss_even_with_rank_gains():
    base_ranks = [40] * 100
    adapter_ranks = [10] * 99 + [90]
    evidence = paired_bootstrap_evidence(
        paired_items(base_ranks),
        paired_items(adapter_ranks),
        noninferiority_margin=0.05,
        bootstrap_repetitions=2_000,
        bootstrap_seed=13,
    )
    assert evidence["estimates"]["mrr"]["paired_point_delta"] > 0
    assert not evidence["checks"]["r50_observed_nonloss"]
    assert not evidence["passed"]
