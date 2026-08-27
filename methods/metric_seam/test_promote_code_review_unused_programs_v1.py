from __future__ import annotations

import json

from methods.metric_seam import promote_code_review_unused_programs_v1 as promotion
from methods.metric_seam.hierarchy_code_runner import build_execution_plan


def test_additive_gate_is_59_without_rewriting_canonical() -> None:
    result = promotion.build()
    assert result["summary"]["relation_local_static_fidelity_count"] == 59
    assert result["summary"]["whole_construct_exact_count"] == 0
    assert {
        level: result["summary"]["by_level"][level][
            "relation_local_static_fidelity_count"
        ]
        for level in ("R1", "R2", "R3")
    } == {"R1": 17, "R2": 19, "R3": 23}
    canonical = json.loads(promotion.CANONICAL.read_text(encoding="utf-8"))
    assert canonical["summary"]["relation_local_static_fidelity_count"] == 56


def test_runner_accepts_promoted_gate_and_groups_programs() -> None:
    plans = build_execution_plan(promotion.build())
    assert len(plans) == 32
    new_ids = {"a35", "a72", "a400", "a181", "a309", "a25"}
    assert new_ids <= {plan["aspect_id"] for plan in plans}
    assert sum(len(plan["relations"]) for plan in plans) == 59


def test_checked_promoted_gate_rebuilds_exactly() -> None:
    checked = json.loads(promotion.OUT.read_text(encoding="utf-8"))
    assert checked == promotion.build()
