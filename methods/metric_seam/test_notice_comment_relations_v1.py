from __future__ import annotations

import json
from pathlib import Path

from methods.metric_seam import notice_comment_relations_v1 as notice


ROOT = Path(__file__).resolve().parents[2]


def _relation(text: str, name: str) -> dict:
    return notice.analyze(text)["relations"][name]


def test_actionable_request_uses_dependency_target_not_modal_presence_alone() -> None:
    actionable = _relation(
        "EPA should revise the proposed fee schedule.",
        "actionable_target_dependency",
    )
    assert actionable["score"] > 0
    assert actionable["certificate"]["matches"][0]["action"] == "revise"
    assert actionable["certificate"]["matches"][0]["target_head"] in {
        "fee",
        "schedule",
    }

    no_target = _relation("This outcome should be possible.", "actionable_target_dependency")
    assert no_target["status"] == "measured"
    assert no_target["score"] == 0.0


def test_legal_and_pinpoint_links_require_an_action_in_the_same_sentence() -> None:
    text = (
        "FDA should amend 21 CFR § 101.9(c)(6)(i) because the present definition "
        "excludes resistant starch."
    )
    result = notice.analyze(text)["relations"]
    assert result["legal_authority_action_link"]["score"] > 0
    assert result["pinpoint_provision_action_link"]["score"] > 0
    assert result["causal_support_action_link"]["score"] > 0

    descriptive = notice.analyze(
        "21 CFR § 101.9(c)(6)(i) contains the dietary-fiber definition."
    )["relations"]
    assert descriptive["legal_authority_action_link"]["score"] == 0.0
    assert descriptive["pinpoint_provision_action_link"]["score"] == 0.0


def test_legal_citation_numbers_do_not_become_quantified_claims() -> None:
    citation_only = _relation(
        "FDA should amend 21 CFR § 101.9(c)(6)(i).",
        "quantified_action_link",
    )
    assert citation_only["score"] == 0.0

    quantitative = _relation(
        "The agency should reduce the fee by 25 percent.",
        "quantified_action_link",
    )
    assert quantitative["score"] > 0
    assert quantitative["certificate"]["matches"][0]["span"].strip() == "25 percent"


def test_bare_identifiers_and_rule_years_are_not_quantitative_support() -> None:
    identifier_cases = [
        "The 510(k) Third Party Review Program should be strengthened.",
        "The agency should clarify Part 325 and NWP GC 20.",
        "The 2016 Regulations should be withdrawn.",
        "Paragraph 161.7(e)(2) should be changed.",
    ]
    for text in identifier_cases:
        relations = notice.analyze(text)["relations"]
        assert relations["quantified_action_link"]["score"] == 0.0
        supported = relations["supported_actionable_target_graph"]
        assert "quantity" not in {
            kind
            for match in supported["certificate"]["matches"]
            for kind in match["support_types"]
        }

    measured = _relation(
        "The inspection interval should be increased to 1000 flight hours.",
        "quantified_action_link",
    )
    assert measured["score"] > 0


def test_distributional_relation_requires_group_and_impact_predicate() -> None:
    linked = _relation(
        "The proposal would burden rural households, so EPA should revise it.",
        "distributional_group_impact_link",
    )
    assert linked["score"] > 0
    unlinked = _relation(
        "We met with a rural community yesterday.",
        "distributional_group_impact_link",
    )
    assert unlinked["score"] == 0.0


def test_specialized_structures_are_typed_presence_relations_not_policy_truth() -> None:
    burden = _relation(
        "The estimate covers 100 respondents and 2 annual responses requiring 3 hours each.",
        "burden_breakdown_relation",
    )
    assert burden["score"] > 0
    cost = _relation(
        "The contractor alternative costs $200 compared with $100 for in-house performance.",
        "cost_comparison_relation",
    )
    assert cost["score"] > 0
    uncertainty = _relation(
        "Sensitivity analysis gives a range from 5 percent to 12 percent.",
        "uncertainty_bound_relation",
    )
    assert uncertainty["score"] > 0
    time_value = _relation(
        "Using a 3 percent discount rate produces the reported net present value.",
        "time_value_relation",
    )
    assert time_value["score"] > 0
    assert notice.RELATION_DEPTHS["supported_actionable_target_graph"] == 3
    assert {
        depth for name, depth in notice.RELATION_DEPTHS.items()
        if name != "supported_actionable_target_graph"
    } == {2}


def test_filtered_action_relations_preserve_specific_function() -> None:
    corrective = notice.analyze(
        "The agency should correct the erroneous cross-reference."
    )["relations"]
    assert corrective["corrective_target_dependency"]["score"] > 0
    assert corrective["privacy_restriction_action_link"]["score"] == 0.0

    privacy = notice.analyze(
        "The portal should prohibit personally identifiable information in public comments."
    )["relations"]
    assert privacy["privacy_restriction_action_link"]["score"] > 0

    identity = notice.analyze(
        "The agency should require authentic consent for third-party submissions."
    )["relations"]
    assert identity["identity_authenticity_action_link"]["score"] > 0


def test_supported_action_graph_combines_dependency_target_with_local_evidence() -> None:
    supported = _relation(
        "EPA should reduce the fee by 25 percent because small firms cannot absorb the cost.",
        "supported_actionable_target_graph",
    )
    assert supported["score"] > 0
    types = supported["certificate"]["matches"][0]["support_types"]
    assert "causal_marker" in types
    assert "quantity" in types
    unsupported = _relation(
        "EPA should reduce the fee.",
        "supported_actionable_target_graph",
    )
    assert unsupported["score"] == 0.0


def test_empty_input_abstains_and_real_rows_are_label_free() -> None:
    empty = notice.analyze("")
    assert set(empty["relations"]) == set(notice.RELATION_DEPTHS)
    assert all(row["status"] == "abstained" for row in empty["relations"].values())

    rows = json.loads(
        (
            ROOT
            / "outputs/metric_seam_pilot/hierarchy_r123/items_v2/notice-and-comment/compiler_train.json"
        ).read_text(encoding="utf-8")
    )
    assert len(rows) == 150
    assert all(set(row) == {"item_key", "ctext"} for row in rows)
    output = notice.analyze(rows[0]["ctext"])
    assert output == notice.analyze(rows[0]["ctext"])
    assert output["parser_model"] == "en_core_web_sm"
    assert "parser" in output["parser_pipes"]
