from __future__ import annotations

import json
from pathlib import Path

from methods.metric_seam import grant_structure_v1 as grant


ROOT = Path(__file__).resolve().parents[2]


def _relation(text: str, name: str) -> dict:
    return grant.analyze(text)["relations"][name]


def test_budget_arithmetic_checks_itemized_sum_and_abstains_without_total() -> None:
    exact = """
    Budget
    • $70,000 for scholarships
    • $65,000 for tutoring
    • $9,000 for computers
    • $7,500 for supplies
    This $151,500 investment will support the project.
    """
    result = _relation(exact, "budget_sum_consistency")
    assert result["status"] == "measured"
    assert result["score"] == 1.0
    assert result["certificate"]["n_items"] == 4

    mismatch = exact.replace("$151,500 investment", "$140,000 investment")
    mismatch_result = _relation(mismatch, "budget_sum_consistency")
    assert mismatch_result["status"] == "measured"
    assert 0.0 <= mismatch_result["score"] < 1.0
    assert mismatch_result["certificate"]["relative_error"] > 0

    abstained = _relation("Budget includes $50,000 for staff.", "budget_sum_consistency")
    assert abstained["status"] == "abstained"
    assert abstained["score"] is None


def test_budget_selects_last_checkable_total_not_best_fitting_total() -> None:
    text = """
    - $60 for activity A
    - $40 for activity B
    Total: $100.
    - $30 for activity C
    - $30 for activity D
    Final total: $100.
    """
    result = _relation(text, "budget_sum_consistency")
    assert result["certificate"]["selection_rule"] == (
        "last checkable stated total in ctext order"
    )
    assert result["certificate"]["declared_total"] == 100.0
    assert result["score"] < 1.0


def test_citation_relation_requires_a_linked_claim_not_bibliography_presence() -> None:
    linked = _relation(
        "Prior work showed a 30% reduction (Smith et al., 2022).",
        "citation_claim_link",
    )
    assert linked["score"] > 0
    bibliography = _relation(
        "References\nSmith, J. (2022). A useful paper.",
        "citation_claim_link",
    )
    assert bibliography["status"] == "measured"
    assert bibliography["score"] == 0.0


def test_deep_relations_require_local_relational_pairs() -> None:
    risk = _relation(
        "Recruitment attrition is a major risk. If enrollment is slow, we will add a second site.",
        "risk_mitigation_graph",
    )
    assert risk["score"] > 0
    unpaired = _relation(
        "Recruitment attrition is a major risk.\n" + "Unrelated filler. " * 10 + "We have alternatives.",
        "risk_mitigation_graph",
    )
    assert unpaired["score"] == 0.0

    hypothesis = _relation(
        "We hypothesize that tutoring improves retention. Aim 1 tests this prediction in 80 students.",
        "aim_hypothesis_experiment_graph",
    )
    assert hypothesis["score"] > 0


def test_quantified_gap_handles_percentages_without_not_only_false_positive() -> None:
    true_gap = _relation(
        "Students have a 90% pass rate compared with the national average of 19%.",
        "quantified_need_gap",
    )
    assert true_gap["score"] > 0
    not_only = _relation(
        "The consortium supported more than 4,000 projects, transforming not only policy but practice.",
        "quantified_need_gap",
    )
    assert not_only["score"] == 0.0


def test_role_and_schedule_relations_reject_shared_token_and_title_year_collisions() -> None:
    software = _relation(
        "Package managers provide search functions for software repositories.",
        "role_responsibility_graph",
    )
    assert software["score"] == 0.0
    assigned = _relation(
        "The project manager will oversee recruitment and reporting.",
        "role_responsibility_graph",
    )
    assert assigned["score"] > 0

    title_year = _relation(
        "Application for the Open Science Fellowship 2024. The project aims to map software.",
        "schedule_dependency_graph",
    )
    assert title_year["score"] == 0.0
    scheduled = _relation(
        "By 2025, Aim 1 will complete recruitment and begin analysis.",
        "schedule_dependency_graph",
    )
    assert scheduled["score"] > 0


def test_front_matter_and_outline_are_structural_not_whole_quality_judgments() -> None:
    text = """
    1. Executive Summary
    The problem is low retention. We propose a mentoring program for 100 students
    that will improve graduation outcomes.
    2. Project Description
    3. Budget and Justification
    4. Evaluation
    """
    result = grant.analyze(text)
    assert result["relations"]["front_matter_coverage"]["score"] == 1.0
    assert result["relations"]["document_outline_structure"]["score"] > 0
    assert "whole_construct_score" not in result


def test_program_emits_all_declared_relations_and_empty_input_abstains() -> None:
    result = grant.analyze("")
    assert result["schema"] == grant.SCHEMA
    assert set(result["relations"]) == set(grant.RELATION_DEPTHS)
    assert all(row["status"] == "abstained" for row in result["relations"].values())


def test_real_grant_ctext_smoke_is_deterministic_and_label_free() -> None:
    rows = json.loads(
        (
            ROOT
            / "outputs/metric_seam_pilot/hierarchy_r123/items_v2/grant-funding/compiler_train.json"
        ).read_text(encoding="utf-8")
    )
    assert len(rows) == 103
    first = grant.analyze(rows[0]["ctext"])
    assert first == grant.analyze(rows[0]["ctext"])
    assert first["relations"]["budget_sum_consistency"]["score"] == 1.0
    assert all(set(row) == {"item_key", "ctext"} for row in rows)
