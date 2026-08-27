from scripts.tools.silver_match_v3.prepare_boundary_enriched_teacher_pack import (
    legal_stratum,
    peer_stratum,
)


def test_peer_boundary_strata_cover_audited_confusions() -> None:
    assert peer_stratum({"norm": "Please clarify the experimental method"}) == "methods_transparency_rigor"
    assert peer_stratum({"norm": "The theoretical claims lack evidence"}) == "evidence_theory_correctness"
    assert peer_stratum({"norm": "There is only one dataset"}) == "generalization_sampling"
    assert peer_stratum({"norm": "The problem is interesting and important"}) == "question_significance_motivation"


def test_legal_boundary_strata_cover_audited_confusions() -> None:
    assert legal_stratum({"norm": "the discharge was unlawful"}) == "bare_holding_or_substantive_rule"
    assert legal_stratum({"norm": "not supported by substantial evidence"}) == "accuracy_evidence_record"
    assert legal_stratum({"norm": "the term does not mean cyber security information"}) == "terminology_definition"
    assert legal_stratum({"norm": "the overlooked standing issue was waived"}) == "issue_analogy_preservation"
