"""Construct-derived adversarial and metamorphic tests for symbolic steps."""

from methods.metric_seam.hybrids.ops_symbolic_steps_v1 import (
    analyze_document,
    verify_expression_pair,
)


def test_polynomial_expansion_is_exact_positive_witness():
    row = verify_expression_pair(r"(x+1)^2", r"x^2+2x+1")
    assert row["status"] == "verified_rational_identity"
    assert row["positive_code_witness"] is True
    assert row["domain_nonzero_obligations"] == []


def test_variable_renaming_and_side_reversal_are_metamorphic_invariants():
    first = verify_expression_pair(r"a\cdot(b+c)", r"a\cdot b+a\cdot c")
    renamed = verify_expression_pair(r"u\cdot(v+w)", r"u\cdot v+u\cdot w")
    reversed_pair = verify_expression_pair(r"a\cdot b+a\cdot c", r"a\cdot(b+c)")
    assert {first["status"], renamed["status"], reversed_pair["status"]} == {
        "verified_rational_identity"
    }


def test_cancellation_certificate_surfaces_its_domain_obligation():
    row = verify_expression_pair(r"\frac{x^2-1}{x-1}", r"x+1")
    assert row["status"] == "verified_rational_identity"
    assert row["domain_nonzero_obligations"] == ["x - 1 != 0"]


def test_false_universal_identity_gets_exact_counterexample():
    row = verify_expression_pair(
        r"(x+1)^2", r"x^2+1", declared_universal_scope=True
    )
    assert row["status"] == "universal_identity_counterexample"
    assert row["counterexample_assignment"]
    assert row["criterion_defect_witness"] is True


def test_same_nonidentity_is_not_a_document_defect_without_claim_scope():
    row = verify_expression_pair(r"x", r"1")
    assert row["status"] == "exact_nonidentity_witness"
    assert row["positive_code_witness"] is True
    assert row["claim_scope_required"] is True
    assert row["criterion_defect_witness"] is False


def test_branch_sensitive_transcendental_identity_abstains():
    row = verify_expression_pair(r"\sin^2 x+\cos^2 x", r"1")
    assert row["status"] == "parse_noncoverage"
    assert row["positive_code_witness"] is False


def test_malformed_latex_abstains_instead_of_becoming_negative_evidence():
    row = verify_expression_pair(r"\frac{x+1", r"x")
    assert row["status"] == "parse_noncoverage"
    assert row["criterion_defect_witness"] is False


def test_document_analysis_uses_answer_and_emits_no_whole_scalar():
    text = (
        "Question: Is $x=7$?\n\n"
        "Answer: For every $x$, the algebraic step "
        "$ (x+1)^2=x^2+2x+1 $ is an identity."
    )
    row = analyze_document(text)
    assert row["verified_rational_identity_count"] == 1
    assert row["whole_criterion_fidelity"] == "UNAVAILABLE"
    assert row["whole_criterion_scalar"] is None


def test_document_analysis_never_infers_universal_scope_from_an_equation():
    row = analyze_document("Question: Solve.\n\nAnswer: $x=1$.")
    assert row["exact_nonidentity_witness_count"] == 1
    assert row["universal_identity_counterexample_count"] == 0
    assert row["criterion_defect_witness_count"] == 0


def test_no_executable_pair_is_explicit_abstention():
    row = analyze_document("Question: Why?\n\nAnswer: The theorem applies by compactness.")
    assert row["abstained"] is True
    assert row["positive_code_witness_count"] == 0


def test_equation_chain_checks_every_adjacent_step():
    row = analyze_document(
        r"Question: Expand.\n\nAnswer: $ (x+1)^2=x^2+2x+1=x\cdot(x+2)+1 $."
    )
    assert row["pair_candidate_count"] == 2
    assert row["verified_rational_identity_count"] == 2
