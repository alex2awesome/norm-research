"""Active coding census a104 h0: automated-test presence and design quality.

This is not the legacy ``f2p_mock`` a104.  It is an outcome-blind seed for
the current 250-item diff-only census.  Presence, test/source balance,
test-to-changed-symbol correspondence, and assertion structure are separate
code-verifiable sub-relations.  Whether assertions express the intended
behaviour remains outside this code-only h0.
"""

LLM_FIELDS = {}
PROGRAM_PROVENANCE = "active_code_review_census_outcome_blind_cpu_v1"
SUBRELATIONS = {
    "test_presence": "code_verifiable",
    "test_source_balance": "code_verifiable_proxy",
    "test_to_changed_symbol_correspondence": "code_verifiable",
    "assertion_structure": "code_verifiable_proxy",
    "behavioural_intent_and_oracle_validity": "unresolved_without_execution_or_prompt",
}


def score(text: str, extracted: dict, ops) -> float:
    try:
        profile = ops.test_design_profile(text)
        if not profile["source_files"] and not profile["test_files"]:
            return 0.5
        if profile["test_files"] and not profile["source_files"]:
            return 0.8
        value = (
            0.05
            + 0.20 * profile["presence"]
            + 0.15 * profile["line_balance"]
            + 0.40 * profile["correspondence"]
            + 0.20 * profile["assertion_density"]
        )
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.5

