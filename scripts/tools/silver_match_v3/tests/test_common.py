from scripts.tools.silver_match_v3.common import (
    extract_norm,
    norm_query,
    norm_query_views,
    normalize_name,
    normalize_space,
    stable_uid,
)


def test_normalize_name_typographic_variants():
    assert normalize_name("Setup–surprise mechanics") == normalize_name(
        "Setup-surprise  mechanics"
    )


def test_stable_uid_is_row_sensitive():
    assert stable_uid("x", 0, "doc", "norm") != stable_uid("x", 1, "doc", "norm")


def test_extract_norm_accepts_new_peer_review_schema():
    assert extract_norm({"signal_text": "  needs   clearer methods "}) == "needs clearer methods"


def test_invalid_surrogate_is_replaced():
    assert normalize_space("bad\ud83d text") == "bad? text"


def test_norm_query_includes_evidence_and_places_weak_hint_last():
    value = norm_query(
        {
            "task": "code-review",
            "norm": "This should be cached.",
            "context": "This should be cached because the call is repeated in the loop.",
            "aspect": "efficiency",
        }
    )
    assert "Evidence passage:" in value
    assert value.index("Human evaluative statement:") < value.index("Evidence passage:")
    assert value.index("Evidence passage:") < value.index("Weak extraction aspect hint:")


def test_norm_query_does_not_duplicate_identical_context():
    value = norm_query(
        {"task": "humor", "norm": "The ending drags.", "context": "The ending drags."}
    )
    assert "Evidence passage:" not in value
    assert len(norm_query_views({"task": "humor", "norm": "The ending drags."})) == 1
