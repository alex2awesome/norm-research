from pathlib import Path


def test_independent_boundary_guides_do_not_leak_verifier_only_decisions() -> None:
    prompt_root = Path(__file__).parents[1] / "prompts"
    for name in ("label_peer_boundaries_v1.txt", "label_legal_boundaries_v1.txt"):
        text = (prompt_root / name).read_text()
        assert "BETTER_CANDIDATE" not in text
        assert "CONFIRM_MATCH" not in text
        assert "AMBIGUOUS_MATCH" not in text
        assert "MATCH_FAMILY_ONLY" in text
        assert "precision dominates yield" in text.lower()
