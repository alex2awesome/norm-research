from __future__ import annotations

from methods.metric_seam.verifiers.math_a12_symbolic import (
    ablate_witness_lines,
    extract_equality_pairs,
    verify_pair,
)


def test_extracts_file_qualified_adjacent_pairs() -> None:
    text = "Question: simplify\n\nAnswer:\nWe have $$x+x=2x=3x.$$\nDone."
    pairs = extract_equality_pairs(text, item_key="train_0001")
    assert [(pair.lhs, pair.rhs) for pair in pairs] == [("x+x", "2x"), ("2x", "3x")]
    assert all(pair.witness.path == "answer.md" for pair in pairs)
    assert all(pair.witness.start_line == 4 for pair in pairs)
    assert len({pair.pair_id for pair in pairs}) == 2


def test_symbolic_verdict_has_three_states_and_load_bearing_witness() -> None:
    text = "Answer:\n$$x+x=2x$$\n$$x+x=3x$$\n$$f(x)=x$$"
    pairs = extract_equality_pairs(text, item_key="train_0002")
    values = [verify_pair(pair) for pair in pairs]
    assert values[0].applies and not values[0].violated
    assert values[1].applies and values[1].violated
    assert not values[2].applies
    assert values[0].witnesses == (pairs[0].witness,)
    assert values[1].witnesses == (pairs[1].witness,)
    assert not values[2].witnesses

    ablated = ablate_witness_lines(text, values[1].witnesses)
    remaining = extract_equality_pairs(ablated, item_key="train_0002")
    assert all(pair.pair_id != pairs[1].pair_id for pair in remaining)


def test_pair_request_value_contains_no_score_or_confidence() -> None:
    pair = extract_equality_pairs("Answer:\n$$x=x$$", item_key="train_0003")[0]
    value = pair.to_request_value()
    assert set(value) == {"item_key", "pair_id", "lhs", "rhs", "source_span"}
    assert "score" not in str(value).lower()
    assert "confidence" not in str(value).lower()
