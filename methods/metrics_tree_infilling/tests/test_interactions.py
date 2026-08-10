"""Unit tests for the composite/interaction screen (§9 fix core)."""

from __future__ import annotations

import numpy as np

from methods.metrics_tree_infilling.interactions import (
    RULES, apply_rule, best_combination, binarize, combine_with_rule,
)


def test_xor_is_recovered_where_neither_primitive_is_marginal():
    """The §9 case: label = A xor B; A and B each ~50% and carry no marginal signal."""
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.integers(0, 2, n).astype(float)
    b = rng.integers(0, 2, n).astype(float)
    y = (a.astype(bool) ^ b.astype(bool)).astype(int)

    rule, acc = best_combination(a, b, y)
    # marginals are ~chance, but xor is perfect -> a genuine interaction is found
    assert rule == "xor", f"expected xor, got {rule} (acc {acc})"
    assert acc > 0.99, acc
    # sanity: each primitive alone is ~chance
    assert max(float((binarize(a) == y).mean()), float((binarize(b) == y).mean())) < 0.55


def test_main_effect_is_not_reported_as_interaction():
    """When the label IS a single primitive, no rule should be flagged (margin not beaten)."""
    rng = np.random.default_rng(1)
    n = 2000
    a = rng.integers(0, 2, n).astype(float)
    b = rng.integers(0, 2, n).astype(float)
    y = a.astype(int)                       # pure main effect of a
    rule, acc = best_combination(a, b, y)
    assert rule is None, f"flagged {rule} for a pure main effect (acc {acc})"
    assert acc > 0.95                        # the marginal already nails it


def test_anti_correlated_main_effect_not_an_interaction():
    """y = NOT-a is still a (inverted) main effect, not an interaction — must not be flagged."""
    rng = np.random.default_rng(3)
    n = 2000
    a = rng.integers(0, 2, n).astype(float)
    b = rng.integers(0, 2, n).astype(float)
    y = 1 - a.astype(int)
    rule, acc = best_combination(a, b, y)
    assert rule is None, f"flagged {rule} for an inverted main effect (acc {acc})"
    assert acc > 0.95


def test_and_not_b_recovered():
    rng = np.random.default_rng(2)
    n = 2000
    a = rng.integers(0, 2, n).astype(float)
    b = rng.integers(0, 2, n).astype(float)
    y = ((a == 1) & (b == 0)).astype(int)   # a AND NOT b
    rule, acc = best_combination(a, b, y)
    assert rule == "a_not_b", (rule, acc)
    assert acc > 0.99


def test_xnor_is_recovered_via_inverted_polarity():
    """y = NOT(A xor B) (XNOR): the raw xor column is anti-correlated with y, but its complement
    is perfect. best_combination must credit the better polarity and return xor (the GLM inverts),
    not drop it as no_interaction. Regression for the inverted-composite bug (Codex #4)."""
    rng = np.random.default_rng(7)
    n = 2000
    a = rng.integers(0, 2, n).astype(float)
    b = rng.integers(0, 2, n).astype(float)
    y = (~(a.astype(bool) ^ b.astype(bool))).astype(int)   # XNOR
    rule, acc = best_combination(a, b, y)
    assert rule == "xor", f"expected xor (inverted by GLM), got {rule} (acc {acc})"
    assert acc > 0.99, acc
    # marginals are ~chance
    assert max(float((binarize(a) == y).mean()), float((binarize(b) == y).mean())) < 0.55


def test_combine_with_rule_roundtrip():
    a = np.array([1.0, 0.0, 1.0, 0.0])
    b = np.array([1.0, 1.0, 0.0, 0.0])
    # xor: 0,1,1,0
    assert list(combine_with_rule(a, b, "xor")) == [0, 1, 1, 0]
    # and: 1,0,0,0
    assert list(combine_with_rule(a, b, "and")) == [1, 0, 0, 0]
    # nan-safe
    assert list(binarize(np.array([np.nan, 0.6]))) == [0, 1]


def test_all_rules_have_opposites_where_applicable():
    # xor is its own inverse-ish; a_not_b vs b_not_a are distinct
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 1, 0])
    assert list(apply_rule(a, b, "a_not_b")) == [0, 1, 0, 0]
    assert list(apply_rule(a, b, "b_not_a")) == [0, 0, 1, 0]
    assert len(RULES) == 5
