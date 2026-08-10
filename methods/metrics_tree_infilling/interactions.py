"""Composite / interaction features — the §9 limitation fix.

The base loop discovers missing MAIN effects only: one feature, materialized, reinserted. It
provably misses an **interaction of absent features** (a root XOR) — two textual properties
neither of which has marginal label signal, but whose boolean combination fully determines the
label. In such a gap every single-feature proposal fails to separate (the contrast looks like
noise), so the loop gives up.

This module closes that gap with a **composite**: up to two primitive features combined by a
boolean rule (and / or / xor / a-and-not-b / b-and-not-a). The rule is fit on data
(``best_combination``), not trusted from the proposer. Pure numpy — no LLM — so the capability is
unit-testable on a planted XOR.

The live path: ``feature_gen.propose_composite_feature`` asks the LLM for the primitives + a
candidate rule; ``loop.run_infill`` (when ``cfg.enable_composite_proposer``) materializes the
primitives, fits the best rule on discover, applies it on test, and reinserts the composite
column through the normal guards.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

RULES = ("and", "or", "xor", "a_not_b", "b_not_a")


def binarize(col: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """NaN-safe threshold to {0,1} (NaN / non-applicable -> 0)."""
    return (np.nan_to_num(col, nan=0.0) > thr).astype(int)


def apply_rule(a: np.ndarray, b: np.ndarray, rule: str) -> np.ndarray:
    """Combine two binary {0,1} columns by ``rule`` -> binary column."""
    a = a.astype(bool)
    b = b.astype(bool)
    if rule == "and":
        return (a & b).astype(int)
    if rule == "or":
        return (a | b).astype(int)
    if rule == "xor":
        return (a ^ b).astype(int)
    if rule == "a_not_b":
        return (a & ~b).astype(int)
    if rule == "b_not_a":
        return (b & ~a).astype(int)
    raise ValueError(f"unknown rule {rule!r}")


def _acc(yhat: np.ndarray, y: np.ndarray) -> float:
    return float((yhat == y).mean()) if len(y) else 0.0


def best_combination(a: np.ndarray, b: np.ndarray, y: np.ndarray,
                     margin: float = 0.02) -> Tuple[Optional[str], float]:
    """Best boolean rule combining primitive columns ``a``, ``b`` to predict ``y``.

    Returns ``(rule, accuracy)`` where ``rule`` is non-None ONLY if some combination beats *both*
    marginal single-column accuracies by ``margin`` — i.e. it is a genuine interaction, not a
    main effect the base loop would already find. Columns are binarized at 0.5 first.
    """
    ab = binarize(a)
    bb = binarize(b)
    # baseline: the best any SINGLE primitive does, in EITHER polarity (an inverted main effect
    # y = 1-a is still a main effect, not an interaction)
    base = max(_acc(ab, y), _acc(1 - ab, y), _acc(bb, y), _acc(1 - bb, y))
    best_rule, best_acc = None, base
    for r in RULES:
        col = apply_rule(ab, bb, r)
        # The within-node GLM can invert any composite through a negative coefficient, so credit
        # the better of the two polarities. This recovers XNOR (= NOT xor), NAND, NOR etc., where
        # the raw rule column is anti-correlated with y but its complement is perfect — without
        # this, such a gap looks like noise and the composite is dropped at the no_interaction gate.
        acc = max(_acc(col, y), _acc(1 - col, y))
        if acc > best_acc:
            best_rule, best_acc = r, acc
    if best_rule is not None and best_acc >= base + margin:
        return best_rule, best_acc
    return None, base


def combine_with_rule(a: np.ndarray, b: np.ndarray, rule: Optional[str]) -> np.ndarray:
    """Apply a chosen ``rule`` (or, if None, the better marginal) to produce a level column."""
    ab = binarize(a)
    bb = binarize(b)
    if rule is None:
        return ab if ab.sum() >= bb.sum() else bb
    return apply_rule(ab, bb, rule)
