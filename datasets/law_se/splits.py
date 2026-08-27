"""Single shared split_of(question_id) function for Law.SE.

Imported by every stage (pool build, propensity balancing, pairwise builder).
This module is the single source of truth so the partition can never drift
between stages — the same defensive pattern as CR.SE v2 / Math.SE v3.3.

Convention: md5(question_id) -> first 8 hex -> u in [0,1).
  u <  0.80           -> "train"
  0.80 <= u <  0.90   -> "eval"
  0.90 <= u           -> "test"

This matches build_v3_position_matched.py / CR.SE v2 splits.py byte-for-byte,
so Law.SE / Math.SE / CR.SE all use one identical group-split convention.
"""
from __future__ import annotations

import hashlib


TRAIN_FRAC = 0.80
EVAL_FRAC = 0.10  # implies TEST_FRAC = 0.10


def split_of(question_id, train_frac: float = TRAIN_FRAC,
             eval_frac: float = EVAL_FRAC) -> str:
    """Deterministic group split: all answers to a question land in the same
    split."""
    h = int(hashlib.md5(str(question_id).encode("utf-8")).hexdigest()[:8], 16)
    u = h / 0xFFFFFFFF
    if u < train_frac:
        return "train"
    if u < train_frac + eval_frac:
        return "eval"
    return "test"
