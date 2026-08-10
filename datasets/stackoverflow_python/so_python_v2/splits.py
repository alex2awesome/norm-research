"""Single shared split_of(question_id) function for SO[python] v2.

Imported by every stage (pool build, balancing, floor tests, pairwise builder).
Identical convention to CR.SE v2 (datasets/code-review/crse_balanced_v2/splits.py):

  md5(question_id) -> first 8 hex -> u in [0,1).
    u <  0.80           -> "train"
    0.80 <= u <  0.90   -> "eval"
    0.90 <= u           -> "test"
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
