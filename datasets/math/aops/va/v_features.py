#!/usr/bin/env python3
"""Deterministic, label-blind V features for AoPS forum solutions.

Input is the FORUM POST BODY only (the problem statement is stripped by the
caller, so statement length/LaTeX cannot leak into a "solution style" feature).
The same-approach label is neither accepted nor inspected.

The nine lexicon flags are ported VERBATIM (same regexes, same names) from
`datasets/math/aops/scripts/v_features_from_tfidf.py`, which produced the
published AoPS V readout, plus that script's three deterministic surface columns
(`log_len`, `n_display_math`, `latex_density`) and a small set of additional
purely-countable features.

NOT included, deliberately: `is_correct_f`, `has_answer_f` (LLM-judge fields, not
deterministic) and `p_ed` (an out-of-fold LEARNED editorial-register score). Those
three sit in the published .706 "V" number but are not a surface channel; a V
column here is something a regex can compute from the post text alone.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List

# ---- verbatim from scripts/v_features_from_tfidf.py -------------------------
LEX = {
    "v_boxed": r"\\boxed",
    "v_hide_block": r"\[hide",
    "v_answer_stmt": r"\banswer (?:is|of)\b|\bour answer\b|\bthe answer\b",
    "v_deductive": r"\b(?:thus|therefore|hence|implies|it follows)\b",
    "v_meta_doubt": r"\b(?:wrong|incorrect|mistake|why|how do|help|stuck|hint|"
                    r"tried|guessed|confused|understand|not sure|seems)\b",
    "v_first_person": r"\b(?:i|my|me|im|i'm|i've|i think)\b",
    "v_heavy_machinery": r"\b(?:inversion|inversive|projective|pascal|desargues|"
                         r"generating function|determinant|vmatrix|"
                         r"roots? of unity filter|complex bash|"
                         r"trig bash|bary|barycentric|cauchy|schwarz|jensen|"
                         r"lagrange|induction on|pigeonhole)\b",
    "v_standard_tech": r"\b(?:right triangle|pythagorean|pythag|similar triangles?|"
                       r"casework|cases?|counting|ways|total|substitut|"
                       r"factor|symmetry|telescop)\b",
    "v_proof_framing": r"\b(?:prove|proof|q\.?e\.?d|wts|we want to show)\b|\\blacksquare",
}
_LEX_RE = {k: re.compile(v, re.I) for k, v in LEX.items()}

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
DISPLAY_RE = re.compile(r"\\\[|\\begin\{align")
IMATH_RE = re.compile(r"<imath>|</imath>")
QUOTE_BLOCK_RE = re.compile(r"\[quote", re.I)
LIST_MARK_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", re.M)
SENTENCE_END_RE = re.compile(r"[.!?]+(?:\s|$)")

V_NAMES = [
    "v_boxed", "v_hide_block", "v_answer_stmt", "v_deductive", "v_meta_doubt",
    "v_first_person", "v_heavy_machinery", "v_standard_tech", "v_proof_framing",
    "v_numeral_density", "v_question_marks",
    "v_log_len", "v_n_display_math", "v_latex_density", "v_imath_tag_count",
    "v_word_count", "v_sentence_count", "v_avg_sentence_words",
    "v_alpha_share", "v_uppercase_letter_ratio", "v_linebreak_count",
    "v_list_marker_count", "v_quote_block_count", "v_type_token_ratio",
]


def v_features(body: str) -> Dict[str, float]:
    b = body or ""
    n = len(b)
    words = WORD_RE.findall(b)
    nw = len(words)
    letters = [c for c in b if c.isalpha()]
    n_sent = max(len(SENTENCE_END_RE.findall(b)), 1 if b else 0)
    out = {k: float(len(_LEX_RE[k].findall(b))) for k in LEX}
    out["v_numeral_density"] = float(sum(c.isdigit() for c in b) / (n + 1))
    out["v_question_marks"] = float(b.count("?"))
    out["v_log_len"] = float(math.log1p(n))
    out["v_n_display_math"] = float(len(DISPLAY_RE.findall(b)))
    out["v_latex_density"] = float(b.count("$") / (n + 1))
    out["v_imath_tag_count"] = float(len(IMATH_RE.findall(b)))
    out["v_word_count"] = float(nw)
    out["v_sentence_count"] = float(n_sent)
    out["v_avg_sentence_words"] = float(nw / max(n_sent, 1))
    out["v_alpha_share"] = float(len(letters) / (n + 1))
    out["v_uppercase_letter_ratio"] = float(
        sum(c.isupper() for c in letters) / max(len(letters), 1))
    out["v_linebreak_count"] = float(b.count("\n"))
    out["v_list_marker_count"] = float(len(LIST_MARK_RE.findall(b)))
    out["v_quote_block_count"] = float(len(QUOTE_BLOCK_RE.findall(b)))
    lows = [w.lower() for w in words]
    out["v_type_token_ratio"] = float(len(set(lows)) / max(nw, 1))
    ordered = {k: out[k] for k in V_NAMES}
    assert list(ordered) == V_NAMES
    assert all(math.isfinite(v) for v in ordered.values())
    return ordered


def vector(body: str, names: Iterable[str] = V_NAMES) -> List[float]:
    vals = v_features(body)
    return [vals[k] for k in names]


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    print(json.dumps(v_features(ap.parse_args().text), indent=2))
