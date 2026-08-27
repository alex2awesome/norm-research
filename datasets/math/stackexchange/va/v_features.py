#!/usr/bin/env python3
"""Deterministic, label-blind V features for math.stackexchange ANSWERS
(math.SE V2 multi-y rebuild).

Input is the answer body only (the question title is stripped by the caller, so
question length cannot leak into an "answer style" feature). Neither the accept
verdict nor the vote score is accepted or inspected.

Every column is a surface count/ratio a regex can compute. Lexicon families
mirror the AoPS V module (`datasets/math/aops/va/v_features.py`) so the two math
cells share a comparable surface channel; the math.SE-specific additions are the
answer-register markers this corpus actually carries (hedging, references to
sources, edit/meta markers, enumerated structure).
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List

LEX = {
    "v_deductive": r"\b(?:thus|therefore|hence|implies|it follows|consequently|so that)\b",
    "v_proof_framing": r"\b(?:prove|proof|q\.?e\.?d|we want to show|suffices to show|"
                       r"claim|lemma|theorem)\b|\\blacksquare|\\square",
    "v_hedging": r"\b(?:i think|maybe|perhaps|probably|not sure|might be|seems|"
                 r"i believe|if i(?:'m| am) not mistaken)\b",
    "v_first_person": r"\b(?:i|my|me|i'm|i've)\b",
    "v_second_person": r"\b(?:you|your|you're|you've)\b",
    "v_imperative_hint": r"\b(?:hint|note that|observe that|recall that|consider|"
                         r"try|notice)\b",
    "v_reference_pointer": r"\b(?:see|cf\.|refer to|wikipedia|textbook|as in|"
                           r"this answer|this question|link)\b|https?://",
    "v_heavy_machinery": r"\b(?:cauchy|schwarz|jensen|lagrange|fubini|zorn|"
                         r"banach|hahn|residue theorem|generating function|"
                         r"determinant|induction on|pigeonhole|compactness|"
                         r"galois|tensor|functor)\b",
    "v_standard_tech": r"\b(?:substitut|integrat|differentiat|factor|expand|"
                       r"telescop|symmetry|casework|by parts|change of variable|"
                       r"counting|combinator)\b",
    "v_generality_marker": r"\b(?:in general|more generally|for all|for every|"
                           r"arbitrary|any such)\b",
    "v_edge_case_marker": r"\b(?:edge case|boundary|degenerate|trivial case|"
                          r"assume(?:s|d)? (?:that )?(?:x|n|a)?\s*(?:>|<|=|\\neq)|"
                          r"provided that|as long as)\b",
    "v_meta_edit": r"\b(?:edit|update|added|corrected|typo|oops|thanks)\b",
}
_LEX_RE = {k: re.compile(v, re.I) for k, v in LEX.items()}

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
DISPLAY_RE = re.compile(r"\$\$|\\\[|\\begin\{(?:align|equation|gather|array)")
INLINE_MATH_RE = re.compile(r"\$")
SENTENCE_END_RE = re.compile(r"[.!?]+(?:\s|$)")
LIST_MARK_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", re.M)
BACKSLASH_CMD_RE = re.compile(r"\\[A-Za-z]+")

V_NAMES = list(LEX) + [
    "v_log_len", "v_word_count", "v_sentence_count", "v_avg_sentence_words",
    "v_n_display_math", "v_inline_math_delims", "v_latex_cmd_count",
    "v_latex_density", "v_numeral_density", "v_alpha_share",
    "v_question_marks", "v_uppercase_letter_ratio", "v_linebreak_count",
    "v_paragraph_count", "v_list_marker_count", "v_type_token_ratio",
]


def v_features(body: str) -> Dict[str, float]:
    b = body or ""
    n = len(b)
    words = WORD_RE.findall(b)
    nw = len(words)
    letters = [c for c in b if c.isalpha()]
    n_sent = max(len(SENTENCE_END_RE.findall(b)), 1 if b else 0)
    out = {k: float(len(_LEX_RE[k].findall(b))) for k in LEX}
    out["v_log_len"] = float(math.log1p(n))
    out["v_word_count"] = float(nw)
    out["v_sentence_count"] = float(n_sent)
    out["v_avg_sentence_words"] = float(nw / max(n_sent, 1))
    out["v_n_display_math"] = float(len(DISPLAY_RE.findall(b)))
    out["v_inline_math_delims"] = float(len(INLINE_MATH_RE.findall(b)))
    out["v_latex_cmd_count"] = float(len(BACKSLASH_CMD_RE.findall(b)))
    out["v_latex_density"] = float(len(BACKSLASH_CMD_RE.findall(b)) / (nw + 1))
    out["v_numeral_density"] = float(sum(c.isdigit() for c in b) / (n + 1))
    out["v_alpha_share"] = float(len(letters) / (n + 1))
    out["v_question_marks"] = float(b.count("?"))
    out["v_uppercase_letter_ratio"] = float(
        sum(c.isupper() for c in letters) / max(len(letters), 1))
    out["v_linebreak_count"] = float(b.count("\n"))
    out["v_paragraph_count"] = float(b.count("\n\n") + 1)
    out["v_list_marker_count"] = float(len(LIST_MARK_RE.findall(b)))
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
