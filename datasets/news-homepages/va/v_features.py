#!/usr/bin/env python3
"""Deterministic, label-blind V features for homepage-curation items.

Input is the HEADLINE only (the CONTEXT field -- the other headlines on the same
capture -- is deliberately excluded: it is shared across every row of a snapshot,
so any feature computed on it is a snapshot-identity feature, not a property of
the item). The placement label is neither accepted nor inspected.

Deliberately conservative: this is the nuisance channel, so every column is a
count or ratio a regex can compute from the headline string.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
CAPWORD_RE = re.compile(r"\b[A-Z][a-z']+\b")
ALLCAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
QUOTE_RE = re.compile(r"[\"“”‘’']")
COLON_DASH_RE = re.compile(r"[:—–|]")
MONEY_RE = re.compile(r"[$£€]\s?\d|\b\d+\s?(?:billion|million|trillion|percent|%)", re.I)
FUNCTION_WORDS = {"the", "a", "an", "of", "in", "to", "for", "on", "and", "is",
                  "at", "with", "as", "by", "from", "that", "it", "its", "his",
                  "her", "their"}
HEDGE_RE = re.compile(r"\b(?:may|might|could|reportedly|allegedly|appears?|"
                      r"expected to|set to|is said to)\b", re.I)
FUTURE_RE = re.compile(r"\b(?:will|to (?:be|hold|meet|face|visit|announce)|ahead of|"
                       r"next (?:week|month|year))\b", re.I)
WHQ_RE = re.compile(r"^\s*(?:what|why|how|who|when|where|which|is|are|do|does|can)\b", re.I)

V_NAMES = [
    "v_char_count", "v_word_count", "v_avg_word_length", "v_type_token_ratio",
    "v_capitalized_word_count", "v_capitalized_word_ratio",
    "v_all_caps_token_count", "v_uppercase_letter_ratio",
    "v_number_count", "v_money_or_magnitude_count", "v_digit_ratio",
    "v_quote_mark_count", "v_colon_dash_count", "v_comma_count",
    "v_question_mark", "v_exclamation_mark", "v_apostrophe_count",
    "v_function_word_ratio", "v_hedge_marker_count", "v_future_marker_count",
    "v_interrogative_opening", "v_long_word_ratio", "v_flesch_reading_ease",
]


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    g = len(re.findall(r"[aeiouy]+", w))
    if len(w) > 3 and w.endswith("e") and not w.endswith(("le", "ye")):
        g -= 1
    return max(g, 1)


def headline_of(text: str) -> str:
    """Split the stored item text; V is computed on the HEADLINE half only."""
    t = str(text or "")
    head = t.split("\n\nCONTEXT: ", 1)[0]
    return head.removeprefix("HEADLINE: ").strip()


def v_features(headline: str) -> Dict[str, float]:
    h = headline or ""
    n = len(h)
    words = WORD_RE.findall(h)
    nw = len(words)
    lows = [w.lower() for w in words]
    letters = [c for c in h if c.isalpha()]
    syll = sum(_syllables(w) for w in words)
    caps = len(CAPWORD_RE.findall(h))
    out = {
        "v_char_count": float(n),
        "v_word_count": float(nw),
        "v_avg_word_length": float(sum(len(w) for w in words) / max(nw, 1)),
        "v_type_token_ratio": float(len(set(lows)) / max(nw, 1)),
        "v_capitalized_word_count": float(caps),
        "v_capitalized_word_ratio": float(caps / max(nw, 1)),
        "v_all_caps_token_count": float(len(ALLCAPS_RE.findall(h))),
        "v_uppercase_letter_ratio": float(
            sum(c.isupper() for c in letters) / max(len(letters), 1)),
        "v_number_count": float(len(NUMBER_RE.findall(h))),
        "v_money_or_magnitude_count": float(len(MONEY_RE.findall(h))),
        "v_digit_ratio": float(sum(c.isdigit() for c in h) / (n + 1)),
        "v_quote_mark_count": float(len(QUOTE_RE.findall(h))),
        "v_colon_dash_count": float(len(COLON_DASH_RE.findall(h))),
        "v_comma_count": float(h.count(",")),
        "v_question_mark": float(h.count("?")),
        "v_exclamation_mark": float(h.count("!")),
        "v_apostrophe_count": float(h.count("'") + h.count("’")),
        "v_function_word_ratio": float(
            sum(w in FUNCTION_WORDS for w in lows) / max(nw, 1)),
        "v_hedge_marker_count": float(len(HEDGE_RE.findall(h))),
        "v_future_marker_count": float(len(FUTURE_RE.findall(h))),
        "v_interrogative_opening": float(bool(WHQ_RE.match(h))),
        "v_long_word_ratio": float(sum(len(w) >= 8 for w in words) / max(nw, 1)),
        "v_flesch_reading_ease": float(
            206.835 - 1.015 * nw - 84.6 * (syll / max(nw, 1))) if nw else 0.0,
    }
    assert list(out) == V_NAMES
    assert all(math.isfinite(v) for v in out.values())
    return out


def vector(headline: str, names: Iterable[str] = V_NAMES) -> List[float]:
    vals = v_features(headline)
    return [vals[k] for k in names]


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    print(json.dumps(v_features(headline_of(ap.parse_args().text)), indent=2))
