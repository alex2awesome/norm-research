#!/usr/bin/env python3
"""Deterministic, label-blind V features for Style Invitational entries.

Only ``entry_text`` and ``contest_prompt`` are accepted.  The functions do not
read tiers, week identifiers, author histories, or any fitted corpus state.
Pronunciation-like features use transparent orthographic approximations so the
file remains runnable without third-party NLP packages.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Mapping

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)?")
ALPHA_RE = re.compile(r"[A-Za-z]")
SENTENCE_RE = re.compile(r"[.!?]+(?:[\"'’”)]*)")
PUNCT_RE = re.compile(r"[^\w\s]")
VOWEL_RUN_RE = re.compile(r"[aeiouy]+")

V_NAMES = [
    "v_char_count",
    "v_word_count",
    "v_avg_word_length",
    "v_sentence_count",
    "v_line_count",
    "v_punctuation_density",
    "v_question_count",
    "v_exclamation_count",
    "v_digit_count",
    "v_uppercase_ratio",
    "v_type_token_ratio",
    "v_hapax_ratio",
    "v_flesch_reading_ease",
    "v_repeated_token_ratio",
    "v_adjacent_alliteration_rate",
    "v_internal_rhyme_rate",
    "v_end_rhyme_rate",
    "v_prompt_token_jaccard",
    "v_prompt_longest_shared_run",
]


def words(text: str) -> list[str]:
    return [m.group(0).lower().replace("’", "'") for m in WORD_RE.finditer(text or "")]


def _alpha_words(text: str) -> list[str]:
    return [w for w in words(text) if any(c.isalpha() for c in w)]


def _syllables(word: str) -> int:
    """A deterministic English syllable approximation used only for readability."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = len(VOWEL_RUN_RE.findall(w))
    if len(w) > 3 and w.endswith("e") and not w.endswith(("le", "ye")):
        groups -= 1
    if len(w) > 4 and w.endswith("es") and not w.endswith(("aes", "ees", "oes")):
        groups -= 1
    return max(groups, 1)


def _rhyme_key(word: str) -> str:
    """Return an orthographic final-vowel rhyme key, or an empty key."""
    w = re.sub(r"[^a-z]", "", word.lower())
    matches = list(VOWEL_RUN_RE.finditer(w))
    if not matches:
        return ""
    start = matches[-1].start()
    key = w[start:]
    return key if len(key) >= 2 else ""


def _longest_shared_run(a: list[str], b: list[str]) -> int:
    """Longest contiguous shared token run, using O(min(n,m)) memory."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    best = 0
    for aw in a:
        cur = [0] * (len(b) + 1)
        for j, bw in enumerate(b, 1):
            if aw == bw:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def _line_end_words(text: str) -> list[str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) <= 1:
        # Commas and semicolons often preserve light-verse line endings after
        # plain-text archive parsing has flattened original line breaks.
        lines = [x.strip() for x in re.split(r"[,;]\s+|[.!?]+\s+", text or "") if x.strip()]
    out = []
    for line in lines:
        ws = _alpha_words(line)
        if ws:
            out.append(ws[-1])
    return out


def v_features(entry_text: str, contest_prompt: str) -> dict[str, float]:
    text = entry_text or ""
    prompt = contest_prompt or ""
    toks = words(text)
    alpha = _alpha_words(text)
    ntok = len(toks)
    nalpha = len(alpha)
    counts = Counter(toks)

    sentence_count = max(len(SENTENCE_RE.findall(text)), 1 if text.strip() else 0)
    line_count = len([x for x in text.splitlines() if x.strip()]) or (1 if text.strip() else 0)
    letters = ALPHA_RE.findall(text)
    uppercase = sum(c.isupper() for c in letters)
    syllables = sum(_syllables(w) for w in alpha)
    flesch = (
        206.835
        - 1.015 * (nalpha / max(sentence_count, 1))
        - 84.6 * (syllables / max(nalpha, 1))
        if nalpha
        else 0.0
    )

    repeated = sum(c for c in counts.values() if c > 1)
    allit_pairs = sum(
        a[0].lower() == b[0].lower()
        for a, b in zip(alpha, alpha[1:])
        if a and b
    )

    # Internal rhyme: repeated nonempty rhyme keys among words within a
    # six-token window, excluding identical lexical tokens.
    rhyme_pairs = 0
    rhyme_opportunities = 0
    for i, a in enumerate(alpha):
        ka = _rhyme_key(a)
        for b in alpha[i + 1 : i + 7]:
            kb = _rhyme_key(b)
            if ka and kb and a != b:
                rhyme_opportunities += 1
                rhyme_pairs += ka == kb

    ends = _line_end_words(text)
    end_pairs = 0
    for a, b in zip(ends, ends[1:]):
        ka, kb = _rhyme_key(a), _rhyme_key(b)
        end_pairs += bool(ka and kb and ka == kb and a != b)

    ptoks = words(prompt)
    eset, pset = set(toks), set(ptoks)
    union = eset | pset

    feats = {
        "v_char_count": float(len(text)),
        "v_word_count": float(ntok),
        "v_avg_word_length": float(sum(len(w) for w in toks) / max(ntok, 1)),
        "v_sentence_count": float(sentence_count),
        "v_line_count": float(line_count),
        "v_punctuation_density": float(len(PUNCT_RE.findall(text)) / max(len(text), 1)),
        "v_question_count": float(text.count("?")),
        "v_exclamation_count": float(text.count("!")),
        "v_digit_count": float(sum(c.isdigit() for c in text)),
        "v_uppercase_ratio": float(uppercase / max(len(letters), 1)),
        "v_type_token_ratio": float(len(counts) / max(ntok, 1)),
        "v_hapax_ratio": float(sum(c == 1 for c in counts.values()) / max(len(counts), 1)),
        "v_flesch_reading_ease": float(flesch),
        "v_repeated_token_ratio": float(repeated / max(ntok, 1)),
        "v_adjacent_alliteration_rate": float(allit_pairs / max(nalpha - 1, 1)),
        "v_internal_rhyme_rate": float(rhyme_pairs / max(rhyme_opportunities, 1)),
        "v_end_rhyme_rate": float(end_pairs / max(len(ends) - 1, 1)),
        "v_prompt_token_jaccard": float(len(eset & pset) / max(len(union), 1)),
        "v_prompt_longest_shared_run": float(_longest_shared_run(toks, ptoks)),
    }
    if list(feats) != V_NAMES:
        raise AssertionError("V feature order drift")
    if not all(math.isfinite(v) for v in feats.values()):
        raise ValueError("non-finite V feature")
    return feats


def vector(entry_text: str, contest_prompt: str) -> list[float]:
    values: Mapping[str, float] = v_features(entry_text, contest_prompt)
    return [values[name] for name in V_NAMES]


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("entry_text")
    ap.add_argument("--prompt", default="")
    args = ap.parse_args()
    print(json.dumps(v_features(args.entry_text, args.prompt), indent=2))
