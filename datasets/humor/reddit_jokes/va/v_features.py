#!/usr/bin/env python3
"""Deterministic, label-blind V features for r/Jokes posts (reddit-jokes
community cell).

Only the joke text is accepted.  The upvote-derived verdict is neither
accepted nor inspected.  Standard library only, Unicode-aware where practical.

Adapted from datasets/humor/hashtagwars/va/v_features.py: the prompt-dependent
features (`v_prompt_shared_substring_ratio`, hashtag/mention counts) are dropped
because an r/Jokes post has no contest prompt, and joke-form surface proxies
(setup/punchline shape, dialogue, list structure) are added in their place.
Every feature is a surface count or ratio -- none of them reads meaning.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}", re.I)
ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…)")
SENTENCE_END_RE = re.compile(r"[.!?]+(?:\s|$)")
QUOTE_RE = re.compile(r"[\"“”]")
DIALOGUE_RE = re.compile(
    r"\b(?:said|says|asked|asks|replied|replies|answered|answers|shouted|yells?)\b", re.I)
NARRATIVE_OPEN_RE = re.compile(
    r"^\s*(?:a|an|the|so|two|three|my|this)\b[^.?!]{0,60}\b"
    r"(?:walk|walks|walked|goes|went|enters?|entered|sits?|sat|says?|said|are|is|was|were)\b",
    re.I)
RIDDLE_OPEN_RE = re.compile(
    r"^\s*(?:what|why|how|when|where|who|which)\b", re.I)
LIST_MARK_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", re.M)


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = len(re.findall(r"[aeiouy]+", w))
    if len(w) > 3 and w.endswith("e") and not w.endswith(("le", "ye")):
        groups -= 1
    return max(groups, 1)


def _emoji_count(text: str) -> int:
    return sum(1 for ch in text
               if ord(ch) > 127 and unicodedata.category(ch) in {"So", "Sk"})


def _alliteration_ratio(words: List[str]) -> float:
    pairs = [(a, b) for a, b in zip(words, words[1:]) if a and b]
    if not pairs:
        return 0.0
    return sum(a[0].lower() == b[0].lower() for a, b in pairs) / len(pairs)


def _rhyme_ratio(words: List[str]) -> float:
    """Fraction of nearby word pairs sharing a 3-letter suffix (surface proxy)."""
    clean = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
    pairs = []
    for i, left in enumerate(clean):
        if len(left) < 4:
            continue
        for right in clean[i + 1:i + 4]:
            if len(right) >= 4 and left != right:
                pairs.append((left, right))
    if not pairs:
        return 0.0
    return sum(a[-3:] == b[-3:] for a, b in pairs) / len(pairs)


def _tail_share(text: str) -> float:
    """Share of characters that fall after the last sentence boundary --
    a crude 'how much of the text is the final beat' proxy."""
    if not text:
        return 0.0
    ends = list(SENTENCE_END_RE.finditer(text))
    if not ends:
        return 1.0
    last = ends[-1].end()
    if last >= len(text):
        last = ends[-2].end() if len(ends) > 1 else 0
    return (len(text) - last) / len(text)


V_NAMES = [
    "v_char_count",
    "v_token_count",
    "v_avg_token_length",
    "v_type_token_ratio",
    "v_sentence_count",
    "v_avg_sentence_tokens",
    "v_final_beat_char_share",
    "v_uppercase_letter_ratio",
    "v_all_caps_token_count",
    "v_question_count",
    "v_exclamation_count",
    "v_ellipsis_count",
    "v_quote_mark_count",
    "v_dialogue_verb_count",
    "v_linebreak_count",
    "v_list_marker_count",
    "v_url_count",
    "v_emoji_count",
    "v_digit_count",
    "v_repeated_char_run_count",
    "v_repeated_token_count",
    "v_adjacent_alliteration_ratio",
    "v_nearby_suffix_rhyme_ratio",
    "v_riddle_opening",
    "v_narrative_opening",
    "v_automated_readability_index",
    "v_flesch_reading_ease",
]


def v_features(text: str) -> Dict[str, float]:
    """Return the fixed 27-feature V bank for one joke."""
    raw = text or ""
    content = URL_RE.sub(" ", raw)
    content = re.sub(r"[ \t]+", " ", content).strip()
    words = WORD_RE.findall(content)
    lowers = [w.lower().replace("’", "'") for w in words]
    n_words = len(words)
    letters = [ch for ch in content if ch.isalpha()]
    counts = Counter(lowers)
    n_sent = max(len(SENTENCE_END_RE.findall(content)), 1 if content else 0)
    syll = sum(_syllables(w) for w in words)
    ari = (4.71 * (sum(len(w) for w in words) / max(n_words, 1))
           + 0.5 * (n_words / max(n_sent, 1)) - 21.43) if content else 0.0
    fre = (206.835 - 1.015 * (n_words / max(n_sent, 1))
           - 84.6 * (syll / max(n_words, 1))) if content else 0.0
    out = {
        "v_char_count": float(len(content)),
        "v_token_count": float(n_words),
        "v_avg_token_length": float(sum(len(w) for w in words) / max(n_words, 1)),
        "v_type_token_ratio": float(len(counts) / max(n_words, 1)),
        "v_sentence_count": float(n_sent),
        "v_avg_sentence_tokens": float(n_words / max(n_sent, 1)),
        "v_final_beat_char_share": float(_tail_share(content)),
        "v_uppercase_letter_ratio": float(
            sum(ch.isupper() for ch in letters) / max(len(letters), 1)),
        "v_all_caps_token_count": float(len(ALL_CAPS_RE.findall(content))),
        "v_question_count": float(content.count("?")),
        "v_exclamation_count": float(content.count("!")),
        "v_ellipsis_count": float(len(ELLIPSIS_RE.findall(content))),
        "v_quote_mark_count": float(len(QUOTE_RE.findall(content))),
        "v_dialogue_verb_count": float(len(DIALOGUE_RE.findall(content))),
        "v_linebreak_count": float(raw.count("\n")),
        "v_list_marker_count": float(len(LIST_MARK_RE.findall(raw))),
        "v_url_count": float(len(URL_RE.findall(raw))),
        "v_emoji_count": float(_emoji_count(raw)),
        "v_digit_count": float(sum(ch.isdigit() for ch in content)),
        "v_repeated_char_run_count": float(len(REPEATED_CHAR_RE.findall(content))),
        "v_repeated_token_count": float(sum(v - 1 for v in counts.values() if v > 1)),
        "v_adjacent_alliteration_ratio": float(_alliteration_ratio(words)),
        "v_nearby_suffix_rhyme_ratio": float(_rhyme_ratio(words)),
        "v_riddle_opening": float(bool(RIDDLE_OPEN_RE.match(content))),
        "v_narrative_opening": float(bool(NARRATIVE_OPEN_RE.match(content))),
        "v_automated_readability_index": float(ari),
        "v_flesch_reading_ease": float(fre),
    }
    assert list(out) == V_NAMES
    assert all(math.isfinite(v) for v in out.values()), out
    return out


def vector(text: str, names: Iterable[str] = V_NAMES) -> List[float]:
    values = v_features(text)
    return [values[name] for name in names]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()
    print(json.dumps(v_features(args.text), indent=2))
