#!/usr/bin/env python3
"""Deterministic, label-blind V features for #HashtagWars tweets.

Only the tweet and the contest hashtag are accepted.  The verdict is neither
accepted nor inspected.  Counts are Unicode-aware where practical and require
only the Python standard library.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-z0-9_]+")
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}", re.I)
ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…)")
SENTENCE_END_RE = re.compile(r"[.!?]+(?:\s|$)")


def prompt_words(hashtag: str) -> List[str]:
    """Split filename-style or CamelCase hashtags into lowercase words."""
    value = hashtag.strip().lstrip("#").replace("_", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return [w.lower() for w in WORD_RE.findall(value)]


def content_text(text: str, hashtag: str) -> str:
    """Remove submission plumbing, but preserve all authored content."""
    value = URL_RE.sub(" ", text or "")
    value = MENTION_RE.sub(" ", value)
    value = HASHTAG_RE.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def _syllables(word: str) -> int:
    """Small deterministic vowel-group approximation used for readability."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = len(re.findall(r"[aeiouy]+", w))
    if len(w) > 3 and w.endswith("e") and not w.endswith(("le", "ye")):
        groups -= 1
    return max(groups, 1)


def _emoji_count(text: str) -> int:
    # Symbols and pictographs are mostly So; exclude ordinary ASCII symbols.
    return sum(
        1
        for ch in text
        if ord(ch) > 127 and unicodedata.category(ch) in {"So", "Sk"}
    )


def _alliteration_ratio(words: List[str]) -> float:
    pairs = [(a, b) for a, b in zip(words, words[1:]) if a and b]
    if not pairs:
        return 0.0
    return sum(a[0].lower() == b[0].lower() for a, b in pairs) / len(pairs)


def _rhyme_ratio(words: List[str]) -> float:
    """Fraction of nearby word pairs sharing a 3-letter suffix.

    This is deliberately a surface proxy, not a claim about pronunciation.
    """
    clean = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
    pairs = []
    for i, left in enumerate(clean):
        if len(left) < 4:
            continue
        for right in clean[i + 1 : i + 4]:
            if len(right) >= 4 and left != right:
                pairs.append((left, right))
    if not pairs:
        return 0.0
    return sum(a[-3:] == b[-3:] for a, b in pairs) / len(pairs)


def _prompt_substring_ratio(words: List[str], pwords: List[str]) -> float:
    """Share of content words with a >=4-character prompt substring."""
    if not words or not pwords:
        return 0.0
    hits = 0
    for word in words:
        w = word.lower()
        found = False
        for prompt in pwords:
            for n in range(4, min(len(w), len(prompt)) + 1):
                if w[:n] in prompt or prompt[:n] in w:
                    found = True
                    break
            if found:
                break
        hits += found
    return hits / len(words)


V_NAMES = [
    "v_char_count",
    "v_token_count",
    "v_avg_token_length",
    "v_type_token_ratio",
    "v_uppercase_letter_ratio",
    "v_all_caps_token_count",
    "v_question_count",
    "v_exclamation_count",
    "v_ellipsis_count",
    "v_hashtag_count",
    "v_mention_count",
    "v_url_count",
    "v_emoji_count",
    "v_digit_count",
    "v_repeated_char_run_count",
    "v_repeated_token_count",
    "v_adjacent_alliteration_ratio",
    "v_nearby_suffix_rhyme_ratio",
    "v_prompt_shared_substring_ratio",
    "v_automated_readability_index",
]


def v_features(text: str, hashtag: str) -> Dict[str, float]:
    """Return the fixed 20-feature V bank."""
    raw = text or ""
    content = content_text(raw, hashtag)
    words = WORD_RE.findall(content)
    lowers = [w.lower().replace("’", "'") for w in words]
    n_words = len(words)
    n_chars = len(content)
    letters = [ch for ch in content if ch.isalpha()]
    counts = Counter(lowers)
    pwords = prompt_words(hashtag)
    sentence_count = max(len(SENTENCE_END_RE.findall(content)), 1 if content else 0)
    ari = (
        4.71 * (sum(len(w) for w in words) / max(n_words, 1))
        + 0.5 * (n_words / max(sentence_count, 1))
        - 21.43
        if content
        else 0.0
    )
    out = {
        "v_char_count": float(n_chars),
        "v_token_count": float(n_words),
        "v_avg_token_length": float(sum(len(w) for w in words) / max(n_words, 1)),
        "v_type_token_ratio": float(len(counts) / max(n_words, 1)),
        "v_uppercase_letter_ratio": float(
            sum(ch.isupper() for ch in letters) / max(len(letters), 1)
        ),
        "v_all_caps_token_count": float(len(ALL_CAPS_RE.findall(content))),
        "v_question_count": float(content.count("?")),
        "v_exclamation_count": float(content.count("!")),
        "v_ellipsis_count": float(len(ELLIPSIS_RE.findall(content))),
        "v_hashtag_count": float(len(HASHTAG_RE.findall(raw))),
        "v_mention_count": float(len(MENTION_RE.findall(raw))),
        "v_url_count": float(len(URL_RE.findall(raw))),
        "v_emoji_count": float(_emoji_count(raw)),
        "v_digit_count": float(sum(ch.isdigit() for ch in content)),
        "v_repeated_char_run_count": float(len(REPEATED_CHAR_RE.findall(content))),
        "v_repeated_token_count": float(sum(v - 1 for v in counts.values() if v > 1)),
        "v_adjacent_alliteration_ratio": float(_alliteration_ratio(words)),
        "v_nearby_suffix_rhyme_ratio": float(_rhyme_ratio(words)),
        "v_prompt_shared_substring_ratio": float(
            _prompt_substring_ratio(words, pwords)
        ),
        "v_automated_readability_index": float(ari),
    }
    assert list(out) == V_NAMES
    assert all(math.isfinite(v) for v in out.values())
    return out


def vector(text: str, hashtag: str, names: Iterable[str] = V_NAMES) -> List[float]:
    values = v_features(text, hashtag)
    return [values[name] for name in names]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--hashtag", required=True)
    args = parser.parse_args()
    print(json.dumps(v_features(args.text, args.hashtag), indent=2))
