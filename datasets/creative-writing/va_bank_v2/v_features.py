#!/usr/bin/env python3
"""Deterministic surface-feature bank for WritingPrompts stories.

All features are computed on the full story body (the prompt is excluded).  The A
judge uses a separately documented deterministic head+tail truncation.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"'”’)\]]+)?\s+")
DIALOGUE_RE = re.compile(r'(?:"[^"\n]{1,1000}"|“[^”\n]{1,1000}”)')
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.I)

V_NAMES = [
    "v_log_chars",
    "v_log_words",
    "v_mean_word_chars",
    "v_mattr_50",
    "v_hapax_fraction",
    "v_log_paragraphs",
    "v_mean_paragraph_words",
    "v_paragraph_length_cv",
    "v_dialogue_word_fraction",
    "v_mean_sentence_words",
    "v_sentence_length_cv",
    "v_question_per_100_words",
    "v_exclaim_per_100_words",
    "v_punctuation_per_100_words",
    "v_flesch_reading_ease",
]


def _safe_cv(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if mean > 0 else 0.0


def _mattr(tokens: list[str], window: int = 50) -> float:
    n = len(tokens)
    if not n:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    counts = Counter(tokens[:window])
    diversity_sum = float(len(counts))
    for i in range(window, n):
        outgoing = tokens[i - window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[tokens[i]] += 1
        diversity_sum += len(counts)
    return diversity_sum / (n - window + 1) / window


def _syllables(word: str) -> int:
    """Small deterministic approximation used only for the readability check."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = len(VOWEL_GROUP_RE.findall(w))
    if len(w) > 2 and w.endswith("e") and not w.endswith(("le", "ye")):
        groups -= 1
    if len(w) > 3 and w.endswith("es") and not w.endswith(("aes", "ees", "oes")):
        groups -= 1
    return max(1, groups)


def compute_v_features(story: str) -> dict[str, float]:
    text = story or ""
    words_original = WORD_RE.findall(text)
    words = [w.lower().replace("’", "'") for w in words_original]
    n_words = len(words)
    denom = max(n_words, 1)

    paragraphs = [p for p in re.split(r"\n\s*\n+", text) if p.strip()]
    para_lengths = [len(WORD_RE.findall(p)) for p in paragraphs]
    sentences = [s for s in SENTENCE_RE.split(text) if WORD_RE.search(s)]
    sent_lengths = [len(WORD_RE.findall(s)) for s in sentences]

    dialogue_words = sum(len(WORD_RE.findall(m.group(0))) for m in DIALOGUE_RE.finditer(text))
    counts = Counter(words)
    hapax = sum(1 for count in counts.values() if count == 1)
    syllables = sum(_syllables(w) for w in words)
    n_sent = max(len(sentences), 1)
    flesch = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / denom)
    punctuation = sum(text.count(ch) for ch in ".,;:!?—-…")

    values = {
        "v_log_chars": math.log1p(len(text)),
        "v_log_words": math.log1p(n_words),
        "v_mean_word_chars": sum(len(w) for w in words) / denom,
        "v_mattr_50": _mattr(words, 50),
        "v_hapax_fraction": hapax / max(len(counts), 1),
        "v_log_paragraphs": math.log1p(len(paragraphs)),
        "v_mean_paragraph_words": float(np.mean(para_lengths)) if para_lengths else 0.0,
        "v_paragraph_length_cv": _safe_cv(para_lengths),
        "v_dialogue_word_fraction": dialogue_words / denom,
        "v_mean_sentence_words": float(np.mean(sent_lengths)) if sent_lengths else 0.0,
        "v_sentence_length_cv": _safe_cv(sent_lengths),
        "v_question_per_100_words": 100.0 * text.count("?") / denom,
        "v_exclaim_per_100_words": 100.0 * text.count("!") / denom,
        "v_punctuation_per_100_words": 100.0 * punctuation / denom,
        "v_flesch_reading_ease": float(flesch),
    }
    return {name: float(values[name]) for name in V_NAMES}


def feature_vector(story: str) -> list[float]:
    values = compute_v_features(story)
    return [values[name] for name in V_NAMES]


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(compute_v_features(sys.stdin.read()), indent=2, sort_keys=True))
