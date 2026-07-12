"""g6: Presentation register — hedging/suggestion register rate (per 1000 words).

Promoted from the inline g-metrics of scripts/se_ladder/se_ladder_score.py
(SE ladder, 2026-06-11). ARTIFACT = "body" (see g1 docstring).
"""
from __future__ import annotations

import re
from typing import Optional

ASPECT_ID = "g6"
ASPECT_NAME = "Presentation register: hedging-register rate"
TIER = 1
TOOLS: list = []
APPLIES_TO_LANGS = ["any"]
CLASSIFICATION = "THICK"
ARTIFACT = "body"

_HEDGE_RE = re.compile(
    r"\b(i would|i'd|you could|you might|consider|note that|personally|"
    r"imho|in my opinion|perhaps|maybe|instead|alternatively|by the way|"
    r"i think|i believe|suggest)\b", re.IGNORECASE)


def _is_diff(text: str) -> bool:
    return text.lstrip().startswith("diff --git")


def applies(text: str) -> bool:
    return isinstance(text, str) and len(text) >= 5 and not _is_diff(text)


def score(text: str) -> Optional[float]:
    if not applies(text):
        return None
    n_words = max(1, len(text.split()))
    return float(1000.0 * len(_HEDGE_RE.findall(text)) / n_words)
