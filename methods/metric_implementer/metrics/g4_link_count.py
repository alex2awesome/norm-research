"""g4: Presentation register — link count (bare URLs + markdown links).

Promoted from the inline g-metrics of scripts/se_ladder/se_ladder_score.py
(SE ladder, 2026-06-11). ARTIFACT = "body" (see g1 docstring).
"""
from __future__ import annotations

import re
from typing import Optional

ASPECT_ID = "g4"
ASPECT_NAME = "Presentation register: link count"
TIER = 1
TOOLS: list = []
APPLIES_TO_LANGS = ["any"]
CLASSIFICATION = "THIN"
ARTIFACT = "body"

_LINK_RE = re.compile(r"https?://\S+|\[[^\]]+\]\([^)]+\)")


def _is_diff(text: str) -> bool:
    return text.lstrip().startswith("diff --git")


def applies(text: str) -> bool:
    return isinstance(text, str) and len(text) >= 5 and not _is_diff(text)


def score(text: str) -> Optional[float]:
    if not applies(text):
        return None
    return float(len(_LINK_RE.findall(text)))
