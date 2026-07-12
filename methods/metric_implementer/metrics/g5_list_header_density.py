"""g5: Presentation register — list/header/quote line density (per 1000 chars).

Promoted from the inline g-metrics of scripts/se_ladder/se_ladder_score.py
(SE ladder, 2026-06-11). ARTIFACT = "body" (see g1 docstring).
"""
from __future__ import annotations

import re
from typing import Optional

ASPECT_ID = "g5"
ASPECT_NAME = "Presentation register: list/header density"
TIER = 1
TOOLS: list = []
APPLIES_TO_LANGS = ["any"]
CLASSIFICATION = "THIN"
ARTIFACT = "body"

_LIST_HDR_RE = re.compile(r"^[ \t]*(?:[-*+] |\d+\. |#{1,6} |> )", re.MULTILINE)


def _is_diff(text: str) -> bool:
    return text.lstrip().startswith("diff --git")


def applies(text: str) -> bool:
    return isinstance(text, str) and len(text) >= 5 and not _is_diff(text)


def score(text: str) -> Optional[float]:
    if not applies(text):
        return None
    return float(1000.0 * len(_LIST_HDR_RE.findall(text)) / len(text))
