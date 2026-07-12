"""g3: Presentation register — inline-code density (per 1000 chars).

Promoted from the inline g-metrics of scripts/se_ladder/se_ladder_score.py
(SE ladder, 2026-06-11). ARTIFACT = "body" (see g1 docstring). Top positive
LR coefficient on every SO slice — prose that references identifiers in
`backticks` wins.
"""
from __future__ import annotations

import re
from typing import Optional

ASPECT_ID = "g3"
ASPECT_NAME = "Presentation register: inline-code density"
TIER = 1
TOOLS: list = []
APPLIES_TO_LANGS = ["any"]
CLASSIFICATION = "THIN"
ARTIFACT = "body"

_INLINE_CODE_RE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")


def _is_diff(text: str) -> bool:
    return text.lstrip().startswith("diff --git")


def applies(text: str) -> bool:
    return isinstance(text, str) and len(text) >= 5 and not _is_diff(text)


def score(text: str) -> Optional[float]:
    if not applies(text):
        return None
    return float(1000.0 * len(_INLINE_CODE_RE.findall(text)) / len(text))
