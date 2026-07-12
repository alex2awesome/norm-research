"""g2: Presentation register — fraction of body chars inside fenced code.

Promoted from the inline g-metrics of scripts/se_ladder/se_ladder_score.py
(SE ladder, 2026-06-11). ARTIFACT = "body" (see g1 docstring). Note: in the
ladder LR this coefficient is consistently NEGATIVE on SO slices — all-code
answers lose.
"""
from __future__ import annotations

import re
from typing import Optional

ASPECT_ID = "g2"
ASPECT_NAME = "Presentation register: code-character fraction"
TIER = 1
TOOLS: list = []
APPLIES_TO_LANGS = ["any"]
CLASSIFICATION = "THIN"
ARTIFACT = "body"

_FENCE_BLOCK_RE = re.compile(
    r"^[ \t]*(?:```|~~~)[^\n]*\n(.*?)^[ \t]*(?:```|~~~)[ \t]*$",
    re.DOTALL | re.MULTILINE)


def _is_diff(text: str) -> bool:
    return text.lstrip().startswith("diff --git")


def applies(text: str) -> bool:
    return isinstance(text, str) and len(text) >= 5 and not _is_diff(text)


def score(text: str) -> Optional[float]:
    if not applies(text):
        return None
    blocks = _FENCE_BLOCK_RE.findall(text)
    return float(sum(len(b) for b in blocks) / max(1, len(text)))
