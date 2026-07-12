"""g1: Presentation register — fenced code-block count.

Promoted from the inline g-metrics of scripts/se_ladder/se_ladder_score.py
(SE ladder, 2026-06-11) so they are reusable bank members. Unlike the aNNN
metrics, the g-metrics score the FULL ANSWER BODY (markdown or de-HTMLed
text), not a synthetic diff: ARTIFACT = "body". Consumers that score diffs
get applies()=False via the diff guard; body-aware scorers (se_ladder_score)
route the answer body here.

Score = number of fenced (``` / ~~~) code blocks. Raw count, standardized
downstream by the model pipeline.
"""
from __future__ import annotations

import re
from typing import Optional

ASPECT_ID = "g1"
ASPECT_NAME = "Presentation register: fenced code-block count"
TIER = 1
TOOLS: list = []
APPLIES_TO_LANGS = ["any"]
CLASSIFICATION = "THIN"
ARTIFACT = "body"  # full answer body, NOT a diff

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
    return float(len(_FENCE_BLOCK_RE.findall(text)))
