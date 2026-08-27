"""a476: Python verbose multi-paragraph docstring marker.

User solutions sometimes carry a multi-paragraph docstring describing
the chosen approach ("Approach:", "Method:", "Solution:", "Trick:").
Editorial code keeps the same content out of band, in surrounding prose.

Detected: at least one triple-quoted docstring whose contents
  - span more than 3 non-blank lines, AND/OR
  - contain at least one header keyword:
    Approach / Method / Solution / Trick / Idea / Algorithm /
    Intuition / Explanation / Steps / Observation

Score: tanh(matches / 1.5). 1 verbose docstring -> 0.58, 2 -> 0.86.

Classification: THIN, register-only.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a476"
ASPECT_NAME = "Python verbose explanatory docstring"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
APPLIES_TO_EXTS = [".py", ".pyi"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_RX_TRIPLE = re.compile(r'(?P<q>"""|\'\'\')(.+?)(?P=q)', re.S)
_HEADER = re.compile(
    r"\b(approach|method|solution|trick|idea|algorithm|intuition|"
    r"explanation|steps|observation|strategy|insight|key idea)\b\s*[:\-]",
    re.I,
)


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    matches = 0
    for m in _RX_TRIPLE.finditer(text):
        body = m.group(2)
        lines = [ln for ln in body.split("\n") if ln.strip()]
        long_enough = len(lines) > 3
        has_header = _HEADER.search(body) is not None
        if long_enough and has_header:
            matches += 1
        elif long_enough or has_header:
            matches += 0.5
    return float(math.tanh(matches / 1.5))


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    scs = [s for s in (_file_score(c) for c in by_path.values()) if s is not None]
    if not scs:
        return None
    return float(sum(scs) / len(scs))
