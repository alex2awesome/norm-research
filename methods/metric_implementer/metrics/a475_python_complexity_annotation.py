"""a475: Python big-O complexity annotation marker.

Captures the explicit `# Time: O(n)` / `# Space: O(1)` / `# O(n log n)`
register that user solutions adopt to advertise their analysis. Editorial
posts usually keep complexity in the surrounding prose, not in trailing
code comments.

Detected patterns (regex over comment lines):
  - bare `# O(...)` or `# o(...)` at end-of-line or stand-alone
  - `# Time:` / `# time complexity:` / `# T:` lines
  - `# Space:` / `# space complexity:` / `# S:` lines
  - `# Complexity:`
  - `# T.C:` / `# S.C:` / `# TC:` / `# SC:` shorthand

Score: tanh(annotation_count / 2). 0 -> 0, 2+ -> 0.76.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a475"
ASPECT_NAME = "Python big-O complexity annotation"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
APPLIES_TO_EXTS = [".py", ".pyi"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

# big-O parenthesized expression
_RX_BIG_O = re.compile(r"#[^\n]*\b[OoΘΩ]\s*\(\s*[^)\n]{1,60}\)")
_RX_TIME = re.compile(
    r"#[ \t]*(?:time[ \t]*complexity|time|t\.?c\.?|tc)\s*[:=]",
    re.I,
)
_RX_SPACE = re.compile(
    r"#[ \t]*(?:space[ \t]*complexity|space|s\.?c\.?|sc)\s*[:=]",
    re.I,
)
_RX_COMPLEXITY = re.compile(r"#[ \t]*complexity\s*[:=]", re.I)


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    hits = 0
    for line in text.split("\n"):
        if "#" not in line:
            continue
        if _RX_BIG_O.search(line):
            hits += 1
            continue
        if _RX_TIME.search(line) or _RX_SPACE.search(line) or _RX_COMPLEXITY.search(line):
            hits += 1
    # Also look inside triple-quoted blocks for these patterns
    for m in re.finditer(r'"""(.*?)"""', text, re.S):
        blk = m.group(1)
        hits += len(_RX_BIG_O.findall(blk)) // 2  # count weakly inside docstrings
        if _RX_TIME.search(blk):
            hits += 1
        if _RX_SPACE.search(blk):
            hits += 1
    return float(math.tanh(hits / 2.0))


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
