"""a477: Python non-English / non-ASCII comments marker.

Editorial Python on LeetCode is always written in English. User
solutions sometimes carry Chinese / Russian / Korean / etc. commentary,
which is a strong register cue distinguishing them from editorial code.

Detected: any non-ASCII code point appearing inside a `#`-comment line
or inside a triple-quoted string. We ignore non-ASCII inside regular
string literals (those can be legitimately part of test data).

Score: tanh(non_ascii_comment_chars / 10).
0 -> 0.0
10 chars -> 0.76
50 chars -> ~1.0
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a477"
ASPECT_NAME = "Python non-English (non-ASCII) comments"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
APPLIES_TO_EXTS = [".py", ".pyi"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_NON_ASCII = re.compile(r"[^\x00-\x7F]")
_TRIPLE = re.compile(r'(?P<q>"""|\'\'\')(.+?)(?P=q)', re.S)


def _count_non_ascii(text: str) -> int:
    n = 0
    # comment-line non-ASCII
    for line in text.split("\n"):
        if "#" in line:
            # take everything from first '#' onward; ignore '#' inside strings
            # (cheap approximation: we don't strip strings, but tab/space rules
            # mean comments dominate)
            idx = line.find("#")
            if idx >= 0:
                comment = line[idx:]
                n += len(_NON_ASCII.findall(comment))
    # triple-quoted string non-ASCII
    for m in _TRIPLE.finditer(text):
        n += len(_NON_ASCII.findall(m.group(2)))
    return n


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    n = _count_non_ascii(text)
    return float(math.tanh(n / 10.0))


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
