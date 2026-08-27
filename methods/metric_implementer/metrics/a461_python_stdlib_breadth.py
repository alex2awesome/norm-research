"""a461: Modern Python stdlib breadth.

Counts the number of *distinct* modules from a fixed "modern stdlib"
allowlist that the answer touches. The allowlist captures the families
that experienced Python answers tend to reach for instead of writing the
loop by hand:

    itertools, functools, collections, contextlib, pathlib, enum,
    dataclasses, typing, operator, heapq, bisect, dataclasses,
    statistics, fractions, math, re

Detection is two-pass: explicit imports (regex) AND attribute use of the
form ``modname.<symbol>`` even without an explicit import (sometimes the
review post writes it that way). Final score is ``log(1+k) / log(1+K)``
clipped to [0, 1] with K=7 -- by ~7 distinct modules we saturate.

Classification: THIN.
"""
from __future__ import annotations

import math
import re
from typing import Optional, Set

from ..sandbox import added_files_by_ext

ASPECT_ID = "a461"
ASPECT_NAME = "Modern Python stdlib breadth"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

ALLOW = (
    "itertools", "functools", "collections", "contextlib", "pathlib",
    "enum", "dataclasses", "typing", "operator", "heapq", "bisect",
    "statistics", "fractions", "math", "re",
)
_IMPORT = re.compile(
    r"^[ \t]*(?:from[ \t]+([a-zA-Z_][\w.]*)|import[ \t]+([a-zA-Z_][\w.]*(?:[ \t]*,[ \t]*[a-zA-Z_][\w.]*)*))",
    re.M,
)
# attribute use without import
_ATTR_USE = {
    name: re.compile(rf"(?:^|[^\w.]){name}\.[a-zA-Z_]\w*")
    for name in ALLOW
}


def _modules_touched(text: str) -> Set[str]:
    found: Set[str] = set()
    for m in _IMPORT.finditer(text):
        mod = m.group(1) or m.group(2) or ""
        for piece in mod.split(","):
            top = piece.strip().split(".", 1)[0]
            if top in ALLOW:
                found.add(top)
    for name, rx in _ATTR_USE.items():
        if name in found:
            continue
        if rx.search(text):
            found.add(name)
    return found


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    k = len(_modules_touched(text))
    K = 7.0
    return float(min(1.0, math.log(1 + k) / math.log(1 + K)))


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
