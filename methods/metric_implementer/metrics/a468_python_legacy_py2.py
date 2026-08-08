"""a468: Python 2 legacy marker penalty.

NEGATIVE-direction: counts occurrences of Py2-only constructs:
  - ``print`` statement without parentheses
  - ``xrange``
  - ``raw_input``
  - ``.iteritems`` / ``.iterkeys`` / ``.itervalues``
  - ``has_key``
  - unicode-literal prefix ``u'...'`` / ``u"..."``

Score = ``1 - tanh(legacy_count / 5)``. Zero hits -> 1.0 (clean), 5+
hits -> ~0.

Classification: THIN.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a468"
ASPECT_NAME = "Python 2 legacy markers (penalty)"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_PATTERNS = [
    re.compile(r"^\s*print[ \t]+[^(]", re.M),  # print foo (NOT print(foo))
    re.compile(r"\bxrange\s*\("),
    re.compile(r"\braw_input\s*\("),
    re.compile(r"\.(?:iteritems|iterkeys|itervalues)\s*\("),
    re.compile(r"\.has_key\s*\("),
    re.compile(r"(?:^|[^A-Za-z_0-9])[uU](?:[rR])?['\"][^'\"\\\n]*['\"]"),
]


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    hits = 0
    for rx in _PATTERNS:
        hits += len(rx.findall(text))
    return float(1.0 - math.tanh(hits / 5.0))


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
