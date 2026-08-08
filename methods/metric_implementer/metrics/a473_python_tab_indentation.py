"""a473: Python tab-indentation marker.

Binary marker: 1.0 if the source contains a literal TAB character
used for indentation, 0.0 otherwise.

Editorial-style Python on LeetCode is almost universally space-indented
(PEP 8). User-submitted solutions that arrive with tab indentation tend
to be copy-pasted from personal IDEs / scratch files and read as
register-distinct from editorial code.

We only flag TABs that appear at the start of a line (indentation
context). Inline tabs inside strings are ignored.

Classification: THIN, register-only.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a473"
ASPECT_NAME = "Python tab-indentation marker"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
APPLIES_TO_EXTS = [".py", ".pyi"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_LEADING_TAB = re.compile(r"^\t", re.M)
_INTERIOR_TAB = re.compile(r"^ +\t", re.M)


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    if _LEADING_TAB.search(text) is not None:
        return 1.0
    if _INTERIOR_TAB.search(text) is not None:
        return 1.0
    return 0.0


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
