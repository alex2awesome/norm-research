"""a601: Fast-I/O sync_with_stdio + cin.tie usage flag.

Binary: 1.0 if the candidate calls `ios_base::sync_with_stdio(false)` AND/OR
`cin.tie(0)` / `cin.tie(NULL)`. Diagnostic of competitive C++ style.

CLASSIFICATION: THIN — exact regex.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a601"
ASPECT_NAME = "Fast-I/O sync_with_stdio / cin.tie usage"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"


def applies(diff_text: str) -> bool:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return False
    return any(looks_like_cpp(c) for c in by_path.values())


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    any_fast = False
    found_any = False
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found_any = True
        if f["sync_with_stdio"] or f["cin_tie_zero"]:
            any_fast = True
    if not found_any:
        return None
    return float(1.0 if any_fast else 0.0)
