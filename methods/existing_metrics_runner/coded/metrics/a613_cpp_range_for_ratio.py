"""a613: Range-based for / total for ratio.

Score = range_for / (range_for + classic_for). 1.0 means all loops are
range-based; 0.0 means all are classic C-style with semicolons; None when
no for loop is detected. Modern-idiom indicator.

CLASSIFICATION: THIN — regex.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a613"
ASPECT_NAME = "Range-based for ratio"
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
    r = 0.0
    c = 0.0
    found = False
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found = True
        r += f["range_for_count"]
        c += f["classic_for_count"]
    if not found:
        return None
    total = r + c
    if total == 0:
        return None
    return float(r / total)
