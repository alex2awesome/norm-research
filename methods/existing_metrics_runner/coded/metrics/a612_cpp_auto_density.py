"""a612: `auto` keyword density per 100 LOC.

Modern-C++11+ idiom. Older Olympic submissions write `int` / `vector<...>`
explicitly; modern candidates use `auto`. Higher = more modern.

CLASSIFICATION: THIN — regex; counts the keyword `auto` only.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a612"
ASPECT_NAME = "auto keyword density per 100 LOC"
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
    n_auto = 0.0
    nloc = 0.0
    found = False
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found = True
        n_auto += f["auto_count"]
        nloc += f["nloc_strip"]
    if not found or nloc <= 0:
        return None
    return float(100.0 * n_auto / nloc)
