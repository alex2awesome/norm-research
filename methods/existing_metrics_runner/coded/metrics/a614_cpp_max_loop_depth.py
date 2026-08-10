"""a614: Max nested-loop depth.

Integer >= 0. Counts the max depth of nested for/while/do-while structures
in the candidate (via tree-sitter). 3+ is common in DP/brute-force; 0-1
suggests simulation or single-pass algorithm.

CLASSIFICATION: PARTIALLY_THIN — tree-sitter primary, regex fallback.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a614"
ASPECT_NAME = "Max loop nesting depth"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "PARTIALLY_THIN"


def applies(diff_text: str) -> bool:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return False
    return any(looks_like_cpp(c) for c in by_path.values())


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    best = 0.0
    found = False
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found = True
        if f["max_loop_depth"] > best:
            best = f["max_loop_depth"]
    if not found:
        return None
    return float(best)
