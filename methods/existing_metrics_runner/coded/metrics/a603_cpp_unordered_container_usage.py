"""a603: Use of `unordered_map` / `unordered_set` (hash-based STL).

Binary: 1.0 if any unordered_* container is used; 0.0 if no STL maps/sets
are present at all; 0.0 if only ordered (map/set) variants are used.
Distinguishes hash-aware programmers from tree-based defaults.

CLASSIFICATION: THIN — exact regex.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a603"
ASPECT_NAME = "unordered_map/set usage"
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
    has = 0.0
    found = False
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found = True
        if f["use_unordered"]:
            has = 1.0
    if not found:
        return None
    return float(has)
