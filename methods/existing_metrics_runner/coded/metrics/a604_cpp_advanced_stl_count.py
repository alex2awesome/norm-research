"""a604: Count of advanced STL containers used (priority_queue, deque, bitset).

Integer in [0, 3]: how many of {priority_queue, deque, bitset} appear in the
candidate. Distinguishes vanilla candidates from those reaching for
specialized containers.

CLASSIFICATION: THIN — exact regex of canonical type names.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a604"
ASPECT_NAME = "Advanced STL container count"
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
    found = False
    flags = [0, 0, 0]
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found = True
        if f["use_priority_queue"]:
            flags[0] = 1
        if f["use_deque"]:
            flags[1] = 1
        if f["use_bitset"]:
            flags[2] = 1
    if not found:
        return None
    return float(sum(flags))
