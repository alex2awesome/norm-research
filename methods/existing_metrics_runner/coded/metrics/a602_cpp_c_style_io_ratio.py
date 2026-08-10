"""a602: C-style I/O ratio (scanf/printf vs cin/cout).

Score = scanf_printf / (scanf_printf + cin_cout). 1.0 = all C-style,
0.0 = all C++ streams. None if no I/O detected.

Identifies a sub-class of competitive programmers who prefer printf for speed.

CLASSIFICATION: THIN — regex counted, ratio.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a602"
ASPECT_NAME = "C-style I/O usage ratio"
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
    c_io = 0.0
    cpp_io = 0.0
    found = False
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found = True
        c_io += f["scanf_printf_count"]
        cpp_io += f["cin_cout_count"]
    if not found:
        return None
    total = c_io + cpp_io
    if total == 0:
        return None
    return float(c_io / total)
