"""a448: C++ closing-brace density.

Computes ``count('}') / non-blank lines`` across added C++ files.

High values indicate verbose, deeply nested or explicitly-blocked code;
LC-community style favours single-line ``if`` bodies and tight loops.

Returns NaN when there are no non-blank lines.

Tier 1. THIN.
"""
from __future__ import annotations
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a448"
ASPECT_NAME = "C++ closing-brace density"
TIER = 1
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    n_close = 0
    n_lines = 0
    for content in by_path.values():
        for ln in content.split("\n"):
            if ln.strip():
                n_lines += 1
        n_close += content.count("}")
    if n_lines == 0:
        return None
    return n_close / n_lines
