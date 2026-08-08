"""a449: C++ semicolon density.

Computes ``count(';') / non-blank lines`` across added C++ files.

High values indicate many statements per line (often LC-community style
``a++,b++;``) **or** verbose multi-statement-per-line code. We expect
this to vary across the LC / industrial axis even though the sign is not
fixed a priori.

Returns NaN when there are no non-blank lines.

Tier 1. THIN.
"""
from __future__ import annotations
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a449"
ASPECT_NAME = "C++ semicolon density"
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
    n_semi = 0
    n_lines = 0
    for content in by_path.values():
        for ln in content.split("\n"):
            if ln.strip():
                n_lines += 1
        n_semi += content.count(";")
    if n_lines == 0:
        return None
    return n_semi / n_lines
