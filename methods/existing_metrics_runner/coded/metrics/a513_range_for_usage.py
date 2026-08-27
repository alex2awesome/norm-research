"""a513: Range-for fraction of all for-loops in C++.

Counts the proportion of `for (auto ...)` style range-based loops over all
`for (...)` loops in the added C++ code. Returns range_count / total_count.

This OVERLAPS partially with a410 (modern C++ idioms density), but is
strictly a SINGLE-DIMENSION ratio rather than a composite — orthogonal as a
feature column in the LR model.

CANDIDATE CODE ALONE. CLASSIFICATION: THIN.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a513"
ASPECT_NAME = "Range-for fraction"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]

_FOR_ANY = re.compile(r"\bfor\s*\(")
# range-for: `for (auto X : Y)` or `for (const auto& X : Y)`
_FOR_RANGE = re.compile(r"\bfor\s*\(\s*(?:const\s+)?auto\b[^;{)]*?:")


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    total = 0
    range_n = 0
    for content in by_path.values():
        total += len(_FOR_ANY.findall(content))
        range_n += len(_FOR_RANGE.findall(content))
    if total == 0:
        return None
    return float(range_n / total)
