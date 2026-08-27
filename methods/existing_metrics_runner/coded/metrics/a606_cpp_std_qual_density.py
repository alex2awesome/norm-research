"""a606: `std::` qualification density per 100 LOC.

Count of `std::X` references divided by NLOC, scaled to per-100. High = no
`using namespace std`, deliberate qualification; low + a605=1 = idiomatic
competitive C++. Returns 0.0 with using_namespace_std=1 (typical), >0 for
production-style code.

CLASSIFICATION: THIN — regex count.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a606"
ASPECT_NAME = "std:: qualification density per 100 LOC"
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
    total = 0.0
    nloc = 0.0
    found = False
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found = True
        total += f["std_qual_count"]
        nloc += f["nloc_strip"]
    if not found or nloc <= 0:
        return None
    return float(100.0 * total / nloc)
