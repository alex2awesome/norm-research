"""a514: `auto` declaration density.

Counts uses of `auto` as a type specifier in declarations vs total
declarations (rough). Distinct from a513 (which is the range-for fraction
specifically) and complementary to a410 (composite modern-idioms density).

Returns auto_count / max(1, total_decl_count), clamped to [0,1].

CANDIDATE CODE ALONE. CLASSIFICATION: THIN.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a514"
ASPECT_NAME = "auto declaration density"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]
# `auto` as a type specifier (not the keyword `auto` in old K&R).
# Must be followed by an identifier (variable name) or `&` / `*` then ident.
_AUTO = re.compile(r"\bauto\b\s*(?:const\s+)?(?:&|\*|&&)?\s*\w+")
# rough total declaration: lines starting with a type-ish word followed by ident
_DECL = re.compile(
    r"\b(int|long|short|char|bool|double|float|unsigned|size_t|"
    r"string|vector|map|set|pair|auto|const)\b[\s\*&]*\w+",
)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    auto_n = 0
    decl_n = 0
    for content in by_path.values():
        auto_n += len(_AUTO.findall(content))
        decl_n += len(_DECL.findall(content))
    if decl_n == 0:
        return None
    return float(max(0.0, min(1.0, auto_n / decl_n)))
