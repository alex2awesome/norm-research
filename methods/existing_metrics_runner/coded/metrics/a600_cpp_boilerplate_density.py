"""a600: Competitive-programming boilerplate macro density (per 100 LOC).

Counts the count of `#define int long long`, `typedef long long ll`, and
`#define rep(...)` / `#define FOR(...)` shortcut macros per 100 NLOC.
Higher => more competitive-programming idiom.

Returns float >= 0 on C++ candidates; abstains otherwise.

CLASSIFICATION: THIN — regex-derived, validated.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a600"
ASPECT_NAME = "Competitive-boilerplate macro density per 100 LOC"
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
    total_count = 0.0
    total_nloc = 0.0
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        total_count += f["boilerplate_count"]
        total_nloc += f["nloc_strip"]
    if total_nloc <= 0:
        return None
    return float(100.0 * total_count / total_nloc)
