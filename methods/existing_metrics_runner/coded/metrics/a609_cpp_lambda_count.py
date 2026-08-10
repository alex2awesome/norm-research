"""a609: Lambda expression count.

Integer count of C++ lambdas `[...](...) { ... }`. Modern C++ idiom often
absent in older competitive submissions.

Uses tree-sitter when available, regex fallback otherwise.

CLASSIFICATION: PARTIALLY_THIN.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a609"
ASPECT_NAME = "Lambda expression count"
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
    total = 0.0
    found = False
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found = True
        total += f["lambda_count_ts"]
    if not found:
        return None
    return float(total)
