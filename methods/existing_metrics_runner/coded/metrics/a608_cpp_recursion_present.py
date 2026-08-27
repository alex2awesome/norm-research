"""a608: Self-call recursion present.

Binary: 1.0 if any non-`main` function calls itself (lexical self-reference
in its own body); 0.0 otherwise. Detects recursive solutions vs iterative.

Uses tree-sitter to localize 'within own body'; regex fallback uses
multi-occurrence of the function name as a call site.

CLASSIFICATION: PARTIALLY_THIN — tree-sitter primary, regex fallback.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a608"
ASPECT_NAME = "Self-call recursion present"
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
    found = False
    any_rec = 0.0
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found = True
        if f["self_call_recursion"] > 0:
            any_rec = 1.0
    if not found:
        return None
    return float(any_rec)
