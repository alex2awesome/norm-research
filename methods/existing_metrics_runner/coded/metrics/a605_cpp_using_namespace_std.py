"""a605: `using namespace std;` flag.

Binary: 1.0 if `using namespace std;` appears at any scope; 0.0 otherwise.
Strong signal of competitive-programming style (production C++ avoids it).

CLASSIFICATION: THIN — exact regex.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext
from .._cpp_struct_analyzer import CPP_EXTS, analyze, looks_like_cpp

ASPECT_ID = "a605"
ASPECT_NAME = "using namespace std flag"
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
    any_using = 0.0
    for content in by_path.values():
        if not looks_like_cpp(content):
            continue
        f = analyze(content)
        if not f:
            continue
        found = True
        if f["using_namespace_std"]:
            any_using = 1.0
    if not found:
        return None
    return float(any_using)
