"""a511: `#define int long long` macro (Olympiad overflow hack).

Common ICPC/Olympiad hack to avoid manual long-long type-casting on every
literal. Binary 1.0 / 0.0; abstains on non-C++ diffs.

CLASSIFICATION: THIN — exact regex match.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a511"
ASPECT_NAME = "#define int long long"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]
_PAT = re.compile(r"#\s*define\s+int\s+long\s+long\b")


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    for content in by_path.values():
        if _PAT.search(content):
            return 1.0
    return 0.0
