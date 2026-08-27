"""a515: Preprocessor `#define` density.

Count `#define` lines divided by total LOC. High density is characteristic
of competitive-programming hack macros (`#define ll long long`, `#define rep`
`#define endl '\\n'`, etc.).

Returns clamp(define_lines / max(1, loc) * 10, 0, 1). So 10% of lines being
defines saturates the score.

CANDIDATE CODE ALONE. CLASSIFICATION: THIN.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a515"
ASPECT_NAME = "Preprocessor define density"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["C", "C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
_DEFINE = re.compile(r"^\s*#\s*define\b", flags=re.MULTILINE)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    loc = 0
    defs = 0
    for content in by_path.values():
        loc += sum(1 for ln in content.splitlines() if ln.strip())
        defs += len(_DEFINE.findall(content))
    if loc == 0:
        return None
    return float(max(0.0, min(1.0, (defs / loc) * 10.0)))
