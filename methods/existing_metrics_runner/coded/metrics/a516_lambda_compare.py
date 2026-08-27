"""a516: Lambda-in-sort/compare presence.

Detects whether the candidate uses a lambda in conjunction with a sort,
stable_sort, nth_element, partial_sort, or similar STL ordering call. This
is a stylistic marker — competition code often uses bespoke comparators.

Binary signal: 1.0 if any sort()/stable_sort()/etc call has a lambda among
its arguments (within ~120 chars of the call); 0.0 otherwise.

CANDIDATE CODE ALONE. CLASSIFICATION: THIN.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a516"
ASPECT_NAME = "Lambda-in-sort/compare"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]

_SORT_CALL = re.compile(
    r"\b(sort|stable_sort|partial_sort|nth_element|partition|"
    r"upper_bound|lower_bound|priority_queue)\b"
)
# crude lambda regex
_LAMBDA = re.compile(r"\[\s*[\w&=, ]*\]\s*\([^)]*\)\s*(?:->|\{|mutable)")


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    for content in by_path.values():
        for sm in _SORT_CALL.finditer(content):
            # search lambda within 240 chars of the call
            window = content[sm.start():sm.start() + 240]
            if _LAMBDA.search(window):
                return 1.0
    return 0.0
