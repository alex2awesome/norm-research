"""a445: C++ class-Solution wrapper stripped.

Returns 1.0 when the LeetCode-style ``class Solution { public: ... };``
wrapper is absent (bare free function), 0.0 when it is present.

Captures the LC-community style of submitting a bare function rather than
the canonical class wrapper. Abstains when the snippet does not look like
a single LeetCode-style answer (e.g. multiple top-level classes, no
functions at all).

Tier 1. THIN.
"""
from __future__ import annotations
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a445"
ASPECT_NAME = "C++ class-Solution wrapper stripped"
TIER = 1
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]

# REGEX_OK: tool_output — surface-syntax check for the LeetCode wrapper.
RE_CLASS_SOLUTION = re.compile(r"\bclass\s+Solution\b\s*[\{:]")
# Cheap function-definition probe: anything that looks like a top-level
# or method-level signature followed by a brace. We only use this to
# decide whether the snippet has *any* code to score.
RE_FUNC_LIKE = re.compile(r"\)\s*(const)?\s*\{")


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    saw_any_code = False
    saw_wrapper = False
    for content in by_path.values():
        if RE_FUNC_LIKE.search(content):
            saw_any_code = True
        if RE_CLASS_SOLUTION.search(content):
            saw_wrapper = True
    if not saw_any_code:
        return None
    return 0.0 if saw_wrapper else 1.0
