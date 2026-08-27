"""a501: Control-flow fingerprint — branching density.

Language-agnostic regex-counter over the added lines: counts the number of
control-flow keywords (for/while/if/else/switch/case/return/goto/break/
continue) and normalizes by lines of code (LOC). Returns a univariate
"branching density" score in [0,1].

A high score means dense branching (many decisions per line — often hand
written competition code with custom corner-case handling); a low score
means mostly straight-line code (often library wrappers or arithmetic).
This is a function of the CANDIDATE CODE ALONE.

We use a word-boundary regex restricted to lines that look like CODE (skipping
lines whose stripped content begins with `#` (C/Python comment), `//`, `/*`,
`*`). String/char literals are stripped first so e.g. `"if"` inside a printf
doesn't count.

Score = clamp(density / TARGET_DENSITY, 0, 1), where TARGET_DENSITY = 0.5
(half the lines having a control-flow keyword is "saturated").

CLASSIFICATION: THIN — deterministic count over a regex-defined alphabet.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a501"
ASPECT_NAME = "Control-flow branching density"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

CONTROL_KW = re.compile(
    r"\b(for|while|if|elif|else|switch|case|return|goto|break|continue|"
    r"try|except|catch)\b"
)
_STR_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def _strip_comments_and_strings(line: str, lang_c: bool) -> str:
    s = _STR_LIT.sub("", line)
    if lang_c:
        # strip from // onward
        idx = s.find("//")
        if idx >= 0:
            s = s[:idx]
        # strip /* ... */ (single-line case)
        s = re.sub(r"/\*.*?\*/", "", s)
    else:
        # python # comment
        idx = s.find("#")
        if idx >= 0:
            s = s[:idx]
    return s


def _is_real_code_line(stripped: str) -> bool:
    if not stripped:
        return False
    return True


def _density(content: str, is_c: bool) -> Optional[float]:
    loc = 0
    branches = 0
    for raw in content.splitlines():
        s = _strip_comments_and_strings(raw, is_c).strip()
        if not _is_real_code_line(s):
            continue
        loc += 1
        branches += len(CONTROL_KW.findall(s))
    if loc == 0:
        return None
    return branches / loc


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    total_loc = 0
    total_branches = 0
    for path, content in by_path.items():
        is_c = not path.lower().endswith((".py", ".pyi"))
        for raw in content.splitlines():
            s = _strip_comments_and_strings(raw, is_c).strip()
            if not s:
                continue
            total_loc += 1
            total_branches += len(CONTROL_KW.findall(s))
    if total_loc == 0:
        return None
    density = total_branches / total_loc
    # saturate at density=0.5 (half the lines branch)
    return float(max(0.0, min(1.0, density / 0.5)))
