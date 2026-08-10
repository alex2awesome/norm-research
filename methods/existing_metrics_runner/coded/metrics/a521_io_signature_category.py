"""a521: I/O signature category as an ordinal feature.

Univariate function of CANDIDATE CODE ALONE. Classifies the candidate's
I/O style into one of four buckets:

  1 = fast I/O           (sync_with_stdio(false) / scanf / sys.stdin)
  2 = standard streams   (cin/cout, print() default)
  3 = ad-hoc / no I/O    (none of the above detected)
  4 = mixed              (both fast-style and standard-style present)

applies() True iff any read/write pattern is detected. Otherwise abstain
(no I/O at all — common in headers, library code).

Note a512 already measures the C++ stream/C I/O ratio. This metric is
*orthogonal*: it includes Python and singles out the FAST-I/O bucket
(sync_with_stdio(false) + tie(0) is a Luogu/Codeforces hallmark).

CLASSIFICATION: THIN.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a521"
ASPECT_NAME = "I/O signature category"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

_STR_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

# Fast I/O signals
_CPP_FAST = re.compile(
    r"sync_with_stdio\s*\(\s*(?:false|0)\s*\)|"
    r"\bscanf\s*\(|\bprintf\s*\(|\bgetchar\b|\bputchar\b|"
    r"\btie\s*\(\s*(?:0|NULL|nullptr)\s*\)"
)
_PY_FAST = re.compile(
    r"sys\.stdin|sys\.stdout|input\s*=\s*sys\.stdin\.readline|"
    r"input\s*=\s*sys\.stdin|readlines\s*\(\s*\)|"
    r"sys\.stdout\.write"
)

# Standard stream signals
_CPP_STD = re.compile(
    r"\bcin\b|\bcout\b|\bcerr\b|\bgetline\s*\(|\bendl\b"
)
_PY_STD = re.compile(r"(?<![A-Za-z_])print\s*\(|(?<![A-Za-z_])input\s*\(\s*\)")


def _strip(content: str) -> str:
    s = re.sub(r"/\*.*?\*/", " ", content, flags=re.DOTALL)
    s = _STR_LIT.sub(' "" ', s)
    out = []
    for ln in s.splitlines():
        for marker in ("//",):
            idx = ln.find(marker)
            if idx >= 0:
                ln = ln[:idx]
        # NOTE: don't strip `#` here — Python comments may include input()
        # but # in C++ may be a preprocessor directive (#include). We strip
        # # comments only on Python files at a higher level if needed.
        out.append(ln)
    return "\n".join(out)


def _detect(content: str, is_py: bool):
    s = _strip(content)
    fast = False
    std = False
    if is_py:
        if _PY_FAST.search(s):
            fast = True
        if _PY_STD.search(s):
            std = True
    else:
        if _CPP_FAST.search(s):
            fast = True
        if _CPP_STD.search(s):
            std = True
    return fast, std


def applies(diff_text: str) -> bool:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return False
    for path, content in by_path.items():
        is_py = path.lower().endswith((".py", ".pyi"))
        fast, std = _detect(content, is_py)
        if fast or std:
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    any_fast = False
    any_std = False
    for path, content in by_path.items():
        is_py = path.lower().endswith((".py", ".pyi"))
        f, st = _detect(content, is_py)
        any_fast = any_fast or f
        any_std = any_std or st
    if any_fast and any_std:
        return 4.0  # mixed
    if any_fast:
        return 1.0  # fast
    if any_std:
        return 2.0  # standard
    # We declared applies() False above when neither set; this branch
    # should be unreachable but keep it defensive.
    return 3.0  # ad-hoc / no I/O detected
