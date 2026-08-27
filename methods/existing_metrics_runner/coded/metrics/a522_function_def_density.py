"""a522: Function-definition density per 100 lines of code.

Univariate function of CANDIDATE CODE ALONE. Counts top-level and nested
function/lambda definitions in the candidate, divided by max(1, NLOC) and
scaled to per-100-line density.

  Python: `def NAME(...)`, `async def NAME(...)`, `lambda ...:`
  C/C++:  `RETTYPE NAME(...)` { ... }, lambdas `[]( ... ) { ... }`

A high density implies modular / well-structured code; a single monolithic
`main()` implies low density (typical of competitive submissions).

Return: float >= 0.0 (defs per 100 non-blank lines).

applies() True iff at least one .py/.cpp/.cc/etc. file is present.

CLASSIFICATION: PARTIALLY_THIN — heuristic regex on stripped code.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a522"
ASPECT_NAME = "Function definition density per 100 LOC"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "PARTIALLY_THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

_STR_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

# Python defs
_PY_DEF = re.compile(r"^\s*(?:async\s+)?def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
                     re.MULTILINE)
_PY_LAMBDA = re.compile(r"\blambda\b")

# C/C++ function definition: at top of a line, a returntype-like prefix
# followed by `name(` and an opening brace before next semicolon. This is
# the typical hardest-to-get-right regex in C++; we use a conservative
# heuristic that allows up to 3 type-tokens before the function name.
_CPP_DEF = re.compile(
    r"^[ \t]*"
    r"(?:(?:static|inline|virtual|constexpr|explicit|extern|friend)\s+)*"
    r"(?:[A-Za-z_][A-Za-z0-9_:<>,\s\*&]*\s+)"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{)]*\)\s*(?:const\s*)?\{",
    re.MULTILINE,
)
# C++ lambda
_CPP_LAMBDA = re.compile(r"\[[^\]]*\]\s*\([^;{)]*\)\s*(?:->\s*[A-Za-z_:<>0-9\s\*&,]+)?\s*\{")


def _strip_for_regex(content: str) -> str:
    s = re.sub(r"/\*.*?\*/", " ", content, flags=re.DOTALL)
    s = _STR_LIT.sub(' "" ', s)
    out = []
    for ln in s.splitlines():
        for marker in ("//",):
            idx = ln.find(marker)
            if idx >= 0:
                ln = ln[:idx]
        out.append(ln)
    return "\n".join(out)


def _nloc(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip())


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    total_defs = 0
    total_nloc = 0
    for path, content in by_path.items():
        if not content.strip():
            continue
        is_py = path.lower().endswith((".py", ".pyi"))
        s = _strip_for_regex(content)
        nloc = _nloc(s)
        total_nloc += nloc
        if is_py:
            total_defs += len(_PY_DEF.findall(s))
            total_defs += len(_PY_LAMBDA.findall(s))
        else:
            total_defs += len(_CPP_DEF.findall(s))
            total_defs += len(_CPP_LAMBDA.findall(s))
    if total_nloc == 0:
        return None
    return float(100.0 * total_defs / total_nloc)
