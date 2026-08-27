"""a462: Pythonic idiom density.

Counts occurrences of the following Pythonic constructs and normalizes
per 100 LOC:

  - list / dict / set comprehensions
  - generator expressions
  - f-strings (formatted_value inside string)
  - walrus operator (``:=``)
  - decorators

We prefer ``ast`` because tree-sitter sometimes mis-classifies generator
exprs vs tuples. If ``ast.parse`` fails (incomplete snippet, mid-function
context) we fall back to regex over the source.

Final score: ``tanh(idioms_per_100loc / 10)`` -- 10/100 loc -> 0.76.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a462"
ASPECT_NAME = "Pythonic idiom density"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]


def _ast_count(tree: ast.AST) -> int:
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                              ast.GeneratorExp)):
            n += 1
        elif isinstance(node, ast.JoinedStr):  # f-string
            n += 1
        elif isinstance(node, ast.NamedExpr):  # walrus
            n += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
            n += len(getattr(node, "decorator_list", []) or [])
    return n


_RX_FSTR = re.compile(r"""(?:^|[^A-Za-z_0-9])[fF][rR]?['"]""")
_RX_WALRUS = re.compile(r":=")
_RX_DEC = re.compile(r"^[ \t]*@\w", re.M)
_RX_COMP = re.compile(
    r"\bfor\b[^:\n]*\b(in)\b",  # rough; will under-count nested comps
)


def _regex_count(text: str) -> int:
    n = 0
    n += len(_RX_FSTR.findall(text))
    n += len(_RX_WALRUS.findall(text))
    n += len(_RX_DEC.findall(text))
    # comprehensions are tricky w/o AST; approximate by `[... for ... in ...]`
    n += len(re.findall(r"[\[\{][^\[\]\{\}\n]*\bfor\b[^\[\]\{\}\n]*\bin\b",
                         text))
    n += len(re.findall(r"\([^()\n]*\bfor\b[^()\n]*\bin\b[^()\n]*\)", text))
    return n


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    n_lines = max(1, text.count("\n"))
    try:
        tree = ast.parse(text)
        n = _ast_count(tree)
    except SyntaxError:
        n = _regex_count(text)
    rate = n * 100.0 / n_lines
    return float(math.tanh(rate / 10.0))


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    scs = [s for s in (_file_score(c) for c in by_path.values()) if s is not None]
    if not scs:
        return None
    return float(sum(scs) / len(scs))
