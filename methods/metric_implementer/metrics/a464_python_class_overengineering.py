"""a464: Python class over-engineering penalty.

NEGATIVE-direction metric: detects the "wrap a few attributes in a class"
smell that experienced reviewers often call out -- a class whose body is
just ``__init__`` (or ``__init__`` + dunder helpers) and nothing else.
Such a class is almost always better written as a ``dataclass``, a
``NamedTuple``, or just a ``dict``.

For each ``class`` we ask: is the body ONLY assignments + ``__init__`` +
dunders (``__repr__``, ``__eq__``, ...)? If yes, that's an over-engineered
class.

Score = 1 - (over_engineered / total_classes), so HIGH is good (no
over-engineering).

Classification: THIN, structural.
"""
from __future__ import annotations

import ast
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a464"
ASPECT_NAME = "Python class over-engineering (init+attrs only)"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_DUNDER = re.compile(r"^__[A-Za-z0-9_]+__$")


def _is_overengineered(cls: ast.ClassDef) -> bool:
    # All body statements are either Pass, Assign (class attr), AnnAssign,
    # or FunctionDef whose name is dunder. NO non-dunder methods.
    has_init = False
    only_dunder_methods = True
    has_real_method = False
    for stmt in cls.body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Expr)):
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name == "__init__":
                has_init = True
                continue
            if _DUNDER.match(stmt.name):
                continue
            has_real_method = True
            only_dunder_methods = False
        else:
            # nested classes, etc -- not over-engineered
            return False
    # Over-engineered iff has __init__ and only dunder methods otherwise
    return has_init and only_dunder_methods and not has_real_method


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if not classes:
        return None
    bad = sum(1 for c in classes if _is_overengineered(c))
    return float(1.0 - bad / len(classes))


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
