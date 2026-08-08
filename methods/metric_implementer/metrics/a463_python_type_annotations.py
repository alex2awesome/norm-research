"""a463: Python function annotation coverage (lightweight).

For each function definition in the file, ask: is there at least one
annotated argument, OR a ``->`` return annotation? Score = fraction of
function defs satisfying that. This is intentionally laxer than a303
(which measures per-arg coverage and ``Any``-overuse): we want a register
signal -- does this answer "do types at all".

Falls back to regex on ``def NAME(... -> ...:`` and per-arg
``name: type`` when AST fails.

Classification: THIN.
"""
from __future__ import annotations

import ast
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a463"
ASPECT_NAME = "Python function annotation coverage (lax)"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]


def _fn_has_annotation(fn: ast.AST) -> bool:
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if fn.returns is not None:
        return True
    args = fn.args
    for arg_list in (args.args, args.posonlyargs, args.kwonlyargs):
        for a in arg_list or []:
            if a.annotation is not None and a.arg not in ("self", "cls"):
                return True
    return False


_RX_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(->[^:]*)?:",
                     re.M)
_RX_ARG_ANNOT = re.compile(r"\b([A-Za-z_]\w*)[ \t]*:[ \t]*[A-Za-z_\[]")


def _regex_score(text: str) -> Optional[float]:
    defs = _RX_DEF.findall(text)
    if not defs:
        return None
    annotated = 0
    for _, args, ret in defs:
        if ret and ret.strip():
            annotated += 1
            continue
        if _RX_ARG_ANNOT.search(args or ""):
            annotated += 1
    return annotated / len(defs)


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _regex_score(text)
    fns = [n for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not fns:
        return None
    annotated = sum(1 for fn in fns if _fn_has_annotation(fn))
    return float(annotated / len(fns))


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
