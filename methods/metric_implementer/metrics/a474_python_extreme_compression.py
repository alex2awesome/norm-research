"""a474: Python extreme-compression / golfed-one-liner penalty.

Captures the "over-clever" register: cramming a full algorithm into
one expression. Editorial Python prefers a readable 5-15 line solution,
not a `reduce(lambda a, b: ..., xs, init)` one-liner.

Detected patterns (AST-first, regex fallback):
  - `reduce(lambda ...)` call
  - walrus-chained expression (`(a := ...)` used twice or more)
  - metaclass `type('', (), {...})` trick
  - lambda that itself contains a lambda (lambda-of-lambda)
  - a method/function whose entire body is a single `return` of a
    multi-clause comprehension or chained ternaries
  - `Solution.<method>` whose body is one expression statement

Score: tanh(matches / 2). 0 -> 0 (no golfing). 1 match -> 0.46.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a474"
ASPECT_NAME = "Python extreme-compression / golfed register"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
APPLIES_TO_EXTS = [".py", ".pyi"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]


def _walk(node):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk(child)


def _count_lambdas(node) -> int:
    return sum(1 for n in _walk(node) if isinstance(n, ast.Lambda))


def _is_ternary_chain(expr) -> bool:
    # IfExp nested inside IfExp body OR orelse
    if not isinstance(expr, ast.IfExp):
        return False
    return isinstance(expr.body, ast.IfExp) or isinstance(expr.orelse, ast.IfExp)


def _is_complex_comp(expr) -> bool:
    if not isinstance(expr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return False
    return len(expr.generators) >= 2 or any(g.ifs for g in expr.generators)


def _ast_signals(tree) -> int:
    sig = 0
    walruses = 0
    for n in _walk(tree):
        # reduce(lambda ...)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "reduce":
            if any(isinstance(a, ast.Lambda) for a in n.args):
                sig += 1
        # walrus chain
        if isinstance(n, ast.NamedExpr):
            walruses += 1
        # metaclass `type('', (), {...})` trick
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "type" and len(n.args) == 3):
            sig += 1
        # lambda-of-lambda
        if isinstance(n, ast.Lambda) and _count_lambdas(n.body) >= 1:
            sig += 1
        # single-statement function body that is a complex expression
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [s for s in n.body if not isinstance(s, ast.Expr)
                    or not (isinstance(s.value, ast.Constant)
                            and isinstance(s.value.value, str))]
            if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
                rv = body[0].value
                if _is_complex_comp(rv) or _is_ternary_chain(rv):
                    sig += 1
                elif _count_lambdas(rv) >= 1:
                    sig += 1
    if walruses >= 2:
        sig += 1
    return sig


# regex fallbacks (when ast.parse fails)
_RX_REDUCE_LAMBDA = re.compile(r"\breduce\s*\(\s*lambda\b")
_RX_TYPE_META = re.compile(r"\btype\s*\(\s*['\"][^'\"]*['\"]\s*,\s*\(")
_RX_WALRUS = re.compile(r":=")


def _regex_signals(text: str) -> int:
    sig = 0
    sig += len(_RX_REDUCE_LAMBDA.findall(text))
    sig += len(_RX_TYPE_META.findall(text))
    if len(_RX_WALRUS.findall(text)) >= 2:
        sig += 1
    return sig


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    try:
        tree = ast.parse(text)
        sig = _ast_signals(tree)
    except SyntaxError:
        sig = _regex_signals(text)
    return float(math.tanh(sig / 2.0))


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
