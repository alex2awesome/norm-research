"""a469: Math closed-form / bit-trick density.

Counts uses of compact numerical idioms that distinguish a "knew the
formula" answer from a "computed it in a loop" answer:

  - ``math.<x>`` calls (math.gcd, math.isqrt, math.factorial, ...)
  - ``.bit_length()`` / ``.bit_count()``
  - ``divmod(...)``
  - 3-arg ``pow(a, b, m)`` -- the modular exponentiation form
  - bitwise operators ``<<`` / ``>>`` / ``&`` / ``|`` / ``^``
  - ``Fraction``, ``Decimal`` (precision-aware arithmetic)

Per 100 LOC, squashed with tanh. Higher = denser math idioms.

Classification: THIN.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a469"
ASPECT_NAME = "Math closed-form / bit-trick density"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]


def _ast_count(tree: ast.AST) -> int:
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            # math.* call
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                    and f.value.id == "math":
                n += 1
            # method calls bit_length / bit_count
            elif isinstance(f, ast.Attribute) and f.attr in ("bit_length",
                                                              "bit_count"):
                n += 1
            elif isinstance(f, ast.Name) and f.id == "divmod":
                n += 1
            elif isinstance(f, ast.Name) and f.id == "pow" and len(node.args) == 3:
                n += 1
        elif isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr,
                           ast.BitXor)):
            n += 1
    return n


_RX_FALLBACK = [
    re.compile(r"\bmath\.\w+\s*\("),
    re.compile(r"\.bit_length\s*\("),
    re.compile(r"\.bit_count\s*\("),
    re.compile(r"\bdivmod\s*\("),
    re.compile(r"\bpow\s*\([^()]*,[^()]*,[^()]*\)"),
    re.compile(r"[^<]<<[^<]"),
    re.compile(r"[^>]>>[^>]"),
    re.compile(r"\bFraction\s*\("),
    re.compile(r"\bDecimal\s*\("),
]


def _regex_count(text: str) -> int:
    return sum(len(rx.findall(text)) for rx in _RX_FALLBACK)


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
    return float(math.tanh(rate / 6.0))


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
