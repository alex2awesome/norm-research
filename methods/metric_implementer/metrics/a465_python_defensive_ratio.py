"""a465: Defensive-programming density.

How often does the answer use defensive constructs (input validation,
``try/except``, ``isinstance`` checks, None-guards, explicit ``raise``
in ``if`` branches)? Counted per statement and squashed.

Per the CR.SE deep dive, winners tend to be moderately defensive --
neither bare loops (loser) nor swallowing-all (loser). We don't try to
detect "swallow all"; just rate of defensive constructs vs total stmts.

Final score = tanh(defensive_per_stmt * 8) -- defensive_rate=0.125 -> 0.76.

Classification: THIN.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a465"
ASPECT_NAME = "Python defensive-construct density"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]


def _ast_counts(tree: ast.AST):
    total = 0
    defensive = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            total += 1
        if isinstance(node, ast.Try):
            defensive += 1
        elif isinstance(node, ast.Assert):
            defensive += 1
        elif isinstance(node, ast.Raise):
            defensive += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "isinstance":
            defensive += 1
        elif isinstance(node, ast.Compare):
            for cmp_op, comp in zip(node.ops, node.comparators):
                if isinstance(cmp_op, (ast.Is, ast.IsNot)) \
                        and isinstance(comp, ast.Constant) \
                        and comp.value is None:
                    defensive += 1
                    break
    return total, defensive


_RX_STMT = re.compile(r"^[^\n#]*\S", re.M)
_RX_DEFENSIVE = [
    re.compile(r"^\s*try\s*:", re.M),
    re.compile(r"^\s*except\b", re.M),
    re.compile(r"^\s*raise\b", re.M),
    re.compile(r"^\s*assert\b", re.M),
    re.compile(r"\bisinstance\s*\("),
    re.compile(r"\b(?:is\s+None|is\s+not\s+None)\b"),
]


def _regex_counts(text: str):
    total = max(1, len(_RX_STMT.findall(text)))
    d = sum(len(rx.findall(text)) for rx in _RX_DEFENSIVE)
    return total, d


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    try:
        tree = ast.parse(text)
        total, defensive = _ast_counts(tree)
    except SyntaxError:
        total, defensive = _regex_counts(text)
    if total == 0:
        return None
    rate = defensive / total
    return float(math.tanh(rate * 8.0))


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
