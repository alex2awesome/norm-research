"""a466: PEP-8 naming compliance for variables and functions.

For each LOCAL assignment target and each function name, classify as
snake_case (good), camelCase / PascalCase (bad for non-class names), or
SHOUTY_CASE (acceptable for module-level constants but penalized for
locals). Imports are excluded -- you can't rename a third-party symbol.

Score = snake_case / (snake_case + camel_case + 0.5*shouty), clipped to
[0, 1].

Classification: THIN.
"""
from __future__ import annotations

import ast
import re
from typing import Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a466"
ASPECT_NAME = "Python snake_case naming compliance"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_SNAKE = re.compile(r"^_?[a-z][a-z0-9_]*$")
_CAMEL = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]+)+$")
_PASCAL = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+$")
_SHOUTY = re.compile(r"^[A-Z][A-Z0-9_]+$")


def _classify(name: str) -> str:
    if not name or name.startswith("__"):
        return "skip"
    if _SNAKE.match(name):
        return "snake"
    if _CAMEL.match(name) or _PASCAL.match(name):
        return "camel"
    if _SHOUTY.match(name):
        return "shouty"
    return "skip"


def _collect_names(tree: ast.AST):
    snake = camel = shouty = 0
    imported = set()
    # First pass: collect imported names so we skip them
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])

    for node in ast.walk(tree):
        # Function defs
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name in imported:
                continue
            kind = _classify(name)
            if kind == "snake":
                snake += 1
            elif kind == "camel":
                camel += 1
            elif kind == "shouty":
                shouty += 1
        # Assignment targets
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                names = _names_from_target(tgt)
                for nm in names:
                    if nm in imported:
                        continue
                    kind = _classify(nm)
                    if kind == "snake":
                        snake += 1
                    elif kind == "camel":
                        camel += 1
                    elif kind == "shouty":
                        shouty += 1
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nm = node.target.id
            if nm in imported:
                continue
            kind = _classify(nm)
            if kind == "snake":
                snake += 1
            elif kind == "camel":
                camel += 1
            elif kind == "shouty":
                shouty += 1
    return snake, camel, shouty


def _names_from_target(tgt):
    if isinstance(tgt, ast.Name):
        return [tgt.id]
    if isinstance(tgt, (ast.Tuple, ast.List)):
        out = []
        for el in tgt.elts:
            out.extend(_names_from_target(el))
        return out
    if isinstance(tgt, ast.Starred):
        return _names_from_target(tgt.value)
    return []


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    snake, camel, shouty = _collect_names(tree)
    total = snake + camel + 0.5 * shouty
    if total == 0:
        return None
    return float(min(1.0, snake / total))


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
