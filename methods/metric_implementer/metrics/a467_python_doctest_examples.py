"""a467: Doctest / "Example:" presence inside docstrings.

For each function or class with a docstring, count it as "exemplified" if
the docstring contains EITHER a doctest block (``>>>`` prompt) OR a
named section (Example, Examples, Usage). Score = exemplified / total
docstring-bearing defs.

Falls back to a regex over the source if AST parse fails: count
``>>>`` lines and triple-quoted blocks with an Example header.

Classification: THIN -- pure surface presence.
"""
from __future__ import annotations

import ast
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a467"
ASPECT_NAME = "Docstring example / doctest presence"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_EXAMPLE_HEADER = re.compile(r"^\s*(?:Example|Examples|Usage)\s*[:\n]",
                              re.M | re.I)
_DOCTEST = re.compile(r"^\s*>>>", re.M)


def _has_example(docstring: Optional[str]) -> bool:
    if not docstring:
        return False
    if _DOCTEST.search(docstring):
        return True
    if _EXAMPLE_HEADER.search(docstring):
        return True
    return False


def _ast_score(tree: ast.AST) -> Optional[float]:
    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef, ast.Module))]
    documented = []
    exemplified = 0
    for d in defs:
        ds = ast.get_docstring(d)
        if ds is None:
            continue
        documented.append(d)
        if _has_example(ds):
            exemplified += 1
    if not documented:
        return None
    return float(exemplified / len(documented))


_RX_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+\w", re.M)


def _regex_score(text: str) -> Optional[float]:
    n_defs = len(_RX_DEF.findall(text))
    n_doc = len(re.findall(r'""".*?"""', text, re.S)) + \
            len(re.findall(r"'''.*?'''", text, re.S))
    if n_doc == 0:
        return None
    has_doctest = 1 if _DOCTEST.search(text) else 0
    has_example = 1 if _EXAMPLE_HEADER.search(text) else 0
    return float(min(1.0, (has_doctest + has_example) / max(1, n_doc)))


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    try:
        tree = ast.parse(text)
        return _ast_score(tree)
    except SyntaxError:
        return _regex_score(text)


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
