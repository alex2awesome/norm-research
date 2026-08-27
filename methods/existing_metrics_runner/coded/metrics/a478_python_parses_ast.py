"""a478: Python source parses as a valid AST.

Binary: 1.0 if `ast.parse(code)` succeeds without SyntaxError, else 0.0.

A floor metric — answers that don't even parse as Python cannot be
production-grade. Most pasted solutions parse cleanly; the ones that
don't are often pseudo-code, partial snippets, or interactive REPL
transcripts. Returns None when no Python file is added.

Classification: THIN. Trivial cost.
"""
from __future__ import annotations

import ast
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a478"
ASPECT_NAME = "Python parses as valid AST"
TIER = 1
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]


def _parses(text: str) -> Optional[float]:
    if not text.strip():
        return None
    try:
        ast.parse(text)
        return 1.0
    except SyntaxError:
        return 0.0
    except Exception:
        return None


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    vals = [v for v in (_parses(c) for c in by_path.values()) if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))
