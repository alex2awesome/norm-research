"""a460: Python review-post register density.

Counts surface markers that distinguish a Code-Review-Stack-Exchange-style
review post from a single straight-up code dump:

  - `>>>` interactive REPL prompts at start of a line
  - `...` continuation prompts at start of a line
  - elision dots `...` appearing as a stand-alone statement (the
    "rest of the code unchanged" idiom)
  - number of distinct code blocks (separated by 1+ blank lines)

We normalize to "per 100 lines" and squash with `tanh` so a busy review
gives ~1 and a single bare snippet gives near 0.

Classification: THIN, register-only. No semantic claim.
Tier 2: pure regex on added Python source.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a460"
ASPECT_NAME = "Python review-post register density"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_REPL = re.compile(r"^[ \t]*>>>[ \t]", re.M)
_REPL_CONT = re.compile(r"^[ \t]*\.\.\.[ \t]", re.M)
# elision/placeholder statement -- a line containing ONLY `...`
_ELIDE = re.compile(r"^[ \t]*\.\.\.[ \t]*$", re.M)
# block separator: 2+ newlines
_BLOCK_SEP = re.compile(r"\n[ \t]*\n")


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    n_lines = max(1, text.count("\n"))
    repl = len(_REPL.findall(text))
    cont = len(_REPL_CONT.findall(text))
    elide = len(_ELIDE.findall(text))
    blocks = len(_BLOCK_SEP.findall(text)) + 1
    # raw register density per 100 lines
    raw = (repl + cont + elide + max(0, blocks - 1)) * 100.0 / n_lines
    # squash: tanh of raw/8 -- raw=8 -> 0.66, raw=16 -> 0.92
    return float(math.tanh(raw / 8.0))


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
