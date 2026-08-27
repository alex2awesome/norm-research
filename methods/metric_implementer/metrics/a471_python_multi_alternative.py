"""a471: Multi-alternative-solution density.

A CR.SE answer often shows MULTIPLE versions of "the same code" --
"here is the naive version", then "here is the cleaner version", then
"and here is the one-liner". This is a strong winner signal that the
single-snippet bank metrics miss.

We detect alternatives heuristically by splitting the file into blank-
line-separated blocks, extracting the identifier set of each block, and
counting *sibling* pairs whose Jaccard overlap is >= 0.5. The score is
the number of such alternative pairs, normalized by log.

Classification: THIN.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Set

from ..sandbox import added_files_by_ext

ASPECT_ID = "a471"
ASPECT_NAME = "Multi-alternative solution density"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

_IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
_KEYWORDS = frozenset({
    "def", "class", "for", "if", "else", "elif", "while", "return",
    "import", "from", "as", "in", "is", "not", "and", "or", "with",
    "try", "except", "finally", "raise", "yield", "lambda", "pass",
    "break", "continue", "True", "False", "None", "self", "cls",
    "global", "nonlocal", "async", "await",
})


def _block_identifiers(block: str) -> Set[str]:
    return {m for m in _IDENT.findall(block) if m not in _KEYWORDS}


def _split_blocks(text: str) -> List[str]:
    raw_blocks = re.split(r"\n[ \t]*\n+", text)
    return [b for b in raw_blocks if b.strip() and any(
        line.strip() and not line.strip().startswith("#")
        for line in b.split("\n")
    )]


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    blocks = _split_blocks(text)
    if len(blocks) < 2:
        return 0.0
    sigs = [_block_identifiers(b) for b in blocks]
    sigs = [s for s in sigs if len(s) >= 2]
    if len(sigs) < 2:
        return 0.0
    alt_pairs = 0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            a, b = sigs[i], sigs[j]
            inter = len(a & b)
            uni = len(a | b)
            if uni == 0:
                continue
            j_idx = inter / uni
            if j_idx >= 0.5:
                alt_pairs += 1
    # log normalize: 1 pair -> 0.50, 3 pairs -> 0.78, 7 pairs -> 1.0
    return float(min(1.0, math.log(1 + alt_pairs) / math.log(1 + 7)))


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
