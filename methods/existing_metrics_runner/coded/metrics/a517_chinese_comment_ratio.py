"""a517: Chinese-character ratio in comments (any language).

Many Luogu submissions are commented in Chinese (Luogu is a Chinese
competitive programming platform). This metric measures the fraction of
characters in comments that fall in the CJK Unified Ideographs Unicode
block.

Returns chinese_chars / max(1, total_comment_chars). 0 if no comments.

Note: the source distribution itself has comparable Luogu/CC mass, so this
metric is partly *correlated* with platform — it is intended as a
verifiable surface feature of the code, not as a label-leakage proxy.

CANDIDATE CODE ALONE. CLASSIFICATION: THIN.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a517"
ASPECT_NAME = "Chinese-character comment ratio"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

# CJK Unified Ideographs primary block
_CJK = re.compile(r"[一-鿿]")
# C-style line comment and block comment, Python # comment
_C_LINE = re.compile(r"//(.*)")
_C_BLOCK = re.compile(r"/\*(.*?)\*/", flags=re.DOTALL)
_PY_LINE = re.compile(r"#(.*)")


def _extract_comments(content: str, is_py: bool) -> str:
    parts = []
    if is_py:
        for m in _PY_LINE.finditer(content):
            parts.append(m.group(1))
    else:
        for m in _C_LINE.finditer(content):
            parts.append(m.group(1))
        for m in _C_BLOCK.finditer(content):
            parts.append(m.group(1))
    return "\n".join(parts)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    total = 0
    cjk = 0
    for path, content in by_path.items():
        is_py = path.lower().endswith((".py", ".pyi"))
        comments = _extract_comments(content, is_py)
        if not comments:
            continue
        total += len(comments)
        cjk += len(_CJK.findall(comments))
    if total == 0:
        return None
    return float(cjk / total)
