"""a512: scanf/printf vs cin/cout I/O style ratio.

C-style I/O (scanf/printf/puts/getchar/putchar) vs C++-stream I/O
(cin/cout/getline/endl). Returns the proportion of C++-stream calls:

  cpp_count / (cpp_count + c_count + epsilon)

So a fully C++-stream user scores ~1.0; a fully C-style user (common in
competition programming) scores ~0.0. Mixed: in between.

Univariate on CANDIDATE CODE ALONE. CLASSIFICATION: THIN.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a512"
ASPECT_NAME = "C++ stream vs C I/O ratio"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]

_C_IO = re.compile(
    r"\b(scanf|printf|fscanf|fprintf|sprintf|snprintf|puts|gets|"
    r"getchar|putchar|fgets|fputs)\s*\("
)
_CPP_IO = re.compile(
    r"\b(cin|cout|cerr|clog|getline|endl)\b"
)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    c_count = 0
    cpp_count = 0
    for content in by_path.values():
        c_count += len(_C_IO.findall(content))
        cpp_count += len(_CPP_IO.findall(content))
    total = c_count + cpp_count
    if total == 0:
        return None  # no I/O at all — abstain
    return float(cpp_count / total)
