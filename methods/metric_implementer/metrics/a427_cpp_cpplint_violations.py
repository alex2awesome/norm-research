"""a427: cpplint Google-style violations density.

Score = exp(-violations_per_loc * 30).

Tier 3. THIN.
"""
from __future__ import annotations
import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a427"
ASPECT_NAME = "C++ cpplint violations density"
TIER = 3
TOOLS = ["cpplint"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
# REGEX_OK: tool_output — counts lines emitted by cpplint stderr (it prints
#                       "file:line: msg [category] [severity]").
_LINE_RE = re.compile(r"^[^:]+:\d+:", re.M)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    if not have_tool("cpplint"):
        return None
    import shutil
    td = write_temp_files(by_path)
    try:
        total_v = 0
        total_loc = 0
        for p in td.iterdir():
            loc = max(1, p.read_text(errors="ignore").count("\n"))
            total_loc += loc
            rc, stdout, stderr = run(
                ["cpplint", "--quiet", str(p)], timeout=8.0)
            total_v += len(_LINE_RE.findall(stderr))
        if total_loc == 0:
            return None
        return math.exp(-(total_v / total_loc) * 30.0)
    finally:
        shutil.rmtree(td, ignore_errors=True)
