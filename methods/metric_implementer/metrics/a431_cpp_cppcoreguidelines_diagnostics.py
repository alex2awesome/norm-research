"""a431: clang-tidy cppcoreguidelines- checks.

Run `clang-tidy -checks="-*,cppcoreguidelines-*"` per added file. Count diagnostics.
Score = exp(-diag_density * 10.0).

Tier 3. THIN.
"""
from __future__ import annotations
import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a431"
ASPECT_NAME = "C++ cppcoreguidelines-* clang-tidy density"
TIER = 3
TOOLS = ["clang-tidy"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
# REGEX_OK: tool_output — counts "warning:" lines from clang-tidy stdout.
_W_RE = re.compile(r": warning:")


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    if not have_tool("clang-tidy"):
        return None
    import shutil
    td = write_temp_files(by_path)
    try:
        total_w = 0
        total_loc = 0
        for p in td.iterdir():
            loc = max(1, p.read_text(errors="ignore").count("\n"))
            total_loc += loc
            rc, stdout, stderr = run(
                ["clang-tidy",
                 "-checks=-*,cppcoreguidelines-*",
                 "--quiet", str(p), "--",
                 "-std=c++17", "-x", "c++"],
                timeout=12.0)
            total_w += len(_W_RE.findall(stdout))
        if total_loc == 0:
            return None
        return math.exp(-(total_w / total_loc) * 10.0)
    finally:
        shutil.rmtree(td, ignore_errors=True)
