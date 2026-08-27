"""a426: clang-format compliance.

Run `clang-format --output-replacements-xml` and count number of suggested
replacements. Score = 1 / (1 + n_replacements / max(loc, 1) * 100).

Tier 3. THIN (clang-format is deterministic given a style).
"""
from __future__ import annotations
import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a426"
ASPECT_NAME = "C++ clang-format compliance"
TIER = 3
TOOLS = ["clang-format"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
# REGEX_OK: tool_output — count <replacement> tags in clang-format XML.
_REPL_RE = re.compile(r"<replacement ")


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    if not have_tool("clang-format"):
        return None
    import shutil
    td = write_temp_files(by_path)
    try:
        total_repl = 0
        total_loc = 0
        for p in td.iterdir():
            txt = p.read_text(errors="ignore")
            loc = max(1, txt.count("\n"))
            total_loc += loc
            rc, stdout, stderr = run(
                ["clang-format", "--output-replacements-xml", "-style=Google",
                 str(p)], timeout=5.0)
            if rc != 0 and not stdout:
                continue
            n = len(_REPL_RE.findall(stdout))
            total_repl += n
        if total_loc == 0:
            return None
        density = total_repl / total_loc
        return 1.0 / (1.0 + density * 5.0)
    finally:
        shutil.rmtree(td, ignore_errors=True)
