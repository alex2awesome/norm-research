"""a414: cppcheck severity-weighted diagnostics density.

We run cppcheck on the added C/C++ files with the full enable set and weight
each finding by severity:

    error       4.0
    warning     2.0
    portability 1.0
    performance 1.0
    style       1.0
    information 0.0 (noise: missing-include, etc.)

Total weight divided by added lines gives a density. Score = exp(-density * 5).
  - 0/line     → 1.0
  - 0.1/line   → exp(-0.5) ≈ 0.607
  - 0.5/line   → exp(-2.5) ≈ 0.082

This complements a145 (buffer/string subset of cppcheck findings) by
considering *all* cppcheck issue classes — style, portability, performance —
not just the buffer-safety subset.

CLASSIFICATION THIN. Tier 3. Tool: cppcheck.

If cppcheck is missing, abstain (None).
"""
from __future__ import annotations

import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a414"
ASPECT_NAME = "cppcheck severity-weighted density"
TIER = 3
TOOLS = ["cppcheck"]
APPLIES_TO_LANGS = ["C", "C++"]
CLASSIFICATION = "THIN"

C_EXTS = [".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".c++"]

SEV_W = {
    "error": 4.0,
    "warning": 2.0,
    "portability": 1.0,
    "performance": 1.0,
    "style": 1.0,
    "information": 0.0,
}


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, C_EXTS))


def score(diff_text: str) -> Optional[float]:
    if not have_tool("cppcheck"):
        return None
    by_path = added_files_by_ext(diff_text, C_EXTS)
    if not by_path:
        return None
    total_lines = sum(1 + t.count("\n") for t in by_path.values())
    if total_lines == 0:
        return None
    td = write_temp_files(by_path)
    try:
        rc, out, err = run(
            ["cppcheck",
             "--enable=warning,style,performance,portability",
             "--quiet",
             "--template={severity}|{id}",
             str(td)],
            timeout=30.0,
        )
        if rc < 0:
            return None
        # cppcheck writes findings to stderr by default with `--template`.
        text = (err or "") + "\n" + (out or "")
        weighted = 0.0
        for line in text.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            # REGEX_OK: tool_output — parsing cppcheck stable template.
            parts = line.split("|", 1)
            sev = parts[0].strip()
            weighted += SEV_W.get(sev, 0.0)
        density = weighted / max(total_lines, 1)
        return float(math.exp(-density * 5.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
