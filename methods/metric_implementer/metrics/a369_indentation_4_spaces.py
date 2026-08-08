"""a369: Indentation — 4 spaces per level.

Norm: indent each block/level by exactly 4 spaces, consistently. This is
the PEP 8 convention for Python. Other languages disagree about width and
about tabs-vs-spaces (Go uses tabs, JS commonly uses 2), so we restrict
this metric to Python — the only language where "4 spaces per level" is
the canonical, deterministic rule.

Tool: `ruff` with the indentation-specific rule subset:
  - W191  indentation contains tabs
  - E111  indentation is not a multiple of 4
  - E114  indentation is not a multiple of 4 (comment)
  - E117  over-indented

Norm conformance: 1.0 = zero indentation violations across added Python
lines, decaying with violation density (same exp-decay shape as a181, but
scoped to indentation rules only). Mirrors a181_warnings_as_errors but
narrows the rule set.

Applies only to diffs that add at least one Python (.py / .pyi) file.
Other languages -> applies()=False (router-gated; runner records NaN).
"""
from __future__ import annotations

import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a369"
ASPECT_NAME = "Indentation: 4 spaces per level"
TIER = 3
TOOLS = ["ruff"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"
# PARTIALLY_THIN: the rule "4 spaces per level" is deterministically
# checkable in Python (PEP 8, ruff W191/E111/E114/E117), but the norm as
# stated is language-conditional — we honestly cover only Python here.

PY_EXTS = [".py", ".pyi"]

# Ruff codes that target indentation specifically.
INDENT_RULES = "W191,E111,E114,E117"


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    if not have_tool("ruff"):
        return None
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    total_lines = sum(1 + t.count("\n") for t in by_path.values())
    if total_lines == 0:
        return None
    td = write_temp_files(by_path)
    try:
        rc, out, _ = run(
            ["ruff", "check", "--no-cache",
             "--select", INDENT_RULES,
             "--output-format=concise",
             "--exit-zero", str(td)],
            timeout=15.0,
        )
        if rc < 0:
            return None
        # ruff concise output: one diagnostic per non-empty line.
        n_violations = sum(1 for ln in out.splitlines() if ln.strip())
        density = n_violations / max(total_lines, 1)
        # Indentation violations are usually localized and rare in normal
        # code; use the same shape as a181 so feature scales align.
        return float(math.exp(-density * 20.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
