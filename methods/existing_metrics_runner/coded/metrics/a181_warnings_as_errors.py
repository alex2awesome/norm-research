"""a181: Treat warnings/lints as errors.

We measure Python lint violation density of code ADDED in the diff using ruff
(default ruleset). Other languages: not measured here (would need eslint,
golangci-lint, etc. — add as separate per-language metrics later).

Norm conformance: 1.0 = zero violations per added line, decaying with density.

Score = exp(-violations_per_line * 20). Chosen so 0/line = 1.0,
0.05/line = 0.37, 0.1/line = 0.14.

Note: We grade ONLY the added Python code, not the whole file. This matches
how reviewers actually evaluate diffs — they care about what's being added,
not pre-existing lint debt.
"""
from __future__ import annotations

import json
import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a181"
ASPECT_NAME = "Treat warnings/lints as errors"
TIER = 3
TOOLS = ["ruff"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    if not have_tool("ruff"):
        return None
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    total_lines = sum(
        t.count("\n") + (0 if t.endswith("\n") else 1)
        for t in by_path.values()
    )
    if total_lines == 0:
        return None
    td = write_temp_files(by_path)
    try:
        # JSON avoids counting Ruff's human summary (including
        # "All checks passed!") as a diagnostic.
        rc, out, _ = run(
            ["ruff", "check", "--no-cache", "--output-format=json",
             "--exit-zero", str(td)],
            timeout=15.0,
        )
        if rc < 0 or not out.strip():
            return None
        try:
            diagnostics = json.loads(out)
        except json.JSONDecodeError:
            return None
        if not isinstance(diagnostics, list):
            return None
        n_violations = len(diagnostics)
        density = n_violations / max(total_lines, 1)
        return float(math.exp(-density * 20.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
