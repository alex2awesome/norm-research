"""a5: Avoid duplication (DRY) with pragmatism.

We measure whether code ADDED in the diff introduces copy-paste duplication.
Tool: `jscpd` (Copy-Paste Detector), a multi-language token-stream clone
detector that supports 100+ languages including Python/JS/TS/Java/Go.
We invoke jscpd's JSON reporter and compute the fraction of added lines that
participate in any detected clone fragment.

Pragmatism — the Rule of Three angle:
A small amount of repeated boilerplate is fine; jscpd is configured with
`--min-lines 5` and `--min-tokens 50` (its defaults), so trivially small
repeats are not counted. Tests are not specially exempted here (test files
appear in the diff like any other); the downstream model can learn from the
applied-flag if it wants to.

Norm conformance: 1.0 = no clones detected. Score decays exponentially with
the fraction of added lines participating in a clone:
    score = exp(-dup_fraction * 5)
That gives 0% dup -> 1.0, 10% -> 0.61, 20% -> 0.37, 40% -> 0.14.
We use 5 instead of 20 (cf. a181) because clone fractions are typically much
larger than lint-violation density — a single duplicated function can easily
push the fraction above 30%.

Why this lives at Tier 3 not Tier 1:
A regex or `difflib` over raw diff text would see "function header looks like
function header" and pretend that's a clone. jscpd parses with Prism-style
tokenizers per language, ignores comments/whitespace by default (`mild` mode),
and reports actual token-stream clones — the real measurement target.

Per the GUIDE: applies() is cheap (diff parsing only); score() returns None
when jscpd is absent, when no files match supported languages, or when jscpd
fails on the (often partial) reconstructed files.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a5"
ASPECT_NAME = "Avoid duplication (DRY) pragmatic"
TIER = 3
TOOLS = ["jscpd"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go",
                    "C", "C++", "C#", "Ruby", "PHP", "Kotlin", "Swift", "Rust"]
CLASSIFICATION = "THIN"

# Extensions jscpd understands well and that appear in code_review fixtures.
SUPPORTED_EXTS = [
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".java",
    ".go",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".kt", ".kts",
    ".swift",
    ".rs",
]


def applies(diff_text: str) -> bool:
    """True iff the diff adds lines in at least one file of a supported language."""
    return bool(added_files_by_ext(diff_text, SUPPORTED_EXTS))


def _total_added_lines(by_path: Dict[str, str]) -> int:
    total = 0
    for content in by_path.values():
        if not content:
            continue
        # Match jscpd line-count convention (count physical newlines + 1 if no trailing nl).
        total += content.count("\n") + (0 if content.endswith("\n") else 1)
    return total


def score(diff_text: str) -> Optional[float]:
    if not have_tool("jscpd"):
        return None
    by_path = added_files_by_ext(diff_text, SUPPORTED_EXTS)
    if not by_path:
        return None
    total_added = _total_added_lines(by_path)
    if total_added < 5:
        # jscpd's default min-lines is 5; below that no clone is possible.
        return None

    src_dir = write_temp_files(by_path)
    report_dir = Path(tempfile.mkdtemp(prefix="jscpd_rep_"))
    try:
        rc, _out, _err = run(
            [
                "jscpd",
                "--silent",
                "--reporters", "json",
                "--output", str(report_dir),
                "--min-lines", "5",
                "--min-tokens", "50",
                "--mode", "mild",
                "--no-gitignore",
                "--noTips",
                str(src_dir),
            ],
            timeout=30.0,
        )
        if rc < 0:
            return None
        report_path = report_dir / "jscpd-report.json"
        if not report_path.exists():
            # jscpd ran but produced no JSON: treat as unmeasurable, not 0.
            return None
        try:
            data = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        stats = data.get("statistics", {}).get("total", {})
        total_lines = stats.get("lines", 0)
        dup_lines = stats.get("duplicatedLines", 0)
        if not isinstance(total_lines, (int, float)) or total_lines <= 0:
            return None
        dup_fraction = max(0.0, min(1.0, float(dup_lines) / float(total_lines)))
        return float(math.exp(-dup_fraction * 5.0))
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)
        shutil.rmtree(report_dir, ignore_errors=True)
