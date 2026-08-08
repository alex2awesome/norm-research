"""a413: clang-tidy diagnostics compliance on added C++ code.

We run clang-tidy with a broad but stable checks set on the added .cpp/.hpp
content and count warnings per added line. Score = exp(-density * 5).

  - 0 warnings/line       → 1.0
  - 0.1 warnings/line     → exp(-0.5) ≈ 0.607
  - 0.5 warnings/line     → exp(-2.5) ≈ 0.082
  - 1 warning/line        → exp(-5) ≈ 0.007

We pass `-- -std=c++17` so clang-tidy parses the snippet without a
compile_commands.json. The diff snippets are not real translation units and
will produce many "header not found" diagnostics — we explicitly disable
those via `--checks` selection and filter out diagnostic lines that aren't
warnings.

Tools: clang-tidy. Tier 3. CLASSIFICATION THIN — clang-tidy is the canonical
C++ linter (analogous to ruff for Python).

If clang-tidy is missing from PATH (sk3 may need a `brew install llvm` or
`apt install clang-tidy` step), we abstain by returning None on tool error.
"""
from __future__ import annotations

import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a413"
ASPECT_NAME = "clang-tidy diagnostics compliance"
TIER = 3
TOOLS = ["clang-tidy"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh"]

# Stable set of practical checks. We deliberately skip whole categories
# (-fuchsia-*, -google-readability-todo, -llvmlibc-*) that don't generalize.
CHECKS = ",".join([
    "-*",
    "bugprone-*",
    "modernize-*",
    "performance-*",
    "readability-*",
    "cppcoreguidelines-*",
    "clang-analyzer-*",
    # exclude noisy or pedantic checks
    "-modernize-use-trailing-return-type",
    "-readability-magic-numbers",
    "-cppcoreguidelines-avoid-magic-numbers",
    "-readability-identifier-length",
])


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    if not have_tool("clang-tidy"):
        return None
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    total_lines = sum(1 + t.count("\n") for t in by_path.values())
    if total_lines == 0:
        return None
    td = write_temp_files(by_path)
    try:
        # Run clang-tidy file by file so a missing-include in one file
        # doesn't abort the rest.
        n_warnings = 0
        any_ok = False
        for p in sorted(td.iterdir()):
            if p.suffix.lower() not in CPP_EXTS:
                continue
            rc, out, err = run(
                ["clang-tidy", f"--checks={CHECKS}", "--quiet",
                 str(p), "--", "-std=c++17",
                 # silence missing-system-headers
                 "-Wno-everything"],
                timeout=30.0,
            )
            if rc < 0:
                # timeout or missing tool: abstain only if we got nothing
                continue
            any_ok = True
            text = (out or "") + "\n" + (err or "")
            for line in text.splitlines():
                # clang-tidy warnings look like:
                #   path:line:col: warning: msg [check-name]
                # REGEX_OK: tool_output — parsing clang-tidy stable output.
                if ": warning:" in line and "[" in line and line.endswith("]"):
                    n_warnings += 1
        if not any_ok:
            return None
        density = n_warnings / max(total_lines, 1)
        return float(math.exp(-density * 5.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
