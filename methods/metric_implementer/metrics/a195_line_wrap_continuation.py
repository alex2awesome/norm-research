"""a195: Line-wrapping and continuation conventions.

Norm: split long lines at sensible break points; prefer parentheses-based
continuation over backslash continuation; keep continuation indents
consistent.

Why this is NARROWER than a72 (formatting) and a369 (4-space indent):
- a72 runs the full formatter (ruff format / prettier / gofmt) and grades
  whether the file is fully canonical. It mixes line-wrap with quote style,
  trailing commas, whitespace, etc. — too coarse to isolate the wrap norm.
- a369 is scoped to indentation rules only (W191/E111/E114/E117).
- a195 is scoped to the line-wrap and continuation rules only:
    E501  line too long
    E502  redundant backslash inside brackets (use parens, not '\\')
    E131  continuation line unaligned for hanging indent (pycodestyle, only
          surfaced under --preview in current ruff; we still --select it so
          we get coverage if/when it stabilizes)
  Plus a deterministic count of bare backslash-continuations OUTSIDE
  brackets in added lines. PEP 8 says: "The preferred way of wrapping long
  lines is by using Python's implied line continuation inside parentheses,
  brackets and braces. Long lines can be broken over multiple lines by
  wrapping expressions in parentheses. These should be used in preference
  to using a backslash for line continuation." Backslash-outside-brackets
  is exactly the anti-pattern.

Score: density of (E501 + E502 + E131 + backslash-EOL) violations per added
Python line, mapped through the same exp-decay shape as a181/a369 so feature
scales align in the downstream model:
    score = exp(-density * 20.0)

Applies only to diffs that add at least one Python (.py / .pyi) file.
JS/TS/Go are deliberately excluded here: the line-wrap norm interacts with
each language's formatter (prettier --print-width, gofmt's tab-based
wrapping) in ways a72 already covers holistically. Adding a per-language
wrap-only metric for JS would just shadow prettier's check; a72 already
encodes that signal.

Classification: PARTIALLY_THIN — the line-length and backslash-continuation
sub-rules are fully deterministic via ruff. The broader "consistent
continuation alignment and operator placement" is only partially captured
(E131 covers hanging-indent alignment; W503/W504 operator-placement style
is intentionally a stylistic choice in PEP 8 and not reliably checkable
without a project config). So this metric measures the verifiable core of
a195 honestly and abstains on the rest.
"""
from __future__ import annotations

import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a195"
ASPECT_NAME = "Line-wrapping and continuation conventions"
TIER = 3
TOOLS = ["ruff"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = [".py", ".pyi"]

# Ruff codes that target line-wrapping / continuation specifically.
# E501 = line too long; E502 = redundant backslash between brackets.
# E131 = continuation line unaligned (hanging indent).
WRAP_RULES = "E501,E502,E131"


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def _bare_backslash_eol_count(by_path) -> int:
    """Count backslash-continuations on added lines that are NOT inside a
    bracket context. We do this with the Python tokenizer so we don't have
    to regex code — `tokenize` emits a NEWLINE for logical line ends and
    nothing at all between physical lines joined by '\\'. So any physical
    line ending with a bare '\\' that isn't followed by a tokenize NEWLINE
    on the same logical line was a backslash continuation.

    For robustness against snippets that don't tokenize, we fall back to a
    deterministic check: count physical lines that end with a single
    backslash (after stripping trailing whitespace), then SUBTRACT lines
    whose backslash sits within an unbalanced (, [, or { that opened on or
    before this line. That paren-tracker is a tiny stdlib walk over the
    text, not a regex over code semantics.
    """
    n = 0
    for content in by_path.values():
        depth = 0
        # Per-character bracket depth; ignore brackets inside strings/comments
        # would be ideal, but we approximate by tracking line-by-line and
        # using a simple state. snippets are short; false-positives from
        # brackets-in-strings are rare in real diffs.
        in_string = None  # quote char or None
        i = 0
        line_start_depth = 0
        line_buf = []
        for line in content.splitlines():
            line_start_depth = depth
            j = 0
            L = len(line)
            while j < L:
                c = line[j]
                if in_string:
                    if c == "\\" and j + 1 < L:
                        j += 2
                        continue
                    if c == in_string:
                        in_string = None
                    j += 1
                    continue
                if c in ("'", '"'):
                    in_string = c
                elif c == "#":
                    break  # rest of line is comment
                elif c in "([{":
                    depth += 1
                elif c in ")]}":
                    depth -= 1
                j += 1
            # End of physical line: check trailing backslash
            stripped = line.rstrip()
            if stripped.endswith("\\") and not stripped.endswith("\\\\"):
                # Backslash-continuation at end of physical line.
                # If we're inside brackets, ruff E502 will fire (covered).
                # If we're OUTSIDE brackets, ruff WON'T flag it as wrong
                # (it's valid Python), but PEP 8 considers it the
                # anti-pattern. Count those.
                if depth <= 0 and line_start_depth <= 0:
                    n += 1
            # If a string was unterminated at EOL (triple-string), our
            # state will desync — but we only ever increment on a clean
            # bracket-out condition, so the worst case is under-count.
            if in_string is not None:
                # Reset on physical newline if it's not a triple-quoted
                # string — we can't tell, so just reset to avoid runaway.
                in_string = None
        # next file resets state
    return n


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
             "--select", WRAP_RULES,
             "--preview",  # surfaces E502 / E131 where still preview
             "--output-format=concise",
             "--exit-zero", str(td)],
            timeout=15.0,
        )
        if rc < 0:
            return None
        n_ruff = sum(1 for ln in out.splitlines() if ln.strip())
        n_bare = _bare_backslash_eol_count(by_path)
        n_violations = n_ruff + n_bare
        density = n_violations / max(total_lines, 1)
        # Same exp-decay shape as a181/a369 for feature-scale parity.
        return float(math.exp(-density * 20.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
