"""a226: SQL formatting conventions (SELECT, JOIN, DDL).

The norm: "Apply consistent, readable SQL formatting: aligned/indented JOINs
and filters, standardized SELECT lists, trailing commas in multi-line DDL,
and consistent clause alignment for repeated attributes."

Measurement strategy: use `sqlfluff` — a real SQL parser/linter that
understands SQL structure (CTEs, JOINs, SELECT targets, DDL columns). We
restrict to LAYOUT rules ("LT..") because the norm is explicitly about
formatting/alignment, not semantic correctness, capitalization choice, or
identifier policy.

LT-family rules sqlfluff enforces (4.x):
    LT01 whitespace/spacing
    LT02 indentation
    LT03 binary operators on new line
    LT04 comma placement
    LT05 line length
    LT06 keyword/identifier spacing
    LT07 with-CTE bracket placement
    LT08 blank line after CTE
    LT09 SELECT targets on new line
    LT10 SELECT modifier placement
    LT11 SET operators on new line
    LT12 final newline
    LT13 leading whitespace
    LT14 keywords (WHERE/etc.) start new line
    LT15 reserved keywords as identifiers

Score = exp(-layout_violations_per_added_line * 8). Tuned so 0 viol/line=1.0,
0.05 -> 0.67, 0.1 -> 0.45, 0.5 -> 0.018.

Scope decision: we ONLY apply this metric to *.sql files in the diff. SQL
embedded in Python/JS string literals would need AST walking + heuristics to
identify which strings are SQL — possible (tree-sitter + DB-call detection)
but research-grade, easy to over- or under-apply, and out of scope for an
honest single-norm metric. If a PR has no .sql files we abstain.

Dialect handling: PR diffs don't tell us the SQL dialect, and sqlfluff
defaults to "ansi" which is strict. We try a small panel of common dialects
(ansi, postgres, mysql, sqlite) and take the MINIMUM violation count across
them. This is the lenient/charitable interpretation: if a file parses
cleanly under ANY common dialect, we believe the author and use that.
"""
from __future__ import annotations

import json
import math
from typing import List, Optional

from ..sandbox import added_files_by_ext, have_tool, run

ASPECT_ID = "a226"
ASPECT_NAME = "SQL formatting conventions (SELECT, JOIN, DDL)"
TIER = 3
TOOLS = ["sqlfluff"]
APPLIES_TO_LANGS = ["SQL"]
CLASSIFICATION = "THIN"

SQL_EXTS = [".sql", ".ddl", ".dml"]

# Layout rules — what the norm is actually about. We deliberately ignore
# non-layout families (CP=capitalization, AL=aliasing, RF=references,
# CV=conventions, ST=structure, AM=ambiguity, JJ=jinja) so the score reflects
# formatting conformance and is not polluted by stylistic policies orthogonal
# to the norm.
LAYOUT_CODES = {f"LT{i:02d}" for i in range(1, 16)}

# Dialects we'll try; we take the minimum violation count across them as the
# charitable interpretation when dialect is unknown.
DIALECTS = ["ansi", "postgres", "mysql", "sqlite"]


def applies(diff_text: str) -> bool:
    """True iff the diff adds any lines to a .sql/.ddl/.dml file.

    We do NOT attempt to detect SQL inside string literals here — see module
    docstring. Cheap, conservative, and easy to extend later if we want to
    move to a research-grade applies-gate.
    """
    return bool(added_files_by_ext(diff_text, SQL_EXTS))


def _lint_one(content: str, dialect: str) -> Optional[int]:
    """Run sqlfluff lint on a single SQL snippet for one dialect.

    Returns the number of layout violations, or None if the parser failed
    (templating error, fully unparseable, etc.). We pass --nofail so a
    nonzero exit (= violations found) still gives us JSON stdout.
    """
    rc, out, _err = run(
        ["sqlfluff", "lint", "--dialect", dialect, "--format", "json",
         "--disable-progress-bar", "--nofail", "-"],
        stdin=content,
        timeout=20.0,
    )
    if rc < 0:
        return None
    try:
        data = json.loads(out) if out.strip() else []
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        # Empty list = nothing to lint (e.g. comment-only file). Treat as 0.
        return 0
    n_layout = 0
    n_fatal = 0
    for fileobj in data:
        for v in fileobj.get("violations", []) or []:
            code = v.get("code", "")
            # PRS = parse error, TMP = templater error. If the whole snippet
            # failed to parse, sqlfluff usually emits one PRS/TMP and no other
            # violations — that's the "unmeasurable" signal we abstain on.
            if code.startswith("PRS") or code.startswith("TMP"):
                n_fatal += 1
                continue
            if code in LAYOUT_CODES:
                n_layout += 1
    # If parsing fundamentally failed and we got nothing else, abstain for
    # this dialect by returning None — the caller will try other dialects.
    if n_fatal > 0 and n_layout == 0:
        return None
    return n_layout


def _best_layout_violations(content: str) -> Optional[int]:
    """Try each dialect; return the MIN layout-violation count.

    Returns None iff every dialect failed to parse the snippet — then we
    can't measure formatting conformance and we abstain.
    """
    results: List[int] = []
    for dialect in DIALECTS:
        v = _lint_one(content, dialect)
        if v is not None:
            results.append(v)
    if not results:
        return None
    return min(results)


def _line_count(s: str) -> int:
    if not s:
        return 0
    return s.count("\n") + (0 if s.endswith("\n") else 1)


def score(diff_text: str) -> Optional[float]:
    if not have_tool("sqlfluff"):
        return None
    by_path = added_files_by_ext(diff_text, SQL_EXTS)
    if not by_path:
        return None

    total_lines = 0
    total_violations = 0
    n_measurable = 0
    for _path, content in by_path.items():
        if not content.strip():
            continue
        n_v = _best_layout_violations(content)
        if n_v is None:
            # snippet unparseable under every dialect -> skip
            continue
        n_measurable += 1
        total_lines += _line_count(content)
        total_violations += n_v

    if n_measurable == 0 or total_lines == 0:
        return None

    density = total_violations / total_lines
    return float(math.exp(-density * 8.0))
