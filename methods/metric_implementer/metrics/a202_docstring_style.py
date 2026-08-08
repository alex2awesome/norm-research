"""a202: Documentation style, placement, and docstring formatting.

What we measure
---------------
For Python files added in the diff, we measure DOCSTRING STYLE CONFORMANCE
(not coverage) using ruff's pydocstyle rules. The aspect description targets
formatting/placement issues like "leading/trailing whitespace", "consistent
indentation", "ordered, non-empty Param/Returns/Throws tags", and capitalization
— all of which map directly to specific pydocstyle (D-prefix) rules.

Coverage vs. style trade-off
----------------------------
We chose STYLE because the aspect description emphasizes formatting (whitespace,
capitalization, ordering of sections, placement relative to decorators), not
presence/absence. A separate metric could measure coverage with `interrogate`
if needed.

Language scope
--------------
Python only. The aspect description also mentions JSDoc and ScalaDoc, but:
  - JSDoc placement/format checking would require eslint with jsdoc plugin,
    which is not in the sandbox tool registry.
  - ScalaDoc has no widely-installed CLI checker.
  - extending later is straightforward: dispatch by language and run the
    appropriate checker (eslint --select=jsdoc/* for JS/TS, etc.).
We therefore restrict to Python and rely on the docstring-style ruff rule set.

Tooling
-------
We invoke `ruff check --select=D --exit-zero` over the added Python files.
Ruff's D-rules implement pydocstyle (which the upstream project has
deprecated in favor of ruff). Selected rules cover:
  - D200-D215: docstring whitespace, quotes, single-line shape, indentation
  - D400-D419: capitalization, ending period, summary-line shape, section
               formatting (Google/Numpy style), section ordering
  - D201-D202: blank lines around docstrings (placement)
  - D210: surrounding whitespace
  - D205: blank line between summary and body
  - D212/D213: multi-line docstring start-line placement

Scoring
-------
Let `n_defs` = number of def/class nodes in the added code that HAVE a
docstring (we check style only on existing docstrings — missing docstrings
are a coverage concern, not a style concern, and would unfairly dominate the
score for sparsely-documented diffs).

Let `n_viol` = number of ruff D-rule findings, EXCLUDING D100-D107 (which
flag MISSING docstrings, a coverage concern).

  score = exp( - violations_per_documented_def * 1.5 )

So 0 violations -> 1.0, 1 violation per documented def -> ~0.22.

Three return states:
  - None when applies()=False (no Python in diff)
  - None when ruff missing OR no documented defs in added code (cannot measure
    style — there is no docstring to grade)
  - float in [0,1] otherwise
"""
from __future__ import annotations

import ast
import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a202"
ASPECT_NAME = "Documentation style, placement, and docstring formatting"
TIER = 3
TOOLS = ["ruff"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

# Coverage-style D rules we EXCLUDE (these flag missing docstrings, not bad
# formatting of existing ones). D100..D107.
COVERAGE_CODES = {f"D10{i}" for i in range(8)}


def _count_documented_defs(source: str) -> int:
    """Count function/class definitions in `source` that have a docstring.

    Uses the stdlib ast (no regex on code). Returns 0 on parse error — the
    score() caller will abstain in that case via the (n_defs == 0) guard.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            if ast.get_docstring(node) is not None:
                n += 1
    return n


def applies(diff_text: str) -> bool:
    """True iff the diff adds any Python code (we don't probe PATH here)."""
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    """See module docstring. Returns None when un-measurable."""
    if not have_tool("ruff"):
        return None
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None

    n_documented_defs = sum(_count_documented_defs(s) for s in by_path.values())
    if n_documented_defs == 0:
        # No docstrings to grade -> style is undefined, abstain.
        return None

    td = write_temp_files(by_path)
    try:
        # Run ruff D-rule subset. --exit-zero so non-empty findings don't
        # cause rc != 0. --output-format=concise emits one finding per line.
        # We use --no-cache for hermeticity. --preview enables some newer
        # D-rules but we leave it off to stay deterministic across ruff
        # versions in the registry.
        rc, out, _err = run(
            ["ruff", "check", "--no-cache", "--exit-zero",
             "--select", "D",
             # Explicitly ignore coverage rules (D100..D107). Some are off by
             # default but pydocstyle/pydoclint groupings vary by ruff
             # version; we list them to be safe.
             "--ignore", ",".join(sorted(COVERAGE_CODES)),
             "--output-format=concise",
             str(td)],
            timeout=20.0,
        )
        if rc < 0:
            return None
        n_viol = 0
        for ln in out.splitlines():
            if not ln.strip():
                continue
            # Defensive belt-and-suspenders: if a D10x slipped past --ignore
            # (e.g., older ruff), drop it. The concise line looks like
            # "<path>:<line>:<col>: D200 ..." — we just substring-check.
            if any(c in ln for c in COVERAGE_CODES):
                continue
            n_viol += 1

        density = n_viol / n_documented_defs
        return float(math.exp(-density * 1.5))
    finally:
        shutil.rmtree(td, ignore_errors=True)
