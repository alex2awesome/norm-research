"""a165: Operator spacing and parentheses / delimiter whitespace.

Norm: "Use consistent spacing around operators and after separators; avoid
spaces inside matched delimiters; use parentheses only for precedence or
clarity."

Why this is NARROWER than a72 / a195 / a369 (and not subsumed by a72):
- a72 runs the whole formatter (ruff format / prettier / gofmt) and grades
  whether each file is fully canonical. That mixes operator spacing with
  quote style, trailing commas, line wrap, indent, blank lines, import
  ordering, etc. — it cannot isolate the spacing-around-operators norm.
- a195 covers ONLY line-wrap / continuation rules (E501, E502, E131).
- a369 covers ONLY indentation rules (W191, E111, E114, E117).
- This metric (a165) covers ONLY whitespace around operators and inside /
  next to delimiters — the exact PEP 8 sub-rules. These ruff codes don't
  overlap with the rule sets used by a72's `ruff format --check` either:
  `ruff format` would reformat the file (and rolls all PEP 8 whitespace
  rules together into a binary "is the file canonical?" check), while
  `ruff check --select <whitespace codes>` reports each individual
  whitespace diagnostic on the file as-is. The signals are different:
  a72 is "would the formatter touch this file at all?"; a165 is "how many
  operator/delimiter whitespace violations per added line?". An added
  file can fail a72 because of e.g. trailing-comma style while having
  ZERO a165 violations, or vice versa.

The exact PEP 8 / ruff `pycodestyle` codes selected here:
    E201  whitespace after '(' '[' '{'
    E202  whitespace before ')' ']' '}'
    E203  whitespace before ',', ';', ':'
    E211  whitespace before '(' or '['
    E221  multiple spaces before operator
    E222  multiple spaces after operator
    E225  missing whitespace around operator
    E226  missing whitespace around arithmetic operator
    E227  missing whitespace around bitwise or shift operator
    E228  missing whitespace around modulo operator
    E231  missing whitespace after ','/';'/':'
    E251  unexpected spaces around keyword / parameter equals
    E252  missing whitespace around parameter equals (annotated default)
    E261  at least two spaces before inline comment
    E262  inline comment should start with '# '
    E271  multiple spaces after keyword
    E272  multiple spaces before keyword
    E275  missing whitespace after keyword

NOTE on the "parentheses only for precedence or clarity" half of the norm:
PEP 8 does NOT define a deterministic check for "unnecessary parentheses";
pylint has `C0325 superfluous-parens` but it only catches a narrow set
(parens around `return`, `print` etc. — and modern Python code rarely
trips it). We do NOT bolt on a regex over code to detect "redundant
parens" — that would be exactly the anti-pattern the GUIDE warns against
(regex pretending to parse code). The whitespace half is fully measurable;
the redundant-paren half is not, so we honestly under-measure rather than
fake it. Hence CLASSIFICATION = "PARTIALLY_THIN".

Score: density of selected violations per added Python line, mapped through
the same exp(-density * 20.0) shape used by a181/a369/a195 so feature
scales align in the downstream model.

Applies only to diffs that add at least one Python (.py / .pyi) file. Other
languages have different operator-spacing conventions (Go: gofmt enforces
its own rules; JS/TS: prettier; Java: google-java-format). a72 already
covers the holistic per-language formatter check; adding language-specific
operator-spacing-only metrics would just shadow those formatters with
weaker signals. The Python-only scope here mirrors a195 and a369.
"""
from __future__ import annotations

import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a165"
ASPECT_NAME = "Operator spacing and delimiter whitespace"
TIER = 3
TOOLS = ["ruff"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"
# PARTIALLY_THIN: the whitespace-around-operators / inside-delimiters half
# of the norm is fully deterministic via ruff's pycodestyle E2xx subset.
# The "use parentheses only for precedence or clarity" half has no reliable
# deterministic checker (pylint C0325 covers only a tiny slice). We measure
# the verifiable half honestly and abstain on the rest, rather than faking
# a regex over code semantics.

PY_EXTS = [".py", ".pyi"]

# pycodestyle E2xx subset: operator spacing + delimiter whitespace + keyword
# whitespace. Excludes indentation (E1xx -> a369), line-wrap (E5xx -> a195),
# and blank-line / import rules (covered by other metrics).
SPACING_RULES = ",".join([
    "E201", "E202", "E203", "E211",
    "E221", "E222",
    "E225", "E226", "E227", "E228",
    "E231",
    "E251", "E252",
    "E261", "E262",
    "E271", "E272", "E275",
])


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
             "--select", SPACING_RULES,
             # E226/E227/E228 are off-by-default in pycodestyle and surface
             # in ruff only under --preview; enable so we get coverage.
             "--preview",
             "--output-format=concise",
             "--exit-zero", str(td)],
            timeout=15.0,
        )
        if rc < 0:
            return None
        # ruff concise output: one diagnostic per non-empty line.
        n_violations = sum(1 for ln in out.splitlines() if ln.strip())
        density = n_violations / max(total_lines, 1)
        # Same exp-decay shape as a181 / a369 / a195 for feature-scale parity.
        return float(math.exp(-density * 20.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
