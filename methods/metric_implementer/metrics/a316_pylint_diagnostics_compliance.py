"""a316: Python linter diagnostics compliance.

Aspect description (verbatim from aspects.json):
  "Code resolves reported Python linter issues (pylint/ruff/flake8 etc.),
   including argument/keyword compatibility, async/context usage, exception
   types and instantiation, import/style problems, and invalid options or
   APIs."

How this differs from a181 ("Treat warnings/lints as errors"):
  a181 runs `ruff` with its default ruleset, which is dominated by *style*
  rules (E/W from pycodestyle, F from pyflakes — mostly trivial cleanups).
  a316 targets the *bug-shaped* diagnostics the aspect actually names:
  wrong argument counts, raising non-exceptions, catching non-exceptions,
  dangerous defaults, broad excepts, lost exceptions, etc. Those are
  pylint's E-codes and a curated subset of W-codes.

Tool choice: pylint (richer semantic checks than ruff's default).

Why not just `--errors-only`?
  pylint's E-set includes `import-error` (E0401), `no-name-in-module`
  (E0611), `no-member` (E1101), and `undefined-variable` (E0602). On
  diff-derived partial files (we only see added lines, not the whole
  file or its dependencies), these codes mostly produce false positives:
  any third-party import looks "unresolvable", any symbol defined on a
  non-added line looks "undefined". Including them would tank precision
  and yield a noise-dominated score.

So we enable a curated allow-list of pylint codes whose semantics depend
only on the snippet itself (syntax, exception/raise/except shape,
arg-keyword mismatches when callable is visible, dangerous-default-value,
broad-except, lost-exception, deprecated-method, await-outside-async).
This matches the aspect's named categories without poisoning the
measurement with cross-file resolution errors.

Score = exp(-violations_per_added_line * 30). 0 violations/line → 1.0;
0.03/line → ~0.41; 0.05/line → ~0.22. Tighter decay than a181 because
each retained diagnostic is a real bug, not a style nit.

Applies: PR diff adds at least one .py / .pyi file.
"""
from __future__ import annotations

import json
import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a316"
ASPECT_NAME = "Python linter diagnostics compliance"
TIER = 3
TOOLS = ["pylint"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

# Curated allow-list of pylint codes that the aspect description names AND
# whose semantics don't require cross-file resolution. Each code's category
# tag below maps to a phrase from the aspect description.
ENABLED_CODES = [
    # syntax (catches invalid options or APIs at parse time)
    "E0001",  # syntax-error
    # exception types and instantiation
    "E0701",  # bad-except-order
    "E0702",  # raising-bad-type
    "E0703",  # raising-bad-type
    "E0704",  # misplaced-bare-raise
    "E0710",  # raising-non-exception
    "E0711",  # notimplemented-raised
    "E0712",  # catching-non-exception
    "W0702",  # bare-except
    "W0703",  # broad-except (older name) / broad-exception-caught
    "W0706",  # try-except-raise (lost-exception cousin)
    "W0150",  # lost-exception
    "W0719",  # broad-exception-raised
    # argument/keyword compatibility (fire only when callable visible)
    "E1120",  # no-value-for-parameter
    "E1121",  # too-many-function-args
    "E1123",  # unexpected-keyword-arg
    "E1124",  # redundant-keyword-arg
    "E1125",  # missing-kwoa
    "E1126",  # invalid-sequence-index
    "E1127",  # invalid-slice-index
    # async/context usage
    "E1142",  # await-outside-async
    "W1641",  # eq-without-hash
    # invalid options or APIs (where pylint knows the symbol)
    "W1505",  # deprecated-method
    "W1508",  # invalid-envvar-default
    "W1509",  # subprocess-popen-preexec-fn
    "W4901",  # deprecated-module
    "W4902",  # deprecated-method (2024-rename)
    # import/style problems that are intrinsic to the snippet
    "W0401",  # wildcard-import
    "W0404",  # reimported
    "W0410",  # misplaced-future
    # other genuine-bug warnings
    "W0102",  # dangerous-default-value
    "W0104",  # pointless-statement
    "W0106",  # expression-not-assigned
    "W0107",  # unnecessary-pass
    "W0143",  # comparison-with-callable
    "W1503",  # redundant-unittest-assert
]

# Codes we explicitly disable because they require cross-file resolution
# that a diff snippet cannot provide. Documented so reviewers see the
# precision/recall trade we made.
DISABLED_NOISE_CODES = [
    "import-error",         # E0401 — diff snippets miss most deps
    "no-name-in-module",    # E0611
    "no-member",            # E1101 — needs type info
    "undefined-variable",   # E0602 — symbol may be defined elsewhere in file
    "used-before-assignment",  # E0601 — same
]


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    if not have_tool("pylint"):
        return None
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    total_lines = sum(1 + t.count("\n") for t in by_path.values())
    if total_lines == 0:
        return None
    td = write_temp_files(by_path)
    try:
        cmd = [
            "pylint",
            "--output-format=json",
            "--score=no",
            "--persistent=no",
            "--disable=all",
            "--enable=" + ",".join(ENABLED_CODES),
            # Belt-and-braces: even though --disable=all suppresses these,
            # name them explicitly so a future maintainer sees the intent.
            "--disable=" + ",".join(DISABLED_NOISE_CODES),
            str(td),
        ]
        rc, out, _ = run(cmd, timeout=30.0)
        # pylint exits non-zero whenever it finds messages; that's fine.
        # rc == -1 (timeout) or -2 (missing tool) → abstain.
        if rc < 0:
            return None
        try:
            findings = json.loads(out) if out.strip() else []
        except json.JSONDecodeError:
            return None
        if not isinstance(findings, list):
            return None
        # Defensive: drop anything outside the enabled set (in case of
        # pylint default-enabled internal codes leaking through).
        n_violations = sum(
            1 for f in findings
            if isinstance(f, dict)
            and f.get("message-id") in ENABLED_CODES
        )
        density = n_violations / max(total_lines, 1)
        return float(math.exp(-density * 30.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
