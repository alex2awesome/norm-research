"""a267: Injection prevention and input handling.

The norm: "Prevent injection by strictly validating inputs with allow-lists,
parameterizing commands/queries, and applying context-appropriate output
encoding/sanitization for HTML/JS/CSS/XML; avoid reliance on denylists."

We measure injection-vulnerability density of code ADDED in the diff using
`bandit` (Python static security analyzer) filtered to a curated set of
injection-relevant test IDs. Other languages are not covered by this metric
(semgrep p/security-audit would extend to JS/Java/Go but is much heavier and
multi-language fixtures here are dominated by Python anyway — see runner
report). Following the per-language pattern established by a181
(ruff/Python), other-language injection metrics belong in sibling files.

Norm conformance: 1.0 = zero bandit injection findings per added Python line,
decaying with density.

Score = exp(-findings_per_line * 50). 0/line = 1.0,
0.02/line = 0.37, 0.05/line = 0.08. Findings are rarer than lint diagnostics
so the decay constant is steeper than a181's (20).

Restricted bandit tests (CWE-77/78/79/89/90/94/95/611/943):
  - B102 exec_used
  - B301 pickle (unsafe deserialization)
  - B307 eval
  - B321 ftplib
  - B411 xmlrpc
  - B602 subprocess_popen_with_shell_equals_true
  - B603 subprocess_without_shell_equals_true (low-priority; kept for context)
  - B604 any_other_function_with_shell_equals_true
  - B605 start_process_with_a_shell (os.system / os.popen)
  - B606 start_process_with_no_shell (low; kept for completeness)
  - B607 start_process_with_partial_path
  - B608 hardcoded_sql_expressions (SQL injection)
  - B609 linux_commands_wildcard_injection
  - B610/B611 Django raw SQL / extra() with user input
  - B703 django_mark_safe
  - B704 markupsafe.Markup with non-literal (XSS via templates)

Bandit is invoked with --skip none and JSON output for stable parsing.

Note: Most code_review diffs won't contain injection patterns at all — a
score of 1.0 for clean diffs is the correct, expected behavior. The metric
is informative on the long tail where reviewers actually push back on
injection risks.
"""
from __future__ import annotations

import json
import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a267"
ASPECT_NAME = "Injection prevention and input handling"
TIER = 3
TOOLS = ["bandit"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

# CWE-77/78/79/89/90/94/95/611/943 — injection/eval/shell/SQL/XSS family
INJECTION_TESTS = ",".join([
    "B102",  # exec_used
    "B301",  # pickle
    "B307",  # eval
    "B321",  # ftplib (credential injection)
    "B411",  # xmlrpc
    "B602",  # subprocess shell=True
    "B604",  # any other function w/ shell=True
    "B605",  # os.system / os.popen (shell)
    "B607",  # partial-path process start
    "B608",  # hardcoded SQL expressions
    "B609",  # wildcard injection in shell commands
    "B610",  # django extra() raw SQL
    "B611",  # django RawSQL with user input
    "B703",  # django mark_safe
    "B704",  # markupsafe.Markup non-literal (XSS)
])


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    if not have_tool("bandit"):
        return None
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    total_lines = sum(1 + t.count("\n") for t in by_path.values())
    if total_lines == 0:
        return None
    td = write_temp_files(by_path)
    try:
        # -q: quiet, --tests: restrict to injection rules,
        # -f json: stable parse, -r: recurse temp dir.
        rc, out, err = run(
            ["bandit", "-q", "-r", "-f", "json",
             "--tests", INJECTION_TESTS, str(td)],
            timeout=30.0,
        )
        # bandit returns non-zero when findings exist; that's expected.
        if rc < 0 or not out.strip():
            return None
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            return None
        results = payload.get("results", [])
        # Weight HIGH severity findings 2x, MEDIUM 1x, LOW 0.5x.
        weight = {"HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.5}
        weighted = 0.0
        for r in results:
            sev = r.get("issue_severity", "MEDIUM").upper()
            weighted += weight.get(sev, 1.0)
        density = weighted / max(total_lines, 1)
        return float(math.exp(-density * 50.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
