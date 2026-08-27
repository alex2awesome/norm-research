"""a206: Dead and unreachable code elimination.

Aspect description (verbatim from aspects.json):
  "Identify and remove unreachable code paths, unused variables, and
   commented-out/dead code to keep the codebase lean and maintainable."

How we measure it on a diff
---------------------------
We score the added Python code (from the unified diff) on two complementary
deterministic signals, both producing line-level findings:

  1. **vulture** (subprocess, Tier 3). Detects unused functions, classes,
     methods, variables, attributes, imports, AND unreachable code after
     a `return`/`raise`. We run with `--min-confidence 80` to suppress
     vulture's well-documented false-positive class (entry-point names,
     decorator-registered callables) while keeping unreachable-after-return
     at 100% confidence and unused-imports at 90%.

  2. **AST pass for unreachable code** (Tier 2). Walks each function body
     and counts statements after a terminator (`return`, `raise`, `break`,
     `continue`) in the same block. This catches unreachable-after-break
     and unreachable-after-continue that vulture's pyflakes-style pass
     does not flag, and works even on snippets where vulture chokes on
     missing imports / syntax fragments.

Findings from both passes are deduplicated by (file, line), so vulture and
the AST pass agreeing on "line 47 is unreachable" counts once. The
denominator is the number of added Python lines.

Score = exp(-findings_per_added_line * 25). 0/line → 1.0, 0.04/line →
~0.37, 0.08/line → ~0.14. Chosen tighter than a181's *20* multiplier
because each finding here is a real bug-shaped removal opportunity, not a
style nit, but looser than a316's *30* because vulture has a small but
non-zero noise rate on diff snippets where global usage is invisible.

What we deliberately do NOT measure
-----------------------------------
* **Commented-out code.** The aspect description names it, but reliable
  detection is heuristic at best (is `# x = compute()` commented-out code
  or a docstring example?). A regex would collapse into "count `#`-lines",
  which is text length again. We leave this dimension to a future
  THICK-marked sibling metric if needed.
* **JavaScript/TypeScript unreachable.** ESLint's `no-unreachable` would
  work, but eslint requires a working node-modules tree and a config; on
  diff-only fragments precision is poor (TS imports unresolved → spurious
  errors). Scoping this metric to Python keeps the signal honest. A
  separate `a206_js_*` metric can be added once eslint sandbox plumbing
  is in place.

Applies: PR diff adds at least one .py / .pyi file.
"""
from __future__ import annotations

import ast
import math
import shutil
from typing import Optional, Set, Tuple

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a206"
ASPECT_NAME = "Dead and unreachable code elimination"
TIER = 3
TOOLS = ["vulture"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = [".py", ".pyi"]

# vulture confidence floor. 80 keeps:
#   - unreachable code after return/raise (100)
#   - unused imports (90)
#   - unused functions/methods/classes (60-90 depending on heuristics)
# and drops:
#   - unused variables (60) — high false-positive rate on diff snippets
# We re-introduce a stricter unused-variable check via ruff/F841 would be
# duplicative with a181; the AST pass below picks up dead branches instead.
VULTURE_MIN_CONFIDENCE = 80

# Statements that make subsequent statements in the same block unreachable.
_TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def _vulture_findings(by_path) -> Tuple[Optional[Set[Tuple[str, int]]], int]:
    """Run vulture per-file; return (findings_set, n_parsed_files).

    Per-file invocation isolates parse failures: one snippet that won't
    compile shouldn't suppress findings from siblings that do. We classify
    "no parse error" as "vulture produced output OR exited 0 with no
    stderr error line", and count those files in n_parsed_files. Findings
    are accumulated only from successfully-parsed files.

    Returns (None, 0) only when vulture is missing or timed out on every
    file. Returns (set(), 0) when every file failed to parse — caller
    treats that as "no signal".
    """
    if not have_tool("vulture"):
        return None, 0
    findings: Set[Tuple[str, int]] = set()
    n_parsed = 0
    any_executed = False
    td = write_temp_files(by_path)
    try:
        for child in sorted(td.iterdir()):
            if not child.is_file():
                continue
            rc, out, err = run(
                ["vulture", f"--min-confidence={VULTURE_MIN_CONFIDENCE}", str(child)],
                timeout=10.0,
            )
            if rc == -2:  # tool missing
                return None, 0
            if rc == -1:  # timeout on this file; skip but keep going
                continue
            any_executed = True
            # rc 0 = no findings, 3 = findings, 1 = parse/usage error.
            if rc == 1:
                # parse failure — don't count this file as parsed
                continue
            n_parsed += 1
            for line in out.splitlines():
                if not line.strip():
                    continue
                parts = line.split(":", 2)  # REGEX_OK: tool_output (vulture fixed format)
                if len(parts) < 3:
                    continue
                try:
                    lineno = int(parts[1])
                except ValueError:
                    continue
                findings.add((parts[0].rsplit("/", 1)[-1], lineno))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    if not any_executed:
        return None, 0
    return findings, n_parsed


def _ast_unreachable_findings(by_path) -> Set[Tuple[str, int]]:
    """AST pass: count statements after a terminator in the same block.

    Returns set of (basename, line) tuples for stable dedup with vulture's
    own unreachable-after-return findings.
    """
    findings: Set[Tuple[str, int]] = set()
    for path, src in by_path.items():
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            # Partial diff snippets often won't parse cleanly. Skip silently;
            # vulture's own AST pass is more tolerant via its own front-end.
            continue
        basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0] + ".py"
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            _scan_block(body, basename, findings)
            # also scan else/finalbody/handlers bodies
            for attr in ("orelse", "finalbody"):
                blk = getattr(node, attr, None)
                if isinstance(blk, list):
                    _scan_block(blk, basename, findings)
            handlers = getattr(node, "handlers", None)
            if isinstance(handlers, list):
                for h in handlers:
                    hb = getattr(h, "body", None)
                    if isinstance(hb, list):
                        _scan_block(hb, basename, findings)
    return findings


def _scan_block(block, basename: str, findings: Set[Tuple[str, int]]) -> None:
    """If block contains a terminator, mark all subsequent statements as findings."""
    seen_terminator = False
    for stmt in block:
        if seen_terminator:
            lineno = getattr(stmt, "lineno", None)
            if lineno is not None:
                findings.add((basename, lineno))
            continue
        if isinstance(stmt, _TERMINATORS):
            seen_terminator = True


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    total_lines = sum(1 + t.count("\n") for t in by_path.values())
    if total_lines == 0:
        return None

    vf, n_vulture_parsed = _vulture_findings(by_path)
    af = _ast_unreachable_findings(by_path)
    # n_ast_parsed: count files whose source parsed cleanly via ast.parse.
    n_ast_parsed = 0
    for src in by_path.values():
        try:
            ast.parse(src)
            n_ast_parsed += 1
        except SyntaxError:
            pass

    # Abstain when no parser succeeded on ANY file: we cannot distinguish
    # "clean" from "unparseable diff fragment". This matches a316's
    # conservative posture for partial snippets.
    if n_vulture_parsed == 0 and n_ast_parsed == 0:
        return None

    findings = (vf or set()) | af
    n = len(findings)
    density = n / max(total_lines, 1)
    return float(math.exp(-density * 25.0))
