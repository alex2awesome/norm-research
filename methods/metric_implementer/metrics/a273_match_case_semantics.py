"""a273: Python structural pattern matching semantics.

Python 3.10+ `match` / `case` has a small number of well-known semantic
footguns. PEP 634/635/636 spell out the rules; the norm being measured is
"does the added code apply those rules correctly?"

The hardest mistakes to catch by eye, and what we check via `ast`:

  R1. Sequence-pattern star rule. A `MatchSequence` may contain at most
      one `MatchStar`. More than one is a SyntaxError at parse time, but a
      `MatchStar` *inside* a nested OR (`MatchOr`) alternative is unusual
      and worth flagging — we count only the legal-but-suspicious cases
      conservatively.
  R2. Mapping-pattern double-rest. `MatchMapping` allows at most one
      `**rest` capture; multiple is a SyntaxError. We treat it as a
      structure check via the `rest` field.
  R3. Capture-vs-constant footgun. A bare lowercase `Name` in a case is a
      CAPTURE pattern (binds the subject), NOT a comparison against a
      previously-defined constant. The fix is to dotted-attribute it
      (`case Color.RED:`), or to use a guard. We flag a bare-name
      `MatchAs(pattern=None, name=x)` where `x` is also assigned as a
      module-level constant in the same file. Wildcard `_` is excluded.
  R4. Singleton vs equality comparison. `None` / `True` / `False` MUST be
      matched as `MatchSingleton` (identity), not via class- or value-
      pattern wrappers. Python's parser already enforces this for the
      bare literal, but `case bool() as b if b is True:` style anti-
      patterns are visible. We award the simple form.
  R5. Irrefutable case placement. An irrefutable pattern (bare wildcard
      or capture without guard) before the end of the match block makes
      subsequent cases unreachable. PEP 634 says this is a SyntaxError
      only for the bare wildcard; capture-without-guard before another
      case is a *semantic* mistake we DO flag.
  R6. Guard-only fallthrough. A `case _ if cond:` arm provides
      conditional fallthrough; if it's the only "default-looking" arm and
      `cond` can be False, the match is non-exhaustive. We do not flag
      this (it is legal style) — included for documentation.

Score per match-statement = fraction of (R1..R5) satisfied. Average over
all match statements in the diff.

The metric ABSTAINS when the diff adds no syntactically parseable Python
file containing a `match` statement. Most diffs.
"""
from __future__ import annotations

import ast
from typing import List, Optional, Set

from ..sandbox import added_files_by_ext

ASPECT_ID = "a273"
ASPECT_NAME = "Python match/case semantics"
TIER = 2
TOOLS = ["ast"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]


# ---------- module-level-constant collection ----------

def _module_constants(tree: ast.AST) -> Set[str]:
    """Names assigned at module top-level to a literal — i.e. what users
    most commonly want to *match against*, which is the classic capture-vs-
    constant trap.
    """
    out: Set[str] = set()
    if not isinstance(tree, ast.Module):
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Constant):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        out.add(tgt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.value, ast.Constant) and isinstance(
                    node.target, ast.Name):
                out.add(node.target.id)
    return out


# ---------- per-pattern checks ----------

def _is_wildcard(pat: ast.pattern) -> bool:
    """Bare `_` wildcard: MatchAs with no pattern and no name."""
    return isinstance(pat, ast.MatchAs) and pat.pattern is None and pat.name is None


def _is_bare_capture(pat: ast.pattern) -> Optional[str]:
    """Return capture name iff `pat` is `case x:` (bare name, no sub-pattern)."""
    if isinstance(pat, ast.MatchAs) and pat.pattern is None and pat.name:
        return pat.name
    return None


def _is_irrefutable(case: ast.match_case) -> bool:
    """A case is irrefutable iff its pattern always matches AND it has no
    guard. PEP 634: bare wildcard or bare capture are irrefutable patterns.
    """
    if case.guard is not None:
        return False
    p = case.pattern
    if _is_wildcard(p):
        return True
    if _is_bare_capture(p) is not None:
        return True
    # OR-pattern is irrefutable iff every alternative is
    if isinstance(p, ast.MatchOr):
        return all(
            _is_wildcard(alt) or _is_bare_capture(alt) is not None
            for alt in p.patterns
        )
    return False


def _walk_patterns(pat: ast.pattern):
    yield pat
    if isinstance(pat, ast.MatchSequence):
        for sub in pat.patterns:
            yield from _walk_patterns(sub)
    elif isinstance(pat, ast.MatchMapping):
        for sub in pat.patterns:
            yield from _walk_patterns(sub)
    elif isinstance(pat, ast.MatchClass):
        for sub in pat.patterns:
            yield from _walk_patterns(sub)
        for sub in pat.kwd_patterns:
            yield from _walk_patterns(sub)
    elif isinstance(pat, ast.MatchOr):
        for sub in pat.patterns:
            yield from _walk_patterns(sub)
    elif isinstance(pat, ast.MatchAs):
        if pat.pattern is not None:
            yield from _walk_patterns(pat.pattern)
    elif isinstance(pat, ast.MatchStar):
        return


def _has_nested_star_in_or(pat: ast.pattern) -> bool:
    """R1 suspicious: MatchStar inside a MatchOr alternative inside a
    sequence pattern. Rare but pathological for readability.
    """
    if isinstance(pat, ast.MatchSequence):
        for sub in pat.patterns:
            if isinstance(sub, ast.MatchOr):
                for alt in sub.patterns:
                    if isinstance(alt, ast.MatchStar):
                        return True
    for sub in _walk_patterns(pat):
        if sub is pat:
            continue
        if _has_nested_star_in_or(sub):
            return True
    return False


def _r1_sequence_star_ok(match_node: ast.Match) -> bool:
    """At most one MatchStar per MatchSequence (parser usually enforces
    this; we still scan, and additionally count nested-OR star as a fail)."""
    for case in match_node.cases:
        for p in _walk_patterns(case.pattern):
            if isinstance(p, ast.MatchSequence):
                stars = sum(1 for x in p.patterns if isinstance(x, ast.MatchStar))
                if stars > 1:
                    return False
        if _has_nested_star_in_or(case.pattern):
            return False
    return True


def _r2_mapping_rest_ok(match_node: ast.Match) -> bool:
    """At most one `**rest` per mapping pattern. The AST `rest` is a single
    optional name, so we just trust the parser's per-pattern guarantee but
    also flag if `rest` shares a name with one of the value patterns'
    keys — a different antipattern.
    """
    for case in match_node.cases:
        for p in _walk_patterns(case.pattern):
            if isinstance(p, ast.MatchMapping):
                # AST guarantees a single `rest` slot; nothing structural
                # to fail. Sanity-check key duplication.
                seen = set()
                for k in p.keys:
                    if isinstance(k, ast.Constant):
                        if k.value in seen:
                            return False
                        seen.add(k.value)
    return True


def _r3_capture_vs_constant_ok(
        match_node: ast.Match, module_constants: Set[str]) -> bool:
    """R3: a bare-name case that shadows a module-level constant is almost
    always the capture-vs-constant footgun. The user *meant* equality.
    """
    for case in match_node.cases:
        nm = _is_bare_capture(case.pattern)
        if nm is not None and nm in module_constants:
            return False
        # check inside OR patterns too
        if isinstance(case.pattern, ast.MatchOr):
            for alt in case.pattern.patterns:
                nm2 = _is_bare_capture(alt)
                if nm2 is not None and nm2 in module_constants:
                    return False
    return True


def _r4_singletons_ok(match_node: ast.Match) -> bool:
    """`None` / `True` / `False` patterns are MatchSingleton (identity).
    The AST parser enforces this — if the user wrote `case None:` we get
    MatchSingleton automatically. We instead check that no case uses a
    constant-class style like `case bool():` followed by a `is True`
    guard — a noisy way to write a singleton match.
    """
    for case in match_node.cases:
        if case.guard is None:
            continue
        # MatchClass(cls=Name(bool|None|...)) with guard "x is True"
        p = case.pattern
        if isinstance(p, ast.MatchClass) and isinstance(p.cls, ast.Name):
            if p.cls.id in {"bool", "NoneType"}:
                # Look for guard containing `is True/False/None`
                for sub in ast.walk(case.guard):
                    if (isinstance(sub, ast.Compare) and
                            any(isinstance(op, ast.Is) for op in sub.ops)):
                        return False
    return True


def _r5_irrefutable_placement_ok(match_node: ast.Match) -> bool:
    """An irrefutable case (wildcard or unguarded bare capture) before the
    last case makes following cases unreachable. The parser flags this for
    bare `_`; for bare-name captures it is silent but still a bug.
    """
    cases = match_node.cases
    for i, case in enumerate(cases[:-1]):
        if _is_irrefutable(case):
            return False
    return True


# ---------- per-file scoring ----------

def _score_file(code: bytes) -> Optional[float]:
    """Parse one Python source; return mean score over its match
    statements, or None if no parse / no match statements found."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    matches: List[ast.Match] = [
        n for n in ast.walk(tree) if isinstance(n, ast.Match)]
    if not matches:
        return None

    constants = _module_constants(tree)

    per_match = []
    for m in matches:
        checks = [
            _r1_sequence_star_ok(m),
            _r2_mapping_rest_ok(m),
            _r3_capture_vs_constant_ok(m, constants),
            _r4_singletons_ok(m),
            _r5_irrefutable_placement_ok(m),
        ]
        per_match.append(sum(1 for x in checks if x) / len(checks))
    return sum(per_match) / len(per_match)


# ---------- contract ----------

def _file_has_match_keyword(code: str) -> bool:
    """Cheap pre-filter so we don't ast.parse the world. Looks for the
    soft keyword `match` on a line, followed later by a `case` line.
    Not a parse — just a hint to skip ast.parse for files with no chance.
    """
    saw_match = False
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("match ") and stripped.rstrip().endswith(":"):
            saw_match = True
        elif saw_match and stripped.startswith("case "):
            return True
    return False


def _candidate_files(diff_text: str):
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    return {p: c for p, c in by_path.items() if _file_has_match_keyword(c)}


def applies(diff_text: str) -> bool:
    """True iff the diff adds Python source that contains at least one
    plausible `match`/`case` pair (cheap text pre-filter only)."""
    return bool(_candidate_files(diff_text))


def score(diff_text: str) -> Optional[float]:
    by_path = _candidate_files(diff_text)
    if not by_path:
        return None
    scs: List[float] = []
    for path, content in by_path.items():
        s = _score_file(content.encode("utf8", errors="replace"))
        if s is not None:
            scs.append(s)
    if not scs:
        return None
    return float(sum(scs) / len(scs))
