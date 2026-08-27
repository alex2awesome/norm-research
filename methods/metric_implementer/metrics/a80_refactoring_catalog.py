"""a80: Refactoring techniques and application — PARTIALLY_THIN catalog detector.

The norm (Fowler-style): apply *named* refactorings — Extract Method, Inline
Function, Move Method, Rename, Replace Conditional with Polymorphism, Decompose
Conditional, Introduce Parameter Object, etc. — to improve clarity and design
without changing behavior. This is ADJACENT to a37 ("Refactoring quality and
practice").

Is this subsumed by a37? a37 explicitly writes:

    "What would unblock a THIN measurement: A multi-language refactoring
     detector that classifies hunks as one of the named Fowler refactorings
     (Extract Method, Rename, Inline, Move Method, etc.)."

That is exactly what a80 names. So a80 is the SUB-NORM that a37 abstains
from — a37 measures the *shape* of the diff (line balance, edits-existing
fraction, anti-cosmetic churn), while a80 should detect *which* refactoring
techniques are being applied. The two norms answer different questions:

  a37 asks: "Does this diff LOOK like a refactor at all?" (shape)
  a80 asks: "Which named refactorings does this diff perform?" (catalog)

A diff can score high on a37 (balanced replace, existing files, no cosmetic
churn) without applying ANY recognizable Fowler refactoring — e.g., a
behaviour-changing rewrite of an algorithm. Conversely a diff can apply a
clean Extract Method while scoring poorly on a37's balance metric (e.g.,
short extracted helper added to a mostly-new file). So a80 is distinct.

How we detect the catalog (Python only, abstain elsewhere):

We use Python's `ast` module on the PRE-state (removed lines reconstructed)
and POST-state (added lines reconstructed) of each Python source file. This
gives us the diff at function-definition resolution.

  R1. EXTRACT METHOD. A function `def F(...)` newly defined in the post-state
      (not present in pre-state), AND a call `F(...)` newly appearing in the
      post-state. Inverse Fowler signature: code is *moved into* a named
      function and the caller invokes it.

  R2. INLINE FUNCTION. A function `def F(...)` removed (present in pre,
      absent in post), AND call sites of `F` removed. Inverse of R1.

  R3. RENAME (function). A function `def OLD(...)` removed and a function
      `def NEW(...)` added with substantially similar body size (within
      0.7-1.4x by line count) — heuristic match.

  R4. DECOMPOSE CONDITIONAL / GUARD CLAUSE. A removed function body
      containing top-level `if`/`else` is replaced by a post-state function
      with early-return guards (i.e., `if cond: return` patterns) and a
      reduced max nesting depth. Captures Fowler's "Decompose Conditional"
      and "Replace Nested Conditional with Guard Clauses".

  R5. INTRODUCE PARAMETER OBJECT (very weak). Function arg-list shrinks
      from N≥4 args to a single non-self argument in the renamed/kept
      function. Pure heuristic.

Score = sigmoid-like saturating function of (# distinct refactoring patterns
detected) / (# changed Python functions). Bounded [0,1].

Why PARTIALLY_THIN (not THIN):
  - Detection is structural and one-shot: a function added + called could
    be a new feature, not Extract Method. We have no way to confirm the
    extracted body came from the removed lines without semantic alignment.
  - Cross-file refactorings (Move Method/Move Class) require resolving
    references across files, which our per-file diff parsing does not.
  - Renames must be matched by body similarity; we use a coarse length
    proxy. True body matching needs AST diff.
  - We abstain on non-Python languages — coverage is partial by design.

Why not THICK:
  - Several named refactorings ARE detectable at AST-diff resolution for
    Python: Extract Method (R1) and Inline Function (R2) have crisp
    structural signatures. The detector is genuinely measuring the norm,
    not text length.

applies(): True iff the diff edits at least one Python source file with
both an added and a removed line (so pre/post AST comparison is possible).
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

import whatthepatch

ASPECT_ID = "a80"
ASPECT_NAME = "Refactoring techniques and application"
TIER = 2
TOOLS = []  # Python stdlib `ast` + whatthepatch only
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = (".py", ".pyi")


def _parse_diff_pre_post(diff_text: str) -> Dict[str, Tuple[str, str]]:
    """For each Python file in the diff, return (pre_text, post_text).

    pre_text  = concatenation of context lines + removed lines (recovers the
                file as it existed before the change, modulo unseen regions
                of the file).
    post_text = concatenation of context lines + added lines (post-state).

    Note: this is the diff's *visible* slice, not the full file. That's fine
    because we only check function defs and calls that appear in the change.
    """
    idx = diff_text.find("diff --git")
    if idx == -1:
        return {}
    try:
        diffs = list(whatthepatch.parse_patch(diff_text[idx:]))
    except Exception:
        return {}

    out: Dict[str, Tuple[List[str], List[str]]] = {}
    for d in diffs:
        if d is None:
            continue
        new_path = d.header.new_path or ""
        old_path = d.header.old_path or ""
        if new_path.startswith("b/"):
            new_path = new_path[2:]
        if old_path.startswith("a/"):
            old_path = old_path[2:]
        path = new_path or old_path
        if not path or path == "/dev/null":
            continue
        if not any(path.lower().endswith(e) for e in PY_EXTS):
            continue
        if new_path == "/dev/null" or old_path == "/dev/null":
            continue  # pure create / delete: no refactor signal

        pre_lines: List[str] = []
        post_lines: List[str] = []
        for ch in (d.changes or []):
            if ch.line is None:
                continue
            if ch.old is not None and ch.new is not None:
                pre_lines.append(ch.line)
                post_lines.append(ch.line)
            elif ch.new is None and ch.old is not None:
                pre_lines.append(ch.line)
            elif ch.old is None and ch.new is not None:
                post_lines.append(ch.line)
        out.setdefault(path, ([], []))
        out[path][0].extend(pre_lines)
        out[path][1].extend(post_lines)

    return {p: ("\n".join(pre), "\n".join(post))
            for p, (pre, post) in out.items() if pre or post}


def _safe_parse(src: str) -> Optional[ast.AST]:
    """Try to parse a diff slice as Python. The slice is usually incomplete
    (mid-function context), so we try a couple of cheap repairs."""
    try:
        return ast.parse(src)
    except SyntaxError:
        # Try wrapping in a dummy try/except to mop up dangling code.
        pass
    # Heuristic: strip lines until we find a `def`/`class` and parse from
    # there. Many diff hunks start mid-function with leading whitespace.
    lines = src.split("\n")
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "async def ")):
            sub = "\n".join(l[len(line) - len(stripped):] if len(l) >= len(line) - len(stripped) else l
                            for l in lines[i:])
            try:
                return ast.parse(sub)
            except SyntaxError:
                continue
    return None


def _collect_defs_and_calls(tree: ast.AST) -> Tuple[Dict[str, ast.AST],
                                                     set,
                                                     Dict[str, int]]:
    """Return (defs_by_name, called_names, def_body_linecounts).

    defs_by_name: top-level + nested function/method definitions, keyed by
                  bare name (last component for methods).
    called_names: names appearing as the function in a Call expression.
    def_body_linecounts: estimated body line count per def (end_lineno
                         - lineno) — used as a coarse body-size proxy.
    """
    defs: Dict[str, ast.AST] = {}
    body_lines: Dict[str, int] = {}
    calls: set = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = node
            try:
                body_lines[node.name] = max(1,
                    (getattr(node, "end_lineno", node.lineno) or node.lineno)
                    - node.lineno + 1)
            except Exception:
                body_lines[node.name] = 1
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                calls.add(f.id)
            elif isinstance(f, ast.Attribute):
                calls.add(f.attr)
    return defs, calls, body_lines


def _max_nesting(tree: ast.AST) -> int:
    """Maximum nesting depth of If/For/While/Try inside any function."""
    best = 0

    def walk(node, depth):
        nonlocal best
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try,
                              ast.AsyncFor, ast.With, ast.AsyncWith)):
            depth += 1
            best = max(best, depth)
        for c in ast.iter_child_nodes(node):
            walk(c, depth)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            walk(node, 0)
    return best


def _has_guard_clauses(tree: ast.AST) -> int:
    """Count `if cond: return ...` patterns at top of a function body
    (guard clauses — Fowler's 'Replace Nested Conditional with Guards')."""
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in node.body[:4]:
                if isinstance(stmt, ast.If):
                    if any(isinstance(b, (ast.Return, ast.Raise))
                           for b in stmt.body):
                        n += 1
    return n


def _detect_refactorings(pre_src: str, post_src: str) -> Tuple[int, int]:
    """Return (n_refactorings_detected, n_changed_funcs).

    n_refactorings_detected: count of distinct refactoring signals fired
                             on this file (capped at # changed funcs).
    n_changed_funcs: number of function defs that differ between pre/post
                     (denominator for the per-file ratio).
    """
    pre_tree = _safe_parse(pre_src) if pre_src.strip() else None
    post_tree = _safe_parse(post_src) if post_src.strip() else None
    if pre_tree is None and post_tree is None:
        return 0, 0

    pre_defs, pre_calls, pre_body = (
        _collect_defs_and_calls(pre_tree) if pre_tree else ({}, set(), {})
    )
    post_defs, post_calls, post_body = (
        _collect_defs_and_calls(post_tree) if post_tree else ({}, set(), {})
    )

    added_defs = set(post_defs) - set(pre_defs)
    removed_defs = set(pre_defs) - set(post_defs)
    kept_defs = set(post_defs) & set(pre_defs)

    detected = 0

    # R1. Extract Method: new def + new call to that def.
    for name in added_defs:
        if name in post_calls and name not in pre_calls:
            detected += 1
            break  # at most one Extract-Method credit per file

    # R2. Inline Function: removed def + removed calls.
    for name in removed_defs:
        if name in pre_calls and name not in post_calls:
            detected += 1
            break

    # R3. Rename: removed def + added def with similar body size (no
    # overlapping name pair — i.e., the names are different).
    if removed_defs and added_defs:
        for old in removed_defs:
            old_size = pre_body.get(old, 1)
            for new in added_defs:
                new_size = post_body.get(new, 1)
                if old_size == 0:
                    continue
                ratio = new_size / old_size
                if 0.7 <= ratio <= 1.4 and old != new:
                    detected += 1
                    break
            else:
                continue
            break

    # R4. Decompose Conditional / Guard Clauses: nesting reduced AND guard
    # clauses introduced (or substantially more of them) in kept funcs.
    if kept_defs and pre_tree and post_tree:
        pre_depth = _max_nesting(pre_tree)
        post_depth = _max_nesting(post_tree)
        pre_guards = _has_guard_clauses(pre_tree)
        post_guards = _has_guard_clauses(post_tree)
        if post_depth < pre_depth and post_guards >= pre_guards + 1:
            detected += 1

    # R5. Introduce Parameter Object (very weak): a kept def whose arg
    # count drops from >=4 to <=2.
    for name in kept_defs:
        pre_node = pre_defs[name]
        post_node = post_defs[name]
        if not isinstance(pre_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not isinstance(post_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        pre_args = len(pre_node.args.args)
        post_args = len(post_node.args.args)
        if pre_args >= 4 and post_args <= 2 and post_args < pre_args:
            detected += 1
            break

    # changed_funcs denominator: added + removed + kept-with-different-body.
    changed_kept = sum(
        1 for n in kept_defs
        if pre_body.get(n, 0) != post_body.get(n, 0)
    )
    n_changed = len(added_defs) + len(removed_defs) + changed_kept

    return detected, n_changed


def applies(diff_text: str) -> bool:
    """True iff the diff edits at least one Python source file with both
    added and removed lines (pre/post comparison is possible)."""
    files = _parse_diff_pre_post(diff_text)
    if not files:
        return False
    for path, (pre, post) in files.items():
        if pre.strip() and post.strip():
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    files = _parse_diff_pre_post(diff_text)
    if not files:
        return None

    total_detected = 0
    total_changed = 0
    for path, (pre, post) in files.items():
        if not pre.strip() or not post.strip():
            continue
        d, c = _detect_refactorings(pre, post)
        total_detected += d
        total_changed += c

    if total_changed == 0:
        # No changed functions to compare — abstain.
        return None

    # Saturating mapping. 0 refactorings → 0; ratio 0.5 → ~0.71; ratio >=1 → 1.
    # Bounded so a small number of detections doesn't peg the score.
    ratio = total_detected / total_changed
    if ratio <= 0:
        return 0.0
    if ratio >= 1.0:
        return 1.0
    # Mildly concave map so partial detections show up.
    import math
    return float(1.0 - math.exp(-2.5 * ratio))
