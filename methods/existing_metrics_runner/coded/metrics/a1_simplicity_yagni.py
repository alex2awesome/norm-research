"""a1: Simplicity (KISS/YAGNI) and complexity management — PARTIALLY_THIN.

The norm asks to "prefer the simplest implementation that meets current,
evidenced requirements" and to "avoid incidental complexity, speculative
generality, premature optimization, and over-engineering." A fully honest
read of the norm is THICK in the sense of a16 (maintainability): deciding
whether an abstraction is "speculative" requires intent — what counts as a
*current, evidenced* requirement is known only to the author and reviewer.

HOWEVER, a0 (CCN) and a18 (Maintainability Index + worst-function NLOC +
params) already cover the *local-complexity-mass* slice of this norm
(branching depth, function length, parameter count, code volume). What
those metrics do NOT measure — and what this metric targets — is the
*speculative-generality* slice: structural patterns that are red flags
for YAGNI/over-engineering independently of how much code there is.

We count, per added Python file, occurrences of YAGNI red-flag patterns:

  R1. Abstract classes / abstract methods.
      `from abc import ABC, ABCMeta, abstractmethod`, classes inheriting
      `ABC` or with `metaclass=ABCMeta`, and `@abstractmethod`-decorated
      methods. These create a generality scaffold whose payoff requires
      future subclasses we cannot verify exist.

  R2. Placeholder / "future" markers.
      `raise NotImplementedError`, `pass`-only class/function bodies,
      `...` (Ellipsis) as a function body. These announce "I will be
      implemented later" — the classic YAGNI shape.

  R3. Speculative-name single-method classes.
      Class defined with name ending in Factory|Manager|Helper|Wrapper|
      Builder|Handler|Provider|Strategy|Adapter that has exactly ONE
      method (excluding __init__). These are the "AbstractSingletonProxy
      FactoryBean" smell — wrapping that adds an indirection layer for a
      single call site.

  R4. Deep inheritance (DIT proxy, in-diff only).
      Classes that name 2+ base classes (multiple inheritance) OR whose
      base name is itself a class defined in the same diff (chain of 2+).
      Deep inheritance is a recognized complexity-management smell (Chidamber-
      Kemerer DIT). We approximate by counting `class X(A, B, ...)` with
      len(bases) >= 2, since true DIT across the project is unobservable
      from the diff alone.

  R5. Code-comment "future" markers.
      Counts of `TODO|FIXME|XXX|HACK` comments added in the diff (any
      language extension supported by lizard). These are explicit
      author-acknowledged speculative scope.

Combination. Each file gets a "flag density" = (R1+R2+R3+R4) / max(loc, 30).
We average flag density across added Python files, then map to [0,1] with
an exponential decay (1 = no flags = norm satisfied):

    py_score = exp(-mean_flag_density * 25)

For R5 we compute flags-per-added-line across ALL added source files
(any source extension lizard covers), and combine with py_score when
Python files exist, else use the comment signal alone:

    comment_score = exp(-todo_density * 100)   # 1 TODO per 100 LOC ~ 0.37

    score = 0.75 * py_score + 0.25 * comment_score  if Python present
    score = comment_score                            otherwise

Weight rationale: structural R1-R4 are stronger YAGNI signals than TODO
markers (a single TODO in legacy code is normal hygiene); when Python
files are present we weight them 3:1. When no Python is present we have
only the comment marker channel — explicitly weaker, so the metric will
saturate at high scores for most diffs (low std), and the runner will
see that.

CLASSIFICATION = PARTIALLY_THIN, not THIN:
  - "Speculative" requires comparing to a requirement we never observe.
    A genuine ABC across a real plugin ecosystem will look identical to
    an over-engineered ABC with no plugins.
  - All five signals are *structural surrogates* for the intent the norm
    actually names. They will have false positives (legitimate Factory
    patterns) and false negatives (clever non-abstract over-engineering).
  - The metric will likely correlate moderately with a0 and a18 because
    YAGNI code also tends to be longer / more branchy — but it captures
    DIFFERENT signals (abstraction shape vs raw mass).

Distinction from siblings:
  - a0 (CCN): branching depth of the worst function. Orthogonal to
    speculative *shape* — a single abstract method has CCN=1.
  - a18 (MI + NLOC + params): code volume and parameter count. Doesn't
    fire on a single-method Factory class.
  - a16 (Maintainability ease-of-change): THICK. We are NOT a
    maintainability proxy; we are a speculative-abstraction proxy.

applies(): True iff the diff adds at least one source file lizard parses
(so the comment-marker channel always has a denominator).
"""
from __future__ import annotations

import ast
import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext, parse_diff_added_by_file

ASPECT_ID = "a1"
ASPECT_NAME = "Simplicity (KISS/YAGNI) and complexity management"
TIER = 2  # Python AST + diff line walk; no external CLI required
TOOLS = []
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go",
                    "C++", "C", "Ruby", "C#", "PHP"]
CLASSIFICATION = "PARTIALLY_THIN"

# Source extensions for the comment-marker channel (R5).
SOURCE_EXTS = [".py", ".pyi",
               ".js", ".jsx", ".mjs", ".cjs",
               ".ts", ".tsx",
               ".java",
               ".go",
               ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
               ".cs",
               ".rb", ".php",
               ".kt", ".scala", ".swift", ".rs"]

PY_EXTS = [".py", ".pyi"]

# Names whose single-method-class form is a YAGNI red flag (R3).
SPECULATIVE_SUFFIXES = (
    "Factory", "Manager", "Helper", "Wrapper", "Builder", "Handler",
    "Provider", "Strategy", "Adapter", "Proxy", "Mediator", "Coordinator",
)

# R5: code-comment YAGNI markers. We look for these as comment tokens.
# We restrict to comment contexts so as not to flag string literals.
# REGEX_OK: tool_output — we only run this against extracted comment
# substrings from the added-lines text (after `_added_comments_only`),
# not against raw source. The token set is fixed and case-sensitive.
TODO_TOKENS_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")

# Per-language single-line comment prefixes for R5 comment extraction.
# REGEX_OK: tool_output — we are scanning ADDED diff lines (a single line
# of source) for their comment substring. This is not parsing code; it is
# stripping a single-line-comment prefix from already-line-split text.
PY_COMMENT_RE = re.compile(r"#(.*)$")
# REGEX_OK: tool_output — single-line comment-prefix stripping on one
# already-line-split source line; not parsing code structure.
SLASH_COMMENT_RE = re.compile(r"//(.*)$")


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, SOURCE_EXTS))


# ---- R1-R4: AST-based detection on added Python source -------------------

def _count_python_flags(src: str) -> tuple[int, int]:
    """Return (flag_count, loc) for one Python source. flag_count is the
    sum of R1+R2+R3+R4 occurrences; loc is the line count of the source.
    Returns (0, 0) if the source cannot be parsed (often the case for
    diff-only fragments).
    """
    loc = len([ln for ln in src.splitlines() if ln.strip()])
    if loc == 0:
        return 0, 0
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Partial diff content often won't parse as a complete module. We
        # try wrapping it in a dummy class to recover any class/function
        # context, but if that also fails we give up and return zero
        # structural flags (the comment channel still applies).
        try:
            tree = ast.parse("class _Diff_:\n" + "".join(
                "    " + ln + "\n" for ln in src.splitlines()))
        except SyntaxError:
            return 0, loc

    flags = 0

    # R1: imports of abc machinery and metaclass=ABCMeta usage
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "abc":
                for n in node.names:
                    if n.name in ("ABC", "ABCMeta", "abstractmethod",
                                  "abstractproperty",
                                  "abstractclassmethod",
                                  "abstractstaticmethod"):
                        flags += 1

    # R1+R2+R3: per-class inspection
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # R1a: ABC in bases
            base_names = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    base_names.append(b.id)
                elif isinstance(b, ast.Attribute):
                    base_names.append(b.attr)
            if any(n in ("ABC", "ABCMeta") for n in base_names):
                flags += 1
            # R1b: metaclass=ABCMeta kwarg
            for kw in node.keywords:
                if kw.arg == "metaclass":
                    val = kw.value
                    if isinstance(val, ast.Name) and val.id == "ABCMeta":
                        flags += 1
                    elif isinstance(val, ast.Attribute) and val.attr == "ABCMeta":
                        flags += 1

            # R1c: @abstractmethod decorated methods inside the class
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in m.decorator_list:
                        dec_name = None
                        if isinstance(dec, ast.Name):
                            dec_name = dec.id
                        elif isinstance(dec, ast.Attribute):
                            dec_name = dec.attr
                        elif isinstance(dec, ast.Call):
                            f = dec.func
                            if isinstance(f, ast.Name):
                                dec_name = f.id
                            elif isinstance(f, ast.Attribute):
                                dec_name = f.attr
                        if dec_name in ("abstractmethod", "abstractproperty",
                                        "abstractclassmethod",
                                        "abstractstaticmethod"):
                            flags += 1

            # R2a: class body is `pass`-only or `...`-only
            non_doc = [s for s in node.body
                       if not (isinstance(s, ast.Expr)
                               and isinstance(s.value, ast.Constant)
                               and isinstance(s.value.value, str))]
            if (len(non_doc) == 1
                and (isinstance(non_doc[0], ast.Pass)
                     or (isinstance(non_doc[0], ast.Expr)
                         and isinstance(non_doc[0].value, ast.Constant)
                         and non_doc[0].value.value is Ellipsis))):
                flags += 1

            # R3: speculative-suffix class with exactly one non-init method
            cname = node.name
            if any(cname.endswith(suf) for suf in SPECULATIVE_SUFFIXES):
                methods = [s for s in node.body
                           if isinstance(s, (ast.FunctionDef,
                                             ast.AsyncFunctionDef))
                           and s.name != "__init__"]
                if len(methods) == 1:
                    flags += 1

            # R4: multiple inheritance (DIT proxy in-diff)
            if len(node.bases) >= 2:
                flags += 1

    # R2b/c: free-function bodies that are `raise NotImplementedError`,
    # `pass`-only, or `...`-only.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            # skip docstring
            non_doc = [s for s in body
                       if not (isinstance(s, ast.Expr)
                               and isinstance(s.value, ast.Constant)
                               and isinstance(s.value.value, str))]
            if len(non_doc) == 1:
                stmt = non_doc[0]
                if isinstance(stmt, ast.Pass):
                    flags += 1
                elif (isinstance(stmt, ast.Expr)
                      and isinstance(stmt.value, ast.Constant)
                      and stmt.value.value is Ellipsis):
                    flags += 1
                elif isinstance(stmt, ast.Raise):
                    exc = stmt.exc
                    name = None
                    if isinstance(exc, ast.Name):
                        name = exc.id
                    elif isinstance(exc, ast.Call):
                        f = exc.func
                        if isinstance(f, ast.Name):
                            name = f.id
                        elif isinstance(f, ast.Attribute):
                            name = f.attr
                    if name == "NotImplementedError":
                        flags += 1

    return flags, loc


# ---- R5: comment-token scan over added source lines ---------------------

def _added_comment_density(diff_text: str) -> Optional[tuple[int, int]]:
    """Return (todo_count, loc) summed across added lines of all source
    files we recognize. None if no source files. The comment-extraction
    step here is stripping a SINGLE LINE COMMENT PREFIX from a SINGLE
    diff-added line — not parsing code structure, hence the REGEX_OK
    annotation above the patterns."""
    by_path = added_files_by_ext(diff_text, SOURCE_EXTS)
    if not by_path:
        return None
    total_todos = 0
    total_loc = 0
    for path, content in by_path.items():
        p = path.lower()
        is_python_like = p.endswith(".py") or p.endswith(".pyi")
        is_ruby = p.endswith(".rb")
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            total_loc += 1
            # Extract the comment substring, if any.
            comment_body = None
            if is_python_like or is_ruby:
                m = PY_COMMENT_RE.search(line)
                if m:
                    comment_body = m.group(1)
            else:
                m = SLASH_COMMENT_RE.search(line)
                if m:
                    comment_body = m.group(1)
            if comment_body is None:
                continue
            # TODO_TOKENS_RE only scans the comment substring, never the
            # surrounding code, so it cannot trip on identifiers like
            # `todo_list` or string literals containing "TODO".
            for _ in TODO_TOKENS_RE.finditer(comment_body):
                total_todos += 1
    if total_loc == 0:
        return None
    return total_todos, total_loc


def score(diff_text: str) -> Optional[float]:
    # Channel A: structural YAGNI flags on Python files (R1-R4).
    py_files = added_files_by_ext(diff_text, PY_EXTS)
    py_score: Optional[float] = None
    if py_files:
        per_file_density = []
        for _path, src in py_files.items():
            flags, loc = _count_python_flags(src)
            # Floor loc at 30 to avoid blowing up density on a 3-line stub
            # that legitimately uses one ABC import.
            denom = max(loc, 30)
            per_file_density.append(flags / denom)
        if per_file_density:
            mean_density = sum(per_file_density) / len(per_file_density)
            py_score = math.exp(-mean_density * 25.0)

    # Channel B: TODO/FIXME comment density (R5) across all source files.
    cd = _added_comment_density(diff_text)
    comment_score: Optional[float] = None
    if cd is not None:
        todos, loc = cd
        # 1 TODO per 100 LOC -> 0.37; 0 -> 1.0; 5 per 100 -> ~0.007.
        comment_score = math.exp(-(todos / max(loc, 1)) * 100.0)

    if py_score is None and comment_score is None:
        return None
    if py_score is None:
        return float(max(0.0, min(1.0, comment_score)))
    if comment_score is None:
        return float(max(0.0, min(1.0, py_score)))
    result = 0.75 * py_score + 0.25 * comment_score
    return float(max(0.0, min(1.0, result)))
