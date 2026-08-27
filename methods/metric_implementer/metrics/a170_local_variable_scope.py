"""a170: Local variable scope and initialization.

The norm: declare each local variable in the narrowest scope, initialize it
at its declaration, place that declaration close to its first use, and never
read it before it has been assigned.

We measure conformance by language using deterministic structural signals:

  Python (Tier 3 + Tier 2):
    - ruff `F841` (`Local variable ... assigned but never used`) flags
      variables whose lifespan is wholly wasted — a strong negative signal
      for "minimize lifespan".
    - ruff `F823` / `F821` flag local-read-before-assignment patterns.
    - AST pass over each `function_definition`: for every local variable
      first assigned at top of the function body but first read deeper
      (inside an `if`/`for`/`while`/`with`/`try` further down), we record
      "declared too early". This proxies the "declare close to first use"
      rule. Variables declared and used at the same nesting depth count
      as conformant.

  JavaScript / TypeScript (Tier 2, tree-sitter):
    - Count `var x = ...` (function-scoped, legacy, encourages hoisting and
      wider lifespan) vs `let` / `const` (block-scoped, narrower lifespan).
      Score component = fraction of declarations using `let`/`const`.
    - A `let` whose declarator has no initializer (`let x; ...; x = 1;`)
      counts against initialization-at-declaration.

  Java (Tier 2, tree-sitter):
    - For each `local_variable_declaration` inside a method body, check
      (a) the declarator has an initializer (initialize-at-declaration),
      (b) the declaration carries the `final` modifier (immutable preferred
      under modern Java style).
    - Score = mean of (a) and (b) across local variable declarations.

Per-file score is averaged across the three sub-scores that fire; overall
score is the unweighted mean across scored files. Abstain (None) when the
diff contains no Python / JS / TS / Java additions OR when no local
variable declaration is observable in those additions (e.g. the diff is all
comments, imports, JSON, markdown).

CLASSIFICATION = PARTIALLY_THIN. F841 and `var` vs `let` are objectively
detectable rule violations (THIN). "Declare close to first use" and
"prefer final" are stylistic conventions that have widespread but not
universal adoption — measurable, but not enforced. Hence PARTIALLY_THIN.
"""
from __future__ import annotations

import math
import shutil
from typing import Dict, List, Optional, Tuple

from ..sandbox import (added_files_by_ext, have_tool, parse_diff_added_by_file,
                       run, write_temp_files)

ASPECT_ID = "a170"
ASPECT_NAME = "Local variable scope and initialization"
TIER = 3
TOOLS = ["ruff", "tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = [".py", ".pyi"]
JS_EXTS = [".js", ".jsx", ".mjs", ".cjs"]
TS_EXTS = [".ts", ".tsx"]
JAVA_EXTS = [".java"]


# ----- parser cache ---------------------------------------------------------

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "py":
            import tree_sitter_python as m
            L = Language(m.language())
        elif lang == "js":
            import tree_sitter_javascript as m
            L = Language(m.language())
        elif lang == "ts":
            import tree_sitter_typescript as m
            L = Language(m.language_typescript())
        elif lang == "tsx":
            import tree_sitter_typescript as m
            L = Language(m.language_tsx())
        elif lang == "java":
            import tree_sitter_java as m
            L = Language(m.language())
        else:
            _PARSERS[lang] = None
            return None
        _PARSERS[lang] = Parser(L)
    except Exception:
        _PARSERS[lang] = None
    return _PARSERS[lang]


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# ----- Python ---------------------------------------------------------------

# Container node types whose CHILDREN sit at a deeper nesting level than the
# function body itself. Used to detect "declared at top, first used deeper".
_PY_NESTING_BLOCKS = frozenset({
    "if_statement", "elif_clause", "else_clause",
    "for_statement", "while_statement",
    "with_statement", "try_statement", "except_clause", "finally_clause",
    "match_statement", "case_clause",
})


def _py_collect_locals_close_to_use(root, src: bytes) -> Tuple[int, int]:
    """For each function_definition, walk its body. For each plain
    `var = ...` at the top level of that body (depth-0), check whether
    `var` is first *read* at depth-0 too. If first read is at depth >= 1,
    that's "declared earlier than needed". Returns (n_locals_examined,
    n_close_to_first_use).
    """
    n_total = 0
    n_close = 0

    def walk_fn(fn_node):
        nonlocal n_total, n_close
        # locate the body block of this function_definition
        body = None
        for c in fn_node.children:
            if c.type == "block":
                body = c
                break
        if body is None:
            return
        # collect top-of-body assignments: name -> position index
        # We restrict to the simplest LHS: a single identifier.
        top_assigns: Dict[str, int] = {}
        # children of `block` are statements; expression_statement may wrap
        # an `assignment` node.
        statements = list(body.children)
        for idx, stmt in enumerate(statements):
            if stmt.type != "expression_statement":
                continue
            for c in stmt.children:
                if c.type != "assignment":
                    continue
                lhs = c.children[0] if c.children else None
                if lhs is not None and lhs.type == "identifier":
                    name = _text(lhs, src)
                    if name not in top_assigns:
                        top_assigns[name] = idx
        if not top_assigns:
            return
        # For each top-of-body assignment, find the first time `name`
        # appears as an identifier read in a statement after the assignment.
        # If that first read sits inside a nested block (depth >= 1), mark
        # "declared too early"; otherwise close-to-first-use.
        for name, decl_idx in top_assigns.items():
            n_total += 1
            first_read_depth = _first_read_depth_in(
                statements[decl_idx + 1:], name, src)
            if first_read_depth is None:
                # No read at all: F841-style waste, count as NOT close.
                # F841 handles this separately too, but counting it here
                # means a missed F841 doesn't double-let it pass.
                continue
            if first_read_depth == 0:
                n_close += 1

    def walk(node):
        if node.type == "function_definition":
            walk_fn(node)
        for c in node.children:
            walk(c)

    walk(root)
    return n_total, n_close


def _first_read_depth_in(stmts, name: str, src: bytes) -> Optional[int]:
    """Find first identifier-read of `name` in the given list of statements.
    Return the nesting depth (0 if at top of this stmt list, >=1 if inside
    a control-flow block). Returns None if never read.

    "Read" = appears as an `identifier` token NOT on the LHS of an
    assignment statement.
    """
    for stmt in stmts:
        d = _find_read_depth(stmt, name, src, depth=0,
                             skip_assign_lhs=True)
        if d is not None:
            return d
    return None


def _find_read_depth(node, name: str, src: bytes, depth: int,
                     skip_assign_lhs: bool) -> Optional[int]:
    # If this node is an assignment, skip the LHS identifier (a write, not
    # a read), but DO scan the RHS.
    if node.type == "assignment" and skip_assign_lhs and node.children:
        # children layout: [lhs, '=', rhs] typically.
        for c in node.children[1:]:
            d = _find_read_depth(c, name, src, depth,
                                 skip_assign_lhs=True)
            if d is not None:
                return d
        return None
    if node.type == "identifier":
        if _text(node, src) == name:
            return depth
        return None
    # Nesting: any control-flow construct OR a `block` opens a new depth.
    new_depth = depth
    if node.type in _PY_NESTING_BLOCKS:
        new_depth = depth + 1
    for c in node.children:
        d = _find_read_depth(c, name, src, new_depth,
                             skip_assign_lhs=skip_assign_lhs)
        if d is not None:
            return d
    return None


def _py_ruff_f841_rate(by_path: Dict[str, str]) -> Optional[float]:
    """Fraction of added Python lines that DO NOT carry an F841 finding.
    Returns None if ruff unavailable or no file present.
    """
    if not by_path:
        return None
    if not have_tool("ruff"):
        return None
    total_lines = sum(1 + t.count("\n") for t in by_path.values())
    if total_lines == 0:
        return None
    td = write_temp_files(by_path)
    try:
        rc, out, _ = run(
            ["ruff", "check", "--no-cache",
             "--select=F841,F823,F821",
             "--output-format=concise", "--exit-zero", str(td)],
            timeout=15.0,
        )
        if rc < 0:
            return None
        n_findings = sum(1 for ln in out.splitlines()
                         if ln.strip()
                         and ("F84" in ln or "F82" in ln))
        # Smooth: exp(-rate * 40) so 0 findings = 1.0, 0.025/line = 0.37
        density = n_findings / max(total_lines, 1)
        return float(math.exp(-density * 40.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _py_file_score(code: bytes) -> Optional[float]:
    """Tree-sitter half: fraction of top-of-body declarations whose first
    read is also at top-of-body depth (i.e. declared close to first use).
    """
    p = _get_parser("py")
    if p is None:
        return None
    root = p.parse(code).root_node
    n_total, n_close = _py_collect_locals_close_to_use(root, code)
    if n_total == 0:
        return None
    return n_close / n_total


# ----- JavaScript / TypeScript ---------------------------------------------

def _js_walk(code: bytes, lang: str) -> Tuple[int, int, int, int]:
    """Returns (n_var_decls, n_let_const_decls, n_let_uninit, n_let_total)
    for body-level local variables.
    """
    parser = _get_parser(lang)
    if parser is None:
        return (0, 0, 0, 0)
    root = parser.parse(code).root_node
    n_var = n_let_const = n_let_uninit = n_let = 0

    def walk(node):
        nonlocal n_var, n_let_const, n_let_uninit, n_let
        t = node.type
        if t == "variable_declaration":
            # `var x = ...;` — one or more declarators
            n_var += sum(1 for c in node.children
                         if c.type == "variable_declarator")
        elif t == "lexical_declaration":
            is_let = any(c.type == "let" for c in node.children)
            n_decls = 0
            n_no_init = 0
            for c in node.children:
                if c.type == "variable_declarator":
                    n_decls += 1
                    has_init = any(cc.type == "="
                                   for cc in c.children)
                    if not has_init:
                        n_no_init += 1
            n_let_const += n_decls
            if is_let:
                n_let += n_decls
                n_let_uninit += n_no_init
        for c in node.children:
            walk(c)

    walk(root)
    return (n_var, n_let_const, n_let_uninit, n_let)


def _js_file_score(code: bytes, lang: str) -> Optional[float]:
    n_var, n_lc, n_uninit, n_let = _js_walk(code, lang)
    n_total = n_var + n_lc
    if n_total == 0:
        return None
    # Two parts: (1) fraction NOT using `var`, (2) fraction of `let` that
    # were initialized at declaration. We average them; if there are zero
    # `let`, use only (1).
    s_var = n_lc / n_total
    if n_let == 0:
        return float(s_var)
    s_init = 1.0 - (n_uninit / n_let)
    return float(0.5 * s_var + 0.5 * s_init)


# ----- Java -----------------------------------------------------------------

def _java_walk(code: bytes) -> Tuple[int, int, int]:
    """Returns (n_locals, n_initialized, n_final). Only `local_variable_
    declaration` inside method/constructor bodies is counted.
    """
    parser = _get_parser("java")
    if parser is None:
        return (0, 0, 0)
    root = parser.parse(code).root_node
    n_locals = n_init = n_final = 0

    def walk(node, inside_method: bool):
        nonlocal n_locals, n_init, n_final
        t = node.type
        if t in ("method_declaration", "constructor_declaration"):
            for c in node.children:
                walk(c, True)
            return
        if inside_method and t == "local_variable_declaration":
            modifiers_txt = ""
            for c in node.children:
                if c.type == "modifiers":
                    modifiers_txt = _text(c, code)
            has_final = "final" in modifiers_txt
            for c in node.children:
                if c.type == "variable_declarator":
                    n_locals += 1
                    if has_final:
                        n_final += 1
                    # An initialized declarator has `=` among its children.
                    if any(cc.type == "=" for cc in c.children):
                        n_init += 1
        for c in node.children:
            walk(c, inside_method)

    walk(root, inside_method=False)
    return n_locals, n_init, n_final


def _java_file_score(code: bytes) -> Optional[float]:
    n_loc, n_init, n_final = _java_walk(code)
    if n_loc == 0:
        return None
    s_init = n_init / n_loc
    s_final = n_final / n_loc
    return float(0.5 * s_init + 0.5 * s_final)


# ----- Dispatch -------------------------------------------------------------

def _ext(path: str) -> str:
    return "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""


def _lang_for(path: str) -> Optional[str]:
    e = _ext(path)
    if e in PY_EXTS:
        return "py"
    if e in JS_EXTS:
        return "js"
    if e == ".tsx":
        return "tsx"
    if e in TS_EXTS:
        return "ts"
    if e in JAVA_EXTS:
        return "java"
    return None


def applies(diff_text: str) -> bool:
    """True if any added file is Python / JS / TS / Java."""
    by_path = parse_diff_added_by_file(diff_text)
    return any(_lang_for(p) is not None for p in by_path)


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    file_scores: List[float] = []

    # Python: combine ruff F841 density with tree-sitter "close to first
    # use" measurement.
    py_files = {p: t for p, t in by_path.items() if _lang_for(p) == "py"}
    if py_files:
        s_ruff = _py_ruff_f841_rate(py_files)
        per_file_close: List[float] = []
        for path, body in py_files.items():
            s = _py_file_score(body.encode("utf8", errors="replace"))
            if s is not None:
                per_file_close.append(s)
        s_close = (sum(per_file_close) / len(per_file_close)
                   if per_file_close else None)
        # Combine: average of available components.
        parts = [x for x in (s_ruff, s_close) if x is not None]
        if parts:
            file_scores.append(sum(parts) / len(parts))

    # JS/TS: per-file score over the var/let/const balance.
    for path, body in by_path.items():
        lang = _lang_for(path)
        if lang in ("js", "ts", "tsx"):
            s = _js_file_score(body.encode("utf8", errors="replace"), lang)
            if s is not None:
                file_scores.append(s)

    # Java: per-file initialization + final fraction.
    for path, body in by_path.items():
        if _lang_for(path) == "java":
            s = _java_file_score(body.encode("utf8", errors="replace"))
            if s is not None:
                file_scores.append(s)

    if not file_scores:
        return None
    return float(sum(file_scores) / len(file_scores))
