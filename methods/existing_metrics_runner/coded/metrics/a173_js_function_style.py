"""a173: JavaScript function declarations vs expressions and placement.

The norm has two pieces:

  (1) Adopt a CONSISTENT policy for function-style: pick a side between
      `function foo() {}` (hoisted declarations) and `const foo = () => {}`
      / `const foo = function() {}` (expressions). Style guides differ on
      *which* side, so we measure DOMINANCE, not direction.

  (2) Avoid declaring functions inside non-function blocks (if / else / for
      / while / do / switch / try / catch / finally / labeled / bare-block).
      ES5 strict-mode and most style guides forbid this; tree-sitter parses
      these as `function_declaration` whose immediate parent is the
      `statement_block` of a control-flow construct rather than a function
      body or the program.

We walk the JS/TS AST with tree-sitter (no regex on code), tally:
  - n_decl       : `function_declaration` nodes (only those NOT inside a
                   non-function block — counted toward dominance)
  - n_expr_named : `function_expression` nodes (typically RHS of a
                   variable_declarator or property)
  - n_arrow      : `arrow_function` nodes
  - n_misplaced  : `function_declaration` whose enclosing scope is a
                   non-function block

Score combines two sub-scores into [0,1]:

  s_consistency = max(n_decl, n_expr_named + n_arrow) / total
                  (1 if every function is the same style, ~0.5 if split
                   50/50; on N < 2 functions consistency is trivially 1.0)

  s_placement   = 1 - n_misplaced / total_decls
                  (1 if every `function_declaration` sits in a function
                   scope / program scope; 0 if every one is nested under a
                   non-function block)

  score         = 0.5 * s_consistency + 0.5 * s_placement

CLASSIFICATION is PARTIALLY_THIN: (2) is a hard rule with deterministic
tool support (eslint `no-inner-declarations`), but (1) is convention —
neither "all declarations" nor "all arrows" is objectively wrong, so the
consistency half is a style heuristic, not a verifier.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a173"
ASPECT_NAME = "JS function declarations vs expressions and placement"
TIER = 2
TOOLS = ["tree-sitter-javascript", "tree-sitter-typescript"]
APPLIES_TO_LANGS = ["JavaScript", "TypeScript"]
CLASSIFICATION = "PARTIALLY_THIN"

JS_EXTS = (".js", ".jsx", ".mjs", ".cjs")
TS_EXTS = (".ts", ".tsx")

# Block container types that are NOT function bodies. A `function_declaration`
# whose nearest enclosing block is one of these is "misplaced".
NON_FUNCTION_BLOCK_PARENTS = frozenset({
    "if_statement", "else_clause",
    "for_statement", "for_in_statement", "for_of_statement",
    "while_statement", "do_statement",
    "switch_case", "switch_default", "switch_statement",
    "try_statement", "catch_clause", "finally_clause",
    "labeled_statement",
    "with_statement",
})

# Containers that DO count as "in a function scope":
FUNCTION_BODY_PARENTS = frozenset({
    "function_declaration", "function_expression", "arrow_function",
    "method_definition", "generator_function_declaration",
    "generator_function",
    "program", "module",  # top-level placement is fine
    "class_body",         # methods inside classes are fine
})


_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "js":
            import tree_sitter_javascript as m
            L = Language(m.language())
        elif lang == "ts":
            import tree_sitter_typescript as m
            L = Language(m.language_typescript())
        elif lang == "tsx":
            import tree_sitter_typescript as m
            L = Language(m.language_tsx())
        else:
            _PARSERS[lang] = None
            return None
        _PARSERS[lang] = Parser(L)
    except Exception:
        _PARSERS[lang] = None
    return _PARSERS[lang]


def _lang_for_ext(ext: str) -> Optional[str]:
    ext = ext.lower()
    if ext in JS_EXTS:
        return "js"
    if ext == ".tsx":
        return "tsx"
    if ext in TS_EXTS:
        return "ts"
    return None


def _is_misplaced_decl(node) -> bool:
    """Walk up from a function_declaration: is the nearest enclosing
    *scope-bearing* ancestor a non-function block?

    We climb past `statement_block` because it's the body container, then
    look at its parent — that's what answers "what kind of block am I in?".
    """
    cur = node.parent
    while cur is not None:
        t = cur.type
        # Skip the immediate statement_block wrapping the decl's siblings;
        # the meaning lives in its parent.
        if t == "statement_block":
            cur = cur.parent
            continue
        if t in FUNCTION_BODY_PARENTS:
            return False
        if t in NON_FUNCTION_BLOCK_PARENTS:
            return True
        # ERROR nodes (partial diffs) -> climb past.
        cur = cur.parent
    # Reached the top without classification -> treat as top-level (fine).
    return False


def _tally(code: bytes, lang: str) -> Tuple[int, int, int, int]:
    """Return (n_decl, n_expr_named, n_arrow, n_misplaced_decl)."""
    parser = _get_parser(lang)
    if parser is None:
        return (0, 0, 0, 0)
    root = parser.parse(code).root_node
    n_decl = n_expr = n_arrow = n_misplaced = 0

    def walk(n):
        nonlocal n_decl, n_expr, n_arrow, n_misplaced
        t = n.type
        if t == "function_declaration":
            if _is_misplaced_decl(n):
                n_misplaced += 1
            n_decl += 1
        elif t == "function_expression":
            n_expr += 1
        elif t == "arrow_function":
            n_arrow += 1
        for c in n.children:
            walk(c)

    walk(root)
    return (n_decl, n_expr, n_arrow, n_misplaced)


def _collect(diff_text: str) -> Optional[Tuple[int, int, int, int]]:
    """Sum per-file tallies across JS/TS files in the diff's added lines.

    Returns None if no JS/TS file present.
    """
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    totals = [0, 0, 0, 0]
    any_lang_file = False
    for path, body in by_path.items():
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = _lang_for_ext(ext)
        if lang is None:
            continue
        any_lang_file = True
        d, e, a, m = _tally(body.encode("utf8", errors="replace"), lang)
        totals[0] += d
        totals[1] += e
        totals[2] += a
        totals[3] += m
    if not any_lang_file:
        return None
    return tuple(totals)  # type: ignore[return-value]


def applies(diff_text: str) -> bool:
    """Apply iff the diff adds a JS/TS file AND we observe at least one
    function-like construct (otherwise there's nothing to score)."""
    tot = _collect(diff_text)
    if tot is None:
        return False
    n_decl, n_expr, n_arrow, _ = tot
    return (n_decl + n_expr + n_arrow) > 0


def score(diff_text: str) -> Optional[float]:
    tot = _collect(diff_text)
    if tot is None:
        return None
    n_decl, n_expr, n_arrow, n_misplaced = tot
    total = n_decl + n_expr + n_arrow
    if total == 0:
        return None

    # Consistency: how dominant is the more-used style?
    decl_side = n_decl
    expr_side = n_expr + n_arrow
    if total < 2:
        s_consistency = 1.0
    else:
        s_consistency = max(decl_side, expr_side) / total

    # Placement: fraction of declarations NOT in a non-function block.
    if n_decl == 0:
        s_placement = 1.0
    else:
        s_placement = 1.0 - (n_misplaced / n_decl)

    return float(0.5 * s_consistency + 0.5 * s_placement)
