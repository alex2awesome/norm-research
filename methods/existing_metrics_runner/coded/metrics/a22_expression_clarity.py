"""a22: Expression-level clarity and explicitness.

The norm: prefer clear, readable expressions and statements — sensible
operand ordering, appropriate parentheses/grouping, explicit forms when
they aid understanding, and correct scoping of try/except (or
try/catch). Reviewers most often flag *dense* expressions that should be
split into named subexpressions or statements.

We measure that density with four structural signals computed per added
source file via tree-sitter (Python, JavaScript, TypeScript, Java, Go):

  1. Max ternary / conditional-expression nesting depth.
     Nested ternaries are the canonical "should be an if/elif" smell.
  2. Max comprehension nesting depth (list/dict/set/generator).
     Double or triple comprehensions are the Python equivalent of nested
     ternaries (one expression encoding three loops).
  3. Max lambda / arrow / func-literal "body complexity".
     Number of boolean operators, conditionals and call/attribute nodes
     inside any one lambda body. Big lambdas should be named functions.
  4. Max method / attribute chain length (a.b.c.d().e()).
     Long fluent chains are hard to debug and parenthesize correctly.

Plus one statement-scoping signal (Tier 2, language-aware):

  5. Mean try-block body length (lines).
     "Correct scoping of try/exception handling" means try blocks should
     wrap only the lines that can raise the targeted exception. Very long
     try bodies (>10 lines) are the over-broad-try smell.

Per-signal normalization (all to [0,1], 1 = healthy):

    tern_norm  = exp(-max(0, max_tern_depth - 1) / 1.5)
        # 1 level is fine (1.00), 2 levels = 0.51, 3 = 0.26
    comp_norm  = exp(-max(0, max_comp_depth - 1) / 1.5)
        # same shape — single comprehension fine, double = 0.51
    lam_norm   = exp(-max(0, max_lambda_complexity - 3) / 4.0)
        # tiny lambdas fine (<=3 ops), 7 ops ~ 0.37
    chain_norm = exp(-max(0, max_chain_len - 3) / 3.0)
        # chains up to 3 dots fine, 6 dots ~ 0.37
    try_norm   = exp(-max(0, mean_try_body_lines - 5) / 8.0)
        # try wrapping <=5 lines fine, 13 lines ~ 0.37

Composite score (equal weight on the four expression signals, half-weight
on the try-scoping signal because the AST partial-snippet has higher
uncertainty there):

    score = (tern_norm + comp_norm + lam_norm + chain_norm) * 0.225
          + try_norm * 0.10

The 4 * 0.225 + 0.10 = 1.00 normalization keeps the output in [0,1].

Tier: 2 (pure AST via tree-sitter, no subprocess).

Classification: PARTIALLY_THIN. The signals are deterministic and the
formula is closed-form, but "expression-level clarity" also covers
parenthesization style and operand ordering — those are taste calls
deliberately omitted because no tool can score them without ground truth
from human reviewers. We measure the *density* axis, which is the part
that admits a structural definition.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a22"
ASPECT_NAME = "Expression-level clarity and explicitness"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# Per-language node-type tables. Tree-sitter grammars use slightly
# different names per language, so we route by lang_short.
TERNARY_TYPES = {
    "py": {"conditional_expression"},
    "js": {"ternary_expression"},
    "ts": {"ternary_expression"},
    "java": {"ternary_expression"},
    "go": set(),  # Go has no ternary
}
COMPREHENSION_TYPES = {
    "py": {"list_comprehension", "dictionary_comprehension",
           "set_comprehension", "generator_expression"},
    "js": set(),
    "ts": set(),
    "java": set(),
    "go": set(),
}
LAMBDA_TYPES = {
    "py": {"lambda"},
    "js": {"arrow_function", "function_expression"},
    "ts": {"arrow_function", "function_expression"},
    "java": {"lambda_expression"},
    "go": {"func_literal"},
}
# Nodes that count as one "operation" inside a lambda body
LAMBDA_OP_TYPES = {
    "py": {"boolean_operator", "comparison_operator", "conditional_expression",
           "call", "binary_operator", "unary_operator"},
    "js": {"binary_expression", "logical_expression",
           "ternary_expression", "call_expression", "unary_expression"},
    "ts": {"binary_expression", "logical_expression",
           "ternary_expression", "call_expression", "unary_expression"},
    "java": {"binary_expression", "ternary_expression",
             "method_invocation", "unary_expression"},
    "go": {"binary_expression", "call_expression", "unary_expression"},
}
# Chain heads: the outer node that contains the chain.
ATTRIBUTE_TYPES = {
    "py": {"attribute"},
    "js": {"member_expression"},
    "ts": {"member_expression"},
    "java": {"field_access"},  # method_invocation is the call wrapper
    "go": {"selector_expression"},
}
CALL_TYPES = {
    "py": {"call"},
    "js": {"call_expression"},
    "ts": {"call_expression"},
    "java": {"method_invocation"},
    "go": {"call_expression"},
}
TRY_TYPES = {
    "py": {"try_statement"},
    "js": {"try_statement"},
    "ts": {"try_statement"},
    "java": {"try_statement"},
    "go": set(),  # Go uses error returns, not try/catch
}

_PARSERS: Dict[str, object] = {}


def _get_parser(lang_short: str):
    if lang_short in _PARSERS:
        return _PARSERS[lang_short]
    try:
        from tree_sitter import Language, Parser
        if lang_short == "py":
            import tree_sitter_python as m; lang = m.language()
        elif lang_short == "js":
            import tree_sitter_javascript as m; lang = m.language()
        elif lang_short == "ts":
            import tree_sitter_typescript as m
            lang = m.language_typescript()
        elif lang_short == "java":
            import tree_sitter_java as m; lang = m.language()
        elif lang_short == "go":
            import tree_sitter_go as m; lang = m.language()
        else:
            _PARSERS[lang_short] = None
            return None
        _PARSERS[lang_short] = Parser(Language(lang))
        return _PARSERS[lang_short]
    except Exception:
        _PARSERS[lang_short] = None
        return None


def _ext_lang(path: str) -> Optional[str]:
    if "." not in path:
        return None
    ext = "." + path.rsplit(".", 1)[-1].lower()
    return EXT_TO_LANG.get(ext)


def _max_depth_of_type(node, target_types, depth: int = 0,
                       skip_root: bool = True) -> int:
    """Max nesting depth of nodes whose .type is in target_types, anywhere
    under `node`. depth=current depth (not counting node itself when
    skip_root). Used for ternary + comprehension nesting."""
    if not target_types:
        return 0
    is_target = node.type in target_types and not skip_root
    new_depth = depth + 1 if is_target else depth
    best = new_depth
    for c in node.children:
        sub = _max_depth_of_type(c, target_types, new_depth, skip_root=False)
        if sub > best:
            best = sub
    return best


def _lambda_body_complexity(node, op_types) -> int:
    """Count nodes in op_types under this lambda's body."""
    count = 0
    stack = [node]
    while stack:
        n = stack.pop()
        if n is not node and n.type in op_types:
            count += 1
        # Do NOT descend into nested lambdas — those are scored on their own.
        if n is not node and n.type in {"lambda", "arrow_function",
                                        "function_expression",
                                        "lambda_expression", "func_literal"}:
            continue
        for c in n.children:
            stack.append(c)
    return count


def _chain_length_at(node, attr_types, call_types) -> int:
    """Length of the dot/call chain rooted at this node.

    A "chain" is a sequence of attribute accesses and calls chained on each
    other. e.g. `a.b.c().d().e` has chain length 5 (b, c(), d(), e).
    We walk down the leftmost-receiver child until we leave the chain.
    """
    length = 0
    cur = node
    chain_types = attr_types | call_types
    while cur is not None and cur.type in chain_types:
        length += 1
        # The receiver is typically the first child (or the named field
        # "object"/"function"). Walk into it.
        nxt = None
        for c in cur.children:
            if c.is_named:
                nxt = c
                break
        cur = nxt
    return length


def _max_chain_anywhere(root, attr_types, call_types) -> int:
    """Max chain length over all "chain heads" — nodes whose parent is NOT
    itself a chain node. Avoids double counting inner chain prefixes.
    """
    chain_types = attr_types | call_types
    best = 0

    def walk(n, parent_is_chain):
        nonlocal best
        is_chain = n.type in chain_types
        if is_chain and not parent_is_chain:
            length = _chain_length_at(n, attr_types, call_types)
            if length > best:
                best = length
        for c in n.children:
            walk(c, is_chain)

    walk(root, False)
    return best


def _try_body_lines(root, try_types) -> Tuple[int, int]:
    """Return (sum_lines, count) over try-block bodies under root."""
    if not try_types:
        return 0, 0
    total = 0
    count = 0
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in try_types:
            # The try body is the first "block"/"compound_statement" child.
            body = None
            for c in n.children:
                if c.type in ("block", "compound_statement", "constructor_body"):
                    body = c
                    break
            if body is not None:
                lines = max(1, body.end_point[0] - body.start_point[0])
                total += lines
                count += 1
        for c in n.children:
            stack.append(c)
    return total, count


def _analyze_file(code: bytes, lang_short: str) -> Optional[dict]:
    parser = _get_parser(lang_short)
    if parser is None:
        return None
    try:
        tree = parser.parse(code)
    except Exception:
        return None
    root = tree.root_node

    tern_types = TERNARY_TYPES.get(lang_short, set())
    comp_types = COMPREHENSION_TYPES.get(lang_short, set())
    lam_types = LAMBDA_TYPES.get(lang_short, set())
    op_types = LAMBDA_OP_TYPES.get(lang_short, set())
    attr_types = ATTRIBUTE_TYPES.get(lang_short, set())
    call_types = CALL_TYPES.get(lang_short, set())
    try_types = TRY_TYPES.get(lang_short, set())

    max_tern = _max_depth_of_type(root, tern_types) if tern_types else 0
    max_comp = _max_depth_of_type(root, comp_types) if comp_types else 0

    max_lam = 0
    # find every lambda node, score its body
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in lam_types:
            c = _lambda_body_complexity(n, op_types)
            if c > max_lam:
                max_lam = c
        for c in n.children:
            stack.append(c)

    max_chain = _max_chain_anywhere(root, attr_types, call_types)
    try_sum, try_n = _try_body_lines(root, try_types)

    return {
        "max_tern": max_tern,
        "max_comp": max_comp,
        "max_lam": max_lam,
        "max_chain": max_chain,
        "try_sum": try_sum,
        "try_n": try_n,
        "have_try": bool(try_types),
    }


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    for p in by_path:
        if _ext_lang(p) is not None:
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    agg_tern = 0
    agg_comp = 0
    agg_lam = 0
    agg_chain = 0
    try_sum_all = 0
    try_n_all = 0
    n_files = 0
    have_try_lang = False
    for path, src in by_path.items():
        lang = _ext_lang(path)
        if lang is None:
            continue
        info = _analyze_file(src.encode("utf8", errors="replace"), lang)
        if info is None:
            continue
        n_files += 1
        if info["max_tern"] > agg_tern:
            agg_tern = info["max_tern"]
        if info["max_comp"] > agg_comp:
            agg_comp = info["max_comp"]
        if info["max_lam"] > agg_lam:
            agg_lam = info["max_lam"]
        if info["max_chain"] > agg_chain:
            agg_chain = info["max_chain"]
        if info["have_try"]:
            have_try_lang = True
        try_sum_all += info["try_sum"]
        try_n_all += info["try_n"]

    if n_files == 0:
        return None

    tern_norm = math.exp(-max(0, agg_tern - 1) / 1.5)
    comp_norm = math.exp(-max(0, agg_comp - 1) / 1.5)
    lam_norm = math.exp(-max(0, agg_lam - 3) / 4.0)
    chain_norm = math.exp(-max(0, agg_chain - 3) / 3.0)

    if have_try_lang and try_n_all > 0:
        mean_try = try_sum_all / try_n_all
        try_norm = math.exp(-max(0, mean_try - 5) / 8.0)
    else:
        # No try blocks seen: this signal is silent. Renormalize the four
        # expression signals to sum to 1.0.
        return float((tern_norm + comp_norm + lam_norm + chain_norm) / 4.0)

    return float(
        (tern_norm + comp_norm + lam_norm + chain_norm) * 0.225
        + try_norm * 0.10
    )
