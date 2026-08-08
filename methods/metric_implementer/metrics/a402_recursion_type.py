"""a402: Recursion type classifier (Python).

For each function definition, count self-calls and classify:
  - 0 self-calls       : not recursive            -> score 0.5 (neutral)
  - tail-recursive     : exactly one self-call, and it appears as the
                         direct return expression of the function
                         (return f(...) with no surrounding arith) -> 1.0
  - linear recursion   : exactly one self-call (not in tail position) -> 0.7
  - tree recursion     : >=2 self-calls           -> 0.5 (often correct but
                         expensive — penalised unless memoised)
  - tree+memo          : >=2 self-calls AND @functools.lru_cache/@cache
                         decorator -> 0.85
  - mutual recursion   : function A calls B and B calls A in the same file
                         -> 0.65 (legal but often unintended)

Per-file score: mean over recursive functions. If no function is recursive
at all, return 0.5 (neutral signal: not applicable to "recursion quality").

We DO NOT score non-recursive functions because the norm is about HOW
recursion is done, not whether it's used. Files with zero recursion get
None (abstain) — keeps the signal sharp.

Examples:
  + def fact(n): return 1 if n<2 else n*fact(n-1)   # linear, non-tail -> 0.7
  + def loop(n): return loop(n-1) if n else 0       # tail-call -> 1.0
  + def fib(n): return fib(n-1)+fib(n-2)            # tree, no memo -> 0.5
  + @lru_cache
  + def fib(n): return fib(n-1)+fib(n-2)            # tree + memo -> 0.85
  + def is_even(n): return True if n==0 else is_odd(n-1)
  + def is_odd(n):  return False if n==0 else is_even(n-1)  # mutual -> 0.65

CLASSIFICATION: PARTIALLY_THIN — tail-call detection is syntactic
(direct `return f(...)`); Python doesn't optimize TCO so the value of
"tail-recursion" is stylistic. Mutual recursion is detected only within
one file (we don't see cross-file call graphs in a diff).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a402"
ASPECT_NAME = "Recursion type and quality"
TIER = 2
TOOLS = ["tree-sitter-python"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = [".py", ".pyi"]
_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        try:
            import tree_sitter_python
            from tree_sitter import Language, Parser
            _PARSER = Parser(Language(tree_sitter_python.language()))
        except ImportError:
            return None
    return _PARSER


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _function_name(fn_node, src: bytes) -> str:
    for c in fn_node.children:
        if c.type == "identifier":
            return _text(c, src)
    return ""


def _body_block(fn_node):
    for c in fn_node.children:
        if c.type == "block":
            return c
    return None


def _has_memo(fn_node, src: bytes) -> bool:
    parent = fn_node.parent
    nodes = []
    if parent is not None and parent.type == "decorated_definition":
        nodes.extend(parent.children)
    for n in nodes:
        if n.type == "decorator":
            t = _text(n, src)
            if "lru_cache" in t or "@cache" in t or "functools.cache" in t:
                return True
    return False


def _calls(node, src: bytes) -> List[str]:
    """Return [callee_name] for each call in subtree (identifier callees
    only)."""
    out: List[str] = []

    def walk(n):
        if n.type == "call":
            first = n.children[0] if n.children else None
            if first is not None and first.type == "identifier":
                out.append(_text(first, src))
        for c in n.children:
            walk(c)

    walk(node)
    return out


def _self_call_count(fn_node, name: str, src: bytes) -> int:
    body = _body_block(fn_node)
    if body is None or not name:
        return 0
    return sum(1 for c in _calls(body, src) if c == name)


def _is_tail_call(return_node, name: str, src: bytes) -> bool:
    """A return_statement whose immediate expression is a call to `name`
    (no arithmetic wrapping)."""
    # return_statement -> return [expression]
    children = [c for c in return_node.children if c.type != "return"]
    if not children:
        return False
    expr = children[0]
    # tolerate parenthesized expression
    while expr.type == "parenthesized_expression":
        inner = [c for c in expr.children if c.type
                 not in ("(", ")")]
        if not inner:
            return False
        expr = inner[0]
    # Direct call
    if expr.type == "call":
        first = expr.children[0] if expr.children else None
        if first is not None and first.type == "identifier" \
                and _text(first, src) == name:
            return True
    # Ternary: conditional_expression (e.g. `return f(n-1) if n else 0`)
    if expr.type == "conditional_expression":
        for sub in expr.children:
            if sub.type == "call":
                first = sub.children[0] if sub.children else None
                if first is not None and first.type == "identifier" \
                        and _text(first, src) == name:
                    return True
    return False


def _classify_function(fn_node, src: bytes) -> Optional[Tuple[str, float]]:
    name = _function_name(fn_node, src)
    if not name:
        return None
    body = _body_block(fn_node)
    if body is None:
        return None
    n_self = _self_call_count(fn_node, name, src)
    if n_self == 0:
        return None  # not recursive — abstain at function level
    memoised = _has_memo(fn_node, src)
    if n_self == 1:
        # check if any return statement is a tail call
        tail = False

        def walk(n):
            nonlocal tail
            if n.type == "return_statement":
                if _is_tail_call(n, name, src):
                    tail = True
            for c in n.children:
                walk(c)

        walk(body)
        if tail:
            return ("tail", 1.0)
        return ("linear", 0.7)
    # n_self >= 2
    if memoised:
        return ("tree+memo", 0.85)
    return ("tree", 0.5)


def _file_recursion_score(code: bytes) -> Optional[Tuple[float, Set[str],
                                                          Dict[str, Set[str]]]]:
    """Return (mean_score_across_recursive_fns, fn_names, callgraph)
    where callgraph maps fn -> set of identifier-callees (for mutual-recursion
    detection at the file level)."""
    parser = _get_parser()
    if parser is None:
        return None
    try:
        tree = parser.parse(code)
    except Exception:
        return None
    fn_scores: List[float] = []
    fn_names: Set[str] = set()
    callgraph: Dict[str, Set[str]] = {}

    def walk(n):
        if n.type == "function_definition":
            name = _function_name(n, code)
            if name:
                fn_names.add(name)
                body = _body_block(n)
                if body is not None:
                    callgraph[name] = set(_calls(body, code))
            cls = _classify_function(n, code)
            if cls is not None:
                fn_scores.append(cls[1])
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    if not fn_scores:
        return None

    # Mutual-recursion penalty: any pair (A,B) where A calls B and B calls A
    mutual_pairs = 0
    names_list = list(fn_names)
    for i, a in enumerate(names_list):
        for b in names_list[i + 1:]:
            if a == b:
                continue
            if b in callgraph.get(a, set()) and a in callgraph.get(b, set()):
                mutual_pairs += 1
    avg = sum(fn_scores) / len(fn_scores)
    if mutual_pairs > 0:
        # blend with mutual = 0.65 for transparency
        avg = 0.5 * avg + 0.5 * 0.65
    return (avg, fn_names, callgraph)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    if _get_parser() is None:
        return None
    scs: List[float] = []
    for content in by_path.values():
        r = _file_recursion_score(content.encode("utf8", errors="replace"))
        if r is not None:
            scs.append(r[0])
    if not scs:
        return None
    return float(sum(scs) / len(scs))
