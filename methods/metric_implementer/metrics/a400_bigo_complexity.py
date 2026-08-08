"""a400: Asymptotic (Big-O) complexity classifier.

Detects nested-loop depth and recursion in the ADDED Python code of a diff,
then maps to a complexity class. We walk the tree-sitter-python AST.

Classification ladder (per function):
  - 0: O(1)        — no loops, no recursion
  - 1: O(log n)    — single loop whose iterator step is *= / //= /
                     a binary-search-shape (low/mid/high pattern); heuristic
  - 2: O(n)        — single loop OR linear recursion (one self-call)
  - 3: O(n log n)  — loop containing a recursive call, OR call to sorted()
                     in the same body as a linear loop
  - 4: O(n^2)      — depth-2 nested loop, or two-self-call recursion (tree)
  - 5: O(n^3)      — depth-3 nested loop
  - 6: O(2^n)      — recursion with >=2 self-calls per branch AND no memo
                     decorator (functools.lru_cache / functools.cache)

Per function we take the max class index reached; per file we take the max
across functions; per diff we take the max across files. Score = exp(-k/3.0).

Why a heuristic and not a static tool: there is no off-the-shelf big-O
classifier — even Loopy / SLAyer require manually annotated invariants.
Our ladder is recognised pattern detection on the AST, not symbolic analysis.

Examples:
  + def f(x): return x + 1               -> O(1)   score ~1.00
  + for i in range(n): a[i] = i          -> O(n)   score ~0.51
  + for i in range(n):
  +     for j in range(n): a[i][j] = 0   -> O(n^2) score ~0.26
  + def fib(n):
  +     if n<2: return n
  +     return fib(n-1) + fib(n-2)       -> O(2^n) score ~0.13
  + def bsearch(arr, t):
  +     lo, hi = 0, len(arr)
  +     while lo < hi:
  +         mid = (lo+hi)//2
  +         if arr[mid]==t: return mid
  +         elif arr[mid]<t: lo = mid+1
  +         else: hi = mid
  +     return -1                        -> O(log n) score ~0.72

CLASSIFICATION: PARTIALLY_THIN — the binary-search and memoisation heuristics
are surface signals; some logarithmic patterns will be misclassified as O(n).
"""
from __future__ import annotations

import math
from typing import List, Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a400"
ASPECT_NAME = "Asymptotic complexity (Big-O) class"
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


def _has_memo_decorator(fn_node, src: bytes) -> bool:
    """Walk siblings backward for @lru_cache / @cache / @functools.cache."""
    parent = fn_node.parent
    if parent is None:
        return False
    # tree-sitter-python attaches decorators as decorator nodes that are
    # siblings of function_definition under decorated_definition. Check the
    # function_definition's own children too (some grammar versions).
    nodes = []
    if parent.type == "decorated_definition":
        nodes.extend(parent.children)
    nodes.extend(fn_node.children)
    for n in nodes:
        if n.type == "decorator":
            txt = _text(n, src)
            if ("lru_cache" in txt or "@cache" in txt
                    or "functools.cache" in txt):
                return True
    return False


def _loop_depth(node) -> int:
    """Max nested loop depth strictly inside this node (excluding self)."""
    best = 0

    def walk(n, depth):
        nonlocal best
        if n.type in ("for_statement", "while_statement"):
            depth += 1
            if depth > best:
                best = depth
        for c in n.children:
            walk(c, depth)

    walk(node, 0)
    return best


def _is_binary_search_loop(loop_node, src: bytes) -> bool:
    """Heuristic: while-loop body mentions 'mid', '//2' OR '/2', and updates
    a 'lo'/'low'/'left' or 'hi'/'high'/'right' variable. Cheap signal."""
    if loop_node.type != "while_statement":
        return False
    body_text = _text(loop_node, src)
    has_mid = "mid" in body_text or "middle" in body_text
    has_half = "//2" in body_text or "// 2" in body_text or ">> 1" in body_text
    has_bounds = any(k in body_text for k in
                     ("lo,", "lo ", "low,", "low ", "left",
                      "hi,", "hi ", "high,", "high ", "right"))
    return has_mid and has_half and has_bounds


def _is_log_step_loop(loop_node, src: bytes) -> bool:
    """Loop whose induction variable updates with *=, //=, /=, or >>=.
    e.g. while i < n: i *= 2."""
    body = loop_node
    body_text = _text(body, src)
    for op in ("*= 2", "*=2", "//= 2", "//=2", ">>= 1", ">>=1", "/= 2"):
        if op in body_text:
            return True
    return False


def _self_call_count(fn_node, name: str, src: bytes) -> int:
    """Count direct recursive calls of this function in its body."""
    count = 0

    def walk(n):
        nonlocal count
        if n.type == "call":
            # call -> identifier(args) or attribute(...)
            first = n.children[0] if n.children else None
            if first is not None and first.type == "identifier":
                if _text(first, src) == name:
                    count += 1
        for c in n.children:
            walk(c)

    body = None
    for c in fn_node.children:
        if c.type == "block":
            body = c
            break
    if body is None:
        return 0
    walk(body)
    return count


def _has_sorted_call(node, src: bytes) -> bool:
    def walk(n):
        if n.type == "call":
            first = n.children[0] if n.children else None
            if first is not None and first.type == "identifier":
                if _text(first, src) in ("sorted", "sort"):
                    return True
        return any(walk(c) for c in n.children)

    return walk(node)


def _has_recursive_call_in_loop(fn_node, name: str, src: bytes) -> bool:
    """True if a recursive call appears inside a loop body."""
    found = False

    def walk(n, in_loop):
        nonlocal found
        if found:
            return
        if n.type in ("for_statement", "while_statement"):
            in_loop = True
        if in_loop and n.type == "call":
            first = n.children[0] if n.children else None
            if first is not None and first.type == "identifier" \
                    and _text(first, src) == name:
                found = True
                return
        for c in n.children:
            walk(c, in_loop)

    walk(fn_node, False)
    return found


def _classify_function(fn_node, src: bytes) -> int:
    """Return complexity class index (see ladder above)."""
    # function name
    name = ""
    for c in fn_node.children:
        if c.type == "identifier":
            name = _text(c, src)
            break

    depth = _loop_depth(fn_node)
    self_calls = _self_call_count(fn_node, name, src) if name else 0
    memoised = _has_memo_decorator(fn_node, src)

    # exponential: >=2 self-calls and no memo
    if self_calls >= 2 and not memoised:
        return 6

    # n log n: loop + recursive call
    if depth >= 1 and self_calls >= 1:
        return 3
    if depth >= 1 and _has_sorted_call(fn_node, src):
        # sorted is O(n log n); a single linear loop + sorted = n log n
        return 3

    # nested loop heuristics
    if depth >= 3:
        return 5
    if depth == 2:
        return 4

    if depth == 1:
        # check for log-step loop pattern
        # find the (outer) while/for in the function body
        is_log = False

        def find_loops(n, acc):
            if n.type in ("while_statement", "for_statement"):
                acc.append(n)
                return  # don't recurse into nested loop
            for c in n.children:
                find_loops(c, acc)

        loops: List = []
        find_loops(fn_node, loops)
        for lp in loops:
            if _is_binary_search_loop(lp, src) or _is_log_step_loop(lp, src):
                is_log = True
                break
        if is_log:
            return 1
        return 2

    # no loops
    if self_calls == 1:
        return 2  # linear recursion ~ O(n)
    return 0


def _file_classes(code: bytes) -> List[int]:
    parser = _get_parser()
    if parser is None:
        return []
    try:
        tree = parser.parse(code)
    except Exception:
        return []
    root = tree.root_node
    out: List[int] = []

    def walk(n):
        if n.type == "function_definition":
            out.append(_classify_function(n, code))
        for c in n.children:
            walk(c)

    walk(root)
    return out


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    if _get_parser() is None:
        return None
    max_k = -1
    n_seen = 0
    for content in by_path.values():
        ks = _file_classes(content.encode("utf8", errors="replace"))
        for k in ks:
            n_seen += 1
            if k > max_k:
                max_k = k
    if n_seen == 0:
        return None
    return float(math.exp(-max_k / 3.0))
