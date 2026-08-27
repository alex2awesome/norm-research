"""a404: Data-structure choice appropriateness (Python).

Walks the AST and flags:
  Positive signals (+1 each occurrence, capped):
    - heapq.* call             (priority queue)
    - collections.deque(...)   (O(1) ends)
    - collections.Counter(...)
    - collections.defaultdict(...)
    - collections.OrderedDict(...)
    - set / frozenset constructor or set literal of size > 1
    - bisect.* call            (sorted-list binary insert/search)

  Negative signals (-1 each):
    - list-as-queue:           `lst.pop(0)`   (O(n) pop)
    - membership test on a *list literal*:   `x in [1,2,3]`  (use set/tuple)
    - membership test on a name that was assigned a list literal earlier in
      the same file (cheap intra-file flow): `xs = [1,2,3]; if x in xs: ...`
    - manual sort: `sorted(...)` followed by `[0]`/`[-1]` ONLY (use
      min/max); flagged for `sorted(...)[0]` and `sorted(...)[-1]`

Score = 0.5 + 0.5 * (good - bad) / (good + bad + 1), clipped to [0,1].

Examples:
  + import heapq
  + heapq.heappush(heap, x)              -> +1
  + q.pop(0)                              -> -1
  + if x in [1, 2, 3]: ...                -> -1
  + sorted(xs)[0]                         -> -1 (should be min)
  + d = defaultdict(list)                 -> +1
  + sorted(xs)[2]                         ->  0 (general indexing, not a smell)

CLASSIFICATION: PARTIALLY_THIN — appropriateness depends on access
patterns we don't see. We measure surface evidence of "knows about the
stdlib data structures" vs "uses the slowest path".
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from ..sandbox import added_files_by_ext

ASPECT_ID = "a404"
ASPECT_NAME = "Data-structure choice appropriateness"
TIER = 2
TOOLS = ["tree-sitter-python"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = [".py", ".pyi"]

GOOD_CALL_NAMES = {"deque", "Counter", "defaultdict", "OrderedDict",
                   "heappush", "heappop", "heapify", "nlargest", "nsmallest",
                   "insort", "insort_left", "insort_right", "bisect",
                   "bisect_left", "bisect_right", "frozenset"}

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


def _call_callee_name(call_node, src: bytes) -> Optional[str]:
    first = call_node.children[0] if call_node.children else None
    if first is None:
        return None
    if first.type == "identifier":
        return _text(first, src)
    if first.type == "attribute":
        txt = _text(first, src)
        return txt.rsplit(".", 1)[-1]
    return None


def _is_list_pop_zero(call_node, src: bytes) -> bool:
    """call: attribute_or_identifier( args ); detect `<x>.pop(0)`."""
    first = call_node.children[0] if call_node.children else None
    if first is None or first.type != "attribute":
        return False
    txt = _text(first, src)
    if not txt.endswith(".pop"):
        return False
    # Check argument list for a literal 0
    for c in call_node.children:
        if c.type == "argument_list":
            args = [a for a in c.children
                    if a.type not in ("(", ")", ",")]
            if len(args) == 1 and args[0].type == "integer" \
                    and _text(args[0], src) == "0":
                return True
    return False


def _find_list_literal_assignments(root, src: bytes) -> Set[str]:
    """Return set of identifier names assigned to a list literal at any
    scope in the file."""
    out: Set[str] = set()

    def walk(n):
        if n.type == "assignment":
            # children: [lhs, '=', rhs]
            lhs = n.children[0] if n.children else None
            rhs = n.children[-1] if len(n.children) >= 3 else None
            if lhs is not None and lhs.type == "identifier" \
                    and rhs is not None and rhs.type == "list":
                out.add(_text(lhs, src))
        for c in n.children:
            walk(c)

    walk(root)
    return out


def _count_signals(root, src: bytes):
    good = 0
    bad = 0
    list_vars = _find_list_literal_assignments(root, src)

    def walk(node):
        nonlocal good, bad
        t = node.type
        if t == "call":
            nm = _call_callee_name(node, src)
            if nm in GOOD_CALL_NAMES:
                good += 1
            if _is_list_pop_zero(node, src):
                bad += 1
            # sorted(...)[0] or sorted(...)[-1] handled via subscript below
        elif t == "set":
            # set literal {1,2,3}, count once if length > 1
            n_items = sum(1 for c in node.children
                          if c.type not in ("{", "}", ","))
            if n_items >= 2:
                good += 1
        elif t == "comparison_operator":
            # detect `x in <list literal>` and `x in <name in list_vars>`
            kids = node.children
            ops = [k for k in kids if k.type == "in"]
            if ops:
                # Comparison structure is left op right ...
                # The RHS we care about is whatever comes after 'in'
                idx = None
                for i, k in enumerate(kids):
                    if k.type == "in":
                        idx = i
                        break
                if idx is not None and idx + 1 < len(kids):
                    rhs = kids[idx + 1]
                    if rhs.type == "list":
                        bad += 1
                    elif rhs.type == "identifier" \
                            and _text(rhs, src) in list_vars:
                        bad += 1
        elif t == "subscript":
            # sorted(...)[0] / sorted(...)[-1] -> bad
            base = node.children[0] if node.children else None
            if base is not None and base.type == "call":
                first = base.children[0] if base.children else None
                if first is not None and first.type == "identifier" \
                        and _text(first, src) == "sorted":
                    # find the subscript value
                    inside = [c for c in node.children
                              if c.type not in ("[", "]", base.type)]
                    # Heuristic: look at slice/index node text
                    for c in node.children:
                        if c.type in ("integer",):
                            if _text(c, src) in ("0",):
                                bad += 1
                        elif c.type == "unary_operator":
                            t2 = _text(c, src).strip()
                            if t2 == "-1":
                                bad += 1
        for c in node.children:
            walk(c)

    walk(root)
    return good, bad


def _file_score(code: bytes) -> Optional[float]:
    parser = _get_parser()
    if parser is None:
        return None
    try:
        tree = parser.parse(code)
    except Exception:
        return None
    good, bad = _count_signals(tree.root_node, code)
    if good + bad == 0:
        return None
    r = (good - bad) / (good + bad + 1)
    return float(max(0.0, min(1.0, 0.5 + 0.5 * r)))


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    scs: List[float] = []
    for content in by_path.values():
        s = _file_score(content.encode("utf8", errors="replace"))
        if s is not None:
            scs.append(s)
    if not scs:
        return None
    return float(sum(scs) / len(scs))
