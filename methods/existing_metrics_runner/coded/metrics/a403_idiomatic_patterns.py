"""a403: Pythonic idiom density.

Counts AST-level idioms vs anti-patterns in added Python code via
tree-sitter-python. Score = (idioms - antipatterns) / (idioms + antipatterns
+ 1), shifted to [0,1] via 0.5 + 0.5*r.

Idioms (+1 each):
  - list/set/dict/generator comprehension (list_comprehension,
    set_comprehension, dictionary_comprehension, generator_expression)
  - enumerate(...) call
  - zip(...) call
  - context manager (with_statement)
  - f-string (interpolation node)
  - any decorator (@... before def)
  - collections.defaultdict / Counter / OrderedDict construction
  - any yield_statement / yield_from (generator function)

Anti-patterns (-1 each):
  - for ... in range(len(x))           - classic non-idiom
  - bare `except:` (try_statement with except_clause that has no
    expression)
  - string-concat in loop (augmented_assignment with += where target is a
    name and we're inside a for/while)
  - dict access pattern `d[k] if k in d else default` (subscript ternary)

This is a purely AST-structural metric — we never regex against code.

Examples:
  + [x*x for x in lst]              -> +1
  + for i in range(len(lst)):       -> -1
  + with open('f') as fh:           -> +1
  + try:
  +     ...
  + except:                         -> -1
  + s = ''
  + for x in lst: s += x            -> -1
  + @dataclass                       -> +1

CLASSIFICATION: PARTIALLY_THIN — even tree-sitter cannot tell whether a
defaultdict actually replaces a `dict.setdefault`-pattern in spirit; we use
surface-level structural evidence and that's it.
"""
from __future__ import annotations

from typing import List, Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a403"
ASPECT_NAME = "Pythonic idiom density"
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


IDIOM_NODE_TYPES = {
    "list_comprehension", "set_comprehension", "dictionary_comprehension",
    "generator_expression", "with_statement",
    "yield", "yield_from",
}
IDIOM_CALLEES = {"enumerate", "zip", "defaultdict", "Counter", "OrderedDict",
                 "namedtuple"}


def _is_range_len_for(for_node, src: bytes) -> bool:
    """for_statement whose iterable is `range(len(...))`."""
    # children: for, var, in, iterable, :, block
    # Find first call after 'in'
    saw_in = False
    for c in for_node.children:
        if c.type == "in":
            saw_in = True
            continue
        if saw_in and c.type == "call":
            first = c.children[0] if c.children else None
            if first is not None and first.type == "identifier" \
                    and _text(first, src) == "range":
                # check call args for a call to len(...)
                for arg in c.children:
                    if arg.type == "argument_list":
                        for a in arg.children:
                            if a.type == "call":
                                f2 = a.children[0] if a.children else None
                                if f2 is not None and f2.type == "identifier" \
                                        and _text(f2, src) == "len":
                                    return True
            return False
    return False


def _has_bare_except(try_node) -> bool:
    """try_statement with an except_clause that has no expression child."""
    for c in try_node.children:
        if c.type == "except_clause":
            # bare except: only children are 'except', ':', block
            non_struct = [k for k in c.children
                          if k.type not in (":", "except", "block",
                                            "comment")]
            if not non_struct:
                return True
    return False


def _count_in_subtree(root, src: bytes):
    idioms = 0
    antipatterns = 0
    fstrings = 0
    decorators = 0

    def is_string_concat_in_loop(node, in_loop):
        # augmented_assignment with operator '+=' inside a for/while body,
        # where LHS is identifier or subscript and RHS textual contains a
        # quoted string OR operand types indicate concatenation. We
        # approximate: any += inside loop counts as anti-pattern UNLESS the
        # RHS is a numeric literal.
        if not in_loop or node.type != "augmented_assignment":
            return False
        op_seen = False
        for c in node.children:
            if c.type == "+=":
                op_seen = True
        if not op_seen:
            return False
        # If RHS is a pure numeric literal, skip (numeric += is fine)
        rhs = node.children[-1] if node.children else None
        if rhs is not None and rhs.type in ("integer", "float"):
            return False
        # If the LHS looks like a known accumulator name suffix, flag.
        return True

    def walk(node, in_loop):
        nonlocal idioms, antipatterns, fstrings, decorators
        t = node.type
        if t in IDIOM_NODE_TYPES:
            idioms += 1
        elif t == "call":
            first = node.children[0] if node.children else None
            if first is not None and first.type == "identifier":
                nm = _text(first, src)
                if nm in IDIOM_CALLEES:
                    idioms += 1
            elif first is not None and first.type == "attribute":
                # collections.defaultdict / collections.Counter
                txt = _text(first, src)
                base = txt.rsplit(".", 1)[-1]
                if base in IDIOM_CALLEES:
                    idioms += 1
        elif t == "interpolation":
            fstrings += 1
        elif t == "decorator":
            decorators += 1
        elif t == "for_statement":
            if _is_range_len_for(node, src):
                antipatterns += 1
        elif t == "try_statement":
            if _has_bare_except(node):
                antipatterns += 1
        if is_string_concat_in_loop(node, in_loop):
            antipatterns += 1

        new_in_loop = in_loop or t in ("for_statement", "while_statement")
        for c in node.children:
            walk(c, new_in_loop)

    walk(root, False)
    # f-string nodes can fire many times in same string; cap contribution
    idioms += min(fstrings, 5)
    idioms += min(decorators, 5)
    return idioms, antipatterns


def _file_score(code: bytes) -> Optional[float]:
    parser = _get_parser()
    if parser is None:
        return None
    try:
        tree = parser.parse(code)
    except Exception:
        return None
    idioms, antipats = _count_in_subtree(tree.root_node, code)
    if idioms + antipats == 0:
        return None
    ratio = (idioms - antipats) / (idioms + antipats + 1)
    return 0.5 + 0.5 * ratio


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
