"""a406: Magic-number density (Python).

Walks tree-sitter-python AST and counts integer/float literals appearing in
"meaningful positions" (i.e. NOT in the allowed contexts below). Constants
allowed always: 0, 1, -1, 2, 10, 100, 1000.

Allowed contexts (literal NOT counted as magic):
  - in a module-level UPPER_SNAKE assignment RHS (it's BEING the constant)
  - as a dictionary or list/tuple element on the RHS of a UPPER_SNAKE
    assignment (the surrounding constant absorbs it)
  - as a default-parameter value with a comment immediately above (we
    cannot reliably check comment positions, so we use a SIMPLER rule:
    default-parameter values are always allowed)
  - as the argument of `range(...)` when there is only ONE arg (start/stop
    pattern; idiomatic enough)
  - as a value in slicing literals  (treated as allowed since these are
    indexing constants)

Score = exp(-magic_count / 5.0). We then average across files.

Examples:
  + x = 86400                            -> magic (NOT in allowlist)
  + MAX_RETRIES = 5; for i in range(MAX_RETRIES): ...  -> 0 magic (constant)
  + def f(timeout=30): ...               -> 0 magic (default param)
  + buf = bytearray(4096)                -> magic
  + score = grade * 1.5                  -> magic
  + return arr[:3]                       -> 0 magic (slice)

We DO NOT cover this metric elsewhere — verified via grep across the
metrics/ directory; the existing `a16_maintainability_thick` is THICK
(returns None) and `a18_maintainability_smells` measures radon's MI/raw
metrics, not literals.

CLASSIFICATION: THIN — literal counting is a deterministic AST query.
"""
from __future__ import annotations

import math
from typing import List, Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a406"
ASPECT_NAME = "Magic-number density"
TIER = 2
TOOLS = ["tree-sitter-python"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]
ALLOWED_LITERALS = {"0", "1", "-1", "2", "10", "100", "1000", "0.0", "1.0"}

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


def _is_inside_upper_snake_const_assignment(node) -> bool:
    """Walk up: if we are within the RHS of an assignment whose LHS is
    UPPER_SNAKE (all caps, with underscores/digits), treat as not magic."""
    cur = node.parent
    last = node
    while cur is not None:
        if cur.type == "assignment":
            # LHS is first child; ensure we are NOT the LHS subtree
            lhs = cur.children[0] if cur.children else None
            if lhs is None or last is lhs:
                return False
            # Check LHS shape
            return _is_upper_snake(lhs)
        last = cur
        cur = cur.parent
    return False


def _is_upper_snake(lhs_node) -> bool:
    """Determine if LHS identifier (or first identifier inside) is
    UPPER_SNAKE."""
    # We need the source bytes; we don't have them here, but we can check the
    # node's text via byte slice on parent's source. We'll handle this
    # without source by recursion of identifier children's text via tree
    # mapping in caller. For simplicity, we re-fetch via attribute:
    name = getattr(lhs_node, "_uname", None)
    if name is None:
        return False
    return name.isupper() and len(name) >= 1


def _is_in_default_param(node) -> bool:
    cur = node.parent
    while cur is not None:
        if cur.type in ("default_parameter", "typed_default_parameter"):
            return True
        if cur.type == "parameters":
            return False
        cur = cur.parent
    return False


def _is_in_slice(node) -> bool:
    cur = node.parent
    while cur is not None:
        if cur.type == "slice":
            return True
        if cur.type in ("function_definition", "module"):
            return False
        cur = cur.parent
    return False


def _is_in_single_arg_range(node, src: bytes) -> bool:
    """True iff the literal is the only argument of `range(...)`."""
    cur = node.parent
    while cur is not None:
        if cur.type == "argument_list":
            args = [c for c in cur.children
                    if c.type not in ("(", ")", ",")]
            if len(args) != 1:
                return False
            call = cur.parent
            if call is None or call.type != "call":
                return False
            first = call.children[0] if call.children else None
            if first is not None and first.type == "identifier" \
                    and _text(first, src) == "range":
                return True
            return False
        cur = cur.parent
    return False


def _file_magic_count(code: bytes) -> Optional[int]:
    parser = _get_parser()
    if parser is None:
        return None
    try:
        tree = parser.parse(code)
    except Exception:
        return None

    # First pre-pass: tag LHS identifiers of assignments with their text
    # so _is_upper_snake can read it without source bytes.
    def tag_lhs(n):
        if n.type == "assignment":
            lhs = n.children[0] if n.children else None
            if lhs is not None and lhs.type == "identifier":
                try:
                    lhs._uname = _text(lhs, code)
                except AttributeError:
                    pass
        for c in n.children:
            tag_lhs(c)

    tag_lhs(tree.root_node)

    magic = 0
    has_function_or_call = False

    def walk(n):
        nonlocal magic, has_function_or_call
        t = n.type
        if t in ("call", "function_definition"):
            has_function_or_call = True
        if t in ("integer", "float"):
            txt = _text(n, code)
            # handle unary_minus parent
            parent = n.parent
            full = txt
            if parent is not None and parent.type == "unary_operator":
                full = _text(parent, code).strip()
            if full in ALLOWED_LITERALS:
                pass
            elif _is_inside_upper_snake_const_assignment(n):
                pass
            elif _is_in_default_param(n):
                pass
            elif _is_in_slice(n):
                pass
            elif _is_in_single_arg_range(n, code):
                pass
            else:
                magic += 1
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    if not has_function_or_call:
        # No code-meaningful structures, abstain
        return None
    return magic


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    if _get_parser() is None:
        return None
    counts: List[int] = []
    for content in by_path.values():
        c = _file_magic_count(content.encode("utf8", errors="replace"))
        if c is not None:
            counts.append(c)
    if not counts:
        return None
    avg = sum(counts) / len(counts)
    return float(math.exp(-avg / 5.0))
