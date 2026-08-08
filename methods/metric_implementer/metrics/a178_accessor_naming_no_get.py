"""a178: Accessor naming without `get_` prefix.

In Python and Go, idiomatic style names accessors after the *property*, not
with a `get_` prefix (PEP 8 / Effective Go). Java is the opposite (the
JavaBeans convention requires `getX()`), so we exclude Java entirely.

Detection (Python / Go only):
  - Walk the source via tree-sitter to find function/method declarations
    whose body is a *single* `return` statement reading a field on the
    receiver/self (the classic accessor shape):
        def foo(self): return self._foo
        func (r *T) Foo() int { return r.foo }
  - Count an accessor as conforming iff its name does NOT start with `get_`
    (Python) or `Get` (Go — exported accessor named `GetFoo` is the same
    anti-pattern in Effective Go).
  - Documented exception: `__get__`/`__getitem__`/`__getattr__` etc. are
    Python dunder protocol hooks, not user-facing accessors — exempt.

Score per file = conforming_accessors / total_accessors. Overall score =
mean across files that contained at least one accessor. ABSTAINS when no
accessor-shaped method was observed in the added code.

Narrow-applicability, PARTIALLY_THIN: the *naming* check is mechanical, but
the *shape* check (one-line `return self.x`) is a heuristic for "is this
really an accessor". A method named `get_total` that does real arithmetic
is correctly NOT flagged.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a178"
ASPECT_NAME = "Accessor naming without get_ prefix"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".go": "go",
}

# Dunder-protocol "get" names are exempt — they're descriptor / mapping hooks,
# not user-facing accessors. Naming is fixed by the language.
PY_DUNDER_EXEMPT = frozenset({
    "__get__", "__getitem__", "__getattr__", "__getattribute__",
    "__getstate__", "__getnewargs__", "__getnewargs_ex__",
})

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "py":
            import tree_sitter_python as m
            L = m.language()
        elif lang == "go":
            import tree_sitter_go as m
            L = m.language()
        else:
            return None
        _PARSERS[lang] = Parser(Language(L))
        return _PARSERS[lang]
    except ImportError:
        return None


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# ----- Python ----------------------------------------------------------------

def _py_first_child(node, types):
    for c in node.children:
        if c.type in types:
            return c
    return None


def _py_is_self_attribute_return(body_node, src: bytes,
                                 self_name: str) -> bool:
    """Is the body a single `return self.x` (or `return x` after self)?

    body_node is a `block` containing exactly one statement which must be
    `return_statement` whose return expression is an `attribute` whose object
    is the receiver name (`self` / `cls` / typical Python convention).
    """
    # block's children include indent/dedent/newline noise — pick statements
    stmts = [c for c in body_node.children if c.type not in (
        "comment", "string", "\n")]
    # `string` at start of block = docstring; tolerate one docstring + return.
    real_stmts = []
    for c in body_node.children:
        if c.type in ("comment",):
            continue
        if c.type == "expression_statement":
            # could be a bare docstring
            only_str = (len(c.children) == 1
                        and c.children[0].type == "string")
            if only_str:
                continue
        real_stmts.append(c)
    if len(real_stmts) != 1:
        return False
    stmt = real_stmts[0]
    if stmt.type != "return_statement":
        return False
    # find return expression
    expr = None
    for c in stmt.children:
        if c.type not in ("return", "comment"):
            expr = c
            break
    if expr is None or expr.type != "attribute":
        return False
    # attribute = object "." attribute_name
    obj = expr.children[0] if expr.children else None
    if obj is None or obj.type != "identifier":
        return False
    return _text(obj, src) == self_name


def _py_walk(root, src: bytes) -> List[Tuple[str, bool]]:
    """Return [(name, is_get_prefixed)] for each accessor-shaped method
    found at any class scope.
    """
    out: List[Tuple[str, bool]] = []

    def walk(node, in_class: bool, self_name: Optional[str]):
        t = node.type
        if t == "class_definition":
            for c in node.children:
                walk(c, True, "self")
            return
        if t == "function_definition" and in_class:
            name_node = _py_first_child(node, ("identifier",))
            params_node = _py_first_child(node, ("parameters",))
            body_node = _py_first_child(node, ("block",))
            if name_node is not None and params_node is not None \
                    and body_node is not None:
                # Receiver name = first parameter (typically self/cls).
                first_param = None
                for c in params_node.children:
                    if c.type == "identifier":
                        first_param = _text(c, src)
                        break
                    if c.type == "typed_parameter":
                        for cc in c.children:
                            if cc.type == "identifier":
                                first_param = _text(cc, src)
                                break
                        if first_param:
                            break
                name = _text(name_node, src)
                if (first_param is not None
                        and name not in PY_DUNDER_EXEMPT
                        and _py_is_self_attribute_return(
                            body_node, src, first_param)):
                    out.append((name, name.startswith("get_")))
            # don't descend into nested functions
            return
        for c in node.children:
            walk(c, in_class, self_name)

    walk(root, in_class=False, self_name=None)
    return out


# ----- Go --------------------------------------------------------------------

def _go_method_receiver_name(method_node, src: bytes) -> Optional[str]:
    """Pull the receiver variable name from a method_declaration."""
    for c in method_node.children:
        if c.type == "parameter_list":
            # receiver parameter is the first parameter_list before the method name
            for fp in c.children:
                if fp.type == "parameter_declaration":
                    for cc in fp.children:
                        if cc.type == "identifier":
                            return _text(cc, src)
            return None
    return None


def _go_is_single_field_return(body_node, src: bytes,
                               recv_name: str) -> bool:
    """True iff body is a single `return r.x` (or `return r.x, nil` style is
    NOT a simple accessor and is rejected).
    """
    # block contains "{" statements "}"
    stmts = [c for c in body_node.children
             if c.type not in ("{", "}", "comment")]
    if len(stmts) != 1:
        return False
    stmt = stmts[0]
    if stmt.type != "return_statement":
        return False
    # children: "return", expression_list
    expr_list = None
    for c in stmt.children:
        if c.type == "expression_list":
            expr_list = c
            break
    if expr_list is None:
        return False
    # exactly one expression, and it's a selector_expression with operand
    # equal to the receiver name
    exprs = [c for c in expr_list.children if c.type != ","]
    if len(exprs) != 1:
        return False
    e = exprs[0]
    if e.type != "selector_expression":
        return False
    operand = e.children[0] if e.children else None
    if operand is None or operand.type != "identifier":
        return False
    return _text(operand, src) == recv_name


def _go_walk(root, src: bytes) -> List[Tuple[str, bool]]:
    out: List[Tuple[str, bool]] = []

    def walk(node):
        t = node.type
        if t == "method_declaration":
            # method_declaration = "func" receiver name parameters [result] body
            recv_name = _go_method_receiver_name(node, src)
            # find method name (field_identifier) and body (block)
            name = None
            body = None
            for c in node.children:
                if c.type == "field_identifier" and name is None:
                    name = _text(c, src)
                elif c.type == "block":
                    body = c
            if (recv_name is not None and name is not None
                    and body is not None
                    and _go_is_single_field_return(body, src, recv_name)):
                # Go anti-pattern is `GetX` (exported accessor) — Effective Go
                # says the getter should just be `X`. Lower-case `getX` is
                # also non-idiomatic but rare; we flag both.
                is_get = (name.startswith("Get")
                          and len(name) > 3
                          and name[3].isupper()) or (
                    name.startswith("get")
                    and len(name) > 3
                    and name[3].isupper())
                out.append((name, is_get))
        for c in node.children:
            walk(c)

    walk(root)
    return out


# ----- Dispatch --------------------------------------------------------------

def _file_pair(code: bytes, lang: str) -> Optional[Tuple[int, int]]:
    """Return (conforming_count, total_accessors) for one source file, or
    None if the parser is unavailable.
    """
    parser = _get_parser(lang)
    if parser is None:
        return None
    tree = parser.parse(code)
    root = tree.root_node
    if lang == "py":
        items = _py_walk(root, code)
    elif lang == "go":
        items = _go_walk(root, code)
    else:
        return None
    if not items:
        return (0, 0)
    total = len(items)
    conforming = sum(1 for _, is_get in items if not is_get)
    return (conforming, total)


def _path_lang(path: str) -> Optional[str]:
    p = path.lower()
    for ext, lang in EXT_TO_LANG.items():
        if p.endswith(ext):
            return lang
    return None


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    return any(_path_lang(p) is not None for p in by_path)


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    file_scores: List[float] = []
    for path, content in by_path.items():
        lang = _path_lang(path)
        if lang is None:
            continue
        res = _file_pair(content.encode("utf8", errors="replace"), lang)
        if res is None:
            continue
        conforming, total = res
        if total == 0:
            continue
        file_scores.append(conforming / total)
    if not file_scores:
        return None
    return float(sum(file_scores) / len(file_scores))
