"""a422: C++ intention-revealing naming.

Penalizes uses of generic placeholder names (tmp, temp, data, foo, bar, baz,
val, val1, val2, ret, res, retval, dummy, ans, num, arr, lst, mp, st, ll, pq)
for declared variables and parameters. These reveal nothing about purpose.

Score = 1 - (#generic / #declared_vars). Abstain if no declared vars.

Tier 2. PARTIALLY_THIN.
"""
from __future__ import annotations
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a422"
ASPECT_NAME = "C++ intention-revealing naming"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "PARTIALLY_THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
GENERIC = frozenset({"tmp", "temp", "data", "foo", "bar", "baz",
                     "val", "val1", "val2", "val3",
                     "ret", "res", "result", "retval",
                     "dummy", "ans", "answer",
                     "num", "nums",
                     "arr", "array", "lst",
                     "mp", "st", "ll", "pq", "vec",
                     "a", "b", "c", "p", "q", "t",
                     "x1", "x2", "y1", "y2",
                     "obj", "thing", "stuff", "var", "value"})

_PARSER = None

def _get_parser():
    global _PARSER
    if _PARSER is None:
        try:
            import tree_sitter_cpp
            from tree_sitter import Language, Parser
            _PARSER = Parser(Language(tree_sitter_cpp.language()))
        except Exception:
            return None
    return _PARSER

def _walk_all(node):
    yield node
    for c in node.children:
        yield from _walk_all(c)

def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _collect_var_names(root, src: bytes):
    out = []
    for n in _walk_all(root):
        t = n.type
        if t in ("init_declarator", "declarator", "parameter_declaration"):
            for c in _walk_all(n):
                if c.type == "identifier":
                    nm = _text(c, src)
                    out.append(nm)
                    break
    return out


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    parser = _get_parser()
    if parser is None:
        return None
    n_total = n_generic = 0
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for nm in _collect_var_names(tree.root_node, src):
            n_total += 1
            if nm.lower() in GENERIC or (len(nm) == 1 and nm not in ("i","j","k","n","m")):
                n_generic += 1
    if n_total == 0:
        return None
    return 1.0 - (n_generic / n_total)
