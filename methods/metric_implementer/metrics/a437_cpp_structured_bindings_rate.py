"""a437: C++ structured-binding adoption (C++17).

Returns the rate of `auto [a, b] = …` per declaration. 1 - exp(-rate*5).

Tier 2. THIN.
"""
from __future__ import annotations
import math
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a437"
ASPECT_NAME = "C++ structured bindings adoption"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]

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


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    parser = _get_parser()
    if parser is None:
        return None
    n_sb = n_decls = 0
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for n in _walk_all(tree.root_node):
            if n.type == "structured_binding_declarator":
                n_sb += 1
            elif n.type == "declaration":
                n_decls += 1
    if n_decls == 0:
        return None
    return 1.0 - math.exp(-(n_sb / n_decls) * 5.0)
