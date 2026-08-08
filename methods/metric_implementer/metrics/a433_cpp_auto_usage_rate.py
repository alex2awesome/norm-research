"""a433: C++ auto-keyword adoption.

Among local variable declarations, fraction whose type specifier is `auto`.
Higher = more modern type deduction usage.

Tier 2. THIN.
"""
from __future__ import annotations
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a433"
ASPECT_NAME = "C++ auto keyword adoption rate"
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
    n_auto = n_decl = 0
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for n in _walk_all(tree.root_node):
            if n.type == "declaration":
                # check first child for type
                txt = _text(n, src)
                if len(txt) > 200:
                    continue
                # look for auto specifier
                has_auto = False
                for c in n.children:
                    if c.type in ("placeholder_type_specifier", "auto"):
                        has_auto = True
                        break
                    # auto specifier appears as primitive_type "auto" too
                    if c.type == "primitive_type" and _text(c, src) == "auto":
                        has_auto = True
                        break
                n_decl += 1
                if has_auto:
                    n_auto += 1
    if n_decl == 0:
        return None
    return n_auto / n_decl
