"""a435: C++ lambda density.

Score = 1 - exp(-lambdas / max(funcs, 1)).

Reflects whether code uses lambdas at all (common in modern STL algorithms).

Tier 2. THIN.
"""
from __future__ import annotations
import math
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a435"
ASPECT_NAME = "C++ lambda density"
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
    n_lam = n_funcs = 0
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for n in _walk_all(tree.root_node):
            if n.type == "lambda_expression":
                n_lam += 1
            elif n.type == "function_definition":
                n_funcs += 1
    if n_funcs == 0:
        return None
    return 1.0 - math.exp(-(n_lam / n_funcs))
