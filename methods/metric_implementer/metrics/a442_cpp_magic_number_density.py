"""a442: C++ magic-number density.

Counts integer_literal / number_literal nodes that are not in the trivial
set {0,1,-1,2}. Normalizes by total declarations. Lower density = better
named constants. Returns exp(-density*5).

Tier 2. PARTIALLY_THIN.
"""
from __future__ import annotations
import math
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a442"
ASPECT_NAME = "C++ magic-number density"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "PARTIALLY_THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
TRIVIAL = {"0","1","-1","2"}

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
    n_magic = n_lines = 0
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        n_lines += max(1, content.count("\n"))
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for n in _walk_all(tree.root_node):
            if n.type == "number_literal":
                tok = _text(n, src).strip()
                if tok not in TRIVIAL and not tok.endswith("ULL") and                         tok not in ("0x0","0x1"):
                    n_magic += 1
    if n_lines == 0:
        return None
    return math.exp(-(n_magic / n_lines) * 5.0)
