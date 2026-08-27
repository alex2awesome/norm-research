"""a450: C++ C-style for-loop fraction.

For each for-loop in the added C++ files, classify as either:
  - C-style: ``for (init; cond; incr) { ... }`` (tree-sitter ``for_statement``)
  - range-based: ``for (T x : container) { ... }`` (tree-sitter ``for_range_loop``)

Returns the fraction that are C-style.

LC-community style favours the C-style index loop for index access /
manual indexing; industrial style favours range-based ``for (auto& x : v)``.

Returns NaN if no for-loop is present.

Tier 2. THIN.
"""
from __future__ import annotations
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a450"
ASPECT_NAME = "C++ C-style for-loop fraction"
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


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    parser = _get_parser()
    if parser is None:
        return None
    n_c = 0
    n_range = 0
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for n in _walk_all(tree.root_node):
            if n.type == "for_statement":
                n_c += 1
            elif n.type == "for_range_loop":
                n_range += 1
    total = n_c + n_range
    if total == 0:
        return None
    return n_c / total
