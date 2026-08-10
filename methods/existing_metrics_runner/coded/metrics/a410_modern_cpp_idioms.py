"""a410: Modern C++ idioms density (C++11/14/17 surface usage).

We walk the tree-sitter-cpp parse of added C++ lines and count occurrences of:

  positive (modern):
    - `auto` as type specifier (placeholder_type_specifier)
    - range-based for (for_range_loop)
    - lambda expressions (lambda_expression)
    - structured bindings (structured_binding_declarator) — C++17
    - using-aliases (alias_declaration)

  negative (legacy):
    - `typedef` (type_definition)
    - C-style counted for-loops (for_statement with int init + cond + update)

Score = positive_density / (positive_density + negative_density + epsilon).

Classification THIN: structural — what counts as `auto` is unambiguous.
Tier 2: pure tree-sitter, no subprocess.

We do NOT measure correctness of the modern construct (e.g. whether using
`auto` *here* hurts readability) — only frequency density of modern over
legacy idioms in the added lines. Conformance interpretation: a project
that fully adopted C++11/17 should score high; one that still writes
`typedef int U32;` and bare `for (int i…; i<…; i++)` scores low.
"""
from __future__ import annotations

from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a410"
ASPECT_NAME = "Modern C++ idioms density"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh"]

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        try:
            import tree_sitter_cpp
            from tree_sitter import Language, Parser
            _PARSER = Parser(Language(tree_sitter_cpp.language()))
        except ImportError:
            return None
    return _PARSER


def _count_idioms(source: bytes):
    parser = _get_parser()
    if parser is None:
        return None
    tree = parser.parse(source)
    pos = 0
    neg = 0

    def walk(node):
        nonlocal pos, neg
        t = node.type
        # positive — modern
        if t == "placeholder_type_specifier":
            # `auto` keyword usage
            if any(c.type == "auto" for c in node.children):
                pos += 1
        elif t == "for_range_loop":
            pos += 1
        elif t == "lambda_expression":
            pos += 1
        elif t == "structured_binding_declarator":
            pos += 1
        elif t == "alias_declaration":
            pos += 1
        # negative — legacy
        elif t == "type_definition":
            # `typedef` form (tree-sitter emits type_definition for typedef)
            if any(c.type == "typedef" for c in node.children):
                neg += 1
        elif t == "for_statement":
            # treat C-style for as negative regardless of body
            neg += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return pos, neg


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    total_pos = 0
    total_neg = 0
    any_seen = False
    for content in by_path.values():
        res = _count_idioms(content.encode("utf8", errors="replace"))
        if res is None:
            return None
        p, n = res
        total_pos += p
        total_neg += n
        if p + n > 0:
            any_seen = True
    if not any_seen:
        return None
    # epsilon=1 in denominator avoids div-by-zero when only one side seen.
    return float(total_pos / (total_pos + total_neg + 1))
