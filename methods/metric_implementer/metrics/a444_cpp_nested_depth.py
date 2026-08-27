"""a444: C++ control-flow nesting depth.

For each function, compute max nesting depth of compound_statement /
if_statement / for / while. Score = mean(max_depth) per file, mapped via
exp(-mean_depth/5).

Tier 2. THIN.
"""
from __future__ import annotations
import math
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a444"
ASPECT_NAME = "C++ control-flow nesting depth"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
NESTERS = {"compound_statement","if_statement","for_statement",
           "for_range_loop","while_statement","do_statement",
           "switch_statement","try_statement"}

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


def _max_depth(node, depth=0):
    md = depth
    for c in node.children:
        nd = depth + (1 if c.type in NESTERS else 0)
        sub = _max_depth(c, nd)
        if sub > md:
            md = sub
    return md


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    parser = _get_parser()
    if parser is None:
        return None
    depths = []
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for n in _walk_all(tree.root_node):
            if n.type == "function_definition":
                depths.append(_max_depth(n))
    if not depths:
        return None
    mean_d = sum(depths) / len(depths)
    return math.exp(-mean_d / 5.0)
