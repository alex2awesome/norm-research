"""a424: C++ function decomposition.

Score = 1 - clamp(mean_function_body_lines / 50, 0, 1).

A code unit that decomposes work into multiple small functions has small
mean body length; a single huge function pushes the score toward 0.

Tier 2. THIN.
"""
from __future__ import annotations
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a424"
ASPECT_NAME = "C++ function decomposition"
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
    lens = []
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for n in _walk_all(tree.root_node):
            if n.type == "function_definition":
                body = None
                for c in n.children:
                    if c.type == "compound_statement":
                        body = c
                        break
                if body is None:
                    continue
                txt = _text(body, src)
                lens.append(max(1, txt.count("\n")))
    if not lens:
        return None
    mean_len = sum(lens) / len(lens)
    val = 1.0 - min(1.0, mean_len / 50.0)
    return float(val)
