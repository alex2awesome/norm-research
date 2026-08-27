"""a425: C++ local-variable scoping.

Heuristic: tight-scope code declares variables inside the loop/branch where
they are used. Loose-scope code hoists declarations to function top.

We compute, per function: fraction of init_declarator nodes whose nearest
enclosing scope is the function body (depth 1) vs a nested block (depth>=2).

Score = mean over functions of (1 - depth1_fraction).

Tier 2. PARTIALLY_THIN — proxy, not direct lifetime analysis.
"""
from __future__ import annotations
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a425"
ASPECT_NAME = "C++ local-variable scope tightness"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "PARTIALLY_THIN"

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


def _function_scope_score(func_body, src: bytes) -> Optional[float]:
    if func_body is None or func_body.type != "compound_statement":
        return None
    decls = []  # (depth, node)
    def walk(node, depth):
        for c in node.children:
            if c.type == "init_declarator" or (c.type == "declaration"):
                decls.append((depth, c))
            if c.type == "compound_statement":
                walk(c, depth + 1)
            else:
                walk(c, depth)
    walk(func_body, 1)
    if not decls:
        return None
    n_deep = sum(1 for d, _ in decls if d >= 2)
    return n_deep / len(decls)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    parser = _get_parser()
    if parser is None:
        return None
    func_scores = []
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
                s = _function_scope_score(body, src)
                if s is not None:
                    func_scores.append(s)
    if not func_scores:
        return None
    return sum(func_scores) / len(func_scores)
