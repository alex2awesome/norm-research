"""a439: C++ reference-or-pointer vs value parameter passing for non-trivial types.

For parameters whose type spelling looks non-trivial (>= 5 chars, not in
trivial primitive set), what fraction are passed by reference / pointer?
Passing large objects by value is a perf smell.

Tier 2. PARTIALLY_THIN.
"""
from __future__ import annotations
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a439"
ASPECT_NAME = "C++ ref/ptr param vs value for non-trivial types"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "PARTIALLY_THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
TRIVIAL = {"int","long","short","char","bool","float","double",
           "size_t","ssize_t","int32_t","int64_t","uint32_t","uint64_t",
           "auto","void"}

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
    n_nontriv = n_ref = 0
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for n in _walk_all(tree.root_node):
            if n.type != "parameter_declaration":
                continue
            txt = _text(n, src)
            # crude type token = first token
            tok = txt.split()[0] if txt.split() else ""
            tok = tok.replace("const","").strip()
            if not tok:
                continue
            # is type non-trivial? Heuristic: has < or :: or length >= 6 and not in TRIVIAL
            nontriv = ("<" in txt) or ("::" in txt) or                        (tok not in TRIVIAL and len(tok) >= 6)
            if not nontriv:
                continue
            n_nontriv += 1
            if "&" in txt or "*" in txt:
                n_ref += 1
    if n_nontriv == 0:
        return None
    return n_ref / n_nontriv
