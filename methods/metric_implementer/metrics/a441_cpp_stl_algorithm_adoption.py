"""a441: C++ STL algorithm adoption (header <algorithm> usage proxy).

Counts call_expression nodes whose callee spelling references known STL
algorithms (std::sort, std::find, std::accumulate, std::transform,
std::for_each, std::count, std::any_of, std::all_of, std::none_of,
std::partition, std::unique, std::reverse, std::rotate, std::lower_bound,
std::upper_bound, std::binary_search, std::min/max/min_element/max_element).

Divides by total number of loops + algorithm calls; high ratio = idiomatic
STL.

Tier 2. THIN.
"""
from __future__ import annotations
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a441"
ASPECT_NAME = "C++ STL algorithm adoption rate"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
ALGO_NAMES = {"sort","stable_sort","find","find_if","accumulate","transform",
              "for_each","count","count_if","any_of","all_of","none_of",
              "partition","unique","reverse","rotate","lower_bound",
              "upper_bound","binary_search","min","max","min_element",
              "max_element","copy","copy_if","fill","iota","generate",
              "remove","remove_if","replace","merge","includes","equal",
              "lexicographical_compare","next_permutation",
              "prev_permutation","nth_element"}

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
    n_algo = n_loops = 0
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for n in _walk_all(tree.root_node):
            if n.type in ("for_statement","for_range_loop","while_statement"):
                n_loops += 1
            elif n.type == "call_expression":
                # callee is first child
                if n.children:
                    callee_txt = _text(n.children[0], src)
                    leaf = callee_txt.split("::")[-1].split("(")[0].strip()
                    if leaf in ALGO_NAMES:
                        n_algo += 1
    denom = n_algo + n_loops
    if denom == 0:
        return None
    return n_algo / denom
