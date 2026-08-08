"""a423: C++ commenting strategy.

Counts comment tokens via tree-sitter and computes:
  - comment_density = comment_lines / total_lines
  - per-comment mean alphabetic-word count

Score = sigmoid(2 * (density + 0.5 * mean_word_count_scale - 0.1)).

Penalizes both no-comments and 1-word junk comments ("done", "todo", "tmp").

Tier 2. PARTIALLY_THIN.
"""
from __future__ import annotations
import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a423"
ASPECT_NAME = "C++ inline comment quality"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "PARTIALLY_THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
# REGEX_OK: tool_output — word tokens inside comment strings (not source).
_WORD = re.compile(r"[A-Za-z]{2,}")

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
    total_lines = 0
    cmt_lines = 0
    word_counts = []
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        nlines = max(1, content.count("\n") + 1)
        total_lines += nlines
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for n in _walk_all(tree.root_node):
            if n.type == "comment":
                txt = _text(n, src)
                cmt_lines += txt.count("\n") + 1
                wc = len(_WORD.findall(txt))
                word_counts.append(wc)
    if total_lines == 0:
        return None
    density = cmt_lines / total_lines
    mean_wc = (sum(word_counts) / len(word_counts)) if word_counts else 0.0
    # Compress mean_wc into roughly [0,1] using min(1, mean_wc/8).
    wc_scale = min(1.0, mean_wc / 8.0)
    raw = density + 0.5 * wc_scale - 0.1
    return 1.0 / (1.0 + math.exp(-2.0 * raw))
