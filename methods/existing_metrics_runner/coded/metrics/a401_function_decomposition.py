"""a401: Function decomposition — many small functions vs one giant one.

Per file we extract function definitions via tree-sitter (Python, JS, TS,
Java, Go) and compute:
  - n_fns: number of function/method declarations
  - lines per function (end_row - start_row + 1)
  - mean_len, max_len

Score per file (a "decomposition health" signal):
  - If n_fns < 1: not applicable (e.g. pure data file)
  - If n_fns == 1:
        single function => penalise long monolith.
        score = exp(-(len - 30)/40) clipped to [0,1] with len=30 -> 1.0
  - If n_fns >= 2:
        good if mean_len < 30 (small focused functions): score = 1.0
        decay as mean_len grows past 30.
        max-len kicker: if any function > 100 lines, subtract 0.2 from score.

Per-diff score: mean over per-file scores.

Why not subsume into a0 (CCN)? CCN measures branching INSIDE a function.
This measures whether the code was decomposed AT ALL. A 200-line linear
function has CCN=1 but is exactly what this metric flags.

Why not subsume into a232 / a80? a80 detects Fowler "Extract Method"
refactorings (deltas between pre/post). a232 detects OO class-level
refactorings. Neither measures the standing decomposition health of the
file at the snapshot we observe.

Examples:
  + Single 200-line function           -> ~0.0 (monolith)
  + Five 15-line functions             -> 1.0
  + Two 80-line functions              -> ~0.3
  + One 25-line function               -> 1.0
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a401"
ASPECT_NAME = "Function decomposition (many small vs one giant)"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# function-defining node types per language
FN_NODES = {
    "py": {"function_definition"},
    "js": {"function_declaration", "method_definition", "arrow_function",
           "function_expression"},
    "ts": {"function_declaration", "method_definition", "arrow_function",
           "function_expression", "function_signature"},
    "java": {"method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration"},
}

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "py":
            import tree_sitter_python as m; L = m.language()
        elif lang == "js":
            import tree_sitter_javascript as m; L = m.language()
        elif lang == "ts":
            import tree_sitter_typescript as m; L = m.language_typescript()
        elif lang == "java":
            import tree_sitter_java as m; L = m.language()
        elif lang == "go":
            import tree_sitter_go as m; L = m.language()
        else:
            return None
        _PARSERS[lang] = Parser(Language(L))
        return _PARSERS[lang]
    except ImportError:
        return None


def _collect_fn_lengths(code: bytes, lang: str) -> List[int]:
    parser = _get_parser(lang)
    if parser is None:
        return []
    try:
        tree = parser.parse(code)
    except Exception:
        return []
    targets = FN_NODES.get(lang, set())
    lengths: List[int] = []

    def walk(n):
        if n.type in targets:
            length = n.end_point[0] - n.start_point[0] + 1
            lengths.append(length)
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return lengths


def _file_score(lengths: List[int]) -> Optional[float]:
    if not lengths:
        return None
    n = len(lengths)
    mean_len = sum(lengths) / n
    max_len = max(lengths)
    if n == 1:
        # single function: penalise long monolith
        if mean_len <= 30:
            return 1.0
        return float(max(0.0, math.exp(-(mean_len - 30) / 40.0)))
    # multiple functions
    if mean_len <= 30:
        s = 1.0
    else:
        s = math.exp(-(mean_len - 30) / 40.0)
    if max_len > 100:
        s -= 0.2
    return float(max(0.0, min(1.0, s)))


def _path_lang(path: str) -> Optional[str]:
    p = path.lower()
    for ext, lang in EXT_TO_LANG.items():
        if p.endswith(ext):
            return lang
    return None


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    return any(_path_lang(p) is not None for p in by_path)


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    scs: List[float] = []
    for path, content in by_path.items():
        lang = _path_lang(path)
        if lang is None:
            continue
        lens = _collect_fn_lengths(
            content.encode("utf8", errors="replace"), lang)
        s = _file_score(lens)
        if s is not None:
            scs.append(s)
    if not scs:
        return None
    return float(sum(scs) / len(scs))
