"""a518: AST node-type sequence entropy via tree-sitter.

Goal: "How structurally distinctive is this code?" — univariate property
of CANDIDATE CODE ALONE, orthogonal to surface style (whitespace, names,
comments).

Method:
  - Parse the added candidate file(s) with tree-sitter Python or C++.
  - Walk the tree, collecting only AST node TYPES (no identifiers, no
    literals, no whitespace).
  - Compute Shannon entropy (base e) of the empirical distribution over
    node types, normalized by log(N_types_observed) for length-invariance.
  - Higher entropy = more structurally varied code. Lower = more templated
    / repetitive (typical competition stubs with a loop and a print).

If tree-sitter bindings (`tree_sitter_python`, `tree_sitter_cpp`) are not
available at import time, falls back to a regex-based proxy: counts a fixed
set of structural keyword categories and reports the entropy of that
distribution. The fallback is documented but materially weaker.

applies() True iff the candidate has >= 30 non-whitespace lines (per spec).

CLASSIFICATION: THIN. Univariate function of CANDIDATE CODE ALONE.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a518"
ASPECT_NAME = "AST node-type sequence entropy"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

# Try to load tree-sitter; on failure, regex fallback is used.
_TS_OK = False
_PY_PARSER = None
_CPP_PARSER = None
try:  # pragma: no cover - environment dependent
    import tree_sitter_python as _tspy
    import tree_sitter_cpp as _tscpp
    from tree_sitter import Language as _Language, Parser as _Parser
    _PY_LANG = _Language(_tspy.language())
    _CPP_LANG = _Language(_tscpp.language())
    _PY_PARSER = _Parser(_PY_LANG)
    _CPP_PARSER = _Parser(_CPP_LANG)
    _TS_OK = True
except Exception:
    _TS_OK = False


# Regex fallback: structural-keyword class counts
_FALLBACK_CLASSES = {
    "decl": re.compile(
        r"\b(def|class|struct|template|typedef|namespace|enum|using|"
        r"int|long|short|char|void|bool|float|double|auto|static|extern)\b"),
    "ctrl": re.compile(
        r"\b(for|while|if|elif|else|switch|case|return|break|continue|"
        r"goto|try|catch|except|finally|yield|throw|raise|do)\b"),
    "call": re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\("),
    "binop": re.compile(r"==|!=|<=|>=|&&|\|\||<<|>>|->|::|[\+\-\*/%<>]"),
    "assign": re.compile(r"(?<![=!<>])=(?!=)"),
    "brace": re.compile(r"[\{\}]"),
    "bracket": re.compile(r"[\[\]]"),
    "paren": re.compile(r"[\(\)]"),
    "semi": re.compile(r";"),
    "str": re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'"),
    "num": re.compile(r"\b\d+(?:\.\d+)?\b"),
}


def _nonblank_line_count(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip())


def applies(diff_text: str) -> bool:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return False
    total = 0
    for content in by_path.values():
        total += _nonblank_line_count(content)
        if total >= 30:
            return True
    return False


def _collect_node_types(node) -> list:
    """Iterative DFS collecting only NAMED node types (skip anonymous/punct)."""
    out = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n.is_named:
            out.append(n.type)
        # children appended in reverse so traversal is left-to-right
        for c in reversed(n.children):
            stack.append(c)
    return out


def _entropy_natural(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h


def _score_treesitter(by_path):
    seqs = []
    for path, content in by_path.items():
        is_py = path.lower().endswith((".py", ".pyi"))
        parser = _PY_PARSER if is_py else _CPP_PARSER
        try:
            tree = parser.parse(content.encode("utf-8", errors="ignore"))
        except Exception:
            continue
        seqs.extend(_collect_node_types(tree.root_node))
    if not seqs:
        return None
    cnt = Counter(seqs)
    h = _entropy_natural(cnt)
    # length-normalize by log(N distinct types observed) so the score
    # represents "how evenly the node-type budget is spread"
    k = len(cnt)
    if k <= 1:
        return 0.0
    return float(max(0.0, min(1.0, h / math.log(k))))


def _score_fallback(by_path):
    counts = Counter()
    for content in by_path.values():
        for cls_name, pat in _FALLBACK_CLASSES.items():
            counts[cls_name] += len(pat.findall(content))
    if sum(counts.values()) == 0:
        return None
    h = _entropy_natural(counts)
    k = sum(1 for v in counts.values() if v > 0)
    if k <= 1:
        return 0.0
    return float(max(0.0, min(1.0, h / math.log(k))))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    if _TS_OK:
        return _score_treesitter(by_path)
    return _score_fallback(by_path)
