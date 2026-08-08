"""a421: C++ identifier expressiveness.

Per declared identifier, compute:
  - length in chars
  - Shannon entropy over chars
  - token count (split on _ / camel boundaries)

Score = mean over identifiers of:
   clamp((entropy>=1.5) + (len>=3) + (tokens>=2), 0, 3) / 3.

Penalizes single-letter or low-entropy names like `tmp`, `x`, `a`.

Tier 2. PARTIALLY_THIN.
"""
from __future__ import annotations
import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a421"
ASPECT_NAME = "C++ identifier expressiveness"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "PARTIALLY_THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
EXEMPT = frozenset({"i", "j", "k", "n", "m", "x", "y", "z", "_",
                    "self", "this"})

# REGEX_OK: tool_output — splits identifier strings.
_CAMEL_BREAK = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

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


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    n = len(s)
    cnt = Counter(s.lower())
    return -sum((c/n) * math.log2(c/n) for c in cnt.values())


def _tokens(name: str) -> int:
    parts = []
    for chunk in name.split("_"):
        parts.extend(_CAMEL_BREAK.split(chunk))
    return sum(1 for p in parts if p)


def _ident_score(name: str) -> Optional[float]:
    if not name or name in EXEMPT or name.startswith("__"):
        return None
    pts = 0
    if _entropy(name) >= 1.5:
        pts += 1
    if len(name) >= 3:
        pts += 1
    if _tokens(name) >= 2:
        pts += 1
    return pts / 3.0


def _collect_names(root, src: bytes):
    names = []
    for n in _walk_all(root):
        if n.type in ("identifier", "field_identifier", "type_identifier"):
            # filter to declarations only by checking parent context
            parent = n.parent
            if parent is None:
                continue
            pt = parent.type
            if pt in ("function_declarator", "init_declarator",
                      "field_declaration", "parameter_declaration",
                      "class_specifier", "struct_specifier",
                      "type_definition", "alias_declaration",
                      "enumerator"):
                names.append(_text(n, src))
    return names


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    parser = _get_parser()
    if parser is None:
        return None
    scores = []
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for nm in _collect_names(tree.root_node, src):
            s = _ident_score(nm)
            if s is not None:
                scores.append(s)
    if not scores:
        return None
    return sum(scores) / len(scores)
