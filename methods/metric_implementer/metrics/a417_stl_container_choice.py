"""a417: STL container choice idiomaticity.

We walk the added C++ tree-sitter parse and tally container-typed
declarations (variables, parameters, fields, return types). A container is
"idiomatic" or "anti-idiomatic" by a small lookup:

  IDIOMATIC (positive):
    - std::vector       (sequence, default choice)
    - std::array        (fixed-size sequence)
    - std::unordered_map / unordered_set   (hash lookup)
    - std::map / std::set                  (sorted ordered lookup)
    - std::string                          (text)
    - std::span         (C++20 view)
    - std::optional                        (nullable single value)

  ANTI-PATTERN (negative):
    - C-style array (array_declarator with non-string element type) where a
      std::array/std::vector would do — except when the size is small (≤4),
      we forgive it (`int rgba[4]` is fine).
    - std::list  (rarely appropriate; default-bad)
    - std::deque (often premature; we leave NEUTRAL — don't count)

Score = idiomatic / (idiomatic + anti). If no containers at all observed,
applies but returns None (no signal).

Tier 2, CLASSIFICATION PARTIALLY_THIN (some judgment calls — when is
std::list actually right? — but the typical case is unambiguous).
"""
from __future__ import annotations

from typing import Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a417"
ASPECT_NAME = "C++ STL container choice idiomaticity"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "PARTIALLY_THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]

IDIOMATIC_CONTAINERS = {
    "vector", "array", "unordered_map", "unordered_set",
    "map", "set", "string", "span", "optional", "string_view",
    "variant", "tuple", "pair",
}

ANTI_CONTAINERS = {"list", "forward_list"}

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


def _text(n) -> str:
    return n.text.decode("utf8", errors="replace")


def _template_base_name(template_type_node) -> Optional[str]:
    """Return the type_identifier child of a template_type, e.g.
    `vector<int>` → 'vector'."""
    for c in template_type_node.children:
        if c.type == "type_identifier":
            return _text(c)
    return None


def _qid_template_name(qid_node) -> Optional[str]:
    """qualified_identifier like std::vector<int> → return 'vector'."""
    for c in qid_node.children:
        if c.type == "template_type":
            return _template_base_name(c)
        if c.type == "qualified_identifier":
            nm = _qid_template_name(c)
            if nm is not None:
                return nm
    return None


def _count(source: bytes) -> Optional[Tuple[int, int]]:
    parser = _get_parser()
    if parser is None:
        return None
    tree = parser.parse(source)
    idiomatic = 0
    anti = 0

    def walk(node):
        nonlocal idiomatic, anti
        t = node.type
        if t == "template_type":
            base = _template_base_name(node)
            if base in IDIOMATIC_CONTAINERS:
                idiomatic += 1
            elif base in ANTI_CONTAINERS:
                anti += 1
        elif t == "qualified_identifier":
            base = _qid_template_name(node)
            if base in IDIOMATIC_CONTAINERS:
                idiomatic += 1
                # Don't recurse to avoid double-counting the inner template_type
                return
            elif base in ANTI_CONTAINERS:
                anti += 1
                return
        elif t == "type_identifier":
            # bare `string` or `vector` (after `using namespace std`)
            name = _text(node)
            # only count if direct, not part of larger template_type
            if name == "string":
                # Could be std::string after using namespace std — treat as
                # idiomatic
                idiomatic += 1
        elif t == "array_declarator":
            # C-style array. Try to find the size literal. If small (<=4) or
            # if the element type is `char` (likely a string buffer used for
            # interop) skip; otherwise count as anti-pattern.
            size = None
            for c in node.children:
                if c.type == "number_literal":
                    try:
                        size = int(_text(c))
                    except ValueError:
                        pass
            # Find ancestor's element type by walking up via .parent
            elem_type = None
            par = node.parent
            if par is not None:
                for c in par.children:
                    if c.type == "primitive_type":
                        elem_type = _text(c)
                        break
            if elem_type == "char":
                pass  # interop buffer, skip
            elif size is not None and size <= 4:
                pass  # small fixed-size like rgba[4]
            else:
                anti += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return idiomatic, anti


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    total_i = 0
    total_a = 0
    for content in by_path.values():
        res = _count(content.encode("utf8", errors="replace"))
        if res is None:
            return None
        i, a = res
        total_i += i
        total_a += a
    if total_i + total_a == 0:
        return None
    return float(total_i / (total_i + total_a))
