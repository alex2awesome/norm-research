"""a446: C++ terse identifier density.

Walks tree-sitter-cpp AST of added C++ files and computes the fraction of
*declared* identifiers (variables, parameters, fields, type aliases, but
NOT C++ keywords and NOT STL type names) whose name is at most three
characters long.

LC-community style favours ``res``, ``cnt``, ``dp``, ``mp``, ``i`` while
industrial style favours ``result``, ``count``, ``adjacencyMap``. A high
ratio means the code is terse.

Returns NaN if no usable identifier is found.

Tier 2. THIN.
"""
from __future__ import annotations
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a446"
ASPECT_NAME = "C++ terse identifier density"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]

# Built-in / STL identifiers we never want to count.
STL_AND_BUILTINS = frozenset({
    # primitive types
    "int", "long", "short", "char", "float", "double", "bool", "void",
    "unsigned", "signed", "auto", "size_t", "ssize_t", "uint8_t",
    "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t",
    "int64_t",
    # STL types we expect in LC C++
    "string", "vector", "map", "set", "unordered_map", "unordered_set",
    "multiset", "multimap", "deque", "queue", "stack", "priority_queue",
    "list", "array", "pair", "tuple", "bitset", "function",
    "shared_ptr", "unique_ptr", "weak_ptr", "optional", "variant",
    "any", "iterator", "const_iterator", "reverse_iterator",
    # C++ keywords / qualifiers that can land in identifier slots
    "const", "static", "virtual", "override", "final", "inline",
    "explicit", "constexpr", "noexcept", "mutable", "volatile",
    "public", "private", "protected", "this", "self", "operator",
    "namespace", "std", "true", "false", "nullptr",
    # common LeetCode aliases
    "TreeNode", "ListNode", "Node",
})

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


# Node types whose direct `identifier` children are *declarations*.
DECL_PARENT_TYPES = {
    "init_declarator", "declarator", "parameter_declaration",
    "field_declaration", "alias_declaration",
}


def _collect_declared(root, src: bytes):
    names = []
    for n in _walk_all(root):
        if n.type == "identifier":
            # Climb to nearest meaningful parent.
            p = n.parent
            if p is None:
                continue
            if p.type in {"init_declarator", "parameter_declaration",
                          "field_declaration"}:
                names.append(_text(n, src))
            elif p.type == "pointer_declarator" or p.type == "reference_declarator":
                names.append(_text(n, src))
            elif p.type == "array_declarator":
                # only the leftmost identifier child counts as the name
                first_id = None
                for c in p.children:
                    if c.type == "identifier":
                        first_id = c
                        break
                if first_id is not None and first_id == n:
                    names.append(_text(n, src))
        elif n.type == "type_identifier":
            # Skip — we don't count type names here.
            continue
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
    n_short = n_total = 0
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for nm in _collect_declared(tree.root_node, src):
            if not nm or nm in STL_AND_BUILTINS:
                continue
            n_total += 1
            if len(nm) <= 3:
                n_short += 1
    if n_total == 0:
        return None
    return n_short / n_total
