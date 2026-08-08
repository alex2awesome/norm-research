"""a412: Smart pointers vs raw memory management.

We walk the tree-sitter-cpp AST of added C++ lines and count:

  positive (modern memory management):
    - `std::unique_ptr`, `std::shared_ptr`, `std::weak_ptr` references
      (as qualified_identifier or template_type in type position)
    - `std::make_unique`, `std::make_shared` calls (call_expression with
      these names)

  negative (manual memory management):
    - `new` expressions (new_expression)
    - `delete` expressions (delete_expression)
    - calls to `malloc`, `calloc`, `realloc`, `free` (call_expression)

Score = smart_count / (smart_count + raw_count + 1).
  - all-smart, no raw           → smart_count / (smart_count + 1) ≈ 1.0 for large N
  - 1 smart, 0 raw              → 0.5
  - 0 smart, 0 raw              → applies()=True but score=None (no signal)
  - 0 smart, many raw           → 0.0

Tier 2 (tree-sitter only), CLASSIFICATION THIN.
"""
from __future__ import annotations

from typing import Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a412"
ASPECT_NAME = "C++ smart pointers vs raw memory"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]

SMART_TYPE_NAMES = {"unique_ptr", "shared_ptr", "weak_ptr"}
SMART_FACTORIES = {"make_unique", "make_shared", "allocate_shared"}
RAW_ALLOC_FUNCS = {"malloc", "calloc", "realloc", "free", "aligned_alloc"}

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


def _call_target_name(call_node) -> Optional[str]:
    """Return the called function's bare name (last component of qualified name)."""
    if not call_node.children:
        return None
    head = call_node.children[0]
    if head.type == "identifier":
        return _text(head)
    if head.type == "qualified_identifier":
        # last name component
        last = None
        for c in head.children:
            if c.type in ("identifier", "template_function", "field_identifier"):
                last = c
        if last is not None:
            if last.type == "template_function":
                # name<...> — first child is identifier
                for c in last.children:
                    if c.type == "identifier":
                        return _text(c)
            return _text(last)
    if head.type == "template_function":
        for c in head.children:
            if c.type == "identifier":
                return _text(c)
    if head.type == "field_expression":
        # obj.method() — return method name
        for c in head.children:
            if c.type == "field_identifier":
                return _text(c)
    return None


def _is_smart_type_usage(node) -> bool:
    """Detect qualified_identifier / template_type whose base is a smart ptr."""
    t = node.type
    if t == "template_type":
        for c in node.children:
            if c.type == "type_identifier" and _text(c) in SMART_TYPE_NAMES:
                return True
    elif t == "qualified_identifier":
        # std::unique_ptr<T> — qualified_identifier wraps namespace + template_type
        for c in node.children:
            if c.type == "template_type":
                for cc in c.children:
                    if cc.type == "type_identifier" and _text(cc) in SMART_TYPE_NAMES:
                        return True
    return False


def _count(source: bytes) -> Optional[Tuple[int, int]]:
    parser = _get_parser()
    if parser is None:
        return None
    tree = parser.parse(source)
    smart = 0
    raw = 0

    def walk(node):
        nonlocal smart, raw
        t = node.type
        if t in ("template_type", "qualified_identifier"):
            if _is_smart_type_usage(node):
                smart += 1
                # don't recurse into the template args looking for further
                # type matches — they don't double-count the same site
        if t == "call_expression":
            name = _call_target_name(node)
            if name is not None:
                if name in SMART_FACTORIES:
                    smart += 1
                elif name in RAW_ALLOC_FUNCS:
                    raw += 1
        if t == "new_expression":
            raw += 1
        if t == "delete_expression":
            raw += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return smart, raw


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    total_smart = 0
    total_raw = 0
    for content in by_path.values():
        res = _count(content.encode("utf8", errors="replace"))
        if res is None:
            return None
        s, r = res
        total_smart += s
        total_raw += r
    if total_smart + total_raw == 0:
        # No memory-management signal at all in this diff.
        return None
    return float(total_smart / (total_smart + total_raw + 1))
