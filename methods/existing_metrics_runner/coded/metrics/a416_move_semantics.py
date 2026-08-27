"""a416: C++ move semantics awareness.

We count move-semantics-aware constructs in added C++ code via tree-sitter:

  positive signals:
    - calls to `std::move` / `std::forward`
    - rvalue-reference parameter declarators (`&&`) on user-defined function
      parameters (excluding template `T&&` in templated contexts where it is
      forwarding-reference — still counts as awareness)
    - `noexcept` specifiers on function definitions (important for move
      constructors to participate in standard-library optimizations)

  reference point (denominator): number of function definitions in the added
  code (proxy for "opportunities to be move-aware") + 1.

Score = move_signals / (function_defs + 1), clamped to [0, 1].

This is a *density* score, not a correctness check. A high score means the
author wrote move-aware code; a low score means functions and returns are
present but no move-related construct appears.

Tier 2, CLASSIFICATION THIN.
"""
from __future__ import annotations

from typing import Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a416"
ASPECT_NAME = "C++ move semantics density"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h"]

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
    if not call_node.children:
        return None
    head = call_node.children[0]
    if head.type == "identifier":
        return _text(head)
    if head.type == "qualified_identifier":
        last = None
        for c in head.children:
            if c.type in ("identifier", "template_function", "field_identifier"):
                last = c
        if last is not None:
            if last.type == "template_function":
                for c in last.children:
                    if c.type == "identifier":
                        return _text(c)
            return _text(last)
    if head.type == "template_function":
        for c in head.children:
            if c.type == "identifier":
                return _text(c)
    return None


def _count(source: bytes) -> Optional[Tuple[int, int]]:
    parser = _get_parser()
    if parser is None:
        return None
    tree = parser.parse(source)
    move_signals = 0
    func_defs = 0

    def walk(node):
        nonlocal move_signals, func_defs
        t = node.type
        if t == "function_definition":
            func_defs += 1
            # detect `noexcept` on this function
            txt = node.text.decode("utf8", errors="replace")
            # tree-sitter doesn't always expose noexcept_specifier; the keyword
            # always appears in the function declarator text — and since
            # `noexcept` isn't a valid identifier elsewhere, presence in the
            # declarator-level text is reliable. We restrict to the first
            # 200 chars (signature, before body).
            sig = txt[:200]
            # REGEX_OK: tool_output — keyword scan in known-bounded signature text
            if "noexcept" in sig:
                move_signals += 1
        if t == "call_expression":
            name = _call_target_name(node)
            if name in ("move", "forward"):
                move_signals += 1
        if t == "abstract_reference_declarator" or t == "reference_declarator":
            # rvalue reference detection: contains two `&` tokens in children
            amps = sum(1 for c in node.children if c.type == "&")
            if amps >= 2:
                move_signals += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return move_signals, func_defs


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    total_signals = 0
    total_funcs = 0
    for content in by_path.values():
        res = _count(content.encode("utf8", errors="replace"))
        if res is None:
            return None
        s, f = res
        total_signals += s
        total_funcs += f
    if total_funcs == 0 and total_signals == 0:
        # No functions in this diff — move semantics doesn't apply
        return None
    raw = total_signals / (total_funcs + 1)
    return float(min(raw, 1.0))
