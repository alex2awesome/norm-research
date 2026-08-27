"""a415: C++ header hygiene.

Composite of three sub-criteria computed via tree-sitter-cpp parse of added
content (no regex on code):

  (a) Include ordering: convention is system (<...>) → third-party → project
      (\"...\"). We approximate the convention as "all <...> includes appear
      before any \"...\" includes" within each file. Score 1.0 if ordered,
      0.0 otherwise. Files with <2 includes total don't apply this criterion.

  (b) Header guards: header files (.h, .hpp, .hh, .hxx) should use either
      `#pragma once` OR an `#ifndef X / #define X / #endif` triplet. Score
      1.0 if either present in a header, 0.0 if neither.

  (c) Duplicate includes: count duplicated include paths within a single
      file. Score = 1.0 if zero dupes, else 1.0 / (1 + dupes).

Final metric score is the unweighted mean of whichever sub-scores actually
applied (some files contribute (a) only, headers contribute (a)+(b)+(c)).

Tree-sitter-cpp emits `preproc_include`, `preproc_ifdef`, `preproc_def`,
`#pragma once` (as preproc_call with `pragma` directive). We walk these
preprocessor nodes only.

Tier 2. CLASSIFICATION THIN.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a415"
ASPECT_NAME = "C++ header hygiene"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C", "C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]
HEADER_EXTS = (".h", ".hpp", ".hxx", ".hh")

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


def _file_scores(source: bytes, is_header: bool):
    """Return a list of sub-scores in [0,1] for the criteria that apply to
    this file. Empty list if no criterion applied."""
    parser = _get_parser()
    if parser is None:
        return None
    tree = parser.parse(source)
    includes: List[Tuple[str, str]] = []  # (kind, path)
    has_pragma_once = False
    has_ifndef_define_triplet = False

    # Track top-level preprocessor includes & the first preproc_ifdef
    # together with its #define for guard detection.
    def walk(node, depth=0):
        nonlocal has_pragma_once, has_ifndef_define_triplet
        t = node.type
        if t == "preproc_include":
            # children: '#include' + system_lib_string OR string_literal
            kind = None
            path = None
            for c in node.children:
                if c.type == "system_lib_string":
                    kind = "system"
                    path = _text(c)
                elif c.type == "string_literal":
                    kind = "project"
                    path = _text(c)
            if kind is not None and path is not None:
                includes.append((kind, path))
        elif t == "preproc_call":
            # `#pragma once` or other directives
            txt = _text(node)
            # REGEX_OK: format_header — single-line preproc directive text
            if "pragma" in txt and "once" in txt:
                has_pragma_once = True
        elif t == "preproc_ifdef":
            # check first child for "#ifndef" + the next preproc_def for matching macro
            children = node.children
            if children and children[0].type == "#ifndef":
                # find a preproc_def child
                for c in children:
                    if c.type == "preproc_def":
                        has_ifndef_define_triplet = True
                        break
        for c in node.children:
            walk(c, depth + 1)

    walk(tree.root_node)

    scores: List[float] = []

    # (a) ordering
    if len(includes) >= 2:
        # all system before any project?
        seen_project = False
        ordered = True
        for kind, _ in includes:
            if kind == "project":
                seen_project = True
            elif kind == "system" and seen_project:
                ordered = False
                break
        scores.append(1.0 if ordered else 0.0)

    # (b) guards (headers only)
    if is_header:
        scores.append(1.0 if (has_pragma_once or has_ifndef_define_triplet) else 0.0)

    # (c) duplicates
    if includes:
        paths = [p for _, p in includes]
        dupes = len(paths) - len(set(paths))
        scores.append(1.0 / (1.0 + dupes))

    return scores


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    all_sub: List[float] = []
    for path, content in by_path.items():
        is_header = path.lower().endswith(HEADER_EXTS)
        sub = _file_scores(content.encode("utf8", errors="replace"), is_header)
        if sub is None:
            return None
        all_sub.extend(sub)
    if not all_sub:
        return None
    return float(sum(all_sub) / len(all_sub))
