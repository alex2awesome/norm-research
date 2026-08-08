"""a387: Java type/class/interface/constructor UpperCamelCase naming.

This is the Java-specific subset of a164 (multi-language identifier naming).
a164 already enforces PascalCase on Java class/interface/enum/record names,
so to provide an INDEPENDENT signal we focus on the two Java-specific checks
a164 does NOT make:

  (1) CONSTRUCTOR identity rule. A Java constructor's identifier MUST exactly
      match its enclosing class identifier (character-for-character). This is
      stricter than "is PascalCase" — `class Foo { foo() {...} }` would pass
      a164's PascalCase check on the class but FAIL the constructor naming
      norm because `foo` does not equal `Foo`. (`foo()` is not actually a
      constructor under javac, but a tree-sitter constructor_declaration
      node uses whatever identifier the author wrote. We detect both: the
      mismatched casing of an intended constructor, and any constructor
      whose identifier diverges from the enclosing class.)

  (2) TYPE PARAMETER naming (generics `<T>`, `<E>`, `<K, V>`). The Java
      convention (JLS, Effective Java item 56) is single capital letters or
      a single capital letter followed by digits/short uppercase words, all
      UpperCamelCase starting with a capital. a164 ignores type parameters
      entirely.

We deliberately do NOT re-check class/interface/enum/record/annotation
declaration names — that overlaps with a164. If you want the union signal,
read both metrics.

Scoring per file:
  - n_ok / n_total over the (constructor + type-parameter) identifiers found.
  - applies() = True iff the diff added any Java content.
  - score() = mean over files that yielded at least one judgeable identifier.
  - Abstain when no constructor or type parameter was observed.
"""
from __future__ import annotations

# REGEX_OK: tool_output — character-class membership test on identifier strings,
# not parsing source code.
import re
from typing import List, Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a387"
ASPECT_NAME = "Java type/constructor UpperCamelCase naming"
TIER = 2
TOOLS = ["tree-sitter-java"]
APPLIES_TO_LANGS = ["Java"]
CLASSIFICATION = "THIN"

JAVA_EXTS = [".java"]

# REGEX_OK: tool_output — PascalCase (UpperCamelCase) check on identifier strings.
PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
# REGEX_OK: tool_output — Java type-parameter convention: single capital letter,
# or single capital with trailing digits, or short UpperCamel (e.g. T, E, K, V,
# T1, KEY, KeyT). All must start with a capital.
TYPE_PARAM_OK = re.compile(r"^[A-Z][A-Za-z0-9]*$")

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        try:
            import tree_sitter_java
            from tree_sitter import Language, Parser
            _PARSER = Parser(Language(tree_sitter_java.language()))
        except ImportError:
            return None
    return _PARSER


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _enclosing_type_name(node, src: bytes) -> Optional[str]:
    """Walk up to the nearest class/record/enum/interface and return its
    declared identifier."""
    cur = node.parent
    while cur is not None:
        if cur.type in ("class_declaration", "record_declaration",
                        "enum_declaration", "interface_declaration"):
            for c in cur.children:
                if c.type == "identifier":
                    return _text(c, src)
            return None
        cur = cur.parent
    return None


def _has_error_ancestor(node) -> bool:
    cur = node.parent
    while cur is not None:
        if cur.type == "ERROR":
            return True
        cur = cur.parent
    return False


def _walk_java(root, src: bytes) -> List[Tuple[str, str, bool]]:
    """Return list of (kind, name, ok) where:
      - kind in {"ctor", "tparam"}
      - name = the offending/checked identifier
      - ok   = True iff identifier satisfies the kind's naming rule
    """
    out: List[Tuple[str, str, bool]] = []

    def walk(node):
        t = node.type

        if t == "constructor_declaration" and not _has_error_ancestor(node):
            ctor_name = None
            for c in node.children:
                if c.type == "identifier":
                    ctor_name = _text(c, src)
                    break
            if ctor_name is not None:
                cls = _enclosing_type_name(node, src)
                # Two-part judgment: (a) UpperCamelCase, (b) matches class name.
                # We require BOTH for ok=True. The "matches class name" check
                # is the signal a164 lacks.
                is_pascal = bool(PASCAL_CASE.match(ctor_name))
                matches = (cls is not None and ctor_name == cls)
                out.append(("ctor", ctor_name, is_pascal and matches))

        elif t == "type_parameter" and not _has_error_ancestor(node):
            # type_parameter children: (optional annotations) identifier
            # (optional type_bound). We want the bare identifier.
            for c in node.children:
                if c.type == "type_identifier" or c.type == "identifier":
                    name = _text(c, src)
                    ok = bool(TYPE_PARAM_OK.match(name))
                    out.append(("tparam", name, ok))
                    break

        for c in node.children:
            walk(c)

    walk(root)
    return out


def _file_score(source: bytes) -> Optional[Tuple[int, int]]:
    """Return (n_ok, n_total) for one Java file, or None if unparseable."""
    parser = _get_parser()
    if parser is None:
        return None
    # Wrap loose snippets so tree-sitter has a class context (mirrors a283).
    src = source if (b"class " in source or b"interface " in source
                     or b"record " in source or b"enum " in source) else (
        b"class __Snip {\n" + source + b"\n}\n")
    tree = parser.parse(src)
    items = _walk_java(tree.root_node, src)
    if not items:
        return None
    n_ok = sum(1 for _, _, ok in items if ok)
    return n_ok, len(items)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, JAVA_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, JAVA_EXTS)
    if not by_path:
        return None
    total_ok = total_n = 0
    any_observed = False
    for content in by_path.values():
        res = _file_score(content.encode("utf8", errors="replace"))
        if res is None:
            continue
        ok, n = res
        total_ok += ok
        total_n += n
        any_observed = True
    if not any_observed or total_n == 0:
        # No constructor or type parameter found in added Java — norm doesn't
        # fire on this diff. Abstain rather than score 1.0 (which would be
        # indistinguishable from "no violation").
        return None
    return float(total_ok / total_n)
