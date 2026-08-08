"""a420: C++ identifier naming conventions.

Walks tree-sitter-cpp AST of added C++ files and checks naming convention
conformance for declared identifiers:

  - classes / structs / typedefs / type_alias  -> PascalCase
  - functions / methods                         -> snake_case OR camelCase (either tolerated)
  - macros / enum constants                     -> UPPER_SNAKE
  - other variables / fields                    -> snake_case OR camelCase

Returns conformance ratio in [0,1]. Abstains when no declared identifier
is observed.

Tier 2. THIN.
"""
from __future__ import annotations
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a420"
ASPECT_NAME = "C++ identifier naming conventions"
TIER = 2
TOOLS = ["tree-sitter-cpp"]
APPLIES_TO_LANGS = ["C++"]
CLASSIFICATION = "THIN"

CPP_EXTS = [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".hh", ".h", ".c"]

# REGEX_OK: tool_output — pure identifier-string shape checks.
RE_PASCAL = re.compile(r"^_?[A-Z][a-zA-Z0-9]*$")
RE_SNAKE = re.compile(r"^_{0,2}[a-z][a-z0-9_]*_?$")
RE_CAMEL = re.compile(r"^[a-z][a-zA-Z0-9]*_?$")
RE_UPPER = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
EXEMPT = frozenset({"_", "self", "this", "operator"})

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

def _ok_pascal(n): return bool(RE_PASCAL.match(n))
def _ok_snake_or_camel(n): return bool(RE_SNAKE.match(n) or RE_CAMEL.match(n))
def _ok_upper(n): return bool(RE_UPPER.match(n))


def _collect(root, src: bytes):
    items = []  # (name, role)
    for n in _walk_all(root):
        t = n.type
        if t in ("class_specifier", "struct_specifier", "union_specifier"):
            for c in n.children:
                if c.type == "type_identifier":
                    items.append((_text(c, src), "type"))
                    break
        elif t == "enum_specifier":
            for c in n.children:
                if c.type == "type_identifier":
                    items.append((_text(c, src), "type"))
                    break
            # enumerator constants
            for c in _walk_all(n):
                if c.type == "enumerator":
                    for cc in c.children:
                        if cc.type == "identifier":
                            items.append((_text(cc, src), "const"))
                            break
        elif t == "alias_declaration":
            for c in n.children:
                if c.type == "type_identifier":
                    items.append((_text(c, src), "type"))
                    break
        elif t == "type_definition":  # typedef
            # last identifier-like child is the alias name
            tid = None
            for c in n.children:
                if c.type == "type_identifier":
                    tid = c
            if tid is not None:
                items.append((_text(tid, src), "type"))
        elif t == "function_definition":
            for c in _walk_all(n):
                if c.type == "function_declarator":
                    for cc in c.children:
                        if cc.type in ("identifier", "field_identifier",
                                       "qualified_identifier"):
                            nm = _text(cc, src).split("::")[-1]
                            # operator overloads / destructors / ctors skipped
                            if nm and not nm.startswith("operator")                                     and not nm.startswith("~"):
                                items.append((nm, "func"))
                            break
                    break
        elif t == "preproc_def":
            # #define FOO ...
            for c in n.children:
                if c.type == "identifier":
                    items.append((_text(c, src), "macro"))
                    break
    return items


def _judge(name: str, role: str):
    if not name or name in EXEMPT or name.startswith("__"):
        return None
    if role == "type":
        return _ok_pascal(name)
    if role == "func":
        return _ok_snake_or_camel(name)
    if role == "macro":
        return _ok_upper(name)
    if role == "const":
        return _ok_upper(name) or _ok_pascal(name)  # kEnumValue also tolerated
    return None


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, CPP_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, CPP_EXTS)
    if not by_path:
        return None
    parser = _get_parser()
    if parser is None:
        return None
    n_ok = n_total = 0
    for content in by_path.values():
        src = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(src)
        except Exception:
            continue
        for name, role in _collect(tree.root_node, src):
            v = _judge(name, role)
            if v is None:
                continue
            n_total += 1
            if v:
                n_ok += 1
    if n_total == 0:
        return None
    return n_ok / n_total
