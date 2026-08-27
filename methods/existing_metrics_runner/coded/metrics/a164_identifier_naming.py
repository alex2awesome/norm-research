"""a164: Identifier naming and casing conventions.

Walks AST per language (tree-sitter) to extract declared identifiers paired
with the kind of thing they declare (function, class, constant, field, etc.),
then checks the identifier's casing against the per-language convention:

Python (PEP 8):
  - functions, methods, variables, parameters -> snake_case
  - classes                                   -> PascalCase
  - module-level UPPER assignments            -> UPPER_SNAKE (constants)
  - dunder names (__foo__) are exempt; single underscore "_" is exempt

JavaScript / TypeScript:
  - functions, methods, variables             -> camelCase
  - classes, TS interfaces, TS type aliases   -> PascalCase
  - `const FOO = ...` ALL-UPPER               -> UPPER_SNAKE (constants)

Java:
  - methods, fields, parameters               -> camelCase
  - classes, interfaces, enums, records       -> PascalCase
  - `static final` fields                     -> UPPER_SNAKE

Go (Effective Go):
  - any top-level identifier starting upper   -> PascalCase (exported)
  - any top-level identifier starting lower   -> camelCase (unexported)
  - both must be free of underscores (idiomatic Go avoids `_` in names)

Scoring: per file, fraction of declared identifiers that conform to the rule
for their role. Overall score = mean across files. Abstain when no declared
identifier was observed (e.g. diff has only expressions / comments / data).

Ambiguity notes:
  - Python sometimes uses PascalCase for type aliases (PEP 8 silent); we do
    not penalize that because we can't reliably distinguish a class-typed
    constant from a type alias at the AST level. We only enforce
    UPPER_SNAKE on assignments whose RHS is a literal (int/string/None/bool
    /tuple/list/dict/set) — leaving aliases like `MyType = list[int]`
    un-flagged.
  - Go: `_` is the blank identifier, exempt.
  - TS: the `IFoo` "Hungarian-I" prefix is allowed (still PascalCase).
"""
from __future__ import annotations

# REGEX_OK: tool_output — these patterns check identifier STRINGS (sequences
# of bytes), not source code. Regex is the right tool for character-class
# membership tests like "is this snake_case".
import re
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a164"
ASPECT_NAME = "Identifier naming and casing conventions"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# REGEX_OK: tool_output — pure character-class patterns on identifier strings.
SNAKE_CASE = re.compile(r"^_{0,2}[a-z][a-z0-9_]*_?$")
# REGEX_OK: tool_output — camelCase identifier-string check (not source).
# Trailing _ allowed for keyword-collision avoidance like `class_`.
CAMEL_CASE = re.compile(r"^[a-z][a-zA-Z0-9]*_?$")
# REGEX_OK: tool_output — PascalCase identifier-string check.
PASCAL_CASE = re.compile(r"^_?[A-Z][a-zA-Z0-9]*$")
# REGEX_OK: tool_output — UPPER_SNAKE identifier-string check.
UPPER_SNAKE = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
# REGEX_OK: tool_output — dunder name shape like __init__.
DUNDER = re.compile(r"^__[a-zA-Z0-9_]+__$")
# Truly trivial names we never penalize.
EXEMPT_NAMES = frozenset({"_", "__", "self", "cls"})

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


def _ok_snake(name: str) -> bool:
    if DUNDER.match(name):
        return True
    return bool(SNAKE_CASE.match(name))


def _ok_camel(name: str) -> bool:
    return bool(CAMEL_CASE.match(name))


def _ok_pascal(name: str) -> bool:
    return bool(PASCAL_CASE.match(name))


def _ok_upper_snake(name: str) -> bool:
    return bool(UPPER_SNAKE.match(name))


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# ----- Python ----------------------------------------------------------------

_LITERAL_NODES = {"integer", "float", "string", "true", "false", "none",
                  "tuple", "list", "dictionary", "set",
                  "concatenated_string"}


def _py_walk(root, src: bytes) -> List[Tuple[str, str]]:
    """Return [(name, role)] for each declaration we can classify in a Python
    file. role in {"func","class","const","var"}.
    """
    out: List[Tuple[str, str]] = []

    def first_id(node):
        for c in node.children:
            if c.type == "identifier":
                return c
        return None

    def walk(node, at_module_top: bool):
        t = node.type
        if t == "function_definition":
            nm = first_id(node)
            if nm is not None:
                out.append((_text(nm, src), "func"))
            for c in node.children:
                walk(c, False)
            return
        if t == "class_definition":
            nm = first_id(node)
            if nm is not None:
                out.append((_text(nm, src), "class"))
            for c in node.children:
                walk(c, False)
            return
        if at_module_top and t == "expression_statement":
            # Module-level assignment -> potentially a constant.
            for c in node.children:
                if c.type == "assignment":
                    lhs = c.children[0] if c.children else None
                    rhs = c.children[-1] if len(c.children) >= 3 else None
                    if lhs is not None and lhs.type == "identifier":
                        name = _text(lhs, src)
                        # Only enforce UPPER_SNAKE if the name LOOKS like a
                        # constant AND the RHS is a literal. Otherwise treat
                        # as a regular variable (snake_case).
                        if rhs is not None and rhs.type in _LITERAL_NODES \
                                and name.isupper():
                            out.append((name, "const"))
                        else:
                            out.append((name, "var"))
            return
        for c in node.children:
            walk(c, at_module_top and (t in ("module", "program")))

    walk(root, at_module_top=True)
    return out


def _py_judge(name: str, role: str) -> Optional[bool]:
    if name in EXEMPT_NAMES:
        return None
    if role == "func":
        return _ok_snake(name)
    if role == "class":
        return _ok_pascal(name)
    if role == "const":
        return _ok_upper_snake(name)
    if role == "var":
        return _ok_snake(name)
    return None


# ----- JavaScript / TypeScript ----------------------------------------------

_JS_LITERAL_NODES = {"number", "string", "true", "false", "null", "undefined",
                     "template_string", "regex", "array", "object",
                     "unary_expression"}


def _js_walk(root, src: bytes, is_ts: bool) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    def walk(node):
        t = node.type
        if t == "function_declaration":
            for c in node.children:
                if c.type == "identifier":
                    out.append((_text(c, src), "func"))
                    break
        elif t == "class_declaration":
            for c in node.children:
                if c.type in ("identifier", "type_identifier"):
                    out.append((_text(c, src), "class"))
                    break
        elif is_ts and t == "interface_declaration":
            for c in node.children:
                if c.type in ("type_identifier", "identifier"):
                    out.append((_text(c, src), "class"))  # treat as Pascal
                    break
        elif is_ts and t == "type_alias_declaration":
            for c in node.children:
                if c.type in ("type_identifier", "identifier"):
                    out.append((_text(c, src), "class"))  # Pascal
                    break
        elif is_ts and t == "enum_declaration":
            for c in node.children:
                if c.type in ("identifier", "type_identifier"):
                    out.append((_text(c, src), "class"))
                    break
        elif t == "method_definition":
            for c in node.children:
                if c.type == "property_identifier":
                    nm = _text(c, src)
                    if nm not in ("constructor",):
                        out.append((nm, "func"))
                    break
        elif t == "lexical_declaration":
            is_const = any(c.type == "const" for c in node.children)
            for c in node.children:
                if c.type == "variable_declarator":
                    nm_node = None
                    val_node = None
                    seen_eq = False
                    for cc in c.children:
                        if cc.type == "identifier" and nm_node is None:
                            nm_node = cc
                        elif cc.type == "=":
                            seen_eq = True
                        elif seen_eq and val_node is None:
                            val_node = cc
                    if nm_node is None:
                        continue
                    nm = _text(nm_node, src)
                    if is_const and val_node is not None and \
                            val_node.type in _JS_LITERAL_NODES \
                            and nm.isupper():
                        out.append((nm, "const"))
                    elif val_node is not None and \
                            val_node.type in ("arrow_function",
                                              "function_expression",
                                              "function"):
                        out.append((nm, "func"))
                    else:
                        out.append((nm, "var"))
        elif t == "variable_declaration":  # var x = ...
            for c in node.children:
                if c.type == "variable_declarator":
                    for cc in c.children:
                        if cc.type == "identifier":
                            out.append((_text(cc, src), "var"))
                            break
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _js_judge(name: str, role: str) -> Optional[bool]:
    if name in EXEMPT_NAMES or name.startswith("$"):
        return None
    if role == "func":
        return _ok_camel(name)
    if role == "class":
        return _ok_pascal(name)
    if role == "const":
        return _ok_upper_snake(name)
    if role == "var":
        return _ok_camel(name)
    return None


# ----- Java ------------------------------------------------------------------

def _java_walk(root, src: bytes) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    def walk(node):
        t = node.type
        if t in ("class_declaration", "interface_declaration",
                 "enum_declaration", "record_declaration",
                 "annotation_type_declaration"):
            for c in node.children:
                if c.type == "identifier":
                    out.append((_text(c, src), "class"))
                    break
        elif t == "method_declaration":
            for c in node.children:
                if c.type == "identifier":
                    out.append((_text(c, src), "func"))
                    break
        elif t == "field_declaration":
            modifiers_txt = ""
            for c in node.children:
                if c.type == "modifiers":
                    modifiers_txt = _text(c, src)
            is_const = ("static" in modifiers_txt
                        and "final" in modifiers_txt)
            for c in node.children:
                if c.type == "variable_declarator":
                    for cc in c.children:
                        if cc.type == "identifier":
                            out.append((_text(cc, src),
                                        "const" if is_const else "var"))
                            break
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _java_judge(name: str, role: str) -> Optional[bool]:
    if name in EXEMPT_NAMES:
        return None
    if role == "class":
        return _ok_pascal(name)
    if role == "func":
        return _ok_camel(name)
    if role == "const":
        return _ok_upper_snake(name)
    if role == "var":
        return _ok_camel(name)
    return None


# ----- Go --------------------------------------------------------------------

def _go_walk(root, src: bytes) -> List[Tuple[str, str]]:
    """Returns [(name, role)] where role is one of "go_decl" (any declared
    identifier). Go cares about exported vs unexported (initial case), not
    role-specific casing — both forms are camel/Pascal of words with no
    underscores.
    """
    out: List[Tuple[str, str]] = []

    def add_ids_from(node, role):
        # Generic helper: collect direct-child identifiers.
        for c in node.children:
            if c.type in ("identifier", "field_identifier",
                          "type_identifier"):
                out.append((_text(c, src), role))

    def walk(node):
        t = node.type
        if t == "function_declaration":
            add_ids_from(node, "go_decl")
        elif t == "method_declaration":
            for c in node.children:
                if c.type == "field_identifier":
                    out.append((_text(c, src), "go_decl"))
                    break
        elif t == "type_spec":
            for c in node.children:
                if c.type == "type_identifier":
                    out.append((_text(c, src), "go_decl"))
                    break
        elif t == "const_spec" or t == "var_spec":
            for c in node.children:
                if c.type == "identifier":
                    out.append((_text(c, src), "go_decl"))
        for c in node.children:
            walk(c)

    walk(root)
    return out


# REGEX_OK: tool_output — Go identifier shape check; testing strings, not code.
_GO_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


def _go_judge(name: str, role: str) -> Optional[bool]:
    if name in EXEMPT_NAMES:
        return None
    # Go convention: MixedCaps (Pascal or camel), no underscores.
    return bool(_GO_NAME_RE.match(name))


# ----- Dispatch --------------------------------------------------------------

def _file_score(code: bytes, lang: str) -> Optional[float]:
    parser = _get_parser(lang)
    if parser is None:
        return None
    tree = parser.parse(code)
    root = tree.root_node
    if lang == "py":
        items = _py_walk(root, code)
        judge = _py_judge
    elif lang in ("js",):
        items = _js_walk(root, code, is_ts=False)
        judge = _js_judge
    elif lang == "ts":
        items = _js_walk(root, code, is_ts=True)
        judge = _js_judge
    elif lang == "java":
        items = _java_walk(root, code)
        judge = _java_judge
    elif lang == "go":
        items = _go_walk(root, code)
        judge = _go_judge
    else:
        return None
    n_total = n_ok = 0
    for name, role in items:
        v = judge(name, role)
        if v is None:
            continue
        n_total += 1
        if v:
            n_ok += 1
    if n_total == 0:
        return None
    return n_ok / n_total


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
        s = _file_score(content.encode("utf8", errors="replace"), lang)
        if s is not None:
            scs.append(s)
    if not scs:
        return None
    return float(sum(scs) / len(scs))
