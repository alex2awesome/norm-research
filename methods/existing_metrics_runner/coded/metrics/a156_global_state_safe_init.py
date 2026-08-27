"""a156: Avoid global state and ensure safe initialization.

Aspect (aspects.json): "Minimize or avoid mutable global state; prefer
locals/module scope or dependency injection. Prevent initialization-order
hazards from global objects with constructors/destructors."

This metric measures *density of mutable global state* per file using
tree-sitter parsers (no subprocess). It is multi-language and reports a
per-file density that is then inverted to a [0,1] conformance score, where
1.0 means "no detected mutable globals" and decays with each addition.

Per-language signals:

  Python (.py, .pyi):
    Two complementary signals.
    A. Module-level *mutable* assignments. An assignment at module top is
       counted as mutable when:
         - the LHS is a simple identifier (not dunder, not ALL-UPPER
           constant-style, not a TYPE alias of a literal-only RHS), and
         - the RHS is a mutable-container literal (list/dict/set) OR a call
           expression (e.g. `defaultdict()`, `Counter()`, `Lock()`).
       Immutable literals (int/str/float/bool/None/tuple) and ALL-UPPER
       constants are ignored — those are *constants*, not mutable state.
       Type aliases like `Foo = Bar | None` are ignored (RHS not in our
       mutable set).
    B. ``global`` statements inside functions: each occurrence is a strong
       signal of cross-call mutable state.
    file_density = (n_mutable_module_assigns + n_global_stmts)
                   / max(n_top_level_decls + n_functions, 1)
    file_score   = exp(-density)

  Java (.java):
    A. ``static`` non-``final`` fields anywhere in the file.
       (``static final`` is a constant; only `static` without `final` is
       mutable shared state.)
    file_density = n_static_nonfinal / max(n_classes + n_methods, 1)
    file_score   = exp(-density)

  Go (.go):
    A. Package-level ``var`` declarations (mutable). ``const`` is fine and
       ignored. We do NOT count function-local var.
    file_density = n_top_var_specs / max(n_top_decls, 1)
    file_score   = exp(-density)

  JavaScript / TypeScript (.js/.jsx/.mjs/.cjs/.ts/.tsx):
    A. Top-level ``let`` / ``var`` declarations (mutable). ``const`` is
       ignored. Module-level ``let`` is package-scope mutable state across
       all importers.
    file_density = n_top_mutable_lex / max(n_top_decls, 1)
    file_score   = exp(-density)

Score:
  Diff score = mean of file scores. None when no supported file present or
  when no top-level declarations were observed in any added file.

Classification: PARTIALLY_THIN.
  - Counting mutable global decls IS thin (pure tree-sitter rules).
  - The decay shape (`exp(-density)`) is a calibration choice rather than a
    universal threshold. Whether one mutable singleton is OK depends on
    architectural context the diff doesn't show.
  - The Python "RHS in mutable set OR call" heuristic over-applies on
    factory functions that return immutables and under-applies on
    expressions that mutate through methods after assignment — these are
    the partial-thinness wrinkles.

Not measured (out of scope, would push to THICK):
  - Whether dependency injection is *the right alternative* in this codebase.
  - C++ static-initialization-order fiasco (no C++ in fixtures; could be
    added with tree-sitter-cpp on demand).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a156"
ASPECT_NAME = "Avoid global state and ensure safe initialization"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

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


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# ----- Python --------------------------------------------------------------

# RHS node types we treat as mutable (lists/dicts/sets and call results).
# Immutables (int/str/float/bool/None/tuple) are intentionally excluded.
_PY_MUTABLE_RHS = {"list", "dictionary", "set", "list_comprehension",
                   "dictionary_comprehension", "set_comprehension",
                   "call"}
# RHS node types we treat as constants (no mutation possible by reassign).
_PY_IMMUTABLE_RHS = {"integer", "float", "string", "true", "false", "none",
                     "tuple", "concatenated_string", "unary_operator",
                     "binary_operator", "conditional_expression"}


def _py_count(root, src: bytes) -> Optional[Dict[str, int]]:
    n_top_decls = 0
    n_funcs = 0
    n_mutable_assigns = 0
    n_global_stmts = 0

    def is_dunder(name: str) -> bool:
        return name.startswith("__") and name.endswith("__")

    def is_const_name(name: str) -> bool:
        # Heuristic: ALL-UPPER (allowing digits and underscore) -> constant.
        return name.isupper() and any(c.isalpha() for c in name)

    def walk_globals(node):
        nonlocal n_global_stmts
        if node.type == "global_statement":
            n_global_stmts += 1
        for c in node.children:
            walk_globals(c)

    # Top-level
    for c in root.children:
        t = c.type
        if t == "function_definition":
            n_top_decls += 1
            n_funcs += 1
            walk_globals(c)
        elif t == "class_definition":
            n_top_decls += 1
            # walk into class for nested funcs that use global
            walk_globals(c)
        elif t == "decorated_definition":
            inner = None
            for cc in c.children:
                if cc.type in ("function_definition", "class_definition"):
                    inner = cc
                    break
            if inner is not None:
                n_top_decls += 1
                if inner.type == "function_definition":
                    n_funcs += 1
                walk_globals(inner)
        elif t == "expression_statement":
            for cc in c.children:
                if cc.type == "assignment":
                    # children typically: [lhs, "=", rhs] or with type annot
                    lhs = None
                    rhs = None
                    seen_eq = False
                    for kid in cc.children:
                        if kid.type == "=":
                            seen_eq = True
                            continue
                        if not seen_eq and lhs is None:
                            lhs = kid
                        elif seen_eq and rhs is None:
                            rhs = kid
                    if lhs is None or rhs is None:
                        continue
                    if lhs.type != "identifier":
                        # tuple unpacking / subscript / attribute -> skip
                        continue
                    name = _text(lhs, src)
                    if is_dunder(name) or name == "__all__":
                        continue
                    if is_const_name(name):
                        n_top_decls += 1
                        continue
                    n_top_decls += 1
                    if rhs.type in _PY_MUTABLE_RHS:
                        # Exclude obvious frozen calls like frozenset(), tuple()
                        if rhs.type == "call":
                            # peek callee name
                            callee = rhs.children[0] if rhs.children else None
                            callee_txt = _text(callee, src) if callee else ""
                            if callee_txt in ("frozenset", "tuple", "int",
                                              "str", "float", "bool",
                                              "bytes", "complex", "range",
                                              "len", "id", "hash",
                                              "TypeVar", "NewType",
                                              "namedtuple", "NamedTuple",
                                              "Enum", "auto",
                                              "logging.getLogger",
                                              "getLogger"):
                                continue
                        n_mutable_assigns += 1
                    elif rhs.type in _PY_IMMUTABLE_RHS:
                        continue
                    # Other RHS shapes (attribute access, identifier alias)
                    # we conservatively skip -- could be a type alias.
        # ignore imports, etc.
    return {
        "n_top_decls": n_top_decls,
        "n_funcs": n_funcs,
        "n_mutable_assigns": n_mutable_assigns,
        "n_global_stmts": n_global_stmts,
    }


def _py_score_file(root, src: bytes) -> Optional[float]:
    c = _py_count(root, src)
    if c is None:
        return None
    denom = c["n_top_decls"] + c["n_funcs"]
    if denom == 0:
        return None
    numer = c["n_mutable_assigns"] + c["n_global_stmts"]
    density = numer / denom
    return math.exp(-density)


# ----- Java ----------------------------------------------------------------

def _java_modifier_text(node, src: bytes) -> str:
    for c in node.children:
        if c.type == "modifiers":
            return _text(c, src)
    return ""


def _java_count(root, src: bytes) -> Optional[Dict[str, int]]:
    n_classes = 0
    n_methods = 0
    n_static_nonfinal = 0
    saw_type = False

    def walk(node):
        nonlocal n_classes, n_methods, n_static_nonfinal, saw_type
        t = node.type
        if t in ("class_declaration", "interface_declaration",
                 "enum_declaration", "record_declaration",
                 "annotation_type_declaration"):
            n_classes += 1
            saw_type = True
        elif t in ("method_declaration", "constructor_declaration"):
            n_methods += 1
        elif t == "field_declaration":
            mods = _java_modifier_text(node, src)
            if "static" in mods and "final" not in mods:
                # Count number of declarators -> each variable counts.
                n_decls = 0
                for c in node.children:
                    if c.type == "variable_declarator":
                        n_decls += 1
                n_static_nonfinal += max(n_decls, 1)
        for c in node.children:
            walk(c)

    walk(root)
    if not saw_type:
        return None
    return {
        "n_classes": n_classes,
        "n_methods": n_methods,
        "n_static_nonfinal": n_static_nonfinal,
    }


def _java_score_file(root, src: bytes) -> Optional[float]:
    c = _java_count(root, src)
    if c is None:
        return None
    denom = c["n_classes"] + c["n_methods"]
    if denom == 0:
        return None
    density = c["n_static_nonfinal"] / denom
    return math.exp(-density)


# ----- Go -------------------------------------------------------------------

def _go_count(root, src: bytes) -> Optional[Dict[str, int]]:
    n_top_decls = 0
    n_top_var_specs = 0
    for c in root.children:
        t = c.type
        if t in ("function_declaration", "method_declaration",
                 "type_declaration", "const_declaration", "var_declaration",
                 "import_declaration"):
            if t == "import_declaration":
                continue
            n_top_decls += 1
            if t == "var_declaration":
                # count spec children
                n_specs = 0
                for spec in c.children:
                    if spec.type == "var_spec":
                        # Count each identifier on LHS as a separate var.
                        for cc in spec.children:
                            if cc.type == "identifier":
                                n_specs += 1
                if n_specs == 0:
                    n_specs = 1
                n_top_var_specs += n_specs
    if n_top_decls == 0:
        return None
    return {"n_top_decls": n_top_decls,
            "n_top_var_specs": n_top_var_specs}


def _go_score_file(root, src: bytes) -> Optional[float]:
    c = _go_count(root, src)
    if c is None:
        return None
    density = c["n_top_var_specs"] / max(c["n_top_decls"], 1)
    return math.exp(-density)


# ----- JS / TS --------------------------------------------------------------

def _js_count(root, src: bytes) -> Optional[Dict[str, int]]:
    n_top_decls = 0
    n_top_mutable = 0
    saw_anything = False

    def count_lex(node, is_top: bool):
        nonlocal n_top_mutable
        # lexical_declaration has a "kind" child or its first child is
        # the keyword text "let" / "const".
        if not is_top:
            return 0
        kind = None
        n_declarators = 0
        for c in node.children:
            if c.type in ("let", "var", "const"):
                kind = c.type
            elif kind is None and c.type == "kind":
                kind = _text(c, src).strip()
            if c.type == "variable_declarator":
                n_declarators += 1
        if n_declarators == 0:
            n_declarators = 1
        if kind in ("let", "var"):
            n_top_mutable += n_declarators
        return n_declarators

    def walk_top():
        nonlocal n_top_decls, saw_anything
        for c in root.children:
            t = c.type
            if t in ("function_declaration", "class_declaration",
                     "generator_function_declaration",
                     "abstract_class_declaration",
                     "interface_declaration", "type_alias_declaration",
                     "enum_declaration"):
                n_top_decls += 1
                saw_anything = True
            elif t == "lexical_declaration":
                n_top_decls += 1
                saw_anything = True
                count_lex(c, is_top=True)
            elif t == "variable_declaration":  # bare `var ...` at top
                n_top_decls += 1
                saw_anything = True
                # count declarators
                n_d = 0
                for cc in c.children:
                    if cc.type == "variable_declarator":
                        n_d += 1
                n_top_mutable_local = max(n_d, 1)
                # treat as mutable
                nonlocal n_top_mutable
                n_top_mutable += n_top_mutable_local
            elif t == "export_statement":
                # peek inside
                for cc in c.children:
                    if cc.type in ("function_declaration",
                                   "class_declaration",
                                   "interface_declaration",
                                   "type_alias_declaration",
                                   "enum_declaration"):
                        n_top_decls += 1
                        saw_anything = True
                    elif cc.type == "lexical_declaration":
                        n_top_decls += 1
                        saw_anything = True
                        count_lex(cc, is_top=True)
                    elif cc.type == "variable_declaration":
                        n_top_decls += 1
                        saw_anything = True
                        n_d = 0
                        for ccc in cc.children:
                            if ccc.type == "variable_declarator":
                                n_d += 1
                        n_top_mutable += max(n_d, 1)

    walk_top()
    if not saw_anything:
        return None
    return {"n_top_decls": n_top_decls,
            "n_top_mutable": n_top_mutable}


def _js_score_file(root, src: bytes) -> Optional[float]:
    c = _js_count(root, src)
    if c is None:
        return None
    density = c["n_top_mutable"] / max(c["n_top_decls"], 1)
    return math.exp(-density)


# ----- Driver --------------------------------------------------------------

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
    file_scores: List[float] = []
    for path, content in by_path.items():
        lang = _path_lang(path)
        if lang is None:
            continue
        parser = _get_parser(lang)
        if parser is None:
            continue
        code = content.encode("utf8", errors="replace")
        try:
            tree = parser.parse(code)
        except Exception:
            continue
        root = tree.root_node
        if lang == "py":
            s = _py_score_file(root, code)
        elif lang == "java":
            s = _java_score_file(root, code)
        elif lang == "go":
            s = _go_score_file(root, code)
        elif lang in ("js", "ts"):
            s = _js_score_file(root, code)
        else:
            s = None
        if s is not None:
            file_scores.append(s)
    if not file_scores:
        return None
    return float(sum(file_scores) / len(file_scores))
