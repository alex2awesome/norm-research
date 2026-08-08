"""a43: Naming clarity and conventions.

The norm asks identifiers to be "clear, descriptive, consistently styled ...
as short as possible but as long as necessary". The CONVENTIONS half is
covered by a164 (multi-language casing) and a387 (Java type/constructor
identity). This metric targets the CLARITY half, which a164/a387 do not
measure: a fully snake_case Python name like ``tmp`` is convention-conformant
under a164 yet violates a43.

Honest framing: semantic clarity is NOT deterministically observable from a
diff. "Descriptive enough" depends on context (`i` is fine in a tight loop,
terrible as a module-level constant). We therefore mark this metric
``PARTIALLY_THIN`` and use three SURROGATES that capture the easiest
unambiguous failures of the norm, each computed structurally over the
tree-sitter declaration list (so the signal does NOT collapse into a
text-length proxy like the codegen_claude regex did).

Surrogate signals (per declared identifier; loop-bound vars excluded):

  (1) **Single-letter / very-short names outside loops.** Names of length 1
      or 2 declared as functions, classes, methods, fields, constants, or
      non-loop variables. ``i``/``j``/``k`` are exempt when they appear as
      the iteration variable of a ``for`` statement.

  (2) **Placeholder/low-information names.** A small curated set of names
      that are universally considered semantically empty when used as
      durable identifiers: ``tmp``, ``temp``, ``foo``, ``bar``, ``baz``,
      ``qux``, ``data``, ``val``, ``value``, ``obj``, ``thing``, ``stuff``,
      ``var``, ``my_var``, ``test1``, ``test2``, etc. These are NOT
      conventionally short ("ctx", "cfg") — those are excluded because
      they are widely accepted abbreviations in their domains.

  (3) **Numeric-suffix laziness.** Names ending in a bare digit that are
      not Java type parameters (``T1``) or version markers (``v2``):
      ``user1``, ``thing2``. These typically indicate a copy-paste that
      should have been renamed.

Scoring: per file, ``score = 1 - (n_unclear / n_total_declarations)``. Mean
across files. Abstain when fewer than 3 declared identifiers were observed
(too few to estimate a rate). Abstain when no supported language is in the
diff.

Why PARTIALLY_THIN rather than THIN:
  - The surrogate is a LOWER BOUND on unclear names. A name like
    ``handle`` or ``process`` is semantically vague but slips through.
  - The surrogate has a CEILING: ``a164`` already enforces case, and any
    fully spelled-out word that passes a164 also passes this metric.
    Beyond that ceiling is genuine taste — call it THICK residual.

Why PARTIALLY_THIN rather than THICK:
  - Catching ``def foo()`` or ``int x = 5`` as a module-level constant IS
    a deterministic failure of "clear, descriptive". A THICK label would
    falsely claim no deterministic signal exists. The honest measurement
    is "we catch the obvious failures, not the subtle ones."

Distinct from a164: a164 returns 1.0 for ``def tmp():`` (snake_case). This
metric returns < 1.0.
Distinct from a387: a387 checks Java constructor name == class name and
type-parameter shape; it does not look at descriptiveness at all.
"""
from __future__ import annotations

# REGEX_OK: tool_output -- character-class checks on identifier STRINGS, not
# parsing source code.
import re
from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a43"
ASPECT_NAME = "Naming clarity and conventions"
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

# Names that are conventionally allowed as loop / index / coordinate
# variables. We do NOT flag these when they appear *inside* a for-loop or
# similar narrow scope. They are still flagged at function/class/constant
# scope.
LOOP_OK_SHORT = frozenset({
    "i", "j", "k", "n", "m", "x", "y", "z", "t", "u", "v",
    "e", "err", "ok", "id",
})

# Always-exempt identifiers (language conventions, dunder, blank).
EXEMPT_NAMES = frozenset({
    "_", "__", "self", "cls", "this",
    "args", "kwargs",
})

# Universally accepted abbreviations across the supported languages. These
# are EXCLUDED from the placeholder list because penalizing them would just
# replicate stylistic preference (the THICK residual).
COMMON_ABBREVIATIONS = frozenset({
    "ctx", "cfg", "req", "res", "resp", "msg", "err", "buf", "len", "src",
    "dst", "dir", "fd", "fn", "fp", "fs", "ip", "url", "uri", "uid", "pid",
    "tid", "db", "io", "os", "ui", "ok", "id", "ms", "ns", "px", "ws",
    "addr", "args", "argv", "argc", "env", "ext", "info", "init", "max",
    "min", "num", "obj", "pos", "ptr", "ref", "reg", "ret", "sub", "sum",
    "tmp",  # appears here intentionally so we treat it once below
    "tot", "val", "vec", "ver", "wd", "regex", "re", "json", "yaml", "xml",
    "html", "css", "sql", "csv", "tsv", "uuid", "guid", "jwt", "rsa", "tls",
    "ssl", "tcp", "udp", "dns", "lru", "fifo", "lifo", "api",
})

# Bona-fide placeholder / low-info / metasyntactic names. These represent
# UNAMBIGUOUS clarity failures even though some overlap with COMMON_ABBR
# (intentional: ``tmp`` as a module-level constant is bad even though it's
# an accepted *local* abbreviation). We keep the list small and surgical.
PLACEHOLDER_NAMES = frozenset({
    "foo", "bar", "baz", "qux", "quux",
    "tmp", "temp", "temporary",
    "thing", "stuff", "things", "myvar", "my_var",
    "data1", "data2", "val1", "val2",
    "test1", "test2", "test3",
    "obj1", "obj2",
    "asdf", "blah", "todo",
})

# Shape check on identifier strings.
# Names ending in a bare digit (1..9) and at least one preceding letter.
# We exclude ``T1``/``T2`` etc. by also requiring length > 2 (so any
# Java/Go type parameter naming convention is unaffected).
# REGEX_OK: tool_output
NUMERIC_SUFFIX = re.compile(r"^[A-Za-z][A-Za-z]+[0-9]$")
# REGEX_OK: tool_output -- "vX" version markers are intentionally allowed
# (v1, v2, ...).
VERSION_MARKER = re.compile(r"^v[0-9]+$", re.IGNORECASE)


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


# -------------------------------------------------------------------- judging

def _is_unclear(name: str, role: str) -> bool:
    """Return True iff this declared identifier matches a clarity-failure
    surrogate. ``role`` is informational; loop-context exemption is applied
    by the caller before we get here.
    """
    if name in EXEMPT_NAMES:
        return False
    # Dunder methods always exempt.
    if name.startswith("__") and name.endswith("__"):
        return False
    base = name.strip("_")  # strip leading/trailing underscores for the
    if not base:
        return False
    base_low = base.lower()

    # (1) Single-letter / 2-char in non-loop context. We catch the
    # non-loop case here because the caller has already filtered loop
    # binders out.
    if len(base) == 1 and base.isalpha():
        return True
    if len(base) == 2 and base.isalpha() and base_low not in COMMON_ABBREVIATIONS:
        # 2-char functions/classes are almost always unclear (e.g. ``def fn``);
        # 2-char loop indices were filtered out above.
        return True

    # (2) Placeholder / metasyntactic.
    if base_low in PLACEHOLDER_NAMES:
        return True

    # (3) Numeric-suffix laziness (but NOT version markers like v2).
    if VERSION_MARKER.match(base):
        return False
    if NUMERIC_SUFFIX.match(base):
        return True

    return False


# ----------------------------------------------------------- per-language AST

def _py_loop_binders(root, src: bytes) -> Set[str]:
    """Collect identifiers bound by ``for <id> in ...`` so we can exempt
    them. We do this on the whole tree; the set is small."""
    out: Set[str] = set()

    def walk(node):
        if node.type == "for_statement":
            # children: 'for' <target> 'in' <iter> ':' <body> [...]
            for c in node.children:
                if c.type == "identifier":
                    out.add(_text(c, src))
                    break
                if c.type in ("pattern_list", "tuple_pattern"):
                    for cc in c.children:
                        if cc.type == "identifier":
                            out.add(_text(cc, src))
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _py_decls(root, src: bytes) -> List[Tuple[str, str]]:
    """Return [(name, role)] for declared identifiers in a Python file.
    role in {"func","class","const","var","arg"}.
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
            # parameters
            for c in node.children:
                if c.type == "parameters":
                    for cc in c.children:
                        if cc.type == "identifier":
                            out.append((_text(cc, src), "arg"))
                        elif cc.type in ("default_parameter",
                                         "typed_parameter",
                                         "typed_default_parameter"):
                            for ccc in cc.children:
                                if ccc.type == "identifier":
                                    out.append((_text(ccc, src), "arg"))
                                    break
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
            for c in node.children:
                if c.type == "assignment":
                    lhs = c.children[0] if c.children else None
                    if lhs is not None and lhs.type == "identifier":
                        name = _text(lhs, src)
                        role = "const" if name.isupper() else "var"
                        out.append((name, role))
        for c in node.children:
            walk(c, at_module_top and (t in ("module", "program")))

    walk(root, at_module_top=True)
    return out


def _js_loop_binders(root, src: bytes) -> Set[str]:
    out: Set[str] = set()

    def walk(node):
        if node.type in ("for_statement", "for_in_statement",
                          "for_of_statement"):
            # heuristic: any direct identifier child or any
            # variable_declarator inside the init slot is a loop binder.
            for c in node.children:
                if c.type == "identifier":
                    out.add(_text(c, src))
                elif c.type in ("lexical_declaration",
                                "variable_declaration"):
                    for cc in c.children:
                        if cc.type == "variable_declarator":
                            for ccc in cc.children:
                                if ccc.type == "identifier":
                                    out.add(_text(ccc, src))
                                    break
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _js_decls(root, src: bytes, is_ts: bool) -> List[Tuple[str, str]]:
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
        elif is_ts and t in ("interface_declaration",
                              "type_alias_declaration",
                              "enum_declaration"):
            for c in node.children:
                if c.type in ("type_identifier", "identifier"):
                    out.append((_text(c, src), "class"))
                    break
        elif t == "method_definition":
            for c in node.children:
                if c.type == "property_identifier":
                    nm = _text(c, src)
                    if nm != "constructor":
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
                    if is_const and val_node is not None and nm.isupper():
                        out.append((nm, "const"))
                    elif val_node is not None and val_node.type in (
                            "arrow_function", "function_expression",
                            "function"):
                        out.append((nm, "func"))
                    else:
                        out.append((nm, "var"))
        elif t == "variable_declaration":
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


def _java_loop_binders(root, src: bytes) -> Set[str]:
    out: Set[str] = set()

    def walk(node):
        if node.type in ("for_statement", "enhanced_for_statement"):
            for c in node.children:
                if c.type == "identifier":
                    out.add(_text(c, src))
                elif c.type == "local_variable_declaration":
                    for cc in c.children:
                        if cc.type == "variable_declarator":
                            for ccc in cc.children:
                                if ccc.type == "identifier":
                                    out.add(_text(ccc, src))
                                    break
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _java_decls(root, src: bytes) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    def walk(node):
        t = node.type
        if t in ("class_declaration", "interface_declaration",
                  "enum_declaration", "record_declaration"):
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
        elif t == "formal_parameter":
            for c in node.children:
                if c.type == "identifier":
                    out.append((_text(c, src), "arg"))
                    break
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _go_loop_binders(root, src: bytes) -> Set[str]:
    out: Set[str] = set()

    def walk(node):
        if node.type in ("for_statement", "range_clause"):
            for c in node.children:
                if c.type == "identifier":
                    out.add(_text(c, src))
                elif c.type == "expression_list":
                    for cc in c.children:
                        if cc.type == "identifier":
                            out.add(_text(cc, src))
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _go_decls(root, src: bytes) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    def walk(node):
        t = node.type
        if t == "function_declaration":
            for c in node.children:
                if c.type == "identifier":
                    out.append((_text(c, src), "func"))
                    break
        elif t == "method_declaration":
            for c in node.children:
                if c.type == "field_identifier":
                    out.append((_text(c, src), "func"))
                    break
        elif t == "type_spec":
            for c in node.children:
                if c.type == "type_identifier":
                    out.append((_text(c, src), "class"))
                    break
        elif t in ("const_spec", "var_spec"):
            for c in node.children:
                if c.type == "identifier":
                    role = "const" if t == "const_spec" else "var"
                    out.append((_text(c, src), role))
        elif t == "parameter_declaration":
            for c in node.children:
                if c.type == "identifier":
                    out.append((_text(c, src), "arg"))
        for c in node.children:
            walk(c)

    walk(root)
    return out


# --------------------------------------------------------------- per-file ---

def _file_score(code: bytes, lang: str) -> Optional[Tuple[int, int]]:
    """Return (n_unclear, n_total) for one file."""
    parser = _get_parser(lang)
    if parser is None:
        return None
    tree = parser.parse(code)
    root = tree.root_node

    if lang == "py":
        loop_binders = _py_loop_binders(root, code)
        decls = _py_decls(root, code)
    elif lang == "js":
        loop_binders = _js_loop_binders(root, code)
        decls = _js_decls(root, code, is_ts=False)
    elif lang == "ts":
        loop_binders = _js_loop_binders(root, code)
        decls = _js_decls(root, code, is_ts=True)
    elif lang == "java":
        loop_binders = _java_loop_binders(root, code)
        decls = _java_decls(root, code)
    elif lang == "go":
        loop_binders = _go_loop_binders(root, code)
        decls = _go_decls(root, code)
    else:
        return None

    if not decls:
        return None

    n_unclear = 0
    n_total = 0
    seen: Set[Tuple[str, str]] = set()
    for name, role in decls:
        if (name, role) in seen:
            # de-dup repeated declarations (e.g. overloaded methods, common
            # in JS class bodies) so one bad name isn't double-counted.
            continue
        seen.add((name, role))
        if name in EXEMPT_NAMES:
            continue
        # Loop binders may legitimately be 1-2 chars; exempt them entirely.
        if name in loop_binders and role in ("var", "arg"):
            continue
        n_total += 1
        if _is_unclear(name, role):
            n_unclear += 1
    if n_total == 0:
        return None
    return n_unclear, n_total


def _path_lang(path: str) -> Optional[str]:
    p = path.lower()
    for ext, lang in EXT_TO_LANG.items():
        if p.endswith(ext):
            return lang
    return None


# --------------------------------------------------------------- public API ---

def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    return any(_path_lang(p) is not None for p in by_path)


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    total_unclear = 0
    total_n = 0
    for path, content in by_path.items():
        lang = _path_lang(path)
        if lang is None:
            continue
        r = _file_score(content.encode("utf8", errors="replace"), lang)
        if r is None:
            continue
        u, n = r
        total_unclear += u
        total_n += n
    # Abstain if too few observations to estimate a rate reliably.
    if total_n < 3:
        return None
    return float(1.0 - (total_unclear / total_n))
