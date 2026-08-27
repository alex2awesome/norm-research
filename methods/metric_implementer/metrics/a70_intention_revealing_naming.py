"""a70: Intention-revealing naming and conventions.

The norm: "names clearly convey intent ... balance brevity with clarity, avoid
ambiguity/obscure abbreviations". This is the SEMANTIC half of a164's STRUCTURAL
casing rules. a164 enforces snake_case / PascalCase / UPPER_SNAKE. a387 enforces
Java constructor identity. Neither checks whether an identifier *says anything*.

This metric measures the residual: per declared identifier, is the name in a
known "intention-obscuring" set, or below a richness floor that suggests a
placeholder rather than a description?

Classification: PARTIALLY_THIN. The full norm ("readable APIs and call sites")
requires reader-side semantic understanding we cannot deterministically compute.
We surrogate it with three checkable surface features on the SAME identifier
strings a164/a387 inspect, but along an ORTHOGONAL axis (semantics, not shape):

  (1) Generic-name blacklist. A curated set of names that empirically convey
      no intent: ``data``, ``info``, ``result``, ``temp``, ``tmp``, ``value``,
      ``val``, ``obj``, ``item``, ``mgr``, ``manager``, ``helper``, ``util``,
      ``utils``, ``handler``, ``processor``, ``foo``, ``bar``, ``baz``,
      ``stuff``, ``thing``, ``misc``. Also single-letter names (other than
      conventional loop counters in narrow scope, which we cannot determine
      from a snippet, so we accept i/j/k/n/x/y/z as exempt).

  (2) Obscure-abbreviation floor. Names that are length 2-3 and not vowel-
      bearing real words (e.g. ``mgr``, ``proc``, ``tmp``, ``buf``, ``ptr``,
      ``ctx``) — we flag the well-known opaque short forms only; we do NOT
      penalize ``id``, ``ok``, ``ip``, ``ms``, ``ns`` which are unambiguous.

  (3) Length floor. Identifiers of 1-2 characters that are not in the
      narrow-scope exempt set are penalized; this catches ``a``, ``b``, ``c``,
      ``d`` used as fields/functions rather than loop counters.

Independence from a164: a164 returns ``snake_case`` PASS for ``data``,
``temp``, ``mgr`` — they obey the convention. This metric returns FAIL on
each. Conversely, a CamelCased generic like ``DataManager`` fails a70 but
passes a164. The signals are by construction non-redundant on the same
artifact.

Independence from a387: a387 only scores Java constructors and type
parameters. a70 scores every declared identifier across Python, JS, TS, Java,
Go (same five languages as a164).

Independence from a43: a43 is the same family-of-norms parent (general
clarity). If a43 is being implemented in parallel by another agent, the
expected division is: a43 covers descriptive-vs-laconic (length, vocabulary),
a70 covers intention-revealing-vs-generic (blacklist). The overlap is real
but small — both can co-exist; downstream model will discount duplication.

Scoring per file: ``n_ok / n_total`` over declared identifiers. Score per PR:
mean across files that yielded at least one judgeable identifier. Abstains
when no declarations are observable.
"""
from __future__ import annotations

# REGEX_OK: tool_output — these patterns check identifier STRINGS, not source.
import re
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a70"
ASPECT_NAME = "Intention-revealing naming and conventions"
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

# Names that empirically convey no intent. Lower-cased for comparison; we
# split CamelCase first so ``DataManager`` decomposes to ``data`` + ``manager``
# and gets two blacklist hits.
GENERIC_TOKENS = frozenset({
    # data shells
    "data", "info", "result", "results", "value", "values", "val", "vals",
    "obj", "object", "objs", "objects", "item", "items", "entry", "entries",
    "element", "elements", "node", "nodes", "thing", "things", "stuff",
    "misc", "etc",
    # generic roles
    "mgr", "manager", "helper", "helpers", "util", "utils", "utility",
    "handler", "handlers", "processor", "processors", "wrapper", "wrappers",
    "controller", "service", "factory",
    # placeholders
    "foo", "bar", "baz", "qux", "quux", "blah",
    # generic temporaries
    "temp", "tmp", "buf", "buff",
    # generic boolean/flag
    "flag", "ok", "test", "thing2",
})

# Opaque short abbreviations: 2-4 chars, no vowel-bearing meaning. We flag
# only well-known ones; benign acronyms (id, ip, ms, url, css, sql, ast) are
# omitted from this set.
OBSCURE_ABBREVS = frozenset({
    "mgr", "ctx", "ptr", "tmp", "buf", "obj", "val", "var", "src", "dst",
    "arr", "lst", "fn", "fns", "cb", "cbs", "cfg", "evt", "msg",
})

# Single/double-letter names exempt because they are conventionally used
# as iteration counters / coordinates / pair temporaries. Reviewed code is
# largely tolerant of these in narrow scope. We cannot determine scope from
# a diff hunk, so accept them globally.
EXEMPT_SHORT = frozenset({
    "i", "j", "k", "m", "n", "x", "y", "z", "_",
    "self", "cls", "this",
    # one-letter generics common as type parameters / language idioms
    "t", "e",
    # conventional pair temporaries
    "kv", "id", "ip", "ok", "io",
})

# REGEX_OK: tool_output — split identifier strings into morphemes; not parsing
# source code, just decomposing a name string.
_CAMEL_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

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


def _tokens(name: str) -> List[str]:
    """Break ``DataMgr`` -> ``data``, ``mgr``; ``data_mgr`` -> same; ``UPPER_SNAKE``
    -> two tokens. Lower-case all morphemes."""
    # Strip leading/trailing underscores (private, dunder).
    core = name.strip("_")
    if not core:
        return []
    # Split on underscores first (snake / UPPER_SNAKE), then CamelCase each chunk.
    out: List[str] = []
    for chunk in core.split("_"):
        if not chunk:
            continue
        morphemes = _CAMEL_SPLIT.findall(chunk)
        if not morphemes:
            out.append(chunk.lower())
        else:
            for mo in morphemes:
                out.append(mo.lower())
    return out


def _judge(name: str) -> Optional[bool]:
    """Return True if name conveys intent, False if it is generic/obscure,
    None if exempt (loop counter, self, this, dunder-like)."""
    # Dunder methods (__init__, __repr__) are language conventions — exempt.
    if name.startswith("__") and name.endswith("__") and len(name) > 4:
        return None
    lname = name.lower().strip("_")
    if not lname:
        return None
    if lname in EXEMPT_SHORT:
        return None
    # Single-letter declared names that are NOT exempt -> obscure.
    if len(lname) == 1:
        return False
    toks = _tokens(name)
    if not toks:
        return None
    # All-token blacklist check: if EVERY morpheme is generic, the name fails.
    # ``DataManager`` -> [data, manager] -> both generic -> fail.
    # ``cacheManager`` -> [cache, manager] -> ``cache`` is informative -> pass.
    nontrivial = [t for t in toks if t not in GENERIC_TOKENS]
    if not nontrivial:
        return False
    # Any morpheme that is an obscure abbreviation AND the name has only one
    # such morpheme (so it's the whole name) -> fail.
    if len(toks) == 1 and toks[0] in OBSCURE_ABBREVS:
        return False
    # Two-character names that aren't in EXEMPT_SHORT -> obscure.
    if len(lname) == 2 and lname not in EXEMPT_SHORT:
        return False
    return True


# ----- Walkers (same shape as a164, but we only need names, not roles) ------

def _py_walk(root, src: bytes) -> List[str]:
    out: List[str] = []

    def first_id(node):
        for c in node.children:
            if c.type == "identifier":
                return c
        return None

    def walk(node):
        t = node.type
        if t == "function_definition":
            nm = first_id(node)
            if nm is not None:
                out.append(_text(nm, src))
            # Parameter names too — they're declared identifiers.
            for c in node.children:
                if c.type == "parameters":
                    for pc in c.children:
                        if pc.type == "identifier":
                            out.append(_text(pc, src))
                        elif pc.type in ("typed_parameter",
                                         "default_parameter",
                                         "typed_default_parameter"):
                            sub = first_id(pc)
                            if sub is not None:
                                out.append(_text(sub, src))
        elif t == "class_definition":
            nm = first_id(node)
            if nm is not None:
                out.append(_text(nm, src))
        elif t == "assignment":
            # Only LHS-identifier assignments. We don't filter for module-level
            # (a164 does that for casing rules); semantic richness applies to
            # all named bindings.
            lhs = node.children[0] if node.children else None
            if lhs is not None and lhs.type == "identifier":
                out.append(_text(lhs, src))
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _js_walk(root, src: bytes, is_ts: bool) -> List[str]:
    out: List[str] = []

    def walk(node):
        t = node.type
        if t == "function_declaration":
            for c in node.children:
                if c.type == "identifier":
                    out.append(_text(c, src))
                    break
        elif t == "class_declaration":
            for c in node.children:
                if c.type in ("identifier", "type_identifier"):
                    out.append(_text(c, src))
                    break
        elif is_ts and t in ("interface_declaration",
                             "type_alias_declaration",
                             "enum_declaration"):
            for c in node.children:
                if c.type in ("type_identifier", "identifier"):
                    out.append(_text(c, src))
                    break
        elif t == "method_definition":
            for c in node.children:
                if c.type == "property_identifier":
                    nm = _text(c, src)
                    if nm != "constructor":
                        out.append(nm)
                    break
        elif t in ("lexical_declaration", "variable_declaration"):
            for c in node.children:
                if c.type == "variable_declarator":
                    for cc in c.children:
                        if cc.type == "identifier":
                            out.append(_text(cc, src))
                            break
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _java_walk(root, src: bytes) -> List[str]:
    out: List[str] = []

    def walk(node):
        t = node.type
        if t in ("class_declaration", "interface_declaration",
                 "enum_declaration", "record_declaration",
                 "annotation_type_declaration"):
            for c in node.children:
                if c.type == "identifier":
                    out.append(_text(c, src))
                    break
        elif t == "method_declaration":
            for c in node.children:
                if c.type == "identifier":
                    out.append(_text(c, src))
                    break
        elif t == "field_declaration":
            for c in node.children:
                if c.type == "variable_declarator":
                    for cc in c.children:
                        if cc.type == "identifier":
                            out.append(_text(cc, src))
                            break
        elif t == "local_variable_declaration":
            for c in node.children:
                if c.type == "variable_declarator":
                    for cc in c.children:
                        if cc.type == "identifier":
                            out.append(_text(cc, src))
                            break
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _go_walk(root, src: bytes) -> List[str]:
    out: List[str] = []

    def walk(node):
        t = node.type
        if t == "function_declaration":
            for c in node.children:
                if c.type in ("identifier", "field_identifier"):
                    out.append(_text(c, src))
                    break
        elif t == "method_declaration":
            for c in node.children:
                if c.type == "field_identifier":
                    out.append(_text(c, src))
                    break
        elif t == "type_spec":
            for c in node.children:
                if c.type == "type_identifier":
                    out.append(_text(c, src))
                    break
        elif t in ("const_spec", "var_spec"):
            for c in node.children:
                if c.type == "identifier":
                    out.append(_text(c, src))
        for c in node.children:
            walk(c)

    walk(root)
    return out


# ----- Dispatch -------------------------------------------------------------

def _file_score(code: bytes, lang: str) -> Optional[float]:
    parser = _get_parser(lang)
    if parser is None:
        return None
    tree = parser.parse(code)
    root = tree.root_node
    if lang == "py":
        names = _py_walk(root, code)
    elif lang == "js":
        names = _js_walk(root, code, is_ts=False)
    elif lang == "ts":
        names = _js_walk(root, code, is_ts=True)
    elif lang == "java":
        names = _java_walk(root, code)
    elif lang == "go":
        names = _go_walk(root, code)
    else:
        return None
    n_total = n_ok = 0
    for nm in names:
        v = _judge(nm)
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
