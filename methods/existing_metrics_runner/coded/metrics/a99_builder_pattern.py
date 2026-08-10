"""a99: Builder pattern for complex construction.

The norm asks: when a class/function has many or optional inputs, provide
a builder or parameter object; conversely, avoid gratuitous builders.

**Honest framing.** Deciding whether construction is "complex enough to
warrant a builder" is a judgment call — that part is THICK. What we *can*
measure deterministically from source is whether the builder pattern is
**present and used** in code that plausibly needs it. So this metric tracks
USAGE FREQUENCY, not APPROPRIATENESS. Mark CLASSIFICATION = PARTIALLY_THIN.

Detection (tree-sitter, no regex on code):

  Java
    - "Builder inner class" : a `class_declaration` whose simple name is
      exactly `Builder` and that lives inside another class/interface/record.
    - "Builder usage"       : a method-call chain of length >= 3 whose
      terminal selector is `.build()` (e.g. `Foo.builder().a(x).b(y).build()`).
    - "Constructor with many params" : any `constructor_declaration` (or
      record header) with >= 4 formal parameters — a class with such a
      constructor is a CANDIDATE for needing a builder.

  JavaScript / TypeScript
    - "Fluent chain" : a `call_expression` whose callee is a member access
      and that is itself the receiver of >= 2 further member-call links
      (chain length >= 3). This is the canonical fluent-builder shape.
    - "Wide constructor" : a `method_definition` named `constructor` with
      >= 4 parameters.

  Python
    - "Fluent class" : a `class_definition` containing >= 2 methods whose
      body ends with `return self`.
    - "Wide __init__" : an `__init__` with >= 4 non-self parameters.

Score
-----
For each language family we compute:

    builder_signal  = builder_usage + builder_class_def
    candidate       = constructors_or_classes_needing_one

  s = (builder_signal + 1) / (builder_signal + candidate + 1)     # Laplace
      clipped to [0,1]

A class with many wide constructors and no builder gets s -> 0; a file with
any builder usage gets s -> 1. The Laplace smoothing prevents div-by-zero
and keeps scores away from extremes when counts are tiny.

We aggregate across files in the diff by summing tallies, then computing s
once. `applies()` requires (a) at least one supported language file added
AND (b) at least one "candidate or builder signal" tally was observed —
otherwise the norm has nothing to say about this diff.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a99"
ASPECT_NAME = "Builder pattern for complex construction"
TIER = 2
TOOLS = ["tree-sitter-java", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-python"]
APPLIES_TO_LANGS = ["Java", "JavaScript", "TypeScript", "Python"]
CLASSIFICATION = "PARTIALLY_THIN"

JAVA_EXTS = (".java",)
JS_EXTS = (".js", ".jsx", ".mjs", ".cjs")
TS_EXTS = (".ts", ".tsx")
PY_EXTS = (".py", ".pyi")

WIDE_PARAM_THRESHOLD = 4  # >= this many params -> "complex construction candidate"

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "java":
            import tree_sitter_java as m
            L = Language(m.language())
        elif lang == "js":
            import tree_sitter_javascript as m
            L = Language(m.language())
        elif lang == "ts":
            import tree_sitter_typescript as m
            L = Language(m.language_typescript())
        elif lang == "tsx":
            import tree_sitter_typescript as m
            L = Language(m.language_tsx())
        elif lang == "python":
            import tree_sitter_python as m
            L = Language(m.language())
        else:
            _PARSERS[lang] = None
            return None
        _PARSERS[lang] = Parser(L)
    except Exception:
        _PARSERS[lang] = None
    return _PARSERS[lang]


def _lang_for_ext(ext: str) -> Optional[str]:
    ext = ext.lower()
    if ext in JAVA_EXTS:
        return "java"
    if ext == ".tsx":
        return "tsx"
    if ext in TS_EXTS:
        return "ts"
    if ext in JS_EXTS:
        return "js"
    if ext in PY_EXTS:
        return "python"
    return None


def _text(node) -> str:
    return node.text.decode("utf8", errors="replace") if node is not None else ""


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------
def _tally_java(code: bytes) -> Tuple[int, int, int]:
    """Return (builder_class, builder_usage, wide_constructors)."""
    parser = _get_parser("java")
    if parser is None:
        return (0, 0, 0)
    # Wrap snippets without a class so tree-sitter still parses something
    if (b"class " not in code and b"interface " not in code
            and b"record " not in code):
        code = b"class __Snip {\n" + code + b"\n}\n"
    root = parser.parse(code).root_node
    bc = bu = wc = 0

    def chain_terminates_in_build(call_node) -> bool:
        """Is this call_expression's final selector `.build`, with a chain
        of length >= 3?"""
        # call_expression children: object (call/identifier/field_access),
        #   method (identifier), argument_list.
        # We treat method-call chain length as number of call_expression
        # nodes along the receiver spine.
        name_child = None
        for ch in call_node.children:
            if ch.type == "identifier":
                name_child = ch
                break
            # field_access: receiver.identifier — the method name is rightmost
        # In tree-sitter-java, method_invocation has a "name" field.
        try:
            name_child = call_node.child_by_field_name("name") or name_child
        except Exception:
            pass
        if name_child is None or _text(name_child) != "build":
            return False
        # walk receiver-spine counting method_invocations
        depth = 1
        recv = call_node.child_by_field_name("object")
        while recv is not None and recv.type == "method_invocation":
            depth += 1
            recv = recv.child_by_field_name("object")
        return depth >= 3

    def walk(n, enclosing_is_class: bool):
        nonlocal bc, bu, wc
        t = n.type
        if t == "class_declaration":
            name = n.child_by_field_name("name")
            if (enclosing_is_class and name is not None
                    and _text(name) == "Builder"):
                bc += 1
            for c in n.children:
                walk(c, True)
            return
        if t in ("constructor_declaration",):
            params = n.child_by_field_name("parameters")
            if params is not None:
                k = sum(1 for c in params.children
                        if c.type == "formal_parameter")
                if k >= WIDE_PARAM_THRESHOLD:
                    wc += 1
        if t == "record_declaration":
            # record headers count as "constructor with N params"
            params = n.child_by_field_name("parameters")
            if params is not None:
                k = sum(1 for c in params.children
                        if c.type == "formal_parameter")
                if k >= WIDE_PARAM_THRESHOLD:
                    wc += 1
        if t == "method_invocation":
            if chain_terminates_in_build(n):
                bu += 1
        for c in n.children:
            walk(c, enclosing_is_class or t in (
                "class_declaration", "interface_declaration",
                "record_declaration", "enum_declaration"))

    walk(root, False)
    return (bc, bu, wc)


# ---------------------------------------------------------------------------
# JS / TS
# ---------------------------------------------------------------------------
def _tally_js_like(code: bytes, lang: str) -> Tuple[int, int, int]:
    """(builder_class, fluent_chains, wide_constructors).

    builder_class is always 0 for JS/TS (no idiomatic Builder class name).
    """
    parser = _get_parser(lang)
    if parser is None:
        return (0, 0, 0)
    root = parser.parse(code).root_node
    bu = wc = 0
    # to avoid double-counting nested chains, only count the OUTERMOST chain
    # (i.e. a call_expression whose parent is NOT itself a call along the
    # chain spine)
    seen_chain_roots = set()

    def chain_length(call_node) -> int:
        """Count consecutive call_expression nodes whose callee is a
        member_expression, walking down the receiver spine."""
        depth = 0
        cur = call_node
        while cur is not None and cur.type == "call_expression":
            callee = cur.child_by_field_name("function")
            if callee is None or callee.type != "member_expression":
                break
            depth += 1
            recv = callee.child_by_field_name("object")
            cur = recv
        return depth

    def is_chain_root(call_node) -> bool:
        """True if this call_expression is the top of a member-call chain
        (its parent is NOT a member_expression whose parent is a
        call_expression continuing the chain)."""
        p = call_node.parent
        if p is None:
            return True
        # If parent is a member_expression and grandparent is a call_expression
        # using that member, then this call is mid-chain (it's the receiver).
        if p.type == "member_expression":
            gp = p.parent
            if gp is not None and gp.type == "call_expression":
                # is call_node the .object of the member_expression?
                obj = p.child_by_field_name("object")
                if obj is not None and (obj.start_byte, obj.end_byte) == (
                        call_node.start_byte, call_node.end_byte):
                    return False
        return True

    def walk(n):
        nonlocal bu, wc
        t = n.type
        if t == "call_expression":
            if is_chain_root(n):
                d = chain_length(n)
                if d >= 3:
                    bu += 1
        if t == "method_definition":
            name = n.child_by_field_name("name")
            if name is not None and _text(name) == "constructor":
                params = n.child_by_field_name("parameters")
                if params is not None:
                    k = sum(1 for c in params.children
                            if c.type in ("required_parameter",
                                          "optional_parameter",
                                          "formal_parameter",
                                          "identifier",
                                          "assignment_pattern",
                                          "rest_pattern",
                                          "object_pattern",
                                          "array_pattern"))
                    if k >= WIDE_PARAM_THRESHOLD:
                        wc += 1
        for c in n.children:
            walk(c)

    walk(root)
    return (0, bu, wc)


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
def _tally_python(code: bytes) -> Tuple[int, int, int]:
    """(builder_class, fluent_methods, wide_inits).

    builder_class: classes with >= 2 methods ending in `return self`.
    fluent_methods: total count of such methods (proxy for "builder usage").
    wide_inits: `__init__`s with >= 4 non-self parameters.
    """
    parser = _get_parser("python")
    if parser is None:
        return (0, 0, 0)
    root = parser.parse(code).root_node
    bc = bu = wc = 0

    def is_return_self(stmt) -> bool:
        if stmt.type != "return_statement":
            return False
        # return_statement -> "return" expression?
        for c in stmt.children:
            if c.type == "identifier" and _text(c) == "self":
                return True
        return False

    def function_returns_self(func_def) -> bool:
        body = func_def.child_by_field_name("body")
        if body is None:
            return False
        # walk children of the function body
        for c in body.children:
            if is_return_self(c):
                return True
            # check `if ... : return self` etc nested one level
            for sub in c.children:
                if is_return_self(sub):
                    return True
        return False

    def walk_class(cls):
        nonlocal bc, bu
        body = cls.child_by_field_name("body")
        if body is None:
            return
        fluent_n = 0
        for c in body.children:
            if c.type == "function_definition":
                if function_returns_self(c):
                    fluent_n += 1
        if fluent_n >= 2:
            bc += 1
        bu += fluent_n

    def walk(n):
        nonlocal wc
        t = n.type
        if t == "class_definition":
            walk_class(n)
        if t == "function_definition":
            name = n.child_by_field_name("name")
            if name is not None and _text(name) == "__init__":
                params = n.child_by_field_name("parameters")
                if params is not None:
                    # subtract `self`
                    k = 0
                    for c in params.children:
                        if c.type in ("identifier", "typed_parameter",
                                      "default_parameter",
                                      "typed_default_parameter",
                                      "list_splat_pattern",
                                      "dictionary_splat_pattern"):
                            if c.type == "identifier" and _text(c) == "self":
                                continue
                            # for typed_parameter, the actual name is a child
                            # but we count it as 1 param regardless
                            k += 1
                    if k >= WIDE_PARAM_THRESHOLD:
                        wc += 1
        for c in n.children:
            walk(c)

    walk(root)
    return (bc, bu, wc)


# ---------------------------------------------------------------------------
# Aggregate across files
# ---------------------------------------------------------------------------
def _collect(diff_text: str) -> Optional[Tuple[int, int, int]]:
    """Sum (builder_signal, candidate, n_files) across supported files.

    builder_signal = builder_class + builder_usage
    candidate      = wide_constructors / wide_inits / classes-with-N-params
    """
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    builder_sig = 0
    candidate = 0
    n_files = 0
    for path, body in by_path.items():
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = _lang_for_ext(ext)
        if lang is None:
            continue
        n_files += 1
        src = body.encode("utf8", errors="replace")
        try:
            if lang == "java":
                bc, bu, wc = _tally_java(src)
            elif lang in ("js", "ts", "tsx"):
                bc, bu, wc = _tally_js_like(src, lang)
            elif lang == "python":
                bc, bu, wc = _tally_python(src)
            else:
                continue
        except Exception:
            # parser blow-up on this snippet — skip rather than crash the run
            continue
        builder_sig += bc + bu
        candidate += wc
    if n_files == 0:
        return None
    return (builder_sig, candidate, n_files)


def applies(diff_text: str) -> bool:
    """Apply when the diff adds a supported-language file AND we observed
    at least one builder signal or one wide-construction candidate.

    Without either, the norm has nothing to say about this diff (no
    construction visible), so abstain by returning False.
    """
    res = _collect(diff_text)
    if res is None:
        return False
    builder_sig, candidate, _ = res
    return (builder_sig + candidate) > 0


def score(diff_text: str) -> Optional[float]:
    res = _collect(diff_text)
    if res is None:
        return None
    builder_sig, candidate, _ = res
    if (builder_sig + candidate) == 0:
        return None
    # Laplace-smoothed ratio: more builder signal relative to candidates -> 1.
    # All candidate, no builder -> trends to 0 as candidate grows.
    s = (builder_sig + 1) / (builder_sig + candidate + 1)
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return float(s)
