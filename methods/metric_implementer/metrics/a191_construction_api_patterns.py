"""a191: Construction and initialization API patterns.

The norm asks for clear, idiomatic construction: align Default/new behavior,
offer constructors/factories/builders where appropriate, support fluent /
object-initializer idioms, prefer simple literals over verbose constructors.

**Relation to a99.** a99 measures **builder pattern presence vs wide
constructor candidates** — i.e. whether complex construction has a builder.
a191 is broader and orthogonal: it measures **how clean the construction
API itself is**, independent of whether a builder exists. Specifically:

  1. **Telescoping constructors** — multiple constructors in the same class
     with monotonically growing parameter lists. This is the canonical
     anti-pattern motivating builders/factories. We count classes that have
     telescoping ctors.
  2. **Named factory methods** — `static of(...)`, `static create(...)`,
     `static from*(...)`, `static newInstance(...)`, Python `@classmethod`s
     named `from_*` / `of_*`. Their presence indicates a clearer
     construction API than raw `new Foo(a,b,c,d)`.
  3. **Default / optional parameters** — in JS/TS `function(a, b = 1)`,
     TS `b?: T`; in Python `def __init__(self, a, b=1)`. Use of these
     compresses what would otherwise be telescoping ctors into a single
     entry point — a positive signal.
  4. **Wide raw constructors** — `>=5` formal parameters with no defaults
     and no neighboring factory. A negative signal.

Score per language family:

    positives  = factory_methods + classes_with_defaults
    negatives  = telescoping_classes + wide_raw_constructors

    s = (positives + 1) / (positives + negatives + 1)        # Laplace
        clipped to [0,1]

A diff with several factory methods and default-parameter ctors trends to
1; a diff that adds 4-arg / 5-arg / 6-arg constructors with no defaults
or factories trends to 0.

`applies()` requires (a) at least one supported-language file in the diff
AND (b) at least one constructor / factory / wide-init signal — otherwise
the norm is silent about the diff.

Classification
--------------
PARTIALLY_THIN. We can structurally detect telescoping constructors,
factory methods, and default parameters with tree-sitter. We cannot
deterministically decide *appropriateness* (judgment); we measure the
prevalence of patterns that the norm rewards/penalizes. THICK overlap
with a99 is bounded: telescoping detection and factory detection are
distinct from a99's builder-class / fluent-chain / wide-ctor counts.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a191"
ASPECT_NAME = "Construction and initialization API patterns"
TIER = 2
TOOLS = ["tree-sitter-java", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-python"]
APPLIES_TO_LANGS = ["Java", "JavaScript", "TypeScript", "Python"]
CLASSIFICATION = "PARTIALLY_THIN"

JAVA_EXTS = (".java",)
JS_EXTS = (".js", ".jsx", ".mjs", ".cjs")
TS_EXTS = (".ts", ".tsx")
PY_EXTS = (".py", ".pyi")

WIDE_RAW_THRESHOLD = 5  # > a99's 4 — a191 only flags truly verbose ctors
FACTORY_NAME_PREFIXES = (
    "of", "from", "create", "make", "new", "build", "valueof", "parse",
    "withinstance", "newinstance", "get",  # `getInstance` etc.
)

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


def _looks_like_factory_name(name: str) -> bool:
    """A static method whose name suggests it returns an instance.

    We treat the common families: of/from/create/make/newInstance/getInstance/
    parse/valueOf. Case-insensitive on the prefix; the method must start with
    the prefix (so `getName` is excluded because `name` is not the suffix of
    a factory-style name — but `getInstance` matches `get` and is fine).
    """
    if not name:
        return False
    low = name.lower()
    # exact common factory names
    if low in {"of", "from", "create", "make", "newinstance", "getinstance",
              "valueof", "parse", "build"}:
        return True
    # camelCase factory: `createXxx`, `fromXxx`, `ofXxx`, `newXxx`, `parseXxx`
    for p in ("create", "from", "of", "make", "valueof", "parse"):
        if low.startswith(p) and len(low) > len(p) and name[len(p)].isupper():
            return True
    # `newInstance`, `newBuilder`
    if low.startswith("new") and len(low) > 3 and name[3].isupper():
        return True
    return False


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------
def _has_modifier(node, kw: str) -> bool:
    """Check tree-sitter-java modifiers (modifier nodes are children)."""
    for c in node.children:
        if c.type == "modifiers":
            if kw in _text(c).split():
                return True
    return False


def _tally_java(code: bytes) -> Dict[str, int]:
    parser = _get_parser("java")
    if parser is None:
        return {"factory": 0, "telescoping": 0, "wide_raw": 0,
                "defaults": 0, "ctors": 0}
    # Wrap loose snippets
    if (b"class " not in code and b"interface " not in code
            and b"record " not in code and b"enum " not in code):
        code = b"class __Snip {\n" + code + b"\n}\n"
    root = parser.parse(code).root_node

    out = {"factory": 0, "telescoping": 0, "wide_raw": 0,
           "defaults": 0, "ctors": 0}

    def param_count(params_node) -> int:
        if params_node is None:
            return 0
        return sum(1 for c in params_node.children
                   if c.type == "formal_parameter")

    def walk_class(cls):
        body = cls.child_by_field_name("body")
        if body is None:
            return
        ctor_param_counts: List[int] = []
        for ch in body.children:
            if ch.type == "constructor_declaration":
                params = ch.child_by_field_name("parameters")
                k = param_count(params)
                ctor_param_counts.append(k)
                out["ctors"] += 1
                if k >= WIDE_RAW_THRESHOLD:
                    out["wide_raw"] += 1
            elif ch.type == "method_declaration":
                # static factory?
                if _has_modifier(ch, "static"):
                    name = ch.child_by_field_name("name")
                    nm = _text(name) if name is not None else ""
                    if _looks_like_factory_name(nm):
                        out["factory"] += 1
        # Telescoping: 2+ constructors whose param counts form a strictly
        # increasing chain (after dedup-sort). This catches the canonical
        # Foo(), Foo(a), Foo(a, b), Foo(a, b, c) pattern.
        if len(ctor_param_counts) >= 2:
            uniq = sorted(set(ctor_param_counts))
            if len(uniq) >= 2 and uniq[-1] - uniq[0] >= 2:
                out["telescoping"] += 1
        # Recurse into nested classes
        for ch in body.children:
            walk(ch)

    def walk(n):
        t = n.type
        if t in ("class_declaration", "interface_declaration",
                 "enum_declaration", "record_declaration"):
            walk_class(n)
            return
        for c in n.children:
            walk(c)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# JS / TS
# ---------------------------------------------------------------------------
def _tally_js_like(code: bytes, lang: str) -> Dict[str, int]:
    parser = _get_parser(lang)
    if parser is None:
        return {"factory": 0, "telescoping": 0, "wide_raw": 0,
                "defaults": 0, "ctors": 0}
    root = parser.parse(code).root_node
    out = {"factory": 0, "telescoping": 0, "wide_raw": 0,
           "defaults": 0, "ctors": 0}

    PARAM_KINDS = ("required_parameter", "optional_parameter",
                   "formal_parameter", "identifier",
                   "assignment_pattern", "rest_pattern",
                   "object_pattern", "array_pattern")

    def is_optional_or_default(p) -> bool:
        # optional_parameter (TS): `x?: T` or `x: T = ...`
        if p.type == "optional_parameter":
            return True
        if p.type == "assignment_pattern":
            return True
        # required_parameter with a value field (default value)
        if p.type == "required_parameter":
            for c in p.children:
                if c.type == "assignment_pattern":
                    return True
                # tree-sitter-typescript: required_parameter has 'value' field
            try:
                if p.child_by_field_name("value") is not None:
                    return True
            except Exception:
                pass
        return False

    def param_counts(params_node):
        if params_node is None:
            return 0, 0  # total, defaults
        total = 0
        defaults = 0
        for c in params_node.children:
            if c.type in PARAM_KINDS:
                total += 1
                if is_optional_or_default(c):
                    defaults += 1
        return total, defaults

    def class_body_walk(cls_body):
        ctor_param_counts: List[int] = []
        for ch in cls_body.children:
            if ch.type == "method_definition":
                name = ch.child_by_field_name("name")
                nm = _text(name) if name is not None else ""
                params = ch.child_by_field_name("parameters")
                total, defaults = param_counts(params)
                if nm == "constructor":
                    ctor_param_counts.append(total)
                    out["ctors"] += 1
                    if total >= WIDE_RAW_THRESHOLD and defaults == 0:
                        out["wide_raw"] += 1
                    if defaults > 0:
                        out["defaults"] += 1
                else:
                    # static factory? In JS/TS, static methods are marked by
                    # a 'static' keyword child of method_definition.
                    is_static = any(
                        c.type == "static" or _text(c) == "static"
                        for c in ch.children if c.type not in
                        ("property_identifier", "formal_parameters",
                         "statement_block"))
                    if is_static and _looks_like_factory_name(nm):
                        out["factory"] += 1
        # ctor overloading is rare in JS but common in TS (signatures).
        # Count method_signature children with name 'constructor'.
        for ch in cls_body.children:
            if ch.type in ("method_signature", "construct_signature",
                           "abstract_method_signature"):
                # TS overload signatures
                name = ch.child_by_field_name("name")
                nm = _text(name) if name is not None else ""
                if nm == "constructor" or ch.type == "construct_signature":
                    params = ch.child_by_field_name("parameters")
                    total, _ = param_counts(params)
                    ctor_param_counts.append(total)
        if len(ctor_param_counts) >= 2:
            uniq = sorted(set(ctor_param_counts))
            if len(uniq) >= 2 and uniq[-1] - uniq[0] >= 2:
                out["telescoping"] += 1

    def walk(n):
        if n.type in ("class_declaration", "class", "class_expression"):
            body = n.child_by_field_name("body")
            if body is not None:
                class_body_walk(body)
        # Top-level function with default params still counts as
        # initialization API (e.g. factory functions exported from a module)
        if n.type in ("function_declaration", "arrow_function",
                      "function_expression"):
            name = n.child_by_field_name("name")
            nm = _text(name) if name is not None else ""
            params = n.child_by_field_name("parameters")
            total, defaults = param_counts(params)
            # treat as factory if the name pattern fits AND it returns
            # something (heuristic: any function with factory-like name
            # at top level)
            if nm and _looks_like_factory_name(nm):
                out["factory"] += 1
            if defaults > 0 and total >= 2:
                out["defaults"] += 1
            if total >= WIDE_RAW_THRESHOLD and defaults == 0 and nm:
                out["wide_raw"] += 1
        for c in n.children:
            walk(c)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
def _tally_python(code: bytes) -> Dict[str, int]:
    parser = _get_parser("python")
    if parser is None:
        return {"factory": 0, "telescoping": 0, "wide_raw": 0,
                "defaults": 0, "ctors": 0}
    root = parser.parse(code).root_node
    out = {"factory": 0, "telescoping": 0, "wide_raw": 0,
           "defaults": 0, "ctors": 0}

    PARAM_KINDS = ("identifier", "typed_parameter", "default_parameter",
                   "typed_default_parameter", "list_splat_pattern",
                   "dictionary_splat_pattern")

    def param_stats(params_node) -> Tuple[int, int]:
        """Return (count_excl_self, defaults_count)."""
        if params_node is None:
            return 0, 0
        total = 0
        defaults = 0
        for c in params_node.children:
            if c.type not in PARAM_KINDS:
                continue
            if c.type == "identifier" and _text(c) == "self":
                continue
            total += 1
            if c.type in ("default_parameter", "typed_default_parameter"):
                defaults += 1
        return total, defaults

    def has_decorator(func_node, name: str) -> bool:
        # function_definition's parent is decorated_definition when decorated
        p = func_node.parent
        if p is None or p.type != "decorated_definition":
            return False
        for c in p.children:
            if c.type == "decorator":
                # decorator -> "@" + expression
                txt = _text(c).strip().lstrip("@").split("(")[0].strip()
                if txt == name or txt.endswith("." + name):
                    return True
        return False

    def walk_class(cls):
        body = cls.child_by_field_name("body")
        if body is None:
            return
        # Telescoping in Python is rare (no overloading) but sometimes
        # simulated with @overload or by adding *args. Mainly we just
        # measure __init__ width and presence of @classmethod factories.
        for stmt in body.children:
            # decorated_definition wraps function_definition
            target = stmt
            if stmt.type == "decorated_definition":
                for c in stmt.children:
                    if c.type == "function_definition":
                        target = c
                        break
            if target.type != "function_definition":
                continue
            name = target.child_by_field_name("name")
            nm = _text(name) if name is not None else ""
            params = target.child_by_field_name("parameters")
            total, defaults = param_stats(params)
            if nm == "__init__":
                out["ctors"] += 1
                if total >= WIDE_RAW_THRESHOLD and defaults == 0:
                    out["wide_raw"] += 1
                if defaults > 0:
                    out["defaults"] += 1
            else:
                # classmethod factories (`from_dict`, `of`, `create_*`)
                if (has_decorator(target, "classmethod")
                        and (nm.startswith("from_") or nm.startswith("of_")
                             or nm.startswith("create_")
                             or nm.startswith("make_")
                             or nm in ("of", "from", "create", "make"))):
                    out["factory"] += 1
                # Module-level factories (caught in walk below)

    def walk(n):
        if n.type == "class_definition":
            walk_class(n)
        # module-level factory functions
        if n.type == "function_definition" and (n.parent is None
                                                or n.parent.type
                                                in ("module", "block",
                                                    "decorated_definition")):
            name = n.child_by_field_name("name")
            nm = _text(name) if name is not None else ""
            if nm and (nm.startswith("create_") or nm.startswith("make_")
                       or nm.startswith("from_") or nm.startswith("new_")
                       or nm in ("create", "make")):
                # only count when not inside a class (walk_class handles those)
                p = n.parent
                in_class = False
                while p is not None:
                    if p.type == "class_definition":
                        in_class = True
                        break
                    p = p.parent
                if not in_class:
                    out["factory"] += 1
        for c in n.children:
            walk(c)

    walk(root)
    return out


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def _collect(diff_text: str) -> Optional[Dict[str, int]]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    agg = {"factory": 0, "telescoping": 0, "wide_raw": 0,
           "defaults": 0, "ctors": 0, "n_files": 0}
    for path, body in by_path.items():
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = _lang_for_ext(ext)
        if lang is None:
            continue
        agg["n_files"] += 1
        src = body.encode("utf8", errors="replace")
        try:
            if lang == "java":
                t = _tally_java(src)
            elif lang in ("js", "ts", "tsx"):
                t = _tally_js_like(src, lang)
            elif lang == "python":
                t = _tally_python(src)
            else:
                continue
        except Exception:
            continue
        for k in ("factory", "telescoping", "wide_raw", "defaults", "ctors"):
            agg[k] += t.get(k, 0)
    if agg["n_files"] == 0:
        return None
    return agg


def applies(diff_text: str) -> bool:
    res = _collect(diff_text)
    if res is None:
        return False
    # Norm has something to say iff we saw any ctor, factory, or default-init
    # signal in supported-language files.
    return (res["ctors"] + res["factory"] + res["defaults"]
            + res["wide_raw"] + res["telescoping"]) > 0


def score(diff_text: str) -> Optional[float]:
    res = _collect(diff_text)
    if res is None:
        return None
    positives = res["factory"] + res["defaults"]
    # Heavy negatives: telescoping ctors and very wide raw ctors are direct
    # anti-patterns under this norm.
    heavy_neg = res["telescoping"] + res["wide_raw"]
    # Soft baseline: every raw constructor without any default parameter is
    # a mild "verbose construction surface" signal. A class offering only a
    # narrow defaultless ctor is mildly less clean than one offering a
    # factory or default parameters.
    raw_ctors_no_defaults = max(0, res["ctors"] - res["defaults"])
    soft_neg = 0.5 * raw_ctors_no_defaults
    negatives = heavy_neg + soft_neg
    if (positives + negatives) == 0:
        return None
    s = (positives + 1) / (positives + negatives + 1)
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return float(s)
