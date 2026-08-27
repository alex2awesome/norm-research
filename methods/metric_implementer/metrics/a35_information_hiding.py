"""a35: Information hiding and stable abstractions.

The norm: "Encapsulate likely-to-change decisions behind clear, stable
interfaces; decouple abstraction and implementation so each can vary
independently, exposing explicit contracts."

Distinct from a3 (minimal cohesive interfaces): a3 measures how big a
public surface is and what fraction of members are private (a count-based
encapsulation surrogate). a35 measures a *qualitatively different* failure
mode: even a tiny public surface can leak implementation. The classic
leak is exposing **fields directly** (public mutable state) rather than
behind accessor methods, and **returning concrete implementation types**
in public signatures (`ArrayList` instead of `List`, `HashMap` instead of
`Map`), which couples callers to a specific implementation.

Three deterministic sub-signals (per language):

  (1) PUBLIC_FIELD_RATIO
      In Java/TS/Python, count public *data members* exposed directly on a
      class vs total public class members (fields + methods). A class with
      `public int counter;` next to a no-getter API leaks state. The score
      is 1 - (public_fields / public_members), with public_fields=0 -> 1.0.

      Mutable-but-final public fields (Java `public static final` constants,
      `final` references) are NOT counted as leaks — they are part of the
      stable contract. Same for TS `readonly` modifiers and Python class
      variables annotated with `Final[...]`.

  (2) ACCESSOR_PRESENCE
      In Java, if any non-final public field exists, the encapsulated form
      provides paired `getX/setX` (or `isX`). Score down per public field
      whose name has no matching accessor in the same class. For Python,
      the @property idiom is the encapsulated form; we credit any property.

  (3) CONCRETE_RETURN_TYPES (Java/TS only)
      Walk public method return-type annotations and penalize concrete
      collection types (`ArrayList`, `LinkedList`, `HashMap`, `TreeMap`,
      `HashSet`, `Vector`) that should be their abstract interface
      (`List`, `Map`, `Set`). Returning the abstract type means the caller
      can't tell whether you switch to a different impl tomorrow -- that's
      the "abstractions can vary independently of implementation" clause
      of the norm.

      Per-class score is `n_abstract_returns / n_returns_we_can_classify`
      among public methods; classes with no classifiable return are
      skipped (don't pull the score down).

Per class: mean of the sub-signals that apply (PUBLIC_FIELD_RATIO always
applies once a class is seen; ACCESSOR_PRESENCE only when there are
public mutable fields; CONCRETE_RETURN_TYPES only when annotations exist).

File score = mean across classes. Diff score = mean across files.

Abstain when:
  - No class/interface body is added to the diff.
  - The only added classes contain neither fields nor methods (e.g.
    empty marker classes).

Classification: PARTIALLY_THIN.
  - PUBLIC_FIELD_RATIO and ACCESSOR_PRESENCE are directly observable from
    structure — that part is thin.
  - CONCRETE_RETURN_TYPES is a heuristic mapping from a hardcoded list of
    concrete types to their abstract supertypes; it misses user-defined
    "leaky" types (e.g. exposing a `RepositoryImpl` instead of a
    `Repository` interface) which would require whole-program type
    resolution. So the metric is honest but partial.

Overlap with a3 (acknowledged): a3 already covers "tiny public surface".
This metric explicitly does NOT count public-method count or arity — that
is a3's job. We focus only on the SHAPE of what is exposed (field vs
method, concrete vs abstract type), not its size.
"""
from __future__ import annotations

# REGEX_OK: tool_output — these patterns probe identifier STRINGS (the names
# pulled out of an AST node), not source code itself. We use a tree-sitter
# walk for all structural decisions.
import re
from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a35"
ASPECT_NAME = "Information hiding and stable abstractions"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java"]
CLASSIFICATION = "PARTIALLY_THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
}

# Concrete collection types in Java that have an abstract interface.
JAVA_CONCRETE = {
    "ArrayList", "LinkedList", "Vector", "Stack",
    "HashMap", "TreeMap", "LinkedHashMap", "Hashtable", "ConcurrentHashMap",
    "HashSet", "TreeSet", "LinkedHashSet",
    "ArrayDeque", "PriorityQueue",
}
JAVA_ABSTRACT = {
    "List", "Map", "Set", "Collection", "Queue", "Deque", "Iterable",
    "SortedMap", "SortedSet", "NavigableMap", "NavigableSet",
    "ConcurrentMap",
}

# TS concrete vs abstract.
TS_CONCRETE = {"Array", "Map", "Set", "WeakMap", "WeakSet"}
# In TS, `ReadonlyArray<T>`, `Iterable<T>`, `Iterator<T>` are stable contracts.
TS_ABSTRACT = {"ReadonlyArray", "ReadonlyMap", "ReadonlySet",
               "Iterable", "Iterator", "AsyncIterable",
               "Readonly", "Partial", "Pick", "Record"}

# REGEX_OK: tool_output — recognize accessor name shapes once we have the
# raw method-name string from the AST. (Names, not source.)
_JAVA_GETTER = re.compile(r"^(?:get|is|has)([A-Z][A-Za-z0-9_]*)$")
_JAVA_SETTER = re.compile(r"^set([A-Z][A-Za-z0-9_]*)$")

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "py":
            import tree_sitter_python as m
            L = m.language()
        elif lang == "js":
            import tree_sitter_javascript as m
            L = m.language()
        elif lang == "ts":
            import tree_sitter_typescript as m
            L = m.language_typescript()
        elif lang == "java":
            import tree_sitter_java as m
            L = m.language()
        else:
            return None
        _PARSERS[lang] = Parser(Language(L))
        return _PARSERS[lang]
    except ImportError:
        return None


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# ----- shared dataclass-ish shape ------------------------------------------

class ClassReport:
    """Per-class accumulator. Each sub-signal is Optional[float] in [0,1];
    None means the sub-signal didn't apply."""

    __slots__ = ("public_field_ratio", "accessor_presence",
                 "abstract_return_ratio")

    def __init__(self):
        self.public_field_ratio: Optional[float] = None
        self.accessor_presence: Optional[float] = None
        self.abstract_return_ratio: Optional[float] = None

    def score(self) -> Optional[float]:
        parts = [v for v in (self.public_field_ratio,
                             self.accessor_presence,
                             self.abstract_return_ratio) if v is not None]
        if not parts:
            return None
        return sum(parts) / len(parts)


# ----- Python ---------------------------------------------------------------

def _py_class_body(node):
    for c in node.children:
        if c.type == "block":
            return c
    return None


def _py_classes(root, src: bytes) -> List[ClassReport]:
    """Python info-hiding signals:
      - public field := class-level assignment whose LHS is identifier not
        starting with '_' AND not annotated `Final[...]` AND not assigned
        a function/lambda.
      - public method := function_definition whose name does not start '_'
      - property := function_definition decorated with @property
      Accessor presence: a property accessor "counts" for any same-named
      public field (rare in Python idiom but allows accurate measurement
      where users still write Java-style getters via @property).
    """
    out: List[ClassReport] = []

    def gather(class_node) -> ClassReport:
        body = _py_class_body(class_node)
        rep = ClassReport()
        if body is None:
            return rep

        public_fields: Set[str] = set()
        public_methods: int = 0
        properties: Set[str] = set()

        def is_function_value(rhs) -> bool:
            return rhs is not None and rhs.type in (
                "lambda", "function_definition")

        def has_final_annotation(annot_node) -> bool:
            if annot_node is None:
                return False
            txt = _text(annot_node, src)
            return "Final" in txt

        for c in body.children:
            t = c.type
            # Plain assignment at class level.
            if t == "expression_statement":
                for cc in c.children:
                    if cc.type == "assignment":
                        lhs = cc.children[0] if cc.children else None
                        rhs = cc.children[-1] if len(cc.children) >= 3 else None
                        if lhs is None or lhs.type != "identifier":
                            continue
                        name = _text(lhs, src)
                        if name.startswith("_"):
                            continue
                        if is_function_value(rhs):
                            continue
                        public_fields.add(name)
            # Type-annotated class attribute: `counter: int` or
            # `counter: Final[int] = 0`.
            elif t == "expression_statement":
                pass  # handled above
            elif t == "function_definition":
                # collect name & decorators
                name = None
                decorated_property = False
                for cc in c.children:
                    if cc.type == "identifier" and name is None:
                        name = _text(cc, src)
                if name is None:
                    continue
                if name.startswith("_"):
                    continue
                public_methods += 1
            elif t == "decorated_definition":
                # decorators come as children of type "decorator"
                dec_texts: List[str] = []
                fdef = None
                for cc in c.children:
                    if cc.type == "decorator":
                        dec_texts.append(_text(cc, src))
                    elif cc.type == "function_definition":
                        fdef = cc
                if fdef is None:
                    continue
                # name from function_definition
                fname = None
                for cc in fdef.children:
                    if cc.type == "identifier":
                        fname = _text(cc, src)
                        break
                if fname is None or fname.startswith("_"):
                    continue
                public_methods += 1
                if any("@property" in d for d in dec_texts):
                    properties.add(fname)
            elif t == "annotated_assignment" or (
                    t == "expression_statement"
                    and any(cc.type == "annotated_assignment"
                            for cc in c.children)):
                node_for_anno = c
                if t == "expression_statement":
                    for cc in c.children:
                        if cc.type == "annotated_assignment":
                            node_for_anno = cc
                            break
                # children: name, ':', type, ['=', value]
                name_node = None
                type_node = None
                value_node = None
                seen_colon = False
                seen_eq = False
                for cc in node_for_anno.children:
                    if cc.type == "identifier" and name_node is None:
                        name_node = cc
                    elif cc.type == ":":
                        seen_colon = True
                    elif seen_colon and type_node is None and cc.type != "=":
                        type_node = cc
                    elif cc.type == "=":
                        seen_eq = True
                    elif seen_eq and value_node is None:
                        value_node = cc
                if name_node is None:
                    continue
                name = _text(name_node, src)
                if name.startswith("_"):
                    continue
                if has_final_annotation(type_node):
                    continue
                if is_function_value(value_node):
                    continue
                public_fields.add(name)

        n_pub_members = len(public_fields) + public_methods
        if n_pub_members == 0:
            return rep  # empty class
        # (1) PUBLIC_FIELD_RATIO
        rep.public_field_ratio = 1.0 - (len(public_fields) / n_pub_members)

        # (2) ACCESSOR_PRESENCE — Python idiom: @property covers it.
        if public_fields:
            covered = sum(1 for f in public_fields if f in properties)
            rep.accessor_presence = covered / len(public_fields)
        return rep

    def walk(node):
        if node.type == "class_definition":
            out.append(gather(node))
        for c in node.children:
            walk(c)

    walk(root)
    return out


# ----- Java -----------------------------------------------------------------

def _java_modifier_text(node, src: bytes) -> str:
    for c in node.children:
        if c.type == "modifiers":
            return _text(c, src)
    return ""


def _java_classes(root, src: bytes) -> List[ClassReport]:
    out: List[ClassReport] = []

    def class_body(node):
        for c in node.children:
            if c.type in ("class_body", "interface_body", "enum_body",
                          "record_body", "annotation_type_body"):
                return c
        return None

    def gather(class_node, is_interface: bool) -> ClassReport:
        body = class_body(class_node)
        rep = ClassReport()
        if body is None:
            return rep

        # Mutable public fields (writable state).
        mutable_public_fields: List[str] = []
        public_field_count = 0   # all public fields incl. final
        public_method_count = 0
        method_names: Set[str] = set()
        public_return_types: List[str] = []

        for c in body.children:
            t = c.type
            if t == "field_declaration":
                mods = _java_modifier_text(c, src)
                # interface bodies: fields are implicitly public static final
                # (constants) — those are stable contract, not state leaks.
                if is_interface:
                    is_pub = True
                    is_final = True
                else:
                    is_pub = "public" in mods
                    is_final = "final" in mods
                if not is_pub:
                    continue
                # walk declarators for names
                for cc in c.children:
                    if cc.type == "variable_declarator":
                        for ccc in cc.children:
                            if ccc.type == "identifier":
                                nm = _text(ccc, src)
                                public_field_count += 1
                                if not is_final:
                                    mutable_public_fields.append(nm)
                                break
            elif t in ("method_declaration", "constructor_declaration"):
                mods = _java_modifier_text(c, src)
                if is_interface:
                    is_pub = True
                elif "public" in mods:
                    is_pub = True
                else:
                    is_pub = False
                if not is_pub:
                    continue
                public_method_count += 1
                # method name
                mname = None
                for cc in c.children:
                    if cc.type == "identifier":
                        mname = _text(cc, src)
                        break
                if mname is not None:
                    method_names.add(mname)
                # return type (only present on method_declaration, not ctor)
                if t == "method_declaration":
                    ret_txt = _java_return_type(c, src)
                    if ret_txt is not None:
                        public_return_types.append(ret_txt)

        n_pub = public_field_count + public_method_count
        if n_pub == 0:
            return rep

        # (1) PUBLIC_FIELD_RATIO — interface constants are not state leaks,
        # so we exempt them. For classes, ANY public field counts as a leak
        # because callers can read it directly; mutable ones are worse.
        if is_interface:
            rep.public_field_ratio = 1.0
        else:
            # Penalize mutable fields fully; final fields half (still an API
            # commitment to a representation).
            leak_weight = (len(mutable_public_fields)
                           + 0.5 * (public_field_count
                                    - len(mutable_public_fields)))
            denom = float(n_pub)
            rep.public_field_ratio = max(0.0,
                                         1.0 - leak_weight / denom)

        # (2) ACCESSOR_PRESENCE — for each mutable public field, is there a
        # matching getX/setX/isX in the class? Cap at 1.0.
        if mutable_public_fields:
            def has_accessor(field_name: str) -> bool:
                cap = field_name[:1].upper() + field_name[1:]
                candidates = {f"get{cap}", f"set{cap}", f"is{cap}",
                              f"has{cap}"}
                return any(m in method_names for m in candidates)
            covered = sum(1 for f in mutable_public_fields if has_accessor(f))
            rep.accessor_presence = covered / len(mutable_public_fields)

        # (3) CONCRETE_RETURN_TYPES
        classified = 0
        abstract = 0
        for rt in public_return_types:
            head = _strip_generic(rt)
            if head in JAVA_CONCRETE:
                classified += 1
                # leak
            elif head in JAVA_ABSTRACT:
                classified += 1
                abstract += 1
        if classified > 0:
            rep.abstract_return_ratio = abstract / classified

        return rep

    def walk(node):
        if node.type in ("class_declaration", "interface_declaration",
                         "enum_declaration", "record_declaration"):
            is_interface = node.type == "interface_declaration"
            out.append(gather(node, is_interface=is_interface))
        for c in node.children:
            walk(c)

    walk(root)
    return out


def _java_return_type(method_node, src: bytes) -> Optional[str]:
    """Return type comes before the method name on `method_declaration`.
    tree-sitter-java exposes it via children with various 'type' nodes.
    """
    for c in method_node.children:
        if c.type in ("type_identifier", "generic_type",
                      "array_type", "void_type",
                      "integral_type", "floating_point_type",
                      "boolean_type"):
            return _text(c, src)
    return None


def _strip_generic(type_text: str) -> str:
    """`List<String>` -> `List`, `Map<K,V>` -> `Map`. Identifier-string op."""
    type_text = type_text.strip()
    # Drop array brackets
    if type_text.endswith("[]"):
        type_text = type_text[:-2].strip()
    lt = type_text.find("<")
    if lt != -1:
        type_text = type_text[:lt]
    # Drop package qualifier
    if "." in type_text:
        type_text = type_text.rsplit(".", 1)[-1]
    return type_text.strip()


# ----- JS / TS --------------------------------------------------------------

def _ts_member_visibility(node, src: bytes) -> str:
    """Return 'public' | 'private'. TS uses accessibility_modifier; the
    `#name` form and `_name` convention also signal private.
    """
    # name first
    name_txt = ""
    for cc in node.children:
        if cc.type in ("property_identifier", "identifier",
                       "private_property_identifier"):
            name_txt = _text(cc, src)
            break
    if name_txt.startswith("#") or name_txt.startswith("_"):
        return "private"
    for cc in node.children:
        if cc.type == "accessibility_modifier":
            txt = _text(cc, src).strip()
            if txt in ("private", "protected"):
                return "private"
    return "public"


def _ts_member_is_readonly(node, src: bytes) -> bool:
    # `readonly` shows up as its own child token in TS.
    for cc in node.children:
        if cc.type == "readonly" or _text(cc, src).strip() == "readonly":
            return True
    return False


def _ts_member_name(node, src: bytes) -> Optional[str]:
    for cc in node.children:
        if cc.type in ("property_identifier", "identifier",
                       "private_property_identifier"):
            return _text(cc, src)
    return None


def _ts_member_return_type(node, src: bytes) -> Optional[str]:
    # type_annotation appears after parameters for methods/signatures.
    for cc in node.children:
        if cc.type == "type_annotation":
            txt = _text(cc, src).strip()
            if txt.startswith(":"):
                txt = txt[1:].strip()
            return txt
    return None


def _js_classes(root, src: bytes, is_ts: bool) -> List[ClassReport]:
    out: List[ClassReport] = []

    def class_body(node):
        for c in node.children:
            if c.type in ("class_body", "object_type", "interface_body"):
                return c
        return None

    def gather(class_node, is_interface: bool) -> ClassReport:
        body = class_body(class_node)
        rep = ClassReport()
        if body is None:
            return rep

        mutable_public_fields: List[str] = []
        public_field_count = 0
        public_method_count = 0
        method_names: Set[str] = set()
        getter_setter_names: Set[str] = set()
        public_return_types: List[str] = []

        for c in body.children:
            t = c.type
            # Field-like
            if t in ("public_field_definition", "field_definition",
                     "property_signature"):
                if is_interface:
                    vis = "public"
                    is_readonly = _ts_member_is_readonly(c, src)
                else:
                    vis = _ts_member_visibility(c, src)
                    is_readonly = _ts_member_is_readonly(c, src)
                if vis != "public":
                    continue
                name = _ts_member_name(c, src)
                if name is None:
                    continue
                public_field_count += 1
                if not is_readonly:
                    mutable_public_fields.append(name)
            elif t in ("method_definition", "method_signature",
                       "abstract_method_signature"):
                if is_interface:
                    vis = "public"
                else:
                    vis = _ts_member_visibility(c, src)
                if vis != "public":
                    continue
                # JS/TS distinguishes get/set accessors via a child token.
                is_getter_setter = False
                for cc in c.children:
                    txt = _text(cc, src).strip()
                    if cc.type in ("get", "set") or txt in ("get", "set"):
                        is_getter_setter = True
                        break
                public_method_count += 1
                name = _ts_member_name(c, src)
                if name is not None:
                    method_names.add(name)
                    if is_getter_setter:
                        getter_setter_names.add(name)
                if is_ts:
                    rt = _ts_member_return_type(c, src)
                    if rt is not None:
                        public_return_types.append(rt)

        n_pub = public_field_count + public_method_count
        if n_pub == 0:
            return rep

        if is_interface:
            # An interface listing properties IS the abstraction; readonly
            # vs mutable is the meaningful distinction.
            leak = len(mutable_public_fields)
            rep.public_field_ratio = max(0.0,
                                         1.0 - leak / n_pub)
        else:
            leak_weight = (len(mutable_public_fields)
                           + 0.5 * (public_field_count
                                    - len(mutable_public_fields)))
            rep.public_field_ratio = max(0.0,
                                         1.0 - leak_weight / n_pub)

        # (2) Accessor presence — JS/TS get/set accessors directly cover the
        # field name (they share an identifier with the field).
        if mutable_public_fields:
            covered = sum(1 for f in mutable_public_fields
                          if f in getter_setter_names)
            rep.accessor_presence = covered / len(mutable_public_fields)

        # (3) Concrete return types (TS only).
        if is_ts:
            classified = 0
            abstract = 0
            for rt in public_return_types:
                head = _strip_generic(rt)
                if head in TS_CONCRETE:
                    classified += 1
                elif head in TS_ABSTRACT:
                    classified += 1
                    abstract += 1
            if classified > 0:
                rep.abstract_return_ratio = abstract / classified

        return rep

    def walk(node):
        kind = node.type
        if kind in ("class_declaration", "interface_declaration"):
            is_interface = kind == "interface_declaration"
            out.append(gather(node, is_interface=is_interface))
        for c in node.children:
            walk(c)

    walk(root)
    return out


# ----- Dispatch -------------------------------------------------------------

def _path_lang(path: str) -> Optional[str]:
    p = path.lower()
    for ext, lang in EXT_TO_LANG.items():
        if p.endswith(ext):
            return lang
    return None


def _file_reports(code: bytes, lang: str) -> List[ClassReport]:
    parser = _get_parser(lang)
    if parser is None:
        return []
    tree = parser.parse(code)
    root = tree.root_node
    if lang == "py":
        return _py_classes(root, code)
    if lang == "java":
        return _java_classes(root, code)
    if lang == "js":
        return _js_classes(root, code, is_ts=False)
    if lang == "ts":
        return _js_classes(root, code, is_ts=True)
    return []


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
        reps = _file_reports(content.encode("utf8", errors="replace"), lang)
        per_class = [r.score() for r in reps]
        per_class = [s for s in per_class if s is not None]
        if not per_class:
            continue
        file_scores.append(sum(per_class) / len(per_class))
    if not file_scores:
        return None
    return float(sum(file_scores) / len(file_scores))
