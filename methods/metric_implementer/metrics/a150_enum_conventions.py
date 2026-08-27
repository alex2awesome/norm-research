"""a150: Enum conventions and constraints.

Detects enum declarations in added code across Python, Java, TypeScript, and
Go, and checks per-language conventions. The norm being measured is "are the
enums added by this diff named/structured idiomatically for the language?"

Per-language rules used (deliberately conservative — every check is a *strong*
convention violation that experienced reviewers would flag, not a stylistic
preference):

Python (PEP 8 + `enum` module docs):
  - Enum *class* name is PascalCase.
  - Members are UPPER_SNAKE (per PEP 8 §"Constants" + the `enum` docs).
  - Do not mix `auto()` and explicit integer values in the same enum —
    Python supports it syntactically but the resulting values become a
    foot-gun (PEP 663 motivation; `enum` documentation §"Using auto").

Java (JLS §8.9 + Effective Java Item 34):
  - Enum *type* name is PascalCase (it is a class).
  - Enum *constants* are UPPER_SNAKE.

TypeScript (TS handbook §"Enums"):
  - Enum name is PascalCase.
  - Members are PascalCase by the handbook, but a substantial fraction of
    real codebases use UPPER_SNAKE. We accept EITHER PascalCase OR
    UPPER_SNAKE per member — and we *do* require them to be consistent
    within a single enum (no mix).

Go (Effective Go §"Constants", commonly-cited iota pattern):
  - The Go enum idiom is a `const ( ... iota ... )` block. A const block
    of typed identifiers with no `iota` and where all values are typed
    integer literals 0,1,2,... is "manual iota" and we flag it (the
    idiomatic form uses `iota`). Pure manual constants without a shared
    type are not enums and we don't classify them.
  - Identifier shape follows Go's MixedCaps (no underscores).

Scoring: each detected enum contributes [0,1] = fraction-of-checks-passed.
Metric score = mean over all enums detected in the diff. Abstains when no
enum was detected (the common case — most diffs don't add enums).

This is narrow-applicability by design: the value of the metric is that
when it does fire, the signal is sharp.
"""
from __future__ import annotations

# REGEX_OK: tool_output — pure character-class identifier-string predicates
# (snake/camel/Pascal). Identifier strings are not source code; regex is the
# right tool for character-class membership tests on a name.
import re
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a150"
ASPECT_NAME = "Enum conventions and constraints"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-typescript",
         "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "Java", "TypeScript", "Go"]
CLASSIFICATION = "THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# REGEX_OK: tool_output — identifier-string character-class predicates.
PASCAL_CASE = re.compile(r"^_?[A-Z][a-zA-Z0-9]*$")
# REGEX_OK: tool_output — identifier-string character-class predicate.
UPPER_SNAKE = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
# REGEX_OK: tool_output — identifier-string character-class predicate
# (Go MixedCaps: no underscores).
GO_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "py":
            import tree_sitter_python as m
            L = m.language()
        elif lang == "ts":
            import tree_sitter_typescript as m
            L = m.language_typescript()
        elif lang == "java":
            import tree_sitter_java as m
            L = m.language()
        elif lang == "go":
            import tree_sitter_go as m
            L = m.language()
        else:
            return None
        _PARSERS[lang] = Parser(Language(L))
        return _PARSERS[lang]
    except ImportError:
        return None


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# ----- Python ----------------------------------------------------------------

def _py_class_bases(class_node, src: bytes) -> List[str]:
    """Return the textual list of base-class names of a Python class
    definition. Handles `class C(Enum):` and `class C(IntEnum, Foo):`.
    """
    for c in class_node.children:
        if c.type == "argument_list":
            out = []
            for cc in c.children:
                if cc.type in ("identifier", "attribute", "dotted_name"):
                    out.append(_text(cc, src))
            return out
    return []


# All standard-library enum subclasses worth detecting as "an enum".
_PY_ENUM_BASES = {"Enum", "IntEnum", "IntFlag", "Flag", "StrEnum",
                  "ReprEnum", "auto"}


def _py_is_enum_class(class_node, src: bytes) -> bool:
    bases = _py_class_bases(class_node, src)
    for b in bases:
        # `Foo(Enum)`, `Foo(enum.Enum)`, `Foo(IntEnum)`, ...
        leaf = b.rsplit(".", 1)[-1]
        if leaf in _PY_ENUM_BASES:
            return True
    return False


def _py_class_name(class_node, src: bytes) -> Optional[str]:
    for c in class_node.children:
        if c.type == "identifier":
            return _text(c, src)
    return None


def _py_enum_members(class_node, src: bytes) -> List[Tuple[str, str]]:
    """Return [(member_name, value_kind)] for each member of a Python enum.

    value_kind:
      - "auto" if RHS is a call to `auto()` (or `enum.auto()`)
      - "int" if RHS is an integer literal (incl. unary +/-)
      - "other" otherwise (string, tuple, call, ...)
    """
    out: List[Tuple[str, str]] = []
    body = None
    for c in class_node.children:
        if c.type == "block":
            body = c
            break
    if body is None:
        return out
    for stmt in body.children:
        if stmt.type != "expression_statement":
            continue
        for assign in stmt.children:
            if assign.type != "assignment":
                continue
            kids = assign.children
            if len(kids) < 3:
                continue
            lhs = kids[0]
            rhs = kids[-1]
            if lhs.type != "identifier":
                continue
            name = _text(lhs, src)
            kind = "other"
            if rhs.type == "call":
                # call: function ( args ) — look at the function
                fn = rhs.children[0] if rhs.children else None
                if fn is not None:
                    fn_txt = _text(fn, src)
                    if fn_txt.rsplit(".", 1)[-1] == "auto":
                        kind = "auto"
            elif rhs.type == "integer":
                kind = "int"
            elif rhs.type == "unary_operator":
                # +1 / -1
                for sub in rhs.children:
                    if sub.type == "integer":
                        kind = "int"
                        break
            out.append((name, kind))
    return out


def _py_score_enum(class_node, src: bytes) -> Optional[float]:
    name = _py_class_name(class_node, src)
    members = _py_enum_members(class_node, src)
    if name is None:
        return None
    if not members:
        # Empty / unparseable body — abstain on this class.
        return None
    checks: List[bool] = []
    # Check 1: class name PascalCase
    checks.append(bool(PASCAL_CASE.match(name)))
    # Check 2: every member UPPER_SNAKE
    checks.append(all(bool(UPPER_SNAKE.match(m)) for m, _ in members))
    # Check 3: no mixing auto() with explicit int values
    kinds = {k for _, k in members}
    checks.append(not ({"auto", "int"} <= kinds))
    return sum(1 for c in checks if c) / len(checks)


def _py_find_enums(root, src: bytes) -> List[float]:
    scores: List[float] = []

    def walk(node):
        if node.type == "class_definition":
            if _py_is_enum_class(node, src):
                s = _py_score_enum(node, src)
                if s is not None:
                    scores.append(s)
        for c in node.children:
            walk(c)

    walk(root)
    return scores


# ----- Java ------------------------------------------------------------------

def _java_enum_name(enum_node, src: bytes) -> Optional[str]:
    for c in enum_node.children:
        if c.type == "identifier":
            return _text(c, src)
    return None


def _java_enum_members(enum_node, src: bytes) -> List[str]:
    out: List[str] = []
    body = None
    for c in enum_node.children:
        if c.type == "enum_body":
            body = c
            break
    if body is None:
        return out
    for c in body.children:
        if c.type == "enum_constant":
            for cc in c.children:
                if cc.type == "identifier":
                    out.append(_text(cc, src))
                    break
    return out


def _java_score_enum(enum_node, src: bytes) -> Optional[float]:
    name = _java_enum_name(enum_node, src)
    members = _java_enum_members(enum_node, src)
    if name is None or not members:
        return None
    checks: List[bool] = []
    checks.append(bool(PASCAL_CASE.match(name)))
    checks.append(all(bool(UPPER_SNAKE.match(m)) for m in members))
    return sum(1 for c in checks if c) / len(checks)


def _java_find_enums(root, src: bytes) -> List[float]:
    scores: List[float] = []

    def walk(node):
        if node.type == "enum_declaration":
            s = _java_score_enum(node, src)
            if s is not None:
                scores.append(s)
        for c in node.children:
            walk(c)

    walk(root)
    return scores


# ----- TypeScript -----------------------------------------------------------

def _ts_enum_name(enum_node, src: bytes) -> Optional[str]:
    for c in enum_node.children:
        if c.type in ("identifier", "type_identifier"):
            return _text(c, src)
    return None


def _ts_enum_members(enum_node, src: bytes) -> List[str]:
    """Return list of member names. Tree-sitter-typescript represents enum
    members under `enum_body` with `property_identifier` (and assignment) or
    `enum_assignment` nodes depending on version. Be tolerant.
    """
    out: List[str] = []
    body = None
    for c in enum_node.children:
        if c.type == "enum_body":
            body = c
            break
    if body is None:
        return out
    for c in body.children:
        if c.type == "property_identifier":
            out.append(_text(c, src))
        elif c.type == "enum_assignment":
            # enum_assignment: name = value
            for cc in c.children:
                if cc.type == "property_identifier":
                    out.append(_text(cc, src))
                    break
        elif c.type == "identifier":
            out.append(_text(c, src))
    return out


def _ts_score_enum(enum_node, src: bytes) -> Optional[float]:
    name = _ts_enum_name(enum_node, src)
    members = _ts_enum_members(enum_node, src)
    if name is None or not members:
        return None
    checks: List[bool] = []
    # Check 1: enum name PascalCase
    checks.append(bool(PASCAL_CASE.match(name)))
    # Check 2: every member matches a single convention across the enum
    all_pascal = all(bool(PASCAL_CASE.match(m)) for m in members)
    all_upper = all(bool(UPPER_SNAKE.match(m)) for m in members)
    checks.append(all_pascal or all_upper)
    return sum(1 for c in checks if c) / len(checks)


def _ts_find_enums(root, src: bytes) -> List[float]:
    scores: List[float] = []

    def walk(node):
        if node.type == "enum_declaration":
            s = _ts_score_enum(node, src)
            if s is not None:
                scores.append(s)
        for c in node.children:
            walk(c)

    walk(root)
    return scores


# ----- Go --------------------------------------------------------------------

def _go_const_block_is_enum(const_node, src: bytes) -> Optional[bool]:
    """Heuristic: a `const ( ... )` block is "enum-like" iff it declares at
    least two identifiers that share an explicit type AND at least one specs
    uses `iota` OR all the RHS values are integer literals 0,1,2,... in
    order. Returns:
      - True if uses iota (idiomatic Go enum)
      - False if "manual iota" (typed constants enumerating 0,1,2,...)
      - None if not enum-like at all
    """
    typed_specs: List[Tuple[List[str], Optional[str], Optional[str]]] = []
    # (names, type_text, value_text)
    for c in const_node.children:
        if c.type != "const_spec":
            continue
        names: List[str] = []
        type_text: Optional[str] = None
        value_text: Optional[str] = None
        seen_eq = False
        for cc in c.children:
            if cc.type == "identifier":
                if not seen_eq and type_text is None:
                    names.append(_text(cc, src))
            elif cc.type == "type_identifier":
                type_text = _text(cc, src)
            elif cc.type == "=":
                seen_eq = True
            elif seen_eq and value_text is None:
                value_text = _text(cc, src).strip()
        typed_specs.append((names, type_text, value_text))

    # Need at least one specification with a named type to look like an enum.
    has_typed = any(t for _, t, _ in typed_specs)
    if not has_typed:
        return None
    flat_names: List[str] = []
    for names, _, _ in typed_specs:
        flat_names.extend(names)
    if len(flat_names) < 2:
        return None
    # uses iota?
    uses_iota = any(v is not None and ("iota" in v)
                    for _, _, v in typed_specs)
    if uses_iota:
        return True
    # manual iota: every RHS is an integer literal 0,1,2,...
    ints: List[Optional[int]] = []
    for _, _, v in typed_specs:
        if v is None:
            ints.append(None)
            continue
        try:
            ints.append(int(v))
        except ValueError:
            return None  # not even close
    if all(x is not None for x in ints) and ints == list(range(len(ints))):
        return False  # manual iota -- non-idiomatic
    return None


def _go_const_block_names(const_node, src: bytes) -> List[str]:
    out: List[str] = []
    for c in const_node.children:
        if c.type != "const_spec":
            continue
        for cc in c.children:
            if cc.type == "identifier":
                out.append(_text(cc, src))
            elif cc.type in ("=", "type_identifier"):
                break
    return out


def _go_score_enum(const_node, src: bytes, idiomatic: bool) -> Optional[float]:
    names = _go_const_block_names(const_node, src)
    if not names:
        return None
    checks: List[bool] = []
    # Check 1: uses iota idiom (idiomatic vs manual)
    checks.append(idiomatic)
    # Check 2: all identifiers follow Go MixedCaps (no underscores)
    checks.append(all(bool(GO_NAME.match(n)) for n in names))
    return sum(1 for c in checks if c) / len(checks)


def _go_find_enums(root, src: bytes) -> List[float]:
    scores: List[float] = []

    def walk(node):
        if node.type == "const_declaration":
            verdict = _go_const_block_is_enum(node, src)
            if verdict is not None:
                s = _go_score_enum(node, src, idiomatic=verdict)
                if s is not None:
                    scores.append(s)
        for c in node.children:
            walk(c)

    walk(root)
    return scores


# ----- Dispatch --------------------------------------------------------------

def _file_scores(code: bytes, lang: str) -> List[float]:
    parser = _get_parser(lang)
    if parser is None:
        return []
    tree = parser.parse(code)
    root = tree.root_node
    if lang == "py":
        return _py_find_enums(root, code)
    if lang == "java":
        return _java_find_enums(root, code)
    if lang == "ts":
        return _ts_find_enums(root, code)
    if lang == "go":
        return _go_find_enums(root, code)
    return []


def _path_lang(path: str) -> Optional[str]:
    p = path.lower()
    for ext, lang in EXT_TO_LANG.items():
        if p.endswith(ext):
            return lang
    return None


def _cheap_hint(text: str, lang: str) -> bool:
    """Fast pre-filter: does this added-content blob plausibly declare an
    enum? Cheaper than tree-sitter parsing if all you want is router-gating.
    """
    if lang == "py":
        # `class X(Enum)` / `(IntEnum)` / `(IntFlag)` / `(StrEnum)` / `(Flag)`
        return ("Enum" in text or "IntFlag" in text or "StrEnum" in text
                or "Flag)" in text)
    if lang == "java":
        return "enum " in text
    if lang == "ts":
        return "enum " in text
    if lang == "go":
        # const block with iota OR typed const declarations
        return ("iota" in text) or ("const (" in text) or ("const(" in text)
    return False


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    for path, content in by_path.items():
        lang = _path_lang(path)
        if lang is None:
            continue
        if _cheap_hint(content, lang):
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    all_scores: List[float] = []
    for path, content in by_path.items():
        lang = _path_lang(path)
        if lang is None:
            continue
        if not _cheap_hint(content, lang):
            continue
        scs = _file_scores(content.encode("utf8", errors="replace"), lang)
        all_scores.extend(scs)
    if not all_scores:
        return None
    return float(sum(all_scores) / len(all_scores))
