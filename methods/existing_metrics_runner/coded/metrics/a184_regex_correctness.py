"""a184: Regular expression correctness and readability.

META-norm: this metric looks at the regex *literals* inside the changed code
and asks whether those regexes are well-formed and free of obvious
catastrophic-backtracking risk. The thing being measured is the diff's own
regexes, not its prose.

What we check, per extracted regex literal:
  1. Compiles under Python's `re` (proxy for "syntactically valid").
     Most regex flavors share the same core grammar; balance/escape errors
     show up regardless of engine.
  2. Catastrophic-backtracking risk: an unbounded quantifier (`+`, `*`,
     `{n,}`) nested inside another unbounded quantifier or alternation.
     This is the (a+)+ family that pumps NFA backtracking exponentially.
     Detected by walking `sre_parse`'s AST, not regex-on-regex.

Extraction is via tree-sitter for each supported language so we never use
regex to find regex:
  - Python: `call` of `re.<func>(<string>, ...)`
  - JavaScript/TypeScript: `regex_pattern` nodes and
    `new RegExp(<string>)` / `RegExp(<string>)`
  - Java: `Pattern.compile(<string>)`, plus String.matches / replaceAll /
    replaceFirst / split (first arg is a regex)
  - Go: `regexp.MustCompile(<string>)`, `regexp.Compile`, `regexp.Match*`

The metric ABSTAINS (applies()=False) when no regex literals appear in the
added lines — most diffs.

Score (mean over found regexes, in [0,1]):
  - 0.0  : did not compile
  - 0.5  : compiled but catastrophic-backtracking risk
  - 1.0  : compiled, no nested-quantifier risk
"""
from __future__ import annotations

# The stdlib regex engine below is our MEASUREMENT TOOL: we validate regex
# literals extracted via tree-sitter; we do NOT use regex to parse code.
import re  # REGEX_OK: tool_output
import sre_constants
import sre_parse
import warnings
from typing import List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a184"
ASPECT_NAME = "Regex correctness and readability"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "THIN"


# ---------- parser cache ----------

_PARSERS = {}


def _parser_for(lang: str):
    """Return cached tree-sitter Parser for language key, or None if missing."""
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "py":
            import tree_sitter_python as mod
            lng = Language(mod.language())
        elif lang == "js":
            import tree_sitter_javascript as mod
            lng = Language(mod.language())
        elif lang == "ts":
            import tree_sitter_typescript as mod
            # tree_sitter_typescript exposes both ts and tsx
            lng = Language(mod.language_typescript())
        elif lang == "java":
            import tree_sitter_java as mod
            lng = Language(mod.language())
        elif lang == "go":
            import tree_sitter_go as mod
            lng = Language(mod.language())
        else:
            _PARSERS[lang] = None
            return None
        _PARSERS[lang] = Parser(lng)
    except Exception:
        _PARSERS[lang] = None
    return _PARSERS[lang]


# ---------- string-literal helpers ----------

def _text(n, src: bytes) -> str:
    return src[n.start_byte:n.end_byte].decode("utf8", errors="replace")


def _py_string_to_pattern(raw: str) -> Optional[str]:
    """Convert a Python source string literal (incl. f-strings, raw strings,
    byte strings, prefixes, triple quotes) to its pattern content.

    Returns None for f-strings (can't statically resolve interpolation).
    """
    s = raw
    # strip prefix
    i = 0
    is_raw = False
    is_f = False
    while i < len(s) and s[i] in "rRbBuUfF":
        if s[i] in "rR":
            is_raw = True
        if s[i] in "fF":
            is_f = True
        i += 1
    if is_f:
        return None
    s = s[i:]
    if s.startswith('"""') or s.startswith("'''"):
        body = s[3:-3]
    elif s.startswith('"') or s.startswith("'"):
        body = s[1:-1]
    else:
        return None
    if is_raw:
        return body
    # decode standard escapes (\n, \t, \xNN, \\)
    try:
        return body.encode("utf8").decode("unicode_escape")
    except Exception:
        return None


def _js_string_to_pattern(raw: str) -> Optional[str]:
    """Convert a JS/TS string literal (single, double, template) to content.

    Templates with ${...} we drop (can't resolve statically).
    """
    if not raw:
        return None
    q = raw[0]
    if q == "`":
        body = raw[1:-1]
        if "${" in body:
            return None
        # process \-escapes
        try:
            return body.encode("utf8").decode("unicode_escape")
        except Exception:
            return body
    if q in ('"', "'"):
        body = raw[1:-1]
        try:
            return body.encode("utf8").decode("unicode_escape")
        except Exception:
            return body
    return None


def _java_string_to_pattern(raw: str) -> Optional[str]:
    """Java strings are double-quoted with backslash escapes (Java syntax)."""
    if not raw.startswith('"'):
        return None
    body = raw[1:-1]
    try:
        return body.encode("utf8").decode("unicode_escape")
    except Exception:
        return body


def _go_string_to_pattern(raw: str) -> Optional[str]:
    """Go has raw backticks (verbatim) and interpreted double quotes."""
    if not raw:
        return None
    if raw.startswith("`"):
        return raw[1:-1]
    if raw.startswith('"'):
        body = raw[1:-1]
        try:
            return body.encode("utf8").decode("unicode_escape")
        except Exception:
            return body
    return None


# ---------- per-language regex extraction ----------

PY_RE_FUNCS = {"compile", "match", "search", "findall", "finditer",
               "fullmatch", "split", "sub", "subn"}


def _extract_py(code: bytes) -> List[str]:
    parser = _parser_for("py")
    if parser is None:
        return []
    root = parser.parse(code).root_node
    out: List[str] = []

    def walk(n):
        if n.type == "call":
            callee = n.child_by_field_name("function")
            args = n.child_by_field_name("arguments")
            if callee is not None and args is not None:
                if callee.type == "attribute":
                    obj = callee.child_by_field_name("object")
                    attr = callee.child_by_field_name("attribute")
                    if (obj is not None and attr is not None and
                            _text(obj, code) == "re" and
                            _text(attr, code) in PY_RE_FUNCS):
                        # first positional string arg
                        for ch in args.children:
                            if ch.type == "string":
                                pat = _py_string_to_pattern(_text(ch, code))
                                if pat is not None:
                                    out.append(pat)
                                break
                            if ch.type not in ("(", ",", ")"):
                                break
        for c in n.children:
            walk(c)
    walk(root)
    return out


def _in_error(n) -> bool:
    """True if any ancestor is a tree-sitter ERROR node — added-lines
    fragments commonly produce these around comments (e.g. `/**` JSDoc
    becomes a spurious `regex_pattern` parented to ERROR)."""
    cur = n.parent
    while cur is not None:
        if cur.type == "ERROR":
            return True
        cur = cur.parent
    return False


def _extract_js(code: bytes, ts: bool = False) -> List[str]:
    parser = _parser_for("ts" if ts else "js")
    if parser is None:
        return []
    root = parser.parse(code).root_node
    out: List[str] = []

    def first_string_arg(args_node):
        for ch in args_node.children:
            if ch.type in ("string", "template_string"):
                return _js_string_to_pattern(_text(ch, code))
            if ch.type not in ("(", ",", ")"):
                return None
        return None

    def walk(n):
        if n.type == "regex_pattern":
            parent = n.parent
            # Filter spurious matches: tree-sitter on added-lines-only often
            # reads `/**` (JSDoc opener) as a regex with a MISSING closing
            # `/`. Require a real, non-error `regex` parent with two `/`
            # delimiters present.
            ok = (parent is not None and parent.type == "regex"
                  and not parent.has_error and not _in_error(n))
            if ok:
                slashes = sum(1 for c in parent.children
                              if c.type == "/" and not c.is_missing)
                if slashes >= 2:
                    out.append(_text(n, code))
        elif n.type in ("new_expression", "call_expression"):
            # RegExp("...") or new RegExp("...")
            callee = n.child_by_field_name("constructor") \
                if n.type == "new_expression" else n.child_by_field_name("function")
            args = n.child_by_field_name("arguments")
            if (callee is not None and args is not None and
                    callee.type == "identifier" and
                    _text(callee, code) == "RegExp"):
                pat = first_string_arg(args)
                if pat is not None:
                    out.append(pat)
        for c in n.children:
            walk(c)
    walk(root)
    return out


# Java methods whose first string arg is a regex.
JAVA_REGEX_METHODS = {"matches", "replaceAll", "replaceFirst", "split"}


def _extract_java(code: bytes) -> List[str]:
    parser = _parser_for("java")
    if parser is None:
        return []
    root = parser.parse(code).root_node
    out: List[str] = []

    def first_string_arg(args_node):
        for ch in args_node.children:
            if ch.type == "string_literal":
                return _java_string_to_pattern(_text(ch, code))
            if ch.type not in ("(", ",", ")"):
                return None
        return None

    def walk(n):
        if n.type == "method_invocation":
            # children: [object?] . name(args) — but tree-sitter-java uses
            # field names: object, name, arguments
            name_node = n.child_by_field_name("name")
            obj_node = n.child_by_field_name("object")
            args = n.child_by_field_name("arguments")
            if name_node is not None and args is not None:
                name = _text(name_node, code)
                obj = _text(obj_node, code) if obj_node is not None else ""
                if name == "compile" and obj == "Pattern":
                    pat = first_string_arg(args)
                    if pat is not None:
                        out.append(pat)
                elif name in JAVA_REGEX_METHODS:
                    # String.matches / replaceAll / replaceFirst / split
                    # we don't try to verify the receiver is a String — too
                    # restrictive without type info; accept if first arg is
                    # a literal string
                    pat = first_string_arg(args)
                    if pat is not None:
                        out.append(pat)
        for c in n.children:
            walk(c)
    walk(root)
    return out


GO_REGEX_FUNCS = {"MustCompile", "Compile", "MustCompilePOSIX", "CompilePOSIX",
                  "MatchString", "Match"}


def _extract_go(code: bytes) -> List[str]:
    parser = _parser_for("go")
    if parser is None:
        return []
    root = parser.parse(code).root_node
    out: List[str] = []

    def first_string_arg(args_node):
        for ch in args_node.children:
            if ch.type == "raw_string_literal":
                return _go_string_to_pattern(_text(ch, code))
            if ch.type == "interpreted_string_literal":
                return _go_string_to_pattern(_text(ch, code))
            if ch.type not in ("(", ",", ")"):
                return None
        return None

    def walk(n):
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            args = n.child_by_field_name("arguments")
            if (fn is not None and args is not None and
                    fn.type == "selector_expression"):
                operand = fn.child_by_field_name("operand")
                field = fn.child_by_field_name("field")
                if (operand is not None and field is not None and
                        _text(operand, code) == "regexp" and
                        _text(field, code) in GO_REGEX_FUNCS):
                    pat = first_string_arg(args)
                    if pat is not None:
                        out.append(pat)
        for c in n.children:
            walk(c)
    walk(root)
    return out


# ---------- regex risk analysis ----------

def _has_nested_unbounded_quantifier(pattern: str) -> bool:
    """Return True if the parsed pattern contains an unbounded quantifier
    nested inside another unbounded quantifier (the classic
    catastrophic-backtracking pattern, e.g. (a+)+, (a*)*, (.+)+).

    Uses Python's stdlib `sre_parse` AST. We swallow DeprecationWarning
    since the module is internal but still ships.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            parsed = sre_parse.parse(pattern)
    except Exception:
        return False

    REPEATS = {sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT}
    SUBPAT = sre_constants.SUBPATTERN
    BRANCH = sre_constants.BRANCH
    MAXR = sre_constants.MAXREPEAT

    def walk(seq, in_unbounded: bool) -> bool:
        for op, val in seq:
            if op in REPEATS:
                _min, _max, sub = val
                unbounded = (_max == MAXR)
                if unbounded and in_unbounded:
                    return True
                if walk(sub, in_unbounded or unbounded):
                    return True
            elif op == SUBPAT:
                # (group, add_flags, del_flags, p)
                sub = val[-1]
                if walk(sub, in_unbounded):
                    return True
            elif op == BRANCH:
                # val = (None, [seq, seq, ...])
                _none, branches = val
                for b in branches:
                    if walk(b, in_unbounded):
                        return True
        return False

    return walk(parsed, in_unbounded=False)


def _score_pattern(pat: str) -> float:
    """Return per-regex score in [0,1]."""
    try:
        # REGEX_OK: tool_output — validating extracted regex literal via the
        # stdlib regex engine as our measurement tool.
        re.compile(pat)
    except re.error:
        return 0.0
    if _has_nested_unbounded_quantifier(pat):
        return 0.5
    return 1.0


# ---------- collector ----------

EXTENSIONS = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}


def _collect_patterns(diff_text: str) -> List[Tuple[str, str]]:
    """Walk added lines per file; return [(lang, pattern), ...]."""
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return []
    out: List[Tuple[str, str]] = []
    for path, body in by_path.items():
        ext = ""
        if "." in path:
            ext = "." + path.rsplit(".", 1)[-1].lower()
        lang = EXTENSIONS.get(ext)
        if lang is None:
            continue
        code = body.encode("utf8", errors="replace")
        if lang == "py":
            pats = _extract_py(code)
        elif lang == "js":
            pats = _extract_js(code, ts=False)
        elif lang == "ts":
            pats = _extract_js(code, ts=True)
        elif lang == "java":
            pats = _extract_java(code)
        elif lang == "go":
            pats = _extract_go(code)
        else:
            pats = []
        for p in pats:
            if p:
                out.append((lang, p))
    return out


def applies(diff_text: str) -> bool:
    return len(_collect_patterns(diff_text)) > 0


def score(diff_text: str) -> Optional[float]:
    pats = _collect_patterns(diff_text)
    if not pats:
        return None
    scores = [_score_pattern(p) for _, p in pats]
    return float(sum(scores) / len(scores))
