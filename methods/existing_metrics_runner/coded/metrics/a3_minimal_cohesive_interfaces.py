"""a3: Minimal, cohesive interfaces that encapsulate complexity.

The norm asks two related things about each class/interface a PR adds:

  (a) MINIMAL — the public surface (number of methods/fields a client can
      call) and per-method arity should be small. A class exposing 30
      public methods, or a constructor with 12 parameters, is the canonical
      "bloated surface" the norm forbids.
  (b) COHESIVE — the methods should belong together (single purpose).
      True cohesion (LCOM-style: do methods touch the same fields?) is
      genuinely THICK: it requires whole-file context plus a notion of
      "purpose" that isn't observable from the diff. We instead use a
      structural surrogate: the ratio of *private/non-public* implementation
      to the *public* surface. A well-encapsulated class hides helpers
      behind its public face (private methods, leading-underscore names in
      Python, non-capital field names in Go) — so a low private:public
      ratio is a "leaky surface" smell that maps onto the norm.

That makes a3 PARTIALLY_THIN: (a) is directly observable (counts), (b) is
a surrogate for cohesion that correlates with the norm without measuring it
exactly.

Per-language extraction (tree-sitter, no subprocess):

  Python:
    - public method  := def whose name does NOT start with "_"
    - private method := def whose name starts with "_" (incl. dunders)
    - method arity   := count of parameter children, minus self/cls
  Java:
    - public  := method/field with `public` modifier
    - private := method/field with `private`/`protected`/(no modifier)
    - interface methods count as public
  JS/TS:
    - public  := method NOT prefixed `_` and NOT marked `private`/`protected`
                 (TS); TS `public` modifier is also public
    - private := opposite
    - TS interfaces: every member is public by definition

For each class we compute three sub-scores in [0,1]:
    minimal_methods  = exp(-max(0, n_public_methods - 5) / 5)
    minimal_params   = exp(-max(0, max_public_arity - 3) / 3)
    encapsulation    = (n_private + 1) / (n_public + n_private + 2)
                       (Laplace-smoothed private fraction)

Per-class score = mean of the three. File score = mean over classes.
Diff score = mean over files. Abstain when no class/interface was observed
in the added code.

What this does NOT measure: whether the methods belong to one purpose
(true cohesion), whether the API names client-meaningful concepts, or
whether the class secretly leaks state via mutable returned references.
Those are THICK and out of scope here.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a3"
ASPECT_NAME = "Minimal, cohesive interfaces"
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
        else:
            return None
        _PARSERS[lang] = Parser(Language(L))
        return _PARSERS[lang]
    except ImportError:
        return None


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# ----- Python ---------------------------------------------------------------

def _py_classes(root, src: bytes) -> List[Tuple[int, int, int]]:
    """Return list of (n_public, n_private, max_public_arity) per class."""
    results: List[Tuple[int, int, int]] = []

    def name_of(node):
        for c in node.children:
            if c.type == "identifier":
                return _text(c, src)
        return None

    def class_body(node):
        for c in node.children:
            if c.type == "block":
                return c
        return None

    def method_arity(func_node) -> int:
        for c in func_node.children:
            if c.type == "parameters":
                arity = 0
                for p in c.children:
                    if p.type in ("identifier", "typed_parameter",
                                  "default_parameter",
                                  "typed_default_parameter",
                                  "list_splat_pattern",
                                  "dictionary_splat_pattern"):
                        # skip self/cls
                        ptxt = _text(p, src).lstrip("*").split(":")[0].split("=")[0].strip()
                        if ptxt in ("self", "cls"):
                            continue
                        arity += 1
                return arity
        return 0

    def walk(node):
        if node.type == "class_definition":
            body = class_body(node)
            n_pub = 0
            n_priv = 0
            max_arity = 0
            if body is not None:
                for c in body.children:
                    if c.type == "function_definition":
                        nm = name_of(c)
                        if nm is None:
                            continue
                        is_pub = not nm.startswith("_")
                        ar = method_arity(c)
                        if is_pub:
                            n_pub += 1
                            if ar > max_arity:
                                max_arity = ar
                        else:
                            n_priv += 1
                    elif c.type == "decorated_definition":
                        # decorated method
                        for cc in c.children:
                            if cc.type == "function_definition":
                                nm = name_of(cc)
                                if nm is None:
                                    continue
                                is_pub = not nm.startswith("_")
                                ar = method_arity(cc)
                                if is_pub:
                                    n_pub += 1
                                    if ar > max_arity:
                                        max_arity = ar
                                else:
                                    n_priv += 1
                                break
            if n_pub + n_priv > 0:
                results.append((n_pub, n_priv, max_arity))
        for c in node.children:
            walk(c)

    walk(root)
    return results


# ----- Java -----------------------------------------------------------------

def _java_modifier_text(node, src: bytes) -> str:
    for c in node.children:
        if c.type == "modifiers":
            return _text(c, src)
    return ""


def _java_method_name(node, src: bytes) -> Optional[str]:
    for c in node.children:
        if c.type == "identifier":
            return _text(c, src)
    return None


def _java_method_arity(node, src: bytes) -> int:
    for c in node.children:
        if c.type == "formal_parameters":
            count = 0
            for fp in c.children:
                if fp.type in ("formal_parameter", "spread_parameter"):
                    count += 1
            return count
    return 0


def _java_classes(root, src: bytes) -> List[Tuple[int, int, int]]:
    results: List[Tuple[int, int, int]] = []

    def class_body(node):
        for c in node.children:
            if c.type in ("class_body", "interface_body", "enum_body",
                          "record_body", "annotation_type_body"):
                return c
        return None

    def walk(node):
        if node.type in ("class_declaration", "interface_declaration",
                         "enum_declaration", "record_declaration"):
            is_interface = node.type == "interface_declaration"
            body = class_body(node)
            n_pub = 0
            n_priv = 0
            max_arity = 0
            if body is not None:
                for c in body.children:
                    if c.type in ("method_declaration",
                                  "constructor_declaration"):
                        mods = _java_modifier_text(c, src)
                        if is_interface:
                            is_pub = True
                        elif "public" in mods:
                            is_pub = True
                        else:
                            is_pub = False
                        ar = _java_method_arity(c, src)
                        if is_pub:
                            n_pub += 1
                            if ar > max_arity:
                                max_arity = ar
                        else:
                            n_priv += 1
            if n_pub + n_priv > 0:
                results.append((n_pub, n_priv, max_arity))
        for c in node.children:
            walk(c)

    walk(root)
    return results


# ----- JS / TS --------------------------------------------------------------

def _js_method_name(node, src: bytes) -> Optional[str]:
    for c in node.children:
        if c.type in ("property_identifier", "identifier",
                      "private_property_identifier"):
            return _text(c, src)
    return None


def _js_method_arity(node, src: bytes) -> int:
    for c in node.children:
        if c.type == "formal_parameters":
            count = 0
            for p in c.children:
                if p.type in ("identifier", "required_parameter",
                              "optional_parameter", "rest_pattern",
                              "object_pattern", "array_pattern",
                              "assignment_pattern"):
                    count += 1
            return count
    return 0


def _ts_member_is_private(c, src: bytes) -> bool:
    """TS uses an `accessibility_modifier` child (`public`/`private`/`protected`).
    JS has no formal private except `#name` (private_property_identifier).
    Names starting with `_` are private by convention."""
    name = _js_method_name(c, src) or ""
    if name.startswith("#") or name.startswith("_"):
        return True
    # explicit TS accessibility modifier
    for cc in c.children:
        if cc.type == "accessibility_modifier":
            txt = _text(cc, src).strip()
            if txt in ("private", "protected"):
                return True
            if txt == "public":
                return False
    return False


def _js_classes(root, src: bytes, is_ts: bool) -> List[Tuple[int, int, int]]:
    results: List[Tuple[int, int, int]] = []

    def class_body(node):
        for c in node.children:
            if c.type in ("class_body", "object_type", "interface_body"):
                return c
        return None

    def member_arity(member):
        # Method definitions / signatures hold formal_parameters directly.
        for c in member.children:
            if c.type == "formal_parameters":
                count = 0
                for p in c.children:
                    if p.type in ("identifier", "required_parameter",
                                  "optional_parameter", "rest_pattern",
                                  "object_pattern", "array_pattern",
                                  "assignment_pattern"):
                        count += 1
                return count
        return 0

    def walk(node):
        kind = node.type
        if kind in ("class_declaration", "interface_declaration"):
            is_interface = kind == "interface_declaration"
            body = class_body(node)
            n_pub = 0
            n_priv = 0
            max_arity = 0
            if body is not None:
                for c in body.children:
                    if c.type in ("method_definition", "method_signature",
                                  "abstract_method_signature",
                                  "property_signature"):
                        if is_interface:
                            is_priv = False
                        else:
                            is_priv = _ts_member_is_private(c, src)
                        ar = member_arity(c)
                        if is_priv:
                            n_priv += 1
                        else:
                            n_pub += 1
                            if ar > max_arity:
                                max_arity = ar
            if n_pub + n_priv > 0:
                results.append((n_pub, n_priv, max_arity))
        for c in node.children:
            walk(c)

    walk(root)
    return results


# ----- Scoring --------------------------------------------------------------

def _score_class(n_pub: int, n_priv: int, max_arity: int) -> float:
    minimal_methods = math.exp(-max(0, n_pub - 5) / 5.0)
    minimal_params = math.exp(-max(0, max_arity - 3) / 3.0)
    encapsulation = (n_priv + 1.0) / (n_pub + n_priv + 2.0)
    return (minimal_methods + minimal_params + encapsulation) / 3.0


def _path_lang(path: str) -> Optional[str]:
    p = path.lower()
    for ext, lang in EXT_TO_LANG.items():
        if p.endswith(ext):
            return lang
    return None


def _file_classes(code: bytes, lang: str) -> List[Tuple[int, int, int]]:
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
        classes = _file_classes(content.encode("utf8", errors="replace"), lang)
        if not classes:
            continue
        per_class = [_score_class(*c) for c in classes]
        file_scores.append(sum(per_class) / len(per_class))
    if not file_scores:
        return None
    return float(sum(file_scores) / len(file_scores))
