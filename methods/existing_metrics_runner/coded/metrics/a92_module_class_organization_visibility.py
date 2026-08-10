"""a92: Module and class organization, and visibility.

The aspect: "Define clear module/class boundaries, organize package/crate
trees logically, and set appropriate visibility to keep APIs maintainable
and evolvable."

Distinct angle vs. a3 / a35:
  - a3 (minimal cohesive interfaces) scores the *per-class* public surface
    (n methods, max arity, private:public ratio inside one class). It is
    blind to how many classes share a file or how broad the *module-level*
    API is.
  - a35 (information hiding / stable abstractions) is THICK in this catalog
    (no metric file): "stability" requires history a single diff cannot show.
  - a92 here measures **file / module-level organization and visibility**:
    how many top-level classes coexist per file, how broad the module's
    top-level exported surface is, and how disciplined visibility
    declarations are (Python ``__all__`` / leading-underscore convention;
    Java class- and member-level access modifiers; Go capitalization-as-
    visibility; JS/TS named-export breadth + TS accessibility modifiers).

Per-language signals (tree-sitter, no subprocess):

  Python (.py, .pyi):
    A. classes_per_file penalty:
         class_score = exp(-max(0, n_top_classes - 2) / 2)
       (2 classes per file is the comfortable ceiling.)
    B. top_export_breadth penalty:
         n_top_public = count of top-level def/class whose name does NOT
                        start with "_", excluding names listed in __all__'s
                        complement.
         export_score = exp(-max(0, n_top_public - 6) / 6)
    C. visibility_discipline:
         If the file has 4+ top-level public names, having ``__all__``
         (explicit export list) earns full credit (1.0); otherwise the
         score is the fraction of top-level names that use the ``_``
         prefix convention to mark intent, capped at 1.0.

  Java (.java):
    A. n_top_classes = count of top-level type declarations (classes,
       interfaces, enums, records). The Java *language* allows only one
       PUBLIC top-level type per file; we score:
         pub_top = count of those declared ``public``.
         pub_top <= 1 -> 1.0 ; >1 -> 0.0 (Java illegal).
       And classes_per_file penalty same as Python.
    B. top_export_breadth penalty: same exp decay over public top-level
       types (>1 already 0 in Java; this is for completeness).
    C. visibility_discipline: per top-level class body, fraction of methods
       and fields with an explicit access modifier (public/private/
       protected). Java's *implicit* package-private is allowed but
       indicates the author didn't make a deliberate choice — full credit
       requires explicit modifiers OR explicit package-private intent (no
       reliable AST distinction; we credit explicit modifiers only).

  Go (.go):
    Go uses capitalization for visibility (Effective Go). Signals:
    A. n_top_decls = count of top-level type/func/var/const declarations.
       n_top_types = top-level type_declaration count. Penalty:
         exp(-max(0, n_top_types - 4) / 4)   (file cohesion)
    B. exported breadth: fraction of top-level identifiers that start
       Upper. A file that exports 20+ symbols with no unexported helpers
       is leaky; we use:
         export_score = exp(-max(0, n_exported - 8) / 8).
    C. visibility_discipline (Go-specific):
         Encourages mixing exported+unexported (a fully-exported file is
         either a tiny API module or a leaky one). Score by entropy of the
         exported/unexported split:
             p = n_exported / (n_top_decls)
             entropy = -p*log2(p) - (1-p)*log2(1-p)  (with 0*log0 := 0)
         entropy in [0,1]; 1.0 when half-and-half.

  JavaScript / TypeScript (.js/.jsx/.mjs/.cjs/.ts/.tsx):
    A. classes_per_file penalty: same exp decay over top-level
       class_declaration count.
    B. export_breadth: count named export declarations
       (export const|let|var|function|class|type|interface, export { ... }).
         export_score = exp(-max(0, n_exports - 8) / 8)
    C. visibility_discipline (TS only):
         For TS files only, fraction of class members that use an
         explicit accessibility_modifier (public/private/protected) OR a
         JS-#private name. JS-only files abstain on (C) and use mean of
         (A)+(B).

Per-file score = mean of available sub-scores (A, B, [C]).
Diff score      = mean of file scores. Abstain when no supported file is
                  present in the diff OR when no top-level declarations
                  were found in any added file (e.g., the diff only
                  modified expressions / docstrings).

Classification: PARTIALLY_THIN.
  - Counting top-level decls, exported names, and access modifiers IS thin
    (pure tree-sitter; no judgement call).
  - The "appropriate visibility" threshold is calibrated, not derived: the
    decay constants (2 classes/file, 6 Python exports, 8 Go exports, 8
    JS/TS exports) are author-chosen anchors that work across the
    fixtures but are not normative law. That calibration is the partial-
    thinness.

This metric does NOT measure:
  - whether the chosen visibility *fits the use case* (would a public
    method be better protected, etc.) — that is taste / a35.
  - whether the package tree is logical across files — a268's THICK
    province (we don't see the repo).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a92"
ASPECT_NAME = "Module and class organization and visibility"
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

def _py_top_level_children(root):
    """Yield direct children of the module root."""
    for c in root.children:
        yield c


def _py_name_of(node, src: bytes) -> Optional[str]:
    for c in node.children:
        if c.type == "identifier":
            return _text(c, src)
    return None


def _py_has_all(root, src: bytes) -> bool:
    """Detect a top-level ``__all__ = [...]`` assignment."""
    for c in _py_top_level_children(root):
        if c.type == "expression_statement":
            for cc in c.children:
                if cc.type == "assignment":
                    # left side identifier
                    for left in cc.children:
                        if left.type == "identifier" and _text(left, src) == "__all__":
                            return True
                        break
    return False


def _py_score_file(root, src: bytes) -> Optional[float]:
    n_top_classes = 0
    n_top_public_names = 0
    n_top_underscore_names = 0
    n_top_total_names = 0
    for c in _py_top_level_children(root):
        if c.type == "class_definition":
            n_top_classes += 1
            nm = _py_name_of(c, src)
            if nm is not None:
                n_top_total_names += 1
                if nm.startswith("_"):
                    n_top_underscore_names += 1
                else:
                    n_top_public_names += 1
        elif c.type == "function_definition":
            nm = _py_name_of(c, src)
            if nm is not None:
                n_top_total_names += 1
                if nm.startswith("_"):
                    n_top_underscore_names += 1
                else:
                    n_top_public_names += 1
        elif c.type == "decorated_definition":
            for cc in c.children:
                if cc.type in ("function_definition", "class_definition"):
                    if cc.type == "class_definition":
                        n_top_classes += 1
                    nm = _py_name_of(cc, src)
                    if nm is not None:
                        n_top_total_names += 1
                        if nm.startswith("_"):
                            n_top_underscore_names += 1
                        else:
                            n_top_public_names += 1
                    break
    if n_top_total_names == 0:
        return None  # nothing top-level was declared
    class_score = math.exp(-max(0, n_top_classes - 2) / 2.0)
    export_score = math.exp(-max(0, n_top_public_names - 6) / 6.0)
    # visibility discipline
    if _py_has_all(root, src):
        vis_score = 1.0
    elif n_top_public_names >= 4:
        # require some underscore-prefixed helpers to mark internal intent
        frac = n_top_underscore_names / max(n_top_total_names, 1)
        vis_score = min(1.0, frac * 2.5)  # 40% underscore -> 1.0
    else:
        # small modules don't need __all__ or _-prefixing
        vis_score = 1.0
    return (class_score + export_score + vis_score) / 3.0


# ----- Java ----------------------------------------------------------------

def _java_modifier_text(node, src: bytes) -> str:
    for c in node.children:
        if c.type == "modifiers":
            return _text(c, src)
    return ""


def _java_score_file(root, src: bytes) -> Optional[float]:
    # Find program-level type declarations
    top_types = []
    for c in root.children:
        if c.type in ("class_declaration", "interface_declaration",
                      "enum_declaration", "record_declaration",
                      "annotation_type_declaration"):
            top_types.append(c)
    if not top_types:
        return None
    n_top_classes = len(top_types)
    n_public_top = sum(1 for t in top_types
                       if "public" in _java_modifier_text(t, src))
    class_score = math.exp(-max(0, n_top_classes - 2) / 2.0)
    # one-public-top rule
    if n_public_top <= 1:
        public_top_score = 1.0
    else:
        public_top_score = 0.0
    # visibility discipline: explicit access modifiers on members
    n_members = 0
    n_explicit_access = 0
    for t in top_types:
        body = None
        for c in t.children:
            if c.type in ("class_body", "interface_body", "enum_body",
                          "record_body", "annotation_type_body"):
                body = c
                break
        if body is None:
            continue
        is_interface = t.type == "interface_declaration"
        for member in body.children:
            if member.type in ("method_declaration",
                               "constructor_declaration",
                               "field_declaration"):
                n_members += 1
                mods = _java_modifier_text(member, src)
                if is_interface:
                    # implicitly public; treat as explicit
                    n_explicit_access += 1
                elif any(k in mods for k in
                         ("public", "private", "protected")):
                    n_explicit_access += 1
    if n_members == 0:
        vis_score = 1.0
    else:
        vis_score = n_explicit_access / n_members
    return (class_score + public_top_score + vis_score) / 3.0


# ----- Go -------------------------------------------------------------------

def _go_decl_name(node, src: bytes) -> Optional[str]:
    """Best-effort: first identifier-like child."""
    for c in node.children:
        if c.type in ("identifier", "field_identifier", "type_identifier"):
            return _text(c, src)
    return None


def _go_collect_top_names(root, src: bytes) -> Tuple[int, List[str]]:
    """Return (n_top_types, list_of_top_identifiers)."""
    n_top_types = 0
    names: List[str] = []
    for c in root.children:
        if c.type == "type_declaration":
            n_top_types += 1
            # walk type_spec children
            for spec in c.children:
                if spec.type == "type_spec":
                    nm = _go_decl_name(spec, src)
                    if nm:
                        names.append(nm)
        elif c.type == "function_declaration":
            # name = first identifier after 'func'
            for cc in c.children:
                if cc.type == "identifier":
                    names.append(_text(cc, src))
                    break
        elif c.type == "method_declaration":
            # methods have a receiver; name = field_identifier
            for cc in c.children:
                if cc.type == "field_identifier":
                    names.append(_text(cc, src))
                    break
        elif c.type in ("var_declaration", "const_declaration"):
            for spec in c.children:
                if spec.type in ("var_spec", "const_spec"):
                    for cc in spec.children:
                        if cc.type == "identifier":
                            names.append(_text(cc, src))
                            break
    return n_top_types, names


def _go_score_file(root, src: bytes) -> Optional[float]:
    n_top_types, names = _go_collect_top_names(root, src)
    n_top_decls = len(names)
    if n_top_decls == 0:
        return None
    n_exported = sum(1 for n in names if n and n[0].isupper())
    class_score = math.exp(-max(0, n_top_types - 4) / 4.0)
    export_score = math.exp(-max(0, n_exported - 8) / 8.0)
    # entropy-based visibility-discipline score
    if n_top_decls <= 1:
        vis_score = 1.0
    else:
        p = n_exported / n_top_decls
        if p == 0.0 or p == 1.0:
            # 100% exported or 100% unexported. Slight penalty: 0.5
            vis_score = 0.5
        else:
            vis_score = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return (class_score + export_score + vis_score) / 3.0


# ----- JS / TS --------------------------------------------------------------

def _js_top_level_children(root):
    # Real ES modules: top-level under program root.
    for c in root.children:
        yield c


def _js_count_named_exports(root, src: bytes) -> int:
    """Count export breadth.

    - `export const|let|var|function|class|type|interface|enum X ...` -> 1
    - `export { a, b as c, d }` -> N (count export_specifier children)
    - `export default ...` -> 1 (counted toward breadth too)
    - bare `export { X } from "..."` -> count specifiers
    """
    n = 0
    for c in _js_top_level_children(root):
        if c.type == "export_statement":
            had_specifier_list = False
            for cc in c.children:
                if cc.type == "export_clause":
                    had_specifier_list = True
                    for spec in cc.children:
                        if spec.type == "export_specifier":
                            n += 1
            if had_specifier_list:
                continue
            n += 1
    return n


def _js_score_file(root, src: bytes, is_ts: bool) -> Optional[float]:
    n_top_classes = 0
    for c in _js_top_level_children(root):
        if c.type == "class_declaration":
            n_top_classes += 1
        elif c.type == "export_statement":
            for cc in c.children:
                if cc.type == "class_declaration":
                    n_top_classes += 1
    n_exports = _js_count_named_exports(root, src)
    if n_top_classes == 0 and n_exports == 0:
        return None
    class_score = math.exp(-max(0, n_top_classes - 2) / 2.0)
    export_score = math.exp(-max(0, n_exports - 8) / 8.0)
    sub = [class_score, export_score]
    if is_ts:
        # TS visibility discipline: fraction of class members with an
        # explicit accessibility_modifier or `#name` private.
        n_members = 0
        n_explicit = 0

        def walk(node):
            nonlocal n_members, n_explicit
            if node.type == "class_body":
                for member in node.children:
                    if member.type in ("method_definition",
                                       "public_field_definition",
                                       "property_signature",
                                       "method_signature"):
                        n_members += 1
                        # explicit access modifier?
                        has_mod = False
                        for cc in member.children:
                            if cc.type == "accessibility_modifier":
                                has_mod = True
                                break
                            if cc.type == "private_property_identifier":
                                has_mod = True
                                break
                        if has_mod:
                            n_explicit += 1
            for cc in node.children:
                walk(cc)

        walk(root)
        if n_members > 0:
            vis_score = n_explicit / n_members
            sub.append(vis_score)
        # else: no class members -> skip C
    return sum(sub) / len(sub)


# ----- Driver ---------------------------------------------------------------

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
        elif lang == "js":
            s = _js_score_file(root, code, is_ts=False)
        elif lang == "ts":
            s = _js_score_file(root, code, is_ts=True)
        else:
            s = None
        if s is not None:
            file_scores.append(s)
    if not file_scores:
        return None
    return float(sum(file_scores) / len(file_scores))
