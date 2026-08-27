"""a253: JS/TS import and export organization.

Four conventions, measured per JS/TS file in the diff:

  1. **ES modules over CommonJS** — count `import`/`export` vs `require(...)`
     and `module.exports`. ES form is the modern norm.
  2. **Import ordering** — convention is external packages → relative paths,
     with side-effect-only imports either grouped together at the end or
     kept in their original position. We score top-down ordering of the
     three groups (external, relative, side-effect) using the same
     rank-violation count as a135.
  3. **Exports after imports** — no top-level export statement should appear
     before the last top-level import statement (mirrors a135 "imports at
     top of file"). Type-only and re-exports count.
  4. **Unused imports** — bound names from `import` declarations should
     appear in non-import siblings. TypeScript `import type {...}` and
     `import { type X }` clauses are treated separately because they may be
     consumed only in type positions that tree-sitter records as
     `type_identifier`, which we *also* scan.

Tree-sitter is used because diff-added lines are usually syntactically
incomplete (mid-function). Tree-sitter has error-recovery and still yields
walkable import/export nodes.

The TS analog of a135 has more moving parts than Python:
  - `require(...)` is a function call, not a syntactic form, so the
    CommonJS check looks for `call_expression` whose function is the
    identifier `require`.
  - `export ... from "..."` (re-export) is both an import and export and
    must not double-penalize ordering.
  - `import type {...}` shouldn't be flagged unused if any of its bound
    names appear as a type — TS bindings live in two namespaces.
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a253"
ASPECT_NAME = "JS/TS import and export organization"
TIER = 2
TOOLS = ["tree-sitter-javascript", "tree-sitter-typescript"]
APPLIES_TO_LANGS = ["JavaScript", "TypeScript"]
CLASSIFICATION = "THIN"

JS_EXTS = [".js", ".jsx", ".mjs", ".cjs"]
TS_EXTS = [".ts", ".tsx"]
ALL_EXTS = JS_EXTS + TS_EXTS

_PARSERS = {}


def _get_parser(lang: str):
    """lang in {'js', 'ts', 'tsx'}."""
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "js":
            import tree_sitter_javascript as mod
            _PARSERS[lang] = Parser(Language(mod.language()))
        elif lang == "ts":
            import tree_sitter_typescript as mod
            _PARSERS[lang] = Parser(Language(mod.language_typescript()))
        elif lang == "tsx":
            import tree_sitter_typescript as mod
            _PARSERS[lang] = Parser(Language(mod.language_tsx()))
        else:
            return None
    except ImportError:
        return None
    return _PARSERS[lang]


def _lang_for_path(path: str) -> str:
    p = path.lower()
    if p.endswith(".tsx"):
        return "tsx"
    if p.endswith(".ts"):
        return "ts"
    return "js"


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _classify_specifier(spec: str) -> str:
    """Return 'relative' for './', '../', else 'external'."""
    s = spec.strip().strip("'\"`")
    if s.startswith(".") or s.startswith("/"):
        return "relative"
    return "external"


def _extract_imports(root, src: bytes):
    """Walk top-level children; collect import_statement records.

    Returns list of dicts with keys:
      row: start line
      group: 'external' | 'relative' | 'side_effect'
      bound: list[str] of names introduced
      type_only: bool — `import type {...}`
    """
    out = []
    for child in root.children:
        if child.type != "import_statement":
            continue
        # the source string is the last string child
        source_node = None
        for c in child.children:
            if c.type == "string":
                source_node = c
        if source_node is None:
            continue
        spec = _text(source_node, src)
        # side-effect only: `import "x"` has no clause between `import` and string
        has_clause = any(c.type == "import_clause" for c in child.children)
        if not has_clause:
            grp = "side_effect"
        else:
            grp = _classify_specifier(spec)
        # detect `import type {...}` — the clause has a `type` keyword
        type_only = False
        bound: List[str] = []
        for c in child.children:
            if c.type == "import_clause":
                # children may be: identifier (default), namespace_import,
                # named_imports, plus optional 'type' keyword
                first_token = c.children[0] if c.children else None
                if first_token and first_token.type == "type":
                    type_only = True
                for cc in c.children:
                    if cc.type == "identifier":
                        bound.append(_text(cc, src))
                    elif cc.type == "namespace_import":
                        # `* as Foo`
                        for n in cc.children:
                            if n.type == "identifier":
                                bound.append(_text(n, src))
                    elif cc.type == "named_imports":
                        # `{ a, b as c, type D }`
                        for spec_node in cc.children:
                            if spec_node.type != "import_specifier":
                                continue
                            # last identifier is the local binding
                            ids = [n for n in spec_node.children
                                   if n.type == "identifier"]
                            if ids:
                                bound.append(_text(ids[-1], src))
        out.append({
            "row": child.start_point[0],
            "group": grp,
            "bound": bound,
            "type_only": type_only,
        })
    return out


def _has_require_call(node, src: bytes, found: List[bool]):
    if node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn is not None and fn.type == "identifier" and \
                _text(fn, src) == "require":
            found.append(True)
            return
    for c in node.children:
        _has_require_call(c, src, found)


def _module_exports_assignment(node, src: bytes, found: List[bool]):
    """Detect `module.exports = ...` or `exports.x = ...`."""
    if node.type == "assignment_expression":
        left = node.child_by_field_name("left")
        if left is not None:
            t = _text(left, src)
            if t.startswith("module.exports") or t.startswith("exports."):
                found.append(True)
                return
    for c in node.children:
        _module_exports_assignment(c, src, found)


def _collect_identifiers(node, src: bytes, used: Set[str]):
    """Collect identifier and type_identifier text in non-import children."""
    if node.type in ("import_statement",):
        return
    if node.type in ("identifier", "type_identifier",
                     "shorthand_property_identifier",
                     "property_identifier"):
        used.add(_text(node, src))
    for c in node.children:
        _collect_identifiers(c, src, used)


def _file_score(code: bytes, lang: str) -> Optional[float]:
    parser = _get_parser(lang)
    if parser is None:
        return None
    tree = parser.parse(code)
    root = tree.root_node
    if root.type not in ("program",):
        return None

    imports = _extract_imports(root, code)
    # Count ES import statements + ES export statements
    n_es_import = len(imports)
    n_es_export = sum(1 for c in root.children
                      if c.type == "export_statement")

    # (1) ES vs CommonJS
    cjs_require: List[bool] = []
    _has_require_call(root, code, cjs_require)
    cjs_exports: List[bool] = []
    _module_exports_assignment(root, code, cjs_exports)
    n_es = n_es_import + n_es_export
    n_cjs = len(cjs_require) + len(cjs_exports)
    if n_es + n_cjs == 0:
        s1 = None  # no import/export signal at all
    else:
        s1 = n_es / (n_es + n_cjs)

    sub_scores: List[float] = []
    if s1 is not None:
        sub_scores.append(s1)

    # (2) ordering: external → relative → side_effect
    if len(imports) >= 2:
        rank = {"external": 0, "relative": 1, "side_effect": 2}
        groups = [im["group"] for im in imports]
        violations = sum(1 for a, b in zip(groups, groups[1:])
                         if rank[a] > rank[b])
        s2 = 1.0 - (violations / max(len(groups) - 1, 1))
        sub_scores.append(s2)

    # (3) exports after imports
    if imports and n_es_export > 0:
        last_import_row = imports[-1]["row"]
        early_exports = 0
        for c in root.children:
            if c.type == "export_statement" and \
                    c.start_point[0] < last_import_row:
                # a re-export (`export ... from "..."`) is import-like; allow
                has_from = any(cc.type == "string" for cc in c.children)
                if not has_from:
                    early_exports += 1
        s3 = 1.0 if early_exports == 0 else \
            max(0.0, 1.0 - early_exports / n_es_export)
        sub_scores.append(s3)

    # (4) unused imports — only over runtime (non-type-only) bindings
    runtime_bound = [n for im in imports if not im["type_only"]
                     for n in im["bound"]]
    type_bound = [n for im in imports if im["type_only"]
                  for n in im["bound"]]
    if runtime_bound or type_bound:
        used: Set[str] = set()
        for c in root.children:
            if c.type == "import_statement":
                continue
            _collect_identifiers(c, code, used)
        # runtime bindings: must appear anywhere
        all_bound = runtime_bound + type_bound
        unused = sum(1 for b in all_bound if b not in used)
        s4 = 1.0 - (unused / len(all_bound))
        sub_scores.append(s4)

    if not sub_scores:
        return None
    return sum(sub_scores) / len(sub_scores)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, ALL_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, ALL_EXTS)
    if not by_path:
        return None
    scs: List[float] = []
    for path, content in by_path.items():
        lang = _lang_for_path(path)
        s = _file_score(content.encode("utf8", errors="replace"), lang)
        if s is not None:
            scs.append(s)
    if not scs:
        return None
    return float(sum(scs) / len(scs))
