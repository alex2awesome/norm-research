"""a200: JS/TS module import/export conventions.

a253 already covers ES-vs-CommonJS, ordering, exports-after-imports, and
unused imports. a200's description points at four conventions that a253 does
NOT measure, so this metric stays distinct rather than collapsing into THICK:

  (A) Prefer NAMED exports over DEFAULT exports — `export default X` is
      discouraged because (i) the consumer chooses an arbitrary local name,
      breaking grep-ability, (ii) IDE refactors can't follow the binding
      across files, and (iii) tree-shaking is less predictable.
      We tally `export default ...` vs named exports
      (`export const|let|var|function|class`, `export { ... }`, named
      `export type ...`) per file.

  (B) Consolidated import per source path — multiple `import` statements
      from the same module path should be merged. We count duplicate source
      strings across top-level `import_statement` nodes in each file.

  (C) Avoid exporting directly from an import — bare re-exports
      (`export ... from "...";`, `export * from "..."`) bypass the local
      module surface and surprise downstream consumers, especially when
      mixed with `export default`. a253 explicitly *exempts*
      `export ... from` from its exports-after-imports check; here we count
      it as a (minor) violation of the a200 convention.

  (D) Use `export type` for type-only re-exports in TypeScript files.
      `import type { X } from "..."` paired with a plain
      `export { X }` defeats `isolatedModules` and produces runtime
      references to a type-only symbol. We measure the fraction of
      re-exported names that originated from `import type` clauses but are
      re-exported WITHOUT `export type`.

CLASSIFICATION = "PARTIALLY_THIN". (A) is a community-divided style choice
(React/Redux team prefers named; Vue/Webpack/most CLIs ship `default`), so
penalizing default exports is a convention, not a verifier. (B), (C), and
(D) are deterministic structural checks.

We parse with tree-sitter (same toolkit as a253/a173). No regex on code.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a200"
ASPECT_NAME = "JS/TS module import/export conventions"
TIER = 2
TOOLS = ["tree-sitter-javascript", "tree-sitter-typescript"]
APPLIES_TO_LANGS = ["JavaScript", "TypeScript"]
CLASSIFICATION = "PARTIALLY_THIN"

JS_EXTS = [".js", ".jsx", ".mjs", ".cjs"]
TS_EXTS = [".ts", ".tsx"]
ALL_EXTS = JS_EXTS + TS_EXTS

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
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


def _import_records(root, src: bytes):
    """For each top-level import_statement: (source_str, type_only,
    type_only_bound_names)."""
    out = []
    for child in root.children:
        if child.type != "import_statement":
            continue
        source_node = None
        for c in child.children:
            if c.type == "string":
                source_node = c
        if source_node is None:
            continue
        src_str = _text(source_node, src).strip("'\"`")
        # whole-clause type-only?
        type_only = False
        type_only_names: Set[str] = set()
        for c in child.children:
            if c.type == "import_clause":
                first = c.children[0] if c.children else None
                clause_type_only = first is not None and first.type == "type"
                if clause_type_only:
                    type_only = True
                for cc in c.children:
                    if cc.type == "named_imports":
                        # `{ a, type B, b as c, type D as E }`
                        for spec in cc.children:
                            if spec.type != "import_specifier":
                                continue
                            spec_text = _text(spec, src).lstrip()
                            spec_type_only = clause_type_only or \
                                spec_text.startswith("type ")
                            ids = [n for n in spec.children
                                   if n.type == "identifier"]
                            if ids and spec_type_only:
                                # last identifier is the local binding
                                type_only_names.add(_text(ids[-1], src))
        out.append((src_str, type_only, type_only_names))
    return out


def _export_records(root, src: bytes):
    """Walk top-level export_statement nodes; categorize each.

    Returns dict with counters and a re-export bookkeeping list.
    """
    rec = {
        "n_default": 0,
        "n_named": 0,
        "n_reexport_from": 0,     # `export ... from "..."`
        "n_export_star": 0,       # `export * from "..."`
        # re-exported names paired with whether `export type` was used
        # entries: (name, used_export_type_keyword)
        "reexported_names": [],
    }
    for child in root.children:
        if child.type != "export_statement":
            continue
        # tree-sitter-javascript: export_statement children include keywords
        # 'export', optionally 'default' or 'type', plus the body.
        kw_default = False
        kw_type = False
        has_from = False
        has_star = False
        named_count = 0
        named_names: List[str] = []
        body_decl = False  # const/let/var/function/class
        for c in child.children:
            t = c.type
            if t == "default":
                kw_default = True
            elif t == "type":
                kw_type = True
            elif t == "string":
                has_from = True
            elif t == "export_clause":
                # `{ a, b as c }`
                for spec in c.children:
                    if spec.type == "export_specifier":
                        ids = [n for n in spec.children
                               if n.type == "identifier"]
                        if ids:
                            # exported name is the LAST identifier in
                            # `local as exported`; for plain `a` it IS the
                            # only id.
                            named_names.append(_text(ids[-1], src))
                            named_count += 1
            elif t == "namespace_export":
                # `export * as X from "..."`
                has_star = True
            elif t in ("lexical_declaration", "variable_declaration",
                       "function_declaration", "class_declaration",
                       "generator_function_declaration",
                       "abstract_class_declaration",
                       "interface_declaration", "type_alias_declaration",
                       "enum_declaration"):
                body_decl = True
            elif t == "*":
                has_star = True

        if kw_default:
            rec["n_default"] += 1
            continue
        if has_star:
            rec["n_export_star"] += 1
            if has_from:
                rec["n_reexport_from"] += 1
            continue
        if has_from:
            rec["n_reexport_from"] += 1
            for nm in named_names:
                rec["reexported_names"].append((nm, kw_type))
            continue
        if body_decl or named_count > 0:
            rec["n_named"] += 1
    return rec


def _file_score(code: bytes, lang: str) -> Optional[float]:
    parser = _get_parser(lang)
    if parser is None:
        return None
    tree = parser.parse(code)
    root = tree.root_node
    if root.type != "program":
        return None

    imports = _import_records(root, code)
    exports = _export_records(root, code)

    sub: List[float] = []

    # (A) Prefer named over default
    n_default = exports["n_default"]
    n_named = exports["n_named"]
    total_ex = n_default + n_named
    if total_ex > 0:
        s_a = n_named / total_ex
        sub.append(s_a)

    # (B) Consolidated import per source path
    if len(imports) >= 2:
        sources = [s for s, _, _ in imports]
        n_dupes = len(sources) - len(set(sources))
        # 0 dupes => 1.0; all imports duplicates of one path => 0.0
        s_b = 1.0 - (n_dupes / max(len(sources) - 1, 1))
        sub.append(max(0.0, min(1.0, s_b)))

    # (C) Avoid exporting directly from imports (re-exports). Counts
    # `export ... from` and `export * from` as norm violations.
    n_reexp = exports["n_reexport_from"] + exports["n_export_star"]
    n_total_exports = n_default + n_named + n_reexp
    if n_total_exports > 0:
        s_c = 1.0 - (n_reexp / n_total_exports)
        sub.append(s_c)

    # (D) Use `export type` for type-only re-exports. Only meaningful in TS
    # and only when at least one re-exported name traces back to an
    # `import type` clause.
    if lang in ("ts", "tsx"):
        # Build set of type-only imported names.
        type_only_imported: Set[str] = set()
        for _src, whole_type, names in imports:
            type_only_imported.update(names)
            # whole-clause `import type {X, Y}`: all names already in `names`
            # because _import_records puts them there when clause_type_only.
            _ = whole_type
        suspect = [(nm, used_kw)
                   for (nm, used_kw) in exports["reexported_names"]
                   if nm in type_only_imported]
        if suspect:
            n_good = sum(1 for _, used_kw in suspect if used_kw)
            s_d = n_good / len(suspect)
            sub.append(s_d)

    if not sub:
        return None
    return sum(sub) / len(sub)


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
