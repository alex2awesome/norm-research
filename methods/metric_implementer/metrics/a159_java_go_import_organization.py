"""a159: Import statements — placement, ordering, and grouping (Java + Go).

a159 ("Place imports at the top, one per line; group by std/third-party/local
with blank lines; apply consistent ordering and formatting across files") is
language-agnostic. The Python and JS/TS instantiations of this norm are
already covered by:

  - a135 (Python: top placement, stdlib→third→local, unused)
  - a253 (JS/TS: external→relative, exports after imports, unused)

This module fills the remaining two majority-language slots in the code_review
fixture set: Java (~20% of fixtures) and Go (~20%). The conventions checked
here are the language-community ones, distinct from the Python/JS ones, so
this is a genuine extension and not a duplicate of a135/a253.

==========
Java norms
==========

1. **Imports after package, before type declarations.** All `import`
   declarations must appear between the (optional) `package` declaration and
   the first `class`/`interface`/`enum`/`record` declaration.
2. **No wildcard imports.** `import foo.bar.*;` is the canonical Java-style
   guide violation (Google Java Style §3.3.1; Sun/Oracle code conventions).
   Static imports of constants are likewise required to name the constant.
3. **Lexicographic ordering within groups.** Google Java Style §3.3.3:
   imports are sorted ASCII-lexicographically; statics first, then
   non-statics, each block internally sorted. We check both blocks for
   sorted-ness.
4. **No duplicate imports.** Same fully-qualified name imported twice.

============
Go norms (gofmt / goimports style)
============

1. **Imports inside one `import (...)` block.** The community convention is
   a single grouped import block, not many `import "x"` lines scattered.
2. **No blank-line-separated subgroups out of order.** `goimports` produces
   stdlib first, then third-party (anything with a `.` in the first path
   segment), then optionally local. Subgroups are separated by blank lines.
   We classify each spec as stdlib (no dot in first segment) vs external
   (dot in first segment) and check rank monotonicity.
3. **No duplicate imports.** Same path imported twice.
4. **No unused imports** would normally be caught by `go build`, but on
   partial diff snippets we use a conservative tree-sitter walk: an
   imported name is considered used if it appears as an identifier in any
   non-import sibling node.

For both languages we use tree-sitter (already a dependency for a135, a253,
a283). Diff-added lines are usually syntactically incomplete; tree-sitter's
error recovery still surfaces the import declarations.

Per-file score: mean of the (applicable) sub-rules above, each in [0,1].
Metric score: mean over files touched in the added diff.
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a159"
ASPECT_NAME = "Java/Go import placement, ordering, grouping"
TIER = 2
TOOLS = ["tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Java", "Go"]
CLASSIFICATION = "THIN"

JAVA_EXTS = [".java"]
GO_EXTS = [".go"]
ALL_EXTS = JAVA_EXTS + GO_EXTS

_PARSERS = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "java":
            import tree_sitter_java as mod
            _PARSERS[lang] = Parser(Language(mod.language()))
        elif lang == "go":
            import tree_sitter_go as mod
            _PARSERS[lang] = Parser(Language(mod.language()))
        else:
            return None
    except ImportError:
        return None
    return _PARSERS[lang]


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------

def _java_extract_imports(root, src: bytes):
    """Return list of dicts with keys:
        row: line number
        text: full FQN string ("a.b.C" or "a.b.*")
        is_static: bool
        is_wildcard: bool
    Only top-level import_declarations are returned.
    """
    out = []
    for child in root.children:
        if child.type != "import_declaration":
            continue
        is_static = False
        is_wildcard = False
        name_parts = []
        for c in child.children:
            t = c.type
            if t == "static":
                is_static = True
            elif t == "asterisk":
                is_wildcard = True
            elif t in ("scoped_identifier", "identifier"):
                name_parts.append(_text(c, src))
        full = ".".join(name_parts) + (".*" if is_wildcard else "")
        out.append({
            "row": child.start_point[0],
            "text": full,
            "is_static": is_static,
            "is_wildcard": is_wildcard,
        })
    return out


def _java_file_score(code: bytes) -> Optional[float]:
    parser = _get_parser("java")
    if parser is None:
        return None
    # Java parses without wrapping; partial snippets may not have a package
    tree = parser.parse(code)
    root = tree.root_node
    if root.type != "program":
        return None
    imports = _java_extract_imports(root, code)
    if not imports:
        return None

    sub_scores: List[float] = []

    # (1) Imports between package and first type declaration
    package_row = -1
    first_type_row = None
    type_kinds = ("class_declaration", "interface_declaration",
                  "enum_declaration", "record_declaration",
                  "annotation_type_declaration")
    for child in root.children:
        if child.type == "package_declaration":
            package_row = child.start_point[0]
        if child.type in type_kinds and first_type_row is None:
            first_type_row = child.start_point[0]
    misplaced = 0
    for im in imports:
        if package_row >= 0 and im["row"] < package_row:
            misplaced += 1
        elif first_type_row is not None and im["row"] > first_type_row:
            misplaced += 1
    s_place = 1.0 - (misplaced / len(imports))
    sub_scores.append(s_place)

    # (2) No wildcard imports (static wildcards excluded —
    # `import static X.*` is acceptable per Google style for test utility
    # classes but normal-import wildcards are the canonical violation).
    non_static_wildcards = sum(
        1 for im in imports if im["is_wildcard"] and not im["is_static"])
    total_non_static = sum(1 for im in imports if not im["is_static"])
    if total_non_static > 0:
        s_wc = 1.0 - (non_static_wildcards / total_non_static)
        sub_scores.append(s_wc)

    # (3) Lexicographic ordering within static/non-static blocks
    statics = [im["text"] for im in imports if im["is_static"]]
    non_statics = [im["text"] for im in imports if not im["is_static"]]

    def _sorted_score(seq: List[str]) -> Optional[float]:
        if len(seq) < 2:
            return None
        inv = sum(1 for a, b in zip(seq, seq[1:]) if a > b)
        return 1.0 - inv / (len(seq) - 1)

    for sub in (statics, non_statics):
        sc = _sorted_score(sub)
        if sc is not None:
            sub_scores.append(sc)

    # (4) No duplicates
    texts = [im["text"] for im in imports]
    uniq = len(set(texts))
    s_dup = uniq / len(texts) if texts else 1.0
    sub_scores.append(s_dup)

    return sum(sub_scores) / len(sub_scores)


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

def _go_extract_import_specs(root, src: bytes):
    """Return list of (row, path, name_or_None, in_block).

    Walks `import_declaration` nodes; if it contains an `import_spec_list`
    (i.e. `import ( ... )`), all child `import_spec`s are in_block=True.
    Otherwise the single import_spec child is in_block=False.
    """
    out = []
    decl_blocks = []  # (start_row, end_row) of grouped blocks
    for child in root.children:
        if child.type != "import_declaration":
            continue
        spec_list = None
        single = None
        for c in child.children:
            if c.type == "import_spec_list":
                spec_list = c
            elif c.type == "import_spec":
                single = c
        if spec_list is not None:
            decl_blocks.append((spec_list.start_point[0],
                                spec_list.end_point[0]))
            for spec in spec_list.children:
                if spec.type != "import_spec":
                    continue
                out.append(_parse_spec(spec, src, in_block=True))
        elif single is not None:
            out.append(_parse_spec(single, src, in_block=False))
    return out, decl_blocks


def _parse_spec(spec, src: bytes, in_block: bool):
    path = ""
    name = None
    for c in spec.children:
        if c.type == "interpreted_string_literal":
            raw = _text(c, src)
            path = raw.strip().strip('"')
        elif c.type == "package_identifier":
            name = _text(c, src)
        elif c.type == "dot":
            name = "."
        elif c.type == "blank_identifier":
            name = "_"
    return {
        "row": spec.start_point[0],
        "path": path,
        "name": name,
        "in_block": in_block,
    }


def _go_classify(path: str) -> str:
    """stdlib if first path segment has no '.', else external."""
    if not path:
        return "stdlib"
    first = path.split("/", 1)[0]
    return "external" if "." in first else "stdlib"


def _go_local_name(spec) -> Optional[str]:
    """The identifier that gets bound when this import is used."""
    if spec["name"] == "_" or spec["name"] == ".":
        return None  # side-effect / dot import — can't track usage
    if spec["name"]:
        return spec["name"]
    if not spec["path"]:
        return None
    return spec["path"].rstrip("/").split("/")[-1]


def _go_collect_identifiers(node, src: bytes, used: Set[str]):
    if node.type == "import_declaration":
        return
    if node.type in ("identifier", "package_identifier",
                     "type_identifier", "field_identifier"):
        used.add(_text(node, src))
    for c in node.children:
        _go_collect_identifiers(c, src, used)


def _go_file_score(code: bytes) -> Optional[float]:
    parser = _get_parser("go")
    if parser is None:
        return None
    # Wrap if there's no package declaration (diff snippets often lack one)
    src = code if code.lstrip().startswith(b"package ") else (
        b"package __snip\n" + code)
    tree = parser.parse(src)
    root = tree.root_node
    if root.type != "source_file":
        return None
    specs, blocks = _go_extract_import_specs(root, src)
    if not specs:
        return None

    sub_scores: List[float] = []

    # (1) Single grouped block: penalize scattered `import "x"` lines when
    # multiple imports exist.
    if len(specs) >= 2:
        in_block = sum(1 for s in specs if s["in_block"])
        s_group = in_block / len(specs)
        sub_scores.append(s_group)

    # (2) Group ordering: within each block, stdlib before external.
    # goimports separates them with a blank line; we approximate the
    # blank-line break by line-row gap >= 2.
    if blocks:
        for (b_start, b_end) in blocks:
            block_specs = sorted(
                [s for s in specs if b_start <= s["row"] <= b_end],
                key=lambda s: s["row"])
            if len(block_specs) < 2:
                continue
            # Subgroup by row-gap
            subgroups = [[block_specs[0]]]
            for prev, cur in zip(block_specs, block_specs[1:]):
                if cur["row"] - prev["row"] >= 2:
                    subgroups.append([cur])
                else:
                    subgroups[-1].append(cur)
            # Each subgroup should be homogeneous; sequence of subgroup
            # majority-classes should be stdlib-before-external.
            sub_classes = []
            mix_penalty = 0
            for sg in subgroups:
                classes = [_go_classify(s["path"]) for s in sg]
                stdlib_n = classes.count("stdlib")
                ext_n = classes.count("external")
                if stdlib_n and ext_n:
                    mix_penalty += 1
                sub_classes.append("stdlib" if stdlib_n >= ext_n
                                   else "external")
            rank = {"stdlib": 0, "external": 1}
            inv = sum(1 for a, b in zip(sub_classes, sub_classes[1:])
                      if rank[a] > rank[b])
            denom = max(len(sub_classes) - 1, 1) + max(len(subgroups), 1)
            s_order = 1.0 - (inv + mix_penalty) / denom
            sub_scores.append(max(0.0, s_order))

    # (3) No duplicate paths
    paths = [s["path"] for s in specs if s["path"]]
    if paths:
        s_dup = len(set(paths)) / len(paths)
        sub_scores.append(s_dup)

    # (4) Unused imports — conservative
    bound = [n for n in (_go_local_name(s) for s in specs) if n]
    if bound:
        used: Set[str] = set()
        for child in root.children:
            if child.type == "import_declaration":
                continue
            _go_collect_identifiers(child, src, used)
        unused = sum(1 for b in bound if b not in used)
        s_unused = 1.0 - (unused / len(bound))
        sub_scores.append(s_unused)

    if not sub_scores:
        return None
    return sum(sub_scores) / len(sub_scores)


# ---------------------------------------------------------------------------
# Metric entrypoints
# ---------------------------------------------------------------------------

def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, ALL_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, ALL_EXTS)
    if not by_path:
        return None
    scs: List[float] = []
    for path, content in by_path.items():
        p = path.lower()
        b = content.encode("utf8", errors="replace")
        if p.endswith(".java"):
            s = _java_file_score(b)
        elif p.endswith(".go"):
            s = _go_file_score(b)
        else:
            s = None
        if s is not None:
            scs.append(s)
    if not scs:
        return None
    return float(sum(scs) / len(scs))
