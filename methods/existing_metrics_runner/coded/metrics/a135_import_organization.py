"""a135: Import organization and hygiene.

Three PEP-8 import conventions, measured per Python file in the diff:
  1. Imports at top of file (before non-import statements).
  2. Stdlib → third-party → local ordering.
  3. No unused imports (bound name appears in code below).

Uses tree-sitter-python (concrete-syntax parser, error-recovery on partial
code). The added lines from a diff are often mid-function or otherwise
syntactically incomplete; tree-sitter parses them anyway and we walk the
visible import nodes plus any identifiers in non-import sibling nodes.
"""
from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a135"
ASPECT_NAME = "Import organization and hygiene"
TIER = 2
TOOLS = ["tree-sitter-python"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]
STDLIB = frozenset(getattr(sys, "stdlib_module_names", set()) or {
    "os", "sys", "re", "math", "json", "typing", "pathlib", "collections",
    "itertools", "functools", "dataclasses", "subprocess", "shutil",
    "tempfile", "io", "logging", "datetime", "time", "string", "abc",
    "asyncio", "enum", "warnings", "unittest", "argparse",
})

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        try:
            import tree_sitter_python
            from tree_sitter import Language, Parser
            _PARSER = Parser(Language(tree_sitter_python.language()))
        except ImportError:
            return None
    return _PARSER


def _top_module(name: str) -> str:
    return (name or "").split(".", 1)[0]


def _classify(mod: str) -> str:
    if mod.startswith("."):
        return "local"
    return "stdlib" if _top_module(mod) in STDLIB else "third"


def _extract_import_info(node, src: bytes) -> List[Tuple[int, str, List[str]]]:
    """Return [(start_row, group, bound_names)] for each import statement at
    the top level (children of module/program node).
    """
    out = []

    def text(n) -> str:
        return src[n.start_byte:n.end_byte].decode("utf8", errors="replace")

    for child in node.children:
        if child.type == "import_statement":
            # import a, b as c, x.y
            for n in child.children:
                if n.type == "dotted_name":
                    out.append((child.start_point[0],
                                _classify(text(n)), [_top_module(text(n))]))
                elif n.type == "aliased_import":
                    mod = next((text(c) for c in n.children
                                if c.type == "dotted_name"), "")
                    alias = next((text(c) for c in n.children
                                  if c.type == "identifier"), "")
                    out.append((child.start_point[0], _classify(mod),
                                [alias or _top_module(mod)]))
        elif child.type == "import_from_statement":
            module_name = ""
            level = 0
            names = []
            for n in child.children:
                if n.type == "import_prefix":
                    level = sum(1 for c in n.children if c.type == ".")
                elif n.type in ("dotted_name", "relative_import"):
                    module_name = text(n)
                elif n.type == "aliased_import":
                    nm = next((text(c) for c in n.children
                               if c.type == "identifier"), "")
                    asn = next((text(c) for c in n.children
                                if c.type == "identifier"), nm)
                    ids = [text(c) for c in n.children if c.type == "identifier"]
                    if len(ids) >= 2:
                        names.append(ids[1])  # alias
                    elif ids:
                        names.append(ids[0])
                elif n.type == "identifier":
                    names.append(text(n))
                elif n.type == "wildcard_import":
                    names = []  # cannot judge `from x import *`
            if names == [] and child.text.endswith(b"*"):
                continue  # wildcard, skip
            grp = "local" if level > 0 else _classify(module_name)
            out.append((child.start_point[0], grp, names))
    return out


def _file_score(code: bytes) -> Optional[float]:
    parser = _get_parser()
    if parser is None:
        return None
    tree = parser.parse(code)
    root = tree.root_node
    if root.type not in ("module", "program"):
        return None
    imports = _extract_import_info(root, code)
    if not imports:
        return None

    # (1) imports at top: scan top-level siblings BEFORE the last import line
    last_import_row = imports[-1][0]
    non_import_before = False
    for child in root.children:
        if child.start_point[0] > last_import_row:
            break
        if child.type in ("import_statement", "import_from_statement",
                          "comment", "expression_statement", "decorator",
                          "future_import_statement"):
            # expression_statement allows module docstring
            continue
        if child.type in ("if_statement", "try_statement"):
            # conditional imports are conventional
            continue
        non_import_before = True
        break
    s1 = 0.0 if non_import_before else 1.0

    # (2) ordering
    rank = {"stdlib": 0, "third": 1, "local": 2}
    groups = [g for _, g, _ in imports]
    violations = sum(1 for a, b in zip(groups, groups[1:])
                     if rank.get(a, 1) > rank.get(b, 1))
    s2 = 1.0 - (violations / max(len(groups) - 1, 1))

    # (3) unused imports: walk identifiers in NON-import nodes; check coverage
    bound = [n for _, _, names in imports for n in names if n]
    if not bound:
        return (s1 + s2) / 2.0
    used = set()
    for child in root.children:
        if child.type in ("import_statement", "import_from_statement"):
            continue

        def collect(node):
            if node.type == "identifier":
                used.add(code[node.start_byte:node.end_byte].decode(
                    "utf8", errors="replace"))
            for c in node.children:
                collect(c)
        collect(child)
    unused = sum(1 for b in bound if b not in used)
    s3 = 1.0 - (unused / len(bound))

    return (s1 + s2 + s3) / 3.0


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    scs = [s for s in (_file_score(c.encode("utf8", errors="replace"))
                       for c in by_path.values()) if s is not None]
    if not scs:
        return None
    return float(sum(scs) / len(scs))
