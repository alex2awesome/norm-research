"""a136: Import statement formatting and aliasing hygiene.

Distinct angle vs a135 (Python imports organization — top/order/unused) and
a253 (JS/TS imports — ES vs CJS / ordering / unused). This metric scores the
*surface* hygiene of import statements: line shape, alias quality, wildcard
imports, and shadowing.

Four sub-checks per Python file in the diff:

  1. **One import per line.** PEP-8 says `import a, b, c` (multiple top-level
     modules on one `import` line) is discouraged. Bare `from x import (a, b)`
     blocks are *not* flagged — that is the canonical "many names from one
     module" form. We only flag the `import a, b` family.

  2. **Alias hygiene.** An aliased import is healthy when the alias is the
     conventional/canonical short name for the module (`pandas as pd`,
     `numpy as np`, `tensorflow as tf`, ...). It is unhealthy when the alias
     is cryptic (1-2 char random), shadows a builtin, or differs from the
     well-known convention for a famous library (e.g. `pandas as panda`).

  3. **No wildcard imports.** `from x import *` pollutes namespace and is the
     classic anti-pattern; flagged hard.

  4. **No shadowing.** The same bound name introduced by two different import
     statements, or an import bound name that shadows a Python builtin
     (`list`, `dict`, `id`, `type`, ...), is flagged.

We use tree-sitter-python for syntax-level extraction so partial diff lines
don't trip line-based regex. This is `PARTIALLY_THIN`: anti-patterns
(wildcard, multi-module-on-one-line, builtin-shadowing, name collision) are
THIN/deterministic; the alias-quality judgment is partly heuristic
(whitelist of conventional aliases + structural cryptic-name detection).
"""
from __future__ import annotations

import builtins
from typing import Dict, List, Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a136"
ASPECT_NAME = "Import formatting and aliasing hygiene"
TIER = 2
TOOLS = ["tree-sitter-python"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = [".py", ".pyi"]

# Canonical aliases agreed by community style guides / package docs.
# Source: scientific-Python stack, statsmodels, plotting, ML, web.
CANONICAL_ALIASES: Dict[str, str] = {
    "numpy": "np",
    "pandas": "pd",
    "tensorflow": "tf",
    "matplotlib.pyplot": "plt",
    "seaborn": "sns",
    "scipy": "sp",
    "statsmodels.api": "sm",
    "statsmodels.formula.api": "smf",
    "networkx": "nx",
    "plotly.express": "px",
    "plotly.graph_objects": "go",
    "altair": "alt",
    "polars": "pl",
    "dask.dataframe": "dd",
    "xarray": "xr",
    "sympy": "sym",
    "pytorch_lightning": "pl",  # also pl; collides w/ polars but both common
    "jax.numpy": "jnp",
    "holoviews": "hv",
    "geopandas": "gpd",
    "pyspark.sql.functions": "F",
    "torch.nn.functional": "F",
}

# Modules that *typically* go un-aliased; aliasing them is mildly suspicious.
COMMON_UNALIASED = frozenset({
    "os", "sys", "re", "json", "math", "logging", "io", "time", "typing",
    "subprocess", "shutil", "pathlib", "collections", "itertools",
    "functools", "asyncio", "argparse",
})

BUILTIN_NAMES = frozenset(dir(builtins))


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


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _is_cryptic_alias(alias: str, module: str) -> bool:
    """Heuristic: very short opaque alias unrelated to the module name."""
    if not alias:
        return False
    top = module.split(".")[-1].lower()
    a = alias.lower()
    if a == top:
        return False  # same name; not aliased meaningfully
    # 1-character aliases are normally bad unless they are an established
    # convention (`F`, `T`, etc.). Allow uppercase 1-letter only.
    if len(alias) == 1 and not alias.isupper():
        return True
    # 2-character alias that shares no letter prefix with module
    if len(alias) <= 2 and not top.startswith(a) and a not in top:
        # but allow exact canonical match
        return CANONICAL_ALIASES.get(module) != alias
    return False


def _walk_imports(root, src: bytes):
    """Yield records describing each top-level import statement.

    Each record:
      {
        kind: 'import' | 'from',
        module: str,                # source module (dotted)
        names: List[Tuple[str,str]],# (imported_name, local_bound_name)
        n_modules_on_line: int,     # for `import a, b, c`
        is_wildcard: bool,
        row: int,
      }
    """
    records = []
    for child in root.children:
        if child.type == "import_statement":
            # Children: 'import' keyword, then a sequence of dotted_name or
            # aliased_import separated by commas.
            modules_on_line = []  # list of (module, local)
            for n in child.children:
                if n.type == "dotted_name":
                    mod = _text(n, src)
                    modules_on_line.append((mod, mod.split(".")[0]))
                elif n.type == "aliased_import":
                    mod = ""
                    alias = ""
                    # children: dotted_name, 'as', identifier
                    for c in n.children:
                        if c.type == "dotted_name":
                            mod = _text(c, src)
                        elif c.type == "identifier":
                            alias = _text(c, src)
                    modules_on_line.append((mod, alias))
            if not modules_on_line:
                continue
            n_mods = len(modules_on_line)
            for mod, local in modules_on_line:
                records.append({
                    "kind": "import",
                    "module": mod,
                    "names": [(mod, local)],
                    "n_modules_on_line": n_mods,
                    "is_wildcard": False,
                    "row": child.start_point[0],
                })
        elif child.type == "import_from_statement":
            module_name = ""
            names: List[Tuple[str, str]] = []
            wildcard = False
            # children: 'from', dotted_name/relative_import, 'import',
            # then names: identifier / aliased_import / wildcard_import,
            # possibly inside parentheses.
            seen_import_kw = False
            for n in child.children:
                if n.type == "from":
                    continue
                if n.type == "import":
                    seen_import_kw = True
                    continue
                if not seen_import_kw:
                    if n.type in ("dotted_name", "relative_import"):
                        module_name = _text(n, src)
                else:
                    if n.type == "wildcard_import":
                        wildcard = True
                    elif n.type == "dotted_name":
                        nm = _text(n, src)
                        names.append((nm, nm))
                    elif n.type == "identifier":
                        nm = _text(n, src)
                        names.append((nm, nm))
                    elif n.type == "aliased_import":
                        ids = [c for c in n.children if c.type
                               in ("identifier", "dotted_name")]
                        if len(ids) >= 2:
                            names.append((_text(ids[0], src),
                                          _text(ids[1], src)))
                        elif ids:
                            t = _text(ids[0], src)
                            names.append((t, t))
            records.append({
                "kind": "from",
                "module": module_name,
                "names": names,
                "n_modules_on_line": 1,
                "is_wildcard": wildcard,
                "row": child.start_point[0],
            })
    return records


def _file_score(code: bytes) -> Optional[float]:
    parser = _get_parser()
    if parser is None:
        return None
    tree = parser.parse(code)
    root = tree.root_node
    if root.type not in ("module", "program"):
        return None
    recs = _walk_imports(root, code)
    if not recs:
        return None

    # (1) one-import-per-line: only the bare `import a, b` family
    multi_import_violations = sum(
        1 for r in recs
        if r["kind"] == "import" and r["n_modules_on_line"] > 1
    )
    n_import_lines = sum(1 for r in recs if r["kind"] == "import"
                         and r["n_modules_on_line"] > 0)
    if n_import_lines > 0:
        # Each violating module beyond the first on the line counts once.
        # Compute fraction of bare `import` records that lived on a multi
        # line; cap at 1.0.
        s1 = 1.0 - min(1.0, multi_import_violations
                       / max(n_import_lines, 1))
    else:
        s1 = None

    # (2) alias hygiene over aliased records (kind=='import' with alias != module
    # or kind=='from' with aliased_import).
    alias_pairs: List[Tuple[str, str]] = []
    for r in recs:
        for src_name, local in r["names"]:
            if local and src_name and local != src_name and \
                    local != src_name.split(".")[-1]:
                alias_pairs.append((src_name if r["kind"] == "import"
                                    else f"{r['module']}.{src_name}", local))
            elif r["kind"] == "import":
                # `import a.b.c as x` is captured above; also catch
                # `import foo as foo` (no-op alias) — record as healthy.
                if local and local != src_name and \
                        local == src_name.split(".")[-1]:
                    alias_pairs.append((src_name, local))
    if alias_pairs:
        bad = 0
        for mod, alias in alias_pairs:
            canonical = CANONICAL_ALIASES.get(mod)
            if canonical is not None:
                if canonical != alias:
                    bad += 1  # famous module, wrong alias
                continue
            if alias in BUILTIN_NAMES:
                bad += 1
                continue
            if _is_cryptic_alias(alias, mod):
                bad += 1
                continue
            # aliasing a stdlib utility module is mildly suspicious unless
            # the alias starts with the same letter
            top = mod.split(".")[0]
            if top in COMMON_UNALIASED and not alias.lower().startswith(
                    top[0]):
                bad += 0.5
        s2 = max(0.0, 1.0 - bad / len(alias_pairs))
    else:
        s2 = 1.0  # no aliases means no alias-quality penalty

    # (3) wildcard imports: each wildcard is a violation
    n_wild = sum(1 for r in recs if r["is_wildcard"])
    if n_wild == 0:
        s3 = 1.0
    else:
        s3 = max(0.0, 1.0 - n_wild / len(recs))

    # (4) shadowing: same local bound name from two records, or shadowing a
    # builtin
    locals_seen: Dict[str, int] = {}
    builtin_shadow = 0
    for r in recs:
        for _, local in r["names"]:
            if not local:
                continue
            locals_seen[local] = locals_seen.get(local, 0) + 1
            if local in BUILTIN_NAMES and local not in {"id", "type"}:
                # `id`/`type` are extremely common locals; many codebases
                # rebind them. Flag the more obviously wrong ones.
                builtin_shadow += 1
            elif local in {"list", "dict", "set", "str", "int", "float",
                           "input", "open", "map", "filter", "range",
                           "sum", "min", "max", "sorted", "reversed",
                           "iter", "next", "object", "vars", "bool"}:
                builtin_shadow += 1
    n_dups = sum(c - 1 for c in locals_seen.values() if c > 1)
    total_names = sum(len(r["names"]) for r in recs)
    if total_names > 0:
        s4 = max(0.0, 1.0 - (n_dups + builtin_shadow) / total_names)
    else:
        s4 = None

    sub = [s for s in (s1, s2, s3, s4) if s is not None]
    if not sub:
        return None
    return float(sum(sub) / len(sub))


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
