"""a479: Fraction of Python imports that resolve to known modules.

Walks the AST for `Import` and `ImportFrom` nodes, extracts the top-level
module name, and computes the fraction that live in the Python standard
library or a curated set of common third-party scientific/algorithmic
packages.

Flags solutions that use made-up modules, broken renames, or one-off
helper modules that won't import on a clean interpreter.

Returns:
  - None if no Python source in the diff or no imports parsed.
  - In [0, 1]: recognized / total imports.

Classification: THIN. Trivial cost.
"""
from __future__ import annotations

import ast
import sys
from typing import Optional, Set

from ..sandbox import added_files_by_ext

ASPECT_ID = "a479"
ASPECT_NAME = "Python imports resolve to known modules"
TIER = 1
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]


# Curated allow-list of common third-party packages seen on LeetCode /
# competitive programming Python answers.
_THIRD_PARTY: Set[str] = {
    "numpy", "np", "scipy", "pandas", "pd",
    "sortedcontainers", "more_itertools", "blist",
    "networkx", "sympy", "matplotlib", "pytest",
    "tqdm", "regex", "z3", "pulp", "cvxpy",
    "intervaltree", "bitarray", "pybind11",
    "torch", "tensorflow", "keras", "sklearn",
    "fractions",  # actually stdlib, kept for safety
    "decimal",    # actually stdlib, kept for safety
}


def _stdlib_set() -> Set[str]:
    # sys.stdlib_module_names is available in Python 3.10+.
    names = set(getattr(sys, "stdlib_module_names", ()) or ())
    # Common ones in case the above is empty (defensive).
    names.update({
        "abc", "argparse", "array", "ast", "asyncio", "base64", "binascii",
        "bisect", "builtins", "calendar", "cmath", "collections", "colorsys",
        "concurrent", "contextlib", "copy", "csv", "ctypes", "datetime",
        "decimal", "dis", "enum", "errno", "fnmatch", "fractions", "functools",
        "gc", "getopt", "glob", "gzip", "hashlib", "heapq", "hmac", "html",
        "http", "importlib", "inspect", "io", "ipaddress", "itertools", "json",
        "logging", "math", "multiprocessing", "numbers", "operator", "os",
        "pathlib", "pickle", "pprint", "queue", "random", "re", "shutil",
        "signal", "socket", "sqlite3", "statistics", "string", "struct",
        "subprocess", "sys", "tempfile", "textwrap", "threading", "time",
        "timeit", "token", "tokenize", "trace", "traceback", "types", "typing",
        "unicodedata", "unittest", "urllib", "uuid", "warnings", "weakref",
        "xml", "zipfile", "zlib", "__future__",
    })
    return names


_STDLIB = _stdlib_set()


def _file_score(text: str) -> Optional[float]:
    try:
        tree = ast.parse(text)
    except Exception:
        return None
    total = 0
    ok = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                total += 1
                if top in _STDLIB or top in _THIRD_PARTY:
                    ok += 1
        elif isinstance(node, ast.ImportFrom):
            # node.module can be None for `from . import x`
            if node.module is None:
                continue  # relative import; not classifiable
            top = node.module.split(".", 1)[0]
            total += 1
            if top in _STDLIB or top in _THIRD_PARTY:
                ok += 1
    if total == 0:
        return None
    return ok / total


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    vals = [v for v in (_file_score(c) for c in by_path.values()) if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))
