"""a470: PEP-8 import organization, review-post-aware.

Replacement for a135 (left in place, unchanged). The difference is that
a135 expected imports at the *top* of a single contiguous file. A
CR.SE answer is often a multi-block review post with code interleaved
by prose; tree-sitter sees only the concatenated code blocks, so a135
unfairly penalizes the "show the imports next to where they are used"
pattern.

This metric instead checks:

  1. WITHIN each contiguous import run, are imports grouped
     stdlib -> third-party -> local in order? (No interleaving.)
  2. Are stdlib / third-party / local clusters present? (Reward
     answers that demonstrate awareness of the three-tier convention
     even if they don't put them all at the top.)
  3. No obviously-unused imports.

Score = mean of three sub-scores, each in [0, 1]. Abstains if no
imports.

Classification: THIN.
"""
from __future__ import annotations

import ast
import re
import sys
from typing import List, Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a470"
ASPECT_NAME = "PEP-8 import organization (review-post-aware)"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

STDLIB = frozenset(getattr(sys, "stdlib_module_names", set()) or {
    "os", "sys", "re", "math", "json", "typing", "pathlib", "collections",
    "itertools", "functools", "dataclasses", "subprocess", "shutil",
    "tempfile", "io", "logging", "datetime", "time", "string", "abc",
    "asyncio", "enum", "warnings", "unittest", "argparse", "heapq",
    "bisect", "operator", "statistics", "fractions",
})

_RANK = {"stdlib": 0, "third": 1, "local": 2}


def _classify(mod: str) -> str:
    if not mod:
        return "third"
    if mod.startswith("."):
        return "local"
    top = mod.split(".", 1)[0]
    return "stdlib" if top in STDLIB else "third"


def _collect_imports(tree: ast.AST) -> List[Tuple[int, str, List[str]]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                grp = _classify(alias.name)
                out.append((node.lineno, grp,
                             [alias.asname or alias.name.split(".")[0]]))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level and node.level > 0:
                grp = "local"
            else:
                grp = _classify(mod)
            names = [a.asname or a.name for a in node.names]
            out.append((node.lineno, grp, names))
    out.sort(key=lambda x: x[0])
    return out


def _run_ordering_score(imports) -> float:
    """For each contiguous run (gap <=2 lines), check ordering."""
    if not imports:
        return 1.0
    runs = [[imports[0]]]
    for prev, cur in zip(imports, imports[1:]):
        if cur[0] - prev[0] <= 3:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    bad = 0
    pairs = 0
    for run in runs:
        for a, b in zip(run, run[1:]):
            pairs += 1
            if _RANK[a[1]] > _RANK[b[1]]:
                bad += 1
    if pairs == 0:
        return 1.0
    return 1.0 - bad / pairs


def _three_tier_awareness(imports) -> float:
    """Bonus if 2+ tiers represented (shows awareness of stdlib vs third)."""
    if not imports:
        return 1.0
    groups = {g for _, g, _ in imports}
    return min(1.0, len(groups) / 2.0)


def _unused_score(tree: ast.AST, imports) -> float:
    bound = set()
    for _, _, names in imports:
        for n in names:
            if n and n != "*":
                bound.add(n)
    if not bound:
        return 1.0
    # collect identifiers from NON-import nodes
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            # walk to the root name
            v = node.value
            while isinstance(v, ast.Attribute):
                v = v.value
            if isinstance(v, ast.Name):
                used.add(v.id)
    unused = sum(1 for b in bound if b not in used)
    return 1.0 - unused / len(bound)


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    imports = _collect_imports(tree)
    if not imports:
        return None
    s1 = _run_ordering_score(imports)
    s2 = _three_tier_awareness(imports)
    s3 = _unused_score(tree, imports)
    return float((s1 + s2 + s3) / 3.0)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    scs = [s for s in (_file_score(c) for c in by_path.values()) if s is not None]
    if not scs:
        return None
    return float(sum(scs) / len(scs))
