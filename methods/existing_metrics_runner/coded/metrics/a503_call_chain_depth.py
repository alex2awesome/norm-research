r"""a503: Function decomposition density / call-chain depth proxy.

Counts unique function definitions and a regex-based estimate of the
max call-chain depth (length of static call chain from main / entry).

For C/C++: function-definitions detected by regex on lines matching
  ^\s*(static\s+|inline\s+)?[\w:&\*<>\s]+\s+\w+\s*\([^;]*\)\s*\{
  (with simple tolerant pattern; not a parser).

For Python: count `def NAME(` lines.

We then build an adjacency map: for each defined name, look at which OTHER
defined names appear (as identifier calls) inside its body. The max depth
of the longest acyclic path from "main"/"solve"/module-level is reported.

Score: clamp(num_functions / 10, 0, 1) * 0.5 + clamp(max_depth / 6, 0, 1) * 0.5.
Higher = more decomposed code. CANDIDATE CODE ALONE.

CLASSIFICATION: PARTIALLY_THIN — regex function-detection is heuristic for
C++ (templates and macros can fool it), but the signal is shape-of-program.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional, Set

from ..sandbox import added_files_by_ext

ASPECT_ID = "a503"
ASPECT_NAME = "Call-chain depth / decomposition"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "PARTIALLY_THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

_STR_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
# C/C++ function def heuristic — opening brace at end-of-line after a paren list
_C_FN = re.compile(
    r"^[\t ]*"                              # indent
    r"(?:static\s+|inline\s+|virtual\s+|extern\s+|constexpr\s+)*"
    r"[\w:<>,&\* ]+?\s+"                    # return type (tolerant)
    r"(\w+)\s*\([^;{]*?\)\s*"               # name and arg list
    r"(?:const\s*)?(?:noexcept\s*)?"
    r"\{\s*$",
    flags=re.MULTILINE,
)
_PY_FN = re.compile(r"^[\t ]*def\s+(\w+)\s*\(", flags=re.MULTILINE)


def _extract_calls_in(body: str, names: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for m in re.finditer(r"\b(\w+)\s*\(", body):
        n = m.group(1)
        if n in names:
            out.add(n)
    return out


def _max_depth(adj, start: str, seen=None) -> int:
    if seen is None:
        seen = set()
    if start in seen:
        return 0
    seen = seen | {start}
    best = 0
    for n in adj.get(start, ()):
        d = _max_depth(adj, n, seen)
        if d > best:
            best = d
    return 1 + best


def _analyze_c(content: str):
    s = _STR_LIT.sub('""', content)
    fn_names = []
    fn_bodies = []
    # find each function and grab body by simple brace balance
    for m in _C_FN.finditer(s):
        name = m.group(1)
        start = m.end()
        depth = 1
        i = start
        L = len(s)
        while i < L and depth > 0:
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        fn_names.append(name)
        fn_bodies.append(s[start:i])
    return fn_names, fn_bodies


def _analyze_py(content: str):
    s = _STR_LIT.sub('""', content)
    fn_names = []
    fn_bodies = []
    matches = list(_PY_FN.finditer(s))
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        fn_names.append(name)
        fn_bodies.append(s[start:end])
    return fn_names, fn_bodies


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    all_names = []
    all_bodies = []
    for path, content in by_path.items():
        if path.lower().endswith((".py", ".pyi")):
            ns, bs = _analyze_py(content)
        else:
            ns, bs = _analyze_c(content)
        all_names.extend(ns)
        all_bodies.extend(bs)
    if not all_names:
        # no functions detected — abstain? for competition single-main code
        # we want a signal, so return mid (0.5 * 0 + 0.5 * 1) effectively 0.5
        return 0.0
    name_set = set(all_names)
    adj = defaultdict(set)
    for n, b in zip(all_names, all_bodies):
        for callee in _extract_calls_in(b, name_set):
            if callee != n:
                adj[n].add(callee)
    # pick a root: prefer 'main' / 'solve', else longest depth from any node
    roots = [n for n in name_set if n in ("main", "solve", "run")]
    if not roots:
        roots = list(name_set)
    max_d = 0
    for r in roots:
        d = _max_depth(adj, r)
        if d > max_d:
            max_d = d
    n_fns = len(name_set)
    a = min(1.0, n_fns / 10.0)
    b = min(1.0, max_d / 6.0)
    return float(0.5 * a + 0.5 * b)
