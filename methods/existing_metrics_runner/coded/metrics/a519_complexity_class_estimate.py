"""a519: Heuristic Big-O complexity-class bucket as an ordinal feature.

Univariate function of CANDIDATE CODE ALONE. Examines:
  - maximum loop nesting depth (brace counter for C/C++, indent for Python)
  - presence of recursion (a function that calls itself by name)
  - presence of memoization (dict-as-cache or @lru_cache / @cache)
  - presence of STL/data-structure operations implying complexity factors
    (sort, std::set/map, std::priority_queue, heapq, bisect)

Returns an integer bucket 1..7:
  1  constant            no loops, no recursion
  2  log                 single loop with /= 2 stride or recursion with /2
  3  linear              single loop, max nesting 1
  4  n log n             one outer loop + sort/map/log structure
  5  quadratic           nesting depth 2
  6  cubic+              nesting depth >= 3
  7  exponential         recursion without memo (Fib-style pattern)

applies() True iff at least one control-flow construct is present.

CLASSIFICATION: PARTIALLY_THIN — heuristic ordinal; deterministic given
text but not a rigorous complexity proof.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a519"
ASPECT_NAME = "Heuristic Big-O complexity bucket"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "PARTIALLY_THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

_STR_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

# Control-flow tokens (any presence triggers applies()).
_CTRL_ANY = re.compile(r"\b(for|while|if)\b")

# Loop tokens that should bump nesting in BOTH languages.
_LOOP_TOKEN = re.compile(r"\b(for|while)\b")

# Common sort / log-factor operations
_LOG_OPS = re.compile(
    r"\b(sort|sorted|std::sort|stable_sort|nth_element|partial_sort|"
    r"std::set|std::map|std::multiset|std::multimap|"
    r"set\s*<|map\s*<|multiset\s*<|multimap\s*<|"
    r"priority_queue|heapq|heappush|heappop|bisect|lower_bound|upper_bound)\b"
)

# Log-halving stride hints (for binary search / log loops)
_LOG_STRIDE = re.compile(
    r"(>>=\s*1|<<=\s*1|/=\s*2|//=\s*2|\*=\s*2|"
    r"mid\s*=\s*\(?\s*(?:lo|low|l)\s*\+\s*\(?\s*(?:hi|high|r)\s*-\s*"
    r"(?:lo|low|l)\s*\)?\s*/\s*2)"
)

# Memo tokens
_MEMO = re.compile(
    r"@lru_cache|@cache|@functools\.lru_cache|@functools\.cache|"
    r"\bmemo\b|memoiz|dp\[|cache\[|\bmemo\s*\[|"
    r"std::unordered_map|unordered_map\s*<|"
    r"std::map.*memo|std::unordered_set"
)

# Function-def patterns
_PY_FUNC = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
_CPP_FUNC = re.compile(
    r"^[ \t]*(?:static\s+|inline\s+|virtual\s+|constexpr\s+|template\s*<[^>]*>\s*)*"
    r"(?:[A-Za-z_][A-Za-z0-9_<>:,\s\*&]*\s+)+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{)]*\)\s*\{",
    re.MULTILINE)


def _strip(content: str) -> str:
    """Remove string literals and comments to avoid false matches inside them."""
    s = re.sub(r"/\*.*?\*/", " ", content, flags=re.DOTALL)
    s = _STR_LIT.sub(' "" ', s)
    out = []
    for ln in s.splitlines():
        for marker in ("//", "#"):
            idx = ln.find(marker)
            if idx >= 0:
                ln = ln[:idx]
        out.append(ln)
    return "\n".join(out)


def _max_brace_loop_depth(text: str) -> int:
    """Walk char-by-char, tracking brace depth and which braces opened a loop.

    Best-effort: when we see for/while keyword followed by a '(' ... ')'
    ... '{', the next '{' begins a loop scope. Pop depth on matching '}'.
    """
    i = 0
    n = len(text)
    max_depth = 0
    loop_stack = []  # bool per brace-level: True if that brace opens a loop
    pending_loop = False
    paren_depth = 0
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = j if j != -1 else n
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = j + 2 if j != -1 else n
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            if word in ("for", "while") and paren_depth == 0:
                pending_loop = True
            i = j
            continue
        if c == "(":
            paren_depth += 1
            i += 1
            continue
        if c == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue
        if c == "{":
            loop_stack.append(pending_loop)
            if pending_loop:
                cur = sum(1 for b in loop_stack if b)
                if cur > max_depth:
                    max_depth = cur
            pending_loop = False
            i += 1
            continue
        if c == "}":
            if loop_stack:
                loop_stack.pop()
            i += 1
            continue
        if c == ";" and paren_depth == 0:
            if pending_loop:
                cur = sum(1 for b in loop_stack if b) + 1
                if cur > max_depth:
                    max_depth = cur
            pending_loop = False
        i += 1
    return max_depth


def _max_indent_loop_depth(text: str) -> int:
    """Python-style: track indent of each `for/while` statement.

    Loop nesting depth = max number of for/while ancestors. We approximate
    by walking lines and maintaining a stack of (indent, is_loop) entries.
    """
    stack = []
    max_d = 0
    for ln in text.splitlines():
        stripped = ln.lstrip(" \t")
        if not stripped:
            continue
        indent = len(ln) - len(stripped)
        # pop stack entries whose indent >= current line's indent
        while stack and stack[-1][0] >= indent:
            stack.pop()
        # is this line a for/while header?
        is_loop = bool(re.match(r"(for|while)\b", stripped))
        # nesting = number of loop ancestors INCLUDING this if it's a loop
        loop_anc = sum(1 for _, ll in stack if ll)
        if is_loop:
            cur = loop_anc + 1
            if cur > max_d:
                max_d = cur
        stack.append((indent, is_loop))
    return max_d


def _has_self_recursion(text: str, is_py: bool) -> bool:
    """A function whose body contains a call to itself by name."""
    if is_py:
        funcs = _PY_FUNC.findall(text)
    else:
        funcs = _CPP_FUNC.findall(text)
    if not funcs:
        return False
    # Crude: does the function name appear MORE than once anywhere?
    # (once for the def, once for the recursive call)
    for name in funcs:
        # word-boundary match outside `def ` / `name (`
        pattern = re.compile(r"\b" + re.escape(name) + r"\s*\(")
        if len(pattern.findall(text)) >= 2:
            return True
    return False


def applies(diff_text: str) -> bool:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return False
    for content in by_path.values():
        if _CTRL_ANY.search(_strip(content)):
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    max_loop_depth = 0
    has_rec = False
    has_memo = False
    has_logops = False
    has_log_stride = False
    has_any_loop = False
    for path, content in by_path.items():
        is_py = path.lower().endswith((".py", ".pyi"))
        s = _strip(content)
        if _LOOP_TOKEN.search(s):
            has_any_loop = True
        if is_py:
            d = _max_indent_loop_depth(s)
        else:
            d = _max_brace_loop_depth(s)
        if d > max_loop_depth:
            max_loop_depth = d
        if _has_self_recursion(s, is_py):
            has_rec = True
        if _MEMO.search(s):
            has_memo = True
        if _LOG_OPS.search(s):
            has_logops = True
        if _LOG_STRIDE.search(s):
            has_log_stride = True

    # Bucket assignment — priority: deepest nesting first, then refine.
    if max_loop_depth >= 3:
        return 6.0  # cubic+
    if max_loop_depth == 2:
        return 5.0  # quadratic
    if has_rec and not has_memo:
        # exponential is plausible; downgrade to log if log-stride detected
        if has_log_stride:
            return 2.0
        return 7.0
    if max_loop_depth == 1 and (has_logops or has_log_stride):
        return 4.0  # n log n
    if max_loop_depth == 1:
        return 3.0  # linear
    if has_rec and has_memo:
        return 4.0  # memoized recursion ~ n log n bucket
    if has_rec:
        return 2.0  # log (recursion with halving or shallow)
    if not has_any_loop and not has_rec:
        return 1.0  # constant
    return 3.0
