"""a502: Maximum loop nesting depth (brace/indent based).

Language-agnostic. Walks the candidate code and tracks the deepest nested
loop construct.

For C/C++: scans for `for (`/`while (` tokens, then tracks balanced braces
to figure out the lexical depth at which each loop sits.

For Python: scans for `for `/`while ` at the start of a line (after strip),
and computes depth from indentation: each 4-space level (or each tab) is
one depth unit.

Returns max_depth mapped to a univariate score: 1 / (1 + max_depth). So:
  depth 0 -> 1.00 (no loops — likely trivial or O(1) code)
  depth 1 -> 0.50
  depth 2 -> 0.33
  depth 3 -> 0.25
  depth 4 -> 0.20

Higher depth = lower score = "messier" or higher-asymptotic-cost code.

This is a function of the CANDIDATE CODE ALONE. CLASSIFICATION: THIN —
the brace/indent counter is deterministic.
"""
from __future__ import annotations

import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a502"
ASPECT_NAME = "Max loop nesting depth"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

_STR_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_C_COMMENT_LINE = re.compile(r"//.*$")
_C_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", flags=re.DOTALL)


def _strip_c(content: str) -> str:
    s = _C_COMMENT_BLOCK.sub("", content)
    s = "\n".join(_C_COMMENT_LINE.sub("", ln) for ln in s.splitlines())
    s = _STR_LIT.sub('""', s)
    return s


def _strip_py(content: str) -> str:
    # very rough: strip # comments and string literals
    out = []
    for ln in content.splitlines():
        s = _STR_LIT.sub('""', ln)
        idx = s.find("#")
        if idx >= 0:
            s = s[:idx]
        out.append(s)
    return "\n".join(out)


def _c_max_depth(text: str) -> int:
    """Track lexical brace depth, and the brace depth at which the nearest
    enclosing loop sits. We approximate: when we see `for(` or `while(`,
    we expect a `{` opening shortly after — increment loop_depth on next `{`
    and decrement when the matching `}` closes.
    """
    text = _strip_c(text)
    pending_loops: list = []  # stack of brace-depths where loops will open
    loop_active_depths: list = []  # currently-open loop brace depths
    brace_depth = 0
    max_depth = 0
    # Tokenize roughly: find keywords + braces in order.
    pat = re.compile(r"\bfor\b|\bwhile\b|\{|\}")
    for m in pat.finditer(text):
        tok = m.group(0)
        if tok in ("for", "while"):
            # require an opening paren shortly after
            tail = text[m.end():m.end() + 64].lstrip()
            if tail.startswith("("):
                pending_loops.append(brace_depth)
        elif tok == "{":
            brace_depth += 1
            # if a pending loop matched at brace_depth-1, mark it open
            if pending_loops and pending_loops[-1] == brace_depth - 1:
                pending_loops.pop()
                loop_active_depths.append(brace_depth)
                if len(loop_active_depths) > max_depth:
                    max_depth = len(loop_active_depths)
        elif tok == "}":
            # close any loop opened at brace_depth
            while loop_active_depths and loop_active_depths[-1] == brace_depth:
                loop_active_depths.pop()
            brace_depth = max(0, brace_depth - 1)
    return max_depth


def _py_max_depth(text: str) -> int:
    """Compute max indent-based loop nesting depth."""
    text = _strip_py(text)
    # We walk lines, push (indent, kind) when we see for/while, pop when
    # indent retreats to or below.
    stack: list = []  # list of indents of enclosing loops
    max_depth = 0
    for raw in text.splitlines():
        if not raw.strip():
            continue
        # compute indent in spaces (tabs = 4)
        ind = 0
        for ch in raw:
            if ch == " ":
                ind += 1
            elif ch == "\t":
                ind += 4
            else:
                break
        # pop stack entries with indent >= current line's indent (we exited)
        while stack and stack[-1] >= ind:
            stack.pop()
        stripped = raw.strip()
        if re.match(r"(for|while)\b", stripped):
            stack.append(ind)
            if len(stack) > max_depth:
                max_depth = len(stack)
    return max_depth


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    global_max = 0
    for path, content in by_path.items():
        if path.lower().endswith((".py", ".pyi")):
            d = _py_max_depth(content)
        else:
            d = _c_max_depth(content)
        if d > global_max:
            global_max = d
    return float(1.0 / (1.0 + global_max))
