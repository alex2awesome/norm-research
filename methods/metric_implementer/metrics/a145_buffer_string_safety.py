"""a145: Buffer and string handling safety.

Scope. The norm itself is about C-style memory safety: correctly sized
buffers (including null terminator), bounds-checked operations, length-aware
APIs, banned hazardous functions. In memory-safe languages (Python, JS/TS,
Java, Go) strings are length-prefixed and bounds-checked by the runtime, so
the norm is trivially satisfied — there is no surface for buffer overflow
in the same sense. We therefore APPLY only to C/C++ source files.

Tools. Two complementary signals on the added C/C++ lines:

  1. ``cppcheck`` (Tier 3, C/C++ static analyzer). We run with
     ``--enable=warning,style --inconclusive`` and count findings whose IDs
     are in BUFFER_RELATED_IDS (bufferAccessOutOfBounds, arrayIndexOutOfBounds,
     getsCalled, invalidscanf, sprintfOverlappingData, etc.). cppcheck
     handles a subset of the norm — notably ``gets``, unbounded ``scanf``,
     and some array-bounds reasoning — but does NOT by default flag the
     classic ``strcpy``/``strcat``/``sprintf`` family.

  2. Token-based ban list (supplements cppcheck). On added lines only, we
     tokenize C/C++ source with a real C tokenizer (``pygments``'s
     CLexer when available, fallback to a small hand-written tokenizer)
     and count occurrences of identifiers in BANNED_FUNCS used as call
     targets. This is NOT regex-on-code: tokens are real lexer tokens, so
     the matcher ignores occurrences inside comments and string literals.

Classification. ``PARTIALLY_THIN``. cppcheck is a real tool but only
catches a slice of the norm; the banned-function check is deterministic but
covers a fixed list. The norm itself has a thick edge (custom buffer
arithmetic, signed/unsigned conversions that a static analyzer can miss).
Applicability is VERY narrow — most fixtures have no C/C++ at all.

Score. Per added C/C++ line, weight cppcheck buffer-related findings by
severity (LOW=1, MEDIUM=2, HIGH=4, error=4, warning=2, style=1, portability=1,
performance=1) and add 2.0 per banned-function call site. Density is
weighted / max(added_c_lines, 1). Score = exp(-density * 6.0).
  - 0 findings, 0 banned calls            -> 1.0
  - 1 strcpy / 40 LOC = 2/40              -> exp(-0.30) = 0.741
  - 1 gets   / 20 LOC = 2/20              -> exp(-0.60) = 0.549
  - 3 strcpy + 1 gets / 50 LOC = 8/50     -> exp(-0.96) = 0.383

Return states:
  - applies()=False on PRs with no C/C++ files added (the common case)
  - score()=None when applies()=True but cppcheck is missing
  - score() in [0,1] otherwise

Note. C/C++ is not covered by any other metric in this catalog at the time
of writing, so a145's narrow applicability is the ONLY signal these PRs
will get from this norm.
"""
from __future__ import annotations

import math
import shutil
from typing import List, Optional, Set, Tuple

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a145"
ASPECT_NAME = "Buffer and string handling safety"
TIER = 3
TOOLS = ["cppcheck"]
APPLIES_TO_LANGS = ["C", "C++"]
CLASSIFICATION = "PARTIALLY_THIN"

C_EXTS = [".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".c++"]

# Cppcheck IDs we treat as buffer/string-safety findings.
BUFFER_RELATED_IDS: Set[str] = {
    "bufferAccessOutOfBounds",
    "arrayIndexOutOfBounds",
    "arrayIndexOutOfBoundsCond",
    "arrayIndexThenCheck",
    "negativeIndex",
    "negativeMemoryAllocationSize",
    "outOfBounds",
    "pointerOutOfBounds",
    "containerOutOfBounds",
    "stringLiteralWrite",
    "bufferAccessOutOfBoundsCond",
    "sprintfOverlappingData",
    "getsCalled",
    "scanfWithoutWidth",
    "invalidscanf",
    "invalidscanf_libc",
    "wrongPipeBufferSize",
    "uninitstring",
    "terminateStrncpy",
    "bufferNotZeroTerminated",
    "AssignSize",  # paranoid
    "memsetZeroBytes",
    "memsetFloat",
    "memsetValueOutOfRange",
    "deallocuse",
    "doubleFree",
    "memleak",  # adjacent class — kept since memleak often co-occurs with bad buf math
    "uninitvar",  # uninitialized buffer
    "signConversion",  # signed/unsigned conversion (called out in description)
    "integerOverflow",
    "shiftTooManyBitsSigned",
}

# Hazardous functions (the "ban these" half of the norm).
# Listed as call-target identifiers; lexer makes the match token-precise.
BANNED_FUNCS: Set[str] = {
    "gets",
    "strcpy",
    "strcat",
    "sprintf",
    "vsprintf",
    "scanf",      # without width
    "sscanf",     # without width — heuristic; safer forms exist
    "wcscpy",
    "wcscat",
    "swprintf",
    "_strcpy", "_strcat",  # MS underscore variants
    "stpcpy",
    "lstrcpy",
    "lstrcat",
}

# Severity -> weight for cppcheck issue weighting.
SEV_W = {
    "error": 4.0,
    "warning": 2.0,
    "style": 1.0,
    "portability": 1.0,
    "performance": 1.0,
    "information": 0.0,  # information findings are noise (missing include etc.)
    "LOW": 1.0, "MEDIUM": 2.0, "HIGH": 4.0,
}


# ---------- tokenizer ----------

def _tokenize_c(code: str) -> List[Tuple[str, str]]:
    """Tokenize C/C++ source. Prefer pygments' CLexer; fall back to a small
    hand-rolled tokenizer that strips comments and string/char literals,
    yielding (token_type, value) pairs. We only need 'name' tokens for the
    banned-call check.
    """
    try:
        from pygments.lexers.c_cpp import CppLexer
        from pygments.token import Name, Token
        lexer = CppLexer(stripnl=False)
        out: List[Tuple[str, str]] = []
        for tok_type, value in lexer.get_tokens(code):
            if tok_type in Name or tok_type is Name.Function or tok_type is Name.Builtin:
                out.append(("name", value))
            elif value == "(":
                out.append(("lparen", value))
            elif value == ")":
                out.append(("rparen", value))
        return out
    except Exception:
        pass

    # Fallback: hand-roll. Strip block + line comments and string/char
    # literals, then walk identifier-like tokens.
    s = []
    i = 0
    n = len(code)
    while i < n:
        c = code[i]
        if c == "/" and i + 1 < n and code[i + 1] == "/":
            j = code.find("\n", i)
            i = n if j == -1 else j
            continue
        if c == "/" and i + 1 < n and code[i + 1] == "*":
            j = code.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if c == '"' or c == "'":
            quote = c
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == quote:
                    j += 1
                    break
                j += 1
            i = j
            continue
        s.append(c)
        i += 1
    cleaned = "".join(s)

    out: List[Tuple[str, str]] = []
    buf = []
    for ch in cleaned:
        if ch.isalnum() or ch == "_":
            buf.append(ch)
        else:
            if buf:
                out.append(("name", "".join(buf)))
                buf = []
            if ch == "(":
                out.append(("lparen", ch))
            elif ch == ")":
                out.append(("rparen", ch))
    if buf:
        out.append(("name", "".join(buf)))
    return out


def _count_banned_calls(code: str) -> int:
    """Count call-sites of banned functions in `code`. A call-site is an
    identifier in BANNED_FUNCS immediately followed by `(`. Names inside
    comments / strings are excluded by the tokenizer.
    """
    toks = _tokenize_c(code)
    n = 0
    for idx, (kind, val) in enumerate(toks):
        if kind == "name" and val in BANNED_FUNCS:
            if idx + 1 < len(toks) and toks[idx + 1][0] == "lparen":
                n += 1
    return n


# ---------- cppcheck ----------

def _cppcheck_weight(td: str) -> Optional[float]:
    """Run cppcheck on directory; return weighted buffer-related issue sum,
    or None on tool error."""
    rc, out, err = run(
        ["cppcheck",
         "--enable=warning,style,portability,performance",
         "--inconclusive",
         "--quiet",
         "--template={severity}|{id}|{file}|{line}|{message}",
         td],
        timeout=25.0,
    )
    if rc < 0:
        return None
    # cppcheck writes findings to stderr with --template format; rc is 0 by
    # default whether or not issues are found unless --error-exitcode is set.
    text = (err or "") + "\n" + (out or "")
    weighted = 0.0
    for line in text.splitlines():
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        sev, cid, _file, _ln, _msg = parts
        if cid not in BUFFER_RELATED_IDS:
            continue
        weighted += SEV_W.get(sev, 1.0)
    return weighted


# ---------- public API ----------

def applies(diff_text: str) -> bool:
    """Applies iff the diff adds any C/C++ source lines."""
    return bool(added_files_by_ext(diff_text, C_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, C_EXTS)
    if not by_path:
        return None
    total_lines = sum(1 + t.count("\n") for t in by_path.values())
    if total_lines == 0:
        return None

    # Banned-function calls in added lines — token-precise, lexer-based.
    banned_total = 0
    for body in by_path.values():
        banned_total += _count_banned_calls(body)

    cppcheck_w = 0.0
    if have_tool("cppcheck"):
        td = write_temp_files(by_path)
        try:
            w = _cppcheck_weight(str(td))
            if w is None:
                # Tool failure: we still have the banned-call signal, but
                # honest abstention is preferred per the GUIDE.
                return None
            cppcheck_w = w
        finally:
            shutil.rmtree(td, ignore_errors=True)
    else:
        # Tool unavailable -> abstain rather than ship a banned-call-only score
        # that would be incomparable across hosts.
        return None

    # Each banned-call gets weight 2.0 (≈ cppcheck WARNING severity).
    total_w = cppcheck_w + 2.0 * banned_total
    density = total_w / max(total_lines, 1)
    return float(math.exp(-density * 6.0))
