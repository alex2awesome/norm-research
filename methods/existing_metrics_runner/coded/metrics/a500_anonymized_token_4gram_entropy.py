"""a500: Anonymized-token 4-gram diversity (Shannon entropy).

Language-agnostic. Strips identifiers (replaces every Python/C++ identifier
token with a single placeholder `V`), keeps operators / punctuation / keywords
/ literals, then computes the Shannon entropy of the empirical distribution
over character 4-grams of the anonymized stream.

Univariate signal of "how mechanically simple is this code?":
  - very small entropy  : highly repetitive boilerplate (one-loop-over-array
                          style competition stubs)
  - mid                 : typical contest code with a few patterns
  - high                : richer mix of operators, control flow, idioms

This is a function of the CANDIDATE CODE ALONE — no editorial reference,
no pair-level computation. It is NOT a similarity metric. It's a structural
property of the code's anonymized token shape.

Score: entropy normalized to [0,1] by dividing by log2(256), then clamped.

CLASSIFICATION: THIN — entropy of a regex-defined anonymized stream is
deterministic given the input.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a500"
ASPECT_NAME = "Anonymized 4-gram entropy"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

# Common language keywords to PRESERVE (don't replace with V).
KEYWORDS = {
    # Python
    "def", "return", "if", "elif", "else", "for", "while", "in", "not",
    "and", "or", "is", "None", "True", "False", "import", "from", "as",
    "class", "with", "try", "except", "finally", "raise", "yield", "lambda",
    "pass", "break", "continue", "global", "nonlocal", "assert", "del",
    # C/C++
    "auto", "int", "long", "short", "char", "void", "bool", "float", "double",
    "unsigned", "signed", "const", "static", "struct", "class", "namespace",
    "using", "typedef", "template", "typename", "public", "private",
    "protected", "virtual", "override", "final", "new", "delete", "this",
    "nullptr", "true", "false", "switch", "case", "default", "do", "goto",
    "sizeof", "extern", "inline", "explicit", "operator", "friend", "mutable",
    "register", "volatile", "throw", "catch", "try", "size_t",
}

# Identifier pattern.
_IDENT = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
# String / char literal pattern (collapse to a single token to avoid letting
# string contents contaminate the n-gram distribution).
_STR_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def _anonymize(text: str) -> str:
    """Replace identifiers with V (preserving keywords) and string lits with S."""
    text = _STR_LIT.sub("S", text)

    def _sub(m: re.Match) -> str:
        tok = m.group(0)
        return tok if tok in KEYWORDS else "V"

    text = _IDENT.sub(_sub, text)
    # collapse runs of whitespace to single space — we care about structural
    # token shape, not formatting
    text = re.sub(r"\s+", " ", text)
    return text


def _ngram_entropy(stream: str, n: int = 4) -> float:
    if len(stream) < n:
        return 0.0
    cnt: Counter = Counter()
    for i in range(len(stream) - n + 1):
        cnt[stream[i:i + n]] += 1
    total = sum(cnt.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in cnt.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    streams = []
    for content in by_path.values():
        streams.append(_anonymize(content))
    if not streams:
        return None
    joined = "\n".join(streams)
    h = _ngram_entropy(joined, n=4)
    # log2(256) = 8 — typical upper bound for a byte-level alphabet
    return float(max(0.0, min(1.0, h / 8.0)))
