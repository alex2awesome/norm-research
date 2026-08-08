"""a472: Python pedagogical step-narration comments.

Captures the "tutorial voice" where comments narrate each
mechanical step the code is about to take:

  # Initialize ...
  # Set ...
  # Mark ...
  # Pop ...
  # Push ...
  # Check if ...
  # Loop through ...
  # Iterate over ...
  # Update ...
  # Return ...

This register is common in beginner / educational solutions on
LeetCode but rare in editorial-style code, which prefers a single
strategy block at the top instead of per-statement narration.

Score: ratio of pedagogical-narration comments to total comment lines,
squashed with tanh so a fully narrated solution -> ~1, a comment-less
one -> 0. Pedagogical narration RATIO (not raw count) so a single
"# Initialize result" in an otherwise terse program does not get
flagged. Returns 0 if no comments at all.

Classification: THIN, register-only.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a472"
ASPECT_NAME = "Python pedagogical step-narration comments"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python"]
APPLIES_TO_EXTS = [".py", ".pyi"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

# Pedagogical-narration vocabulary -- imperative verbs typical of step-by-step
# tutorial commentary. The regex matches a "#" line whose first content word
# (after optional capitalization) is one of these verbs.
_NARRATE_VERBS = (
    "initialize", "init", "set", "create", "make", "build",
    "mark", "flag", "track", "store", "save", "keep",
    "pop", "push", "append", "add", "remove", "delete",
    "check", "test", "verify", "ensure", "validate", "confirm",
    "loop", "iterate", "traverse", "walk", "go",
    "update", "increment", "decrement",
    "return", "yield", "output", "print",
    "find", "search", "look", "scan",
    "compute", "calculate", "compare", "swap",
    "skip", "continue", "break", "stop",
    "start", "begin", "end",
    "handle", "process", "convert", "parse",
    "use", "apply", "call",
    "if",  # "# if x is None ..."
    "now", "then", "first", "second", "next", "finally",
)
_NARRATE_RX = re.compile(
    r"^[ \t]*#[ \t]*([A-Za-z]+)\b",
)


def _is_pedagogical(comment_word: str) -> bool:
    return comment_word.lower() in _NARRATE_VERBS


def _file_score(text: str) -> Optional[float]:
    if not text.strip():
        return None
    total_comment_lines = 0
    pedagogical = 0
    for line in text.split("\n"):
        m = _NARRATE_RX.match(line)
        if m is None:
            # also count bare "#" lines toward total but not pedagogical
            stripped = line.strip()
            if stripped.startswith("#"):
                total_comment_lines += 1
            continue
        total_comment_lines += 1
        if _is_pedagogical(m.group(1)):
            pedagogical += 1
    if total_comment_lines == 0:
        return 0.0
    ratio = pedagogical / total_comment_lines
    # weight by raw pedagogical count too -- a single pedagogical comment in 1
    # total comment is 1.0 ratio but weak signal; multiply by tanh(count/3).
    weight = math.tanh(pedagogical / 3.0)
    return float(ratio * weight)


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
