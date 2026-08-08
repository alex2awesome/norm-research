"""a367: Source file encoding policy (UTF-8, BOM).

The norm: source files should be valid UTF-8 *without* a byte-order mark
(BOM). The BOM (U+FEFF / bytes 0xEF 0xBB 0xBF) is technically legal in
UTF-8 but is widely discouraged in source code because many toolchains
(POSIX shells, Python 2, Go, older compilers) misinterpret or refuse it,
and most language style guides explicitly forbid it (PEP 263, Google
style guides, EditorConfig defaults, etc.).

This is a byte-format check, not a code-semantics check, so no parser is
needed. We score each *new file* added in the diff: a BOM character at
position 0 of its added content counts as a violation. We do not flag the
BOM on file *modifications* because modifications don't show line 1 unless
the hunk touches it — over-flagging would produce noise.

UTF-8 validity itself isn't testable here: the diff is already-decoded
text (Python str), so any byte sequence that wasn't valid UTF-8 would have
been replaced upstream by the diff producer. We document this and only
check the in-band signal we can see, which is the BOM character.

Score = (new files without leading BOM) / (new files added).
- applies() = True iff the diff adds at least one new source file.
- score() returns 1.0 when no BOMs are present, decays toward 0 with the
  fraction of new files that begin with U+FEFF.
- Returns None when no new source files are added (router-gated).
"""
from __future__ import annotations

from typing import Optional

import whatthepatch

ASPECT_ID = "a367"
ASPECT_NAME = "Source file encoding policy (UTF-8, BOM)"
TIER = 2
TOOLS = []  # pure byte/char inspection of diff text
APPLIES_TO_LANGS = [
    "Python", "JavaScript", "TypeScript", "Java", "Go", "C", "C++", "C#",
    "Ruby", "PHP", "Rust", "Kotlin", "Scala", "Swift",
]
CLASSIFICATION = "THIN"

# Source-file extensions where a BOM in the file's first bytes matters.
# Markdown / JSON / YAML *also* care about BOMs but are less consistently a
# "source code" norm; we restrict to programming-language sources here.
SOURCE_EXTS = (
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx",
    ".cs",
    ".kt", ".kts",
    ".scala",
    ".php",
    ".swift",
)

BOM_CHAR = "﻿"  # the BOM as a Python Unicode character (what whatthepatch yields)


def _parse_new_source_files(diff_text: str):
    """Yield (path, first_added_line_or_None) for each NEW source file added
    by this diff. New = old_path is /dev/null.
    """
    idx = diff_text.find("diff --git")
    if idx == -1:
        return
    try:
        diffs = whatthepatch.parse_patch(diff_text[idx:])
    except Exception:
        return
    for d in diffs:
        if d is None or d.header is None:
            continue
        old_path = d.header.old_path or ""
        new_path = d.header.new_path or ""
        # New file: old side is /dev/null (or empty).
        is_new = old_path in ("/dev/null", "") and new_path not in ("", "/dev/null")
        if not is_new:
            continue
        path = new_path[2:] if new_path.startswith("b/") else new_path
        if not any(path.lower().endswith(e) for e in SOURCE_EXTS):
            continue
        # First added line of the new file.
        first_line = None
        for ch in (d.changes or []):
            if ch.old is None and ch.new is not None and ch.line is not None:
                first_line = ch.line
                break
        yield path, first_line


def applies(diff_text: str) -> bool:
    for _path, _line in _parse_new_source_files(diff_text):
        return True
    return False


def score(diff_text: str) -> Optional[float]:
    n_new = 0
    n_clean = 0
    for _path, first_line in _parse_new_source_files(diff_text):
        n_new += 1
        if first_line is None:
            # Hunk header existed but no added line was visible — be
            # conservative and treat as clean (BOM detection is impossible
            # without the first byte).
            n_clean += 1
            continue
        # UTF-8 round-trip: the line is a Python str. encode/decode in
        # strict mode will always succeed on a str (no surrogates expected
        # in diff text), so this is a no-op safety check that documents
        # the contract.
        try:
            first_line.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            # Cannot encode (lone surrogate) -> count as non-clean.
            continue
        if first_line.startswith(BOM_CHAR):
            # BOM present at file start -> violation.
            continue
        n_clean += 1
    if n_new == 0:
        return None
    return float(n_clean / n_new)
