"""a408: Edit-distance to the per-problem LeetCode editorial solution.

The metric is narrowly applicable: it only fires on LeetCode-formatted
diffs whose path identifies the problem slug (e.g.
`leetcode/two-sum.py`) and whose extension identifies the language.
For arbitrary OSS PR diffs there is no per-problem reference, so
`applies(...)` returns False.

Approach:
  1. Load the per-(slug, language) editorial lookup at module import.
  2. Parse the diff with whatthepatch (via sandbox.parse_diff_added_by_file)
     to recover {path: added_lines}.
  3. From the path, attempt to identify a known editorial slug (filename
     stem, or any "leetcode/<slug>." segment).
  4. From the extension, map to a normalised language. Look up the
     canonical_code for (slug, language).
  5. Compute a token-level similarity using difflib.SequenceMatcher
     between the submitted code and the editorial code.
  6. Score = ratio in [0, 1] (1 = identical to editorial,
     0 = maximally different).

Why difflib over ZSS / APTED:
  - SequenceMatcher gives a well-behaved [0, 1] ratio out of the box
    and is multilingual (we have python/java/cpp/javascript/...).
  - Tree-edit-distance via ZSS is more principled but ties us to a
    per-language AST parser; for an already-narrow metric we prefer
    breadth across languages over depth on Python only.

CLASSIFICATION: PARTIALLY_THIN — the comparison is deterministic and the
editorial is a defensible canonical reference, but a single canonical
solution under-represents stylistic variance and tokenisation collapses
local symmetries an AST would preserve.
"""
from __future__ import annotations

import difflib
import re  # REGEX_OK: tool_output — splitting tokenizer output, not parsing diffs/code semantics.
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a408"
ASPECT_NAME = "Edit-distance to editorial solution"
TIER = 3
TOOLS = ["editorial-lookup"]
APPLIES_TO_LANGS = [
    "Python", "Java", "C++", "JavaScript", "TypeScript", "Go", "Rust",
]
CLASSIFICATION = "PARTIALLY_THIN"

# Lookup parquet built by scripts/leetcode_editorials/build_editorial_lookup.py.
_LOOKUP_PATH = (Path(__file__).resolve().parents[3]
                / "datasets/leetcode_editorials/editorial_by_slug.parquet")

# Map of file extensions to the normalised language tag used in the
# lookup parquet (matches build_editorial_lookup.LANG_NORM range).
EXT_TO_LANG = {
    ".py": "python", ".pyi": "python",
    ".java": "java",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".h": "cpp",
    ".c": "c",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin",
    ".sql": "sql",
}

# REGEX_OK: tool_output — splitting code into rough word/punct tokens for
# SequenceMatcher. We deliberately match the same pattern used by Python's
# tokenize fallbacks; this is not a parser.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")

_LOOKUP: Optional[Dict[Tuple[str, str], str]] = None


def _load_lookup() -> Dict[Tuple[str, str], str]:
    global _LOOKUP
    if _LOOKUP is not None:
        return _LOOKUP
    if not _LOOKUP_PATH.exists():
        _LOOKUP = {}
        return _LOOKUP
    try:
        import pandas as pd
        df = pd.read_parquet(_LOOKUP_PATH)
    except Exception:
        _LOOKUP = {}
        return _LOOKUP
    _LOOKUP = {
        (slug, lang): code
        for slug, lang, code in zip(
            df["question_slug"].tolist(),
            df["language"].tolist(),
            df["canonical_code"].tolist(),
        )
        if isinstance(code, str) and code
    }
    return _LOOKUP


# REGEX_OK: tool_output — slug grammar is a known character class
# (lowercase ASCII + digits + hyphens); this is shape validation, not
# semantic parsing of diffs/code.
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]+")


def _extract_slug_from_path(path: str, known_slugs: set) -> Optional[str]:
    """Recover the LC problem slug from a synthetic diff path.

    Recognises two conventions:
      - `leetcode/<slug>.<ext>` (and nested variants like
        `leetcode/python/<slug>.py`)
      - `<slug>.<ext>` directly, when stem already matches a known slug
    """
    p = path.strip().lstrip("./")
    # Strip leading a/ b/ from git path prefixes if any survived.
    if p.startswith(("a/", "b/")):
        p = p[2:]
    parts = p.split("/")
    stem = parts[-1].rsplit(".", 1)[0].lower()
    # Direct stem match first.
    if _SLUG_RE.fullmatch(stem) and stem in known_slugs:
        return stem
    # Then any path segment that looks slug-shaped and is a known slug.
    for seg in parts:
        seg_l = seg.lower()
        if _SLUG_RE.fullmatch(seg_l) and seg_l in known_slugs:
            return seg_l
    return None


def _identify(diff_text: str) -> Optional[Tuple[str, str, str, str]]:
    """Return (slug, language, submitted_code, editorial_code) or None.

    Returns the first identifiable (slug, language) tuple in the diff
    for which we have an editorial. If multiple files in the diff match,
    the longest submitted code wins (most informative comparison).
    """
    lookup = _load_lookup()
    if not lookup:
        return None
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None
    known_slugs = {s for (s, _l) in lookup.keys()}

    best: Optional[Tuple[str, str, str, str]] = None
    best_len = -1
    for path, added in by_path.items():
        slug = _extract_slug_from_path(path, known_slugs)
        if slug is None:
            continue
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = EXT_TO_LANG.get(ext)
        if lang is None:
            continue
        ed = lookup.get((slug, lang))
        if ed is None:
            continue
        if len(added) > best_len:
            best_len = len(added)
            best = (slug, lang, added, ed)
    return best


def applies(diff_text: str) -> bool:
    if not diff_text or "diff --git" not in diff_text:
        return False
    return _identify(diff_text) is not None


def _tokens(code: str) -> list:
    return _TOKEN_RE.findall(code or "")


def score(diff_text: str) -> Optional[float]:
    ident = _identify(diff_text)
    if ident is None:
        return None
    _slug, _lang, submitted, editorial = ident
    a = _tokens(submitted)
    b = _tokens(editorial)
    if not a or not b:
        return None
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ratio = sm.ratio()
    # Clamp defensively; SequenceMatcher already returns [0, 1].
    return max(0.0, min(1.0, float(ratio)))
