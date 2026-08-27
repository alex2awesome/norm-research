"""a409: Comment-to-code ratio + comment quality (Python).

Relation to a50: a50 (Commenting strategy and quality) covers density,
TODO load, and useless-restatement across Python/JS/TS/Java/Go via
tree-sitter. THIS metric (a409) does NOT duplicate a50 — instead it adds
a tighter PYTHON-ONLY "explanatory-comment fraction" signal that a50
does not measure:

  Of all inline comments in the added Python code, what fraction CONTAIN
  an explanation marker — one of the words {"because", "to avoid",
  "edge case", "workaround", "TODO", "FIXME", "NOTE", "WHY", "REASON",
  "ASSUMPTION"}? Comments containing such a marker are evidence of
  intent-explaining commentary (the "why" not the "what").

Score per file = explanatory_fraction. We require >= 3 comments to score
(otherwise abstain — too noisy). Score per diff = mean across files.

Distinction from a50:
  - a50 returns a SHAPE score (density curve + TODO load + restatement);
    the same input that is 20% "why"-comments scores identically to one
    with 20% restatement comments as long as the density curve matches.
  - a409 looks AT the contents of comments for explanation markers; it
    is orthogonal to a50 by construction.

Examples (per file):
  + # increment counter
  + i += 1                                            -> 0 markers
  + # because the API returns offset-by-one indices, we shift here
  + i += 1                                            -> 1 explanation marker

CLASSIFICATION: PARTIALLY_THIN — marker-based detection misses comments
that explain intent without any of these keywords; conversely a comment
containing "because" purely conversationally would be a false positive.
Documented in the file because the call to a50's broader analysis is
the right thing to do for a coarser signal.
"""
from __future__ import annotations

# REGEX_OK: tool_output — string-membership tests on the inner text of a
# tree-sitter `comment` node. We are NOT parsing Python code with regex;
# we are checking a known string (the comment body, post-extraction).
import re
from typing import Dict, List, Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a409"
ASPECT_NAME = "Explanatory-comment fraction (intent comments)"
TIER = 2
TOOLS = ["tree-sitter-python"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = [".py", ".pyi"]

# REGEX_OK: tool_output — explanation-marker keyword set on comment prose.
EXPLAIN_MARKER = re.compile(
    r"\b(because|to avoid|edge case|workaround|TODO|FIXME|NOTE|WHY|REASON|"
    r"assumption|hack|caveat|known issue|gotcha)\b",
    re.IGNORECASE,
)

# REGEX_OK: tool_output — strip leading `#` from a tree-sitter comment node
# body. Operates on the extracted comment node text only.
_HASH_PREFIX = re.compile(r"^\s*#+\s?")

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        try:
            import tree_sitter_python
            from tree_sitter import Language, Parser
            _PARSER = Parser(Language(tree_sitter_python.language()))
        except ImportError:
            return None
    return _PARSER


def _file_explanatory_fraction(code: bytes) -> Optional[float]:
    parser = _get_parser()
    if parser is None:
        return None
    try:
        tree = parser.parse(code)
    except Exception:
        return None
    comments: List[str] = []

    def walk(n):
        if n.type == "comment":
            raw = code[n.start_byte:n.end_byte].decode(
                "utf8", errors="replace")
            stripped = _HASH_PREFIX.sub("", raw).strip()
            if stripped:
                comments.append(stripped)
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    if len(comments) < 3:
        return None
    explanatory = sum(1 for c in comments if EXPLAIN_MARKER.search(c))
    return explanatory / len(comments)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    if _get_parser() is None:
        return None
    scs: List[float] = []
    for content in by_path.values():
        s = _file_explanatory_fraction(
            content.encode("utf8", errors="replace"))
        if s is not None:
            scs.append(s)
    if not scs:
        return None
    return float(sum(scs) / len(scs))
