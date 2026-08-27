"""a15: Comments are clear, intentional, and WHY-focused (rationale).

What we measure
---------------
The aspect says: "Prefer self-explanatory code; use comments sparingly to
capture INTENT, RATIONALE, TRADE-OFFS, and non-obvious context. Write clear,
useful English, avoid restating the code, and keep comments accurate."

a50 (sibling metric) covers commenting STRATEGY at a surface level: density
shape, TODO load, and an "obvious restatement" heuristic. a15 takes a
DIFFERENT angle that a50 does not measure: among the comments that exist,
what fraction are oriented toward EXPLAINING WHY (rationale, trade-offs,
intent) versus describing what the code already says? A pure "increment i"
comment fails this norm. A comment containing "because", "to avoid X",
"since the upstream library", "we picked Y over Z because", "trade-off:"
satisfies it.

Signals
-------
We collect inline comments using tree-sitter (same languages as a50:
Python, JS, TS, Java, Go) and exclude docstrings / JSDoc / Javadoc (those
are documentation, handled by a202).

For each inline comment we compute two flags:

  1. RATIONALE marker — the comment contains at least one rationale
     keyword/phrase. The vocabulary is deliberately broad and conservative:
     "because", "since", "so that", "in order to", "to avoid", "to ensure",
     "to prevent", "to handle", "to allow", "to support", "due to",
     "given that", "otherwise", "trade-off", "tradeoff", "rationale",
     "reason", "intent", "intentional", "intentionally", "we chose",
     "we picked", "we prefer", "we use", "we need", "we must",
     "this is needed", "this is because", "note that", "fixes", "fixme",
     "workaround", "hack", "see ", "ref:", "context:", "why ", "assumes",
     "assumption", "invariant", "must ", "should ", "should not", "must not",
     "do not", "don't", "cannot", "deprecated", "legacy", "performance",
     "perf", "thread-safe", "thread safety", "race", "concurrency",
     "edge case", "corner case", "bug", "issue", "spec ", "rfc ",
     "compat", "compatibility", "backwards", "forward-compat", "for now",
     "until ", "temporarily", "TODO".

  2. RESTATEMENT — short comment (<= 6 words) whose identifier-shape tokens
     all appear in the next non-blank code line. Same detector as a50's
     useless-comment heuristic.

We aggregate across all parsed files in the diff:

    n_comments = total inline comments
    n_rationale = # of comments with a rationale marker
    n_restate   = # of restatement-style comments

    rationale_rate = n_rationale / n_comments
    restate_rate   = n_restate   / n_comments
    why_score      = rationale_rate - restate_rate   # in [-1, 1]
    score          = 0.5 * (why_score + 1.0)         # squashed to [0, 1]

A diff with zero inline comments cannot be measured here (the norm is about
the *content* of comments that exist), so we abstain with `None` instead of
emitting a neutral 0.5. This is consistent with the implementer's rule:
"abstain rather than emit unreliable 0.5 noise."

Classification
--------------
PARTIALLY_THIN. A keyword vocabulary cannot decide whether a "because"
comment is *actually* well-reasoned, accurate, or sufficient — only that
the author oriented the prose toward rationale. The restatement detector
is heuristic and identifier-shape only. Both signals are surface
realizations of the underlying norm, not direct measurements of it.

Distinct from a50
-----------------
a50 outputs a single number combining (density, TODO load, restatement).
a15 conditions on comments existing and measures CONTENT orientation:
the fraction of those comments that are rationale-bearing versus pure
restatement. The numerator vocabularies do not overlap: a50 has no
rationale keyword list; a15 has no density shaping term and no TODO
sub-score. The restatement signal is computed identically in both, but in
a15 it appears as a NEGATIVE component of the why-vs-what balance, not as
a standalone term.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a15"
ASPECT_NAME = "Comments: clear, intentional, why-focused"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# Rationale / intent markers. Lower-cased substring match on stripped
# comment text. Kept broad but each entry is a phrase that, when written
# in a code comment, materially signals *why* rather than *what*.
RATIONALE_PHRASES = (
    "because", "since ", "so that", "in order to", "to avoid",
    "to ensure", "to prevent", "to handle", "to allow", "to support",
    "due to", "given that", "otherwise", "trade-off", "tradeoff",
    "rationale", "reason", "intent", "intentional", "intentionally",
    "we chose", "we picked", "we prefer", "we use", "we need",
    "we must", "this is needed", "this is because", "note that",
    "workaround", "hack", " see ", "ref:", "context:", "why ",
    "assumes", "assumption", "invariant", "must ", "should ",
    "should not", "must not", "do not ", "don't ", "cannot",
    "deprecated", "legacy", "performance", "perf ", "thread-safe",
    "thread safety", "race", "concurrency", "edge case",
    "corner case", "bug ", "issue ", "spec ", "rfc ", "compat",
    "compatibility", "backwards", "forward-compat", "for now",
    "until ", "temporarily", "todo", "fixme",
)

# REGEX_OK: tool_output — strip "#" prefix from a tree-sitter comment node.
_PY_COMMENT_PREFIX = re.compile(r"^\s*#+\s?")
# REGEX_OK: tool_output — strip "//" prefix from a tree-sitter comment node.
_C_LINE_COMMENT_PREFIX = re.compile(r"^\s*//+\s?")
# REGEX_OK: tool_output — identifier-shape word extraction over comment
# prose (already extracted by tree-sitter), not over source code.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

_PARSERS: Dict[str, object] = {}


def _get_parser(lang_short: str):
    if lang_short in _PARSERS:
        return _PARSERS[lang_short]
    try:
        from tree_sitter import Language, Parser
        if lang_short == "py":
            import tree_sitter_python as m
            lang = m.language()
        elif lang_short == "js":
            import tree_sitter_javascript as m
            lang = m.language()
        elif lang_short == "ts":
            import tree_sitter_typescript as m
            lang = m.language_typescript()
        elif lang_short == "java":
            import tree_sitter_java as m
            lang = m.language()
        elif lang_short == "go":
            import tree_sitter_go as m
            lang = m.language()
        else:
            return None
        _PARSERS[lang_short] = Parser(Language(lang))
        return _PARSERS[lang_short]
    except ImportError:
        return None


def _strip_comment_text(raw: str, lang: str) -> str:
    if lang == "py":
        return _PY_COMMENT_PREFIX.sub("", raw).strip()
    s = raw.strip()
    if s.startswith("//"):
        return _C_LINE_COMMENT_PREFIX.sub("", raw).strip()
    if s.startswith("/*"):
        inner = s
        # REGEX_OK: tool_output — strip /* ... */ fences from a tree-sitter
        # block_comment node before reading inner prose.
        inner = re.sub(r"^/\*+", "", inner)
        inner = re.sub(r"\*+/$", "", inner)
        # REGEX_OK: tool_output — strip per-line "*" continuation marker
        # inside a block comment body (Javadoc/JSDoc convention).
        lines = [re.sub(r"^\s*\*+\s?", "", ln) for ln in inner.splitlines()]
        return " ".join(s2.strip() for s2 in lines).strip()
    return s.strip()


def _is_doc_comment(node, source: bytes, lang: str) -> bool:
    if lang == "py":
        return False
    raw = source[node.start_byte:node.end_byte]
    return raw.startswith(b"/**")


def _collect_inline_comments(code: bytes, lang: str) -> List[Tuple[int, str]]:
    parser = _get_parser(lang)
    if parser is None:
        return []
    try:
        tree = parser.parse(code)
    except Exception:
        return []
    results: List[Tuple[int, str]] = []
    comment_types = {"comment", "line_comment", "block_comment"}

    def walk(node):
        if node.type in comment_types:
            if not _is_doc_comment(node, code, lang):
                raw = code[node.start_byte:node.end_byte].decode(
                    "utf8", errors="replace")
                stripped = _strip_comment_text(raw, lang)
                if stripped:
                    line = node.start_point[0]
                    results.append((line, stripped))
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return results


def _has_rationale_marker(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in RATIONALE_PHRASES)


def _is_restatement(comment_text: str, next_code_line: Optional[str]) -> bool:
    """Short comment whose identifier-shape tokens all appear in next line."""
    if not next_code_line:
        return False
    words = comment_text.split()
    if not (1 <= len(words) <= 6):
        return False
    if _has_rationale_marker(comment_text):
        # If the author included a rationale phrase, we don't penalise as
        # restatement even if the comment is short.
        return False
    idents = set(t.lower() for t in _IDENT_RE.findall(comment_text))
    if not idents:
        return False
    next_lower = next_code_line.lower()
    overlap = sum(1 for w in idents if w in next_lower)
    return overlap == len(idents)


def applies(diff_text: str) -> bool:
    by_path = parse_diff_added_by_file(diff_text)
    return any(
        any(p.lower().endswith(e) for e in EXT_TO_LANG)
        for p in by_path
    )


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    n_comments = 0
    n_rationale = 0
    n_restate = 0
    any_parsed = False

    for path, content in by_path.items():
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = EXT_TO_LANG.get(ext)
        if lang is None:
            continue
        code_bytes = content.encode("utf8", errors="replace")
        comments = _collect_inline_comments(code_bytes, lang)
        any_parsed = True
        if not comments:
            continue

        lines_full = content.splitlines()
        for line_no, ctext in comments:
            n_comments += 1
            if _has_rationale_marker(ctext):
                n_rationale += 1
            # Find next non-blank, non-comment code line.
            next_line = None
            for j in range(line_no + 1, min(line_no + 4, len(lines_full))):
                cand = lines_full[j].strip()
                if cand and not cand.startswith(("#", "//", "/*", "*")):
                    next_line = cand
                    break
            if _is_restatement(ctext, next_line):
                n_restate += 1

    if not any_parsed:
        return None
    if n_comments == 0:
        # Norm is about comment CONTENT; without comments there is nothing
        # to grade. Abstain rather than emit neutral noise.
        return None

    rationale_rate = n_rationale / n_comments
    restate_rate = n_restate / n_comments
    why_score = rationale_rate - restate_rate  # in [-1, 1]
    score_val = 0.5 * (why_score + 1.0)
    return float(max(0.0, min(1.0, score_val)))
