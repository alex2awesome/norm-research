"""a50: Commenting strategy and quality — INLINE comments.

What we measure
---------------
The aspect description says: "Favor self-explanatory code; comment the
what/why and non-obvious assumptions; keep comments minimal but sufficient,
accurate, and maintained."

We focus on INLINE comments (// ..., /* ... */, # ...) — NOT docstrings.
Docstring formatting is handled separately by a202. The two metrics are
complementary:
  - a202 grades docstring formatting (D-rule violations on Python docstrings).
  - a50 grades the use of INLINE comments inside code bodies, cross-language.

We use tree-sitter per language (Python, JavaScript, TypeScript, Java, Go)
to identify comment nodes, then exclude docstrings (Python) and JSDoc /
Javadoc-style block comments that are documentation, not inline commentary.

Three surrogate signals (all that static analysis can reach):
  1. Comment density in a healthy range.
     0% density gets ~0.7 (acceptable — "self-explanatory code"), peaks
     around 8-15% (comment the non-obvious), penalised above 35% (over-
     commenting suggests unclear code).
  2. TODO/FIXME/XXX/HACK density.
     Each such marker subtracts; 1 marker per 50 added code lines roughly
     halves this component. Many TODOs = unfinished work.
  3. Useless-comment rate.
     Heuristic for comments that restate the next code line (very short
     identifier-only comment immediately preceding a short line that
     contains the same identifier). Penalises "// increment i" style.

We combine the three:
    score = 0.55*density_norm + 0.25*todo_norm + 0.20*(1 - useless_rate)

Classification
--------------
PARTIALLY_THIN. Static analysis cannot judge whether a comment is ACCURATE
(matches the code's behavior) or MAINTAINED (still true after edits). Those
checks require dynamic correspondence or human reading. We measure
surrogate signals: density-shape, TODO load, and a useless-restatement
heuristic.

Distinct from a202
------------------
a202 reads docstrings only and uses ruff D-rules to grade formatting
(whitespace, capitalisation, section ordering). a50 reads INLINE comments,
explicitly excludes docstrings (Python module/class/function first string,
Javadoc /** ... */, JSDoc /** ... */) and grades commenting STRATEGY,
not formatting. They share no signal.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a50"
ASPECT_NAME = "Commenting strategy and quality"
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

# REGEX_OK: tool_output — pattern over comment text after tree-sitter
# extraction. We are matching well-known issue markers inside a known
# string, not parsing code.
TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK|TBD|REFACTOR)\b", re.IGNORECASE)

# REGEX_OK: tool_output — strip language-specific comment delimiters from a
# comment node's text so we can analyse the inner prose.
# REGEX_OK: tool_output — strip "#" prefix from a tree-sitter comment node.
_PY_COMMENT_PREFIX = re.compile(r"^\s*#+\s?")
# REGEX_OK: tool_output — strip "//" prefix from a tree-sitter comment node.
_C_LINE_COMMENT_PREFIX = re.compile(r"^\s*//+\s?")
# REGEX_OK: tool_output — strip /* */ fence around a tree-sitter block comment.
_C_BLOCK_COMMENT_FENCE = re.compile(r"^\s*/\*+\s?|\s?\*+/\s*$", re.DOTALL)

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
    """Remove comment fences/prefixes so we get the inner prose."""
    if lang == "py":
        return _PY_COMMENT_PREFIX.sub("", raw).strip()
    # c-family
    s = raw.strip()
    if s.startswith("//"):
        return _C_LINE_COMMENT_PREFIX.sub("", raw).strip()
    if s.startswith("/*"):
        # collapse newlines + leading "*" continuation chars
        inner = s
        # REGEX_OK: tool_output — strip /* ... */ fences and per-line "*"
        # continuation markers from a tree-sitter block_comment node.
        inner = re.sub(r"^/\*+", "", inner)
        inner = re.sub(r"\*+/$", "", inner)
        # REGEX_OK: tool_output — strip per-line "*" continuation marker
        # inside a block comment body (Javadoc/JSDoc convention).
        lines = [re.sub(r"^\s*\*+\s?", "", ln)
                 for ln in inner.splitlines()]
        return " ".join(s2.strip() for s2 in lines).strip()
    return s.strip()


def _is_doc_comment(node, source: bytes, lang: str) -> bool:
    """True if this comment is really a documentation comment, not inline.

    Python: any string node that is the first statement of a module / def /
    class is a docstring — but those are `string` nodes, not `comment`
    nodes, so Python comments are always inline by definition. We only
    need to handle the C-family JSDoc / Javadoc case.

    For JS/TS/Java/Go: a block comment starting with `/**` is a doc
    comment (JSDoc / Javadoc convention).
    """
    if lang == "py":
        return False
    raw = source[node.start_byte:node.end_byte]
    return raw.startswith(b"/**")


def _collect_inline_comments(code: bytes, lang: str) -> List[Tuple[int, str]]:
    """Return [(line_no, stripped_text), ...] for inline comments only."""
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


def _code_lines(code: bytes, lang: str) -> List[str]:
    """Return the non-blank lines of code with the comment portions removed.

    We don't need a precise per-line code/comment split here — we only
    use this to (a) count non-blank lines, (b) check whether the line
    immediately following a comment contains a token matching the comment.
    A best-effort strip of trailing `//`/`#` portions is sufficient.
    """
    text = code.decode("utf8", errors="replace")
    out = []
    for ln in text.splitlines():
        # Strip trailing line comments. We don't worry about string
        # literals containing `//` — this is a heuristic, not a parser.
        if lang == "py":
            # REGEX_OK: tool_output — strip trailing python comment from a line
            ln2 = re.sub(r"\s+#.*$", "", ln)
        else:
            # REGEX_OK: tool_output — strip trailing C-style line comment
            ln2 = re.sub(r"\s+//.*$", "", ln)
        if ln2.strip():
            out.append(ln2)
    return out


# Tokens we treat as code-meaningful when checking restatement.
# REGEX_OK: tool_output — identifier-shape word extraction over comment
# prose (already extracted by tree-sitter), not over source code.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _useless_comment(comment_text: str, next_code_line: Optional[str]) -> bool:
    """Heuristic: short comment whose words all appear in the next code line.

    "// increment i" preceding "i++" or "i = i + 1" qualifies as useless.
    We require the comment to be SHORT (<=6 words) and at least one
    identifier-like token (len>=3) from the comment to appear verbatim
    in the next code line.
    """
    if not next_code_line:
        return False
    words = comment_text.split()
    if not (1 <= len(words) <= 6):
        return False
    if TODO_RE.search(comment_text):
        return False  # TODOs are not "useless restatement"
    idents = set(_IDENT_RE.findall(comment_text.lower()))
    if not idents:
        return False
    next_lower = next_code_line.lower()
    overlap = sum(1 for w in idents if w in next_lower)
    # All comment idents present in next line + comment short -> useless
    return overlap >= max(1, len(idents) - 0) and overlap == len(idents)


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

    total_code_lines = 0
    total_comments = 0
    total_todos = 0
    total_useless = 0
    any_parsed = False

    for path, content in by_path.items():
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        lang = EXT_TO_LANG.get(ext)
        if lang is None:
            continue
        code_bytes = content.encode("utf8", errors="replace")
        comments = _collect_inline_comments(code_bytes, lang)
        code_lines = _code_lines(code_bytes, lang)
        if not code_lines and not comments:
            continue
        any_parsed = True
        total_code_lines += len(code_lines)
        total_comments += len(comments)

        # Build a line_no -> code text map for next-line lookups.
        text = content
        lines_full = text.splitlines()

        for line_no, ctext in comments:
            if TODO_RE.search(ctext):
                total_todos += 1
            # Look at the next non-blank line after this comment.
            next_line = None
            for j in range(line_no + 1, min(line_no + 4, len(lines_full))):
                cand = lines_full[j].strip()
                if cand and not cand.startswith(("#", "//", "/*", "*")):
                    next_line = cand
                    break
            if _useless_comment(ctext, next_line):
                total_useless += 1

    if not any_parsed:
        return None
    if total_code_lines == 0 and total_comments == 0:
        return None

    # --- density component ---
    # Density = comments / (comments + code_lines). Healthy zone ~5-15%.
    denom = total_comments + max(total_code_lines, 1)
    density = total_comments / denom

    def density_score(d: float) -> float:
        # Piecewise: 0 -> 0.7, peak 1.0 at ~10%, decays past 35%.
        if d <= 0.10:
            # 0 -> 0.7, 0.10 -> 1.0
            return 0.7 + 3.0 * d
        if d <= 0.35:
            return 1.0 - (d - 0.10) * 0.4 / 0.25  # 0.10->1.0, 0.35->0.6
        # over-commented: exponential decay
        return max(0.0, 0.6 * math.exp(-(d - 0.35) * 4.0))

    density_norm = density_score(density)

    # --- TODO/FIXME density ---
    # 0 TODOs -> 1.0, ratio of 1 TODO per 50 lines -> 0.5
    todo_rate = total_todos / max(total_code_lines, 1)
    todo_norm = math.exp(-todo_rate * 35.0)

    # --- useless-restatement rate ---
    # fraction of comments that look like restatement
    if total_comments > 0:
        useless_rate = total_useless / total_comments
    else:
        useless_rate = 0.0
    useless_norm = 1.0 - useless_rate

    score_val = (0.55 * density_norm
                 + 0.25 * todo_norm
                 + 0.20 * useless_norm)
    return float(max(0.0, min(1.0, score_val)))
