"""
code_blocks.py — robust code-block detection for CR.SE answer bodies.

The pilot's bug: the balanced pool's `text` field is strip_html()'d (tags gone
AND whitespace collapsed), so ``` fences never existed and <pre><code> blocks
lost their line structure. This module operates on the RAW body and sniffs the
format:

  * HTML (the SE data dump's rendered Body — CR.SE's actual source):
      <pre><code>...</code></pre>  (also bare <pre> and <pre class="...">)
  * Markdown (the SO mirror used elsewhere): ``` fenced blocks
  * Fallback: 4-space indented blocks (classic markdown code)
  * Last resort: Python-grammar sniffer for whitespace-collapsed text
    (the pool's stripped `text` field) — only used when no raw body exists.

Also provides a cheap per-block language guesser so the checker can pick the
block matching the claim's language.
"""
from __future__ import annotations
import html as _html
import re

# -- HTML --------------------------------------------------------------------

PRE_CODE_RE = re.compile(
    r"<pre[^>]*>\s*(?:<code[^>]*>)?(.*?)(?:</code>)?\s*</pre>",
    re.DOTALL | re.IGNORECASE,
)
INLINE_HTML_CODE_RE = re.compile(r"<code[^>]*>(.*?)</code>", re.DOTALL | re.IGNORECASE)

# -- Markdown ----------------------------------------------------------------

FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
INLINE_TICK_RE = re.compile(r"`([^`\n]{2,200})`")


def _unescape(s: str) -> str:
    return _html.unescape(s)


def _indented_blocks(text: str) -> list[str]:
    """Classic markdown: contiguous runs of 4-space (or tab) indented lines."""
    blocks, cur = [], []
    for L in text.split("\n"):
        if L.startswith("    ") or L.startswith("\t"):
            cur.append(L[4:] if L.startswith("    ") else L[1:])
        elif cur and not L.strip():
            cur.append("")
        elif cur:
            b = "\n".join(cur).strip()
            if b:
                blocks.append(b)
            cur = []
    if cur:
        b = "\n".join(cur).strip()
        if b:
            blocks.append(b)
    return blocks


def detect_format(body: str) -> str:
    """'html' | 'markdown' | 'plain'. HTML wins if any block-level tag is
    present (the SE dump renders EVERYTHING, even prose, into <p> tags)."""
    if not body:
        return "plain"
    if re.search(r"<(?:pre|p|code|blockquote|ul|ol)\b[^>]*>", body, re.IGNORECASE):
        return "html"
    if "```" in body or _indented_blocks(body):
        return "markdown"
    return "plain"


def extract_code_blocks(body: str, min_chars: int = 8) -> list[str]:
    """Return all code blocks (unescaped, structure preserved) from a raw
    answer/question body, handling HTML and markdown sources."""
    if not body:
        return []
    fmt = detect_format(body)
    blocks: list[str] = []
    if fmt == "html":
        blocks = [_unescape(m.group(1)).strip() for m in PRE_CODE_RE.finditer(body)]
    if not blocks and "```" in body:
        blocks = [m.group(1).strip() for m in FENCE_RE.finditer(body)]
    if not blocks and fmt != "html":
        blocks = _indented_blocks(body)
    return [b for b in blocks if len(b) >= min_chars]


def extract_inline_snippets(body: str) -> list[str]:
    """Short inline code mentions (<code>x</code> or `x`)."""
    if not body:
        return []
    out = [_unescape(m.group(1)).strip() for m in INLINE_HTML_CODE_RE.finditer(body)]
    out += [m.group(1).strip() for m in INLINE_TICK_RE.finditer(body)]
    # de-dup preserving order; drop blocks (multi-line) — those come from
    # extract_code_blocks
    seen, res = set(), []
    for s in out:
        if s and "\n" not in s and s not in seen:
            seen.add(s)
            res.append(s)
    return res


# -- Python-grammar sniffer (last resort for whitespace-collapsed text) -------

PY_SNIFF_RE = re.compile(
    r"\b(?:def\s+\w+\s*\(|class\s+\w+\s*[:(]|import\s+\w+|from\s+\w+\s+import\b|"
    r"for\s+\w+\s+in\s+.+?:|while\s+.+?:|return\s+|lambda\s+\w+:)",
)


def sniff_python_in_collapsed_text(text: str) -> bool:
    return bool(PY_SNIFF_RE.search(text or ""))


# -- language guesser ----------------------------------------------------------

_LANG_CUES = [
    ("python", re.compile(r"^\s*(?:def |class \w+[:(]|import \w|from \w+ import|"
                          r"if __name__|print\()", re.M)),
    ("python", re.compile(r":\s*$\n\s+\S", re.M)),  # colon-newline-indent
    ("java", re.compile(r"\bSystem\.out\.print|public\s+static\s+void\s+main|"
                        r"import\s+java\.|@Override\b")),
    ("c#", re.compile(r"\busing\s+System\b|Console\.Write|namespace\s+\w+|"
                      r"\bpublic\s+(?:static\s+)?(?:string|bool|var)\s+\w+\s*\(")),
    ("java", re.compile(r"\bpublic\s+class\s+\w+|"
                        r"\bpublic\s+(?:static\s+)?(?:void|int|String)\s+\w+\s*\(")),
    ("javascript", re.compile(r"\bfunction\s+\w*\s*\(|=>\s*[{(]|\bconsole\.log|"
                              r"\b(?:var|let|const)\s+\w+\s*=|document\.|require\(")),
    ("c++", re.compile(r"#include\s*<|std::|\bcout\b|\bcin\b|int\s+main\s*\(")),
    ("php", re.compile(r"<\?php|\$\w+\s*=|->\w+\(")),
    ("ruby", re.compile(r"^\s*(?:def \w+$|end$|puts |require ['\"])", re.M)),
    ("sql", re.compile(r"\bSELECT\b.+?\bFROM\b|\bINSERT\s+INTO\b|\bCREATE\s+TABLE\b",
                       re.I | re.S)),
    ("go", re.compile(r"\bfunc\s+\w+\s*\(|\bpackage\s+main\b|fmt\.Print")),
    ("rust", re.compile(r"\bfn\s+\w+\s*\(|let\s+mut\b|println!\(")),
]


def guess_language(code: str) -> str:
    for lang, pat in _LANG_CUES:
        if pat.search(code or ""):
            return lang
    return "unknown"


def blocks_for_language(blocks: list[str], lang: str | None) -> list[str]:
    """Blocks compatible with the claim's language: exact guesses first, then
    unknown-guess blocks, never blocks confidently guessed as another lang."""
    if not blocks:
        return []
    if not lang or lang in ("unknown", "other"):
        return blocks
    exact, neutral = [], []
    for b in blocks:
        g = guess_language(b)
        if g == lang:
            exact.append(b)
        elif g == "unknown":
            neutral.append(b)
    return exact + neutral
