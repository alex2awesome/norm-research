"""
a228: "Separate adjacent inline formulas with words"

Do not place two inline formulas adjacent with only punctuation (or a bare
one-letter connective) between them; insert real intervening words.

Two mechanisms both count as the same underlying "run-on formula" failure
(rate_a / rate_b below):

  rate_a -- internal cram: several fully atomic values/expressions
      (\\frac{}{}s, bare numbers, single letters -- NOT relational clauses
      like "a,b\\in\\mathbb{Z}") packed as a comma list inside a SINGLE
      inline span. Same visual run-on effect, just LaTeX-encoded
      differently; a naive "$...$ $...$ " regex baseline can't see this
      because it never looks inside a span. Requires len(parts)>=4 and
      almost all atomic, so ordinary "a, b, c in R" style declarations
      (which are standard, unproblematic notation) are NOT flagged.

  rate_b -- cross-span adjacency: two separate $...$ / \\(...\\) inline
      spans with nothing but whitespace/punctuation/a tiny connector
      between them. ops.delimiter_health already tracks this, so we read
      its count directly (robust to escaped \\$, display-math exclusion,
      etc.); we also run a local scan that additionally gives partial
      credit to bare one-word connectives ("$X$ or $Y$"), and take the
      more sensitive of the two.

A single LLM field ("formula_style") supplies a holistic thick-input nudge
for cases neither parser mechanism reliably reaches: e.g. an enumeration
like "$x$ axis, $y$ axis, $z$ axis" is grammatically fine (real words in
every gap) yet still reads as a listy run-on to a holistic reader. This is
a small, bounded nudge -- the predicate itself lives in code.
"""

import re

LLM_FIELDS = {
    "formula_style": (
        "In <=8 words: are formulas woven into full explanatory sentences, "
        "or mostly chained/listed with little narration between them? "
        "Else NONE."
    ),
}

_RELATION_RE = re.compile(
    r"\\in\b|\\notin\b|\\neq\b|\\ne\b|\\le\b|\\ge\b|\\leq\b|\\geq\b|"
    r"\\sim\b|\\equiv\b|\\subset\b|\\subseteq\b|[=<>]"
)
_ATOMIC_RE = re.compile(
    r"^\s*-?\\?[a-zA-Z]*frac\{[^{}]*\}\{[^{}]*\}\s*$"
    r"|^\s*-?\d+(\.\d+)?\s*$"
    r"|^\s*[a-zA-Z]\s*$"
)

_WORD_RE = re.compile(r"[A-Za-z]{3,}")
# Only mask genuine block-level display environments -- NOT things like
# pmatrix/cases/array, which are routinely nested *inside* an inline $...$
# (e.g. "$X=\begin{pmatrix}...\end{pmatrix}$" is one inline span, not a
# display block), so a blanket \begin{...}...\end{...} match would wrongly
# shred ordinary inline spans that happen to contain a matrix.
_DISPLAY_ENV_RE = re.compile(
    r"\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|eqnarray\*?|alignat\*?)\}"
    r".*?\\end\{\1\}",
    re.DOTALL,
)
_DISPLAY_RE = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
_ESC_DOLLAR_RE = re.compile(r"\\\$")
_INLINE_RE = re.compile(r"\$([^$]+)\$|\\\((.*?)\\\)", re.DOTALL)

_STYLE_BAD = ("list", "chain", "terse", "enumerat", "symbol", "dump", "bare", "minimal", "run-on", "run on")
_STYLE_GOOD = ("prose", "flow", "explan", "narrat", "full sentence")


def _split_top_level(content, seps=(",", ";")):
    parts, depth, cur = [], 0, []
    for ch in content:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth = max(0, depth - 1)
        if depth == 0 and ch in seps:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _is_atomic(p):
    p = p.strip()
    if not p or _RELATION_RE.search(p):
        return False
    return bool(_ATOMIC_RE.match(p))


def _internal_cram(inline_contents):
    bad = total = 0.0
    for content in inline_contents:
        parts = [p for p in _split_top_level(content) if p.strip()]
        if len(parts) >= 4:
            atomic = sum(1 for p in parts if _is_atomic(p))
            if atomic >= max(3, len(parts) - 1):
                joins = len(parts) - 1
                bad += joins
                total += joins
    return bad, total


def _fallback_adjacent_count(text, n_pairs_hint):
    """Local backup only used if ops.delimiter_health has no matching key."""
    t = _ESC_DOLLAR_RE.sub("\x00", text)
    t = _DISPLAY_RE.sub(" XDISPLAYMATHX ", t)
    t = _DISPLAY_ENV_RE.sub(" XDISPLAYMATHX ", t)
    matches = list(_INLINE_RE.finditer(t))
    bad = 0.0
    for i in range(len(matches) - 1):
        gap = t[matches[i].end():matches[i + 1].start()]
        if len(gap) > 80:
            continue
        words = len(_WORD_RE.findall(gap))
        if words == 0:
            bad += 1.0
        elif words == 1 and len(gap.strip()) <= 6:
            bad += 0.5
    return bad


def score(text: str, extracted: dict, ops) -> float:
    try:
        spans = ops.extract_math_spans(text)
        inline_contents = [c for (k, c) in spans if k == "inline"]
        n_inline = len(inline_contents)

        bad_a, total_a = _internal_cram(inline_contents)
        rate_a = (bad_a / total_a) if total_a > 0 else 0.0

        total_b = max(0, n_inline - 1)
        bad_b = 0.0
        if total_b > 0:
            # Take the more sensitive of (a) the op's own robust count of
            # strict zero-word adjacencies and (b) our local scan, which
            # additionally gives partial credit to bare one-word connectives
            # ("$X$ or $Y$") that read as thin glue rather than real
            # explanation. Using max() means a coarser op implementation
            # never makes us *less* sensitive than the local scan alone.
            fallback = _fallback_adjacent_count(text, total_b)
            ops_count = 0.0
            try:
                health = ops.delimiter_health(text) or {}
                adj_key = next((k for k in health if "adjacen" in str(k).lower()), None)
                if adj_key is not None:
                    ops_count = float(health.get(adj_key, 0) or 0)
            except Exception:
                ops_count = 0.0
            bad_b = max(fallback, ops_count)
        rate_b = (bad_b / total_b) if total_b > 0 else 0.0

        # Judge behavior on the training set is bimodal (near-1.0 or near-0),
        # consistent with "either failure mode alone, in proportion to its
        # own severity, is disqualifying" rather than a diluted document-wide
        # average -- so combine the two mechanisms with a soft-OR instead of
        # a weighted mean.
        rate = 1.0 - (1.0 - rate_a) * (1.0 - rate_b)

        style = ((extracted or {}).get("formula_style") or "").lower()
        nudge = 0.0
        if any(k in style for k in _STYLE_BAD):
            nudge = 0.15
        elif any(k in style for k in _STYLE_GOOD):
            nudge = -0.05

        raw = 1.0 - rate - nudge
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
