"""MATH-specific computation ops for hybrid metric channels — seam pilot v1.

All ops here are COMPUTATION ops (Z = f(X): deterministic functions of the single
document, no corpus/external state).  MathOps subclasses Ops, so hybrid programs
running on the math task get the generic ops (normalize/extract_dates/sent_stats,
plus the retrieve_similar evidence op) AND the math ops below.

Why these ops: the math seam survey (outputs/metric_seam_pilot/tasks/math/
seam_table.json) showed description-compiled regex is nearly useless on math
(best pre-gate rho .41; notation consistency -.01, microtypography -.03,
inline-$-delimiters .17).  The checkable predicates need PARSING — LaTeX span
structure and math-token streams — not raw regex over mixed prose+markup.
Corpus profile (Math StackExchange answers, 5k docs): 97% use $...$, 62% use
$$...$$, \\( / \\[ almost absent, 20% use \\begin{env}; hard breakage (odd $,
brace imbalance) is rare (<1%) but soft hazards are common (naked LaTeX
commands outside math 9%, adjacent inline formulas with no separating words 66%).

Pure stdlib + re.  No sympy (not installed in the eval environment); the light
structure we need is a single-pass delimiter scanner, not a CAS.
Every public op is deterministic, runs well under 50ms on corpus-sized docs
(median 1.3KB, p90 2.9KB), and never raises (guarded by _safe).
"""
import re
from collections import Counter

from ops import Ops

# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------

# display-math environments that appear at top level in MathJax-style markup
_MATH_ENVS = {
    "equation", "align", "alignat", "eqnarray", "gather", "multline",
    "displaymath", "math", "split", "cases", "array", "matrix", "pmatrix",
    "bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "aligned", "gathered",
}
_BEGIN_PAT = re.compile(r"\\begin\{([A-Za-z]+\*?)\}")

# spacing / sizing / style commands dropped by the token normalizer
_DROP_CMDS = {
    "\\,", "\\;", "\\:", "\\!", "\\ ", "\\quad", "\\qquad", "\\thinspace",
    "\\negthinspace", "\\enspace", "\\left", "\\right", "\\big", "\\Big",
    "\\bigg", "\\Bigg", "\\bigl", "\\bigr", "\\Bigl", "\\Bigr", "\\biggl",
    "\\biggr", "\\Biggl", "\\Biggr", "\\bigm", "\\displaystyle", "\\textstyle",
    "\\scriptstyle", "\\limits", "\\nolimits", "\\phantom", "\\hphantom",
    "\\vphantom", "\\strut", "\\mathstrut", "\\allowbreak", "\\nonumber",
    "\\notag",
}
# synonym commands canonicalized by the token normalizer (structure-level
# equivalence; notation_census deliberately keeps raw surface forms so that
# \le-vs-\leq style inconsistency stays measurable)
_CANON_CMDS = {
    "\\le": "\\leq", "\\ge": "\\geq", "\\ne": "\\neq", "\\to": "\\rightarrow",
    "\\gets": "\\leftarrow", "\\Bbb": "\\mathbb", "\\bf": "\\mathbf",
    "\\rm": "\\mathrm", "\\cal": "\\mathcal", "\\frak": "\\mathfrak",
    "\\dfrac": "\\frac", "\\tfrac": "\\frac", "\\dots": "\\ldots",
    "\\cdots": "\\ldots", "\\dotsb": "\\ldots", "\\dotsc": "\\ldots",
    "\\over": "\\frac",
}
_TOKEN_PAT = re.compile(r"\\[A-Za-z]+\*?|\\.|\d+(?:\.\d+)?|[A-Za-z]+|\S")

# multi-letter runs kept as function-name tokens by notation_census (bare
# 'sin' vs '\sin' is a real notational inconsistency worth counting)
_BARE_FUNCS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim",
    "max", "min", "sup", "inf", "det", "gcd", "lcm", "deg", "dim", "ker",
    "mod", "arg", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
}

# unicode chars that signal math notation (in spans OR leaking into prose)
def _is_unicode_math(ch):
    o = ord(ch)
    return (
        0x0370 <= o <= 0x03FF      # Greek
        or 0x2070 <= o <= 0x209F   # super/subscripts
        or 0x2190 <= o <= 0x21FF   # arrows
        or 0x2200 <= o <= 0x22FF   # mathematical operators
        or 0x27C0 <= o <= 0x27EF or 0x2980 <= o <= 0x29FF
        or 0x2A00 <= o <= 0x2AFF   # misc/supplemental math symbols
        or ch in "×÷±∞°′″∎■◻ℝℕℤℚℂℓ√"
    )

_INLINE_WINDOW = 600  # a $ with no closer within this window is literal text

def _scan_math(text):
    """Single pass over the doc. Returns (spans, diag) where spans is a list of
    (kind, content, start, end) in document order — kind in {'inline','display'},
    content excludes delimiters, start/end are offsets of the full span — and
    diag counts scanner-level breakage: unclosed inline/display openers."""
    spans, i, n = [], 0, len(text)
    unclosed_inline = unclosed_display = 0
    while i < n:
        c = text[i]
        if c == "\\":
            nxt = text[i + 1:i + 2]
            if nxt == "$":                       # escaped dollar: literal
                i += 2
                continue
            if nxt == "(":
                j = text.find("\\)", i + 2)
                if j != -1:
                    spans.append(("inline", text[i + 2:j], i, j + 2)); i = j + 2
                else:
                    unclosed_inline += 1; i += 2
                continue
            if nxt == "[":
                j = text.find("\\]", i + 2)
                if j != -1:
                    spans.append(("display", text[i + 2:j], i, j + 2)); i = j + 2
                else:
                    unclosed_display += 1; i += 2
                continue
            m = _BEGIN_PAT.match(text, i)
            if m and m.group(1).rstrip("*") in _MATH_ENVS:
                env = m.group(1)
                end_tag = "\\end{%s}" % env
                j = text.find(end_tag, m.end())
                if j != -1:
                    spans.append(("display", text[m.end():j], i, j + len(end_tag)))
                    i = j + len(end_tag)
                else:
                    unclosed_display += 1; i = m.end()
                continue
            i += 2 if nxt else 1                 # any other command: skip "\x"
            continue
        if c == "$":
            if text[i + 1:i + 2] == "$":         # display $$...$$
                j = text.find("$$", i + 2)
                if j != -1:
                    spans.append(("display", text[i + 2:j], i, j + 2)); i = j + 2
                else:
                    unclosed_display += 1; i += 2
                continue
            # inline $...$: closer must be unescaped, within the window, and
            # not across a paragraph break (else the $ is literal, e.g. money)
            j, limit = i + 1, min(n, i + 1 + _INLINE_WINDOW)
            close = -1
            while j < limit:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "$":
                    close = j
                    break
                if text[j] == "\n" and text[j + 1:j + 2] == "\n":
                    break
                j += 1
            if close != -1 and close > i + 1:
                spans.append(("inline", text[i + 1:close], i, close + 1))
                i = close + 1
            else:
                unclosed_inline += 1; i += 1
            continue
        i += 1
    return spans, {"unclosed_inline": unclosed_inline,
                   "unclosed_display": unclosed_display}


def _safe(default):
    """Never-raise guard: on any exception return default() (or a copy-safe
    scalar). Ops feed unvetted web text; a hybrid must not crash the harness."""
    def deco(fn):
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                return default() if callable(default) else default
        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__
        return wrapped
    return deco


# --------------------------------------------------------------------------
# marker patterns for proof_skeleton
# --------------------------------------------------------------------------
_SKELETON_PATS = {
    "theorem": re.compile(
        r"\\begin\{(?:theorem|lemma|proposition|corollary|claim)\*?\}"
        r"|\b(?:Theorem|Lemma|Proposition|Corollary|Claim)\b"),
    "definition": re.compile(
        r"\\begin\{definition\*?\}"
        r"|\b(?:definitions?|defined?s?|denote[sd]?|is called|we say|notation)\b",
        re.I),
    "proof": re.compile(r"\\begin\{proof\}|\b[Pp]roofs?\b"),
    "qed": re.compile(
        r"\bQED\b|q\.e\.d\.|\\blacksquare|\\square|\\qed|[∎■◻□]"
        r"|as desired|complet(?:es|ing) the proof|which was to be shown", re.I),
    "case": re.compile(
        r"\b[Cc]ases?\s+(?:\d+|[ivxIVX]+\b)"
        r"|\bin (?:either|each|both|this|that|the first|the second) case\b"),
    "connective": re.compile(
        r"\b(?:suppose|assume|assuming|hence|therefore|thus|it follows"
        r"|contradiction|contrapositive|w\.l\.o\.g|wlog"
        r"|without loss of generality|by induction|inductive step|base case"
        r"|conversely|note that|observe that|it suffices)\b", re.I),
}

_NAKED_CMD_PAT = re.compile(r"\\[A-Za-z]{2,}")
_TAGGED_PAT = re.compile(r"\\tag\b|\\label\{")


class MathOps(Ops):
    """Ops + math computation ops (LaTeX-aware parsing over StackExchange-style
    markup). All new ops are COMPUTATION ops: Z = f(X), no external state."""

    # ------------------------------------------------------------------
    @staticmethod
    @_safe(list)
    def extract_math_spans(text):
        """All math spans in document order as (kind, content) with kind in
        {'inline','display'}; handles $..$/$$..$$/\\(..\\)/\\[..\\]/\\begin{env},
        escaped \\$, and broken openers (literal-$ fallback). [computation op]"""
        spans, _ = _scan_math(text)
        return [(k, s) for k, s, _a, _b in spans]

    # ------------------------------------------------------------------
    @staticmethod
    @_safe(list)
    def latex_tokens(span):
        """Normalized LaTeX token stream for one math span: commands (synonyms
        canonicalized, spacing/sizing dropped), letter runs, numbers, symbol
        chars. [computation op]"""
        out = []
        for tok in _TOKEN_PAT.findall(span):
            if tok.startswith("\\"):
                tok = _CANON_CMDS.get(tok, tok)
                if tok in _DROP_CMDS:
                    continue
            out.append(tok)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    @_safe(dict)
    def notation_census(text):
        """Symbol->count census for consistency checks: raw LaTeX commands and
        single-letter identifiers inside math spans, bare function names
        (sin vs \\sin), and unicode math chars anywhere (incl. leaking into
        prose). Surface forms are NOT canonicalized. [computation op]"""
        census = Counter()
        spans, _ = _scan_math(text)
        for _kind, content, _a, _b in spans:
            for tok in _TOKEN_PAT.findall(content):
                if tok.startswith("\\") and len(tok) > 1:
                    census[tok] += 1
                elif tok.isalpha():
                    if len(tok) == 1:
                        census[tok] += 1
                    elif tok.lower() in _BARE_FUNCS:
                        census[tok.lower()] += 1
        for ch in text:
            if _is_unicode_math(ch):
                census[ch] += 1
        return dict(census)

    # ------------------------------------------------------------------
    @staticmethod
    @_safe((0, 0, 0, 0.0))
    def equation_stats(text):
        """(n_display, n_inline, n_numbered, avg_tokens): span counts, display
        spans carrying \\tag/\\label numbering, and mean normalized-token
        length per span (0.0 if no math). [computation op]"""
        spans, _ = _scan_math(text)
        n_display = sum(1 for k, *_ in spans if k == "display")
        n_inline = len(spans) - n_display
        n_numbered = sum(1 for k, s, _a, _b in spans
                         if k == "display" and _TAGGED_PAT.search(s))
        if spans:
            avg = sum(len(MathOps.latex_tokens(s)) for _k, s, _a, _b in spans) / len(spans)
        else:
            avg = 0.0
        return n_display, n_inline, n_numbered, avg

    # ------------------------------------------------------------------
    @staticmethod
    @_safe(lambda: {k: [] for k in _SKELETON_PATS})
    def proof_skeleton(text):
        """Positions of proof-structure markers, {category: [(offset, match)]}
        for categories theorem / definition / proof / qed / case / connective
        (definition markers folded in as a category). [computation op]"""
        out = {}
        for cat, pat in _SKELETON_PATS.items():
            out[cat] = [(m.start(), m.group(0)) for m in pat.finditer(text)]
        return out

    # ------------------------------------------------------------------
    @staticmethod
    @_safe(dict)
    def delimiter_health(text):
        """Unbalanced-delimiter diagnostics (all int counts): odd $ parity,
        unclosed inline/display openers, brace imbalance, \\left/\\right
        mismatch, \\begin/\\end env mismatch, naked LaTeX commands outside
        math, adjacent inline formulas with no separating words.
        [computation op]"""
        spans, diag = _scan_math(text)
        stripped = text.replace("\\$", "")
        health = {
            "odd_dollar": stripped.count("$") % 2,
            "unclosed_inline": diag["unclosed_inline"],
            "unclosed_display": diag["unclosed_display"],
        }
        nobrace = re.sub(r"\\[{}]", "", text)
        health["brace_imbalance"] = abs(nobrace.count("{") - nobrace.count("}"))
        health["left_right_mismatch"] = abs(
            len(re.findall(r"\\left\b", text)) - len(re.findall(r"\\right\b", text)))
        begins = Counter(m.group(1) for m in _BEGIN_PAT.finditer(text))
        ends = Counter(re.findall(r"\\end\{([A-Za-z]+\*?)\}", text))
        health["env_mismatch"] = sum(((begins - ends) + (ends - begins)).values())
        # mask math spans, count LaTeX commands left in prose (markup failure)
        masked, prev = [], 0
        for _k, _s, a, b in spans:
            masked.append(text[prev:a]); prev = b
        masked.append(text[prev:])
        prose = " ".join(masked)
        health["naked_commands"] = sum(
            1 for m in _NAKED_CMD_PAT.finditer(prose)
            if m.group(0) not in ("\\begin", "\\end"))
        # adjacent inline formulas separated only by whitespace (style hazard)
        adjacent = 0
        for (k1, _s1, _a1, b1), (k2, _s2, a2, _b2) in zip(spans, spans[1:]):
            if k1 == "inline" and k2 == "inline" and text[b1:a2].strip() == "":
                adjacent += 1
        health["adjacent_inline"] = adjacent
        return health
