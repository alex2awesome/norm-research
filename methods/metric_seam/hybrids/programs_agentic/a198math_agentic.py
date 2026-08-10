"""
Hybrid metric for a198: "Citations and cross-references: precision, placement,
and attribution" on Math StackExchange answers.  Iterating on the a198_h0
reference (TRAIN rho 0.5887) via the agentic loop.

ROUND 1 changes vs h0, driven by residual diagnosis on h0 itself:

(1) BUG FIX -- internal cross-reference false positives. h0's bare "(n)"
    back-reference regex matched ANY parenthesized integer anywhere in the
    answer, including function-application parens inside ordinary math like
    "\\cos (2)", "q(0)", "\\text{ord}_3(252)" -- these are NOT equation
    back-references, just arguments. Counting repeats of the SAME digit (e.g.
    "(2)" appearing 4x as different cosine arguments in one integral) made
    the false "reused" bonus fire hard (h0 scored several such answers
    0.43-0.46 while the judge gave them 0.0). Fix: a bare "(n)" only counts
    as a REFERENCE if it is the ENTIRE content of its own math span (i.e.
    written standalone as "$(1)$", not embedded inside a larger formula), and
    only counts toward "reused" credit if n also appears in the DEFINED set
    (from \\tag/\\label) -- i.e. genuinely "defined, then referenced again",
    per the criterion's own wording, not merely "a digit recurring".

(2) SCOPE -- whole document, not answer-only. h0 restricted both signals to
    text after "Answer:". Residual inspection showed several judge-credited
    docs (0.15-0.45) have their ONLY citation/cross-ref discipline in the
    QUESTION (asker cites "Section 10.1.3 of ... by Nakahara", "Munkres
    Exercise 70.1", or defines \\tag{1}..\\tag{4} that the answer discusses
    without re-tagging) -- the judge is evidently scoring the citation
    practice of the scraped Q+A document as a whole, not the answerer alone.
    Both signals now scan the full normalized text.

(3) NEW CODE OP -- named external source without a bibliographic trigger
    word. h0's generic-external-ref regex only fires on "the/this/that
    book/paper/...". It missed constructions like "by Bondy and Murty",
    "Munkres Exercise 70.1", "Rudin's ...". Added `_named_source(text)`: a
    proper-name token structurally paired (via 'by NAME', "NAME's", or
    "NAME <Locator> <number>") with either a bibliographic noun or a numbered
    locator -- deliberately EXCLUDING theorem/lemma/prop from the possessive
    trigger list, so "Bayes' theorem" / "Cauchy-Schwarz" (naming a concept,
    not a locatable source) do not trigger it. This is a code-side proxy for
    the same THICK judgment the cited_source LLM field targets, used only as
    a secondary/gating signal (weighted below the LLM field), so it can catch
    QUESTION-side citations the answer-scoped field never sees.

LLM_FIELDS is UNCHANGED from h0 (same single field, same instruction) --
budget frozen; extracted["cited_source"] remains the strongest ext signal
when present.
"""

import re

LLM_FIELDS = {
    "cited_source": (
        "In the ANSWER only (not the question), name the specific external "
        "source it cites and relies on: a book/paper title with author, "
        "another identified person's specific answer/post, or a named "
        "website (e.g. Wikipedia). Reply NONE if it only names a theorem or "
        "concept generically, or vaguely points to 'above'/'a comment' "
        "without a locatable source."
    ),
}

_EXTERNAL_CITE_RE = re.compile(r'\\(cite|citep|citet|autoref)\{')
_SEE_PHRASE_RE = re.compile(r'(?i)\b(see|cf\.|c\.f\.)\b')
_GENERIC_EXTERNAL_REF_RE = re.compile(
    r'(?i)\b(this|that|the)\s+(paper|book|article|thesis|text|source|reference|note)s?\b'
)
# theorem/lemma/.../equation NUMBER, tolerant of LaTeX noise ($, (, ), \) between
# the keyword and the digits -- this is exactly what breaks the naive baseline.
_LOCATOR_RE = re.compile(
    r'(?i)\b(theorem|lemma|proposition|prop|corollary|section|chapter|equation|eq)s?\.?'
    r'[\s\$\(\)\[\]\\]{0,6}(\d+|[ivxlcm]+)\b'
)
_PAGE_RE = re.compile(r'(?i)\bpp?\.[\s\$]{0,3}\d+')

# --- named external source without needing a "book/paper" trigger word ---
_BIB_NOUN = r'(?:book|paper|text|notes?|lecture\s*notes?|thesis|article|monograph|exercises?)'
_LOC_WORD = r'(?:[Ee]xercise|[Tt]heorem|[Pp]roposition|[Ss]ection|[Cc]hapter|[Ll]emma|[Cc]orollary)'
_NAME = r'[A-Z][a-zA-Z]{2,}'
_NAMES = r'%s(?:\s+(?:and|&)\s+%s)?' % (_NAME, _NAME)  # "Bondy and Murty"
_ATTRIB_VERB = r'(?:writes?|says?|states?|argues?|shows?|proves?|notes?|explains?|defines?)'
_NAMED_SOURCE_RE = re.compile("|".join([
    r"%s(?:'s|’s)\s*(?:\w+\s+){0,3}%s" % (_NAMES, _BIB_NOUN),          # Rudin's [foo] paper
    r"%s.{0,40}?\bby\s+%s\b" % (_BIB_NOUN, _NAMES),                         # book ... by Bondy and Murty
    r"\b%s\b.{0,40}?\b%s\s+[\d.]+\b" % (_NAME, _LOC_WORD),                 # Munkres ... Exercise 70.1
    r"\b%s[\s\$]{0,3}[\d.]+\$?\s+of\b.{0,60}?\bby\s+%s\b" % (_LOC_WORD, _NAMES),  # Section $10.1.3$ of ... by Nakahara
    r"\b%s\b.{0,40}?\b%s\s+%s\b" % (_BIB_NOUN, _NAME, _ATTRIB_VERB),       # book "X", Terence Tao writes
]))

# ---- internal cross-reference: DEFINE vs REFERENCE, kept separate ----
_DEFINE_RE = re.compile(r'\\(?:tag|label)\{?\s*\*?\s*([\w*]+)\}?')
_EXPLICIT_REF_RE = re.compile(
    r'\\(?:eqref|ref)\{?\s*\*?\s*([\w*]+)\}?'
    r'|\bEq(?:uation)?s?\b\.?\s*\(?(\d+[a-z]?)\)?', re.IGNORECASE)
# a bare "(n)" only counts as a back-reference when it is the ENTIRE, PAREN-
# WRAPPED content of its own math span (mimics rendered \eqref output, e.g.
# "$(1)$") -- this excludes both lone variable spans ("$x$") and embedded
# function-argument parens inside a larger formula ("\cos (2)").
_STANDALONE_NUM_RE = re.compile(r'^\(\s*(\d+[a-z]?)\s*\)$')


def _clean_field(v):
    v = (v or "").strip()
    if not v or v.upper() in ("NONE", "N/A", "NA", "-"):
        return ""
    return v


def _named_external_source(text):
    return bool(_NAMED_SOURCE_RE.search(text))


def _crossref_signals(text, ops):
    """Returns (breadth, reuse_ct): distinct DEFINED numbered results (via
    \\tag/\\label, capped 4) and how many of those are REFERENCED AGAIN later
    (via \\eqref/\\ref, an "Eq. n" phrase, or a standalone bare "(n)" span --
    i.e. a math span whose ENTIRE content is just that number, which rules
    out embedded function-argument parens like "cos(2)"; capped 3)."""
    defined = set(_DEFINE_RE.findall(text))

    referenced = set()
    for a, b in _EXPLICIT_REF_RE.findall(text):
        n = a or b
        if n:
            referenced.add(n)

    try:
        spans = ops.extract_math_spans(text)
    except Exception:
        spans = []
    for _kind, content in spans:
        m = _STANDALONE_NUM_RE.match(content.strip())
        if m:
            referenced.add(m.group(1))

    try:
        _n_disp, _n_inline, n_numbered, _avg_tok = ops.equation_stats(text)
    except Exception:
        n_numbered = 0

    breadth = min(4, max(len(defined), n_numbered))
    reuse_ct = min(3, len(defined & referenced))
    return breadth, reuse_ct


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text)
    except Exception:
        t = text if isinstance(text, str) else ""

    try:
        # ---- (A) external attribution (whole document) ----
        cited = _clean_field(extracted.get("cited_source") if extracted else "")
        has_locator = bool(_LOCATOR_RE.search(t) or _PAGE_RE.search(t))
        ext = 0.0
        if cited:
            ext = 0.50
        elif _named_external_source(t):
            ext = 0.35
        elif _GENERIC_EXTERNAL_REF_RE.search(t) and has_locator:
            ext = 0.30
        if _EXTERNAL_CITE_RE.search(t):
            ext = max(ext, 0.35)
        # a bare locator (theorem/section+number with no named/generic source)
        # is exactly the naive-baseline proxy the corpus notes warn about --
        # ordinary math prose is full of "Theorem 3"/"Section 2" with no
        # citation intent. It only GATES (amplifies an already-detected
        # external source), never creates ext on its own.
        if has_locator and ext > 0:
            ext += 0.20
        if _SEE_PHRASE_RE.search(t) and ext > 0:
            ext += 0.05
        ext = min(1.0, ext)

        # ---- (B) internal numbered cross-reference discipline (whole doc) ----
        breadth, reuse_ct = _crossref_signals(t, ops)

        internal = 0.0
        if breadth > 0:
            if reuse_ct > 0:
                internal = 0.30 + 0.03 * breadth + 0.10 * reuse_ct
            else:
                internal = 0.10 + 0.03 * breadth
        internal = min(1.0, internal)

        combined = max(ext, internal) + 0.15 * min(ext, internal)
        return max(0.0, min(1.0, combined))
    except Exception:
        return 0.5
