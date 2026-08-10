"""
Hybrid metric for a198: "Citations and cross-references: precision, placement,
and attribution" on Math StackExchange answers.

Design rationale (see corpus known_failure_pattern): the v0 keyword baseline
(counting \\cite{.. / theorem|section|p. + digit) gets rho=0.32 because raw
keyword presence is a weak proxy -- MSE answers are full of "Theorem"/"p."-like
tokens from ordinary math prose that have nothing to do with citation
discipline, and LaTeX noise ($, parens) breaks naive "\\s*\\d" adjacency.

Two independent, decoupled signals actually drive the judge:
  (A) EXTERNAL attribution: does the answer name a specific, locatable outside
      source (a book+author, a named person's answer/post, a website) rather
      than just a generic theorem name ("the mean value theorem") or a vague
      pointer ("as X mentioned above")? This distinction is a THICK semantic
      judgment a regex cannot make reliably -> one LLM field.
  (B) INTERNAL cross-reference discipline: numbered results/equations that are
      DEFINED and then actually REFERENCED AGAIN later (not just tagged once
      and forgotten). This is checkable by code via LaTeX-aware tag/back-
      reference matching, and rewards it independently of (A) -- several
      high-judged answers cite nothing external but tag+reuse their own
      numbered equations extensively.
Both analyses are restricted to the ANSWER span (after "Answer:") so that
citation-like language typed by the QUESTION asker doesn't leak in.
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

_ANSWER_SPLIT_RE = re.compile(r'(?i)\banswer\s*:')
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
# a numbered result defined via \tag{...}/\label{...}/\eqref{...}/\ref{...},
# optionally with an "Eq./Equation" word inside the braces, or a bare (n)/(na).
_NUMBER_TOKEN_RE = re.compile(
    r'\\(?:tag|label|eqref|ref)\{?\s*(?:eqs?\.?s?\.?)?\s*(\d+[a-z]?)\}?'
    r'|\bEq(?:uation)?s?\.?\s*\(?(\d+[a-z]?)\)?'
    r'|\(\s*(\d+[a-z]?)\s*\)',
    re.IGNORECASE,
)


def _split_answer(t):
    m = _ANSWER_SPLIT_RE.search(t)
    return t[m.end():] if m else t


def _clean_field(v):
    v = (v or "").strip()
    if not v or v.upper() in ("NONE", "N/A", "NA", "-"):
        return ""
    return v


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text)
    except Exception:
        t = text if isinstance(text, str) else ""

    try:
        ans = _split_answer(t)

        # ---- (A) external attribution ----
        cited = _clean_field(extracted.get("cited_source") if extracted else "")
        has_locator = bool(_LOCATOR_RE.search(ans) or _PAGE_RE.search(ans))
        ext = 0.0
        if cited:
            ext = 0.50
        elif _GENERIC_EXTERNAL_REF_RE.search(ans) and has_locator:
            # points at a real external document + a precise numbered target,
            # even though the LLM couldn't/didn't pin an exact title.
            ext = 0.30
        if has_locator:
            ext += 0.20
        if _SEE_PHRASE_RE.search(ans) and ext > 0:
            ext += 0.05
        if _EXTERNAL_CITE_RE.search(ans):
            ext += 0.15
        ext = min(1.0, ext)

        # ---- (B) internal numbered cross-reference discipline ----
        matches = _NUMBER_TOKEN_RE.findall(ans)
        nums = [g for groups in matches for g in groups if g]
        counts = {}
        for n in nums:
            counts[n] = counts.get(n, 0) + 1

        # supplement with the math-aware op's own numbered-equation count,
        # in case this doc numbers equations in a style our regex misses.
        try:
            _n_disp, _n_inline, n_numbered, _avg_tok = ops.equation_stats(ans)
        except Exception:
            n_numbered = 0

        breadth = min(4, max(len(counts), n_numbered))
        reused = [n for n, c in counts.items() if c >= 2]
        reuse_ct = min(3, len(reused))

        internal = 0.0
        if breadth > 0:
            if reuse_ct > 0:
                internal = 0.30 + 0.03 * breadth + 0.10 * reuse_ct
            else:
                # tagged/numbered but never referenced again: weak signal only
                # -- presence without reuse is exactly the proxy we decouple.
                internal = 0.10 + 0.03 * breadth
        internal = min(1.0, internal)

        combined = max(ext, internal) + 0.15 * min(ext, internal)
        return max(0.0, min(1.0, combined))
    except Exception:
        return 0.5


if __name__ == "__main__":
    class _Ops:
        @staticmethod
        def normalize(s):
            return s

        @staticmethod
        def equation_stats(s):
            n_disp = len(re.findall(r'\$\$|\\\[|\\begin\{align', s))
            n_numbered = len(re.findall(r'\\tag', s))
            return (n_disp, 0, n_numbered, 10.0)

    ops = _Ops()

    cases = [
        ("no citation, plain proof", "Question: q\n\nAnswer: Yes, this follows directly from the definitions.", {}, 0.0),
        ("internal tag reuse", "Question: q\n\nAnswer: We have $$x=1 \\tag{1}$$ Using Equation 1 above we conclude the result.", {}, None),
        ("named source + locator", "Question: q\n\nAnswer: See Rudin's Principles, Theorem 3.2, page 45 for the argument.", {"cited_source": "Rudin, Principles of Mathematical Analysis"}, None),
        ("vague pointer only", "Question: q\n\nAnswer: As mjw mentioned above, this is straightforward.", {"cited_source": ""}, None),
    ]
    for name, txt, ext_fields, _ in cases:
        print(name, "->", round(score(txt, ext_fields, ops), 3))
