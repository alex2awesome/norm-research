"""
Hybrid metric for a198: "Citations and cross-references: precision, placement,
and attribution" on Math StackExchange answers.

h1 revision of h0's two decoupled signals (external attribution / internal
numbered cross-reference discipline). h0 (train rho 0.589) beat the keyword
baseline but the TRAIN residual cells exposed three GENERAL failure modes in
how the two signals were computed -- not problems with the overall two-signal
design, which is kept:

1. The internal cross-reference detector treated ANY bare parenthetical
   integer, e.g. "(2)", as an equation-reference token. But ordinary math
   prose is full of parenthetical integers that are NOT equation labels:
   function evaluations ("cos(2)", "q(0)"), coefficients/indices ("(2n)"),
   etc. Because this pattern is ubiquitous in ANY math answer (not specific
   to any excerpt), it silently inflated the "internal discipline" score for
   ordinary derivations that never actually tag-and-reuse a numbered result.
   Fix: only accept a bare "(n)" as a reference candidate when it is NOT
   immediately adjacent (modulo whitespace) to an identifier/closing-brace
   (which marks it as a function call / subscripted quantity, not a
   standalone reference marker), and require the parenthesized content to be
   a bare integer (drop the trailing-letter allowance that let coefficient-
   like tokens such as "2n" masquerade as equation labels).

2. The external-attribution field ("cited_source") was trusted at face
   value once non-empty, worth a flat 0.50 regardless of whether the named
   source is genuinely grounded in the ANSWER text. LLM extractors can leak
   content from the QUESTION span into an "answer-only" field (a source
   named only in the question, or a bare link that is never actually
   restated in the answer) -- this is a general grounding-failure mode of
   any thick-input LLM field, not something regex can see, so it needs a
   code-side check: discount `cited_source` entirely unless some
   distinctive word/fragment of it literally occurs in the answer span.

3. Even when a citation IS genuine, h0 gave the same flat 0.50 credit to a
   precisely located citation (book + theorem + page) and to a bare named
   mention with no locatable position at all (a plain URL, a same-thread
   "so-and-so's answer" credit). The criterion explicitly defines precision
   as citing "precise locations (theorem/page numbers, editions/volumes)",
   so the code should reward the CO-OCCURRENCE of a named source and a
   locator much more than either alone, rather than crediting bare naming
   almost as if it already were a precise citation.

None of these three fixes are keyed to any specific document; they are
general corrections to what the two signals actually measure. The
combination formula and everything else about h0's architecture is
unchanged.
"""

import re

LLM_FIELDS = {
    "cited_source": (
        "Using ONLY the sentences after 'Answer:' (ignore the question "
        "entirely), does the answer itself name a specific external source "
        "it relies on -- a book/paper title with author, a specific "
        "locatable web page, or another identified user's specific "
        "answer/comment? Name it in <=12 words if so. Reply NONE if the "
        "answer names no such source, or if the only candidate source is "
        "mentioned solely in the question and not repeated in the answer."
    ),
}

_ANSWER_SPLIT_RE = re.compile(r'(?i)\banswer\s*:')
_EXTERNAL_CITE_RE = re.compile(r'\\(cite|citep|citet|autoref)\{')
_SEE_PHRASE_RE = re.compile(r'(?i)\b(see|cf\.|c\.f\.)\b')
_GENERIC_EXTERNAL_REF_RE = re.compile(
    r'(?i)\b(this|that|the)\s+(paper|book|article|thesis|text|source|reference|note)s?\b'
)
# theorem/lemma/.../equation NUMBER, tolerant of LaTeX noise ($, (, ), \) between
# the keyword and the digits.
_LOCATOR_RE = re.compile(
    r'(?i)\b(theorem|lemma|proposition|prop|corollary|section|chapter|equation|eq)s?\.?'
    r'[\s\$\(\)\[\]\\]{0,6}(\d+|[ivxlcm]+)\b'
)
_PAGE_RE = re.compile(r'(?i)\bpp?\.[\s\$]{0,3}\d+')

# Genuine LaTeX numbering commands (still a strong, unambiguous signal), plus
# an explicit "Eq./Equation N" phrase, plus a bare "(N)" -- but the bare form
# is filtered post-match (see _bare_paren_is_reference) and restricted to a
# pure integer, since a trailing letter ("2n", "3k") is far more often a
# coefficient/index than an equation sub-label.
_NUMBER_TOKEN_RE = re.compile(
    r'\\(?:tag|label|eqref|ref)\{?\s*(?:eqs?\.?s?\.?)?\s*(\d+[a-z]?)\}?'
    r'|\bEq(?:uation)?s?\.?\s*\(?(\d+[a-z]?)\)?'
    r'|\((\d+)\)',
    re.IGNORECASE,
)

_IDENT_ADJ_RE = re.compile(r'[A-Za-z0-9_\}]')
_WORD_RE = re.compile(r'[A-Za-z]{3,}')


def _split_answer(t):
    m = _ANSWER_SPLIT_RE.search(t)
    return t[m.end():] if m else t


def _clean_field(v):
    v = (v or "").strip()
    if not v or v.upper() in ("NONE", "N/A", "NA", "-"):
        return ""
    return v


def _bare_paren_is_reference(t, open_paren_idx):
    """A bare "(N)" only counts as an equation-reference marker if it is NOT
    immediately preceded (modulo whitespace) by an identifier/brace -- that
    adjacency pattern ("cos (2)", "q(0)", "f(1)") marks ordinary function
    application, not a stand-alone cross-reference to a tagged result."""
    i = open_paren_idx - 1
    while i >= 0 and t[i] in ' \t':
        i -= 1
    if i < 0:
        return True
    return not _IDENT_ADJ_RE.match(t[i])


def _grounded_in_answer(cited, ans):
    """Guard against LLM field leakage/hallucination: require at least one
    distinctive word (or, for word-less strings like bare URLs, the
    fragment itself) of the extracted citation to literally occur in the
    answer span -- otherwise we cannot trust it actually came from the
    answer rather than the question or elsewhere."""
    words = _WORD_RE.findall(cited)
    ans_low = ans.lower()
    if not words:
        frag = cited.strip().lower()
        return len(frag) >= 6 and frag in ans_low
    return any(w.lower() in ans_low for w in words)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text)
    except Exception:
        t = text if isinstance(text, str) else ""

    try:
        ans = _split_answer(t)

        # ---- (A) external attribution ----
        cited_raw = _clean_field(extracted.get("cited_source") if extracted else "")
        cited = cited_raw if (cited_raw and _grounded_in_answer(cited_raw, ans)) else ""

        has_locator = bool(_LOCATOR_RE.search(ans) or _PAGE_RE.search(ans))
        has_cite_cmd = bool(_EXTERNAL_CITE_RE.search(ans))
        has_see = bool(_SEE_PHRASE_RE.search(ans))

        if cited and has_locator:
            # named source AND a precise locator -- exactly what the
            # criterion means by "precise locations".
            ext = 0.65
        elif cited:
            # named but no pinpoint locator: a real attribution, but a
            # weaker/imprecise one (bare link, same-thread name-drop).
            ext = 0.40
        elif _GENERIC_EXTERNAL_REF_RE.search(ans) and has_locator:
            ext = 0.30
        elif has_locator:
            ext = 0.20
        else:
            ext = 0.0

        if has_cite_cmd:
            ext += 0.15
        if has_see and ext > 0:
            ext += 0.05
        ext = min(1.0, ext)

        # ---- (B) internal numbered cross-reference discipline ----
        positions = {}
        for m in _NUMBER_TOKEN_RE.finditer(ans):
            g1, g2, g3 = m.group(1), m.group(2), m.group(3)
            if g3 is not None:
                if not _bare_paren_is_reference(ans, m.start()):
                    continue
                val = g3
            else:
                val = g1 or g2
            if val:
                positions.setdefault(val, []).append(m.start())

        n_numbered = 0
        try:
            eq_stats = ops.equation_stats(ans)
            if isinstance(eq_stats, dict):
                for key in ("n_numbered", "numbered", "n_tagged", "tagged"):
                    if key in eq_stats and isinstance(eq_stats[key], (int, float)):
                        n_numbered = eq_stats[key]
                        break
        except Exception:
            n_numbered = 0

        breadth = min(4, max(len(positions), int(n_numbered)))
        reused = [v for v, ps in positions.items() if len(ps) >= 2]
        reuse_ct = min(3, len(reused))

        internal = 0.0
        if breadth > 0:
            if reuse_ct > 0:
                internal = 0.30 + 0.03 * breadth + 0.10 * reuse_ct
            else:
                # tagged/numbered but never referenced again: weak signal only.
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
            n_numbered = len(re.findall(r'\\tag', s))
            return {"n_numbered": n_numbered}

    ops = _Ops()

    cases = [
        ("no citation, plain proof",
         "Question: q\n\nAnswer: Yes, this follows directly from the definitions.",
         {}, 0.0),
        ("function-call parens, not a reference",
         "Question: q\n\nAnswer: We have $\\cos (2)$ and $q(0)$ appearing twice each in the sum.",
         {}, None),
        ("internal tag defined and reused at a distance",
         "Question: q\n\nAnswer: We have $$x=1 \\tag{1}$$ Using Equation 1 above we conclude the result.",
         {}, None),
        ("named source + locator, grounded in answer",
         "Question: q\n\nAnswer: See Rudin's Principles, Theorem 3.2, page 45 for the argument.",
         {"cited_source": "Rudin, Principles of Mathematical Analysis"}, None),
        ("cited field leaked from question, not grounded",
         "Question: This is from Arnold's ODE, a hard problem.\n\nAnswer: Consider the domain D and proceed directly.",
         {"cited_source": "Arnold's ODE"}, None),
        ("vague pointer only",
         "Question: q\n\nAnswer: As mjw mentioned above, this is straightforward.",
         {"cited_source": ""}, None),
    ]
    for name, txt, ext_fields, _ in cases:
        print(name, "->", round(score(txt, ext_fields, ops), 3))
