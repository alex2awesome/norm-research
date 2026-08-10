"""Hybrid metric channel for a168: Symbol and language precision in mathematical prose.

Design rationale (see report to caller for full rationale):
- The description-compiled baseline (train rho=0.079) rewarded raw keyword presence
  ("for all", "such that", ...) and crude balance checks -- both weak/near-noise.
- Reading the 30 train examples, the sharpest recurring low-score pattern is BARE
  logical/quantifier symbols ($\\forall$, $\\in$, $\\land$, ...) dropped as isolated
  inline-math tokens directly into an English sentence in place of the word itself
  (e.g. "... divides |G| $\\forall$ $g$ $\\in$ $G$"), and misuse of a relation symbol
  outside its domain (e.g. $\\cong$ for modular congruence instead of $\\equiv$).
  High-score examples instead translate quantifiers into words ("for all", "such
  that") or keep symbols entirely inside one coherent formula, and show careful
  equality/derivation chains (aligned "&=" steps, \\tag references).
- Two things a parser cannot see reliably: (a) whether a *specific* symbol/relation
  is used incorrectly or left ambiguously unparenthesized, and (b) whether the prose
  states a substantive, precise mathematical claim at all versus a vague non-answer
  (both classes exist in-corpus independent of whether any LaTeX is present -- some
  all-prose answers are exemplary, some are vacuous filler). Both need a reader, so
  they are the two LLM_FIELDS; the checkable predicates (bare-symbol-in-prose,
  delimiter hygiene, chain structure, cong/mod domain mismatch) stay in code.
"""

import re

LLM_FIELDS = {
    "notation_flaw": (
        "In one short phrase, name ONE specific symbol, operator, or relation this "
        "answer uses incorrectly, inconsistently, or ambiguously (wrong relation "
        "symbol, unparenthesized ambiguous formula, a quantifier symbol used as a "
        "word in a sentence); say NONE if notation is used correctly throughout."
    ),
    "claim_quality": (
        "In one or two words: does the answer give a specific, substantive "
        "mathematical claim, definition, or computation using correct technical "
        "terms (SUBSTANTIVE), or is it vague, generic, or a non-answer with no "
        "concrete content (VAGUE)?"
    ),
}

_IDIOM_PHRASES = (
    "for all", "for every", "for each", "there exists", "there exist",
    "if and only if", "such that",
)

# Bare command reduces an entire inline-math span to a single logical/quantifier
# token used as a stand-in for an English word -- the "shorthand in prose" flaw.
_BARE_RE = re.compile(
    r"^\\(forall|exists|in|notin|ni|land|lor|neg|lnot|subset|subseteq|supseteq|supset)$"
)
# Parenthesized single-arrow proof-direction labels, e.g. "(\Longleftarrow)" heading
# a proof case, are a standard, legitimate convention -- excluded from the penalty.
_DIRECTION_RE = re.compile(
    r"^\\(Longleftarrow|Longrightarrow|Leftarrow|Rightarrow|Leftrightarrow|iff|implies|to|gets)$"
)

_CHAIN_RE = re.compile(r"[<>=]\s*[^\s<>=]{1,20}\s*[<>=]")


def _unwrap(s):
    s = s.strip()
    if len(s) >= 2 and s[0] in "([" and s[-1] in ")]":
        return s[1:-1].strip()
    return None


def _count_bare_spans(spans):
    n = 0
    for item in spans:
        try:
            kind, content = item[0], item[1]
        except Exception:
            continue
        try:
            core = content.strip()
            inner = _unwrap(core)
            check = inner if inner is not None else core
            if inner is not None and _DIRECTION_RE.fullmatch(inner):
                continue
            if _BARE_RE.fullmatch(check):
                n += 1
        except Exception:
            continue
    return n


def _hygiene_score(ans, ops):
    try:
        dh = ops.delimiter_health(ans)
        if not isinstance(dh, dict):
            return 0.7
        total = 0.0
        for v in dh.values():
            if isinstance(v, (int, float)):
                total += max(0.0, float(v))
        return 1.0 / (1.0 + total)
    except Exception:
        return 0.7


def _naked_score(ans, ops):
    try:
        spans = ops.extract_math_spans(ans)
        if not isinstance(spans, list):
            return 0.8
        bare = _count_bare_spans(spans)
        return 1.0 / (1.0 + bare)
    except Exception:
        return 0.8


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0

        try:
            norm = ops.normalize(text)
        except Exception:
            norm = text

        m = re.search(r"Answer:\s*", norm)
        ans = norm[m.end():] if m else norm
        if not ans.strip():
            ans = norm

        hygiene = _hygiene_score(ans, ops)
        naked = _naked_score(ans, ops)

        low = ans.lower()
        idiom_count = sum(1 for p in _IDIOM_PHRASES if p in low)
        idiom = min(1.0, idiom_count / 2.0)

        chain_count = len(_CHAIN_RE.findall(ans))
        chain = min(1.0, chain_count / 2.0)

        cong_mod_penalty = 0.15 if (re.search(r"\\cong", ans) and re.search(r"\bmod\b", ans, re.I)) else 0.0

        flaw_text = str(extracted.get("notation_flaw", "") or "").strip()
        flaw_present = bool(flaw_text) and not flaw_text.strip(" .").lower().startswith("none")
        flaw_score = 0.15 if flaw_present else 0.85

        claim_text = str(extracted.get("claim_quality", "") or "").strip().lower()
        if "vague" in claim_text or "generic" in claim_text or "non-answer" in claim_text:
            claim_score = 0.15
        elif "substantive" in claim_text:
            claim_score = 0.85
        else:
            claim_score = 0.5

        W_HYGIENE, W_NAKED, W_IDIOM, W_CHAIN, W_FLAW, W_CLAIM = 0.20, 0.15, 0.10, 0.10, 0.25, 0.20

        final = (
            W_HYGIENE * hygiene
            + W_NAKED * naked
            + W_IDIOM * idiom
            + W_CHAIN * chain
            + W_FLAW * flaw_score
            + W_CLAIM * claim_score
            - cong_mod_penalty
        )
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
