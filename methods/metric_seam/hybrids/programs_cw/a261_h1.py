"""Hybrid channel for a261 (h1): Line-level clarity, precision, and economy.

Revision rationale (train-residual failure mode, not excerpt-specific fixes):
h0 correctly tanks documents that are mechanically broken (typos, missing
apostrophes, run-ons), but the criterion itself is centrally about
DIRECTNESS / COMPRESSION / avoiding purple or over-earnest phrasing -- it
never mentions spelling at all. h0's only semantic signal for that dimension
was an LLM field capped at quoting 2 example phrases, worth at most -0.10
to the final score. On any document where ornate / over-earnest phrasing
recurs throughout (dense metaphor, said-bookisms, over-described dialogue
beats, over-earnest sci-fi/fantasy narration), that capped signal saturates
immediately, so a mechanically-clean-but-consistently-purple document floats
near 0.80-0.88 no matter how pervasive the purple prose actually is. That
mismatch (mechanics dominant + purple-density signal saturating at 2 items)
is the general failure mode, illustrated by -- but not limited to -- the
worst train cells.

Fixes applied here:
  (a) Replace the capped "quote up to 2 phrases" field with an ORDINAL
      DENSITY rating of purple/over-earnest phrasing across the WHOLE
      narration, so the penalty scales with how pervasive the problem is
      instead of saturating after two examples.
  (b) Rebalance the score so purple-density carries roughly as much weight
      as mechanical cleanliness (previously mechanics alone was ~2x the
      capped purple penalty's max swing), matching what the criterion
      actually asks for.
  (c) Broaden end-of-document boilerplate stripping to also catch
      numbered markdown link lists and generic self-promotion lines
      ("subscribe", "check out", "r/___", "thanks for reading"), per the
      contract's own note that boilerplate clusters near the end of
      scraped documents -- this keeps such filler from diluting narration
      word counts and error density.
  (d) Add a small, general (not excerpt-specific) structural penalty for
      adverb-laden dialogue tags ("she said angrily"), a well-known
      over-earnest/"telling" tell, as a complement to the LLM density
      signal -- both are needed because the LLM field saturates on
      pervasiveness while this catches sparse-but-present cases cheaply.
Everything else (mojibake normalization, chat-line/dialogue stripping,
verse detection, mechanical-error regexes, thin-narration branches) is
retained from h0 unchanged, since it was not implicated in the residuals.
"""

import re
import math

LLM_FIELDS = {
    "narration_errors": (
        "Quote up to 3 unintentional spelling or grammar mistakes in the "
        "story's narration (ignore dialogue, chat lines, intentional style); "
        "answer NONE if narration is clean."
    ),
    "prose_density": (
        "Rate how much of the WHOLE story's narration (not just one line) "
        "reads as purple, wordy, or over-earnest: answer NONE, LIGHT, "
        "MODERATE, or HEAVY, optionally with one example phrase."
    ),
}

_CHAT_LINE = re.compile(
    r"^\s*\*?[A-Z][A-Za-z0-9_@.'\- ]{1,30}:\s"
)
_META_LINE = re.compile(
    r"^\s*(\[|>|&gt;|/r/|\(edit|edit:|ps\b|p\.s\.)", re.IGNORECASE
)
_AUTHOR_NOTE = re.compile(
    r"(please let me know|feedback is welcome|my first post|i write .* stories on)",
    re.IGNORECASE,
)
_LINK_LIST_LINE = re.compile(r"^\s*\d+[.)]\s*\[.+?\]\(.+?\)")
_PROMO_LINE = re.compile(
    r"(thanks for reading|check out|more (?:stories|chapters|of my writing)|"
    r"subscribe|r/[A-Za-z]\w+|reddit\.com|constructive criticism)",
    re.IGNORECASE,
)

_APOS_MISSING = re.compile(
    r"\b(cant|dont|wont|didnt|doesnt|isnt|wasnt|werent|arent|havent|hasnt|"
    r"hadnt|couldnt|wouldnt|shouldnt|im|ive|youre|theyre|thats|theres|whats)\b",
    re.IGNORECASE,
)
_TYPOS = re.compile(
    r"\b(alot|should of|could of|would of|must of|to late|to many|to much|"
    r"teh|wierd|recieve|definately|becuase|untill|seperate|occured|"
    r"apon|thier|freind)\b",
    re.IGNORECASE,
)
_LOWER_I = re.compile(r"(?<![A-Za-z0-9_'])i(?![A-Za-z0-9_'])")
_SPACED_PUNCT = re.compile(r"[ \t]+(?:[,;:](?![,;:])|\.(?!\.))")
_GLUED_STOP = re.compile(r"[a-z][.!?][A-Z]")
_GLUED_COMMA = re.compile(r",(?=[A-Za-z])")
_A_BEFORE_VOWEL = re.compile(
    r"\ba (?![A-Z]{2,}\b)"
    r"(?!(?:one\b|once\b|uni\w*|unit\w*|usual\w*|use\w*|user\w*|unique\w*|"
    r"utopia\w*|euro\w*|ubiq\w*|ufo\w*|u\b))"
    r"([aeiou]\w+)",
    re.IGNORECASE,
)
_DOUBLED = re.compile(r"\b(the|a|an|to|of|in|is|was|and) \1\b", re.IGNORECASE)
_INTENSIFIER = re.compile(
    r"\b(very|really|quite|literally|suddenly|definitely|absolutely|totally|"
    r"completely|extremely|incredibly|truly|utterly|somewhat|basically)\b|"
    r"\b(started to|began to|proceeded to|seemed to)\b",
    re.IGNORECASE,
)
_ELLIPSIS = re.compile(r"\.\.\.|…")
_QUOTE_SPANS = [
    re.compile(r'"[^"\n]{0,600}"'),
    re.compile(r"“[^”\n]{0,600}”"),
    re.compile(r"‘[^’\n]{0,600}’"),
]
_SPEECH_VERB = (
    r"(?:said|asked|replied|answered|whispered|shouted|muttered|grinned|"
    r"smiled|laughed|sighed|snapped|hissed|growled|chuckled|giggled|yelled|"
    r"cried|murmured|stated|declared|exclaimed|wept)"
)
_ADV_DIALOGUE_TAG = re.compile(
    r"\b" + _SPEECH_VERB + r"\b[^.!?\n]{0,20}?\b\w{4,}ly\b", re.IGNORECASE
)


def _clean(text, ops):
    try:
        t = ops.normalize(text)
        if not isinstance(t, str) or not t.strip():
            t = text
    except Exception:
        t = text
    t = t.replace("&amp;#x200B;", " ").replace("&#x200B;", " ")
    t = t.replace("&amp;nbsp;", " ").replace("&nbsp;", " ")
    t = t.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    t = t.replace("[...]", " ")
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"^[-_~=—―\s]{4,}$", "", t, flags=re.MULTILINE)
    return t


def _split_lines(t):
    """Return (narrative_lines, n_chat, n_kept, n_unpunct, mean_line_words)."""
    kept, n_chat, n_unpunct, wsum = [], 0, 0, 0
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if (
            _CHAT_LINE.match(s)
            or _META_LINE.match(s)
            or _AUTHOR_NOTE.search(s)
            or _LINK_LIST_LINE.match(s)
            or _PROMO_LINE.search(s)
        ):
            n_chat += 1
            continue
        kept.append(s)
        wsum += len(s.split())
        if s[-1:] not in ".!?\"”":
            n_unpunct += 1
    mean_lw = wsum / max(1, len(kept))
    return kept, n_chat, len(kept), n_unpunct, mean_lw


def _strip_dialogue(lines):
    narr_parts, quote_parts = [], []
    for s in lines:
        for pat in _QUOTE_SPANS:
            quote_parts.extend(x[1:-1] for x in pat.findall(s))
            s = pat.sub(" ", s)
        # dangling multi-line quote: drop from unmatched opener to line end
        for opener in ('"', "“"):
            i = s.find(opener)
            if i >= 0:
                quote_parts.append(s[i + 1:])
                s = s[:i]
        s = s.strip()
        if s:
            narr_parts.append(s)
    return " ".join(narr_parts), " ".join(quote_parts)


def _count_field_items(ans):
    """Code predicate over an LLM extraction: how many items did it return?"""
    if not isinstance(ans, str):
        return None  # field absent -> no information
    a = ans.strip().strip(".").strip()
    if not a or a.lower() in ("none", "no", "n/a", "clean", "nothing"):
        return 0
    quoted = re.findall(r'"[^"]{2,80}"|“[^”]{2,80}”', a)
    if quoted:
        return min(3, len(quoted))
    parts = [p for p in re.split(r"[;\n]| \d[.)] ", a) if p.strip()]
    return min(3, max(1, len(parts)))


_DENSITY_KEYWORDS = (
    ("none", 0.0), ("clean", 0.0), ("no purple", 0.0),
    ("light", 0.33), ("mild", 0.33), ("occasional", 0.33),
    ("isolated", 0.33), ("some", 0.33), ("low", 0.33),
    ("moderate", 0.66), ("several", 0.66), ("medium", 0.66),
    ("recurs", 0.66), ("frequent", 0.85),
    ("heavy", 1.0), ("pervasive", 1.0), ("throughout", 1.0), ("high", 1.0),
)


def _parse_density(ans):
    """Ordinal purple-prose density in [0,1], or None if uninformative.

    Prefers the requested NONE/LIGHT/MODERATE/HEAVY vocabulary but falls
    back gracefully to a count-based estimate if the extractor answers in
    the older "quote phrases" style, so the signal degrades rather than
    breaks on off-format LLM answers.
    """
    if not isinstance(ans, str):
        return None
    a = ans.strip()
    if not a:
        return None
    low = a.lower()
    for key, val in _DENSITY_KEYWORDS:
        if key in low:
            return val
    if low.strip(".") in ("n/a",):
        return None
    quoted = re.findall(r'"[^"]{2,80}"|“[^”]{2,80}”', a)
    if quoted:
        n = len(quoted)
        return 0.33 if n <= 1 else (0.66 if n == 2 else 1.0)
    # non-empty, unrecognized format: assume a mild presence rather than 0
    return 0.4


def _mech_score(seg, halflife=7.0):
    words = re.findall(r"[A-Za-z']+", seg)
    W = max(1, len(words))
    per_k = 1000.0 / W
    err = 0.0
    err += 1.0 * len(_LOWER_I.findall(seg))
    err += 1.0 * len(_APOS_MISSING.findall(seg))
    err += 1.0 * len(_TYPOS.findall(seg))
    err += 1.0 * len(_SPACED_PUNCT.findall(seg))
    err += 1.0 * len(_GLUED_STOP.findall(seg))
    err += 1.0 * len(_GLUED_COMMA.findall(seg))
    err += 0.5 * len(_A_BEFORE_VOWEL.findall(seg))
    err += 0.5 * len(_DOUBLED.findall(seg))
    return math.exp(-(err * per_k) / halflife)


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t = _clean(text, ops)
        lines, n_chat, n_kept, n_unpunct, mean_lw = _split_lines(t)
        total_lines = max(1, n_chat + n_kept)
        chat_frac = n_chat / total_lines
        # verse/doggerel: many shortish lines left dangling without end punctuation
        verse = (
            n_kept >= 8
            and (n_unpunct / max(1, n_kept)) >= 0.5
            and mean_lw <= 14
        )

        narr, quoted = _strip_dialogue(lines)
        words = re.findall(r"[A-Za-z']+", narr)
        W = len(words)
        Q = len(re.findall(r"[A-Za-z']+", quoted))

        ex = extracted if isinstance(extracted, dict) else {}
        n_err = _count_field_items(ex.get("narration_errors"))
        density = _parse_density(ex.get("prose_density"))

        if W < 40:
            if Q >= 60:
                # Nearly pure dialogue: judge mechanics on the speech itself
                # (economy norms don't transfer to voiced lines).
                base = 0.20 + 0.62 * _mech_score(quoted, halflife=10.0)
            else:
                # Chat-log / transcript form: the narrator's line-level craft
                # is unobservable; corpus prior is mildly positive.
                base = 0.72
        else:
            per_k = 1000.0 / W
            mech = _mech_score(narr)

            sents = [s for s in re.split(r"(?<=[.!?])\s+", narr) if s.split()]
            slens = [len(s.split()) for s in sents] or [0]
            frac_run = sum(1 for L in slens if L > 40) / len(slens)
            run_pen = min(1.0, frac_run / 0.25)

            n_int = len(_INTENSIFIER.findall(narr))
            clut_pen = min(1.0, max(0.0, n_int * per_k - 10.0) / 25.0)

            n_adv = sum(
                1 for w in words if len(w) > 5 and w.lower().endswith("ly")
            )
            adv_pen = min(1.0, max(0.0, n_adv * per_k - 12.0) / 25.0)

            ell_pen = min(1.0, max(0.0, len(_ELLIPSIS.findall(narr)) * per_k - 3.0) / 10.0)
            exc_pen = min(1.0, max(0.0, narr.count("!") * per_k - 4.0) / 12.0)

            # General structural tell for over-earnest "telling": adverb
            # riding a speech verb ("she said angrily"). Raw count, not
            # length-normalized, since dialogue tags are naturally sparse.
            n_advtag = len(_ADV_DIALOGUE_TAG.findall(t))
            advtag_pen = min(1.0, n_advtag / 4.0)

            econ = max(
                0.0,
                1.0
                - 0.30 * run_pen
                - 0.20 * clut_pen
                - 0.12 * adv_pen
                - 0.10 * ell_pen
                - 0.14 * exc_pen
                - 0.14 * advtag_pen,
            )

            # Purple/over-earnest density: co-equal pillar with mechanics,
            # not a bounded side-penalty, because the criterion is centrally
            # about directness/compression rather than spelling. Missing
            # field info defaults to a mild-presence prior (0.25) rather
            # than assuming clean prose.
            dens = density if density is not None else 0.25
            purple_component = 1.0 - dens

            base = 0.06 + 0.34 * mech + 0.20 * econ + 0.30 * purple_component
            if verse and chat_frac < 0.3:
                base -= 0.12  # rhymed/line-broken doggerel reads as low-craft here

        delta = 0.0
        if n_err is not None:
            delta += 0.04 if n_err == 0 else -0.05 * min(3, n_err)

        return max(0.0, min(1.0, base + delta))
    except Exception:
        return 0.5
