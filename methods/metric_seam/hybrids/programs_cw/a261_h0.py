"""Hybrid channel for a261: Line-level clarity, precision, and economy.

Core idea learned from train residuals: the judge rewards clean, controlled,
economical NARRATION. Low scorers are mechanically broken (lowercase "i",
missing apostrophes, glued/spaced punctuation, run-ons, ellipsis/exclamation
melodrama); high scorers are clean and punchy. Crucially, one 0.95 story is a
chat-log full of INTENTIONAL typos inside chat lines, so all mechanics checks
run on narration only (quoted dialogue and chat-format lines stripped).
LLM fields cover what regex can't see: agreement/homophone errors in
narration, and purple/wordy phrasing.
"""

import re
import math

LLM_FIELDS = {
    "narration_errors": (
        "Quote up to 3 unintentional spelling or grammar mistakes in the "
        "story's narration (ignore dialogue, chat lines, intentional style); "
        "answer NONE if narration is clean."
    ),
    "overwritten_phrases": (
        "Quote up to 2 wordy, purple, or over-earnest phrases a strict line "
        "editor would cut; answer NONE if the prose is lean and direct."
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
        if _CHAT_LINE.match(s) or _META_LINE.match(s) or _AUTHOR_NOTE.search(s):
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

            econ = max(
                0.0,
                1.0
                - 0.35 * run_pen
                - 0.25 * clut_pen
                - 0.15 * adv_pen
                - 0.15 * ell_pen
                - 0.25 * exc_pen,
            )
            base = 0.10 + 0.50 * mech + 0.28 * econ
            if verse and chat_frac < 0.3:
                base -= 0.15  # rhymed/line-broken doggerel reads as low-craft here

        delta = 0.0
        ex = extracted if isinstance(extracted, dict) else {}
        n_err = _count_field_items(ex.get("narration_errors"))
        if n_err is not None:
            delta += 0.07 if n_err == 0 else -0.07 * min(3, n_err)
        n_purple = _count_field_items(ex.get("overwritten_phrases"))
        if n_purple is not None:
            delta += 0.05 if n_purple == 0 else -0.05 * min(2, n_purple)

        return max(0.0, min(1.0, base + delta))
    except Exception:
        return 0.5
