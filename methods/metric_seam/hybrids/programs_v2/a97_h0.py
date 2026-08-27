"""a97 hybrid -- "Specific, complete, checkable facts and claims".

Predicate: saturating DENSITY of concrete, checkable fact tokens (money,
percents, dated events, counts-with-units, statutory citations, clock
times, tickers, attributed quotes) counted ONLY inside prose sentences.
Nav chrome, link lists, tables and footer contact blocks contribute
nothing (presence != quality: a number only counts when embedded in a
real sentence, and the score is a per-word density, not a raw count).
Release-likeness damping: documents with almost no prose floor near 0.
Overlap double-counting is prevented by masking matched spans between
pattern passes.
"""
import math
import re

LLM_FIELDS = {
    "main_fact": "In <=15 words, state the single most specific new fact/result/event this document announces; answer NONE if it announces nothing specific.",
    "fact_count": "Answer only an integer 0-20: how many distinct concrete checkable facts (figures, dates, amounts, prices, named partners) support this document's main announcement? NONE if none.",
}

# --- prose detection -------------------------------------------------------
_FUNC_WORDS = frozenset("""
the of and to in for with that is are was were has have had will would on as
by at from its it this these those be been being an or which their our we he
she they but not can could may might should than into after before during
over under about more most also
de la el en que los las una para con por del al es un le des und der die
""".split())

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-zÀ-ſ']+")
# abbreviation-final fragments get re-merged so "Pub. L. 94-409", "Richard S.
# Hartunian" or "Sept. 10, 2019" are not shredded by the sentence splitter
_ABBREV_END = re.compile(
    r"(?:\b[A-Z]|\b(?:Mr|Mrs|Ms|Dr|St|No|Pub|Inc|Corp|Ltd|Co|vs|Gov|Sen|Rep|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec))\.$")


def _sentences(line):
    parts = _SENT_SPLIT.split(line)
    merged = []
    for p in parts:
        if merged and _ABBREV_END.search(merged[-1].rstrip()):
            merged[-1] = merged[-1].rstrip() + " " + p
        else:
            merged.append(p)
    return merged


def _prose_segments(t):
    """Yield unique sentence-like segments that look like running prose.

    Dedup matters: scraped pages often repeat whole blocks (head==tail),
    which would otherwise dilute density and double-count capped facts.
    """
    out = []
    seen = set()
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = _sentences(line) if len(line) > 180 else [line]
        for seg in parts:
            words = _WORD.findall(seg)
            if len(words) < 8:
                continue
            nfunc = sum(1 for w in words if w.lower() in _FUNC_WORDS)
            if nfunc < 2 or nfunc / len(words) < 0.08:
                continue  # nav lists / table rows / title-case link stacks
            key = re.sub(r"\s+", " ", seg.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            out.append((seg, len(words)))
    return out


# --- fact-token patterns (priority order; matched spans are masked) --------
_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_FACT_PATTERNS = [
    # (weight, cap, compiled_regex)
    (1.2, 12, re.compile(r"(?:[$£€]|\bUS\$|\bUSD\s|\bEUR\s|\bGBP\s)\s?\d[\d,]*(?:\.\d+)?\s?(?:billion|million|trillion|bn|m\b)?", re.I)),
    (1.0, 12, re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent(?:age\s+points?)?|per\s+cent)", re.I)),
    (1.0, 8, re.compile(r"\b\d[\d,]*(?:\.\d+)?\s(?:billion|million|trillion)\b(?:\s?(?:kroner|dollars|euros|pounds))?", re.I)),
    (0.8, 8, re.compile(r"\b" + _MONTH + r"\.?\s+\d{1,2}(?:\s*,\s*\d{4})?\b|\b\d{1,2}\s+" + _MONTH + r"\w*\.?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.I)),
    (0.5, 4, re.compile(r"\b" + _MONTH + r"\w*\.?\s+\d{4}\b")),
    (0.7, 4, re.compile(r"\b\d{1,2}:\d{2}\s?(?:a\.?m\.?|p\.?m\.?)", re.I)),
    (0.7, 3, re.compile(r"\b(?:NYSE|NASDAQ|ASX|LSE|TSX|OTC|AMEX|Euronext|AIM)\s?:\s?[A-Z][A-Z.]{0,6}")),
    (0.7, 6, re.compile(r"\b(?:Pub\.?\s?L\.?\s?[\d-]+|Form\s+[0-9][0-9A-Z-]*|Rule\s+\d[\w().-]*|Release\s+No\.?\s?[\w-]*\d[\w-]*|No\.\s?[A-Z0-9-]*\d[\w-]*|\d+\s+FR\s+\d+|U\d{2}[A-Z]{2}\d+|Section\s+\d+)", re.I)),
    (0.7, 10, re.compile(r"\b\d[\d,]*(?:\.\d+)?[\s-](?:[a-z-]+\s)?(?:people|employees|units|shares|stores|facilities|customers|patients|countries|offices|associates|analysts|consultants|properties|listings|members|staff|locations|jobs|square\s+(?:feet|met(?:er|re)s)|nautical\s+miles|miles|hours?|minutes?|days?|weeks?|months?|years?)\b", re.I)),
    # named, titled sources = checkable attribution ("Jane Doe, chief x officer")
    (0.5, 5, re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.?)*\s+[A-Z][a-z]+\s*,\s*(?:(?:[a-z]+\s+){0,3}(?:president|chief|director|officer|secretary|analyst|attorney|adviser|advisor|manager|head|lead|principal|scientist|investigator|professor|economist|strategist|counsel|founder|partner)|Special\s+Agent|U\.S\.\s+Attorney|CEO|CTO|CFO)\b")),
    (0.5, 3, re.compile(r"\b(?:United\s+States\s+(?:Attorney|Marshal)|U\.S\.\s+Attorney|Attorney\s+General|Special\s+Agent\s+in\s+Charge|Foreign\s+Secretary|Secretary-General)\s+[A-Z][a-z]+")),
    (0.5, 2, re.compile(r"\b\d+\s+[A-Z][\w]*\s+(?:Street|St\.|Avenue|Ave\.?|Boulevard|Blvd\.?|Road|Rd\.?|Drive|Place|Plaza)\b")),
    (0.5, 3, re.compile(r"\b\d+(?:\.\d+)?x\b")),
    (0.4, 4, re.compile(r"\bQ[1-4]\s?(?:20\d{2}|1\d)?\b|\b(?:first|second|third|fourth)\s+quarter\b", re.I)),
    (0.6, 6, re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")),
    (0.3, 2, re.compile(r"\(\d{3}\)\s?\d{3}[-.\s]\d{4}\b|\b\d{3}[-.]\d{3}[-.]\d{4}\b|\+\d{1,3}[\s.-]?\d{2,4}[\s.-]\d{3,4}[\s.-]\d{2,4}\b")),
    (0.15, 6, re.compile(r"\b(?:19|20)\d{2}\b")),
    (0.08, 10, re.compile(r"\b\d+(?:\.\d+)?\b")),
]

_QUOTE = re.compile(r'"[^"]{25,400}"')
_SAID = re.compile(r"\b(?:said|says|stated|added|noted|announced|commented)\b", re.I)
_RELEASE_MARK = re.compile(
    r"PRNewswire|GLOBE\s?NEWSWIRE|Business\s?Wire|Marketwired|FOR\s+IMMEDIATE\s+RELEASE|PRESS\s+RELEASE|News\s+Releases?", re.I)


def _fact_mass(prose_text):
    """Weighted count of checkable-fact tokens; spans masked between passes."""
    buf = prose_text
    total = 0.0
    for weight, cap, pat in _FACT_PATTERNS:
        hits = 0
        pieces = []
        last = 0
        for m in pat.finditer(buf):
            hits += 1
            pieces.append(buf[last:m.start()])
            pieces.append(" " * (m.end() - m.start()))
            last = m.end()
        if hits:
            pieces.append(buf[last:])
            buf = "".join(pieces)
            total += weight * min(hits, cap)
    return total


def score(text, extracted, ops):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        segs = _prose_segments(t)
        n_prose_words = sum(n for _, n in segs)
        if n_prose_words < 25:
            return 0.02  # nav chrome / link farm: no checkable prose at all
        prose = "\n".join(s for s, _ in segs)

        facts = _fact_mass(prose)
        # attributed quoted claims (named sourcing = checkable specificity)
        n_q = len(_QUOTE.findall(prose))
        n_said = len(_SAID.findall(prose))
        facts += 0.5 * min(n_q, n_said, 5)
        # explicit incompleteness: claims flagged as unverifiable/undisclosed
        n_undisc = len(re.findall(
            r"\b(?:were|was|is|are)?\s?not\s+disclosed\b|\bundisclosed\s+(?:sum|amount|terms|price)\b",
            prose, re.I))
        facts -= 0.5 * min(n_undisc, 2)
        facts = max(0.0, facts)

        density = 100.0 * facts / max(n_prose_words, 80)
        spec = 1.0 - math.exp(-density / 2.0)

        gate = min(1.0, n_prose_words / 90.0)
        bonus = 0.04 if _RELEASE_MARK.search(t) else 0.0
        code_score = gate * min(1.0, 0.06 + 0.88 * spec + bonus)

        # ---- optional thick-input blending (safe under extracted={}) ------
        fc_raw = (extracted or {}).get("fact_count", "") or ""
        mf_raw = ((extracted or {}).get("main_fact", "") or "").strip()
        m = re.search(r"\d+", fc_raw)
        if m:  # extractor ran and produced a count
            n = min(int(m.group()), 20)
            llm_spec = gate * min(1.0, 0.06 + 0.88 * (1.0 - math.exp(-n / 5.0)))
            blended = 0.6 * code_score + 0.4 * llm_spec
            if not mf_raw or mf_raw.upper() == "NONE":
                blended *= 0.55  # announces nothing specific
            return max(0.0, min(1.0, blended))
        return max(0.0, min(1.0, code_score))
    except Exception:
        return 0.5
