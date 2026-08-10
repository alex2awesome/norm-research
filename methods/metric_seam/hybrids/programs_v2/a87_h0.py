"""Hybrid metric channel for aspect a87 — "Human, humble spokesperson tone".

Criterion: spokesperson quotes should sound natural, respectful, and unboastful
rather than canned or grandiose.

Judge behaviour observed in the TRAIN pack (30 ex):
  * nav-chrome / marketing / off-topic pages with NO attributed human quote  -> ~0
  * quotes that are boastful/grandiose ("world's premier", "highest ever") or
    aggressive/attacking ("sabotage", "state terrorism")                     -> 0.0-0.2
  * canned corporate self-congratulation ("we had a better year", "our
    highest ever") from execs                                                -> ~0.4
  * natural, measured, human quotes from analysts / scientists / directors,
    especially humble ones ("humbled and honored", "we would welcome ...")   -> 0.9-1.0

Design: the criterion has NARROW applicability (~115/250 NA in adjudication).
Documents where nobody is genuinely quoted score LOW/MID rather than erratically:
a quote-less genuine release lands mid-low, quote-less nav chrome lands near 0.
When an attributed human quote exists, we start high (natural human voice) and
subtract for boastful / grandiose / aggressive / self-promotional content and add
a little for humility.  PRESENCE-IS-NOT-QUALITY: we score the *content* of the
quote (boast vs humility vs aggression), not the mere presence of quotes/keywords.
"""
import re

# ---------------------------------------------------------------- LLM fields --
# Optional thick-input grounding.  Code works fully with extracted={}.
LLM_FIELDS = {
    "quote_tone": (
        "One word for the tone of the main spokesperson/person quote: humble, "
        "natural, neutral, corporate, boastful, or aggressive; answer none if "
        "nobody is directly quoted."
    ),
    "doc_type": (
        "Answer RELEASE if this is a genuine press release or news article with "
        "real reported content, or OTHER if it is a navigation / marketing / "
        "boilerplate page with no news."
    ),
}

# ------------------------------------------------------------------ lexicons --
_ATTR = re.compile(
    r"\b(said|says|saying|stated|according to|added|adds|noted|notes|"
    r"commented|explained|explains|told|charged|remarked|announced|"
    r"asked|continued|observed|recalled|wrote|writes)\b",
    re.I,
)

_BOAST = [
    "world-class", "world class", "best-in-class", "best in class",
    "industry-leading", "industry leading", "cutting-edge", "cutting edge",
    "revolutionary", "unparalleled", "unrivaled", "unrivalled", "premier",
    "premiere", "groundbreaking", "ground-breaking", "game-changing",
    "game changer", "unprecedented", "world's leading", "world leader",
    "global leader", "market leader", "leading provider", "number one",
    "#1", "dominant", "best ever", "best quarter", "best year", "highest ever",
    "record-high", "record high", "record low", "record-low", "most profitable",
    "strongest ever", "biggest ever", "largest ever", "world's best",
    "world's premier", "best possible", "speaks volumes", "second to none",
    "the very best", "leading the", "leader in", "our highest",
    "our best", "our most", "our strongest", "our largest",
]

_HUMBLE = [
    "humbled", "humbling", "humble", "honored", "honoured", "privileged",
    "grateful", "gratitude", "fortunate", "thankful", "appreciate",
    "appreciative", "much to do", "still have much", "still much", "learn from",
    "we owe", "could not have", "couldn't have", "thanks to", "in service of",
    "listen to", "grounded", "modest",
]

_AGGR = [
    "sabotage", "disgraced", "terrorism", "assassination", "condemn",
    "thumbing", "oppression", "disaster", "impeach", "impeachment", "scandal",
    "corrupt", "outrage", "unlawful", "meek response", "conceal", "sham",
    "betray", "reckless", "disgrace", "green light", "state terror",
    "extrajudicial", "sinister", "shameful", "hostile",
]

# self-congratulatory corporate cues (weighted lighter than outright boast)
_SELFPROMO_ACHV = [
    "proud", "pleased", "delighted", "excited", "thrilled", "success",
    "successful", "strong", "record", "milestone", "excellence", "leading",
    "well-positioned", "well positioned", "delivered", "profit", "growth",
    "outperform", "momentum", "robust",
]

_PR_MARKERS = [
    "prnewswire", "pr newswire", "news provided by", "for immediate release",
    "press release", "news release", "media contact", "globenewswire",
    "businesswire", "business wire", "/prweb", "source ",
]

_FP_SINGULAR = re.compile(r"\bI\b")          # case-sensitive first-person singular
_FP_SING_LC = re.compile(r"\b(my|me|myself)\b", re.I)
_FP_ANY = re.compile(r"\b(we|our|us|i|my|me)\b", re.I)


def _clamp(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _count(hay, needles):
    return sum(hay.count(n) for n in needles)


def _quotes(text):
    """Quote spans by pairing SEQUENTIAL quote marks (open/close), which is
    robust to short in-line quotes (nicknames, scare quotes) that break naive
    `"..."` regex pairing.  Keeps spans of plausible sentence length."""
    marks = [i for i, c in enumerate(text) if c == '"']
    out = []
    for k in range(0, len(marks) - 1, 2):
        s, e = marks[k], marks[k + 1]
        content = text[s + 1:e]
        if 12 <= len(content) <= 700:
            out.append((s, e + 1, content))
    return out


def _genuine_quotes(text, spans):
    """Keep spans that read as attributed human SPEECH (a real spokesperson
    sentence), not slogans / scare-quoted fragments / nav."""
    out = []
    for s, e, q in spans:
        if len(re.findall(r"[A-Za-z']+", q)) < 5:      # sentence-length only
            continue
        letters = [c for c in q if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            continue                                    # drop all-caps banners
        ctx = text[max(0, s - 90):s] + text[e:e + 90]
        if _ATTR.search(ctx) or _FP_ANY.search(q):
            out.append(q)
    return out


def _release_likeness(text):
    """0..1 crude estimate that the doc is a real release/news article."""
    t = text.lower()
    sc = 0.0
    for mk in _PR_MARKERS:
        if mk in t:
            sc += 0.28
            break
    # dateline:  CITY, Month Day, Year   /   CITY, State, Month Day
    if re.search(r"[A-Z][A-Za-z.]+,\s+[A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4}", text):
        sc += 0.22
    if re.search(r"\b(announce[ds]?|announcing|unveil|report(?:ed|s)?)\b", t):
        sc += 0.12
    if _ATTR.search(text):
        sc += 0.16
    return _clamp(sc)


def score(text, extracted, ops):
    try:
        text = ops.normalize(text)
    except Exception:
        pass

    spans = _quotes(text)
    quotes = _genuine_quotes(text, spans)
    rel = _release_likeness(text)

    # ---- code-only channel -------------------------------------------------
    if not quotes:
        # criterion barely applies: quote-less doc.  Legit release -> mid-low,
        # nav-chrome junk -> near 0.  Personal first-person voice (a blog) lifts.
        fp_sing = len(_FP_SINGULAR.findall(text)) + len(_FP_SING_LC.findall(text))
        fp_bonus = min(0.30, 0.035 * fp_sing) if fp_sing >= 3 else 0.0
        code = 0.10 + 0.34 * rel + fp_bonus
    else:
        q = " ".join(quotes).lower()
        words = re.findall(r"[a-z']+", q)
        n = max(1, len(words))

        b = _count(q, _BOAST)
        h = _count(q, _HUMBLE)
        a = _count(q, _AGGR)
        sp = _count(q, _SELFPROMO_ACHV)
        excl = q.count("!")
        # density-aware boast: superlatives concentrated in short quotes hurt more
        boast_dens = b / (n / 40.0 + 1.0)

        has_i = bool(re.search(r"\bI\b", " ".join(quotes)) or
                     re.search(r"\bmy\b", q))
        has_we = bool(re.search(r"\b(we|our|us)\b", q))
        # canned corporate courtesy: "we're pleased/honored/proud to <verb>"
        # (a formulaic release cliche, distinct from a personal humble quote)
        courtesy = bool(re.search(
            r"\b(honored|honoured|pleased|proud|delighted|thrilled|excited)"
            r"\s+to\b", q)) and has_we and not has_i

        base = 0.80
        # humility lifts -- but not when the humble word is just a courtesy cliche
        if not courtesy:
            base += min(0.16, 0.06 * h)
        if has_i:
            base += 0.04                                 # personal 1st-person voice

        # grandiosity + canned self-promo + courtesy share ONE capped demotion so
        # overlapping cues ("proud"/"leading") don't triple-count. Non-aggressive
        # corporate quotes floor around ~0.25 (the judge's mid tier), not zero.
        demote = 0.20 * b + 0.10 * boast_dens + 0.06 * sp + (0.12 if courtesy else 0)
        base -= min(0.55, demote)
        base -= min(0.15, 0.05 * excl)                   # hype punctuation
        # aggression is separate and CAN sink a hostile quote toward zero
        base -= min(0.72, 0.34 * a)
        # very sparse "quote" (single short attributed fragment) -> less credit
        if n < 8 and len(quotes) == 1:
            base -= 0.10
        code = base

    code = _clamp(code)

    # ---- fold in LLM tone/type when available ------------------------------
    tone = str(extracted.get("quote_tone", "") or "").strip().lower()
    dtype = str(extracted.get("doc_type", "") or "").strip().lower()

    tone_map = {
        "humble": 1.00, "natural": 0.86, "neutral": 0.62,
        "corporate": 0.42, "canned": 0.40, "formulaic": 0.42,
        "boastful": 0.16, "grandiose": 0.14, "aggressive": 0.05,
    }
    tone_key = next((k for k in tone_map if k in tone), None)

    if tone_key is not None:
        # blend: LLM tone leads, code moderates (injection-resistant backstop)
        return _clamp(0.70 * tone_map[tone_key] + 0.30 * code)

    if tone in ("none", "no", "n/a", "na", "") and dtype:
        if "other" in dtype:
            return _clamp(min(code, 0.15))
        if "release" in dtype and not quotes:
            return _clamp(max(code, 0.30))

    return code
