"""a75 hybrid — Headline/topline clarity and accuracy.

Construct: does the document foreground a clear, ACCURATE, SPECIFIC news headline
(actor + news action + concrete numbers/dates), with tone matched to the substance?

Design (decoupling presence from quality — injection-gate aware):
  * H (headline quality): scan the top of the doc for news-action verbs; each hit is
    scored by the SPECIFICITY of its local window (%, $, dates, proper-noun pairs) and
    penalized for promotional/feature-style tone. A verb with no specifics earns little.
  * B (body news evidence): attributed quotes, quantified facts; wire/ticker markers get
    only a small weight so injecting boilerplate cannot flip a score.
  * Damping: non-English text, second-person marketing copy ("you/your" density — releases
    are third-person), and nav-chrome line density (gentle: real releases are often
    embedded in scraped chrome).
  * LLM fields (optional thick input): verbatim headline + its concrete news fact; the
    PREDICATE (verb/specificity/tone) stays in code, fields only supply cleaner input.
"""
import re

LLM_FIELDS = {
    "headline": "Quote verbatim the document's main news headline or title (max 20 words); answer NONE if the page has no news headline.",
    "news_specifics": "In <=20 words state the headline's concrete news fact (actor + action + number/date); answer NONE if there is none.",
}

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
          r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")

# Finite news-action verbs (avoid bare infinitives/nouns that riddle marketing chrome;
# no "reports"/"sign(s)" alone — those are nav-chrome nouns like "Reports", "Sign In").
_STRONG = re.compile(
    r"\b(?:announc(?:es|ed|ing)|completes|completed|acquir(?:es|ed)|acquisition\s+of|"
    r"signs?\s+(?:an?\s+)?(?:agreement|deal|contract|pact|accord)|signed|"
    r"joins|joined|to\s+join|appoints|appointed|appointment\s+of|"
    r"names|named|launch(?:es|ed)|unveil(?:s|ed)?|begins|began|to\s+begin|"
    r"purchas(?:es|ed)|wins|won|awarded|reported|declin(?:es|ed)|fell|falls|"
    r"dropped|rose|rises?|grew|grows|surpass(?:es|ed)|total(?:s|led|ed)|reach(?:es|ed)|"
    r"agrees|agreed|fil(?:es|ed)|su(?:es|ed)|expand(?:s|ed)|introduc(?:es|ed)|"
    r"elect(?:s|ed)|resigns?|retires)\b", re.I)
_PRONOUNS = {"They", "We", "He", "She", "It", "You", "I", "Who", "This", "That"}
# A long hyphenated URL slug is the page's own headline (news CMS artifact).
_SLUG = re.compile(r"\b(?:[a-z0-9]{2,}-){6,}[a-z0-9]{2,}\b")
# "X Says/Said ..." only with a capitalized subject right before (kills "surveyed said").
_CAP_SAYS = re.compile(r"\b[A-Z][\w&.'-]+\s+(?:Says|says|Said|said)\b")
# Weak, hedged/feature-style claims (mid-tier tell).
_WEAK = re.compile(r"\b(?:will|would|could|may)\s+[A-Za-z]+", re.I)

_PCT = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent|per cent)", re.I)
_MONEY = re.compile(r"\$\s?\d")
_DATEFULL = re.compile(_MONTH + r"\.?\s+\d{1,2}|\b\d{1,2}\s+" + _MONTH, re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_ANYDIG = re.compile(r"\d")
_PROPER = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}")
_PROMO = re.compile(
    r"\b(?:world'?s\s+(?:first|leading|largest|best)|premier|best-in-class|award-winning|"
    r"cutting-edge|revolutionary|unparalleled|state-of-the-art|industry-leading|"
    r"most compelling|hot streak|must-have)\b", re.I)
_SCAREQ = re.compile(r"'[A-Z][^']{2,28}'")
_FEATURE_START = re.compile(r"^\s*(?:How|Why|What|Top\s+\d+|The\s+End\s+of)\b", re.I)

_QUOTE_SAID = re.compile(r'"[^"]{15,400}"\s*,?\s*(?:said|says)\s')
_SAID_NAME = re.compile(r"\b(?:said|says)\s+[A-Z][a-z]")
_NAME_SAID = re.compile(r"\b[A-Z][\w.'-]+\s+(?:said|says)[\s.,:]")
_ACC_TO = re.compile(r"\baccording to\s+[A-Z]")
_WIRE = re.compile(r"PRNewswire|For Immediate Release|News provided by|Business Wire|"
                   r"GlobeNewswire|\bSOURCE\s+[A-Z][A-Za-z]", re.I)
_TICKER = re.compile(r"\((?:NYSE|NASDAQ|Nasdaq|TSX|LSE|OTC)\s*:")
_STOPS = re.compile(r"\b(?:the|of|and|to|is|that|with|for|this|are|was|has|have|will|"
                    r"from|it|its|be|by|at|or|as|on)\b", re.I)
_YOU = re.compile(r"\b(?:you|your|yours|yourself)\b", re.I)


def _spec(s):
    """Specificity of a headline window: quantities/dates >> bare digits."""
    v = 0.0
    if _PCT.search(s):
        v += 0.45
    if _MONEY.search(s):
        v += 0.45
    if _DATEFULL.search(s):
        v += 0.35
    elif _YEAR.search(s):
        v += 0.20
    elif _ANYDIG.search(s):
        v += 0.12
    return min(1.0, v)


def _tone_pen(s):
    pen = min(0.16, 0.08 * len(_PROMO.findall(s)))
    if "?" in s:
        pen += 0.06
    if "!" in s:
        pen += 0.06
    if _SCAREQ.search(s):
        pen += 0.06
    return pen


def _win_score(w, strong):
    sp = _spec(w)
    pr = 1.0 if _PROPER.search(w) else 0.0
    if strong:
        h = (0.30 + 0.10 * pr) if sp == 0 else (0.38 + 0.52 * sp + 0.10 * pr)
    else:
        h = min(0.50, 0.18 + 0.25 * sp + 0.05 * pr)
    return max(0.0, h - _tone_pen(w))


def _headline_H(t):
    """Best news-headline evidence in the top of the doc."""
    zone = t[:2600]
    best = 0.0
    for kind, mm in ([("v", m) for m in _STRONG.finditer(zone)]
                     + [("s", m) for m in _CAP_SAYS.finditer(zone)]):
        w = zone[max(0, mm.start() - 100):mm.end() + 100]
        # A headline verb needs a named actor: capitalized non-pronoun subject just
        # before it, else it's body/blog prose -> weak credit only.
        pre = zone[max(0, mm.start() - 30):mm.start()]
        caps = re.findall(r"[A-Z][\w&.'-]*", pre)
        named = any(c.rstrip(".,'&-") not in _PRONOUNS for c in caps)
        s = _win_score(w, named)
        if kind == "s" and _spec(w) == 0:
            s = min(s, 0.36)   # spec-less "X said" = reaction quote, not a topline
        best = max(best, s)
    if best < 0.35:
        for m in _WEAK.finditer(zone):
            # asymmetric: hedged claims own their preceding context; forward text is
            # often unrelated chrome/bio dates
            w = zone[max(0, m.start() - 90):m.end() + 55]
            best = max(best, _win_score(w, False))
    if _SLUG.search(zone):
        best = max(best, 0.50)   # news-CMS URL slug = the page's own headline
    return best


def _headline_quality(ht):
    """Same predicate applied to an LLM-extracted headline string."""
    n = len(ht.split())
    if n < 3:
        return 0.0
    if _STRONG.search(ht) or _CAP_SAYS.search(ht):
        base = 0.45
    elif _WEAK.search(ht):
        base = 0.26
    else:
        base = 0.14
    h = base + 0.45 * _spec(ht) + 0.10 * (1.0 if _PROPER.search(ht) else 0.0)
    if not (4 <= n <= 24):
        h -= 0.12
    if _FEATURE_START.match(ht):
        h -= 0.12
    return max(0.0, min(1.0, h - _tone_pen(ht)))


def _body_B(t):
    b = 0.0
    if (_QUOTE_SAID.search(t) or _SAID_NAME.search(t) or _NAME_SAID.search(t)
            or _ACC_TO.search(t)):
        b += 0.45
    if _PCT.search(t) or _MONEY.search(t):
        b += 0.30
    if _WIRE.search(t):
        b += 0.15   # weak on purpose: injectable boilerplate
    if _TICKER.search(t):
        b += 0.10
    return min(1.0, b)


def score(text, extracted, ops):
    try:
        t = ops.normalize(text or "")
        words = re.findall(r"[A-Za-z']+", t)
        nw = len(words)
        if nw < 25:
            return 0.0

        H = _headline_H(t)
        B = _body_B(t)

        ex = extracted or {}
        ht = (ex.get("headline") or "").strip()
        if ht and ht.upper() != "NONE":
            H = max(H, _headline_quality(ht))
        elif "headline" in ex:          # extractor looked and found no headline
            H *= 0.5
        ns = (ex.get("news_specifics") or "").strip()
        if ns and ns.upper() != "NONE":
            if _ANYDIG.search(ns) or _DATEFULL.search(ns):
                B = min(1.0, B + 0.20)  # concrete fact confirmed
        elif "news_specifics" in ex:
            B *= 0.75

        raw = 0.78 * H + 0.22 * B

        # --- damping: non-release / non-English / marketing-voice tells ---
        en = len(_STOPS.findall(t)) / nw
        f_en = max(0.10, min(1.0, en / 0.12))
        yr = len(_YOU.findall(t)) / nw
        f_you = 1.0 if yr <= 0.008 else max(0.35, 1.0 - (yr - 0.008) * 30.0)
        if B >= 0.70:
            # strong attributed-quote + quantified-fact evidence => "you" is site
            # chrome around a real news story, not marketing voice
            f_you = max(f_you, 0.75)
        lines = [ln for ln in t[:2600].splitlines() if ln.strip()][:50]
        navd = (sum(1 for ln in lines if len(ln.split()) <= 2) / len(lines)) if lines else 0.0
        f_nav = 1.0 - 0.28 * max(0.0, navd - 0.35)

        return max(0.0, min(1.0, raw * f_en * f_you * f_nav))
    except Exception:
        return 0.5
