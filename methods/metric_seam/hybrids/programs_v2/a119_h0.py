"""a119 Originality and substantive new information — hybrid channel.

Construct (from judge behavior on train): high scores go to genuine press
releases that deliver dense, SPECIFIC, previously-unknown facts (results with
figures, transactions, first-ever events); ~0 goes to non-releases (nav chrome,
blogs, articles, leaflets); low-but-nonzero goes to "thin" releases: award/
recognition announcements, reaction/opinion statements, routine periodic
disclosures, and procedural notices — present-tense release FORM without new
substance. So: score = release-likeness x fact-specificity, damped by
thin-content penalties and by near-duplicate mass in the corpus (evidence op).
Presence-of-novelty keywords ("new", "launch") is deliberately NOT the driver;
specificity (numbers, attributed quotes, dateline structure) is.
"""
import re

LLM_FIELDS = {
    "new_fact": "In <=20 words, the single most important NEW fact/event/figure this document announces publicly; answer NONE if it announces nothing new.",
    "doc_kind": "One word: 'release' if this is a press release or company announcement, 'article' if news/blog/commentary/aggregator page, else 'other'.",
}

_MON = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)")

_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"}

# CITY[, State/Country], Month D  (dateline); city group captured for weekday check
_DATELINE = re.compile(
    r"\b([A-Z][A-Z .'’-]{2,30}|[A-Z][a-z]+(?:\s[A-Z][a-zA-Z.]+)?)\s*,\s*"
    r"(?:(?:[A-Z][a-z]{1,12}\.?|[A-Z]{2}|D\.C\.)\s*,?\s*)?"
    + _MON + r"\.?\s+\d{1,2}\b")

_FULL_DATE = re.compile(
    r"\b(?:" + _MON + r"\.?\s+\d{1,2}\s*,?\s+\d{4}|\d{1,2}\s+" + _MON +
    r"[a-z]*\.?\s+\d{4}|\d{1,2}-" + _MON + r"-\d{4})\b", re.I)

_WIRE = ("prnewswire", "business wire", "businesswire", "globe newswire",
         "globenewswire", "marketwired", "marketwire", "accesswire",
         "rns number", "news provided by", "for immediate release")

_TICKER = re.compile(
    r"\((?:NYSE|NASDAQ|AMEX|TSX|LSE|ASX|Euronext)\s*(?:MKT|American)?\s*[:.]?\s*"
    r"[A-Z][A-Z.]{0,6}\s*\)", re.I)

_ANNOUNCE_STRONG = re.compile(
    r"\b(?:today\s+(?:announc|report|releas|unveil|introduc|launch)\w*|"
    r"(?:announc|report|releas|unveil|introduc|launch)\w{1,4}\s+today)\b", re.I)

_RESULTS_HEADLINE = re.compile(
    r"\b(?:announces?|reports?|posts?|delivers?)\s+(?:its\s+)?(?:record\s+)?"
    r"(?:full[- ]year|fourth|first|second|third|q[1-4]|\d{4}|quarterly|annual|"
    r"holiday|financial|preliminary)?[-\s]*(?:quarter)?[-\s]*"
    r"(?:results?|earnings|profit|sales|revenue|loss)\b", re.I)

_ANNOUNCE_WEAK = re.compile(
    r"\b(?:announces?|announced|unveil(?:s|ed)?|launch(?:es|ed)?|"
    r"introduc(?:es|ed)|to\s+acquire|acquires|to\s+nominate|nominates|"
    r"signs\s+agreement|will\s+begin|declares)\b", re.I)

_REPORT_RELEASE = re.compile(
    r"\bnew report\b|\breport\s+(?:documents|finds|shows|reveals|details|"
    r"estimates|concludes)\b|\bannual\s+[\w\s'-]{0,40}\breport\b", re.I)

_MONEY = re.compile(
    r"[$£€]\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:billion|million|trillion|bn|mn))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:billion|million|trillion)\b", re.I)

_PCT = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|per\s+cent)", re.I)

_QUOTE_ATTR = re.compile(
    r"\"[^\"]{25,600}\"\s*,?\s*(?:said|says|stated|noted|added)\s+[A-Z]")
_SAID_NAME = re.compile(
    r"\b(?:said|says|stated)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+")

# thin-content penalty lexicons (release form without substantive new info)
_SOFT_AWARD = ("named as one of", "named one of", "most ethical companies",
               "best places to work", "best workplaces", "award", "honored",
               "honoured", "recognized as", "recognised as", "recognition",
               "anniversary", "celebrates", "prestigious")
_REACTION = ("condemn", "denounce", "calls on", "calls upon", "urges",
             "in a statement", "in response to", "expressed concern",
             "welcomed the decision", "reiterates")
_ROUTINE = ("portfolio composition", "net asset value", "distribution rate",
            "monthly distribution", "declaration of dividend",
            "portfolio holdings and weightings", "as of month-end")
_PROCEDURAL = ("notice is hereby given", "sunshine act", "open meeting will",
               "matters to be considered", "hereby notifies")
_ARTICLE = ("this article was written", "editor's note", "seeking alpha",
            "follower s", "followers", "read more...", "login to comment",
            "written by", "comments (", "comment s (", "related articles",
            "leave a comment", "add a comment", "views expressed in this")
_IR_CHROME = ("click here for webcast", "click here for transcript",
              "quarterly results archives", "eps estimates",
              "earnings history", "stock chart", "stock calculator",
              "analyst coverage", "historical price look up")

_STOP = frozenset("the of and to a in that for is on with as by at from it its "
                  "was were are be this has have will an or we our their".split())


def _clip(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def _count(pat, s, cap):
    n = 0
    for _ in pat.finditer(s):
        n += 1
        if n >= cap:
            break
    return n


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
    except Exception:
        t = text or ""
    if not t.strip():
        return 0.0
    tl = t.lower()
    head = t[:800]

    # ---- release-likeness (structural form) ----
    wire = any(w in tl for w in _WIRE)
    press_lit = "press release" in tl
    dm = None
    for m in _DATELINE.finditer(t[:3000]):
        city_words = {w.lower().strip(".,") for w in m.group(1).split()}
        if not (city_words & _WEEKDAYS):
            dm = m
            break
    ticker = bool(_TICKER.search(t))
    ann_strong = bool(_ANNOUNCE_STRONG.search(t))
    results_hl = bool(_RESULTS_HEADLINE.search(t))
    ann_weak = bool(_ANNOUNCE_WEAK.search(t))
    report_rel = bool(_REPORT_RELEASE.search(t))
    early_date = bool(_FULL_DATE.search(head))

    rel = _clip(0.45 * wire + 0.35 * bool(dm) + 0.25 * ticker
                + 0.15 * press_lit + 0.30 * ann_strong + 0.25 * results_hl
                + 0.20 * report_rel + 0.10 * ann_weak + 0.20 * early_date)

    # ---- fact specificity (substance, not keyword presence) ----
    money = _count(_MONEY, t, 8)
    pct = _count(_PCT, t, 10)
    quotes = min(2, _count(_QUOTE_ATTR, t, 3) + _count(_SAID_NAME, t, 3))
    words = re.findall(r"[A-Za-z']+", tl)
    nw = len(words) or 1
    stop_frac = sum(1 for w in words if w in _STOP) / nw
    prose = _clip((stop_frac - 0.10) / 0.14, 0.25, 1.0)
    digit_toks = len(re.findall(r"\b\d[\d,.%]*\b", t))
    numdens = _clip(digit_toks / (nw / 100.0) / 8.0) * prose

    sub_raw = _clip(0.11 * money + 0.05 * pct + 0.18 * quotes + 0.25 * numdens)
    # substance only counts as NEW information inside release form
    sub = sub_raw * (0.35 + 0.65 * rel)

    s = _clip(0.06 + 0.50 * rel + 0.52 * sub)

    # ---- thin-content damping (form without new substance) ----
    def hits(lex):
        return sum(1 for k in lex if k in tl)

    # article/aggregator chrome only demotes pages that LOOK release-like;
    # at rel~0 the doc is already near-floor and this would sink true mids
    # below plain nav-chrome zeros.
    art_hits = hits(_ARTICLE)
    if rel > 0.25:
        if art_hits >= 2:
            s *= 0.30
        elif art_hits == 1:
            s *= 0.65
    if art_hits >= 2:
        s = min(s, 0.16)  # platform-article chrome caps originality low
    if hits(_SOFT_AWARD) >= 2 and money <= 2:
        s *= 0.32
    if hits(_REACTION) >= 2 and money == 0:
        s *= 0.55
    if hits(_ROUTINE) >= 2:
        s *= 0.45
    if hits(_PROCEDURAL) >= 1:
        s *= 0.50
    if hits(_IR_CHROME) >= 2:
        s *= 0.45
    s *= prose  # nav-chrome damping

    # ---- evidence op: near-duplicate mass argues AGAINST originality ----
    try:
        sims = ops.retrieve_similar(t, k=6) or []
        others = [sv for sv, _id in sims if sv < 0.985]  # drop self-match
        if others:
            top = max(others)
            if top > 0.70:  # heavy template/near-dup mass only
                s *= 1.0 - 0.45 * _clip((top - 0.70) / 0.30)
            dup_mass = sum(1 for sv in others if sv > 0.60)
            if dup_mass >= 3:
                s *= 0.80
    except Exception:
        pass

    # ---- LLM thick-input grounding (optional; code-only path unaffected) ----
    nf = (extracted or {}).get("new_fact", "").strip()
    dk = (extracted or {}).get("doc_kind", "").strip().lower()
    if dk:
        if dk.startswith("release"):
            s = max(s, _clip(0.25 + 0.75 * sub_raw))
        elif dk.startswith(("article", "other")):
            s *= 0.35
    if nf:
        if nf.upper() == "NONE":
            s *= 0.40
        elif any(ch.isdigit() for ch in nf):
            s = max(s, min(0.55, s + 0.15))

    return _clip(s)
