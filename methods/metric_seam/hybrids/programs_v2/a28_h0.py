"""a28_h0 — Accuracy, evidence, and independent verifiability (hybrid channel).

Judge behavior reverse-engineered from the pack:
  * non-releases (nav chrome, blogs, leaflets, news aggregators, error pages) -> ~0
  * promotional / advocacy / one-sided releases with thin checkable evidence -> ~0.25-0.3
  * data-dense or official releases (figures, tables, legal/regulatory citations,
    primary-data links, named contacts) -> 0.9-1.0

Design: release-likeness gate  x  evidence density  x  hype/advocacy damp.
PRESENCE != QUALITY: evidence is measured as *density of hard, checkable content*
(numbers, full dates, resolvable URLs, legal cites, reachable contacts), never as
bare keyword hits; hype is measured relative to prose volume and only damps.
Works code-only (extracted={}); LLM fields nudge the gate / evidence when present.
"""
import math
import re

LLM_FIELDS = {
    "doc_type": ("Answer RELEASE if this page is a press release or official "
                 "announcement; answer OTHER if it is navigation, blog, news "
                 "article, ad, or error page."),
    "primary_source": ("Name the most authoritative independently checkable source "
                       "cited for the central claims (dataset, filing, law, report, "
                       "URL); answer NONE if none."),
}

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
          r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
          r"Dec(?:ember)?)")

# ---- release-likeness -------------------------------------------------------
_WIRE_STRONG = re.compile(
    r"/\s*PR\s?Newswire\s*/|\(BUSINESS WIRE\)|\(GLOBE NEWSWIRE\)|News provided by|"
    r"FOR IMMEDIATE RELEASE|RNS Number|/\s*CNW\s*/|PRESS RELEASE\s*\d|"
    r"\bSOURCE\s+[A-Z][A-Za-z]|\n\s*#\s?#\s?#\s*\n")
_WIRE_WEAK = re.compile(
    r"\b(?:PR\s?Newswire|Globe\s?Newswire|Business\s?Wire|Press Releases?)\b",
    re.I)
_BYLINE = re.compile(r"\bBy\s+[A-Z][a-z]+\s+[A-Z]")
_DATELINE = re.compile(
    r"[A-Z][A-Za-z.\- ]{2,25},\s*(?:[A-Z][a-zA-Z.]{1,15}\s*,?\s+)?" + _MONTH +
    r"\.?\s+\d{1,2},?\s+\d{4}")
_EURODATE_HDR = re.compile(r"\b\d{1,2}\s+" + _MONTH + r"[A-Za-z]*\.?\s+\d{4}\b|"
                           r"\b\d{1,2}-" + _MONTH + r"-\d{4}\b", re.I)
_ANNOUNCE = re.compile(
    r"\btoday[^.\n]{0,120}\b(?:announc|report|launch|fil(?:e|ed|ing)|issu|propos|"
    r"releas|state|unveil|introduc|complet)|"
    r"\b(?:announc|report|launch|fil|issu|propos|unveil|stat)[a-z]*[^.\n]{0,120}\btoday\b",
    re.I)
_OFFICIAL = re.compile(
    r"Notice is hereby given|pursuant to (?:the )?(?:provisions|Rule|Section|Pub)|"
    r"Panel (?:found|ordered)|violated Rules?\s*\d|PENALTY\s*:|"
    r"fine[^.\n]{0,40}in the amount of|"
    r"(?:United States|U\.?S\.?)\s+Attorney[^\n]{0,100}\bannounc|"
    r"(?:Commission|Department|Bureau|Agency)[^.\n]{0,80}"
    r"\b(?:announced|proposed|adopted|charged|ordered|issued)\b", re.I)
_ERROR_PAGE = re.compile(
    r"page (?:that )?(?:doesn't|does not|no longer) exist|isn't available|"
    r"Error 404|Page Not Found|link you requested", re.I)

# ---- evidence ---------------------------------------------------------------
_PHONE = re.compile(r"(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)\s?|\b\d{3}[-.]\s?)\d{3}[-.]\d{4}\b")
_HARDNUM = re.compile(
    r"[$£€]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion|bn)\b)?|"
    r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|per cent\b|million\b|billion\b|trillion\b)|"
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|"
    r"\b\d+\.\d+\b", re.I)
_URL = re.compile(r"(?:https?://|www\.)[^\s\"'<>)\]]+", re.I)
_URL_BOILER = re.compile(
    r"facebook|twitter|linkedin|youtube|instagram|pinterest|prnewswire\.com|cision|"
    r"globenewswire\.com|businesswire\.com|addthis|plus\.google", re.I)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\[email.{0,3}protected\]")
_CITE_PATS = [
    r"\baccording to\b",
    r"\b(?:data|figures|results|statistics|findings) (?:from|provided by|released by|compiled by)\b",
    r"\bpreliminary (?:results|data)\b",
    r"\b(?:available|accessible) (?:at|on|from)\b",
    r"\bfor more information,? (?:visit|see|contact)\b",
    r"\bfiled? (?:with|in|by)\b",
    r"\bForm\s+(?:10-[KQ]|8-K|S-1)\b",
    r"\bAnnual Report\b",
    r"\bPub\.?\s?L(?:aw)?\.?\s?\d",
    r"\b\d+\s+FR\s+\d+\b",
    r"\bRelease No\b",
    r"\bRules?\s+\d",
    r"\bU\.S\.C\.",
    r"\bFile No\b",
    r"\bin accordance with\b",
    r"\bFederal Register\b",
    r"\bFact Sheet\b",
    r"RELEASE\s+\d{4}-\d+",

    r"\b(?:NYSE|NASDAQ|LSE)\s?:\s?[A-Z]{1,6}\b",
]
_CITES = [re.compile(p, re.I) for p in _CITE_PATS]
_CONTACT_CTX = re.compile(r"contact|inquir|information|relations|media|press|tel|phone|direct", re.I)
_CONTACT_BOILER = re.compile(r"Cision|Helpline|customer support|Toll Free", re.I)
_FULLDATE = re.compile(_MONTH + r"\.?\s+\d{1,2},?\s+\d{4}|\b\d{1,2}\s+" + _MONTH +
                       r"[A-Za-z]*\.?\s+\d{4}", re.I)
_MONTHYEAR = re.compile(_MONTH + r"[A-Za-z]*\.?\s+\d{4}", re.I)

# ---- hype / advocacy damp ---------------------------------------------------
_HYPE = ["world's leading", "world's largest", "world's premier", "world-class",
         "industry-leading", "best-in-class", "iconic", "legendary", "revolutionary",
         "ground-breaking", "groundbreaking", "first of its kind", "first-of-its-kind",
         "cutting-edge", "state-of-the-art", "unparalleled", "illustrious",
         "incredible", "amazing", "thrilled", "delighted", "memorable",
         "barrier breaking", "immersive"]
_ADVOCACY = ["whopping", "disgraced", "outrageous", "sabotage", "misguided", "meek",
             "plague", "nickel-and-dime", "nickel and dime", "condemn", "condemned",
             "condemnation", "sounded the alarm", "thumbing", "oppression",
             "open-borders", "witch hunt", "many believe", "experts say",
             "studies show", "everyone knows", "crisis", "demands that",
             "disaster", "reckless", "radical", "shameful", "outraged"]
_OPINION = ["we believe", "in our view", "we are pleased", "we are excited",
            "we are humbled", "we are proud", "vigorously defend", "no merit",
            "we think", "we estimate"]
_STOP = {"the", "and", "of", "to", "in", "for", "on", "with", "that", "is", "are",
         "was", "be", "by", "at", "from", "as", "it", "this", "will", "has", "have"}


def _sat(x, half):
    return 1.0 - math.exp(-x / max(half, 1e-9))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        low = t.lower()

        # ---------- prose view (>=8-word lines; keeps tables, drops nav chrome)
        lines = t.split("\n")
        prose_lines = [ln for ln in lines if len(ln.split()) >= 8]
        prose = "\n".join(prose_lines)
        pwords = prose.split()
        pw = max(len(pwords), 1)
        stop_frac = sum(1 for w in pwords if w.lower().strip(".,;:'\"()") in _STOP) / pw

        # ---------- release-likeness gate R
        r = 0.08
        wire = bool(_WIRE_STRONG.search(t))
        if wire:
            r += 0.55
        elif _WIRE_WEAK.search(t):
            wire = True
            r += 0.18
        dateline = bool(_DATELINE.search(t) or _EURODATE_HDR.search(t[:1500]))
        if dateline:
            r += 0.35
        if wire and dateline:
            r = max(r, 0.85)
        if _ANNOUNCE.search(t):
            r += 0.30
        if _BYLINE.search(t[:2500]) and _FULLDATE.search(t[:2500]):
            r += 0.15                              # dated authored commentary
        official = bool(_OFFICIAL.search(t))
        if official:
            r = max(r, 0.95)
        r = min(r, 1.0)
        if _ERROR_PAGE.search(t):
            r *= 0.25
        if stop_frac < 0.10:                       # non-English / chrome blob
            r *= max(0.15, stop_frac / 0.10)
        n_sent, _, _ = ops.sent_stats(t)
        if n_sent < 5:                             # near-empty prose
            r *= 0.6

        # ---------- evidence density E (hard, checkable content only)
        pn = _PHONE.sub(" ", prose)                # don't count phones as data
        s_num = _sat(100.0 * len(_HARDNUM.findall(pn)) / pw, 2.0)

        urls = [u for u in _URL.findall(t) if not _URL_BOILER.search(u)]
        s_url = min(1.0, len(urls) / 3.0)

        cite_n = sum(1 for p in _CITES if p.search(t))
        s_cite = min(1.0, cite_n / 4.0)

        # dates: prefer ops.extract_dates, keep only fully specified ones
        full_d = len(set(_FULLDATE.findall(prose + " " + t[:1500])))
        try:
            od = ops.extract_dates(t[:4000])
            full_d = max(full_d, sum(1 for d in od if re.search(r"\d{4}", d)
                                     and re.search(r"[A-Za-z]", d)))
        except Exception:
            pass
        my_d = len(_MONTHYEAR.findall(prose))
        s_date = min(1.0, (full_d + 0.5 * min(my_d, 2)) / 3.0)

        phone_ok = any(not _CONTACT_BOILER.search(t[max(0, m.start() - 80):m.end() + 80])
                       and _CONTACT_CTX.search(t[max(0, m.start() - 150):m.end() + 150])
                       for m in _PHONE.finditer(t))
        email_ok = bool(_EMAIL.search(t))
        s_contact = 0.6 * phone_ok + 0.4 * email_ok

        e = (0.40 * s_num + 0.20 * s_url + 0.20 * s_cite +
             0.12 * s_date + 0.08 * s_contact)
        if official:
            e += 0.28                              # legal/regulatory register
        e = min(e, 1.0)

        # ---------- hype / advocacy damp (relative to prose volume)
        hyp = sum(low.count(w) for w in _HYPE)
        adv = sum(low.count(w) for w in _ADVOCACY)
        opi = sum(low.count(w) for w in _OPINION)
        if "?" in t[:300]:                          # rhetorical headline
            adv += 1
        dens = 100.0 * (hyp + adv + 0.5 * opi) / pw
        damp = max(0.5, 1.0 / (1.0 + 0.45 * max(0.0, dens - 0.15)))

        # ---------- optional LLM-extracted nudges (code-only must still work)
        ex = extracted or {}
        dt = str(ex.get("doc_type", "")).strip().upper()
        if dt:
            if "RELEASE" in dt:
                r = max(r, 0.85)
            elif "OTHER" in dt:
                r *= 0.4
        ps = str(ex.get("primary_source", "")).strip()
        if ps:
            if re.fullmatch(r"none\.?", ps, re.I):
                e *= 0.8
            else:
                e = min(1.0, e + 0.12)

        return max(0.0, min(1.0, r * (0.12 + 0.88 * e) * damp))
    except Exception:
        return 0.5
