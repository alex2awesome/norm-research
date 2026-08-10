"""a81 "Boilerplate and background with links" — hybrid channel (h0).

Judge pattern (from train pack): high scores require REAL release boilerplate
done well — an "About <Org>" prose block / identity-background facts
(founded, headquartered, ticker, "a leading X company"), explicit invitation
links ("for more information, visit <url>"), raw official URLs, and a
media/IR contact block near the END — on a document that actually reads as a
press release. Nav-chrome keyword hits ("About Us", footer "Investor
Relations") on scraped NON-releases score ~0, and even genuine releases
(pol/gov) with no about+links boilerplate score ~0.

Design: PRESENCE != QUALITY. Every component is content-checked (prose
markers after "About X", invitation context around links, contacts only in
the tail region), and the sum is multiplied by a release-likeness damp so
chrome keywords on non-releases cannot buy score.
"""
import re

LLM_FIELDS = {
    "about_boilerplate": "Quote the sentence (usually near the end) describing the issuing org's identity/background (an 'About X' boilerplate); NONE if absent.",
    "official_links": "List up to 3 URLs or domains pointing to the org's official site, newsroom, investor page, or contact/helpline; NONE if absent.",
}

# ---------- component regexes ----------
_ABOUT_RE = re.compile(r"\bAbout\s+(?!Us\b|The\b|This\b|Your\b|Our\b|How\b)[A-Z][\w&.,'\- ]{1,60}")
_ABOUT_MARKERS = ("is a ", "is an ", "is the ", "leading ", "provider of",
                  "headquartered", "founded in", "helps ", "platform",
                  "subsidiar", "mission", "serves ", "offers ",
                  "more information", "www.", "http", "nyse", "nasdaq",
                  "company that", "organization that", "world's")

_FACT_RES = [
    re.compile(r"(?i)\b(?:founded|established|incorporated)\s+in\s+\d{4}"),
    re.compile(r"(?i)\bheadquartered\s+in\b|\bbased\s+in\s+[A-Z]"),
    re.compile(r"(?i)\(\s*(?:nyse|nasdaq|otc|tsx|amex)\s*:\s*[a-z.]{1,6}\s*\)"),
    re.compile(r"(?i)\b(?:a|the)\s+(?:leading|premier|privately[ -]held|next[ -]generation|global)\b[^.\n]{0,60}\b(?:compan(?:y|ies)|platform|provider|firm|leader|organi[sz]ation|agency|manufacturer|developer|institution)\b"),
    re.compile(r"(?i)\bsubsidiary of\b|\bdivision of\b"),
    re.compile(r"(?i)\bmission (?:is|of)\b"),
]

_URL_RE = re.compile(r"https?://[^\s\"<>)]+|\bwww\.[a-z0-9][^\s\"<>)]*", re.I)
_INVITE_RES = [
    # invitation must name a CONCRETE target (url/domain/email/phone), not just "visit our page"
    re.compile(r"(?i)(?:for (?:more|further) information|to learn more|more information)[^.\n]{0,90}?(?:visit|see|go to|available|contact|call|e-?mail)[^.\n]{0,80}?(?:https?://|www\.|[a-z0-9-]{2,}\.(?:com|org|net|gov|io|co)\b|@|\(?\d{3}\)?[\s\-.]\d{3,4}|1-8\d{2})"),
    re.compile(r"(?i)\b(?:please\s+)?visit\s+(?:https?://|www\.|[a-z0-9][a-z0-9-]*\.(?:com|org|net|gov|io|co)\b)"),
    re.compile(r"(?i)\b(?:available|found)\s+(?:at|on)\s[^.\n]{0,60}?(?:https?://|www\.)"),
]
_RELATED_RE = re.compile(r"(?i)\brelated (?:links|stories|articles|releases|news)\b")
_OFFICIAL_TERMS = ("newsroom", "investor relations", "media center",
                   "media centre", "media contacts", "media relations",
                   "press room", "pressroom", "media inquiries", "media kit")

_EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)
_PHONE_RE = re.compile(r"(?:\+?1[\s\-.()]{0,3})?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4}\b|\b1-8(?:00|88|77|66)-[A-Z0-9-]{4,}")
_CLABEL_RE = re.compile(r"(?i)\b(?:media|press|investor)\s+(?:contacts?|relations|inquiries)\b|\bcontact\(?s?\)?\s*:")
_SOURCE_RE = re.compile(r"\bSOURCE\s+[A-Z(]|(?i:\bsource\b\s*:\s*\S)")

# ---------- release-likeness regexes ----------
_WIRE_RE = re.compile(r"(?i)\bpr\s?newswire\b|\bbusiness\s?wire\b|\bglobe\s?newswire\b|\bmarketwired\b|\baccesswire\b|\bnews provided by\b")
_DATELINE_RES = [
    re.compile(r"\b[A-Z][A-Z. ']{1,28},?\s*(?:--|—|–)\s*\(?"),
    re.compile(r"\b[A-Z][A-Za-z.]+(?: [A-Z][A-Za-z.]+){0,3},\s+(?:[A-Z][a-z]+\.?|[A-Z]{2}(?:\.[A-Z]\.?)?|[A-Z]\.[A-Z]\.?),?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"),
]
_ANNOUNCE_RE = re.compile(r"(?i)\btoday,?\s+(?:\w+\s+){0,2}?(?:announc|issue[ds]|report(?:ed|s)|releas|launch|unveil|introduc)|\b(?:announced|reported|launched|unveiled)\s+today\b|\bannounces\b")
_SAID_RE = re.compile(r"[\"”]\s*,?\s*said\b|\bsaid\s+[A-Z][a-z]+|\bsaid:\s*[\"“]|\bsays\s+[A-Z][a-z]+")
_PRLABEL_RE = re.compile(r"(?i)\bpress releases?\b|\bnews releases?\b|\bnewsroom\b|\bmedia ?room\b")
# case-sensitive byline anchor + case-insensitive article markers (kept separate:
# a global (?i) would erase the [A-Z] classes and false-fire on "by Qualcomm Technologies,")
_BYLINE_RE = re.compile(r"(?m)^\s*By\s+[A-Z][a-z]+ [A-Z]")
_ARTICLE_RE = re.compile(r"(?i),\s*(?:national |senior |staff )?(?:editor|reporter)\b|\bauthors?\b|\bposted in\s*:|\bdisclosure\s*:\s*i\b|\bcomments?\s*\(\s*\d")


def _tail(t):
    n = max(1200, min(3500, int(0.4 * len(t))))
    return t[-n:]


def _about_score(t):
    """'About X' header must be followed by descriptive prose, not nav lists."""
    best = 0.0
    n = len(t)
    for m in _ABOUT_RE.finditer(t):
        win = t[m.end():m.end() + 450].lower()
        hits = sum(1 for mk in _ABOUT_MARKERS if mk in win)
        s = 1.0 if hits >= 2 else (0.5 if hits == 1 else 0.0)
        if s and m.start() < 0.55 * n:
            s *= 0.5  # nav 'About' menus live near the head
        best = max(best, s)
    return best


def _grounded(snippet, tl):
    """Cheap grounding check: some 12+-char chunk of the extraction appears in doc."""
    s = re.sub(r"\s+", " ", (snippet or "").lower()).strip(" .\"'")
    if len(s) < 8:
        return False
    for i in range(0, max(1, len(s) - 12), 6):
        if s[i:i + 12] in tl:
            return True
    return False


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if len(t) < 300:
            return 0.0
        tl = t.lower()
        tail = _tail(t)
        tail_l = tail.lower()

        # ---- component 1: organizational identity boilerplate ----
        about = _about_score(t)
        facts = sum(1 for rx in _FACT_RES if rx.search(t))
        ident = min(1.0, 0.55 * about + 0.25 * facts)

        # LLM field: quoted about-boilerplate (grounded + descriptive)
        ab = (extracted or {}).get("about_boilerplate", "")
        if ab and ab.strip().lower() not in ("", "none") and _grounded(ab, tl):
            abl = ab.lower()
            if any(mk in abl for mk in ("leading", "provider", "company", "platform",
                                        "founded", "headquartered", "helps", "global",
                                        "mission", "organization", "agency")):
                ident = max(ident, 0.8)

        # ---- component 2: links (invitation context, explicit URLs) ----
        invite = sum(1 for rx in _INVITE_RES if rx.search(t))
        nurl = min(5, len(_URL_RE.findall(t)))
        related = 1 if _RELATED_RE.search(t) else 0
        official = min(3, sum(1 for term in _OFFICIAL_TERMS if term in tail_l)
                       + (1 if re.search(r"\binvestors\b", tail_l) else 0))
        links = min(1.0, 0.40 * (invite >= 1) + 0.15 * (invite >= 2)
                    + 0.09 * nurl + 0.12 * related + 0.06 * official)

        ol = (extracted or {}).get("official_links", "")
        if ol and ol.strip().lower() not in ("", "none"):
            doms = re.findall(r"(?:https?://|www\.)?([a-z0-9-]{2,}\.(?:com|org|net|gov|io|co)\b)", ol.lower())
            n_ok = sum(1 for d in set(doms) if d in tl)
            if n_ok:
                links = max(links, min(1.0, 0.35 + 0.2 * n_ok))

        # ---- component 3: contact/helpline block in the TAIL ----
        c_email = 1 if _EMAIL_RE.search(tail) else 0
        c_phone = 1 if _PHONE_RE.search(tail) else 0
        c_label = 1 if _CLABEL_RE.search(tail) else 0
        c_source = 1 if _SOURCE_RE.search(tail) else 0
        contact = min(1.0, 0.35 * c_email + 0.25 * c_phone
                      + 0.20 * c_label + 0.30 * c_source)

        base = 0.38 * ident + 0.40 * links + 0.22 * contact

        # ---- release-likeness damp (non-releases judged ~0) ----
        wire = 1 if _WIRE_RE.search(t) else 0
        dateline = 1 if any(rx.search(t) for rx in _DATELINE_RES) else 0
        announce = 1 if _ANNOUNCE_RE.search(t) else 0
        said = 1 if _SAID_RE.search(t) else 0
        prlabel = 1 if _PRLABEL_RE.search(t) else 0
        src = 1 if _SOURCE_RE.search(tail) else 0

        r = (0.35 * wire + 0.25 * dateline + 0.20 * announce
             + 0.20 * said + 0.10 * prlabel + 0.15 * src)
        if c_email and c_label:
            r += 0.10  # genuine media/IR contact block in the tail
        if _BYLINE_RE.search(t) or _ARTICLE_RE.search(t):
            r -= 0.30
        if not (wire or dateline or src):
            head_lines = [ln for ln in t[:1800].split("\n") if ln.strip()]
            if len(head_lines) >= 12:
                short = sum(1 for ln in head_lines if len(ln.split()) <= 3)
                if short / len(head_lines) > 0.6:
                    r -= 0.20  # nav-chrome-dominated head
        r = max(0.0, min(1.0, r))

        return max(0.0, min(1.0, base * (0.25 + 0.75 * r) + 0.08 * r))
    except Exception:
        return 0.0
