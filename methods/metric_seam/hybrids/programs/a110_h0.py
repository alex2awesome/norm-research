"""a110 hybrid: Boilerplate, company info, and publication metadata.

Criterion: complete, up-to-date About section + essential footer elements (company
overview, affiliations, publication metadata, links, expert contacts).

Judge behavior (train pack): strongly bimodal.
  ~1.0  wire-distributed corporate releases with full boilerplate: an 'About <Company>'
        PROSE paragraph (or equivalent company-description prose), SOURCE line, exchange
        ticker, media contact with phone/email, 'for more information, visit <url>',
        PRNewswire/BusinessWire/Cision publication metadata.
  ~0.2  genuine releases with only PARTIAL footer (a contact phone, a forward-looking
        disclaimer, a registered-office block) but no About paragraph.
  ~0.0  non-releases / nav chrome. Crucially these are FULL of 'About Us' nav links,
        'Contact Us' links, URLs and copyright footers — so every cue must be validated
        (About header must be followed by prose, contact cue must have phone/email nearby).

Retrieval (evidence op): 'About X' + footers repeat nearly verbatim across same-issuer
releases, so a near-duplicate corpus hit corroborates boilerplate — used only as a small
gated bonus; degrades to 0 when ops has no corpus.
"""
import re

LLM_FIELDS = {
    "about_section": ("Quote the first sentence of the issuing company's 'About <Company>' "
                      "boilerplate paragraph (company-overview prose near the end, NOT a nav "
                      "link and NOT the PR Newswire/Cision footer); answer NONE if absent."),
    "footer_metadata": ("Quote one issuer footer-metadata line near the end: a 'SOURCE "
                        "<Company>' line, a media/press-contact line with phone or email, or "
                        "a 'For more information, visit <site>' line; answer NONE if absent."),
}

# ---------- About section -----------------------------------------------------
_ABOUT_HDR = re.compile(
    r"\bAbout\s+(?!Us\b|Me\b|The\b|This\b|That\b|These\b|Those\b|Our\b|You\b|Your\b|More\b|"
    r"All\b|PR\s|Cision\b|Press\b|News\b|Author\b|Media\b|Home\b|Blog\b|Store\b|Search\b|"
    r"Careers\b|Site\b|Contact\b|Investor\b|Newsroom\b|Privacy\b|Terms\b|Cookie\b|FAQ\b|"
    r"Help\b|Login\b|Photo\b|Video\b|Services\b)"
    r"[A-Z][\w&.'\-]*(?:\s+(?:[A-Z&][\w&.'\-]*|of|the|and|for|de)){0,6}")
_ABOUT_HDR2 = re.compile(r"(?i)\bAbout (?:the |our )?(?:company|organi[sz]ation|firm)\b")
_ABOUT_VERBS = re.compile(
    r"(?i)\b(is an?|is the|are an?|we are|helps?|provides?|provider|offers?|delivers?|"
    r"develops?|manufactures?|operates?|serves?|specializ\w*|dedicated to|committed to|"
    r"driven to|enables?|empowers?|headquartered|founded|leading|mission|safeguards?)\b")

# company-description prose cues (About-paragraph content, headerless fallback).
# Deliberately case-SENSITIVE lowercase bodies: boilerplate prose, not Title-Case nav.
_ORG_NOUN = (r"(?:company|corporation|organi[sz]ation|provider|firm|leader|platform|"
             r"institution|agency|manufacturer|supplier|producer|bank|insurer)")
_DESC_CUES = [
    re.compile(r"\bis an? (?:leading|global|world[- ]leading|premier|privately[- ]held|"
               r"non[- ]?profit)[a-z, -]{0,60}\b" + _ORG_NOUN + r"\b"),
    re.compile(r"\bis an? [a-z][a-z-]*(?: [a-z-]+){0,3} " + _ORG_NOUN + r"\b"),
    re.compile(r"\ban? (?:leading|global|world[- ]leading|premier|privately[- ]held)"
               r" [a-z, -]{0,45}" + _ORG_NOUN + r"\b"),
    re.compile(r"(?i)\bheadquartered in\b"),
    re.compile(r"(?i)\b[Ff]ounded in (?:19|20)\d{2}\b"),
    re.compile(r"\bemploys (?:approximately |more than |over |about |some )?[\d,]{3,}\b"),
    re.compile(r"\bworld'?s (?:largest|leading|first|biggest)[a-z, -]{0,40}\b"
               + _ORG_NOUN + r"\b"),
    re.compile(r"\bleading (?:provider|supplier|producer|manufacturer|developer) of\b"),
    re.compile(r"\b(?:billion|million) in (?:sales|revenue|assets|combined assets)\b"),
    re.compile(r"\b(?:19|20)\d{2} sales of \$"),
]

# ---------- publication metadata ------------------------------------------------
_WIRE = re.compile(
    r"(?i)(/\s*PR\s*Newswire\s*/|\(BUSINESS\s+WIRE\)|GLOBE\s+NEWSWIRE|GlobeNewswire|"
    r"ACCESSWIRE|Marketwired|\bnews\s+provided\s+by\b|\bPR\s*Newswire\b|\bCision\b|"
    r"\bBusiness\s+Wire\b)")
_SOURCE_LINE = re.compile(r"\bSOURCE[: ]\s{0,3}[A-Z][\w&.,'()\- ]{2,60}")
_TICKER = re.compile(
    r"\((?:NASDAQ|NYSE|AMEX|TSX[V]?|LSE|OTC[A-Z]*|Euronext|HKEX|ASX|FRA)\s*:\s*[A-Z.\- ]{1,8}\)")
_LEGAL = re.compile(r"(?i)\bforward[- ]looking statements?\b|\bsafe[- ]harbor\b")
_TEL_BLOCK = re.compile(r"(?i)\btelephone:\s*\+?[\d(]")

# ---------- contact block --------------------------------------------------------
_ANCHOR = re.compile(
    r"(?i)(\b(?:media|press|news)\s+contacts?\b(?!\s+(?:us|form|page))"
    r"|\b(?:media|press)\s+(?:inquir|enquir)\w*"
    r"|\bmedia\s+relations\b|\bpress\s+office\b"
    r"|\bcontacts?\s*:"
    r"|\bcontact\s+(?:cision|pr\s*newswire)\b"
    r"|\bfor\s+(?:further|more)\s+information\b"
    r"|\binquiries\s+may\s+be\s+directed\b)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}|\[email\s*protected\]")
_PHONE = re.compile(r"(?<![\d\w])(\+?\d{1,4}(?:[ ().\-]{1,3}\d{2,4}){2,5})(?!\d)")
_YEAR_RANGE = re.compile(r"(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}")
_BAD_PHONE_CTX = re.compile(r"(?i)(\$|%|copyright|\ball rights\b)")

# ---------- links / more-info ----------------------------------------------------
_MOREINFO = re.compile(
    r"(?i)(?:for more information|to learn more|learn more about)[^.\n]{0,90}?"
    r"(?:visit|www\.|https?://)"
    r"|(?i:visit)[^.\n]{0,80}?for more information")

_NONE_RE = re.compile(r"(?i)^\s*(none|n/?a|no\b|nothing|unknown|not\s)")
_DISTRIB = re.compile(r"(?i)\b(cision|pr\s*newswire|business\s*wire|globenewswire)\b")


def _prose_after(win):
    """Is the window after an 'About X' header descriptive prose (vs. a nav-link list)?"""
    words = re.findall(r"[A-Za-z][\w'&.\-]*", win)
    if len(words) < 18:
        return False
    low = sum(1 for w in words if w[0].islower())
    if low / len(words) < 0.45:
        return False
    return bool(_ABOUT_VERBS.search(win))


def _about_score(t, tail):
    for m in list(_ABOUT_HDR.finditer(t)) + list(_ABOUT_HDR2.finditer(t)):
        if _prose_after(t[m.end():m.end() + 380]):
            return 1.0
    cues = sum(1 for c in _DESC_CUES if c.search(tail))
    if cues >= 2:
        return 0.75
    if cues == 1:
        return 0.3
    return 0.0


def _phones_in(seg):
    out = []
    for m in _PHONE.finditer(seg):
        s = m.group(1)
        digits = re.sub(r"\D", "", s)
        if not (7 <= len(digits) <= 15):
            continue
        if _YEAR_RANGE.search(s):
            continue
        if _BAD_PHONE_CTX.search(seg[max(0, m.start() - 40):m.start() + len(s)]):
            continue
        out.append(s)
    return out


def _contact_score(tail):
    for m in _ANCHOR.finditer(tail):
        win = tail[max(0, m.start() - 170):m.start() + 320]
        if _EMAIL.search(win) or _phones_in(win):
            return 1.0
    return 0.0


def _retrieval_bonus(text, ops, releaseish):
    """Near-verbatim corpus repeat corroborates shared issuer boilerplate (gated,
    small, and 0 whenever ops has no corpus)."""
    if not releaseish:
        return 0.0
    try:
        hits = ops.retrieve_similar(text, k=6, exclude_id=None) or []
    except Exception:
        return 0.0
    sims = [s for s, _ in hits if s < 0.985]  # ~1.0 == self-match; skip
    if not sims:
        return 0.0
    top = max(sims)
    if top < 0.35:
        return 0.0
    return 0.05 * min(1.0, (top - 0.35) / 0.35)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        tail = t[-3600:]
        tail_desc = t[-4500:]
        tail_link = t[-2000:]

        about = _about_score(t, tail_desc)
        wire = 1.0 if _WIRE.search(t) else 0.0
        source = 1.0 if _SOURCE_LINE.search(tail) else 0.0
        contact = _contact_score(tail)
        moreinfo = 1.0 if _MOREINFO.search(tail_link) else 0.0
        ticker = 1.0 if _TICKER.search(t) else 0.0
        legal = 1.0 if _LEGAL.search(tail) else 0.0
        if contact == 0.0 and _TEL_BLOCK.search(t):
            contact = 0.5  # registered-office phone block (issuer metadata, weaker)

        # --- thick-input grounding: LLM saw the FULL document; predicate stays in code
        ab = ((extracted or {}).get("about_section") or "").strip()
        if len(ab) >= 15 and not _NONE_RE.match(ab) and not _DISTRIB.search(ab):
            grounded = re.sub(r"\s+", " ", ab[:40].lower()) in re.sub(r"\s+", " ", t.lower())
            about = max(about, 0.95 if grounded else 0.8)
        fm = ((extracted or {}).get("footer_metadata") or "").strip()
        if len(fm) >= 8 and not _NONE_RE.match(fm):
            if _EMAIL.search(fm) or _phones_in(fm):
                contact = max(contact, 0.85)
            if re.search(r"\bSOURCE\b", fm):
                source = max(source, 1.0)
            if re.search(r"(?i)visit|www\.|https?://", fm):
                moreinfo = max(moreinfo, 1.0)

        raw = (0.40 * about + 0.24 * wire + 0.14 * source + 0.12 * contact
               + 0.06 * moreinfo + 0.06 * ticker + 0.04 * legal)
        raw += _retrieval_bonus(t, ops, releaseish=(wire > 0 or source > 0 or about >= 0.75))
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
