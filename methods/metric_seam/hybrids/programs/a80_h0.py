"""a80 hybrid: Identification and complete media contacts.

Two halves per the criterion:
  (1) issuer identification  — dateline / 'SOURCE X' / 'News provided by' / ticker / wire marker
  (2) media-contact details  — a REAL press-contact block (name/role + phone/email) near the END,
      NOT distributor chrome (Cision/PRNewswire footer), nav 'Contact Us', subscription or
      customer-service numbers, or byline reporter emails.

Judge behavior (from adjudication + train pack): media-contact presence GATES the score —
issuer identification alone (e.g. a PRNewswire release with 'SOURCE X' but no contact block)
still scores ~0. Contact blocks often sit mid-document (before the chrome footer), so an LLM
extraction field is declared for thick-input grounding; code must degrade gracefully without it.
"""
import re

LLM_FIELDS = {
    "media_contact": ("Name the media/press contact person, team, or department this document "
                      "designates for journalists to reach (with phone/email if given); answer "
                      "NONE if only website chrome, distributor (Cision/PRNewswire) footers, or "
                      "customer-service contacts appear."),
    "issuer": ("Name the organization issuing this press release (the 'SOURCE' / 'News provided "
               "by' organization); answer NONE if this is not a press release."),
}

# --- distributor / page chrome ---------------------------------------------
_CHROME_CUT = re.compile(
    r"(?i)(contact\s+cision|contact\s+pr\s*newswire|cision\s+distribution|"
    r"cision\s+communications?\s+cloud|about\s+pr\s*newswire)")
_CHROME_PHRASES = re.compile(
    r"(?i)\b(contact\s+us|contact\s+information|request\s+a\s+demo|editorial\s+bureaus|"
    r"worldwide\s+offices|general\s+inquiries|media\s+inquiries|cookie\s+settings|"
    r"terms\s+of\s+use|privacy\s+policy|subscription\s+center|sign\s+up|log\s*in|"
    r"888-776-0942)\b")
_CHROME_EMAIL_DOM = re.compile(r"(?i)@(cision|prnewswire|businesswire|globenewswire)\.")

# --- contact-block anchors ---------------------------------------------------
_ANCHOR = re.compile(
    r"(?i)(\b(?:media|press|news)\s+contacts?\b(?!\s+(?:us|form|page))"
    r"|\bcontact\s+person\b"
    r"|\b(?:press|media)\s+(?:inquiries|enquiries)\b"
    r"|\bpress\s+office\b|\bmedia\s+relations\b|\bpublic\s+relations\b"
    r"|\bcontacts?\s*:"
    r"|\bfor\s+(?:further|more)\s+information[^.\n]{0,40}contact\b"
    r"|\binquiries\s+may\s+be\s+directed\b)")

# --- detail detectors --------------------------------------------------------
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}")
# OCR-dropped '@': first.last(at)?domain.tld or first.lastdomain.tld — never www/http
_EMAIL_NOAT = re.compile(
    r"(?<![\w/.])(?!www\.)[a-z][a-z-]{1,20}\.[a-z][a-z-]{1,20}"
    r"\s?(?:\(at\)|\[at\]|\bat\b)?\s?[a-z0-9-]{2,30}\.(?:com|org|net|edu|gov|co)\b")
# phones: allow spaced digit groups ('202 219 7499', '+47 48 02 75 75')
_PHONE = re.compile(r"(?<![\d\w])(\+?\d{1,4}(?:[ ().\-]{1,3}\d{2,4}){2,5})(?!\d)")
_YEAR_RANGE = re.compile(r"(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}")
_SVC_PHONE_CTX = re.compile(
    r"(?i)(customer\s+(?:support|service|care)|call\s+us|helpline|toll[- ]?free|"
    r"hearing\s+impaired|dial[- ]?in|fax|copyright|\$)")
_NAME = re.compile(r"\b[A-Z][a-z]{2,}\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]{2,}\b")
_NAME_STOP = re.compile(
    r"(?i)\b(contact us|terms of|privacy|united states|new york|press release|site map|"
    r"all rights|learn more|read more|about us|cookie|investor relations)\b")
_ROLE = re.compile(
    r"(?i)\b(communications?|public\s+relations|media\s+relations|spokes\w+|press\s+officer|"
    r"director|manager|advis[eo]r|secretary|head\s+of|vice\s+president|vp\b|officer)\b")

# --- issuer detectors --------------------------------------------------------
_WIRE = re.compile(
    r"(?i)(/\s*PRNewswire\s*/?|\(BUSINESS\s+WIRE\)|GLOBE\s+NEWSWIRE|/\s*CNW\s*/|"
    r"ACCESSWIRE|Marketwired|News\s+Direct)")
_PROVIDED_BY = re.compile(r"(?i)\bnews\s+provided\s+by\b")
_SOURCE_LINE = re.compile(r"\bSOURCE[: ]\s{0,3}[A-Z][\w&.,'\- ]{2,60}")
_TICKER = re.compile(r"\((?:NASDAQ|NYSE|AMEX|TSX[V]?|LSE|OTC[A-Z]*|Euronext)\s*:\s*[A-Z.\- ]{1,8}\)")
_DATELINE = re.compile(
    r"\b[A-Z]{3,}[A-Z .]{0,25},\s*[A-Za-z.]{2,15},?\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}")
_DATELINE2 = re.compile(r"\b[A-Z]{4,}(?:[ ,][A-Z.]{2,10})?\s+[-–—]\s+[A-Z]")
_TODAY_VERB = re.compile(r"(?i)\btoday\s+(?:announc|launch|report|introduc|unveil|releas)")
_FOR_JOURNALISTS = re.compile(r"(?i)\bfor\s+journalists\b")
_NEWSROOM = re.compile(r"(?i)\bnews\s*room\b")
_NONE_RE = re.compile(r"(?i)^\s*(none|n/?a|no\b|nothing|unknown|not\s)")


def _phones_in(seg):
    out = []
    for m in _PHONE.finditer(seg):
        s = m.group(1)
        digits = re.sub(r"\D", "", s)
        if not (7 <= len(digits) <= 15):
            continue
        if _YEAR_RANGE.search(s):
            continue
        ctx = seg[max(0, m.start() - 70):m.start()]
        if _SVC_PHONE_CTX.search(ctx):
            continue
        out.append(s)
    return out


def _emails_in(seg):
    out = [m.group(0) for m in _EMAIL.finditer(seg)
           if not _CHROME_EMAIL_DOM.search(m.group(0))]
    out += [m.group(0) for m in _EMAIL_NOAT.finditer(seg)
            if "@" not in m.group(0) and "http" not in seg[max(0, m.start() - 12):m.start()]]
    return out


def _names_in(seg):
    return [m.group(0) for m in _NAME.finditer(seg) if not _NAME_STOP.search(m.group(0))]


def _contact_code_score(tail):
    """Detect a genuine media-contact block in the (de-chromed) tail. [0,1]."""
    best = 0.0
    for m in _ANCHOR.finditer(tail):
        win = tail[m.start():m.start() + 420]
        emails, phones = _emails_in(win), _phones_in(win)
        if not emails and not phones:
            continue  # bare nav label ('Media Contacts' link) — no details, no credit
        s = 0.5
        if emails:
            s += 0.20
        if phones:
            s += 0.15
        if _names_in(win):
            s += 0.10
        if _ROLE.search(win):
            s += 0.05
        best = max(best, min(1.0, s))
    if best == 0.0:
        # anchor-free fallback: personal email + nearby name (e.g. 'Jane Doe ... 555 123 4567
        # jdoe@corp.com' right before a SOURCE line) — still requires person-level evidence
        for m in _EMAIL.finditer(tail):
            if _CHROME_EMAIL_DOM.search(m.group(0)):
                continue
            win = tail[max(0, m.start() - 260):m.start() + 60]
            if _names_in(win) and (_phones_in(win) or _ROLE.search(win)):
                best = max(best, 0.65 + (0.1 if _phones_in(win) else 0.0))
    return best


def _issuer_code_score(t, tail):
    s = 0.0
    if _WIRE.search(t):
        s += 0.35
    if _PROVIDED_BY.search(t[:2500]) or _PROVIDED_BY.search(t):
        s += 0.20
    if _SOURCE_LINE.search(tail):
        s += 0.25
    if _TICKER.search(t):
        s += 0.35
    if _DATELINE.search(t[:3000]) or _DATELINE2.search(t[:3000]):
        s += 0.15
    if _TODAY_VERB.search(t[:3000]):
        s += 0.10
    if re.search(r"(?i)\bpress\s+release", t[:300]):
        s += 0.30
    if _FOR_JOURNALISTS.search(t):
        s += 0.15
    if _NEWSROOM.search(t):
        s += 0.04
    return min(1.0, s)


def _plausible(ans, need_detail=False):
    ans = (ans or "").strip()
    if len(ans) < 3 or _NONE_RE.match(ans):
        return False
    if re.search(r"(?i)\b(cision|pr\s*newswire|business\s*wire)\b", ans):
        return False  # distributor is not the issuer's media contact
    if need_detail:
        return bool(_EMAIL.search(ans) or re.search(r"\d{3}", ans)
                    or _names_in(ans) or _ROLE.search(ans))
    return True


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        # cut distributor footer (everything after the first Cision/PRN footer marker)
        mcut = _CHROME_CUT.search(t, max(0, len(t) - 6000))
        body = t[:mcut.start()] if mcut else t
        # neutralize residual nav-chrome phrases so they can't anchor or feed windows
        clean = _CHROME_PHRASES.sub(" ", body)
        tail = clean[-4200:]

        contact = _contact_code_score(tail)
        issuer = _issuer_code_score(clean, tail)

        # thick-input grounding (predicate stays in code): LLM saw the FULL document
        mc = (extracted or {}).get("media_contact", "")
        if _plausible(mc, need_detail=True):
            contact = max(contact, 0.75 + (0.10 if contact > 0 else 0.0))
        iss = (extracted or {}).get("issuer", "")
        if _plausible(iss):
            issuer = max(issuer, 0.65)

        if contact > 0 and issuer < 0.4:
            # a concrete named contact block itself identifies the issuing org
            issuer = max(issuer, 0.4)

        val = 0.62 * contact + 0.38 * issuer
        if contact == 0.0:
            val = min(val, 0.40)  # judge gates on media contact: issuer ID alone is not usable
            if _SOURCE_LINE.search(tail):
                # release visibly ended (SOURCE sign-off) with NO contact block anywhere:
                # strong evidence the release genuinely omits media contacts
                val *= 0.55
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
