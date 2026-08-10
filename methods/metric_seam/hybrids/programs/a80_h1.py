"""a80 hybrid r1: Identification and complete media contacts.

Round-1 refinement of h0, driven by measured TRAIN disagreements.

UNDER-scoring fixes (in-scope releases the judge scored high, h0 low):
  * contact blocks at the TOP of the document ('FOR IMMEDIATE RELEASE ...
    Contact: <office> <phone> <email>', 'Submitted by <name> Phone: <n>',
    named analyst blocks) -> scan a HEAD segment with the same detector;
  * missing anchors: 'For further information:' (PRNewswire sign-off),
    'please contact <name> (<email>)', 'press information hotlines',
    'Submitted by'; the for-more-information..contact gap was too tight;
  * scrubbed e-mails '[email protected]' (mailto protection) now count as
    e-mail evidence; person names mashed against a following capitalised
    token ('BoatnerDigital Experience ManagerIBM') now match;
  * 'Investor Relations' + BOTH phone AND e-mail right after it counts (the
    judge credits IR blocks on IR-hosted releases); bare nav mention never
    fires because both details are required.

OVER-scoring fixes (judge ~0, h0 mid):
  * release-likeness gate: the LLM media_contact boost needs release evidence
    (code markers or a non-empty LLM issuer or a code-found contact), and a
    document with NO release evidence at all is damped multiplicatively —
    the judge zeroes non-releases (event pages, nav chrome) but still gives
    partial credit to prominent personal contact blocks, hence damp not zero;
  * releases with no contact anywhere got ~0.3 from issuer alone while the
    judge gates on contact -> the no-contact branch is scaled down (order
    among no-contact docs is preserved: pure scaling, no new ties).

LLM_FIELDS are byte-identical to h0 (cached extractions are reused).
Code-only fallback (extracted={}) fully supported.
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
    r"(?i)(\b(?:contact\s+us|contact\s+information|request\s+a\s+demo|editorial\s+bureaus|"
    r"worldwide\s+offices|general\s+inquiries|media\s+inquiries|cookie\s+settings|"
    r"terms\s+of\s+use|privacy\s+policy|subscription\s+center|sign\s+up|log\s*in)\b"
    r"|888[-\s]776[-\s]0942)")  # PRN helpline often runs into next word ('...0942from')
_CHROME_EMAIL_DOM = re.compile(r"(?i)@(cision|prnewswire|businesswire|globenewswire)\.")

# --- contact-block anchors ---------------------------------------------------
_ANCHOR = re.compile(
    r"(?i)(\b(?:media|press|news)\s+contacts?\b(?!\s+(?:us|form|page))"
    r"|\bcontact\s+person\b"
    r"|\b(?:press|media)\s+(?:inquiries|enquiries)\b"
    r"|\bpress\s+office\b|\bmedia\s+relations\b|\bpublic\s+relations\b"
    r"|\bpress\s+information\b"
    r"|\bcontacts?\s*:"
    r"|\bfor\s+(?:further|more)\s+information[^.\n]{0,80}contact\b"
    r"|\bfor\s+(?:further|more)\s+information\s*:"
    r"|\bplease\s+contact\b"
    r"|\bsubmitted\s+by\b"
    r"|\binquiries\s+may\s+be\s+directed\b)")
# 'Investor Relations' is nav chrome almost everywhere -> only credited when
# BOTH an e-mail (incl. scrubbed) AND a phone follow within a short window.
_IR_ANCHOR = re.compile(r"(?i)\binvestor\s+relations\b")

# --- detail detectors --------------------------------------------------------
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}")
# mailto-protection placeholder left by scrapers: '[email protected]'
_EMAIL_SCRUB = re.compile(r"(?i)\[\s*email[ ]?protected\s*\]")
# OCR-dropped '@': first.last(at)?domain.tld or first.lastdomain.tld — never www/http
_EMAIL_NOAT = re.compile(
    r"(?<![\w/.])(?!www\.)[a-z][a-z-]{1,20}\.[a-z][a-z-]{1,20}"
    r"\s?(?:\(at\)|\[at\]|\bat\b)?\s?[a-z0-9-]{2,30}\.(?:com|org|net|edu|gov|co)\b")
# phones: allow spaced digit groups ('202 219 7499', '+47 48 02 75 75')
_PHONE = re.compile(r"(?<![\d\w])(\+?\d{1,4}(?:[ ().\-]{1,3}\d{2,4}){2,5})(?!\d)")
_YEAR_RANGE = re.compile(r"(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}")
_SVC_PHONE_CTX = re.compile(
    r"(?i)(customer\s+(?:support|service|care)|call\s+us|helpline|toll[- ]?free|"
    r"here\s+to\s+help|hearing\s+impaired|dial[- ]?in|fax|copyright|\$)")
# trailing \b -> (?![a-z]) so names mashed against a following capital still match
_NAME = re.compile(r"\b[A-Z][a-z]{2,}\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]{2,}(?![a-z])")
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
# 'ST. LOUIS, Dec. 6, 2016' — CAPS city + mixed-case month + day + year
_DATELINE3 = re.compile(r"\b[A-Z]{2,}[A-Z. ]{0,20},\s+[A-Za-z.]{2,15}\s+\d{1,2},\s+\d{4}")
_TODAY_VERB = re.compile(r"(?i)\btoday\s+(?:announc|launch|report|introduc|unveil|releas)")
_FIR = re.compile(r"(?i)\bfor\s+immediate\s+release\b")
_PRESS_HDR = re.compile(r"(?i)\bpress\s+release")
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
    out += [m.group(0) for m in _EMAIL_SCRUB.finditer(seg)]
    out += [m.group(0) for m in _EMAIL_NOAT.finditer(seg)
            if "@" not in m.group(0) and "http" not in seg[max(0, m.start() - 12):m.start()]]
    return out


def _names_in(seg):
    return [m.group(0) for m in _NAME.finditer(seg) if not _NAME_STOP.search(m.group(0))]


def _contact_code_score(seg):
    """Detect a genuine media-contact block in a (de-chromed) segment. [0,1]."""
    best = 0.0
    for m in _ANCHOR.finditer(seg):
        win = seg[m.start():m.start() + 420]
        emails, phones = _emails_in(win), _phones_in(win)
        if not emails and not phones:
            continue  # bare nav label — no details, no credit
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
    if best < 0.85:
        # IR contact block on IR-hosted releases: require BOTH details
        for m in _IR_ANCHOR.finditer(seg):
            win = seg[m.start():m.start() + 320]
            if _emails_in(win) and _phones_in(win):
                best = max(best, 0.85)
                break
    if best == 0.0:
        # anchor-free fallbacks — person-level evidence required
        ev = [m.start() for m in _EMAIL.finditer(seg)
              if not _CHROME_EMAIL_DOM.search(m.group(0))]
        ev += [m.start() for m in _EMAIL_SCRUB.finditer(seg)]
        for pos in ev:
            win = seg[max(0, pos - 260):pos + 60]
            if _names_in(win) and (_phones_in(win) or _ROLE.search(win)):
                best = max(best, 0.65 + (0.10 if _phones_in(win) else 0.0))
        if best == 0.0:
            # name + phone + ROLE (no e-mail): named spokesperson/analyst block
            for m in _PHONE.finditer(seg):
                s0 = m.group(1)
                digits = re.sub(r"\D", "", s0)
                if not (7 <= len(digits) <= 15) or _YEAR_RANGE.search(s0):
                    continue
                if _SVC_PHONE_CTX.search(seg[max(0, m.start() - 70):m.start()]):
                    continue
                win = seg[max(0, m.start() - 220):m.start() + 140]
                if _names_in(win) and _ROLE.search(win):
                    best = max(best, 0.65 + (0.10 if _emails_in(win) else 0.0))
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
    if _FIR.search(t):
        s += 0.30
    if _PRESS_HDR.search(t[:600]):
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
        head = clean[:3200]

        # contact blocks sit at the END of releases but at the TOP of agency /
        # university / report-style items — scan both segments
        contact = max(_contact_code_score(tail), _contact_code_score(head))
        issuer = _issuer_code_score(clean, tail)

        # release-likeness evidence (code-level)
        rel = bool(_WIRE.search(clean) or _PROVIDED_BY.search(clean)
                   or _SOURCE_LINE.search(tail) or _FIR.search(clean)
                   or _PRESS_HDR.search(clean[:600])
                   or _DATELINE.search(clean[:3000]) or _DATELINE2.search(clean[:3000])
                   or _DATELINE3.search(clean[:3000])
                   or _TICKER.search(clean) or _FOR_JOURNALISTS.search(clean))

        # thick-input grounding (predicate stays in code): LLM saw the FULL document.
        # The contact boost is gated on release evidence so helplines/registration
        # desks on non-release pages can no longer lift the score.
        mc = (extracted or {}).get("media_contact", "")
        iss = (extracted or {}).get("issuer", "")
        iss_ok = _plausible(iss)
        if _plausible(mc, need_detail=True) and (rel or iss_ok or contact > 0):
            contact = max(contact, 0.75 + (0.10 if contact > 0 else 0.0))
        if iss_ok:
            issuer = max(issuer, 0.65)

        if contact > 0 and issuer < 0.4:
            # a concrete named contact block itself identifies the issuing org
            issuer = max(issuer, 0.4)

        val = 0.62 * contact + 0.38 * issuer
        if contact == 0.0:
            # judge gates on media contact: issuer ID alone is worth little
            val = min(val, 0.40) * 0.70
            if _SOURCE_LINE.search(tail):
                # release visibly ended (SOURCE sign-off) with NO contact block
                val *= 0.55
        if not rel and not iss_ok:
            # no release evidence at all: judge mostly zeroes these, but keeps
            # partial credit for prominent personal contact blocks -> damp
            val *= 0.55
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
