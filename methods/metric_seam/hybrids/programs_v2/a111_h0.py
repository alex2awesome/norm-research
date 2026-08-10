"""a111 Clear calls to action and resource links -- hybrid (code + optional LLM fields).

Judged construct: ONE clear ask + obvious next steps with links to resources,
demos, samples, contacts, or subscriptions.

Design: score = cta_quality * (floor + (1-floor) * release_likeness).
 - cta_quality decouples PRESENCE from QUALITY: bare keyword/imperative hits are
   worth ~nothing; credit requires a directive COUPLED to a resource (verb + URL/
   phone/email on the same line), an explicit how-to ("To register..., go to..."),
   a "for more information, visit/contact X" framing, or an explicit pointer to a
   named artifact ("read the full report here"). Passive availability and
   boilerplate contact blocks get only small weight; privacy/cookie/terms-linked
   directives are excluded as chrome.
 - release_likeness damps non-releases (nav chrome, articles, ToS pages), which
   the judge scores ~0 regardless of stray CTA keywords; the gate floor is higher
   (0.4 vs 0.3) when a coupled/core CTA exists, since the judge gives partial
   credit to genuinely CTA-bearing non-release pages.
 - nav-density penalty guards against link-farm chrome inflating quality.
"""
import re

LLM_FIELDS = {
    "main_ask": "Quote the single clearest call-to-action sentence telling the reader a concrete next step (visit/register/call/download/subscribe); answer NONE if there is none.",
    "doc_kind": "One word: 'release' if a press release or official announcement, 'article' if news/blog commentary, 'page' if a website/product/nav/legal page.",
}

_ENG = {"the", "and", "of", "to", "in", "for", "is", "that", "with", "on", "as",
        "are", "this", "be", "by", "at", "from", "it", "or", "will", "has",
        "have", "more", "about", "our", "we", "you", "was", "an", "its"}

_BAN_CTX = ("privacy", "cookie", "terms of", "ad choices", "unsubscribe", "javascript")

# ---------- CTA-quality patterns (lowercase text) ----------
_P_DIRECTIVE_INFO = re.compile(
    r"(?:for (?:more|further|additional|general) (?:information|details|info|media inquiries|inquiries)"
    r"|to learn more|to find out more)"
    r"[^.\n]{0,80}?(?:please\s+)?(?:visit|contact|call|email|e-mail|go to|see|check out|click|dial)"
    r"|contact person for more information")
_P_HOWTO = re.compile(
    r"\bto (?:pre-?register|register|experience|access|attend|participate|apply|sign up"
    r"|get started|subscribe|claim|redeem|rsvp|request)"
    r"[^.\n]{0,90}?(?:please\s+|simply\s+)?(?:visit|go to|message|call|email|text|click|use|enter|follow|dial)"
    r"|\buse this link to\b|\bgo to the following link\b|\bclick here to\b")
_P_ACTION_RES = re.compile(
    r"\b(?:visit|call|dial|email|e-mail|contact|text|message|go to)\s"
    r"[^.\n]{0,60}?(?:https?://|www\.|[\w.+-]+@[\w-]+[.\w-]*\.\w{2,}|\+?\(?\d[\d()\s.\-]{6,}\d)")
_P_STRONG_ASK = re.compile(
    r"\b(?:order (?:yours|now|today)|pre-?order (?:now|today)|buy now"
    r"|activate (?:your )?subscription|start (?:your )?free trial|start now"
    r"|get your (?:free|7-day|7 day)|request a demo|schedule a demo"
    r"|apply (?:now|today)|register (?:now|today|here)|join (?:now|today)"
    r"|donate now|rsvp|book now|shop now|reserve (?:your|now))")
_P_READ_RES = re.compile(
    r"\b(?:read|download|access|view)\s+(?:the\s+|our\s+|this\s+|a\s+|full\s+)*"
    r"[\w\-&+ ]{0,30}?(?:report|filing|study|white\s?paper|fact\s?sheet|infographic|pdf"
    r"|guide|brochure|prospectus)\b")
_P_READ_HERE = re.compile(
    r"\b(?:read|check out|view|download|see)\s+(?:the|our|this|a)\s+[\w\- ]{0,30}?\s?(?:here|below)\b")
_P_SIGNUP = re.compile(
    r"\bsign\s?up\b|\bsubscrib\w*\b|\bsubscription\b|\bnewsletter\b|\brss\b"
    r"|\bemail (?:updates|alerts)\b|\bin your inbox\b|\bregister\b|\bfree trial\b")
_P_CONTACT_BLOCK = re.compile(
    r"(?:media contacts?|media inquiries|press office|press contacts?|media relations|\bcontact:)"
    r"\s*.{0,200}?(?:[\w.+-]+@[\w-]+\.\w{2,}|\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}|\+\d{1,3}[\s\-(]?\d)",
    re.S)
_P_GERUND_CONTACT = re.compile(
    r"\b(?:email|call|text|visit)ing\s+[^.\n]{0,40}?(?:@|\d{3}[\s.\-]\d)")
_P_PASSIVE = re.compile(
    r"(?:available|accessible)\s+(?:at|on|via|from)\s+(?:https?://|www\.|[a-z0-9\-]+\.(?:com|org|gov|net|io))"
    r"|can access [^.\n]{0,40}?(?:at\s+)?https?://")
_P_URL = re.compile(r"https?://\S+|www\.[\w\-]+\.\w\S*")

# ---------- release-likeness patterns ----------
_P_WIRE = re.compile(r"/PRNewswire/|PRNewswire|PR Newswire|BUSINESS WIRE|Business Wire"
                     r"|GLOBE NEWSWIRE|Globe Newswire|News provided by|RNS Number")
_P_IMMEDIATE = re.compile(r"FOR IMMEDIATE RELEASE", re.I)
_P_DATELINE1 = re.compile(
    r"\b[A-Z][A-Za-z. ]{2,30},\s*(?:[A-Za-z.]{2,20},?\s+)?"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}")
_P_DATELINE2 = re.compile(
    r"\b[A-Z]{3,}(?:\s+[A-Z]{2,})*\s*,\s*[A-Z][a-z]+\.?(?:\s+[A-Z][a-z]+)?\s*[-–—]")
_P_ANNOUNCED = re.compile(
    r"\btoday (?:announce[sd]|report(?:s|ed)|file[sd]|launch(?:es|ed)|release[sd]"
    r"|introduce[sd]|unveil(?:s|ed)|issue[sd]|marks|kicks off)\b"
    r"|\bannounced today\b|\breleased today\b|\bis pleased to announce\b")
_P_ANNOUNCE_WEAK = re.compile(r"\bannounc(?:es|ed|ing|ement)\b")
_P_SAID = re.compile(r"[\"”]\s*,?\s*said\s|,\s*said\s+[A-Z][a-z]")
_P_PRNAV = re.compile(r"\bpress releases?\b|\bnewsroom\b|\bnews releases?\b|\bmedia cent(?:er|re)\b")
_P_HEADLINE_VERB = re.compile(
    r"\b[A-Z][\w&'.\- ]{1,50}\s(?:Releases|Announces|Launches|Unveils|Introduces"
    r"|Publishes|Schedules|Files|Reports)\s")
_P_NOTICE = re.compile(r"notice is hereby given|matters to be considered"
                       r"|contact person for more information|sunshine act")
_P_MEDIA_CUE = re.compile(r"\bmedia contacts?\b|\bpress office\b|\bmedia inquiries\b"
                          r"|\bmedia relations\b|\bnote to journalists\b")
_P_HASHES = re.compile(r"\n\s*###\s*(?:\n|$)")
_P_OFFER = re.compile(r"\blimited.time offer\b|\bredeem\b|\beligible\b|\bon us\b")
_P_ABOUT = re.compile(r"\bAbout (?!Us\b|us\b|This\b)[A-Z][A-Za-z]+")
_P_ARTICLE = re.compile(r"This article was written by|Editor'?s Note|Seeking Alpha"
                        r"|\bFollower\s?s?\b|min read|Posted In:|Comments?\s*\(\s*\d+\s*\)")
_P_TOS = re.compile(r"terms of (?:use|service)|terms and conditions")

_ASK_VERB = re.compile(r"\b(?:visit|call|contact|email|register|sign up|subscribe|download"
                       r"|go to|click|order|activate|join|donate|apply|request|message|dial|rsvp)\b")
_ASK_RES = re.compile(r"https?://|www\.|\.com|\.org|\.gov|@|\blink\b|\bwebsite\b|\d{3}[\s.\-]\d{3,4}")


def _clean_hits(pat, tl):
    """Count matches whose local context is not privacy/cookie/ToS chrome."""
    n = 0
    for m in pat.finditer(tl):
        window = tl[m.start(): m.end() + 45]
        if not any(b in window for b in _BAN_CTX):
            n += 1
    return n


def _release_likeness(t, tl):
    r = 0.0
    if _P_WIRE.search(t):
        r += 0.45
    if _P_IMMEDIATE.search(t):
        r += 0.45
    if _P_DATELINE1.search(t):
        r += 0.3
    elif _P_DATELINE2.search(t):
        r += 0.25
    if _P_ANNOUNCED.search(tl):
        r += 0.3
    elif _P_ANNOUNCE_WEAK.search(tl):
        r += 0.15
    if _P_HEADLINE_VERB.search(t):
        r += 0.2
    if _P_SAID.search(t):
        r += 0.2
    if _P_PRNAV.search(tl):
        r += 0.15
    if _P_NOTICE.search(tl):
        r += 0.35
    if _P_MEDIA_CUE.search(tl):
        r += 0.2
    if _P_HASHES.search(t):
        r += 0.2
    n_offer = len(_P_OFFER.findall(tl))
    if n_offer >= 2:   # offer announcement (redemption/eligibility framing)
        r += 0.25
    elif n_offer == 1:
        r += 0.1
    if _P_ABOUT.search(t):
        r += 0.1
    # anti-cues
    if _P_ARTICLE.search(t):
        r -= 0.5
    if len(_P_TOS.findall(tl)) >= 2:
        r -= 0.25
    alpha = [c for c in t if c.isalpha()]
    if len(alpha) > 500:
        caps = sum(1 for c in alpha if c.isupper()) / len(alpha)
        if caps > 0.3:
            r -= 0.3
    return max(0.0, min(1.0, r))


def _cta_quality(t, tl):
    comps = []   # weights for noisy-or aggregation
    core = False # coupled/explicit CTA present (raises the gate floor)
    if _clean_hits(_P_DIRECTIVE_INFO, tl):
        comps.append(0.5)
        core = True
    n_howto = _clean_hits(_P_HOWTO, tl)
    if n_howto:
        comps.append(0.5 if n_howto >= 2 else 0.4)
        core = True
    if _clean_hits(_P_ACTION_RES, tl):
        comps.append(0.35)
        core = True
    if _P_STRONG_ASK.search(tl):
        comps.append(0.35)
        core = True
    n_here = _clean_hits(_P_READ_HERE, tl)
    if n_here:
        comps.append(0.3 if n_here >= 2 else 0.25)
        core = True
    if _P_READ_RES.search(tl):
        comps.append(0.25)
        core = True
    n_signup = len(_P_SIGNUP.findall(tl))
    if n_signup >= 3:
        comps.append(0.2)
    elif n_signup >= 1:
        comps.append(0.12)
    if _P_CONTACT_BLOCK.search(tl):
        comps.append(0.1)
    if _P_GERUND_CONTACT.search(tl):
        comps.append(0.1)
    if _P_PASSIVE.search(tl):
        comps.append(0.08)
    if len(_P_URL.findall(t)) >= 2:
        comps.append(0.08)
    q = 1.0
    for w in comps:
        q *= (1.0 - w)
    return 1.0 - q, core


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if len(t) < 50:
            return 0.0
        # strip syndication-footer chrome (Cision/PRNewswire boilerplate carries
        # fake CTAs like "Request a Demo" on every hosted release)
        for marker in ("Contact Cision", "Cision Distribution 888"):
            idx = t.find(marker)
            if idx > len(t) * 0.4:
                t = t[:idx]
                break
        tl = t.lower()

        # non-English gate
        words = re.findall(r"[a-zA-Z']+", tl)
        if words:
            eng_ratio = sum(1 for w in words if w in _ENG) / len(words)
            if eng_ratio < 0.06:
                return 0.02

        q, core = _cta_quality(t, tl)
        r = _release_likeness(t, tl)

        # nav/link-farm density penalty: chrome pages are walls of <=3-word lines.
        # Softened for release-like docs (nav chrome wrapping a real release body).
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        if len(lines) >= 15:
            frac_short = sum(1 for ln in lines if len(ln.split()) <= 3) / len(lines)
            if frac_short > 0.7:
                q *= (1.0 - min(0.35, (frac_short - 0.7) * 2.0) * (1.0 - 0.5 * r))

        # wire-syndicated releases: full body carries hyperlink CTAs the scrape
        # flattens; keep them a notch above true zero-CTA chrome
        if q < 0.05 and _P_WIRE.search(t):
            q = 0.05

        # ---- optional LLM thick-input signal (no-op when extracted == {}) ----
        kind = (extracted or {}).get("doc_kind")
        if kind is not None and kind.strip():
            k = kind.strip().lower()
            if "release" in k or "announce" in k:
                r = max(r, 0.85)
            elif "article" in k or "page" in k:
                r = min(r, 0.45)
        ask = (extracted or {}).get("main_ask")
        if ask is not None:
            a = ask.strip().lower()
            if a and a not in ("none", "n/a", "no", "-", "null"):
                if _ASK_VERB.search(a):
                    q = 1.0 - (1.0 - q) * 0.6
                    core = True
                if _ASK_RES.search(a):
                    q = 1.0 - (1.0 - q) * 0.85
            else:
                q *= 0.75

        floor = 0.4 if core else 0.3
        return max(0.0, min(1.0, q * (floor + (1.0 - floor) * r)))
    except Exception:
        return 0.5
