"""a42 Audience-centric messaging and engagement — hybrid channel (h0).

Design (from train-pack structure):
  Judge bands: ~0.0 non-releases (nav chrome / spec tables / FAQ / blogs),
  ~0.2-0.25 coherent reader-addressed articles/marketing AND releases with NO
  reader next-steps, ~0.65-0.85 releases/announcements with concrete audience
  apparatus (media contact w/ email+phone, register/webcast/dial-in how-tos,
  press kits / photos / b-roll, "for more information visit <url>", sign-ups).

  score = article_credit(prose)                      if no announce evidence
        = damp * (base + spread * Q_audience)        if announce evidence
  where Q demands SPECIFIC apparatus (emails, phones, URLs adjacent to
  instructions) — bare CTA keywords ("click", "learn more") are worth ~0,
  which is exactly where the v1 baseline failed (presence != quality).
"""
import re
import math

LLM_FIELDS = {
    "doc_type": "One word: press release/company announcement=release; news/blog/analyst piece=article; product or marketing page=marketing; navigation/list/chrome dump=navigation.",
    "reader_next_step": "Quote (<=20 words) the clearest concrete next step offered to readers/journalists (URL, contact, registration, download); NONE if none.",
}

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}|\+\d{2}[ ]?\d{2}[ ]?\d{2}[ ]?\d{2}[ ]?\d{2}")
_URL = re.compile(r"https?://\S+|www\.[\w.-]+\.\w{2,}|\b[\w-]+\.(?:com|org|gov|net)/\S+", re.I)

# --- strong announce evidence (any one gates the release channel) ---
_WIRE = re.compile(
    r"/?PRNewswire/?|BUSINESS WIRE|GLOBE NEWSWIRE|Marketwired|ACCESSWIRE|"
    r"FOR IMMEDIATE RELEASE|PRESS RELEASE|News provided by|News Releases?/Statements|"
    r"Press Releases?\b|\bSOURCE [A-Z][\w,. ]{2,40}\b")
_NEWSROOM = re.compile(r"press kits?\b|contact [A-Z][\w]* PR\b|news source for media|"
                       r"media and publications|view all (?:press releases|news)|"
                       r"aimed at journalists", re.I)
_DATELINE = re.compile(
    r"\b[A-Z][A-Z'’.-]+(?: [A-Z][A-Z'’.-]+){0,3},(?: [A-Z][a-zA-Z.]{1,14},?)? +"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}")
_DATELINE_EU = re.compile(r"(?:PRESS RELEASE|NEWS RELEASE)[ ,]+\d{1,2} [A-Z]+ \d{4}")
_ANNOUNCE = re.compile(
    r"\btoday(?:'s)?,? (?:announc|launch|issu|unveil|introduc|releas|report)\w*|"
    r"\bannounc\w+ today\b|\bis pleased to announce\b|"
    r"\btoday marks\b|\btoday we (?:issued|announced|released|launched|shared)\b|"
    r"\bwe(?:['’]| a| ?a)?re (?:releasing|sharing|announcing|launching|introducing)\b|"
    r"\bweare (?:releasing|sharing|announcing|launching)\b", re.I)

# --- weak release corroboration ---
_QUOTE_SAID = re.compile(r'"[^"\n]{30,400}"\s*,?\s*(?:said|says)\s+[A-Z]|'
                         r'\bsaid\s+[A-Z][\w.]+ [A-Z][\w.]+\s*,')
_ABOUT = re.compile(r"\bAbout (?:the )?[A-Z][\w&.'-]*")
_TICKER = re.compile(r"[([](?:NYSE|NASDAQ|Nasdaq|OTC|TSX|LSE)\s*:\s*[A-Z.]{1,6}[)\]]")
_FWD = re.compile(r"forward-looking statements", re.I)

# --- audience/next-step apparatus (specific, not bare keywords) ---
_MEDIA_CTX = re.compile(r"media (?:contact|inquir|relations)|press (?:office|contact|inquir)|"
                        r"contact:? |for (?:media|press)|communications? (?:adviser|officer)|"
                        r"\bPR\b", re.I)
_ASSETS = re.compile(r"press kit|b-?roll|hi-?res|download(?:able)?|multimedia|"
                     r"photos? ?[-–(:]|logo ?[-–(:]|webcast|dial-?in|conference call|"
                     r"replay|podcast|fact sheet|infographic|full-?size version", re.I)
_HOWTO = re.compile(r"\b(?:to (?:pre-?register|register|experience|access|participate|join|rsvp|"
                    r"learn more|sign up|get started)|for more information|to learn more|"
                    r"learn more at|more information about|you can read|full (?:session )?details|"
                    r"read the details|find additional resources)\b", re.I)
_SIGNUP = re.compile(r"sign up for|subscribe to|email (?:alerts|updates)|newsletter|"
                     r"receive the latest|stay up-?to-?date", re.I)
_JOURNO = re.compile(r"journalists|news source for media|media and publications|newsroom", re.I)
_TAKEAWAY = re.compile(r"key takeaways|highlights|quick links|at a glance", re.I)
_AUDIENCE = re.compile(r"(?:guide|resources?|tips|toolkit|prompts|stories) for [a-z]+|"
                       r"for (?:teachers|educators|students|developers|journalists|"
                       r"customers|investors|patients|travelers|families)\b|"
                       r"help(?:s|ing)? (?:you|educators|teachers|customers|journalists|readers)", re.I)

# --- non-release / junk markers ---
_FAQ = re.compile(r"frequently asked questions|commonly asked questions|\bFAQ\b|\bQ\s*:\s", re.I)
_BLOGGY = re.compile(r"read more\.\.\.|comments? \(\d+\)|\bfollowers?\b|posted in:|"
                     r"disclosure:\s*i am|seeking alpha|editor'?s note", re.I)


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _near(pat_a, pat_b, text, window=160):
    """True if a pat_a match has a pat_b match within `window` chars after it."""
    for m in pat_a.finditer(text):
        seg = text[m.start(): m.end() + window]
        if pat_b.search(seg):
            return True
    return False


def score(text: str, extracted: dict, ops) -> float:
    t = ops.normalize(text or "")
    if not t.strip():
        return 0.0
    n = len(t)
    tail = t[int(n * 0.60):]           # structural blocks sit near the END
    body = t[: int(n * 0.75)]

    # ---------- junk / chrome profile (on the BODY: scraped tails are chrome-y
    # even for genuine releases, so damp on where the message lives) ----------
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    if lines:
        chrome = sum(1 for ln in lines if len(ln.split()) <= 3) / len(lines)
    else:
        chrome = 0.0
    alnum = sum(1 for c in body if c.isalnum()) or 1
    digit_density = sum(1 for c in body if c.isdigit()) / alnum
    n_sent, mwps, _fl = ops.sent_stats(body)
    faq_hits = len(_FAQ.findall(t))
    blog_hits = len(_BLOGGY.findall(t))

    # prose coherence P: real sentences, moderate length, not a nav/spec dump
    p_sent = _clamp((mwps - 8.0) / 6.0) * _clamp((55.0 - mwps) / 20.0)
    p_chrome = _clamp((0.75 - chrome) / 0.45)
    p_digit = _clamp((0.09 - digit_density) / 0.05)
    P = _clamp(0.45 * p_sent + 0.35 * p_chrome + 0.20 * p_digit)
    if faq_hits >= 3:
        P *= 0.35
    if blog_hits >= 2:
        P *= 0.55

    # ---------- release evidence ----------
    strong = 0
    if _WIRE.search(t):
        strong += 1
    if _DATELINE.search(body) or _DATELINE_EU.search(body):
        strong += 1
    if _ANNOUNCE.search(body):
        strong += 1
    if len(set(m.group(0).lower()[:10] for m in _NEWSROOM.finditer(t))) >= 2:
        strong += 1  # corporate-newsroom apparatus (press kits + "Contact X PR" etc.)
    weak = sum((bool(_QUOTE_SAID.search(t)), bool(_ABOUT.search(tail)),
                bool(_TICKER.search(t)), bool(_FWD.search(tail)),
                bool(_TAKEAWAY.search(t)) or bool(_JOURNO.search(t))))

    # LLM doc_type: thick override for the gate (code-only fallback unaffected)
    dt = (extracted or {}).get("doc_type", "").strip().lower()
    if dt.startswith("release"):
        strong = max(strong, 1)
    elif dt.startswith("navigation"):
        P *= 0.35
        strong = 0
    elif dt.startswith(("article", "marketing")) and strong <= 1 and weak <= 1:
        strong = 0  # lone accidental cue in an article ("...announced today...")

    is_release = strong >= 1 and (strong + weak) >= 2
    weak_release = strong >= 1 and not is_release

    # ---------- audience apparatus Q (specificity required) ----------
    # contact credit only for the RELEASE's own media contact, not the wire
    # distributor's boilerplate (Cision/PR Newswire footer phone numbers)
    def _own_contact(pat):
        for m in pat.finditer(tail):
            ctx = tail[max(0, m.start() - 120): m.end() + 60]
            if not re.search(r"cision|pr ?newswire|business ?wire", ctx, re.I):
                return True
        return False
    has_email = _own_contact(_EMAIL)
    has_phone = _own_contact(_PHONE)
    media_ctx = bool(_MEDIA_CTX.search(tail)) or bool(re.search(r"media@|press@|pr@", t, re.I))
    q_contact = (0.5 * has_email + 0.5 * has_phone) * (1.0 if media_ctx else 0.55)
    q_assets = _clamp(len(set(m.group(0).lower()[:8] for m in _ASSETS.finditer(t))) / 3.0)
    q_howto = 0.0
    if _near(_HOWTO, _URL, t) or _near(_HOWTO, _EMAIL, t) or _near(_HOWTO, _PHONE, t):
        q_howto = 1.0
    elif _HOWTO.search(t):
        q_howto = 0.35  # instruction without a concrete pointer: presence != quality
    q_signup = 1.0 if _SIGNUP.search(t) else 0.0
    q_journo = 1.0 if _JOURNO.search(t) else 0.0
    q_struct = 1.0 if _TAKEAWAY.search(t) else 0.0

    nxt = (extracted or {}).get("reader_next_step", "").strip()
    if nxt and nxt.upper() != "NONE":
        if _URL.search(nxt) or _EMAIL.search(nxt) or _PHONE.search(nxt) or \
           re.search(r"register|subscribe|download|visit|call|contact|rsvp", nxt, re.I):
            q_howto = max(q_howto, 0.8)

    # concision nudge: "respect journalists' time" — legalistic 40+-word
    # sentences read as self-serving corporate copy, not audience-tailored
    q_read = _clamp((38.0 - mwps) / 12.0)
    # explicit audience naming ("guide for teachers", "helps you ...")
    q_aud = _clamp(len(_AUDIENCE.findall(body)) / 2.0)

    Q = _clamp(0.25 * q_contact + 0.18 * q_howto + 0.17 * q_assets +
               0.10 * q_signup + 0.09 * q_journo + 0.07 * q_struct +
               0.08 * q_read + 0.06 * q_aud)

    # ---------- compose ----------
    if is_release or weak_release:
        base = 0.30 if is_release else 0.25
        damp = 0.55 + 0.45 * _clamp(P / 0.55)   # chrome-dump "releases" sink
        s = damp * (base + 0.68 * Q)
    else:
        # coherent reader-addressed article/marketing plateaus near 0.2
        s = 0.24 * P + 0.05 * Q * P
    return _clamp(s)
