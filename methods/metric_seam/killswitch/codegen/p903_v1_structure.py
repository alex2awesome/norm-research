"""p903 v1 -- Corpus distinctiveness, structural/positional approach.

Criterion: the release is distinctive relative to the collection (not
near-duplicate template/boilerplate recycled across many announcements).
Single-document proxy: documents assembled from the standard wire-release
TEMPLATE SLOTS -- "FOR IMMEDIATE RELEASE" header, ALL-CAPS dateline, wire
tag (/PRNewswire/ etc.), stock ticker, "About X" section, contact block,
forward-looking-statements section, "###"/SOURCE end marker, "visit our
website" closer -- are structurally interchangeable with thousands of other
releases and score LOW. Documents whose character mass is dominated by
free-form body prose, with few or no template slots filled and little
navigation chrome, score HIGH. Position matters: boilerplate sections are
detected where templates put them (header zone / final third).

Contract: score(text) -> float in [0, 1]; deterministic; stdlib re only;
returns 0.5 on empty input or unexpected error.
"""

import re

# Mojibake / typographic normalization (longest sequences first).
_MOJIBAKE = (
    ("â€œ", '"'),    # mangled left double quote
    ("â€", '"'),    # mangled right double quote (U+009D end)
    ("â€™", "'"),    # mangled apostrophe
    ("â€“", "-"),    # mangled en dash
    ("â€”", "-"),    # mangled em dash
    ("â€¦", "..."),  # mangled ellipsis
    ("â€", '"'),          # leftover pair (invisible byte stripped)
    ("Â ", " "),          # mangled non-breaking space
    ("Â", ""),                 # stray A-circumflex from mangled nbsp
    (" ", " "),                # real non-breaking space
    ("“", '"'), ("”", '"'),
    ("‘", "'"), ("’", "'"),
    ("–", "-"), ("—", "-"),
    ("…", "..."),
    ("[...]", " "),                 # corpus elision marker, not content
)

_WIRE_RE = re.compile(
    r"prnewswire|business ?wire|globe ?newswire|accesswire|newsfile|"
    r"marketwired|einpresswire|newswire\.c", re.IGNORECASE)
_TICKER_RE = re.compile(
    r"\((?:nasdaq|nyse|otc|otcqb|otcqx|tsx|tsxv|lse|asx|amex|euronext)"
    r"[^)]{0,25}\)", re.IGNORECASE)
_FLS_RE = re.compile(
    r"forward-?looking statements|safe harbor|"
    r"actual results (?:may|could) differ|risks and uncertainties",
    re.IGNORECASE)
_ABOUT_RE = re.compile(r"\n\s*about\s+[A-Z0-9\"']", re.IGNORECASE)
_CONTACT_KW_RE = re.compile(
    r"\b(?:media|press|investor|company)\s+contacts?\b|\bcontacts?\s*:|"
    r"investor relations", re.IGNORECASE)
_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_ENDMARK_RE = re.compile(r"^\s*(?:#\s*#\s*#|###|-\s*30\s*-|source[:\s])",
                         re.IGNORECASE)
_VISIT_RE = re.compile(
    r"for (?:more|further) information|to learn more|"
    r"visit (?:https?://|www\.)", re.IGNORECASE)
_BP_HEADING_RE = re.compile(
    r"^\s*(?:about\s+\S|media contacts?\b|press contacts?\b|contacts?:|"
    r"investor relations|forward-?looking statements|safe harbor|"
    r"for (?:more|further) information|###|source[:\s])", re.IGNORECASE)

_CHROME_WORDS = frozenset((
    "home", "news", "menu", "search", "share", "tweet", "print", "email",
    "twitter", "facebook", "linkedin", "instagram", "youtube", "subscribe",
    "sign up", "sign in", "login", "log in", "register", "contact us",
    "about", "about us", "privacy policy", "terms", "terms of use",
    "cookies", "cookie policy", "back to top", "skip to content",
    "read more", "related posts", "related articles", "previous", "next",
    "categories", "tags", "archives", "advertisement", "sponsored",
))


def _normalize(t):
    for bad, good in _MOJIBAKE:
        t = t.replace(bad, good)
    return t


def _is_dateline(line):
    s = line.strip()
    m = re.match(r"^[A-Z][A-Z .,'()&-]{3,50},", s)
    if not m:
        return False
    return bool(re.search(r"\b(?:19|20)\d\d\b", s) or
                re.search(r"\s[-/]|--", s))


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        t = _normalize(text[:200000])
        lines = [ln for ln in t.split("\n") if ln.strip()]
        if not lines:
            return 0.5
        n_lines = len(lines)
        total_chars = float(sum(len(ln) for ln in lines)) or 1.0
        tlen = float(len(t)) or 1.0

        # ---- template slot census (positional) ----
        slots = 0
        head_zone = lines[:max(3, n_lines // 5)]
        # 1. release header in head zone
        if any(re.search(r"for immediate release|^\s*(?:news|press) release\b",
                         ln, re.IGNORECASE) for ln in head_zone):
            slots += 1
        # 2. ALL-CAPS dateline near the top
        if any(_is_dateline(ln) for ln in lines[:max(5, n_lines // 4)]):
            slots += 1
        # 3. wire-service tag anywhere
        if _WIRE_RE.search(t):
            slots += 1
        # 4. stock ticker
        if _TICKER_RE.search(t):
            slots += 1
        # 5. "About X" heading past the first third (char position)
        m = _ABOUT_RE.search(t)
        if m and m.start() > 0.30 * tlen:
            slots += 1
        # 6. contact block in final 40%
        tail40 = t[int(0.60 * tlen):]
        if (_CONTACT_KW_RE.search(tail40) or
                (_PHONE_RE.search(tail40) and _EMAIL_RE.search(tail40))):
            slots += 1
        # 7. forward-looking / safe-harbor language past 40%
        mf = _FLS_RE.search(t)
        if mf and mf.start() > 0.40 * tlen:
            slots += 1
        # 8. end marker in the last few lines
        if any(_ENDMARK_RE.match(ln) for ln in lines[-4:]):
            slots += 1
        # 9. "for more information / visit ..." closer in final half
        if _VISIT_RE.search(t[int(0.50 * tlen):]):
            slots += 1
        templated = min(1.0, slots / 6.0)

        # ---- body mass: how much of the document is free prose vs.
        #      recognizable template sections ----
        cut = None
        for i in range(n_lines // 2, n_lines):
            if _BP_HEADING_RE.match(lines[i].strip()):
                cut = i
                break
        boiler_chars = sum(len(ln) for ln in lines[cut:]) if cut is not None \
            else 0
        head_bp_chars = sum(
            len(ln) for ln in head_zone
            if re.search(r"for immediate release|^\s*(?:news|press) release\b",
                         ln, re.IGNORECASE) or _is_dateline(ln))
        body_frac = max(0.0, 1.0 - (boiler_chars + head_bp_chars) /
                        total_chars)

        # ---- navigation chrome (scraper leftovers = shared across pages) --
        chrome = 0
        for ln in lines:
            if len(ln.split()) <= 4:
                s = ln.strip().lower().strip(" :|>-*•·\t")
                if s in _CHROME_WORDS:
                    chrome += 1
        chrome_frac = chrome / float(n_lines)

        # ---- combine; a slot-free document that is one long mixed-case
        #      prose block is high by construction ----
        paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
        long_para = any(
            len(re.findall(r"[A-Za-z']+", p)) >= 40 and re.search(r"[a-z]", p)
            for p in paras)

        raw = (0.92 - 0.55 * templated - 0.22 * (1.0 - body_frac)
               - 0.30 * min(1.0, 2.5 * chrome_frac))
        if long_para and slots == 0:
            raw += 0.06
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
