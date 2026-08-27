"""a105 hybrid: Plain language / minimal jargon & cliches.

Two-stage design matching the judge's observed behavior on TRAIN:
  Stage 1 (gate): is there a real prose document here at all? Scraped
    chrome (nav menus, ticker pages, template junk, web-UI instructional
    text) gets gated toward 0; the judge scores those 0.0-0.15.
    Gate signals: absolute prose mass, prose/total ratio, and density of
    web-UI terms INSIDE the prose (cookies/settings/click/browser...),
    damped when the doc is overwhelmingly prose (a real FAQ vs a widget).
  Stage 2 (plainness): on the prose subset, penalize insider signals:
    corporate buzzwords/cliches, legalese boilerplate, insider-technical
    vocabulary (earnings/finance/clinical), unique undefined acronyms
    (full text -- catches ticker spam), trademark-mark spam ((r)/(tm)),
    very long sentences, and (English-only) long-word fraction.
  score = gate * (0.32 + 0.63 * plainness) -> reproduces the judge's
    ~0.0 / ~0.4 / ~0.9 band centers (chrome / jargony prose / plain prose).

All features are ratio-based (per 100 words) since the corpus mixes
non-releases of wildly different lengths. Works with extracted == {};
the two LLM fields sharpen exactly the two stages when present.
"""
import re

LLM_FIELDS = {
    "jargon_terms": "List up to 5 insider buzzwords, cliches, or undefined technical terms in this document, or NONE",
    "content_type": "Is the main content readable prose (article/press release/blog) or mostly site navigation/menus/listings? Answer PROSE or NAV",
}

_BUZZWORDS = [
    "synergy", "synergies", "leverage", "leveraging", "best-in-class",
    "best in class", "cutting-edge", "cutting edge", "world-class",
    "world class", "state-of-the-art", "state of the art", "seamless",
    "seamlessly", "robust", "innovative", "innovation", "solutions",
    "ecosystem", "paradigm", "holistic", "empower", "empowers",
    "empowering", "unlock", "game-changing", "game changer",
    "next-generation", "next-gen", "industry-leading", "industry leading",
    "market-leading", "world's leading", "world's first", "world's most",
    "leading provider", "leading global", "mission-critical",
    "value-added", "win-win", "thought leader", "thought leadership",
    "disruptive", "transformative", "transformational", "revolutionize",
    "revolutionary", "groundbreaking", "ground-breaking",
    "first-of-its-kind", "one-of-a-kind", "unparalleled", "unrivaled",
    "unmatched", "visionary", "turnkey", "end-to-end", "scalable",
    "streamline", "streamlined", "actionable", "deliverables",
    "incentivize", "operationalize", "core competencies",
    "best practices", "stakeholders", "footprint",
    "iconic", "legacy brand", "powerhouse", "bespoke", "epitome",
    "plethora", "renowned", "award-winning", "flagship", "premier",
    "excellence", "passionate", "committed to", "dedicated to",
    "prominent", "proud to announce", "pleased to announce",
    "excited to announce", "excited to", "thrilled to", "delighted to",
    "poised to", "well-positioned", "at the forefront", "forefront of",
    "pave the way", "paving the way", "peace of mind",
    "at your fingertips", "must-have", "top-notch", "cult-following",
    "forward-thinking", "barrier breaking", "breaking barriers",
    "capabilities", "warfighter", "warfighters", "advanced technologies",
    "advanced technology", "enhanced ability", "operational excellence",
    "drive growth", "deliver value", "hidden gem", "precision-guided",
]

_LEGALESE = [
    "terms and conditions", "restrictions", "limitations", "exclusions",
    "applicable", "liability", "liable", "indemnification", "pursuant",
    "herein", "hereby", "thereof", "aforementioned", "notwithstanding",
    "disclaimer", "warranty", "warranties", "subject to", "provisions",
    "forward-looking statements", "risks and uncertainties",
    "securities and exchange commission",
]

# insider-technical vocabulary (finance / earnings / clinical) used
# without definition -- reads as jargon to a general audience
_INSIDER = [
    "assets under management", "annualized", "ex-dividend",
    "basis points", "bookrunner", "subordinated notes", "green bonds",
    "holdings", "credit rating", "credit ratings", "restructuring",
    "operating cash flow", "profit per share", "efficacy",
    "prophylaxis", "protease", "clinical trial", "clinical trials",
    "outlook",
]

# web-UI / site-chrome terms: when these live INSIDE the prose, the
# "document" is a widget/settings/FAQ page, not an announcement
_CHROME_UI = [
    "click here", "click", "cookie", "cookies", "browser", "javascript",
    "sign in", "log in", "login", "register", "subscribe",
    "subscription", "password", "email", "e-mail", "settings",
    "default setting", "download", "website", "web site", "homepage",
    "site map", "sitemap", "privacy policy", "terms of use",
    "all rights reserved", "newsletter", "ad blocker", "www.", ".com",
    "embed", "wi-fi", "wifi", "device", "devices", "hotline",
    "toll-free", "faq",
]

_ACRO_WHITELIST = {
    "USA", "CEO", "CFO", "CTO", "COO", "CIO", "NYSE", "NASDAQ", "FAQ",
    "RSS", "PDF", "GMT", "EST", "EDT", "PST", "PDT", "LLC", "INC", "LTD",
    "PLC", "USD", "GBP", "EUR", "AED", "URL", "HTML", "WWW", "COM",
    "THE", "AND", "FOR", "ALL", "NEW", "NOW", "GET", "YOUR", "HERE",
    "CLICK", "MORE", "HOME", "NEWS", "ABOUT", "CONTACT", "SEARCH",
    "READ", "VIEW", "SIGN", "OUT", "JOIN", "SHOP", "FULL", "SITE",
    "SEND", "TIPS", "CLOSE", "ACCEPT", "LEARN", "NOT", "MAY", "CAN",
    "WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW", "FROM", "WITH",
}

_WORD_RE = re.compile(r"[A-Za-z']+")
_ACRO_RE = re.compile(r"\b[A-Z]{3,8}\b")
_TM_RE = re.compile(r"\((?:r|tm|sm)\)|[®™]", re.I)
_ROMAN_RE = re.compile(r"\((?:i{1,3}|iv|v|vi{0,3})\)")
_ENG_RE = re.compile(r"\b(?:the|and|of|to|in|is|that|for|with|are)\b")

# imperative CTA verbs opening a sentence = marketing/UI chrome, not prose
_CTA_VERBS = {
    "try", "search", "access", "subscribe", "contact", "register",
    "click", "learn", "find", "enter", "visit", "sign", "get", "stay",
    "read", "view", "download", "browse", "join", "follow", "share",
    "save", "select", "choose", "call", "request", "submit", "check",
    "explore", "discover", "shop", "order", "buy", "book", "watch",
    "please",
}


def _lex_re(terms):
    return re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in
                            sorted(terms, key=len, reverse=True)) + r")\b")

_BUZZ_RE = _lex_re(_BUZZWORDS)
_LEGAL_RE = _lex_re(_LEGALESE)
_INSIDER_RE = _lex_re(_INSIDER)
_CHROME_RE = _lex_re([t for t in _CHROME_UI if t not in ("www.", ".com")])
_URL_RE = re.compile(r"www\.|\.com\b")


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _prose_lines(text):
    """Keep lines that look like running prose rather than nav chrome:
    >=8 word tokens and >=50% of alphabetic tokens fully lowercase
    (language-agnostic; Title-Case menus and ticker lists fail).
    Cookie-consent banners and {{template}} lines are dropped entirely:
    they appear on good and bad pages alike and carry no signal."""
    prose, total_words = [], 0
    for raw in text.split("\n"):
        words = _WORD_RE.findall(raw)
        total_words += len(words)
        if len(words) < 8:
            continue
        if "cookie" in raw.lower() or "{{" in raw:
            # drop only the offending sentences, not the whole line
            # (scrapes are often one giant line)
            kept = [s for s in re.split(r"(?<=[.!?])\s+", raw)
                    if "cookie" not in s.lower() and "{{" not in s]
            raw = " ".join(kept)
            words = _WORD_RE.findall(raw)
            if len(words) < 8:
                continue
        lower = sum(1 for w in words if w.islower())
        if lower / len(words) >= 0.5:
            prose.append(raw.strip())
    return prose, total_words


def _cta_fraction(prose):
    """Fraction of sentence-ish fragments opening with an imperative CTA
    verb ('Try our calculator', 'Enter up to 25 symbols', 'Please refer
    to...') -- signature of marketing/UI chrome rather than content."""
    frags = [f for f in re.split(r"[.!?\n]+", "\n".join(prose))
             if len(_WORD_RE.findall(f)) >= 4]
    if not frags:
        return 0.0
    cta = sum(1 for f in frags
              if (_WORD_RE.findall(f) or [""])[0].lower() in _CTA_VERBS)
    return cta / len(frags)


def _undefined_acronyms(text):
    """Unique ALL-CAPS tokens (3-8 letters) never expanded; a token seen
    adjacent to a paren -- '(NAA)' or 'WLAN (wireless local...)' -- counts
    as defined and is excluded."""
    seen, defined = set(), set()
    for m in _ACRO_RE.finditer(text):
        tok = m.group(0)
        if tok in _ACRO_WHITELIST:
            continue
        seen.add(tok)
        if "(" in text[max(0, m.start() - 2):m.start()] or \
           "(" in text[m.end():m.end() + 2]:
            defined.add(tok)
    return seen - defined


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        text = ops.normalize(text)
        prose, total_words = _prose_lines(text)
        prose_text = " ".join(prose)
        n_pw = len(_WORD_RE.findall(prose_text))
        low = prose_text.lower()
        per100 = 100.0 / max(1, n_pw)

        # ---- Stage 1: prose gate ----
        pr = n_pw / max(1, total_words)
        gate = min(_clamp((n_pw - 60) / 180.0), _clamp(pr / 0.40))

        # web-UI vocabulary inside the prose, damped when the doc is
        # overwhelmingly prose and protected by dateline density (real
        # articles/releases carry dates in their prose; widgets don't)
        chrome = (len(_CHROME_RE.findall(low)) +
                  len(_URL_RE.findall(low))) * per100
        damp = 1.0 - 0.5 * _clamp((pr - 0.55) / 0.35)
        dates = len(ops.extract_dates(prose_text)) * per100
        protect = 1.0 - 0.6 * _clamp(dates / 1.2)
        gate *= 1.0 - 0.75 * _clamp((chrome - 1.2) / 2.5) * damp * protect

        # imperative CTA-sentence share (marketing/UI chrome)
        gate *= 1.0 - 0.6 * _clamp((_cta_fraction(prose) - 0.12) / 0.30)

        # {{template}} junk = broken scrape, never a release
        gate *= 1.0 - 0.8 * _clamp(text.count("{{") / 8.0)

        # NOTE (hygiene patch): content_type is a single-category enum answer
        # (PROSE/NAV). Bare "NAV" in ct false-fired on unrelated words
        # containing the token as a substring ("navigation", "coronavirus",
        # "navigate", "ritonavir", etc.). \b-anchored.
        ct = str((extracted or {}).get("content_type", "")).strip().upper()
        if ct:
            if re.search(r"\bNAV\b", ct) and "PROSE" not in ct:
                gate = min(gate, 0.3)
            elif "PROSE" in ct:
                gate = max(gate, 0.6)

        if gate <= 0.03:
            return 0.02

        # ---- Stage 2: plainness of the prose ----
        buzz = len(_BUZZ_RE.findall(low)) * per100
        legal = (len(_LEGAL_RE.findall(low)) +
                 len(_ROMAN_RE.findall(low))) * per100
        insider = len(_INSIDER_RE.findall(low)) * per100
        acro = len(_undefined_acronyms(text)) * 100.0 / max(1, total_words)
        tm = len(_TM_RE.findall(prose_text)) * per100
        _, mws, flw = ops.sent_stats(prose_text)
        is_english = len(_ENG_RE.findall(low)) * per100 >= 6.0

        penalty = (0.45 * _clamp((buzz - 0.45) / 1.6) +
                   0.30 * _clamp((legal - 0.35) / 1.0) +
                   0.35 * _clamp(insider / 1.0) +
                   0.35 * _clamp((acro - 1.6) / 1.5) +
                   0.30 * _clamp(tm / 1.2) +
                   0.30 * _clamp((mws - 28.0) / 18.0))
        if is_english:
            penalty += 0.10 * _clamp((flw - 0.15) / 0.12)

        # LLM-grounded jargon examples (thick input regex can't see)
        jt = str((extracted or {}).get("jargon_terms", "")).strip()
        if jt and jt.upper() != "NONE":
            n_terms = len([p for p in re.split(r"[,;\n]", jt) if p.strip()])
            penalty += 0.15 * _clamp(n_terms / 5.0)

        plainness = 1.0 - _clamp(penalty)
        return _clamp(gate * (0.32 + 0.63 * plainness))
    except Exception:
        return 0.5
