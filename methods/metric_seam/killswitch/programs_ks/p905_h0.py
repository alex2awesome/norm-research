"""p905 h0 -- Authentic authorship, hybrid channel (code + 2 LLM fields + ops).

Criterion: the release reads as written by someone with genuine familiarity
with the company/subject (concrete, specific, internally coherent), not
assembled from generic template marketing language.

Design (residual-driven, from wave-1 train feedback):
  The v1 structural baseline's dominant failure is the BOTTOM tier: pure
  navigation chrome, dead/404 pages, listing pages, image-license pages and
  promo/offer-terms pages (judge ~0.0) receive baseline scores 0.53-0.81.
  Genuine authored bodies (judge 0.65-1.0) score 0.59-0.95, so the two
  clusters barely separate.

  h0 therefore:
    1. PRESENCE gate (code): how much of the document is connected prose
       (long, punctuated, function-word-bearing lines) vs. nav fragments.
    2. JUNK / PROMO penalties (code): template-variable markup ({{ }},
       href=, escaped \\r\\n), error-page phrases, offer/legal boilerplate
       (ALL-CAPS legal runs, "automatically renews", second-person density),
       and line-level repetition.
    3. QUALITY (code): attributed quotations, density of concrete specifics
       (money, percents, dates, scale words, tickers) in the prose, and the
       COVERAGE of distinct specific types (a proxy for "specifics spread
       through a coherent narrative").
    4. Two LLM fields for judgments regex can't reach: does a substantive
       authored body exist at all (body_gist, grounded-checked in code), and
       is that body subject-specific vs interchangeable template marketing
       (template_verdict, mapped to a score in code).
  The predicate stays in code; fields only feed graded components and the
  module degrades gracefully to code-only when fields are absent.

Contract: LLM_FIELDS dict; score(text, extracted, ops) -> float in [0, 1].
Deterministic; stdlib (re, math) + provided ops only; never raises.
"""

import re

LLM_FIELDS = {
    "body_gist": (
        "In at most 12 words, state the specific announcement, story, or "
        "subject the main body text develops in depth; answer NONE if the "
        "page is mostly navigation links, search or error pages, headline "
        "listings, or subscription/offer/legal terms."
    ),
    "template_verdict": (
        "Judge the main body prose: answer 'specific' if it gives concrete "
        "facts unique to its subject (names, numbers, events, quotes), "
        "'generic' if it is interchangeable template marketing language that "
        "could describe any company, or NONE if there is no substantive body "
        "prose."
    ),
}

# ----------------------------------------------------------------------
# Fallback normalization (used only if ops.normalize fails).
# ----------------------------------------------------------------------
_MOJIBAKE = (
    ("â€œ", '"'), ("â€", '"'),
    ("â€™", "'"), ("â€˜", "'"),
    ("â€“", "-"), ("â€”", "-"),
    ("â€¦", "..."), ("Â ", " "),
)


def _fallback_norm(text):
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    text = text.replace(" ", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    return text


# ----------------------------------------------------------------------
# Lexicons / patterns
# ----------------------------------------------------------------------
_SENT_PUNCT_RE = re.compile(r"[.!?][\"')\]]?(?:\s|$)")

_QUOTE_ATTRIB_RE = re.compile(
    r"\"[^\"\n]{15,600}\"\s*[,.]?\s*"
    r"(?:said|says|stated|added|noted|commented|explained|remarked"
    r"|according\s+to)\b",
    re.IGNORECASE)
# AP style: "...," Firstname Lastname said
_QUOTE_NAME_SAID_RE = re.compile(
    r"\"[^\"\n]{15,600}\"\s*,?\s*(?:[A-Z][\w.'-]+\s+){1,4}"
    r"(?:said|says|stated|added|noted|commented|explained|remarked)\b")
_ATTRIB_NAME_RE = re.compile(
    r"\b(?:said|says|stated|added|noted|commented|explained|remarked)\s+"
    r"(?:[A-Z][\w.'-]+|(?:CEO|CTO|CFO|President|Dr\.|Prof\.)\s+[A-Z])")
# NAME said: "..."  (blog-style attribution preceding the quote)
_NAME_SAID_QUOTE_RE = re.compile(
    r"\b[A-Z][\w.'-]+\s+(?:said|says|stated|added|noted|wrote)\s*:?\s*\"")
_PROPN_PAIR_RE = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")

_MONEY_RE = re.compile(r"[$£€]\s?\d[\d,.]*")
_PERCENT_RE = re.compile(r"\b\d[\d,.]*\s?(?:%|percent)\b", re.IGNORECASE)
_SCALE_RE = re.compile(
    r"\b\d[\d,.]*\s?(?:million|billion|trillion|bn)\b", re.IGNORECASE)
_MONTHDAY_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?"
    r"|dec(?:ember)?)\.?\s+\d{1,2}\b",
    re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_TICKER_RE = re.compile(r"\b(?:NYSE|NASDAQ|ASX|LSE)\s*:\s*[A-Z]{1,6}\b")

_JUNK_MARKUP = ("{{", "}}", "href=", "\\r\\n", "xdm:", "&gt;", "cmp-text",
                "<a ", "&quot;")
_ERROR_PHRASES = (
    "page that doesn't exist", "isn't available", "not currently available",
    "link you requested", "page you requested", "try again later",
    "unable to complete your request", "no results found", "error 404",
    "page not found")
_PROMO_PHRASES = (
    "automatically renews", "credit card required", "limited-time offer",
    "cancel anytime", "sign up now", "add to cart", "check order status",
    "promo code", "terms and conditions apply", "activate subscription",
    "request a demo you", "to continue reading you must login")
_CAPS_RUN_RE = re.compile(r"\b[A-Z][A-Z0-9 ,.'+$%/-]{45,}[A-Z.]")
_TEMPLATE_PHRASES = (
    "world's leading", "world-class", "best-in-class", "cutting-edge",
    "state-of-the-art", "proud provider", "industry-leading",
    "passionate", "one-stop", "unlock the", "empower your",
    "unparalleled", "seamless", "innovative solutions")
_SECOND_PERSON_RE = re.compile(r"\byou(?:r|'re|rs)?\b", re.IGNORECASE)

_GIST_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "about", "their",
    "into", "over", "none", "page", "text", "body", "main", "announcement",
    "story", "subject", "company", "release", "press", "news", "website",
}


def _sat(x):
    """Clamp to [0, 1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


_CAPWORD_RE = re.compile(r"^[A-Z0-9]")
_CHUNK_END_RE = re.compile(r"[.!?][\"')\]]?$")

# site-furniture sentences: DROPPED from prose (not a penalty on the doc)
_BOILER_CHUNK = (
    "skip to main content", ".gov website", "secure .gov websites",
    "official website of the united states government",
    "share sensitive information", "(press enter)",
    "use left and right arrow keys", "enable javascript", "your browser",
    "all rights reserved", "terms of use", "privacy policy",
    "cookie", "subscribe to our", "sign up for our newsletter",
    "follow us on")


def _prose_split(norm):
    """Rebuild paragraphs from (possibly hard-wrapped) lines, drop nav
    chrome, and count words living in real sentences.

    Returns (prose_text, prose_words, total_words, distinct_line_ratio).
    """
    lines = [ln.strip() for ln in norm.splitlines()]
    lines = [ln for ln in lines if ln]
    total_words = 0
    content_parts = []
    seen = set()
    n_lines = 0
    for ln in lines:
        words = ln.split()
        nw = len(words)
        total_words += nw
        n_lines += 1
        seen.add(ln)
        has_sent_punct = bool(_SENT_PUNCT_RE.search(ln))
        # chrome: very short punct-less fragments, or short Title-Case
        # nav/headline lines with no sentence punctuation
        if nw <= 4 and not has_sent_punct:
            continue
        if nw <= 10 and not has_sent_punct:
            caps = sum(1 for w in words if _CAPWORD_RE.match(w))
            if caps >= 0.6 * nw:
                continue
        content_parts.append(ln)
    if total_words == 0:
        return "", 0, 0, 1.0
    distinct = (len(seen) / n_lines) if n_lines else 1.0
    # merge surviving lines (fixes hard-wrapped paragraphs), then keep only
    # sentence-shaped chunks: 6-130 words ending in sentence punctuation.
    stream = " ".join(content_parts)
    chunks = re.split(r"(?<=[.!?])\s+", stream)
    kept = []
    prose_words = 0
    for ch in chunks:
        words = ch.split()
        nw = len(words)
        if 6 <= nw <= 130 and _CHUNK_END_RE.search(ch):
            cl = ch.lower()
            if any(b in cl for b in _BOILER_CHUNK):
                continue
            # cap-dominated chunks are glued headline/nav/table runs,
            # not connected prose
            caps = sum(1 for w in words if _CAPWORD_RE.match(w))
            if caps > 0.55 * nw:
                continue
            kept.append(ch)
            prose_words += nw
    return " ".join(kept), prose_words, total_words, distinct


def _count_hits(haystack_lower, phrases):
    n = 0
    for p in phrases:
        n += haystack_lower.count(p)
    return n


def score(text, extracted, ops):
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.05
        try:
            norm = ops.normalize(text)
            if not isinstance(norm, str) or not norm:
                norm = text
        except Exception:
            norm = text
        # idempotent second pass: guarantees curly quotes/mojibake are fixed
        # even if ops.normalize is weaker than expected
        norm = _fallback_norm(norm)

        prose, p_words, t_words, distinct = _prose_split(norm)
        low_all = norm.lower()
        prose_lower = prose.lower()

        # ---------------- presence ----------------
        prose_frac = (p_words / t_words) if t_words else 0.0
        presence = _sat(p_words / 150.0) * (0.3 + 0.7 * _sat(prose_frac / 0.5))

        # ---------------- junk / error / repetition ----------------
        # markup junk is counted in the FRONT of the doc only: contact and
        # newswire boilerplate (which legitimately contains a stray <a href>)
        # sits near the END per corpus hazards, while template/scrape junk
        # contaminates the body itself.
        front = low_all[: max(200, int(0.65 * len(low_all)))]
        junk_hits = _count_hits(front, tuple(m.lower() for m in _JUNK_MARKUP))
        err_hits = _count_hits(low_all, _ERROR_PHRASES)
        rep_pen = 0.0
        if t_words >= 80:
            rep_pen = _sat((0.80 - distinct) / 0.80)
        junk = _sat(0.5 * _sat(junk_hits / 6.0)
                    + 0.35 * _sat(err_hits / 2.0)
                    + 0.15 * rep_pen)

        # ---------------- promo / legal boilerplate ----------------
        promo_hits = _count_hits(low_all, _PROMO_PHRASES)
        caps_chars = sum(len(m) for m in _CAPS_RUN_RE.findall(norm))
        caps_pen = _sat((caps_chars - 100.0) / 350.0)
        you_density = 0.0
        if p_words >= 40:
            you_density = len(_SECOND_PERSON_RE.findall(prose)) / float(p_words)
        you_pen = _sat((you_density - 0.012) / 0.030)
        promo = _sat(0.5 * _sat(promo_hits / 3.0)
                     + 0.3 * caps_pen
                     + 0.2 * you_pen)

        content_code = presence * (1.0 - junk) * (1.0 - 0.8 * promo)

        # ---------------- graded quality ----------------
        n_quotes = len(_QUOTE_ATTRIB_RE.findall(prose)) \
            + len(_QUOTE_NAME_SAID_RE.findall(prose)) \
            + len(_NAME_SAID_QUOTE_RE.findall(prose)) \
            + len(_ATTRIB_NAME_RE.findall(prose))
        quote_sig = _sat(n_quotes / 2.0)

        # specifics inside the prose only (decouple presence from quality)
        dates_n = 0
        try:
            d = ops.extract_dates(prose)
            if d:
                dates_n = len(d)
        except Exception:
            dates_n = 0
        if dates_n == 0:
            dates_n = len(_MONTHDAY_RE.findall(prose))
        counts = {
            "money": len(_MONEY_RE.findall(prose)),
            "pct": len(_PERCENT_RE.findall(prose)),
            "scale": len(_SCALE_RE.findall(prose)),
            "date": dates_n,
            "year": len(_YEAR_RE.findall(prose)),
            "ticker": len(_TICKER_RE.findall(prose)),
        }
        n_spec = sum(counts.values())
        spec_density = (100.0 * n_spec / p_words) if p_words else 0.0
        spec_sig = _sat(spec_density / 2.5)
        type_spread = _sat(sum(1 for v in counts.values() if v > 0) / 4.0)

        # named-entity-ish density: adjacent Capitalized-word pairs in prose
        propn_density = 0.0
        if p_words:
            propn_density = 100.0 * len(_PROPN_PAIR_RE.findall(prose)) / p_words
        propn_sig = _sat(propn_density / 4.0)

        quality_code = (0.35 * quote_sig
                        + 0.30 * spec_sig
                        + 0.20 * type_spread
                        + 0.15 * propn_sig)

        # generic template marketing: penalize ONLY when specifics are thin
        # (keyword presence must not proxy for quality on rich releases)
        t_hits = _count_hits(prose_lower, _TEMPLATE_PHRASES)
        template_pen = _sat(t_hits / 3.0) * (1.0 - spec_sig) * (1.0 - quote_sig)
        quality_code *= (1.0 - 0.6 * template_pen)

        # ---------------- LLM fields (graceful when absent) ----------------
        extracted = extracted if isinstance(extracted, dict) else {}

        def _field(name):
            v = extracted.get(name)
            if not isinstance(v, str):
                return None
            v = v.strip()
            if v.lower() in ("none", "n/a", "na", "null", "no", "-"):
                return ""
            return v

        gist = _field("body_gist")
        verdict = _field("template_verdict")

        content = content_code
        if gist is not None:
            if gist == "":
                g = 0.0
            else:
                toks = [w for w in re.findall(r"[a-z0-9]{4,}", gist.lower())
                        if w not in _GIST_STOP]
                grounded = any(w in low_all for w in toks) if toks else False
                g = 1.0 if grounded else 0.6
            # ground the gist in observed prose so a hallucinated gist on a
            # chrome page cannot rescue it
            g_eff = g * (0.5 + 0.5 * presence)
            content = 0.55 * content_code + 0.45 * g_eff

        quality = quality_code
        if verdict is not None:
            vl = verdict.lower()
            if vl == "":
                v_sig = 0.05
            elif "specific" in vl:
                v_sig = 1.0
            elif "generic" in vl:
                v_sig = 0.35
            else:
                v_sig = 0.5
            quality = 0.6 * quality_code + 0.4 * v_sig

        final = content * (0.25 + 0.75 * quality)
        return _sat(final)
    except Exception:
        return 0.5
