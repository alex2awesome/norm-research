"""a103 hybrid: Clarity, concision, and newsroom style.

Judge behavior on TRAIN falls into bands:
  ~0.0   nav-chrome pages, article indexes, foreign-language leaflets,
         news-aggregator/blog chrome — no release, nothing to be clear about
  ~0.25  real prose that is NOT tight newsroom copy: marketing pages, blog
         posts, analyst commentary — AND real releases written badly
         (shouty multi-clause ALL-CAPS headlines, 60-word lead sentences,
         quote-dominated advocacy copy, rhetorical questions)
  ~0.6+  clean corporate releases: dateline, main point in the first
         sentence, facts/figures early, moderate sentence length, plain
         words; the tighter and more front-loaded, the higher (0.6 -> 0.9)

Design: score = gate * (0.06 + 0.94 * quality)
  gate    = is there a release/document to judge? Release markers (newswire
            tags, FOR IMMEDIATE RELEASE, dateline, ticker, contact/SOURCE)
            or generic prose-mass doc-ness (scaled down, so a blog can reach
            the 0.25 band but not the release band). Damped by nav-chrome
            dominance, blog/index anti-markers, and non-English body (the
            judge scores foreign leaflets ~0 on newsroom style).
  quality = MEASURED style, not keyword presence (robust to counterfactual
            keyword injection): lead directness is only credited inside the
            first two body sentences AFTER the located release anchor and is
            coupled to factual content (digits); everything else is a
            penalty family that cannot be gamed by inserting good words —
            sentence economy (mean words/sentence), lead-sentence length,
            dense-copy long-word fraction, wordy-phrase count, ALL-CAPS
            shouty headline runs, rhetorical ?/!, and quote dominance.

Scrape robustness: hard-wrapped (PDF/BusinessWire) lines are unwrapped
before sentence stats; run-on "bill.The" scrape joins are re-split; body
stats are computed only on prose-like lines outside detected nav runs.
Works with extracted == {}; the two LLM fields sharpen exactly the two
stages (doc kind -> gate, lead style -> quality) when present.
"""
import re

LLM_FIELDS = {
    "doc_kind": ("Is this one press release/announcement (not a nav page, "
                 "article index, blog, news article, or product page)? "
                 "Answer RELEASE, ARTICLE, or NAV"),
    "lead_style": ("Does the first body sentence state the main news "
                   "plainly in one tight sentence? Answer DIRECT, WORDY, "
                   "BURIED, or NONE"),
}

_WORD_RE = re.compile(r"[A-Za-z0-9$%']+")
_ALPHA_RE = re.compile(r"[A-Za-z']+")

# ---- release markers (gate only; never the quality signal) ----
_NEWSWIRE_RE = re.compile(
    r"PRNewswire|PR Newswire|BUSINESS WIRE|Business Wire|GLOBE NEWSWIRE|"
    r"Globe Newswire|Marketwired|ACCESSWIRE|ACCESS Newswire")
_IMMEDIATE_RE = re.compile(r"FOR IMMEDIATE RELEASE|\bPRESS RELEASE\b", re.I)
_DATELINE_RE = re.compile(
    r"\b[A-Z]{3,}[A-Za-z., ]{0,30}[-,(]\s*"
    r"(?:[A-Z][a-z]{2,8}\.?,? \d{1,2},? \d{4}|\d{1,2} [A-Z][a-z]{2,8},? \d{4})"
    r"|\b[A-Z]{3,}[A-Z .]{0,20},\s+[A-Z][a-z]+\.?\s*[-–—]{1,2}\s")
# mixed-case dateline: 'Washington, D.C., March 7, 2013' / 'Egham, UK, August 12, 2009'
_DATELINE2_RE = re.compile(
    r"\b[A-Z][a-zA-Z]{2,12}(?:,\s+[A-Z][.a-zA-Z]{1,14}){1,2},\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2},?\s+\d{4}"
    # European caps: 'PRESS RELEASE 11 FEBRUARY 2013'
    r"|\b\d{1,2}\s+(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|"
    r"SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{4}")
_ANNOUNCE_RE = re.compile(
    r"\btoday (?:announce[ds]?|report(?:ed|s)?|launch(?:ed|es)?|introduced|"
    r"released|issued|unveiled|named|filed|proposed|completed)\b|"
    r"\bannounc(?:es|ed|ing) (?:today|its|the|plans)\b|"
    r"\b(?:announced|reported|launched|unveiled|proposed) today\b", re.I)
_TICKER_RE = re.compile(
    r"\((?:NYSE|NASDAQ|Nasdaq|OTC|TSX|LSE|AMEX)[ A-Za-z]*:\s*[A-Z]{1,6}\b"
    r"|\[(?:NYSE|NASDAQ)[ A-Za-z]*:\s*[A-Z]{1,6}\]")
_SAID_RE = re.compile(
    r"[,\"] ?(?:said|says) [A-Z]|\" ?(?:said|says)|(?:said|says) [A-Z][a-z]+ [A-Z]")
_CONTACT_RE = re.compile(
    r"\b(?:Media Contact|Press Contact|Media Inquiries|Press Office|"
    r"Media Hotline|Press inquiries|For further information|SOURCE [A-Z][A-Za-z]|###)")
_ABOUT_RE = re.compile(r"\bAbout [A-Z][A-Za-z&]")

# blog / index / product-page anti-markers
_ANTI_RE = re.compile(
    r"frequently asked questions|Read [Mm]ore|Learn [Mm]ore|Posted In:|"
    r"View all news|View All\b|Older Posts|Related News|Top Stories|"
    r"Continue reading|Click here|Blog Post|Subscribe to|Sign Up|"
    r"Next Article|Previous Article", re.I)

# lead verbs credited ONLY in the first two anchored body sentences,
# and only alongside factual digits (injection-resistant coupling)
_LEAD_VERB_RE = re.compile(
    r"\b(?:announce[ds]?|report(?:ed|s)|launch(?:ed|es)|introduc(?:ed|es)|"
    r"unveil(?:ed|s)|complet(?:ed|es)|propos(?:ed|es)|sign(?:ed|s)|"
    r"acquir(?:ed|es)|purchas(?:ed|es)|bought|totall?ed|declin(?:ed|es)|"
    r"grew|rose|fell|will begin|has entered|agreed to|according to|"
    r"issued|filed|named|elect(?:ed|s)|award(?:ed|s)|deliver(?:ed|s))\b", re.I)

# wordy / bureaucratic phrasing (distinct-hit penalty; from style guides)
_WORDY = [
    r"\bin order to\b", r"\bdue to the fact that\b", r"\bat this point in time\b",
    r"\bfor the purpose of\b", r"\bin the event that\b", r"\bwith regard to\b",
    r"\bin a (?:position|manner|fashion)\b", r"\bbring to the table\b",
    r"\butili[sz]e[ds]?\b", r"\bleverag(?:e|es|ed|ing)\b", r"\bsynergy\b",
    r"\bvalue[- ]add(?:ed)?\b", r"\bbest[- ]practice[s]?\b",
    r"\bon the heels of\b", r"\bit (?:is|should be) noted that\b",
    r"\bfirst[- ]of[- ]its[- ]kind\b", r"\bgoes without saying\b",
    r"\bneedless to say\b", r"\beach and every\b", r"\blast but not least\b",
]
_WORDY_RE = [re.compile(p, re.I) for p in _WORDY]

# hard promotional hype (mild penalty; plain newsroom copy avoids these)
_HYPE_RE = re.compile(
    r"\brevolutionary\b|\bgame[- ]chang(?:er|ing)\b|\bworld[- ]class\b|"
    r"\bcutting[- ]edge\b|\bstate[- ]of[- ]the[- ]art\b|\bunparalleled\b|"
    r"\bbest[- ]in[- ]class\b|\bincredibl[ye]\b|\bamazing\b|\bexciting\b|"
    r"\bmost compelling\b|\bwhopping\b", re.I)

_ENG_FN_RE = re.compile(r"\b(?:the|and|of|to|in|for|with|that|is|are)\b", re.I)
# scrape furniture excluded from BODY STATS only (never from the gate):
# cookie/browser banners and securities-law boilerplate — the judge does not
# grade releases down for standard legal tails, and cookie banners must not
# become the "lead sentence"
_BOILER_RE = re.compile(
    r"we use cookies|use of cookies|uses cookies|consent to (?:our|the) use|"
    r"no longer support this browser|supported browser|update your browser|"
    r"forward[- ]looking statements?|safe harbor|private securities litigation|"
    r"actual results (?:may|could|to) differ|undue reliance|"
    r"risks(?:,| and) uncertainties|all rights reserved|terms of use|"
    r"privacy (?:policy|statement)|informational purposes only", re.I)
_CAPS_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Z0-9,;:.'&?!$%-]*$")
_QUOTE_SPAN_RE = re.compile(r"\"[^\"]{20,600}\"")
_RUNON_FIX_RE = re.compile(r"([a-z])\.([A-Z])")
_WRAP_END_RE = re.compile(r"[a-z,]$")
_WRAP_START_RE = re.compile(r"^[a-z0-9]")


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _unwrap(text):
    """Re-join hard-wrapped lines (line ends lowercase/comma, next starts
    lowercase/digit): wrap artifacts, not layout."""
    raw = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out = []
    for ln in raw:
        if (out and _WRAP_END_RE.search(out[-1]) and _WRAP_START_RE.match(ln)
                and len(_WORD_RE.findall(out[-1])) >= 3):
            out[-1] = out[-1] + " " + ln
        else:
            out.append(ln)
    return out


def _nav_mask(lines):
    """Nav chrome = runs of >=5 consecutive short number-free lines, plus
    short lines whose exact text repeats 3+ times."""
    wcs = [len(_WORD_RE.findall(ln)) for ln in lines]
    counts = {}
    for ln, wc in zip(lines, wcs):
        if wc <= 6:
            counts[ln] = counts.get(ln, 0) + 1
    navish = [wcs[i] <= 6 and not re.search(r"\d+\.\d|\d+%|\$\s?\d", lines[i])
              for i in range(len(lines))]
    in_nav = [navish[i] and counts.get(lines[i], 0) >= 3
              for i in range(len(lines))]
    i = 0
    while i < len(lines):
        if navish[i]:
            j = i
            while j < len(lines) and navish[j]:
                j += 1
            if j - i >= 5:
                for k in range(i, j):
                    in_nav[k] = True
            i = j
        else:
            i += 1
    return wcs, in_nav


# abbreviations that must not end a sentence ('U.S. Air Force' etc.)
_ABBR_RE = re.compile(
    r"\b(U\.S|U\.K|D\.C|E\.U|Inc|Corp|Co|Ltd|LLC|Mr|Mrs|Ms|Dr|Jr|Sr|St|"
    r"Sen|Gov|Rep|Gen|Col|Maj|Sgt|Prof|Rev|Hon|vs|approx|No|Nos|Fig)\.")
# scrape glue: headline/dateline fused onto the lead sentence
_GLUE_RES = [
    (re.compile(r"/\s*PR\s?Newswire\s*/\s*[-–—]*\s*"), ". "),
    (re.compile(r"\(\s*BUSINESS WIRE\s*\)\s*[-–—]*\s*"), ". "),
    (re.compile(r"\bFOR IMMEDIATE RELEASE\b"), ". FOR IMMEDIATE RELEASE"),
    (re.compile(r"(\d{4})\s*[-–—]{1,2}\s+(?=[A-Z(\"])"), r"\1. "),
]


def _sentences(prose):
    prose = _RUNON_FIX_RE.sub(r"\1. \2", prose)
    for rx, rep in _GLUE_RES:
        prose = rx.sub(rep, prose)
    prose = _ABBR_RE.sub(lambda m: m.group(0).replace(".", "․"), prose)
    sents = [s.strip().replace("․", ".")
             for s in re.split(r"(?<=[.!?])\s+", prose)]
    return [s for s in sents
            if len(_ALPHA_RE.findall(s)) >= 4 and not _BOILER_RE.search(s)]


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        text = ops.normalize(text)
        lines = _unwrap(text)
        if not lines:
            return 0.0
        wcs, in_nav = _nav_mask(lines)
        n_lines = len(lines)
        total_words = sum(wcs) or 1

        # ---------- locate release body (skip chrome; anchor at release) ----
        anchor = None
        for i, ln in enumerate(lines):
            if (_NEWSWIRE_RE.search(ln) or _IMMEDIATE_RE.search(ln)
                    or _DATELINE_RE.search(ln) or _DATELINE2_RE.search(ln)):
                # a marker anchor is only valid if real prose follows it
                # (footer 'Press Release' tags must not swallow the body)
                after = sum(wcs[j] for j in range(i, n_lines) if not in_nav[j])
                if after >= 100:
                    anchor = i
                break
        if anchor is None:
            anchor = 0
            for i, ln in enumerate(lines):
                if (not in_nav[i] and wcs[i] >= 25 and "." in ln
                        and not _BOILER_RE.search(lines[i])):
                    anchor = i
                    break

        body_lines = [lines[i] for i in range(anchor, n_lines)
                      if not in_nav[i] and wcs[i] >= 12
                      and re.search(r"[.!?”\"]", lines[i])]
        sents = _sentences(" ".join(body_lines))
        body = " ".join(sents)          # boilerplate-free prose for stats
        body_words = _ALPHA_RE.findall(body)
        prose_words = len(body_words)

        # ---------- gate: release-likeness / document-ness ----------
        has_dateline = bool(_DATELINE_RE.search(text) or _DATELINE2_RE.search(text))
        pts = (2.0 * bool(_NEWSWIRE_RE.search(text)) +
               1.5 * bool(_IMMEDIATE_RE.search(text)) +
               1.5 * has_dateline +
               1.0 * bool(_ANNOUNCE_RE.search(text)) +
               0.75 * bool(_TICKER_RE.search(text)) +
               0.75 * _clamp(len(_SAID_RE.findall(text)) / 2.0) +
               0.5 * bool(_CONTACT_RE.search(text)) +
               0.5 * bool(_ABOUT_RE.search(text)))
        release = _clamp(pts / 3.0)

        doc_like = min(_clamp((prose_words - 80) / 250.0),
                       _clamp((prose_words / total_words) / 0.45))
        gate = max(release, 0.55 * doc_like)   # prose alone caps at 0.55

        # non-English body: newsroom-style judgment mostly collapses
        if prose_words >= 60:
            eng = len(_ENG_FN_RE.findall(body)) / prose_words
            if eng < 0.10:
                gate *= 0.35 + 0.65 * _clamp(eng / 0.10)

        # nav-chrome damping, relaxed for marker-confirmed releases and for
        # pages whose body prose is substantial despite the chrome
        nav_ratio = sum(1 for v in in_nav if v) / n_lines
        gate *= 1.0 - (0.6 * _clamp((nav_ratio - 0.4) / 0.4)
                       * (1.0 - 0.75 * release)
                       * (1.0 - 0.5 * _clamp((prose_words - 150) / 300.0)))
        if release < 0.6:
            gate *= 1.0 - 0.5 * _clamp(len(_ANTI_RE.findall(text)) / 6.0)
        if not sents:
            gate = min(gate, 0.15)

        # ---------- quality: measured newsroom style ----------
        q = 0.45

        # lead directness: anchored first two sentences, verb + factual digit
        lead = sents[:2]
        lead_txt = " ".join(lead)
        lead_direct = 0.0
        if lead:
            has_verb = bool(_LEAD_VERB_RE.search(lead_txt))
            has_fact = bool(re.search(r"\d", lead_txt))
            lead_direct = (0.6 * has_verb + 0.4 * has_fact)
            if has_verb and has_dateline:
                lead_direct = min(1.0, lead_direct + 0.2)
            # leading with a quotation is a hard newsroom-style violation
            if sents[0].lstrip(",;:- ").startswith(('"', "“")):
                lead_direct = 0.0
                q -= 0.10
        q += 0.22 * lead_direct

        # sentence economy (mean words/sentence over the body)
        if sents:
            slens = [len(_ALPHA_RE.findall(s)) for s in sents]
            mws = sum(slens) / len(slens)
            q -= 0.22 * _clamp((mws - 27.0) / 22.0)
            # lead-sentence bloat (the 60-word advocacy lead)
            q -= 0.18 * _clamp((slens[0] - 38.0) / 32.0)

        # dense copy: long-word fraction of the body
        if body_words:
            flw = sum(1 for w in body_words if len(w) >= 9) / len(body_words)
            q -= 0.12 * _clamp((flw - 0.18) / 0.12)

        # wordy phrasing (distinct hits)
        n_wordy = sum(1 for rx in _WORDY_RE if rx.search(body or text))
        q -= 0.15 * _clamp(n_wordy / 4.0)

        # shouty headline: longest run of consecutive ALL-CAPS tokens
        max_run = run = 0
        for tok in re.findall(r"\S+", " ".join(lines[:max(anchor + 3, 12)])):
            if _CAPS_TOKEN_RE.match(tok) and _ALPHA_RE.search(tok):
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        q -= 0.25 * _clamp((max_run - 6.0) / 18.0)

        # rhetorical punctuation: a question in the headline/lead is a hard
        # newsroom violation; questions elsewhere in the body a soft one
        head_q = sum(lines[i].count("?") for i in range(min(anchor + 2, n_lines))
                     if not in_nav[i] and wcs[i] >= 6)
        q -= 0.22 * _clamp((head_q + lead_txt.count("?")) / 1.5)
        q -= 0.10 * _clamp(body.count("?") / 3.0)
        q -= 0.10 * _clamp(body.count("!") / 2.0)

        # opinion-statement lead ('issued the following statement'): the
        # main point is a stance, not front-loaded news
        if re.search(r"(?:issued|released|made) the following "
                     r"(?:statement|remarks)", body[:500], re.I):
            q -= 0.08

        # quote dominance: fraction of body characters inside quotes
        if body:
            q_chars = sum(len(m) for m in _QUOTE_SPAN_RE.findall(body))
            qfrac = q_chars / max(len(body), 1)
            q -= 0.22 * _clamp((qfrac - 0.35) / 0.30)

        # hard hype (mild)
        q -= 0.10 * _clamp(len(set(_HYPE_RE.findall(body or text))) / 4.0)

        # a marker-confirmed release with real prose never drops below the
        # nav-chrome band, however shouty its style
        if release >= 0.25 and prose_words >= 150:
            q = max(q, 0.12)
        q = _clamp(q)

        # ---------- LLM thick-input grounding (optional) ----------
        dk = str((extracted or {}).get("doc_kind", "")).strip().upper()
        if dk:
            if "RELEASE" in dk:
                gate = max(gate, 0.85)
            elif "NAV" in dk:
                gate = min(gate, 0.2)
            elif "ARTICLE" in dk:
                gate = min(gate, 0.5)
        ls = str((extracted or {}).get("lead_style", "")).strip().upper()
        if ls:
            if "DIRECT" in ls:
                q = _clamp(max(q, 0.55) + 0.15)
            elif "WORDY" in ls:
                q = min(q, 0.35)
            elif "BURIED" in ls:
                q = min(q, 0.30)

        if gate <= 0.04:
            return 0.02
        return _clamp(gate * (0.06 + 0.94 * q))
    except Exception:
        return 0.5
