"""a76 -- News-first lede clarity (hybrid channel).

Criterion: deliver the core announcement (5Ws/impact) immediately in the lede,
suitable for direct-to-public reading.

Design: find the first real PROSE sentences (skipping nav chrome / boilerplate),
then score the best early candidate on the co-occurrence -- inside ONE early
sentence -- of a news predicate (announce/acquire/sign/will-do/quantified
movement), a named actor, concrete magnitude, and a time anchor. Feature-style
ledes (1st/2nd person, questions, quote-first, vague subjects) are damped, as
are non-English pages, ticker dumps, and nav-only pages. Release markers
(dateline / PRNewswire / exchange ticker) add a small bonus ONLY when a
qualifying lede sentence already exists (presence != quality).
"""
import re

LLM_FIELDS = {
    "lede_news": "Ignoring navigation/boilerplate, quote the main article's opening sentence only if it directly states a specific new development or announcement (who did or will do what); answer NONE if the opening is navigation, marketing copy, a product/list page, or a feature-style lede.",
    "lede_when": "State the explicit time reference (e.g. 'today', 'March 6, 2013', 'third quarter 2021') that appears in the article's first two body sentences; answer NONE if there is none.",
}

# --- lexical machinery -------------------------------------------------------
_STRONG_VERB = re.compile(
    r"\b(announc(?:es|ed|ing)|launch(?:es|ed|ing)|unveil(?:s|ed|ing)?"
    r"|introduc(?:es|ed|ing)|complet(?:es|ed|ing)|acquir(?:es|ed|ing)"
    r"|purchas(?:es|ed|ing)|bought|nam(?:es|ed)|appoint(?:s|ed)|award(?:s|ed)"
    r"|issu(?:es|ed)|formaliz(?:es|ed)|enter(?:s|ed)\s+into|agre(?:es|ed)\s+to"
    r"|sign(?:s|ed)\s+(?:an?\s+)?(?:agreement|deal|contract|pact|mou|letter)"
    r"|voted?\s+to|approv(?:es|ed)|merg(?:es|ed)|rais(?:es|ed)"
    r"|invest(?:s|ed)|won|wins|open(?:s|ed)|expand(?:s|ed)|receiv(?:es|ed)"
    r"|plans?\s+to|intends?\s+to"
    r"|will\s+(?:be\s+)?(?:begin|start|unveil|launch|acquire|deliver|open"
    r"|invest|build|reveal|continue|purchase|buy|offer|host|hold|expand"
    r"|join|receive|become|debut|introduce|report|make|create|add)\b)",
    re.I,
)
_WILL_ANY = re.compile(r"\bwill\s+[a-z]+", re.I)
_PASSIVE_PRE = re.compile(r"\b(?:was|were|been|being|be|are|is)\s+$", re.I)
# movement verbs (past/simple only) count as news when quantified in-sentence
_QUANT_VERB = re.compile(
    r"\b(total(?:l)?ed|declined|dropped|rose|grew|fell|increased|decreased"
    r"|surpassed|jumped|exceeded|slid|sank|surged|climbed|plunged)\b",
    re.I,
)
_NUM = re.compile(
    r"\$\s?\d|\d+(?:\.\d+)?\s?(?:%|percent|per\s+cent)"
    r"|\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?\s?(?:million|billion|trillion)\b",
    re.I,
)
_WHEN_EXTRA = re.compile(
    r"\b(today(?!['’]s)|yesterday|tomorrow"
    r"|this\s+(?:week|month|year|morning|quarter|fall|spring|summer|winter)"
    r"|last\s+(?:week|month|year|quarter)"
    r"|(?:first|second|third|fourth)[-\s]quarter|Q[1-4]\b"
    r"|quarter[- ]to[- ]quarter|year[- ]over[- ]year|month[- ]over[- ]month)",
    re.I,
)
_DATELINE = re.compile(
    r"\b[A-Z][A-Za-z. ]{2,24},\s*(?:[A-Z][A-Za-z.]{1,15}\.?,?\s+)?"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
)
_RELEASE_MARK = re.compile(
    r"(/\s?PR\s?Newswire\s?/|PRNewswire|Business\s?Wire|GlobeNewswire"
    r"|For\s+Immediate\s+Release|PRESS\s+RELEASE)", re.I,
)
_TICKER_PAREN = re.compile(r"[\(\[]\s?(?:NYSE|NASDAQ|OTC|TSX|LSE)\s?:")
_CHROME = re.compile(
    r"(cookie|privacy|sign in|sign up|log in|login|subscribe|newsletter"
    r"|javascript|browser|rights reserved|terms of use|terms and conditions"
    r"|skip to|copyright|click|follow us|like us on|read more|learn more"
    r"|advertisement|password|email address|contact us|site\s?map|free trial"
    r"|disclaimer|ad-blocker|share this|view all|download|faq)",
    re.I,
)
_EN_STOP = {"the", "of", "to", "and", "in", "a", "for", "is", "on", "that",
            "with", "as", "are", "was", "at", "by", "it", "from", "has", "an"}
_FUNC = {"the", "a", "an", "of", "to", "in", "for", "on", "with", "and",
         "at", "by", "from", "its", "has", "have", "is", "are", "was", "were"}
_PRONOUN = re.compile(r"\b(?:[Ww]e|[Oo]ur|[Yy]ou(?:rs?)?|[Mm]y|I)\b|\bus\b")
_VAGUE_SUBJ = {"it", "they", "he", "she", "we", "i", "you", "this", "these",
               "those", "there", "here"}


def _words(s):
    return re.findall(r"[A-Za-z][A-Za-z'.-]*", s)


def _prose_sentences(t):
    """Yield (char_offset, sentence) for real prose sentences, chrome skipped."""
    out = []
    offset = 0
    kept_chunks = []
    for line in t.split("\n"):
        ls = line.strip()
        w = _words(ls)
        if len(w) >= 6:
            lower_frac = sum(1 for x in w if x[:1].islower()) / len(w)
            if lower_frac >= 0.4 and not _CHROME.search(ls):
                kept_chunks.append((t.find(line, offset), ls))
        offset += len(line) + 1
    for pos, chunk in kept_chunks:
        # split glued lowercase.Uppercase first
        chunk = re.sub(r"(?<=[a-z])\.(?=[A-Z])", ". ", chunk)
        cursor = 0
        for sent in re.split(r"(?<=[.!?])\s+", chunk):
            s = sent.strip()
            w = _words(s)
            if 8 <= len(w) <= 90:
                capf = sum(1 for x in w if x[:1].isupper()) / len(w)
                nfunc = sum(1 for x in w if x.lower() in _FUNC)
                if capf < 0.6 and nfunc >= 2 and not _CHROME.search(s):
                    out.append((pos + cursor, s))
            cursor += len(sent) + 1
    return out


def _sent_when(sent, ops):
    if _WHEN_EXTRA.search(sent):
        return True
    try:
        return bool(ops.extract_dates(sent))
    except Exception:
        return bool(re.search(r"\b(?:19|20)\d{2}\b", sent))


def _active_strong(sent):
    """A strong news verb used actively (not 'were appointed'/'has been named')."""
    for m in _STRONG_VERB.finditer(sent):
        if not _PASSIVE_PRE.search(sent[max(0, m.start() - 8):m.start()]):
            return True
    return False


def _lede_core(sent, ops):
    """Score one candidate sentence's news-first quality, in [0,1]."""
    num = 1.0 if _NUM.search(sent) else 0.0
    when = 1.0 if _sent_when(sent, ops) else 0.0
    strong = _active_strong(sent) or (bool(_WILL_ANY.search(sent)) and when > 0)
    quant = bool(_QUANT_VERB.search(sent)) and num > 0
    w = _words(sent)
    mid_ent = any(x[:1].isupper() for x in w[1:]) or bool(
        re.search(r"\b(?:Inc|Corp|Co|Company|Group|Ltd|LLC|University|Institute"
                  r"|Department|NYSE|NASDAQ)\b", sent))
    ent = 1.0 if mid_ent else 0.0
    if strong or quant:
        core = 0.45 + 0.15 * ent + 0.2 * num + 0.2 * when
        if not mid_ent:
            core *= 0.55  # a news-first lede names its actor (the WHO)
    else:
        core = 0.12 * (num + when)
    # feature-lede / non-news damping (decouples keyword presence from quality)
    if _PRONOUN.search(sent):
        core *= 0.35
    if "?" in sent:
        core *= 0.4
    first = w[0].lower() if w else ""
    if first in _VAGUE_SUBJ:
        core *= 0.35
    if sent.lstrip()[:1] in "\"'":
        core *= 0.6
    if re.search(r"\bHere (?:are|is)\b|\btop \d+\b|\bstocks? (?:added|to watch|to buy)\b",
                 sent, re.I):
        core *= 0.35  # listicle framing, not an announcement lede
    return core


def score(text, extracted, ops):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        n = len(t)

        sents = _prose_sentences(t)
        cands = sents[:8]

        # global damping factors --------------------------------------------
        body_words = [x.lower() for _, s in sents[:30] for x in _words(s)][:300]
        if body_words:
            stop_frac = sum(1 for x in body_words if x in _EN_STOP) / len(body_words)
        else:
            head_words = [x.lower() for x in _words(t[:2000])][:300]
            stop_frac = (sum(1 for x in head_words if x in _EN_STOP) /
                         max(1, len(head_words)))
        lang_mult = 1.0 if stop_frac >= 0.16 else (0.6 if stop_frac >= 0.10 else 0.1)

        head_toks = t.split()[:200]
        n_tickers = sum(1 for x in head_toks
                        if re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z]{1,2})?,?", x))
        ticker_mult = 0.5 if n_tickers >= 25 else 1.0

        # best early news-first sentence -------------------------------------
        best = 0.0
        best_global = 0.0
        for idx, (pos, s) in enumerate(sents):
            core = _lede_core(s, ops)
            best_global = max(best_global, core)
            if idx < len(cands):
                idx_w = 1.0 / (1.0 + 0.35 * idx)
                frac = pos / max(1, n)
                pos_w = 1.0 if frac < 0.5 else (0.7 if frac < 0.75 else 0.45)
                best = max(best, core * idx_w * pos_w)

        # release markers: small bonus, gated on an actual qualifying lede ---
        bonus = 0.0
        if best >= 0.25:
            headzone = t[:3000]
            if _DATELINE.search(headzone):
                bonus += 0.2
            if _RELEASE_MARK.search(headzone):
                bonus += 0.1
            if _TICKER_PAREN.search(headzone):
                bonus += 0.12
            bonus = min(bonus, 0.25)

        # a fully-formed news lede sentence ANYWHERE past the chrome still
        # signals a news-first document (scraped chrome may bury the lede)
        anywhere = 0.3 * best_global if best_global >= 0.75 else 0.0

        val = max(min(1.0, best + bonus), anywhere) * lang_mult * ticker_mult
        if not cands:
            val = min(val, 0.02)
        elif len(cands) == 1:
            val = min(val, 0.5)
        elif len(cands) >= 4 and lang_mult >= 1.0 and ticker_mult >= 1.0:
            val = max(val, 0.06)  # real English article prose, however weak

        # LLM-extracted thick-input grounding (optional; predicate stays here)
        extracted = extracted or {}
        f1 = (extracted.get("lede_news") or "").strip()
        f2 = (extracted.get("lede_when") or "").strip()
        if "lede_news" in extracted:
            if f1 and f1.upper() != "NONE" and _valid_lede(f1, t):
                llm = 0.5
                if _STRONG_VERB.search(f1) or _QUANT_VERB.search(f1):
                    llm += 0.15
                if _NUM.search(f1):
                    llm += 0.1
                if f2 and f2.upper() != "NONE" and (
                        _WHEN_EXTRA.search(f2)
                        or re.search(r"\d", f2)):
                    llm += 0.1
                val = max(val, min(1.0, 0.3 * val + llm * lang_mult * ticker_mult))
            else:
                val *= 0.5  # extractor confirmed no news-first lede
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.3


def _valid_lede(f1, t):
    """Code-side validation of the LLM field: verb-bearing, named, grounded."""
    w = _words(f1)
    if len(w) < 4:
        return False
    if not any(x[:1].isupper() for x in w):
        return False
    if _PRONOUN.search(f1) and not _STRONG_VERB.search(f1):
        return False
    low = t.lower()[: int(len(t) * 0.7)]
    content = [x.lower() for x in w if len(x) >= 5]
    grounded = sum(1 for x in content if x in low)
    return bool(content) and grounded >= max(1, len(content) // 3)
