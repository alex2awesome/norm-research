"""p905 v2 -- Authentic authorship, holistic composite of weak signals.

Criterion: the release reads as written by someone with genuine
familiarity with the company and its subject matter (concrete, specific,
internally coherent) rather than generic template marketing copy.

Approach: weighted blend of eight weak signals, each mapped to [0, 1]:
  concrete-detail density, mid-sentence proper-noun density,
  inverse template-buzzword density, inverse announcement-cliche count,
  attributed-quotation presence, entity coherence (the same named entity
  recurring across paragraphs/sentences), lexical diversity (windowed
  type-token ratio), and sentence-length variation. Signals that cannot
  be computed on short inputs contribute a neutral 0.5.

Contract: score(text) -> float in [0, 1]; higher = more authentic.
Deterministic (explicit alphabetical tie-breaks); imports limited to
re/math/statistics/collections.
"""

import re
import statistics
from collections import Counter

# ----------------------------------------------------------------------
# Mojibake / unicode normalization.
# All non-ASCII is written as \u escapes so the invisible U+009D is exact.
# ----------------------------------------------------------------------
_MOJIBAKE = (
    ("â€œ", '"'),   # misdecoded left double quote
    ("â€", '"'),   # misdecoded right double quote
    ("â€™", "'"),   # misdecoded apostrophe
    ("â€˜", "'"),   # misdecoded left single quote
    ("â€“", "-"),   # misdecoded en dash
    ("â€”", "-"),   # misdecoded em dash
    ("â€¦", "..."), # misdecoded ellipsis
    ("â€¢", "*"),   # misdecoded bullet
    ("Â ", " "),         # A-circ + nbsp -> space
)
_LEFTOVER_MOJI = re.compile("â€.")


def _normalize(text):
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    text = _LEFTOVER_MOJI.sub("'", text)
    text = text.replace("Â", "")
    text = text.replace(" ", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("[...]", "\n\n")
    return text


# ----------------------------------------------------------------------
# Signal lexicons and patterns
# ----------------------------------------------------------------------
_BUZZ_RE = re.compile(
    r"industry[\s-]+leading|world[\s-]+class|cutting[\s-]+edge"
    r"|state[\s-]+of[\s-]+the[\s-]+art|best[\s-]+in[\s-]+class"
    r"|next[\s-]+generation|leading\s+(?:provider|supplier|manufacturer)"
    r"|premier\s+(?:provider|source|destination)|innovative\s+solutions?"
    r"|comprehensive\s+(?:solutions?|suite)|end[\s-]+to[\s-]+end"
    r"|one[\s-]+stop|turn[\s-]?key|mission[\s-]+critical"
    r"|game[\s-]+chang(?:ing|ers?)|seamless(?:ly)?|leverag(?:e|es|ing)"
    r"|synerg(?:y|ies)|unparalleled|unmatched|unrivall?ed"
    r"|second\s+to\s+none|revolutionar(?:y|ies)|groundbreaking|paradigm"
    r"|empower(?:s|ing|ment)?|transformative|disruptive"
    r"|customer[\s-]+centric|value[\s-]+added|thought\s+leader(?:ship)?"
    r"|proven\s+track\s+record|trusted\s+partner"
    r"|wide\s+(?:range|variety|array)\s+of|vast\s+experience"
    r"|uniquely\s+positioned|at\s+the\s+forefront|world[\s-]+renowned"
    r"|market[\s-]+leading|award[\s-]+winning|holistic|robust|scalable"
    r"|visionary|excellence|innovative|top[\s-]+notch|unwavering"
    r"|exceptional\s+(?:service|quality|value)|high(?:est)?[\s-]+quality",
    re.IGNORECASE)

_CLICHE_RE = re.compile(
    r"is\s+(?:pleased|proud|excited|thrilled|delighted)\s+to\s+announce"
    r"|we\s+are\s+(?:excited|proud|thrilled|delighted)"
    r"|looks?\s+forward\s+to|committed\s+to\s+providing"
    r"|dedicated\s+to\s+helping|striv(?:e|es|ing)\s+to\s+(?:provide|deliver|exceed)"
    r"|takes?\s+(?:great\s+)?pride\s+in|the\s+go[\s-]+to"
    r"|your\s+one[\s-]+stop|passionate\s+about",
    re.IGNORECASE)

_SPEC_RES = (
    re.compile(r"[$£€]\s?\d[\d,]*(?:\.\d+)?"),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:million|billion|trillion)\b", re.IGNORECASE),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE),
    re.compile(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
        r"|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?"
        r"|dec(?:ember)?)\.?\s+\d{1,2}\b", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b[vV]?\d+\.\d+(?:\.\d+)*\b"),
    re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:employees|customers|users|patients|stores"
        r"|locations|countries|patents|units|acres?|miles|kg|km|mw|gwh?|ghz"
        r"|gb|tb|square\s+feet|sq\.?\s?ft\.?)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:CEO|CTO|CFO|COO|Chief\s+\w+\s+Officer|Vice\s+President"
        r"|Co-?[Ff]ounder|Founder|Managing\s+Director|Executive\s+Director)\b"),
)

_QUOTE_ATTRIB_RE = re.compile(
    r"\"[^\"\n]{15,600}\"\s*[,.]?\s*"
    r"(?:said|says|stated|added|noted|commented|explained|according\s+to)\b"
    r"|\b(?:said|says|stated|added|noted|commented)\s+[A-Z][\w.'-]+")
_PLAIN_QUOTE_RE = re.compile(r"\"[^\"\n]{20,600}\"")

_URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\(?\+?\d{1,3}\)?[\s.-]?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_TOKEN_RE = re.compile(r"[A-Za-z][\w'-]*")
_CAPWORD_RE = re.compile(r"^[A-Z][a-z]{2,}$")
_ACRO_RE = re.compile(r"^[A-Z]{2,5}$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_STOPCAPS = frozenset((
    "The", "This", "That", "These", "Those", "There", "Then", "They",
    "Their", "Its", "Our", "Your", "His", "Her", "She", "Him", "You",
    "And", "But", "For", "Not", "With", "From", "About", "After",
    "Before", "When", "While", "Where", "What", "Who", "How", "Why",
    "All", "Any", "Each", "Every", "Some", "Most", "More", "Many",
    "New", "One", "Two", "Three", "First", "Second", "Third", "Last",
    "Also", "Since", "Because", "However", "Additionally", "Today",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "Inc", "Llc", "Ltd", "Corp", "Company",
))

_STOPWORDS = frozenset((
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "will", "have", "has", "had", "its", "our", "their", "your", "them",
    "they", "his", "her", "she", "him", "who", "which", "what", "when",
    "where", "how", "why", "not", "but", "all", "any", "can", "may",
    "more", "most", "other", "some", "such", "than", "then", "there",
    "these", "those", "into", "over", "under", "about", "after", "before",
    "been", "being", "were", "would", "could", "should", "also", "each",
))


def _clip01(x):
    return max(0.0, min(1.0, x))


def _mid_sentence_caps(text):
    """(capitalized-word count incl. capped acronyms, list of cap tokens)."""
    count = 0
    acro = 0
    ents = []
    for sent in _SENT_SPLIT_RE.split(text):
        toks = _TOKEN_RE.findall(sent)
        for tok in toks[1:]:
            if _CAPWORD_RE.match(tok) and tok not in _STOPCAPS:
                count += 1
                ents.append(tok)
            elif _ACRO_RE.match(tok):
                acro += 1
                ents.append(tok)
    return count + 0.5 * min(acro, 6), ents


def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.5
        t = _normalize(text)
        t = _URL_RE.sub(" ", t)
        t = _EMAIL_RE.sub(" ", t)
        t = _PHONE_RE.sub(" ", t)

        # drop short chrome-like lines from signal computation
        lines = [ln.strip() for ln in t.split("\n")]
        kept_lines = [ln for ln in lines
                      if not ln or len(ln) >= 25 or ln[-1:] in ".!?\"'"]
        t = "\n".join(kept_lines)

        words = _WORD_RE.findall(t)
        n = len(words)
        if n < 8:
            return 0.5

        paras = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
        if len(paras) <= 1 and t.count("\n") > 4:
            paras = [p.strip() for p in t.split("\n") if p.strip()]
        sents = [s for s in _SENT_SPLIT_RE.split(t) if s.strip()]

        # 1) concrete-detail density
        spec = sum(len(rx.findall(t)) for rx in _SPEC_RES)
        s_specific = _clip01((100.0 * spec / n) / 5.0)

        # 2) mid-sentence proper-noun density
        prop_units, entities = _mid_sentence_caps(t)
        s_propnoun = _clip01((100.0 * prop_units / n) / 8.0)

        # 3) inverse template-buzzword density
        buzz = len(_BUZZ_RE.findall(t))
        s_notgeneric = 1.0 - _clip01((100.0 * buzz / n) / 4.0)

        # 4) inverse announcement-cliche count
        cliche = len(_CLICHE_RE.findall(t))
        s_notcliche = 1.0 - _clip01(cliche / 3.0)

        # 5) attributed quotation
        if _QUOTE_ATTRIB_RE.search(t):
            s_quote = 1.0
        elif _PLAIN_QUOTE_RE.search(t):
            s_quote = 0.6
        else:
            s_quote = 0.35

        # 6) entity coherence: one named entity recurring across units
        units = paras if len(paras) >= 2 else sents
        if len(units) >= 3 and entities:
            counts = Counter(entities)
            # deterministic tie-break: highest count, then alphabetical
            top = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            frac = sum(1 for u in units if top in u) / float(len(units))
            s_coherence = _clip01(frac / 0.5)
        elif entities:
            s_coherence = 0.5 if len(units) < 3 else 0.3
        else:
            s_coherence = 0.0 if len(units) >= 3 else 0.25

        # 7) lexical diversity (windowed type-token ratio on content words)
        content = [w.lower() for w in words
                   if len(w) >= 3 and w.lower() not in _STOPWORDS]
        if len(content) >= 30:
            win, step = 40, 20
            ttrs = []
            for i in range(0, max(1, len(content) - win + 1), step):
                chunk = content[i:i + win]
                if len(chunk) >= 20:
                    ttrs.append(len(set(chunk)) / float(len(chunk)))
            ttr = statistics.mean(ttrs) if ttrs else (
                len(set(content)) / float(len(content)))
            s_diversity = _clip01((ttr - 0.45) / 0.35)
        else:
            s_diversity = 0.5

        # 8) sentence-length variation
        slens = [len(_WORD_RE.findall(s)) for s in sents]
        slens = [x for x in slens if x > 0]
        if len(slens) >= 3 and statistics.mean(slens) > 0:
            cv = statistics.pstdev(slens) / statistics.mean(slens)
            s_sentvar = _clip01(cv / 0.55)
        else:
            s_sentvar = 0.5

        val = (0.20 * s_specific + 0.14 * s_propnoun + 0.20 * s_notgeneric
               + 0.08 * s_notcliche + 0.08 * s_quote + 0.12 * s_coherence
               + 0.08 * s_diversity + 0.10 * s_sentvar)
        return float(_clip01(val))
    except Exception:
        return 0.5
