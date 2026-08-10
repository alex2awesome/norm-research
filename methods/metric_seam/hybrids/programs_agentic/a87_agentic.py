"""Hybrid metric channel for aspect a87 — "Human, humble spokesperson tone" (agentic v1).

Criterion: spokesperson quotes should sound natural, respectful, and unboastful
rather than canned or grandiose.

GOAL OF THIS PASS: a87 was already CERTIFIED but field-dominated — h0 blends
0.70*LLM_tone + 0.30*code UNIFORMLY for every tone value. Full-TRAIN (79 items,
not just the worst residuals) diagnosis of the h0 field/tone split found that
this uniform weight is badly miscalibrated:

  quote_tone bucket   n    judge mean   judge range
  ------------------  ---  -----------  -------------------
  "corporate"          50   0.49         0.00 .. 0.95  (!) — the MODE label
                                                          carries almost no
                                                          information on its
                                                          own; code must do
                                                          the differentiating.
  "neutral"            11   0.92         0.75 .. 1.00   — h0's constant for
                                                          this bucket (0.62)
                                                          was badly LOW; these
                                                          are measured analyst/
                                                          scientist quotes the
                                                          judge scores near 1.
  ""  (LLM: no quote)   5   0.00         0.00 .. 0.00   — reliably near-zero,
                                                          yet h0's fallback
                                                          branch FLOORS
                                                          quote-less "release"
                                                          docs to >=0.30,
                                                          actively wrong here.
  "natural"/"humble"/"aggressive"/"boastful" (n=4/2/5/2): small-n but
  directionally coherent with the criterion's own theory (measured analyst
  speech -> high, hostile/boastful -> low); worth keeping high LLM trust.

Changes over h0 (biggest levers first; all validated on the FULL 79-item
train set, not just the worst-residual list):

  1. NEW code-only signal: nominalization/buzzword-suffix DENSITY inside the
     genuine-quote text (words ending -tion/-ity/-ment/-ance/-ence/...).
     This is a MEASURED STRUCTURE signal, not a keyword list: h0's exact
     _BOAST/_SELFPROMO lexicons miss most canned corporate PR-speak because
     it rarely uses the literal words in those lists ("fundamentally change
     the way the world lives and works", "in our DNA", "next logical step in
     the evolution of our industry" all score 0 boast-hits in h0, judge rates
     them 0.25-0.35). Suffix density alone has rho=-0.43 vs judge across the
     60 quote-bearing train items (a bare keyword-list channel has no
     comparable general handle on this). Folded into the existing capped
     demotion; grid over the coefficient (2..10) is a flat plateau around
     rho ~0.48 code-only, so this reads as a real signal, not a spike.
  2. RECALIBRATED tone_map constants from the bucket means above (biggest
     single fix: neutral 0.62 -> 0.88 recovers ~11 badly-underscored
     analyst-quote docs) plus mild nudges to boastful/aggressive/humble.
  3. DIFFERENTIAL field trust: "corporate" (the uninformative mode label)
     gets a REDUCED LLM weight (0.55) so code's richer signal (boast/humble/
     aggression lexicons + new suffix density) carries more of the decision;
     the small-n but decisive tags (natural/neutral/humble/aggressive/
     boastful) keep a HIGH LLM weight (0.80) since they are directionally
     reliable. This is the literal "move field work into code" lever: the
     field's blended weight now tracks how much information the tag value
     actually carries, instead of one constant for every value.
  4. BUG FIX: h0's fallback for quote-less docs relied on `doc_type` and
     FLOORED quote-less "release" docs to >=0.30 — but the reliable signal
     in the data is whether the LLM independently also found NO quote
     (`quote_tone==""`), not doc_type: every train item with quote_tone==""
     is judge==0.0 regardless of doc_type (incl. two well-formed release-
     shaped docs the old floor would have re-inflated). Replaced the floor
     with a low gate (<=0.12) keyed on tone=="" AND no code-detected quote.
  5. NEW small text-repair op: a handful of scraped docs have their quote
     marks mangled by a corpus-specific mojibake into a bare lowercase "a"
     glued directly to the next (capitalized) word with no space
     ("...Management. aThere's no getting around..."). A standalone "a"
     fused to a following capitalized word never occurs in real English (a
     real article "a" always has a following space), so repairing it to a
     literal `"` is safe general text hygiene (same category as
     ops.normalize's existing curly-quote table), not a per-item hack. It
     only changes extraction on 1/79 train docs but recovering that one
     quote (judge 0.85, previously quote-less and stuck near 0.34) is worth
     +0.02 rho on its own because Spearman is sensitive to badly-misranked
     items.

TRAIN rho: h0 reference 0.7005 -> this candidate ~0.79-0.80 (see harness run).

Residual failure mode NOT fixed in code (irreducibly field-shaped): a doc
whose only "quote" is dash-attributed prose with no literal quote marks at
all (an analyst-commentary video-transcript page, e.g. "In our view, the key
challenge... - Huw Pill", judge 0.80) is invisible to any quote-mark-based
extractor; only the LLM field sees it. Recovering these would need a much
more aggressive attribution-without-quote-marks heuristic that we couldn't
validate as general on the available train examples (too rare/heterogeneous
to distinguish from ordinary prose without over-fitting).
"""
import re

# ---------------------------------------------------------------- LLM fields --
# SAME field names + instructions as a87_h0.py, verbatim. No new LLM fields.
LLM_FIELDS = {
    "quote_tone": (
        "One word for the tone of the main spokesperson/person quote: humble, "
        "natural, neutral, corporate, boastful, or aggressive; answer none if "
        "nobody is directly quoted."
    ),
    "doc_type": (
        "Answer RELEASE if this is a genuine press release or news article with "
        "real reported content, or OTHER if it is a navigation / marketing / "
        "boilerplate page with no news."
    ),
}

# ------------------------------------------------------------------ lexicons --
_ATTR = re.compile(
    r"\b(said|says|saying|stated|according to|added|adds|noted|notes|"
    r"commented|explained|explains|told|charged|remarked|announced|"
    r"asked|continued|observed|recalled|wrote|writes)\b",
    re.I,
)

_BOAST = [
    "world-class", "world class", "best-in-class", "best in class",
    "industry-leading", "industry leading", "cutting-edge", "cutting edge",
    "revolutionary", "unparalleled", "unrivaled", "unrivalled", "premier",
    "premiere", "groundbreaking", "ground-breaking", "game-changing",
    "game changer", "unprecedented", "world's leading", "world leader",
    "global leader", "market leader", "leading provider", "number one",
    "#1", "dominant", "best ever", "best quarter", "best year", "highest ever",
    "record-high", "record high", "record low", "record-low", "most profitable",
    "strongest ever", "biggest ever", "largest ever", "world's best",
    "world's premier", "best possible", "speaks volumes", "second to none",
    "the very best", "leading the", "leader in", "our highest",
    "our best", "our most", "our strongest", "our largest",
]

_HUMBLE = [
    "humbled", "humbling", "humble", "honored", "honoured", "privileged",
    "grateful", "gratitude", "fortunate", "thankful", "appreciate",
    "appreciative", "much to do", "still have much", "still much", "learn from",
    "we owe", "could not have", "couldn't have", "thanks to", "in service of",
    "listen to", "grounded", "modest",
]

_AGGR = [
    "sabotage", "disgraced", "terrorism", "assassination", "condemn",
    "thumbing", "oppression", "disaster", "impeach", "impeachment", "scandal",
    "corrupt", "outrage", "unlawful", "meek response", "conceal", "sham",
    "betray", "reckless", "disgrace", "green light", "state terror",
    "extrajudicial", "sinister", "shameful", "hostile",
]

# self-congratulatory corporate cues (weighted lighter than outright boast)
_SELFPROMO_ACHV = [
    "proud", "pleased", "delighted", "excited", "thrilled", "success",
    "successful", "strong", "record", "milestone", "excellence", "leading",
    "well-positioned", "well positioned", "delivered", "profit", "growth",
    "outperform", "momentum", "robust",
]

_PR_MARKERS = [
    "prnewswire", "pr newswire", "news provided by", "for immediate release",
    "press release", "news release", "media contact", "globenewswire",
    "businesswire", "business wire", "/prweb", "source ",
]

_FP_SINGULAR = re.compile(r"\bI\b")          # case-sensitive first-person singular
_FP_SING_LC = re.compile(r"\b(my|me|myself)\b", re.I)
_FP_ANY = re.compile(r"\b(we|our|us|i|my|me)\b", re.I)

# NEW: nominalization/buzzword suffix density -- a MEASURED-STRUCTURE proxy for
# canned corporate jargon that keyword lists miss ("in our DNA", "next logical
# step in the evolution of our industry", "fundamentally change the way the
# world lives and works" all have zero _BOAST hits but are dense in abstract
# -tion/-ity/-ment/-ance/-ence nouns).
_JARGON_SUFFIX = re.compile(
    r"\b[a-zA-Z]{6,}(?:tions?|ities|ity|ments?|ances?|ences?|ancy|ency|"
    r"ability|abilities|ivity)\b",
    re.I,
)

# corpus-specific mojibake: quote marks mangled into a bare "a" glued directly
# to the following capitalized word (no legitimate English word does this --
# a real article "a" is always followed by a space). Repairing this recovers
# genuine quotes that would otherwise be invisible to _quotes().
_QUOTEISH_A = re.compile(r'(?<=[\s"“‘(])a(?=[A-Z][a-z])')


def _clamp(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _count(hay, needles):
    return sum(hay.count(n) for n in needles)


def _repair_mojibake_quotes(text):
    """Fix the corpus-specific "a" mangling of curly quote marks (see module
    docstring #5). Safe/general: never fires on ordinary English text."""
    return _QUOTEISH_A.sub('"', text)


def _quotes(text):
    """Quote spans by pairing SEQUENTIAL quote marks (open/close), which is
    robust to short in-line quotes (nicknames, scare quotes) that break naive
    `"..."` regex pairing.  Keeps spans of plausible sentence length."""
    marks = [i for i, c in enumerate(text) if c == '"']
    out = []
    for k in range(0, len(marks) - 1, 2):
        s, e = marks[k], marks[k + 1]
        content = text[s + 1:e]
        if 12 <= len(content) <= 700:
            out.append((s, e + 1, content))
    return out


def _genuine_quotes(text, spans):
    """Keep spans that read as attributed human SPEECH (a real spokesperson
    sentence), not slogans / scare-quoted fragments / nav."""
    out = []
    for s, e, q in spans:
        if len(re.findall(r"[A-Za-z']+", q)) < 5:      # sentence-length only
            continue
        letters = [c for c in q if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            continue                                    # drop all-caps banners
        ctx = text[max(0, s - 90):s] + text[e:e + 90]
        if _ATTR.search(ctx) or _FP_ANY.search(q):
            out.append(q)
    return out


def _release_likeness(text):
    """0..1 crude estimate that the doc is a real release/news article."""
    t = text.lower()
    sc = 0.0
    for mk in _PR_MARKERS:
        if mk in t:
            sc += 0.28
            break
    # dateline:  CITY, Month Day, Year   /   CITY, State, Month Day
    if re.search(r"[A-Z][A-Za-z.]+,\s+[A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4}", text):
        sc += 0.22
    if re.search(r"\b(announce[ds]?|announcing|unveil|report(?:ed|s)?)\b", t):
        sc += 0.12
    if _ATTR.search(text):
        sc += 0.16
    return _clamp(sc)


def _jargon_density(q):
    """Fraction of quote words that are abstract corporate-jargon nominal-
    izations. A general morphological signal, not a fixed phrase list."""
    words = re.findall(r"[A-Za-z']+", q)
    if not words:
        return 0.0
    return len(_JARGON_SUFFIX.findall(q)) / len(words)


def _code_score(text, ops):
    """Code-only channel: everything computable without the LLM fields."""
    try:
        text = ops.normalize(text)
    except Exception:
        pass
    try:
        text = _repair_mojibake_quotes(text)
    except Exception:
        pass

    spans = _quotes(text)
    quotes = _genuine_quotes(text, spans)
    rel = _release_likeness(text)

    if not quotes:
        # criterion barely applies: quote-less doc. Legit release -> mid-low,
        # nav-chrome junk -> near 0. Personal first-person voice (a blog) lifts.
        fp_sing = len(_FP_SINGULAR.findall(text)) + len(_FP_SING_LC.findall(text))
        fp_bonus = min(0.30, 0.035 * fp_sing) if fp_sing >= 3 else 0.0
        code = 0.10 + 0.34 * rel + fp_bonus
        return _clamp(code), quotes

    q = " ".join(quotes).lower()
    words = re.findall(r"[a-z']+", q)
    n = max(1, len(words))

    b = _count(q, _BOAST)
    h = _count(q, _HUMBLE)
    a = _count(q, _AGGR)
    sp = _count(q, _SELFPROMO_ACHV)
    excl = q.count("!")
    # density-aware boast: superlatives concentrated in short quotes hurt more
    boast_dens = b / (n / 40.0 + 1.0)
    jargon = _jargon_density(" ".join(quotes))

    has_i = bool(re.search(r"\bI\b", " ".join(quotes)) or re.search(r"\bmy\b", q))
    has_we = bool(re.search(r"\b(we|our|us)\b", q))
    # canned corporate courtesy: "we're pleased/honored/proud to <verb>"
    # (a formulaic release cliche, distinct from a personal humble quote)
    courtesy = bool(re.search(
        r"\b(honored|honoured|pleased|proud|delighted|thrilled|excited)"
        r"\s+to\b", q)) and has_we and not has_i

    base = 0.80
    # humility lifts -- but not when the humble word is just a courtesy cliche
    if not courtesy:
        base += min(0.16, 0.06 * h)
    if has_i:
        base += 0.04                                 # personal 1st-person voice

    # grandiosity + canned self-promo + courtesy + jargon density share ONE
    # capped demotion so overlapping cues don't triple-count. jargon density
    # is the NEW lever: it catches canned PR-speak with zero literal boast
    # keywords ("fundamentally change the way the world lives and works",
    # "in our DNA", "next logical step in the evolution of our industry").
    demote = (0.20 * b + 0.10 * boast_dens + 0.06 * sp
              + (0.12 if courtesy else 0) + 4.0 * jargon)
    base -= min(0.62, demote)
    base -= min(0.15, 0.05 * excl)                   # hype punctuation
    # aggression is separate and CAN sink a hostile quote toward zero
    base -= min(0.72, 0.34 * a)
    # very sparse "quote" (single short attributed fragment) -> less credit
    if n < 8 and len(quotes) == 1:
        base -= 0.10

    return _clamp(base), quotes


# tone_map recalibrated against FULL-train per-bucket judge means (see module
# docstring): "neutral" was the single biggest miscalibration (0.62 -> 0.88).
_TONE_MAP = {
    "humble": 0.95, "natural": 0.90, "neutral": 0.88,
    "corporate": 0.42, "canned": 0.40, "formulaic": 0.42,
    "boastful": 0.20, "grandiose": 0.16, "aggressive": 0.15,
}
# tags with (near-)no discriminative power on their own (the corpus MODE
# label, spanning judge 0.0-0.95) get a REDUCED LLM blend weight so the
# richer code channel (lexicons + jargon density) carries more of the
# decision; the small-n but directionally-decisive tags keep a high weight.
_LOW_INFO_TONES = {"corporate", "canned", "formulaic"}
_W_LOW_INFO = 0.55
_W_HIGH_INFO = 0.80
_ZERO_GATE = 0.12


def score(text: str, extracted: dict, ops) -> float:
    code, quotes = _code_score(text, ops)

    tone = str((extracted or {}).get("quote_tone", "") or "").strip().lower()
    dtype = str((extracted or {}).get("doc_type", "") or "").strip().lower()

    # BUG FIX (was: doc_type=='release' floored quote-less docs to >=0.30):
    # the reliable full-train signal is the LLM ALSO finding no quote at all
    # (tone==""), not doc_type -- every such train item is judge==0.0 whether
    # or not it is release-shaped.
    if not tone and not quotes:
        return _clamp(min(code, _ZERO_GATE))

    tone_key = next((k for k in _TONE_MAP if k in tone), None)
    if tone_key is not None:
        w = _W_LOW_INFO if tone_key in _LOW_INFO_TONES else _W_HIGH_INFO
        return _clamp(w * _TONE_MAP[tone_key] + (1 - w) * code)

    # unexpected/empty tone value with the code path still having something
    # to say (e.g. code found quotes the LLM missed): trust code, but keep
    # the doc_type nav-chrome cap as a safety net.
    if dtype and "other" in dtype:
        return _clamp(min(code, 0.15))

    return code
