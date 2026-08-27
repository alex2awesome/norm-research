"""a162 "Theme clarity and depth" -- hybrid channel h0.

Design rationale (from train residuals):
  Judge strata: LOW  = gag/shock comedy, author intrusions (edit notes, apologies,
                       self-promo), caps-lock shouting, pop-culture riffing;
                MID  = competent genre twist pieces, some meaning but shallow;
                HIGH = earnest stories that develop an abstract moral/philosophical
                       idea (duty, blame, faith, purpose), controlled prose, refrains.
  Keyword presence of theme words (baseline, rho=.349) is a weak proxy: the judge
  responds to whether the story ARGUES a theme, not whether it mentions one.
  So: two thick-input LLM fields carry the tacit construct (does the story have a
  serious theme, and is its primary aim meaning vs. fun vs. jokes); the predicate,
  calibration, and robustness hedges (author-intrusion / shouting penalties,
  thematic-recurrence and refrain bonuses) stay in code.
"""

import re
from collections import Counter

LLM_FIELDS = {
    "theme": ("State the story's central theme as one abstract idea in at most 8 words "
              "(e.g. 'duty versus love'); answer NONE if the story has no serious theme."),
    "intent": ("Classify the story's primary aim in exactly one word: HUMOR (gag or parody), "
               "ENTERTAINMENT (plot fun), or MEANING (explores an idea or emotion)."),
}

# Abstract thematic concepts (used to grade the extracted theme statement and,
# at small weight, in-text thematic recurrence).
_ABSTRACT = {
    "love", "loss", "grief", "death", "mortality", "duty", "sacrifice", "redemption",
    "forgiveness", "justice", "revenge", "guilt", "blame", "responsibility", "faith",
    "belief", "hope", "despair", "identity", "purpose", "meaning", "freedom", "power",
    "corruption", "greed", "ignorance", "truth", "deception", "loneliness", "connection",
    "kindness", "cruelty", "humanity", "morality", "moral", "choice", "consequence",
    "consequences", "fate", "destiny", "memory", "war", "peace", "fear", "courage",
    "betrayal", "trust", "prejudice", "acceptance", "innocence", "obsession", "isolation",
    "legacy", "regret", "compassion", "empathy", "vengeance", "honor", "pride", "envy",
    "mercy", "suffering", "goodness", "evil", "conscience", "atonement", "solitude",
}

_THEME_PATTERNS = (
    r"\bvs\.?\b", r"\bversus\b", r"\bcost of\b", r"\bnature of\b", r"\bpower of\b",
    r"\bimportance of\b", r"\bdanger(?:s)? of\b", r"\bloss of\b", r"\bsearch for\b",
    r"\bmeaning of\b", r"\bprice of\b", r"\bstruggle\b", r"\bcorrupt", r"\bblind",
)

_NONE_ANSWERS = {"none", "n/a", "na", "no", "nothing", "no theme", "-"}

# Author-intrusion / meta-chrome markers (checked near head+tail where notes live).
_INTRUSION_PATTERNS = (
    r"\bedit\s*\d*\s*:", r"\[\s*wp\s*\]", r"first\s+time\s+post", r"long\s+time\s+lurker",
    r"sorry\s+for\s+(?:the\s+|my\s+|bad\s+)*(?:formatting|grammar|spelling|mistakes)",
    r"wrote\s+this\s+on\s+my\s+phone", r"don'?t\s+tear\s+me\s+apart",
    r"thanks?\s+(?:you\s+)?for\s+reading", r"if\s+you\s+liked\s+this",
    r"feedback\s+is\s+(?:welcome|appreciated)", r"criticism\s+(?:is\s+)?welcome",
    r"\bupvote", r"over\s+at\s+r/", r"\bpls\b", r"\bplz\b", r"my\s+formatting",
    r"since\s+i'?ve\s+written", r"if\s+no\s+one\s+else\s+will\s+post",
    r"still\s+crap\b", r"i\s+know\s+i\s+left\s+.{0,20}plot\s+holes",
)


def _theme_component(theme):
    """Grade the LLM's theme statement: absent -> 0, plot-ish -> low, abstract -> high."""
    s = re.sub(r"\s+", " ", (theme or "")).strip().lower().strip(".!\"'")
    if not s or s in _NONE_ANSWERS or s.startswith("none"):
        return 0.0
    words = re.findall(r"[a-z']+", s)
    hits = sum(1 for w in words if w in _ABSTRACT)
    pat = any(re.search(p, s) for p in _THEME_PATTERNS)
    if hits == 0 and not pat:
        return 0.40  # a theme was named, but reads plot-like / concrete
    v = 0.55 + 0.16 * min(2, hits) + (0.08 if pat else 0.0)
    return min(0.95, v)


def _intent_component(intent):
    """MEANING -> 1, ENTERTAINMENT -> 0.5, HUMOR -> 0; unknown/blank -> 0.5."""
    u = (intent or "").strip().upper()
    if not u:
        return 0.5
    if "MEANING" in u or "EARNEST" in u or "SERIOUS" in u:
        return 1.0
    if ("HUMOR" in u or "HUMOUR" in u or "COMED" in u or "PARODY" in u
            or "JOKE" in u or "GAG" in u or "SATIR" in u):
        return 0.0
    if "ENTERTAIN" in u:
        return 0.5
    return 0.5


def _code_component(raw, ops):
    """Class-level craft/chrome signals kept deliberately coarse to avoid overfit."""
    try:
        t = ops.normalize(raw)
    except Exception:
        t = raw or ""
    if not t.strip():
        return 0.0

    words = re.findall(r"[A-Za-z']+", t)
    n_words = max(1, len(words))
    zone = (t[:900] + "\n" + t[-1100:]).lower()

    # Author intrusions (edit notes, apologies, self-promo) near head/tail.
    hits = sum(1 for p in _INTRUSION_PATTERNS if re.search(p, zone))
    intrusion_pen = min(0.30, 0.15 * hits)

    # Caps-lock shouting: runs of >=2 consecutive all-caps words.
    caps_runs = len(re.findall(r"\b[A-Z]{3,}(?:[\s,!?.:'\-]+[A-Z]{2,})+\b", t))
    caps_pen = min(0.15, 0.05 * caps_runs)

    # Exclamation density (per sentence).
    try:
        n_sent, _mwps, frac_long = ops.sent_stats(t)
        n_sent = max(1, int(n_sent))
        frac_long = float(frac_long or 0.0)
    except Exception:
        n_sent, frac_long = max(1, t.count(".")), 0.0
    ex_pen = min(0.12, 0.25 * max(0.0, (t.count("!") / n_sent) - 0.15))

    # Tamed thematic recurrence: abstract concepts that RECUR in the text.
    counts = Counter(w.lower() for w in words)
    recur = sum(1 for w in _ABSTRACT if counts.get(w, 0) >= 2)
    recur_bonus = min(0.12, 0.04 * recur)

    # Refrain / bookend: a substantial sentence repeated verbatim.
    sents = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) >= 25]
    refrain_bonus = 0.06 if any(v >= 2 for v in Counter(sents).values()) else 0.0

    # Tiny lexical-sophistication nudge.
    soph = max(0.0, min(0.06, (frac_long - 0.15) * 0.3))

    v = 0.5 - intrusion_pen - caps_pen - ex_pen + recur_bonus + refrain_bonus + soph
    return max(0.0, min(1.0, v))


def score(text, extracted, ops):
    try:
        raw = text or ""
        if not raw.strip():
            return 0.0
        ex = extracted or {}
        it = _intent_component(ex.get("intent", ""))
        th = _theme_component(ex.get("theme", ""))
        cc = _code_component(raw, ops)
        v = 0.42 * it + 0.32 * th + 0.26 * cc
        return max(0.0, min(1.0, v))
    except Exception:
        return 0.5
