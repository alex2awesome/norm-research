"""Hybrid metric channel for a99: Sentence rhythm, variety, and musicality.

Code predicate measures the *shape* of the prose (sentence-length dispersion,
run-on / monotony / control failures, anaphora & refrain devices); two
constrained ordinal LLM fields ground the tacit part (deliberateness of the
rhythm, flow-breaking errors) that surface statistics cannot see.
"""

import re
import statistics

LLM_FIELDS = {
    "rhythm_control": (
        "Does the author vary sentence lengths and structures deliberately, "
        "with control (purposeful fragments, cadence, refrains)? Answer exactly one word: "
        "YES, SOMEWHAT, or NO."
    ),
    "flow_problems": (
        "Do run-on sentences, comma splices, missing punctuation, or grammar errors "
        "disrupt the prose flow? Answer exactly one word: FREQUENT, OCCASIONAL, or CLEAN."
    ),
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_CAPS_RE = re.compile(r"\b[A-Z]{4,}\b")
_BANGS_RE = re.compile(r"(!!|\?\?|\?!|!\?)")


def _band(x, lo0, lo1, hi1, hi0):
    """Trapezoid membership: 0 below lo0/above hi0, 1 in [lo1, hi1]."""
    if x <= lo0 or x >= hi0:
        return 0.0
    if lo1 <= x <= hi1:
        return 1.0
    if x < lo1:
        return (x - lo0) / max(1e-9, (lo1 - lo0))
    return (hi0 - x) / max(1e-9, (hi0 - hi1))


def _clean(text):
    # markdown / scrape artifacts
    t = text.replace("\r", "\n")
    t = re.sub(r"&(amp|gt|lt|nbsp|#x200B);", " ", t)
    t = re.sub(r"\*\*?|__|#+\[?\]?\(#\w*\)?", " ", t)
    t = t.replace("[...]", " ")
    t = re.sub(r"\[\s*(WP|Poem|EU|CW|PI|OT)\s*\]", " ", t, flags=re.I)
    # drop pure plug/link/author-note lines (subreddit shoutouts, edit notes)
    kept = []
    for ln in t.split("\n"):
        s = ln.strip()
        low = s.lower()
        if s and (re.match(r"^/?r/\w+$", s)
                  or low.startswith(("edit:", "update:", "part 2", "part 3"))
                  or ("feedback" in low and "welcome" in low)
                  or ("thanks for reading" in low)):
            continue
        kept.append(ln)
    return "\n".join(kept)


def _sentences(t):
    # ellipses become soft boundaries; then split on terminal punct or line breaks
    t = t.replace("…", "...")
    parts = re.split(r"(?:(?<=[.!?])|(?<=\.\.\.))[\"'”’)\]]*\s+|\n+", t)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        words = _WORD_RE.findall(p)
        if words:
            out.append((p, words))
    return out


def _code_score(text, ops):
    try:
        t = ops.normalize(text)
    except Exception:
        t = text
    t = _clean(t)
    sents = _sentences(t)
    n = len(sents)
    if n < 4:
        return 0.4
    lens = [len(w) for _, w in sents]
    total_words = max(1, sum(lens))
    m = statistics.fmean(lens)
    sd = statistics.pstdev(lens)
    cv = sd / max(1e-9, m)

    frac_short = sum(1 for L in lens if L <= 4) / n
    frac_vlong = sum(1 for L in lens if L >= 40) / n

    # --- positive shape signals ---
    s_mean = _band(m, 4.0, 8.0, 19.0, 33.0)          # comfortable cadence band
    s_cv = _band(cv, 0.25, 0.55, 1.05, 1.7)          # genuine length variety
    s_short = _band(frac_short, 0.0, 0.08, 0.38, 0.65)  # punchy fragments, not all-choppy

    # anaphora: consecutive sentences opening with the same two words
    ana = 0
    prev = None
    for _, words in sents:
        key = tuple(w.lower() for w in words[:2])
        if prev is not None and len(key) == 2 and key == prev:
            ana += 1
        prev = key
    s_ana = min(1.0, ana / 3.0)

    # refrain: a 4-12 word sentence repeated verbatim, non-adjacently
    seen = {}
    refrain = 0.0
    for i, (p, words) in enumerate(sents):
        if 4 <= len(words) <= 12:
            key = " ".join(w.lower() for w in words)
            if key in seen and i - seen[key] > 2:
                refrain = 1.0
            seen.setdefault(key, i)

    # measured use of dash / semicolon (density per 100 words)
    dashes = len(re.findall(r"—|–|--|;", t))
    s_punct = _band(100.0 * dashes / total_words, 0.0, 0.15, 2.5, 6.0)

    # --- penalties ---
    # run-ons: very long sentences, 'as'-chains, comma splices
    as_chain = sum(1 for p, w in sents if len(re.findall(r"\bas\b", p.lower())) >= 3
                   and len(w) > 25) / n
    splice = sum(1 for p, w in sents if len(w) > 30 and p.count(",") >= 4) / n
    pen_runon = min(1.0, 3.5 * frac_vlong + 4.0 * as_chain + 3.0 * splice
                    + max(0.0, (m - 24.0) / 18.0))

    # control failures: lowercase sentence-openers, lowercase 'i', caps abuse,
    # stacked !?, gross blank-line padding
    low_start = 0
    for p, _ in sents:
        mt = re.search(r"[A-Za-z]", p)
        if mt and mt.group(0).islower():
            low_start += 1
    low_frac = low_start / n
    lc_i = len(re.findall(r"\bi\b", t))
    caps = len(_CAPS_RE.findall(t)) / total_words
    bangs = len(_BANGS_RE.findall(t)) / n
    padding = len(re.findall(r"\n\s*\n\s*\n\s*\n", t))
    pen_ctrl = min(1.0,
                   max(0.0, low_frac - 0.22) * 2.2
                   + min(1.0, lc_i / 6.0) * 0.5
                   + max(0.0, caps - 0.012) * 22.0
                   + max(0.0, bangs - 0.10) * 2.0
                   + min(0.4, padding * 0.08))

    # monotony: near-uniform sentence lengths (verse-like or metronomic prose)
    pen_mono = 0.35 if cv < 0.38 else 0.0

    pos = (0.30 * s_cv + 0.22 * s_mean + 0.14 * s_short
           + 0.14 * s_ana + 0.10 * refrain + 0.10 * s_punct)
    score = 0.18 + 0.78 * pos - 0.38 * min(1.0, pen_runon + pen_ctrl + pen_mono)
    return min(1.0, max(0.0, score))


def _llm_score(extracted):
    vals = []
    r = (extracted.get("rhythm_control") or "").strip().lower()
    if r:
        if re.search(r"\byes\b", r):
            vals.append(1.0)
        elif "somewhat" in r or "partial" in r or "mixed" in r:
            vals.append(0.5)
        elif re.search(r"\bno\b", r):
            vals.append(0.0)
    f = (extracted.get("flow_problems") or "").strip().lower()
    if f:
        if "frequent" in f:
            vals.append(0.0)
        elif "occasional" in f:
            vals.append(0.55)
        elif "clean" in f:
            vals.append(1.0)
    if not vals:
        return None
    return sum(vals) / len(vals)


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        c = _code_score(text, ops)
        l = _llm_score(extracted if isinstance(extracted, dict) else {})
        if l is None:
            return float(c)
        return float(min(1.0, max(0.0, 0.45 * c + 0.55 * l)))
    except Exception:
        return 0.5
