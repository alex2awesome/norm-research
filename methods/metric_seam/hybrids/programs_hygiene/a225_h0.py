"""Hybrid metric channel for CW criterion a225: Sentence Rhythm, Variety, and Pacing.

Judge construct: deliberate variation in sentence length/structure that creates
rhythm, avoids monotony, and regulates line-level pace and emphasis.

Design (from train-residual study):
  High band (0.8-0.9): short punch sentences ("Nothing." / "And he was dead.")
    set against longer flowing sentences; wide sentence-length range; clean prose.
  Low band (0.2-0.3): EITHER unbroken run-on chains (comma/"as" chains, weak
    punctuation control, typo-riddled) OR uniformly flat mid-length sentences /
    metered verse (low variance = monotone).
  => Predicate: length VARIABILITY + short-vs-long CONTRAST + PUNCH placement,
     minus run-on and craft (typo) penalties. Punctuation-presence alone (the
     v0 baseline) is a weak proxy and is not used.
  Two LLM fields supply the tacit part code cannot see on a noisy scrape:
  whether the variation reads as deliberate, and gross prose-error level.
"""

import math
import re
import statistics

LLM_FIELDS = {
    "rhythm_control": (
        "Considering only sentence rhythm: is sentence-length variation deliberate "
        "and controlled for emphasis, somewhat varied, or monotonous? Answer one "
        "word: controlled, varied, or monotonous."
    ),
    "prose_polish": (
        "Ignoring slang inside dialogue, does the prose have grammar, spelling, or "
        "punctuation mistakes? Answer one word: none, few, or many."
    ),
}

# ---------------------------------------------------------------- helpers

_TAIL_NOTE = re.compile(
    r"(?i)(thanks? for reading|feel free to check|check out|feedback (is )?welcome"
    r"|my first post|word count:|part \d+ (is|in)|http[s]?://|/r/\w+|r/\w+"
    r"|^edit\s*:|subreddit)"
)
_MD_JUNK = re.compile(r"(&amp;?#x200b;?|&amp;nbsp;?|&gt;|&amp;|[*_~`]+|\\-+|\[wp\]|\[.*?\]\(.*?\))",
                      re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9'’-]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _strip_notes(text):
    """Drop author-note / link paragraphs (mostly near the end) and md junk."""
    paras = re.split(r"\n\s*\n", text)
    kept = []
    n = len(paras)
    for i, p in enumerate(paras):
        if i >= n - 3 or i == 0:  # notes live at the very start or end
            if _TAIL_NOTE.search(p):
                continue
        kept.append(p)
    out = "\n\n".join(kept if kept else paras)
    out = _MD_JUNK.sub(" ", out)
    return out


def _sentences(text):
    """Sentence word-counts + quote flags. Newlines always end a sentence
    (hard wraps, verse lines, chat lines)."""
    lens, quoted = [], []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in _SENT_SPLIT.split(line):
            w = _WORD.findall(s)
            if not w:
                continue
            lens.append(len(w))
            quoted.append(s.lstrip().startswith(('"', "“", "'", "‘")))
    return lens, quoted


def _typo_rate(text, n_words):
    """Craft signal: rates (per 100 words) of mechanical errors typical of the
    low band: lowercase pronoun i, missing space after ., space before comma,
    lowercase sentence starts."""
    if n_words <= 0:
        return 0.0
    low_i = len(re.findall(r"(?<![A-Za-z0-9])i(?=[ ,.'])", text))
    glued = len(re.findall(r"[a-z]{2}[.!?][A-Za-z]{2}", text))
    spaced_punct = len(re.findall(r"[A-Za-z] [,.;]", text))
    low_start = len(re.findall(r"(?<!\.)[.!?] [a-z]{2}", text))
    return 100.0 * (low_i + glued + spaced_punct + 0.5 * low_start) / n_words


# ---------------------------------------------------------------- code core

def _code_rhythm(text, ops):
    try:
        text = ops.normalize(text)
    except Exception:
        pass
    text = _strip_notes(text)
    lens, quoted = _sentences(text)
    n = len(lens)
    n_words = sum(lens)
    if n < 5 or n_words < 40:
        return 0.4  # too little prose to exhibit controlled rhythm

    mean = n_words / n
    sd = statistics.pstdev(lens)
    cv = sd / mean if mean else 0.0
    # local alternation: successive-difference contrast (rhythm, not just spread)
    succ = statistics.mean(abs(lens[i + 1] - lens[i]) for i in range(n - 1)) / mean

    frac_short = sum(1 for x in lens if x <= 4) / n
    frac_long = sum(1 for x in lens if x >= 18) / n
    frac_vlong = sum(1 for x in lens if x >= 45) / n

    # punch: a <=4-word sentence adjacent to a >=12-word sentence
    punch = 0
    for i, x in enumerate(lens):
        if x <= 4:
            near = []
            if i > 0:
                near.append(lens[i - 1])
            if i + 1 < n:
                near.append(lens[i + 1])
            if near and max(near) >= 12:
                punch += 1
    punch_sig = _clip(punch / max(3.0, 0.08 * n))

    cv_sig = _clip((cv - 0.35) / 0.45)
    succ_sig = _clip((succ - 0.35) / 0.50)
    # contrast needs both ends of the range represented; a heavy staccato mode
    # (many fragments WITH real punch placement) also counts as the "far end"
    staccato = _clip((frac_short - 0.18) / 0.25) * punch_sig
    range_sig = _clip(frac_short / 0.10) * max(_clip(frac_long / 0.10), staccato)

    base = 0.30 * cv_sig + 0.22 * succ_sig + 0.28 * punch_sig + 0.20 * range_sig

    # penalties
    pen = 0.0
    pen += _clip(3.0 * frac_vlong, 0.0, 0.30)                # unbroken run-ons
    pen += _clip(2.0 * (frac_long - 0.42), 0.0, 0.30)        # long-winded, no relief
    as_rate = 100.0 * len(re.findall(r" as ", text)) / n_words
    pen += _clip(0.22 * (as_rate - 1.6), 0.0, 0.30)          # "as ... as" chains
    pen += _clip(0.08 * (_typo_rate(text, n_words) - 0.35), 0.0, 0.30)
    if cv < 0.35:                                            # monotone / verse
        pen += _clip(1.5 * (0.35 - cv), 0.0, 0.25)
    fq = (sum(1 for q in quoted if q) / n) if n else 0.0
    if fq > 0.65:                                            # wall-to-wall dialogue
        pen += _clip(0.8 * (fq - 0.65), 0.0, 0.15)

    return _clip(0.05 + 0.95 * base - pen)


# ---------------------------------------------------------------- interface

def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        code = _code_rhythm(text, ops)

        comps = [(code, 0.55)]

        rc = str((extracted or {}).get("rhythm_control", "")).strip().lower()
        if rc:
            if "controlled" in rc or "deliberate" in rc:
                comps.append((0.95, 0.30))
            elif "monoton" in rc:
                comps.append((0.10, 0.30))
            elif "varied" in rc or "somewhat" in rc or "mixed" in rc:
                comps.append((0.55, 0.30))

        pp = str((extracted or {}).get("prose_polish", "")).strip().lower()
        if pp:
            if "none" in pp or "no mistake" in pp or re.search(r"\bclean\b", pp):
                comps.append((0.85, 0.15))
            elif "many" in pp:
                comps.append((0.10, 0.15))
            elif "few" in pp or re.search(r"\bsome\b", pp):
                comps.append((0.50, 0.15))

        tot = sum(w for _, w in comps)
        val = sum(v * w for v, w in comps) / tot
        return float(_clip(val))
    except Exception:
        return 0.5
