"""Hybrid metric channel for CW criterion a225: Sentence Rhythm, Variety, and Pacing.

Judge construct: deliberate variation in sentence length/structure that creates
rhythm, avoids monotony, and regulates line-level pace and emphasis.

h1 revision (from train-residual study of h0):
  h0's length-variability / short-vs-long "punch" signal is confounded by
  QUOTED DIALOGUE EXCHANGE. Back-and-forth spoken dialogue (short retorts,
  "I asked."/"she said." tags, one line per turn) naturally alternates short
  and long lines purely as a byproduct of conversational turn-taking and
  paragraph-per-speaker formatting -- not because the author is shaping
  narrative-PROSE rhythm for pace/emphasis. h0 systematically over-scored
  dialogue-heavy scenes (interrogations, arguments, dinner-table exchanges)
  that the judge rated as merely ordinary or weak on this criterion.
  A second, smaller confound: verbatim/near-verbatim repeated short lines
  (refrains) inflate the same short-sentence / contrast signals without
  contributing any real sentence-to-sentence VARIETY.

  Fix (general, not keyed to any excerpt):
    1. Collapse consecutive near-duplicate sentences before computing any
       length statistics, so a repeated refrain counts once, not N times.
    2. Compute the fraction of sentences that are quoted dialogue (fq) and
       continuously discount the contrast/punch "base" score by how
       dialogue-dominated the sentence stream is. Mostly-narrative prose is
       untouched; heavily-dialogue passages get materially less credit for
       the same raw length spread, since that spread is weaker evidence of
       deliberate rhythm craft.
    3. The monotone-penalty (low coefficient of variation) is only charged
       when the passage isn't already dialogue-discounted, since "low
       variance dialogue" and "monotone prose/verse" are not the same
       failure mode and shouldn't be double-punished.
  The two LLM fields are unchanged in shape; the rhythm_control instruction
  is revised to ask the extractor to judge NARRATIVE sentences and
  explicitly discount ordinary spoken back-and-forth, since the LLM field
  showed the identical dialogue-alternation confound as the code signal.
"""

import math
import re
import statistics

LLM_FIELDS = {
    "rhythm_control": (
        "Considering only the narrative prose sentences (ignore ordinary "
        "back-and-forth spoken dialogue, which alternates short/long lines "
        "for conversational reasons, not craft): is sentence-length variation "
        "deliberate and controlled for emphasis, somewhat varied, or "
        "monotonous? Answer one word: controlled, varied, or monotonous."
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
_NON_WORD = re.compile(r"[^a-z0-9]+")


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
    """Sentence word-counts + quote flags + a normalized key (for dedup).
    Newlines always end a sentence (hard wraps, verse lines, chat lines)."""
    lens, quoted, keys = [], [], []
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
            keys.append(_NON_WORD.sub("", s.lower()))
    return lens, quoted, keys


def _collapse_repeats(lens, quoted, keys):
    """Collapse runs of consecutive near-identical sentences (verbatim
    refrains) to a single occurrence. A phrase repeated back-to-back for
    effect contributes real length-contrast signal only once -- counting it
    N times inflates short-sentence / punch stats without adding any actual
    sentence-to-sentence variety."""
    if not lens:
        return lens, quoted
    out_lens, out_quoted = [lens[0]], [quoted[0]]
    for i in range(1, len(lens)):
        if keys[i] and keys[i] == keys[i - 1]:
            continue
        out_lens.append(lens[i])
        out_quoted.append(quoted[i])
    return out_lens, out_quoted


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
    raw_lens, raw_quoted, keys = _sentences(text)
    lens, quoted = _collapse_repeats(raw_lens, raw_quoted, keys)
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

    # Dialogue-alternation discount. Quoted back-and-forth exchange alternates
    # short retorts with longer lines/tags as a byproduct of conversational
    # turn-taking and dialogue-tag formatting, not deliberate narrative-prose
    # rhythm control -- the construct this metric targets. The more of the
    # sentence stream that is quoted dialogue, the less the raw contrast
    # signal above should be trusted as evidence of authorial rhythm craft.
    fq = (sum(1 for q in quoted if q) / n) if n else 0.0
    dlg_discount = _clip((fq - 0.25) / 0.55)
    base *= (1.0 - 0.70 * dlg_discount)

    # penalties
    pen = 0.0
    pen += _clip(3.0 * frac_vlong, 0.0, 0.30)                # unbroken run-ons
    pen += _clip(2.0 * (frac_long - 0.42), 0.0, 0.30)        # long-winded, no relief
    as_rate = 100.0 * len(re.findall(r" as ", text)) / n_words
    pen += _clip(0.22 * (as_rate - 1.6), 0.0, 0.30)          # "as ... as" chains
    pen += _clip(0.08 * (_typo_rate(text, n_words) - 0.35), 0.0, 0.30)
    if cv < 0.35 and dlg_discount < 0.3:
        # "low variance" is only a monotone-prose/verse penalty when it isn't
        # already explained (and discounted) by dialogue exchange, which is a
        # different phenomenon and shouldn't be punished twice.
        pen += _clip(1.5 * (0.35 - cv), 0.0, 0.25)

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
