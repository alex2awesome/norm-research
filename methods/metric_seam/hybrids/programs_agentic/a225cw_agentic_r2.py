"""Hybrid metric channel for CW criterion a225: Sentence Rhythm, Variety, and Pacing.

Judge construct: deliberate variation in sentence length/structure that creates
rhythm, avoids monotony, and regulates line-level pace and emphasis.

r3 revision (full-train diagnostic, not just worst-residual eyeballing):

  r1 (h0) got train rho=0.598. r2's attempt to raise the code-side "punch" bar
  (short<=3/long>=16, higher rate) REGRESSED train rho to 0.501: a full
  univariate scan over all 150 train items showed h0's original punch
  definition (<=4-word sentence next to a >=12-word one) already has better
  standalone correlation with judge (rho=+0.204) than the tightened version
  (rho=+0.064) -- the residual-only view was misleading because the worst-15
  is a biased sample. Also on the full set, dialogue fraction `fq` has
  essentially ZERO raw correlation with judge score (rho=+0.010), so h1's
  dialogue-alternation discount (built from a residual-only read, not a
  full-train correlation check) was very likely fighting noise, not signal
  -- consistent with h1 itself regressing train rho to 0.576 vs h0's 0.598.
  This revision goes back to h0's plain code core (no dialogue discount, no
  tightened punch bar) and instead fixes the blend on top of it, guided by
  two full-train facts:

  1. A REAL BUG, not a design question: the prose_polish LLM field returns
     "" (per the module contract's "" == answered NONE) whenever the
     extractor finds no notable grammar/spelling/punctuation issues -- but
     h0/h1's `if pp:` guard treats "" as "field missing" and silently drops
     the whole prose_polish component for those documents. Checking the
     actual bucket means on the full train set: pp=="many" -> judge mean
     0.317 (n=24), pp=="few" -> 0.496 (n=107), pp=="" -> 0.613 (n=19, the
     HIGHEST of the three, and the literal string "none" never once occurs
     in this field -- "clean prose" always surfaces as ""). So "" is the
     single strongest prose_polish bucket and was being thrown away. Fixed:
     pp=="" now maps to a high "clean" value like the (dead, never-firing)
     "none" branch already intended.

  2. A REWEIGHTING, not a redesign: on the full train set, code alone
     (h0's _code_rhythm) reaches rho=0.386, but the rhythm_control LLM field
     ALONE reaches rho=0.562 -- markedly stronger than code, and stronger
     than the h0/h1 blend gives it credit for (weight 0.30 of a 0.55/0.30/0.15
     split). prose_polish alone (with the "" fix) reaches rho~0.51-0.52,
     also stronger than its 0.15 weight suggests. A weight grid search over
     (code, rhythm_control, prose_polish) on the SAME code core and SAME
     field-to-value mapping shape h0 already uses (just fixing the "" bug)
     found code~0.25 / rhythm_control~0.45 / prose_polish~0.30 clearly
     dominant over h0's 0.55/0.30/0.15, raising full-train rho from 0.598
     towards 0.70+. This is not a new construct or a borrowed pointer -- it
     is the same three ingredients h0 always used, correctly weighted by
     how much each one actually tracks the judge's rhythm construct, plus
     mildly softened controlled/varied anchor values (0.95/0.55 -> 0.78/0.38)
     so the LLM's coarse controlled/varied binary does not get treated as
     near-certain when it is really a strong-but-imperfect proxy.
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
# (unchanged from h0 -- the full-train scan showed h0's nonlinear clipped
#  combination already beats naive linear feature sums; the fix here is the
#  blend on top, not this core.)

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

        # code gets less weight than h0/h1 gave it: full-train rho for code
        # alone (0.386) is well below rhythm_control alone (0.562), so this
        # blend reflects each ingredient's actual discriminating power
        # instead of the earlier majority-on-code split. A local grid search
        # around the coarse optimum found a broad, stable plateau (not a
        # single spike) at roughly code~0.15-0.25 / rhythm_control~0.35-0.45
        # / prose_polish~0.35-0.40 -- these weights sit in the middle of
        # that plateau rather than at its (likely noisier) edge.
        comps = [(code, 0.20)]

        rc = str((extracted or {}).get("rhythm_control", "")).strip().lower()
        if rc:
            if "controlled" in rc or "deliberate" in rc:
                comps.append((0.78, 0.40))
            elif "monoton" in rc:
                comps.append((0.10, 0.40))
            elif "varied" in rc or "somewhat" in rc or "mixed" in rc:
                comps.append((0.38, 0.40))

        pp = str((extracted or {}).get("prose_polish", "")).strip().lower()
        if pp == "":
            # BUG FIX vs h0/h1: "" means the extractor answered NONE (no
            # notable mistakes), per the module contract -- it is the
            # single strongest prose_polish bucket on the full train set
            # (mean judge 0.613, higher than the "few" bucket's 0.496), not
            # a missing/uninformative field. h0/h1's `if pp:` guard threw
            # this away for every document with clean prose.
            comps.append((0.78, 0.35))
        elif "none" in pp or "no mistake" in pp or "clean" in pp:
            comps.append((0.80, 0.35))
        elif "many" in pp:
            comps.append((0.12, 0.35))
        elif "few" in pp or "some" in pp:
            comps.append((0.50, 0.35))

        tot = sum(w for _, w in comps)
        val = sum(v * w for v, w in comps) / tot
        return float(_clip(val))
    except Exception:
        return 0.5
