"""p907_h1 — Criterion: Comprehensive detail (hybrid channel, volume-of-substance).

Round-1 refinement of p907_h0. Diagnosis of the h0 residual (all 12 divergent
train cells are UNDER-predictions, while all 6 anchors sit within ~0.01):

  The h0 volume->score log map is well calibrated wherever prose volume is
  measured correctly (the anchors prove this). The misses are a measurement
  failure on ONE document class: pages whose informative mass is ENUMERATIVE
  — product catalogs, headline / press-release indexes, dated release lists,
  spec-heavy Title-Case technical prose. h0's prose filters (>=8-word
  sentences, stopword fraction >= 0.18, capitalized fraction <= 0.60) discard
  that material wholesale, so such pages fall to (or near) the 0.0 floor and
  tie with true nav stubs — a large rank error under a Spearman gate.

General fix (one mechanism, no per-example patches):
  Keep h0's substantive-prose channel A untouched. Add channel B that credits
  enumerative content at a flat discount: words in sentences that FAIL the
  prose filters, inside lines that (a) contain >= 3 words and (b) do not
  start with nav-chrome vocabulary. Effective volume W = W_prose + 0.4 * W_enum
  feeds the SAME log map and the SAME mild density modulation as h0.
  The 0.4 discount ranks list mass below equal prose mass; the 120-word floor
  still zeroes genuine nav stubs (their non-chrome 3+-word lines are few).

Contract: LLM_FIELDS (empty) + score(text, extracted, ops) -> [0.0, 1.0];
deterministic; stdlib only; never raises (returns 0.5 on internal error).
"""

import math
import re

# No LLM fields: the judged quality (fullness / amount of informative detail)
# remains code-reachable from text volume after chrome filtering.
LLM_FIELDS = {}

# Function words, English + Spanish + other corpus languages (unchanged from h0).
_STOP = frozenset("""
a an the and or but if then than of to in on at by for with from as is are
was were be been being it its this that these those he she they we you i
his her their our your not no nor so what which who whom will would can
could shall should may might must do does did done has have had having
there here when where how why all any both each few more most other some
such only own same very just now also into over under after before between
during about against up down out off again once per
de la el en que y los las un una es se del por con para su al lo como mas
o sus le ya fue este ha si porque esta son entre cuando muy sin sobre ser
tiene tambien me hasta hay donde han quien estan desde todo nos durante
todos uno les ni contra otros ese eso ante ellos e esto mi antes algunos
unos yo otro otras otra tanto esa estos mucho quienes nada muchos cual
poco ella estar estas algunas algo nosotros
le les des du dans est une pour pas qui sur ne au ce il elle nous vous ils
elles sont ont aux cette ces leur leurs mais ou donc car chez
der die das und nicht ist sie ich ein eine mit auch auf werden sind wird
dem den einer eines im zum zur bei nach aus wie oder wenn dass kann
da do em um uma os nao com dos das pelo pela seu sua isso essa esse
""".split())

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]"
                      r"[A-Za-zÀ-ÖØ-öø-ÿ'\-]*")

# Nav-chrome vocabulary (unchanged from h0): lines that START with these are
# navigation, boilerplate, or social chrome — never informative detail.
_CHROME_LINE_RE = re.compile(
    r"(?i)^\s*(?:home|menu|search|subscribe|sign ?in|log ?in|register|"
    r"share(?: this)?(?: article)?|tweet|print|download|"
    r"contact(?: us| cision| pr newswire)?|about(?: us)?|"
    r"privacy(?: policy| statement)?|terms(?: of use| of service)?|"
    r"terms\s*(?:&(?:amp;)?|and)\s*conditions|cookies?(?: policy| settings)?|"
    r"site ?map|careers?|learn more|read more|see more|view all|watch now|"
    r"follow(?: us)?|connect with us|rss|copyright|all rights reserved|"
    r"skip to|newsroom|investors?|media|resources|products|services|"
    r"facebook|twitter|linkedin|youtube|instagram|"
    r"related|featured|explore|my services|quick links|helpful links|"
    r"e?mail|phone|tel|fax|back to|loading)\b")

_SENT_SPLIT_RE = re.compile(r"[.!?]+")

# Volume anchors (unchanged from h0): effective words -> [0,1] on a log scale.
_W_FLOOR = 120.0    # at or below this many effective words -> 0.0
_W_CEIL = 4500.0    # at or above this many effective words -> 1.0
_SENT_CAP = 80      # max words a single "sentence" may contribute
_MIN_SENT_WORDS = 8
_MIN_STOP_FRAC = 0.18
_MAX_CAP_FRAC = 0.60

# NEW: enumerative-content channel.
_ENUM_DISCOUNT = 0.4   # a list/headline/catalog word carries 0.4x a prose word
_MIN_ENUM_LINE_WORDS = 3  # 1-2 word lines are nav tokens ("Home", "About Us")


def _volumes(text):
    """Return (prose_word_count, enum_word_count, total_word_count).

    prose_word_count reproduces h0's substantive-word measurement exactly.
    enum_word_count collects words from sentences that FAIL the prose filters,
    inside lines that hold >= _MIN_ENUM_LINE_WORDS words and are not chrome —
    i.e., enumerated specifics (catalog entries, headline lists, spec prose).
    """
    total = 0
    prose = 0
    enum = 0
    for line in text.split("\n"):
        words_in_line = _WORD_RE.findall(line)
        n_line = len(words_in_line)
        total += n_line
        if n_line == 0:
            continue
        is_chrome = bool(_CHROME_LINE_RE.match(line))
        if n_line < 5:
            # Below h0's prose-line threshold: pure enum candidate.
            if n_line >= _MIN_ENUM_LINE_WORDS and not is_chrome:
                enum += min(n_line, _SENT_CAP)
            continue
        if n_line < 12 and is_chrome:
            continue  # short chrome line: no credit in either channel (as h0)
        for sent in _SENT_SPLIT_RE.split(line):
            words = _WORD_RE.findall(sent)
            n = len(words)
            if n == 0:
                continue
            keep_as_prose = False
            if n >= _MIN_SENT_WORDS:
                lowered = [w.lower() for w in words]
                stop = sum(1 for w in lowered if w in _STOP)
                if stop / float(n) >= _MIN_STOP_FRAC:
                    caps = sum(1 for w in words[1:] if w[:1].isupper())
                    if n <= 1 or caps / float(n - 1) <= _MAX_CAP_FRAC:
                        keep_as_prose = True
            if keep_as_prose:
                prose += min(n, _SENT_CAP)
            elif not is_chrome:
                # Failed prose filters -> enumerative credit (discounted later).
                enum += min(n, _SENT_CAP)
    return prose, enum, total


def score(text, extracted, ops):
    try:
        if not text:
            return 0.0
        try:
            t = ops.normalize(text)
            if not isinstance(t, str) or not t:
                t = text
        except Exception:
            t = text

        w_prose, w_enum, w_tot = _volumes(t)
        w_eff = w_prose + _ENUM_DISCOUNT * w_enum
        if w_eff <= 0 or w_tot <= 0:
            return 0.0

        # Log-scale effective volume (same map as h0).
        if w_eff <= _W_FLOOR:
            v = 0.0
        else:
            v = ((math.log(w_eff) - math.log(_W_FLOOR))
                 / (math.log(_W_CEIL) - math.log(_W_FLOOR)))
            v = max(0.0, min(1.0, v))

        # Mild density modulation (same as h0), on effective volume.
        den = w_eff / float(w_tot)
        dn = min(1.0, den / 0.65)
        out = v * (0.8 + 0.2 * dn)

        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5
