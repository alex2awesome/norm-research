"""p907_h0 — Criterion: Comprehensive detail (hybrid channel, volume-of-substance).

Design: on this criterion the judge tracks the sheer VOLUME of substantive
prose in the document (log-scale), largely independent of genre, language,
or PR-keyword density. Terse stubs and nav-chrome pages score ~0; medium
releases ~0.5; very long prose-dense documents (FAQs, leaflets, benefit
pages, full releases) score ~1.0. The v0 keyword baseline saturates early
on keyword-dense SHORT releases and under-scores long plain-prose docs —
that is the residual this program targets.

Predicate (pure code, no LLM fields):
  1. ops.normalize(text) to fix mojibake.
  2. Line/sentence filter to isolate substantive prose:
     - drop lines with < 5 words (nav tokens, headings, link lists);
     - drop short lines matching nav-chrome vocabulary;
     - within kept lines, keep sentences with >= 8 words, a function-word
       fraction >= 0.18 (English + Spanish stopwords, so non-English prose
       still counts), and a capitalized-word fraction <= 0.60 (kills
       Title-Case link salads that lack sentence punctuation).
  3. W = total substantive words (per-sentence contribution capped at 80).
     v = log-scale map of W with anchors ~120 words -> 0.0, ~4500 -> 1.0.
  4. Mild density modulation: score = v * (0.8 + 0.2 * min(1, den/0.65))
     where den = W / total_words — separates clean long releases from
     equally long but chrome-dominated pages, without zeroing them.

Contract: LLM_FIELDS (empty) + score(text, extracted, ops) -> [0.0, 1.0];
deterministic; stdlib only; never raises (returns 0.5 on internal error).
"""

import math
import re

# No LLM fields: the judged quality here (fullness / amount of informative
# detail) is code-reachable from text volume after chrome filtering.
LLM_FIELDS = {}

# Function words, English + Spanish (corpus contains Spanish documents that
# the judge scores on equal footing). Used to tell running prose from
# navigation/link word-salad.
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

# Nav-chrome vocabulary: lines that START with these and are short are
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

# Volume anchors (substantive words -> [0,1] on a log scale).
_W_FLOOR = 120.0    # at or below this many substantive words -> 0.0
_W_CEIL = 4500.0    # at or above this many substantive words -> 1.0
_SENT_CAP = 80      # max words a single "sentence" may contribute
_MIN_SENT_WORDS = 8
_MIN_STOP_FRAC = 0.18
_MAX_CAP_FRAC = 0.60


def _substantive_words(text):
    """Return (substantive_word_count, total_word_count)."""
    total = 0
    kept = 0
    for line in text.split("\n"):
        words_in_line = _WORD_RE.findall(line)
        total += len(words_in_line)
        n_line = len(words_in_line)
        if n_line < 5:
            continue
        if n_line < 12 and _CHROME_LINE_RE.match(line):
            continue
        for sent in _SENT_SPLIT_RE.split(line):
            words = _WORD_RE.findall(sent)
            n = len(words)
            if n < _MIN_SENT_WORDS:
                continue
            lowered = [w.lower() for w in words]
            stop = sum(1 for w in lowered if w in _STOP)
            if stop / float(n) < _MIN_STOP_FRAC:
                continue
            # Title-Case / ALL-CAPS link salad guard (skip first word).
            caps = sum(1 for w in words[1:] if w[:1].isupper())
            if n > 1 and caps / float(n - 1) > _MAX_CAP_FRAC:
                continue
            kept += min(n, _SENT_CAP)
    return kept, total


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

        w_sub, w_tot = _substantive_words(t)
        if w_sub <= 0 or w_tot <= 0:
            return 0.0

        # Log-scale volume of substantive prose.
        if w_sub <= _W_FLOOR:
            v = 0.0
        else:
            v = ((math.log(w_sub) - math.log(_W_FLOOR))
                 / (math.log(_W_CEIL) - math.log(_W_FLOOR)))
            v = max(0.0, min(1.0, v))

        # Mild density modulation: chrome-dominated pages carry less
        # judged detail than equally long clean prose.
        den = w_sub / float(w_tot)
        dn = min(1.0, den / 0.65)
        out = v * (0.8 + 0.2 * dn)

        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5
