"""Hybrid metric channel for CW criterion a135: Foreshadowing and inevitable surprise.

Idea: judged foreshadowing is a STRUCTURAL property, not a lexical one (the
keyword baseline is ~0). Stories the judge rewards exhibit a plant->payoff
shape that is partly visible in code:
  (1) refrains: distinctive phrases repeated with wide positional span;
  (2) bookend echoes: rare content words in the opening that recur at the close;
  (3) dormant plants: words present early and late but absent in the middle;
  (4) enough narrative runway for a setup to be forgotten before it pays off.
Stories the judge zeroes are short dialogue skits / one-note jokes with none
of that structure, often with apologetic author notes.

The two LLM fields locate the plant (early) and the payoff (late); code then
VERIFIES both against the text (position + recurrence), so a confabulated
answer earns nothing. The predicate stays in code.
"""

import re
import math
from collections import Counter

LLM_FIELDS = {
    "planted_cue": (
        "Quote verbatim (3-8 words) the earliest detail or phrase that is "
        "planted early and later pays off or is echoed near the ending; "
        "answer NONE if nothing planted early pays off later."
    ),
    "ending_reveal": (
        "In at most 12 words, name what the ending reveals or recontextualizes "
        "that earlier parts secretly set up; answer NONE if the ending reveals "
        "nothing set up earlier."
    ),
}

_STOP = set(
    """the a an and or but if then of to in on at for with from by as is are was
    were be been being it its this that these those he she they them him his her
    their i you we my your our me us so not no nor do did does done doing have has
    had having will would could should shall may might must can than into out up
    down over under again very just about there here when where what who whom
    which why how all any both each few more most other some such only own same
    too also once during before after above below between while because until
    said says say went come came like get got go going one two back still even
    now never always really think know see look looked around never little"""
    .split()
)

# Apologetic / amateur author-note markers (class-level WritingPrompts noise;
# confident series-linking notes are deliberately NOT penalized).
_APOLOGY = re.compile(
    r"(?i)(hopefully you (?:liked|enjoyed)|don'?t judge|first (?:time )?post|"
    r"first prompt|long time lurker|not my first language|plot holes|"
    r"don'?t tear me apart|criticism (?:is )?welcome|sorry for (?:the )?"
    r"(?:typos|formatting|grammar)|i guess i gotta contribute)"
)


def _words(t):
    return re.findall(r"[a-z][a-z']*", t.lower())


def _content(w):
    return len(w) >= 4 and w not in _STOP


def _sat(x, k):
    """Saturating count -> [0,1)."""
    return 1.0 - math.exp(-max(0.0, float(x)) / k)


def _positions(toks):
    """word -> list of relative positions in [0,1]."""
    n = max(1, len(toks))
    pos = {}
    for i, w in enumerate(toks):
        pos.setdefault(w, []).append(i / n)
    return pos


def _refrain_score(toks):
    """Distinctive trigrams repeated with wide positional span."""
    n = len(toks)
    if n < 30:
        return 0.0
    tri = {}
    for i in range(n - 2):
        g = (toks[i], toks[i + 1], toks[i + 2])
        if not any(_content(w) for w in g):
            continue
        tri.setdefault(g, []).append(i / n)
    hits = set()
    for g, ps in tri.items():
        if len(ps) >= 2 and (max(ps) - min(ps)) >= 0.30:
            # anchor each refrain by its content words to avoid counting the
            # same repeated sentence many times via overlapping trigrams
            hits.add(tuple(sorted(w for w in g if _content(w))))
    return _sat(len(hits), 4.0)


def _bookend_score(pos, total):
    """Rare content words present in first 20% and last 20%."""
    cnt = 0
    for w, ps in pos.items():
        if not _content(w) or len(ps) > 4:
            continue
        if min(ps) <= 0.20 and max(ps) >= 0.80:
            cnt += 1
    return _sat(cnt, 3.0)


def _dormant_plant_score(pos):
    """Words planted early, silent in the middle, returning late."""
    cnt = 0
    for w, ps in pos.items():
        if not _content(w) or len(ps) > 3:
            continue
        early = any(p <= 0.25 for p in ps)
        late = any(p >= 0.75 for p in ps)
        mid = any(0.30 < p < 0.70 for p in ps)
        if early and late and not mid:
            cnt += 1
    return _sat(cnt, 2.5)


def _field_words(ans):
    return [w for w in _words(ans) if _content(w)]


def _verify_plant(ans, toks, pos):
    """Plant must first occur in the first half AND echo in the last 30%."""
    fw = _field_words(ans)
    if not fw:
        return 0.0
    present = [w for w in fw if w in pos]
    if len(present) < max(1, int(math.ceil(0.5 * len(fw)))):
        return 0.0  # answer not grounded in the text
    early_ok = any(min(pos[w]) <= 0.50 for w in present)
    echo_ok = any(max(pos[w]) >= 0.70 for w in present)
    if early_ok and echo_ok:
        return 1.0
    if early_ok or echo_ok:
        return 0.4
    return 0.0


def _verify_reveal(ans, pos):
    """Reveal must be grounded in the final 30% of the text."""
    fw = _field_words(ans)
    if not fw:
        return 0.0
    present = [w for w in fw if w in pos]
    if not present:
        return 0.0
    late = [w for w in present if max(pos[w]) >= 0.70]
    if not late:
        return 0.0
    # inevitable surprise: the revealed thing must have an early trace
    if any(min(pos[w]) <= 0.50 for w in late):
        return 1.0
    return 0.3


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        try:
            t = ops.normalize(text)
        except Exception:
            t = text
        toks = _words(t)
        n = len(toks)
        if n < 15:
            return 0.15
        pos = _positions(toks)

        # (1) narrative runway: room to plant, forget, and pay off
        runway = min(1.0, n / 450.0)

        # (2) code-detected plant->payoff structure
        refrain = _refrain_score(toks)
        bookend = _bookend_score(pos, n)
        dormant = _dormant_plant_score(pos)
        echo = min(1.0, 0.45 * refrain + 0.40 * bookend + 0.45 * dormant)

        # (3) LLM-located, code-verified plant and payoff
        ex = extracted if isinstance(extracted, dict) else {}
        plant = _verify_plant(str(ex.get("planted_cue", "") or ""), toks, pos)
        reveal = _verify_reveal(str(ex.get("ending_reveal", "") or ""), pos)
        # coherence bonus: plant and reveal about the same thing
        pw = set(_field_words(str(ex.get("planted_cue", "") or "")))
        rw = set(_field_words(str(ex.get("ending_reveal", "") or "")))
        pair = 1.0 if (plant > 0 and reveal > 0 and pw & rw) else 0.0

        # (4) apologetic author-note penalty (small, class-level)
        apology = 1.0 if _APOLOGY.search(t) else 0.0

        # foreshadowing needs distance: field bonuses only count when the
        # story has room for a plant to lie dormant before paying off
        dist = math.sqrt(runway)
        s = (
            0.10
            + 0.16 * runway
            + 0.30 * echo
            + (0.24 * plant + 0.14 * reveal + 0.06 * pair) * dist
            - 0.08 * apology
        )
        return float(max(0.0, min(1.0, s)))
    except Exception:
        return 0.5
