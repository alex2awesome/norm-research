"""Hybrid metric channel for a207: Conflict design, credibility, stakes, and escalation.

Design rationale (from train residuals):
  - Low judge band (0.0-0.1) is dominated by jokes/parodies/non-narratives whose
    conflict keywords fool the v0 baseline (listicle scored 0.5 by baseline, 0.05 by judge).
  - High band (0.5-0.85) = earnest stories where something a character loves is
    credibly endangered, costs are actually paid, and tension rises to a climax.
  - A comic story with a real escalation arc (crab president -> "Prepare the warships!")
    still scores 0.65, so escalation must be measured independently of tone.

Channel = 2 LLM fields (genuineness of conflict; worst credible loss) graded by
code severity tiers + a positional escalation gradient computed in code.
"""

import re
import math

LLM_FIELDS = {
    "conflict_type": (
        "Answer one word: GENUINE if the story's central conflict is a real struggle "
        "with credible opposition and consequences, JOKE if conflict is played for "
        "laughs or parody, NONE if there is no conflict."
    ),
    "worst_loss": (
        "In at most 12 words, name the worst thing a character loses or credibly "
        "risks losing by the story's end; answer NONE if nothing important is endangered."
    ),
}

# ---- severity tiers for the extracted "worst_loss" answer -------------------

_HIGH_STAKES = (
    "life", "lives", "death", "die", "dying", "dead", "kill", "murder",
    "daughter", "son", "child", "baby", "famil", "wife", "husband",
    "sister", "brother", "mother", "father", "soul", "world", "humanity",
    "planet", "mankind", "extinct", "war", "invasion", "invade", "enslav",
    "execut", "annihilat", "destro", "genocide", "sanity", "massacre",
    "slaughter", "eaten", "devour",
)

_MID_STAKES = (
    "freedom", "home", "love", "marriage", "memor", "identity", "job",
    "career", "trust", "friend", "reputation", "future", "purpose",
    "destiny", "deactivat", "capture", "captur", "imprison", "trapped",
    "control", "power", "kingdom", "throne", "crown", "magic", "secret",
    "innocence", "escape", "safety", "body", "mind", "humanity",
)

# ---- lexicon for the positional escalation gradient (code-side craft signal) --

_THREAT_STEMS = (
    "kill", "die", "dying", "dead", "death", "blood", "bleed", "fear",
    "afraid", "terror", "terrif", "scream", "threat", "danger", "risk",
    "weapon", "gun", "sword", "blade", "knife", "war", "attack", "fight",
    "fought", "chase", "chasing", "flee", "fled", "escape", "burn", "fire",
    "flame", "destroy", "murder", "stake", "monster", "lose", "losing",
    "lost", "cry", "crying", "tears", "wound", "pain", "hurt", "stab",
    "shot", "shoot", "corpse", "grave", "doom", "help",
)

_THREAT_RE = re.compile(
    r"\b(" + "|".join(_THREAT_STEMS) + r")[a-z]*\b", re.IGNORECASE
)

_WORD_RE = re.compile(r"[a-zA-Z']+")

# word-boundary prefix matchers (substring matching would let "nobody" hit "body")
_HIGH_RE = re.compile(r"\b(" + "|".join(_HIGH_STAKES) + r")[a-z]*")
_MID_RE = re.compile(r"\b(" + "|".join(_MID_STAKES) + r")[a-z]*")
_JOKE_RE = re.compile(
    r"\b(joke|laugh|parod|comic|comed|satir|humor|farc|absurd)[a-z]*")


def _clean(ans):
    a = (ans or "").lower()
    a = re.sub(r"[^a-z' ]+", " ", a).strip()
    return "" if a in ("", "none", "n a", "na", "nothing") else a


def _severity(ans):
    """Grade the extracted worst-loss answer into a [0,1] severity."""
    a = _clean(ans)
    if not a:
        return 0.0
    if _HIGH_RE.search(a):
        return 1.0
    if _MID_RE.search(a):
        return 0.6
    return 0.35  # named something, but nothing in our tiers


def _genuineness(ans):
    """Grade the extracted conflict-type answer into a [0,1] credibility."""
    a = _clean(ans)
    if not a:
        return 0.0
    # joke-check first so "played for laughs, not serious" lands as JOKE
    if _JOKE_RE.search(a):
        return 0.25
    if "genuine" in a or "real" in a or "serious" in a:
        return 1.0
    return 0.5  # answered, but off-menu: treat as ambiguous


def _escalation(text):
    """Positional craft signal: does threat/cost density rise toward the end,
    and is there an actual climax (threat mass in the final third)?"""
    words = _WORD_RE.findall(text.lower())
    n = len(words)
    if n < 40:
        return 0.0
    third = max(1, n // 3)
    chunks = [words[:third], words[third:2 * third], words[2 * third:]]
    dens = []
    for ch in chunks:
        joined = " ".join(ch)
        hits = len(_THREAT_RE.findall(joined))
        dens.append(100.0 * hits / max(1, len(ch)))
    d_first, d_last = dens[0], dens[2]
    total = sum(dens)
    if total < 0.15:  # essentially no conflict material anywhere
        return 0.05
    gradient = d_last - d_first  # per-100-words rise
    rise = 0.5 + 0.5 * math.tanh(gradient / 1.5)
    climax = min(1.0, d_last / 2.5)
    esc = 0.4 * rise + 0.6 * climax
    # very short pieces cannot build escalation
    esc *= min(1.0, n / 150.0)
    return max(0.0, min(1.0, esc))


def score(text, extracted, ops):
    try:
        try:
            t = ops.normalize(text or "")
        except Exception:
            t = text or ""
        if not (t or "").strip():
            return 0.0
        ext = extracted if isinstance(extracted, dict) else {}

        g = _genuineness(ext.get("conflict_type", ""))
        s = _severity(ext.get("worst_loss", ""))
        e = _escalation(t)

        # Fields carry the full-story grounding; code gradient arbitrates craft.
        val = 0.40 * g + 0.35 * s + 0.25 * e
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
