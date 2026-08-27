"""a45 hybrid: Stakes clarity, vulnerability, and escalating cost.

Train-residual reading: every judge=0.0 example is comedy/parody or a
distanced document format (news parody, history log, joke speech, meta
author notes); high scorers (0.6-0.9) are serious stories where the
protagonist personally risks or pays a concrete, irreversible cost.
Keyword stakes-lexicons fail (baseline rho ~= 0.06) because comedic pieces
are saturated with death/kill vocabulary. So the predicate is
tone-seriousness x personal-loss concreteness: two short LLM fields carry
the tacit part; code keeps the gate, the grading, and small surface
penalties/bonuses (author meta-notes, wire-format chrome, late-story
loss-realization cues).
"""

import math
import re

LLM_FIELDS = {
    "tone": (
        "In one word - comedic, mixed, or serious - what is this story's "
        "dominant tone?"
    ),
    "personal_stake": (
        "What irreplaceable thing does the protagonist personally risk or "
        "lose (a loved one, their life, freedom)? Briefly, or NONE."
    ),
}

_COMEDIC_CUES = (
    "comed", "humor", "humour", "parod", "satir", "absurd", "farc",
    "joke", "silly", "whimsic", "light-hearted", "lighthearted",
    "playful", "irrever", "tongue-in-cheek",
)
_SERIOUS_CUES = (
    "serious", "dark", "dramat", "tragic", "grim", "somber", "sombre",
    "tense", "melanchol", "horror", "poignant", "earnest", "heavy",
    "solemn", "bleak", "suspense",
)

_NONE_RE = re.compile(r"^\s*(none|nothing|n/?a|no(ne)?\.?|unknown)\s*\.?\s*$", re.I)

_PERSONAL_TERMS = re.compile(
    r"\b(life|lives|death|die|dies|dying|daughter|son|child|children|baby|"
    r"wife|husband|fianc|sister|brother|mother|father|parents?|family|"
    r"friend|lover|love(d)?\s*one|love|marriage|soulmate|soul|sanity|mind|"
    r"memor(y|ies)|identity|humanity(?!'s\s)|innocence|freedom|home|body|"
    r"sight|voice|future|himself|herself)\b",
    re.I,
)
_WORLD_TERMS = re.compile(
    r"\b(world|mankind|humankind|humanity's|earth|planet|city|kingdom|"
    r"empire|country|nation|universe|galaxy|species|civilization|"
    r"civilisation|the\s+war)\b",
    re.I,
)

_META_PATTERNS = (
    re.compile(r"(?im)^\s*(final\s+)*edit\s*\d*\s*:"),
    re.compile(r"(?i)\bon (my|a|mobile) phone\b"),
    re.compile(r"(?i)reddit\.com|\br/[A-Za-z_]{3,}\b|\[\s*wp\s*\]"),
    re.compile(r"(?i)\bsubscribe\b|thanks for (the gold|reading)"),
    re.compile(r"(?i)sorry for (bad|the) (formatting|grammar|typos)"),
    re.compile(r"(?i)don'?t judge( the)? typos"),
)
_WIRE_PATTERNS = (
    re.compile(r"\((AP|Reuters|AFP)\)"),
    re.compile(r"(?im)^\s*\*{0,2}(to|from|cc|subject)\*{0,2}\s*:"),
    re.compile(r"(?im)^\s*(accessing|entry\s+\d|.{0,40}history log)"),
)

# Late-story loss-realization / escalation cues (costs coming due).
_LOSS_CUES = re.compile(
    r"(?i)\btoo late\b|\bgoodbye\b|\bnever (again|see|hear)\b|\bgone\b|"
    r"\bi'?m sorry\b|\bforgive me\b|\bwept\b|\bcried\b|\btears\b|"
    r"\blast (time|breath|words?)\b|\bdied\b|\bdead\b|\blost\b|\bsacrifice"
)


def _tone_weight(ans):
    a = (ans or "").strip().lower()
    if not a:
        return 0.5
    c = any(k in a for k in _COMEDIC_CUES)
    s = any(k in a for k in _SERIOUS_CUES)
    if c and s:
        return 0.5
    if c:
        return 0.0
    if s:
        return 1.0
    if "mixed" in a or "both" in a:
        return 0.5
    return 0.5


def _stake_weight(ans):
    a = (ans or "").strip()
    if not a or _NONE_RE.match(a):
        return 0.0
    personal = bool(_PERSONAL_TERMS.search(a))
    world = bool(_WORLD_TERMS.search(a))
    if personal:
        return 1.0
    if world:
        return 0.35  # global/abstract stakes read as weaker than personal
    return 0.6  # concrete but unlisted (possession, position, secret...)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = text or ""
        try:
            t = ops.normalize(t)
        except Exception:
            pass
        if not t.strip():
            return 0.0

        ex = extracted or {}
        tone = _tone_weight(ex.get("tone", ""))
        stake = _stake_weight(ex.get("personal_stake", ""))

        # Core predicate: seriousness gates stakes; personal loss grades it.
        raw = 0.05 + 0.22 * tone + 0.55 * tone * stake

        # Escalation/realization bonus: loss cues concentrated late.
        tail = t[int(len(t) * 0.7):] or t
        hits = len(_LOSS_CUES.findall(tail))
        raw += 0.08 * (1.0 - math.exp(-hits / 2.0))

        # Author meta-chrome penalty (edit notes, phone apologies, sub plugs).
        meta = sum(1 for p in _META_PATTERNS if p.search(t))
        raw -= 0.05 * min(1.0, meta / 2.0 + (0.5 if meta else 0.0))

        # Distanced wire/log/memo formats rarely carry felt personal stakes.
        head = t[:400]
        if any(p.search(head) for p in _WIRE_PATTERNS):
            raw -= 0.04

        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
