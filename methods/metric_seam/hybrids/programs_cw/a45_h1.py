"""a45 hybrid (h1): Stakes clarity, vulnerability, and escalating cost.

Train-residual reading of h0's failures: every over-scored train cell is a
"serious"-toned genre piece (horror/sci-fi/pulp-adventure/twist-reveal) whose
LLM-extracted personal_stake is a BARE self-preservation noun -- "their
life", "his life and freedom", "their humanity" -- with no named relationship
attached. h0 treated any such bare survival mention exactly the same as a
genuinely relational stake (a specific loved one, a promise, a bond), giving
both full weight. But bare "their life is at risk" is present in nearly
every action/horror/sci-fi story ever written and is not, by itself, the
"personal stakes... emotionally charged... vulnerability" the criterion asks
for -- a NAMED relationship or bond is a much stronger, sharper signal of it.
h0 also rewarded loss-cue words landing in the story's final stretch
unconditionally; but several over-scored cells are simply abandoned mid-arc
("To be continued", trailing off) -- an unresolved cliffhanger is not a
"loss used purposefully".

h1 keeps h0's architecture (tone gates, stake grades, small surface
adjustments) and sharpens exactly these two general failure points:
  1. Split the old single "personal stake" tier into RELATIONAL (named other
     person/bond -> full weight) vs BARE-SELF (life/freedom/survival with no
     named relation -> discounted weight), instead of collapsing both to 1.0.
  2. Added a modest penalty for structurally unresolved/cliffhanger endings
     (ellipsis, trailing dash, "to be continued", ending on a bare question),
     since a story that never lands its threat cannot "purposefully use" a
     loss.
Everything else (tone gate, meta-chrome / wire-format surface penalties,
tail loss-cue bonus) is kept from h0, since those are not implicated by the
residual pattern and h0 already clears baseline with them in place.
"""

import math
import re

LLM_FIELDS = {
    "tone": (
        "In one word - comedic, mixed, or serious - what is this story's "
        "dominant tone?"
    ),
    "personal_stake": (
        "Who specifically (a named loved one or relationship) does the "
        "protagonist risk losing? If only their own bare survival/freedom "
        "is at risk with no named relationship, say SELF-ONLY. Or NONE."
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

# Tier 1: a NAMED relationship/bond -- the sharp, discriminating signal of
# personal, emotionally-charged vulnerability the criterion asks for.
_RELATIONAL_TERMS = re.compile(
    r"\b(daughter|son|child|children|kids?|baby|wife|husband|fianc\w*|"
    r"sister|brother|mother|father|mom|dad|parents?|family|families|"
    r"friend(s)?|lover|spouse|partner|soulmate|loved\s*one|"
    r"grandmother|grandfather|grandma|grandpa|crew(mate)?|team\s*mate)\b",
    re.I,
)
_SELF_ONLY_RE = re.compile(r"\bself[-\s]?only\b", re.I)
# Tier 2: bare self-preservation -- present in almost every action / horror /
# sci-fi story regardless of whether it is genuinely personal; discounted
# relative to a named relationship.
_BARE_SELF_TERMS = re.compile(
    r"\b(life|lives|death|die|dies|dying|sanity|mind|memory|memories|"
    r"identity|innocence|freedom|humanity|body|sight|voice|future|"
    r"himself|herself|themselves|survival|soul)\b",
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

# Structural markers of an abandoned / unresolved arc: teasing a cost but
# never landing it is not "loss used purposefully".
_CONTINUED_RE = re.compile(r"(?i)to\s+be\s+continued")
_TRAILING_RE = re.compile(r"(\.\.\.|…|--|—)\s*[\"'”’]?\s*$")


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
    if _RELATIONAL_TERMS.search(a):
        return 1.0
    if _SELF_ONLY_RE.search(a):
        return 0.45
    if _WORLD_TERMS.search(a):
        return 0.3
    if _BARE_SELF_TERMS.search(a):
        return 0.45  # bare survival/freedom, no named relationship
    return 0.55  # concrete but unlisted (possession, position, secret...)


def _unresolved_ending_penalty(t):
    tail = (t or "").rstrip()
    if not tail:
        return 0.0
    window = tail[-160:]
    if _CONTINUED_RE.search(window):
        return 0.08
    if _TRAILING_RE.search(tail):
        return 0.06
    if tail.endswith("?") or tail.endswith('?"') or tail.endswith("?”"):
        return 0.05
    return 0.0


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

        # Core predicate: seriousness gates stakes; relational/personal loss
        # grades it (bare self-survival is discounted vs. a named bond).
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

        # Abandoned / unresolved arc: a teased-but-never-paid cost is not a
        # loss "used purposefully".
        raw -= _unresolved_ending_penalty(t)

        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
