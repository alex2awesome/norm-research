"""Hybrid metric channel for a153: Humor craft and tone (CW short fiction).

h0's failure mode (visible across the worst train residual cells): it treats
the humor_level label (COMEDY / LIGHT / SERIOUS) as a near-total determinant
of score (56% of the final weight), with a technique-NAME-count field
(comic_devices) as a minor top-up (22%), plus a code prior built from
laugh/joke/funny keyword density and "!"/dialogue counts (14%). That
architecture conflates COMEDIC INTENT with COMEDIC CRAFT: a story built on
profanity, shouted dialogue, and a single shock or gross-out beat still reads
as "COMEDY" to an extractor, its techniques are trivially nameable ("absurd
premise, irony, punchline" fits almost any twist-based story), and it
literally narrates people laughing/joking -- so h0's own humor-keyword prior
piles on top instead of discounting it. The result is systematic near-ceiling
scores (0.9+) for crude/juvenile pieces the judge rates 0.3-0.5. Symmetrically,
a piece that stays mostly SERIOUS/LIGHT but lands a genuinely controlled,
ironic beat gets almost no credit, because h0 has no channel for craft outside
the COMEDY bucket -- so a piece the judge rates 0.75 for a well-executed dark
irony can fall near 0.35.

h1's fix is architectural, not case-specific: decouple "is humor the mode"
from "is it executed with craft/heart", and make the second question its own
LLM-grounded signal that can move the score in EITHER direction independent
of the coarse level -- rewarding controlled craft even in a SERIOUS/LIGHT
piece, and penalizing crude/heavy-handed execution even inside a labeled
COMEDY piece. The old device-name-count field is dropped (naming techniques
is nearly free and doesn't discriminate skill); the old positive
humor-keyword code prior is dropped (it fires just as hard on crude pieces
that narrate "laughed"/"joked" a lot, so it can't discriminate craft from
crudeness either) and replaced with a small, sign-flipped surface prior that
only flags shouted/profanity-heavy delivery as a *discount*, never a bonus.
"""

import re

LLM_FIELDS = {
    "humor_level": (
        "Is intentional humor central to this story? Answer exactly one word: "
        "COMEDY (humor is the story's main mode), LIGHT (a few comic touches "
        "in a mostly serious story), or SERIOUS (no real humor)."
    ),
    "craft_register": (
        "In at most 10 words: is this story's humor/tone controlled and "
        "skillful (witty, restrained, ironic, balances humor with genuine "
        "emotional heart/depth), or is it crude and heavy-handed (shock "
        "value, gratuitous profanity or gross-out, yelling, one joke "
        "repeated)? Answer NONE if the question doesn't apply."
    ),
}

_LEVEL_HIGH = ("comedy", "comedic", "hilarious", "farce", "parody", "satire",
               "very funny", "humorous", "central")
_LEVEL_MID = ("light", "mild", "some", "slight", "occasional", "partly",
              "touches", "moments")
_LEVEL_LOW = ("serious", "none", "no humor", "no humour", "not funny",
              "drama", "dramatic", "dark", "grim", "tragic", "somber",
              "sombre", "horror", "no")

_CRAFT_SKILLED = (
    "witty", "wit", "clever", "controll", "restrain", "subtl", "understat",
    "deadpan", "dry", "iron", "nuanc", "elegan", "sharp", "heart", "depth",
    "poignan", "tender", "bittersweet", "balanc", "earned", "timing",
    "skill", "craft", "effective", "polish", "measur", "quiet", "subver",
)

_CRAFT_CRUDE = (
    "crude", "juvenile", "shock", "vulgar", "profan", "gratuit", "gross",
    "yell", "shout", "scream", "one-note", "one note", "heavy-hand",
    "heavy hand", "sophomor", "immatur", "cursing", "swearing", "explicit",
    "raunch", "puerile", "cheap", "gimmick", "repetit", "childish",
    "shallow", "forced", "unearned", "excessive",
)


def _parse_level(ans):
    """Map the humor_level answer to 0.0 / 0.5 / 1.0, or None if unparseable.

    Empty string means the extractor answered NONE, which for this question
    reads as 'no humor' -> 0.0.
    """
    s = re.sub(r"[^a-z ]", " ", (ans or "").strip().lower())
    s = " " + " ".join(s.split()) + " "
    if s.strip() == "":
        return 0.0
    for t in (" not comed", " no comed", " not a comed", " not funny",
              " no humor", " no humour", " not humor", " non comed"):
        if t in s:
            return 0.0
    for t in _LEVEL_HIGH:
        if t in s:
            return 1.0
    for t in _LEVEL_MID:
        if " " + t in s or s.startswith(t):
            return 0.5
    for t in _LEVEL_LOW:
        if " " + t + " " in s or s.startswith(t + " "):
            return 0.0
    if "funny" in s:
        return 1.0
    return None


def _craft_signal(ans):
    """Return (craft in [-1, 1], answered: bool).

    Positive = LLM describes controlled/witty/heartfelt execution.
    Negative = LLM describes crude/juvenile/shock-driven execution.
    This is the axis h0 lacked: it is independent of whether humor is the
    story's main mode, so it can reward craft in a SERIOUS/LIGHT piece and
    penalize crudeness in a COMEDY-labeled one.
    """
    s = (ans or "").strip().lower()
    if s == "" or s in ("n/a", "na", "no", "none") or s.startswith("none"):
        return 0.0, False
    skilled_hits = sum(1 for t in _CRAFT_SKILLED if t in s)
    crude_hits = sum(1 for t in _CRAFT_CRUDE if t in s)
    if skilled_hits == 0 and crude_hits == 0:
        return 0.0, True  # answered but uninformative on this axis
    raw = skilled_hits - crude_hits
    return max(-1.0, min(1.0, raw / 2.0)), True


def _sat(x, k):
    return 1.0 - 1.0 / (1.0 + max(0.0, x) / max(1e-6, k))


def _code_prior(text, ops):
    """Weak surface prior in [-1, 0]; tiebreak/fallback only, never a bonus.

    h0's code prior rewarded raw humor-keyword density ("laugh", "joke",
    "funny", ...) and "!"/dialogue density as POSITIVE evidence of craft.
    But crude pieces narrate laughing/joking just as often (arguably more),
    so that signal cannot discriminate craft from crudeness -- it only adds
    to the inflation. h1 keeps a surface prior but restricts it to detecting
    shouted delivery and gratuitous profanity, and only ever subtracts.
    """
    try:
        t = ops.normalize(text)
    except Exception:
        t = text
    w = max(1, len(re.findall(r"\w+", t)))
    excl_rate = t.count("!") / (w / 100.0)
    caps_words = re.findall(r"\b[A-Z]{4,}\b", t)
    caps_rate = len(caps_words) / (w / 100.0)
    shout = 0.5 * _sat(excl_rate, 1.5) + 0.5 * _sat(caps_rate, 1.0)
    prof = len(re.findall(
        r"\b(fuck\w*|shit\w*|bitch\w*|asshole\w*|goddamn\w*|damn\w*)\b",
        t.lower()))
    prof_rate = _sat(prof / (w / 100.0), 0.6)
    crude = 0.6 * shout + 0.4 * prof_rate
    return -max(0.0, min(1.0, crude))


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        extracted = extracted or {}
        lvl = _parse_level(extracted.get("humor_level", ""))
        if lvl is None:
            lvl = 0.5  # unparseable: neutral register, let craft/cp decide

        craft, craft_answered = _craft_signal(extracted.get("craft_register", ""))
        cp = _code_prior(text, ops)  # in [-1, 0]

        # Coarse register sets a baseline (SERIOUS ~0.12, LIGHT ~0.36,
        # COMEDY ~0.60); craft quality then moves the score substantially in
        # either direction, independent of that baseline.
        baseline = 0.12 + 0.48 * lvl
        if craft_answered:
            out = baseline + 0.30 * craft + 0.06 * cp
        else:
            out = baseline + 0.10 * cp

        return float(max(0.0, min(1.0, out)))
    except Exception:
        return 0.5
