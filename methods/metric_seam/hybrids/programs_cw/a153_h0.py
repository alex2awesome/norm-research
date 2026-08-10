"""Hybrid metric channel for a153: Humor craft and tone (CW short fiction).

Judge structure on train is ~3-level: serious drama (0.0), comic touches in a
mixed-tone story (~0.4), full comedy executed with craft (~0.8). Surface
keywords are decoupled from the judged property (Clouseau parody judge=0.8 has
baseline 0.08; keyword-rich melodrama judge=0.0 has baseline 0.34), so the
thick input comes from two LLM extraction fields; the predicate stays in code:
robust parsing of the 3-way level, a device-count signal over a closed comedy
vocabulary, fusion weights, and a small code-cue prior used as within-level
tiebreak and as fallback when extraction fails.
"""

import re

LLM_FIELDS = {
    "humor_level": (
        "Is intentional humor central to this story? Answer exactly one word: "
        "COMEDY (humor is the story's main mode), LIGHT (a few comic touches "
        "in a mostly serious story), or SERIOUS (no real humor)."
    ),
    "comic_devices": (
        "In at most 8 words, name the comedy techniques this story actually "
        "uses (e.g. parody, absurd premise, deadpan, punchline, running gag); "
        "answer NONE if it is not comedic."
    ),
}

# Closed vocabulary of comedy techniques; code counts distinct matches.
_DEVICE_TERMS = (
    "parody", "satire", "satir", "spoof", "absurd", "deadpan", "punchline",
    "punch line", "iron", "exaggerat", "hyperbole", "slapstick", "wordplay",
    "word play", "pun", "wit", "banter", "timing", "incongru", "farce",
    "farcical", "dark comedy", "black humor", "black humour", "gallows",
    "self-deprecat", "understatement", "running gag", "gag", "one-liner",
    "twist", "juxtapos", "mock", "bathos", "anticlimax", "anti-climax",
    "subver", "comic", "comed", "humor", "humour",
)

_LEVEL_HIGH = ("comedy", "comedic", "hilarious", "farce", "parody", "satire",
               "very funny", "humorous", "central")
_LEVEL_MID = ("light", "mild", "some", "slight", "occasional", "partly",
              "touches", "moments")
_LEVEL_LOW = ("serious", "none", "no humor", "no humour", "not funny",
              "drama", "dramatic", "dark", "grim", "tragic", "somber",
              "sombre", "horror", "no")

_HUMOR_KW = ("laugh", "laughed", "laughing", "joke", "joking", "funny",
             "grin", "grinned", "chuckle", "snort", "giggle", "absurd",
             "ridiculous", "ironic", "sarcas", "smirk")


def _parse_level(ans):
    """Map the humor_level answer to 0.0 / 0.5 / 1.0, or None if unparseable.

    Empty string means the extractor answered NONE, which for this question
    reads as 'no humor' -> 0.0.
    """
    s = re.sub(r"[^a-z ]", " ", (ans or "").strip().lower())
    s = " " + " ".join(s.split()) + " "
    if s.strip() == "":
        return 0.0
    # Negated humor must be caught before HIGH terms ("not comedic" etc.).
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
    if "funny" in s:  # bare "funny" without qualifier
        return 1.0
    return None


def _device_signal(ans):
    """(signal in [0,1], answered: bool). Counts distinct known techniques."""
    s = (ans or "").strip().lower()
    if (s == "" or s in ("n/a", "na", "no") or s.startswith("none")
            or s.startswith("no ") or s.startswith("not ")):
        return 0.0, (s != "")
    hits = set()
    for t in _DEVICE_TERMS:
        if t in s:
            hits.add(t.split()[0][:6])  # collapse stem variants
    n = len(hits)
    if n == 0:
        # It answered something non-NONE we don't recognize: weak evidence
        # that some technique exists.
        return 0.3, True
    return min(1.0, 0.45 + 0.275 * min(n, 3)), True  # 1->0.725, 2->1.0 capped


def _sat(x, k):
    return 1.0 - 1.0 / (1.0 + max(0.0, x) / max(1e-6, k))


def _code_prior(text, ops):
    """Weak surface prior in [0,1]; tiebreak/fallback only."""
    try:
        t = ops.normalize(text)
    except Exception:
        t = text
    tl = t.lower()
    w = max(1, len(re.findall(r"\w+", tl)))
    kw = sum(len(re.findall(r"\b" + re.escape(k), tl)) for k in _HUMOR_KW)
    s1 = _sat(kw / (w / 100.0), 0.7)          # comic-marker density
    s2 = _sat(t.count("!") / (w / 100.0), 0.9)  # exclamatory delivery
    dlg = t.count('"') + t.count("“") + t.count("”")
    s3 = _sat(dlg / (w / 100.0), 1.2)          # dialogue density
    return 0.5 * s1 + 0.25 * s2 + 0.25 * s3


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        extracted = extracted or {}
        lvl = _parse_level(extracted.get("humor_level", ""))
        dev, dev_answered = _device_signal(extracted.get("comic_devices", ""))
        cp = _code_prior(text, ops)

        if lvl is not None:
            llm = 0.72 * lvl + 0.28 * dev
        elif dev_answered:
            llm = dev
        else:
            # Both fields failed: mid-range, code-driven ordering only.
            return float(max(0.0, min(1.0, 0.30 + 0.40 * cp)))

        out = 0.08 + 0.78 * llm + 0.14 * cp
        return float(max(0.0, min(1.0, out)))
    except Exception:
        return 0.5
