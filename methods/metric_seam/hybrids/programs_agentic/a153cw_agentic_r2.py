"""a153: Humor craft and tone (CW short fiction). Agentic candidate, round 2.

Reuses h0's two existing LLM fields BYTE-IDENTICAL (humor_level, comic_devices)
so this candidate has real train signal from round 1 -- h1's rewrite swapped
in a brand-new `craft_register` field that arrives empty during iteration and
scored WORSE than h0 on train (0.658 vs h0's 0.728, measured via this same
harness): a cautionary tale about borrowing an untested pointer instead of
extending real code structure. This candidate declares NO new fields.

Diagnosis, from running h0 itself through agentic_run.py and reading its own
worst-residual list (train rho 0.7280):

1. h0's blend (llm = 0.72*lvl + 0.28*dev) makes `humor_level` an almost-hard
   gate. ANY COMEDY-labeled item with >=1 named device lands at out~0.85-0.97
   NO MATTER what those devices are or how well the joke lands, so h0 cannot
   tell a landed comic piece (d02932 'misdirection, punchline' judge=0.80,
   d01826 'absurd premise, deadpan, juxtaposition' judge=0.80) apart from a
   comedy-labeled piece the judge rates low because the execution is crude or
   shock-driven: d01747 devices literally name 'shock humor' (judge=0.30),
   d02459 'wordplay and slapstick' over a crude phallus gag (judge=0.30),
   d00593 profanity-screaming Satan/Buddha fight (judge=0.40), d03837 '"You
   Bastard!" I screamed' demon scene (judge=0.35), d01610 'double entendre'
   wedding toast (judge=0.30). Device NAMES alone don't discriminate skill --
   'absurd premise'/'irony'/'juxtaposition' appear on BOTH the judge=0.30 and
   judge=0.80 items -- but crude DELIVERY is visible in the raw text
   (profanity, ALL-CAPS shouting) and is sometimes even named directly by the
   extractor itself ('shock humor', 'slapstick', 'double entendre').

2. Symmetrically h0 floors LIGHT (out~0.36-0.58) and SERIOUS (out~0.08-0.22)
   almost regardless of how many devices are actually named, because lvl
   dominates the blend at 0.72 weight. A piece the extractor mislabels
   SERIOUS but still names real devices for (d02485 'Dark humor, subversion,
   punchline', judge=0.75) or labels LIGHT with real craft (d02932, d01887,
   d01155, d01826 -- all judge 0.70-0.80) gets capped far below the judge.

3. h0 has NO channel for "heart" even though the criterion text explicitly
   asks for humor "balancing with heart to add depth" (the ORIGINAL code
   baseline had a HEART keyword term that h0 dropped entirely). The
   best-judged comic pieces in the pack consistently pair the joke with a
   warm beat (Mario/Luigi birthday-party kindness, the hairy-hand
   mother/daughter goodnight scene, the devil hosting Canasta, the Sauronoid
   Thanksgiving letter, the Death/Moira ending) -- h0 gives that zero credit.

Fix, same two fields, three targeted additions:
  - `backbone`: blend lvl and dev so dev's `max(lvl, dev)` term can pull the
    backbone UP when real devices are named under a missed SERIOUS/LIGHT
    label, instead of lvl gating dev down to nothing (fixes #2). NOTE: a
    narrower version tried, that only rescued the SERIOUS+strong-device case,
    was measured (via this harness) to score WORSE than even h0 (0.7147 vs
    h0's 0.7280) -- the same generic device names ("absurd premise", "irony")
    turn out to be just as unreliable paired with SERIOUS as with LIGHT (most
    SERIOUS+"absurd premise, irony" items are genuinely low judge, 0.20-0.30,
    not mislabels), so narrowing the rescue traded one fixed item (d02485)
    for several new SERIOUS-bucket overshoots. The blanket max(lvl_c, dev)
    version measured higher (0.7751) and is kept, with the overshoot it does
    cause on some LIGHT items absorbed by the crude_penalty/heart_bonus terms
    below rather than by gating the rescue itself.
  - `crude_penalty`: NEW code-level, negative-only signal built from (a)
    crude/shock/raunch terms named directly inside the comic_devices answer
    text, and (b) raw-text profanity + ALL-CAPS-shouting density
    (ops.normalize'd), tuned to require real DENSITY (not one incidental
    swear) so a chat-log-style piece where profanity is normal banter/color
    isn't nuked by a single hit. This is the discriminator h0 lacks to pull a
    COMEDY-labeled but crudely-executed item back down inside its own bucket
    without touching well-landed comedy (fixes #1).
  - `heart_bonus`: NEW code-level, positive-only warmth-word-density signal
    (reviving the dropped baseline HEART term), gated by `backbone` so heart
    alone in a pure-drama piece cannot inflate a humor-craft score -- it only
    lifts pieces that already carry some comedic backbone (fixes #3).
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

# ---------------------------------------------------------------------------
# Byte-identical to h0: level parsing + device counting (same fields, so this
# stays a faithful reuse rather than a reinterpretation of the same answers).
# ---------------------------------------------------------------------------
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


def _parse_level(ans):
    """Map the humor_level answer to 0.0 / 0.5 / 1.0, or None if unparseable."""
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


def _device_signal(ans):
    """(signal in [0,1], answered: bool). Counts distinct known techniques."""
    s = (ans or "").strip().lower()
    if (s == "" or s in ("n/a", "na", "no") or s.startswith("none")
            or s.startswith("no ") or s.startswith("not ")):
        return 0.0, (s != "")
    hits = set()
    for t in _DEVICE_TERMS:
        if t in s:
            hits.add(t.split()[0][:6])
    n = len(hits)
    if n == 0:
        return 0.3, True
    return min(1.0, 0.45 + 0.275 * min(n, 3)), True


def _sat(x, k):
    return 1.0 - 1.0 / (1.0 + max(0.0, x) / max(1e-6, k))


def _code_prior_fallback(text, ops):
    """Weak surface prior in [0,1]; used only when BOTH LLM fields fail."""
    try:
        t = ops.normalize(text)
    except Exception:
        t = text
    tl = t.lower()
    w = max(1, len(re.findall(r"\w+", tl)))
    kw = sum(len(re.findall(r"\b" + re.escape(k), tl)) for k in
             ("laugh", "joke", "funny", "grin", "chuckle", "snort", "giggle",
              "absurd", "ridiculous", "ironic", "sarcas", "pun"))
    s1 = _sat(kw / (w / 100.0), 0.7)
    s2 = _sat(t.count("!") / (w / 100.0), 0.9)
    dlg = t.count('"') + t.count("“") + t.count("”")
    s3 = _sat(dlg / (w / 100.0), 1.2)
    return 0.5 * s1 + 0.25 * s2 + 0.25 * s3


# ---------------------------------------------------------------------------
# NEW #1: crude/shock discount. Negative-only. Combines (a) crude-technique
# terms the extractor itself named in comic_devices, and (b) raw-text
# profanity + ALL-CAPS shouting density.
# ---------------------------------------------------------------------------
_CRUDE_DEVICE_TERMS = ("shock", "crude", "vulgar", "gross", "raunch", "toilet",
                       "cringe", "double entendre", "slapstick", "gratuit",
                       "shout", "scream")

_PROFANITY = ("fuck", "shit", "bastard", "bitch", "asshole", "goddamn",
              "dumbass", "crap", "dick", "cock", "whore", "slut")


def _crude_penalty(comic_devices_ans, text, ops):
    s = (comic_devices_ans or "").strip().lower()
    dev_hits = sum(1 for t in _CRUDE_DEVICE_TERMS if t in s)
    dev_pen = min(1.0, 0.5 * dev_hits)

    try:
        t = ops.normalize(text)
    except Exception:
        t = text
    tl = t.lower()
    w = max(1, len(re.findall(r"\w+", tl)))
    prof = sum(len(re.findall(r"\b" + re.escape(k), tl)) for k in _PROFANITY)
    # k=1.3 (vs an earlier k=0.5): a single incidental swear in a long piece
    # (e.g. profanity used as normal banter/color in dialogue-heavy chat-log
    # style fiction) should barely register; only real DENSITY of profanity
    # should read as crude delivery.
    prof_rate = _sat(prof / (w / 100.0), 1.3)
    caps_words = re.findall(r"\b[A-Z]{4,}\b", t)
    caps_rate = _sat(len(caps_words) / (w / 100.0), 0.6)
    text_pen = min(1.0, 0.7 * prof_rate + 0.3 * caps_rate)

    return max(dev_pen, text_pen, 0.5 * (dev_pen + text_pen))


# ---------------------------------------------------------------------------
# NEW #2: heart/warmth bonus. Positive-only, gated by comedic backbone in
# score() so it cannot inflate a pure-drama piece with no comedic register.
# ---------------------------------------------------------------------------
_HEART_KW = ("cared", "care for", "kind", "gentle", "warm", "hug",
             "embrace", "tear", "cried", "missed", "smile", "grateful",
             "tender", "proud")


def _heart_bonus(text, ops):
    try:
        t = ops.normalize(text)
    except Exception:
        t = text
    tl = t.lower()
    w = max(1, len(re.findall(r"\w+", tl)))
    hits = sum(len(re.findall(r"\b" + re.escape(k), tl)) for k in _HEART_KW)
    return _sat(hits / (w / 100.0), 0.9)


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        extracted = extracted or {}
        lvl = _parse_level(extracted.get("humor_level", ""))
        dev_ans = extracted.get("comic_devices", "")
        dev, dev_answered = _device_signal(dev_ans)

        if lvl is None and not dev_answered:
            cp = _code_prior_fallback(text, ops)
            return float(max(0.0, min(1.0, 0.30 + 0.40 * cp)))

        lvl_c = lvl if lvl is not None else 0.4  # neutral prior on parse failure

        # Backbone: average of lvl and dev PLUS their max, so a strong device
        # signal can rescue a mislabeled SERIOUS/LIGHT item instead of being
        # gated down to near-zero by lvl (h0's failure mode #2; see docstring
        # for why the blanket max beats a level-gated version empirically).
        backbone = 0.35 * lvl_c + 0.35 * dev + 0.30 * max(lvl_c, dev)

        cp = _code_prior_fallback(text, ops)
        crude = _crude_penalty(dev_ans, text, ops)
        heart = _heart_bonus(text, ops) * backbone

        out = 0.04 + 0.82 * backbone + 0.05 * cp - 0.42 * crude + 0.10 * heart
        return float(max(0.0, min(1.0, out)))
    except Exception:
        return 0.5
