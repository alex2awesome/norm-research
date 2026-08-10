"""The toy world: a fictional village's verdict on which creatures make fit companions.

This is the single source of truth for the scenario's ground truth. It is a deliberate
miniature of the research problem.

- The village publishes a **Companion Code** — articulated, practical criteria anyone can
  state and check: where the creature ranges (its habitat), how big it is, what it eats, what
  its hide is like. These become the *known metrics* (see ``metrics.py``).
- The Code genuinely works for the common **grove** creatures: a fitting companion is small,
  gentle, and soft. But in the wilder habitats the Code falls silent, and the elders' verdicts
  there are governed by **tacit aesthetic norms** the Code never mentions:
    * in the **marsh** (a large population) — sweet **song** is prized  -> a BROAD tacit norm
    * in the **cavern** (a small population) — bioluminescent **glow** is prized -> a NARROW one
  The infilling loop must rediscover both from the text and read off their *generality* as
  **coverage** (the fraction of the population whose verdict they govern): song covers the
  larger marsh, glow the smaller cavern, so song is measurably the more general norm.
- Two further attributes — **color** and **limbs** — are described in every dossier but have
  **no effect** on the verdict. They are decoys: a discovery method must not be fooled into
  "finding" them.

Every attribute is realized with several interchangeable phrasings, so a scorer must
generalize over wording. Creatures are invented (no real species), so the only signal about a
creature is in its description.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------------------
# Attributes. Each value is realized by several phrasings (inserted verbatim into the text).
# --------------------------------------------------------------------------------------

# Known (Companion Code) attributes
HABITAT = {
    "grove": ["dwells in the sunlit groves", "lives among the orchard glades",
              "roams the flowering groves", "nests in the blossoming thickets"],
    "marsh": ["wades through the reed marshes", "lives in the misty wetlands",
              "haunts the boggy fenlands", "stalks the tangled mire"],
    "cavern": ["dwells deep in the cavern tunnels", "lives in the lightless underground",
               "haunts the subterranean hollows", "makes its home in the deep caves"],
}
SIZE = {
    "tiny": ["no larger than a sparrow", "small enough to perch on a wrist",
             "a diminutive little thing", "barely the size of a housecat"],
    "hulking": ["a hulking brute", "towering over a grown villager",
                "massive and broad-shouldered", "enormous in stature"],
}
FEEDING = {
    "grazer": ["grazes on moss and tender leaves", "feeds quietly on fruit and grasses",
               "browses gently on foliage", "nibbles lichen from the stones"],
    "hunter": ["stalks and devours smaller beasts", "hunts live prey through the night",
               "ambushes creatures that stray too close", "tears into whatever flesh it can catch"],
}
PELT = {
    "furred": ["covered in soft fur", "its coat thick and downy", "wrapped in a dense pelt"],
    "scaled": ["sheathed in hard scales", "armored with overlapping plates", "clad in glossy scales"],
}

# Hidden (tacit) aesthetic attributes
GLOW = {
    "luminous": ["its hide casts a soft glow", "it shimmers with a pale inner light",
                 "faintly luminous in the dark", "gives off a gentle bioluminescent shine",
                 "its markings glow like dim embers"],
    "dim": ["its hide is dull and lightless", "it gives off no light at all",
            "drab and wholly shadowed", "its coloring flat and matte"],
}
SONG = {
    "melodious": ["it sings in clear, melodious tones", "its call is a sweet warbling song",
                  "it voices lilting, musical cries", "its song is gentle and tuneful"],
    "harsh": ["it emits harsh, grating shrieks", "its call is a discordant rasp",
              "it screeches in jarring tones", "its cries are rough and unpleasant"],
}

# Decoy attributes — described in every dossier, but irrelevant to the verdict
COLOR = {
    "azure": ["tinted a deep azure blue", "washed in cool cerulean hues", "its markings a bright sky-blue"],
    "ochre": ["colored a dull ochre", "its hide a muddy yellow-brown", "shaded in earthy tan"],
}
LIMBS = {
    "four-limbed": ["it moves on four sturdy limbs", "a four-legged, even gait", "padding along on four feet"],
    "six-limbed": ["it scuttles on six legs", "six-limbed and many-jointed", "skittering on six clawed feet"],
}

ATTRIBUTES = {
    "habitat": HABITAT, "size": SIZE, "feeding": FEEDING, "pelt": PELT,
    "glow": GLOW, "song": SONG, "color": COLOR, "limbs": LIMBS,
}
KNOWN_ATTRS = ["habitat", "size", "feeding", "pelt"]
HIDDEN_ATTRS = ["glow", "song"]
DECOY_ATTRS = ["color", "limbs"]

POSITIVE_VALUE = {"glow": "luminous", "song": "melodious"}

# Habitat is sampled UNEQUALLY: the common grove, a large marsh, a small cavern.
HABITAT_PROBS = {"grove": 0.45, "marsh": 0.35, "cavern": 0.20}

# --------------------------------------------------------------------------------------
# Label rule — the elders' verdict (a NORM, not a physical fact).
# --------------------------------------------------------------------------------------
#   grove  (Code works):  a fitting companion is small, gentle, and soft (a COMBINATION of
#                         published criteria, each individually modest).
#   marsh  (Code silent): the tacit SONG norm rules (sweet-voiced kept).   -> broad
#   cavern (Code silent): the tacit GLOW norm rules (luminous kept).       -> narrow
#   color / limbs:        described everywhere, but never affect the verdict (decoys).

W_GROVE_EACH = 1.5     # grove: each of {tiny, grazer, furred} nudges toward "kept"
W_MARSH_SONG = 3.0     # marsh: melodious(+) vs harsh(-)
W_CAVERN_GLOW = 3.0    # cavern: luminous(+) vs dim(-)


def label_logit(attrs: Dict[str, str]) -> float:
    h = attrs["habitat"]
    if h == "grove":
        z = 0.0
        z += W_GROVE_EACH * (1.0 if attrs["size"] == "tiny" else -1.0)
        z += W_GROVE_EACH * (1.0 if attrs["feeding"] == "grazer" else -1.0)
        z += W_GROVE_EACH * (1.0 if attrs["pelt"] == "furred" else -1.0)
        return z
    if h == "marsh":
        return W_MARSH_SONG * (1.0 if attrs["song"] == "melodious" else -1.0)
    return W_CAVERN_GLOW * (1.0 if attrs["glow"] == "luminous" else -1.0)  # cavern


# --------------------------------------------------------------------------------------
# Sampling + text synthesis
# --------------------------------------------------------------------------------------

_CV = ("br", "z", "k", "thr", "v", "gl", "pl", "sn", "dr", "m", "x", "qu")
_VOW = ("a", "o", "e", "i", "u", "or", "ax", "en", "ish", "ul")


def _name(rng: np.random.Generator) -> str:
    n = rng.integers(2, 4)
    s = "".join(rng.choice(_CV) + rng.choice(_VOW) for _ in range(n))
    return s.capitalize()


def sample_attrs(rng: np.random.Generator) -> Dict[str, str]:
    attrs = {}
    for a, vals in ATTRIBUTES.items():
        if a == "habitat":
            keys = list(HABITAT_PROBS)
            attrs[a] = str(rng.choice(keys, p=[HABITAT_PROBS[k] for k in keys]))
        else:
            attrs[a] = str(rng.choice(list(vals)))
    return attrs


def build_text(attrs: Dict[str, str], rng: np.random.Generator, name: Optional[str] = None) -> str:
    name = name or _name(rng)

    def pick(attr):
        return str(rng.choice(ATTRIBUTES[attr][attrs[attr]]))

    s1 = f"The {name} is {pick('size')}, {pick('pelt')}, and {pick('color')}."
    s2 = f"It {pick('habitat')}, where it {pick('feeding')}."
    s3 = f"{pick('glow').capitalize()}. {pick('song').capitalize()}."
    s4 = f"In motion, {pick('limbs')}."
    body = [s1, s2, s3, s4]
    # light shuffle of the descriptive sentences for surface variety (keep s1 intro first)
    tail = body[1:]
    rng.shuffle(tail)
    return " ".join([s1] + tail)


# --------------------------------------------------------------------------------------
# Shared attribute detector — used by the known code-metrics AND the offline oracle.
# --------------------------------------------------------------------------------------

def detect(text: str, attr: str) -> Optional[str]:
    """The value of ``attr`` expressed in ``text``, or None if absent.

    Matches the full phrasing as a substring. ``build_text`` inserts phrasings verbatim, so
    this is exact and collision-free."""
    t = text.lower()
    for value, phrasings in ATTRIBUTES[attr].items():
        for ph in phrasings:
            if ph.lower() in t:
                return value
    return None
