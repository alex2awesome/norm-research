"""Synthetic calibration suite — constructs with MECHANICAL ORACLES for instrument
validation ("known-truth calibration").

The battery's instruments are first-in-literature; before trusting them on real
constructs, each must read KNOWN cases correctly. E-tier constructs are mechanically
checkable (the oracle is code), so for a competent judge model we know the ground-truth
answer to every variant prompt:

  tf         -> oracle(item)
  negated    -> NOT oracle(item)
  exclusion  -> NOT oracle(item)   (a compliant judge inverts)
  composed   -> oracle_A(item) AND oracle_B(item)

An instrument that misreads E-tier (e.g., scores an explicit rule as uninvertible) is
mismeasuring; readings on real constructs inherit that calibration bound. G-tier are
graded-but-articulable (oracle = authored design labels, weaker); H-tier has no oracle
(profile only). 40 items, fixed, authored to vary every E-tier property independently.
"""
from __future__ import annotations

import re

# ---- item corpus (40 fixed items; properties varied factorially-ish) -------------------
ITEMS = [
    "The cat sat quietly on the mat.",
    "Look out! The dog is loose again!",
    "There are 3 reasons to reconsider this budget proposal before the meeting.",
    "Wow, 7 parrots just landed on my balcony!",
    "The committee will convene on Thursday to review the amended zoning regulations.",
    "honestly no clue what happened lol",
    "A horse, a goat, and 2 chickens escaped from the county fair this morning.",
    "Please submit the quarterly report by end of day.",
    "Amazing!!! Best pizza I have ever had in my entire life!",
    "The mitochondria is the membrane-bound organelle that produces adenosine triphosphate.",
    "My goldfish learned a trick! He swims through a hoop!",
    "It rained for 40 days straight, which ruined the harvest and flooded 12 villages.",
    "ugh whatever, fine.",
    "The elephant exhibit closes at 5 pm sharp.",
    "We regret to inform you that your application has not been successful at this time.",
    "Did you seriously eat 9 tacos?!",
    "Snakes! Why did it have to be snakes!",
    "The stock rose 14 percent after the earnings call exceeded analyst expectations.",
    "i guess the movie was ok, kinda boring in the middle.",
    "Congratulations on your promotion! You earned it!",
    "The report was submitted late.",
    "A single owl hooted twice, then the forest fell silent.",
    "Termination of the agreement requires 30 days written notice from either party.",
    "best. day. ever!",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Whales migrate thousands of miles every year to reach their breeding grounds.",
    "no.",
    "The invoice total of 1250 dollars is due upon receipt.",
    "Hooray! The lizard finally shed its skin after 6 long weeks!",
    "Management has approved the revised staffing plan effective next quarter.",
    "The tiny hamster stuffed 4 peanuts into its cheeks!",
    "Clouds gathered slowly over the gray harbor.",
    "You must be kidding me! 88 emails in one hour!",
    "The defendant was ordered to pay restitution in the amount of 5000 dollars.",
    "puppies puppies puppies!!!",
    "The seminar covers regression, classification, and 2 clustering methods.",
    "A quiet evening, tea, and an old book.",
    "Rabbits can jump surprisingly high when startled!",
    "The board unanimously ratified the merger agreement yesterday.",
    "meh, 2 stars.",
    # cycle-2 corpus extension (indices 40+; cycle-1 results index-aligned to 0-39)
    "Where did the turtle go?",
    "Can you believe the meeting ran 3 hours over?",
    "What time does the aquarium open tomorrow?",
    "Is this the final version of the contract?",
    "Why would anyone schedule 6 meetings on a Friday?!",
    "Has the committee reviewed the proposal yet?",
    "The ferret darted under the porch before anyone could react.",
    "Payment of 750 dollars is expected within 14 business days.",
]

ANIMALS = ("cat", "dog", "parrot", "horse", "goat", "chicken", "goldfish", "elephant",
           "snake", "owl", "whale", "lizard", "hamster", "puppies", "rabbit", "tiger",
           "fish", "bird")

# ---- constructs -------------------------------------------------------------------------
# oracle: item -> bool (E-tier mechanical; G-tier authored; H-tier None)


def _has_animal(t: str) -> bool:
    low = t.lower()
    return any(a in low for a in ANIMALS)


CONSTRUCTS = {
    "E1_exclaim": {
        "tier": "E",
        "prompt": "The text contains at least one exclamation mark ('!').",
        "oracle": lambda t: "!" in t,
    },
    "E2_long": {
        "tier": "E",
        "prompt": "The text is longer than twelve words.",
        "oracle": lambda t: len(t.split()) > 12,
    },
    "E3_animal": {
        "tier": "E",
        "prompt": "The text mentions an animal.",
        "oracle": _has_animal,
    },
    "E4_digit": {
        "tier": "E",
        "prompt": "The text contains a numeral written in digits (e.g., 3, 14, 1250).",
        "oracle": lambda t: bool(re.search(r"\d", t)),
    },
    "G1_formal": {
        "tier": "G",
        "prompt": "The text is written in a formal, professional register.",
        "oracle": None,   # authored labels below
    },
    "G2_excited": {
        "tier": "G",
        "prompt": "The text expresses excitement or strong positive emotion.",
        "oracle": None,
    },
    "H1_charming": {
        "tier": "H",
        "prompt": "The text is charming.",
        "oracle": None,
    },
}

# authored design labels for G-tier (index-aligned with ITEMS; 1=yes)
G_LABELS = {
    "G1_formal": [0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,
                  0,0,0,1,0,1,0,0,1,0, 0,0,0,0,0,0,0,1],
    "G2_excited": [0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,0,1,0,
                   1,0,1,0,1,0,0,1,0,0, 0,0,0,0,0,0,0,0],
}

COMPOSED_PAIRS = [("E1_exclaim", "E3_animal"), ("E2_long", "E4_digit")]


def oracle_vector(cid: str) -> list | None:
    c = CONSTRUCTS[cid]
    if c["oracle"] is not None:
        return [bool(c["oracle"](t)) for t in ITEMS]
    if cid in G_LABELS:
        return [bool(x) for x in G_LABELS[cid]]
    return None


def oracle_sanity() -> dict:
    """Each E-tier oracle must be non-degenerate on the corpus (both classes >= 25%)."""
    out = {}
    for cid in CONSTRUCTS:
        v = oracle_vector(cid)
        out[cid] = None if v is None else sum(v) / len(v)
    return out


# ---- cycle-2 additions ------------------------------------------------------------------
CONSTRUCTS["E5_qmark"] = {
    "tier": "E",
    "prompt": "The text contains at least one question mark ('?').",
    "oracle": lambda t: "?" in t,
}

# hand-negated predicates (the direct-negation upper bound for the negation instrument)
NEG_DIRECT = {
    "E1_exclaim": "The text contains no exclamation marks at all.",
    "E3_animal": "The text does not mention any animal.",
    "E4_digit": "The text contains no numerals written in digits.",
    "E5_qmark": "The text contains no question marks.",
    "G1_formal": "The text is NOT written in a formal register (it is casual, informal, "
                 "or conversational).",
    "G2_excited": "The text does NOT express excitement or strong positive emotion.",
}

# known-mixture holistic construct: instructed blend of three mechanical properties.
# oracle_score = 0.5*E1 + 0.3*E3 + 0.2*E4 (graded 0-1); validates the graded elicitation +
# span-recovery estimator end-to-end (pipeline positive control).
BLEND_PROMPT = (
    "Rate the text on OVERALL APPEAL from 0 to 10, where appeal is driven by: energetic "
    "punctuation such as exclamation marks (about half the weight), mention of animals "
    "(about a third of the weight), and concrete numbers written in digits (the remaining "
    "weight). Reply with a single integer 0-10.")


def blend_oracle_score(t: str) -> float:
    return (0.5 * ("!" in t) + 0.3 * _has_animal(t)
            + 0.2 * bool(re.search(r"\d", t)))
