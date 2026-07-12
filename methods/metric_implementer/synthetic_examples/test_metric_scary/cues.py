"""The planted `is_scary` cue lexicon — the single source of truth shared by the dataset
builder (which PLANTS cues) and the judge (which DETECTS them), so the two can never drift.

A scary story is built from markers drawn from one or more CUE CATEGORIES. Each category has
two independent vocabularies:

  * ``MARKERS``  — full sentence fragments planted into story TEXT. A category "fires" in a
                   text iff one of its markers appears as a substring. This defines the
                   ground-truth label (scary iff >=1 category fires) and what any judge can
                   actually detect in an item.
  * ``KEYWORDS`` — short words a RUBRIC must mention to "cover" the category. This is what
                   makes the metric *articulable*: a rubric that names a cue can apply it; a
                   vague rubric that names none cannot. Coverage is the rubric-side knob the
                   prompt optimizer moves.

Marker vocab and keyword vocab are deliberately disjoint in surface form (markers are
scene-level fragments; keywords are the abstract category words a rubric would use), so
"what the text contains" and "what the rubric says to look for" are measured independently.
"""

from __future__ import annotations

from typing import Dict, List, Set

# ---- the five scary cue categories ------------------------------------------------------
# Order matters only for deterministic iteration. DREAD is the "obvious" category a crude
# seed rubric already names; the other four are the discoverable headroom for optimization.

CATEGORIES: List[str] = ["DREAD", "SOUND", "PRESENCE", "BODY", "DANGER"]

# Scene-level fragments planted into scary stories (text-side signal).
MARKERS: Dict[str, List[str]] = {
    "DREAD": [
        "a cold wave of dread washed over them",
        "an ominous certainty settled into the room",
        "every instinct told them something was deeply wrong",
        "a nameless terror tightened its grip",
    ],
    "SOUND": [
        "a floorboard creaked somewhere behind them",
        "footsteps echoed down the empty hall",
        "a scream tore through the quiet",
        "something scraped slowly against the window",
    ],
    "PRESENCE": [
        "a figure stood motionless in the doorway",
        "they could not shake the feeling of being watched",
        "a long shadow slid across the far wall",
        "something was following close behind them",
    ],
    "BODY": [
        "their heart hammered against their ribs",
        "their breath caught hard in their throat",
        "a shiver crawled the length of their spine",
        "their hands would not stop trembling",
    ],
    "DANGER": [
        "the blade glinted once in the dark",
        "fresh blood was smeared across the floor",
        "it lunged at them from the shadows",
        "the threat in the air was unmistakable now",
    ],
}

# Length-matched, affect-positive fragments planted into NON-scary stories. None of these
# contains any scary marker substring, so non-scary stories fire zero categories.
CALM_MARKERS: List[str] = [
    "warm sunlight spilled gently across the table",
    "they laughed together at the same old joke",
    "the kettle whistled a cheerful little tune",
    "a soft breeze drifted through the open window",
    "the smell of fresh bread filled the whole kitchen",
    "they sank into the comfortable worn-in chairs",
    "the afternoon stretched on slow and easy",
    "everything felt calm, familiar, and ordinary",
]

# Abstract words a RUBRIC must mention to cover each category (rubric-side knob). Lowercased
# substrings; matched case-insensitively against the rubric body.
KEYWORDS: Dict[str, List[str]] = {
    "DREAD": ["dread", "fear", "terror", "scary", "frighten", "afraid", "ominous",
              "horror", "unsettl"],
    "SOUND": ["sound", "noise", "creak", "footstep", "scream", "silence"],
    "PRESENCE": ["presence", "watch", "figure", "shadow", "someone", "stalk", "follow"],
    "BODY": ["heart", "breath", "tremble", "trembl", "pulse", "shiver", "spine"],
    "DANGER": ["danger", "threat", "blood", "knife", "attack", "weapon", "lunge", "harm"],
}

# A non-degenerate planted score divisor: scary stories carry 2-4 firing categories, so a
# fully-covering rubric lands in roughly [0.67, 1.0] and a 1-category rubric lands at ~0.33.
SCORE_DIVISOR: float = 3.0


def fires(text: str) -> Set[str]:
    """Categories whose markers appear in ``text`` (the text-side, rubric-independent signal)."""
    low = text.lower()
    return {c for c, ms in MARKERS.items() if any(m in low for m in ms)}


def is_scary_label(text: str) -> int:
    """Ground-truth planted label: 1 iff any scary category fires."""
    return int(bool(fires(text)))


def coverage(rubric_body: str) -> Set[str]:
    """Categories a rubric "covers" — i.e. names a keyword for (the rubric-side knob)."""
    low = rubric_body.lower()
    return {c for c, kws in KEYWORDS.items() if any(kw in low for kw in kws)}


def planted_score(rubric_body: str, text: str) -> float:
    """The planted judge's score for one item under one rubric, in [0,1].

    A category contributes iff it BOTH fires in the text AND is covered by the rubric. So a
    crude rubric that names only DREAD recovers only the DREAD-driven part of the signal,
    while a rubric enumerating all five cues recovers the full planted label. Returns 0.5
    (pure "can't tell") when the rubric names no concrete cue at all.
    """
    cov = coverage(rubric_body)
    if not cov:
        return 0.5
    hit = fires(text) & cov
    return min(1.0, len(hit) / SCORE_DIVISOR)
