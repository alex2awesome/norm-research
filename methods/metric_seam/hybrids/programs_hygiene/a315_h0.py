"""Hybrid metric channel for a315 -- Use of Influences: Transformation over Imitation.

Judge axis (from train residuals): the judge punishes stories whose world/characters/
plot/lines are lifted from a SPECIFIC existing source (franchise, film, game, real
person, famous historical event), sits mid for generic-trope territory played
straight, and rewards original conceits or genuine subversion of the familiar.
Keyword clich√© counting (the v0 baseline) is nearly orthogonal to this.

Design: two LLM fields carry the thick judgments (what source is borrowed; is the
central conceit fresh or stock); the code holds the predicate: a tier ladder over
(source-specificity x freshness), a small class-level franchise-marker backup
lexicon in case the extractor misses an obvious fandom, and a tiny corpus-evidence
nudge from TF-IDF neighbors (derivative fics share vocabulary with other responses
in the 5k corpus; originals sit farther from their neighbors).
"""

import re
import math

LLM_FIELDS = {
    "borrowed_source": (
        "If the story's characters, world, plot, or famous lines are taken from a "
        "specific existing franchise, book, film, TV show, game, real person, or "
        "famous historical event, name that source in a few words; otherwise answer NONE."
    ),
    "premise_novelty": (
        "Answer one word: 'fresh' if the story's central conceit is inventive or "
        "subverts a familiar premise into something new; 'stock' if it plays a very "
        "common premise (alien invasion, deal with the devil, grim reaper, chosen "
        "one, fan fiction) straight."
    ),
}

# Answers that mean "no specific source".
_NONE_PAT = re.compile(
    r"^\s*(none|n/?a|no\b|nothing|original|not applicable|unknown|unclear)", re.I
)

# Answers naming only a generic category / trope family, not a specific source.
# These indicate trope-territory, not direct derivation.
_GENERIC_MARKERS = (
    "mytholog", "fairy tale", "fairytale", "fairy-tale", "folklore", "folk tale",
    "bible", "biblical", "christian", "superhero", "comic", "genre", "trope",
    "cliche", "cliché", "fanfic", "fan fiction", "urban legend", "legend",
    "generic", "alien invasion", "science fiction", "sci-fi", "scifi", "fantasy",
    "horror", "creepypasta", "reddit", "writingprompts", "writing prompts",
    "grim reaper", "deal with the devil", "various", "several", "multiple",
)

# Class-level backup lexicon: unmistakable markers of big fandoms common in
# WritingPrompts scrapes. Counted as DISTINCT franchise groups (robust to spam).
_FRANCHISE_GROUPS = [
    re.compile(p, re.I) for p in (
        r"\b(darth vader|skywalker|jedi|sith|lightsaber|dagobah|obi[- ]wan|yoda|padawan|death star)\b",
        r"\b(hogwarts|voldemort|dumbledore|hermione|muggle|gryffindor)\b",
        r"\b(gandalf|frodo|sauron|mordor|middle[- ]earth|bilbo)\b",
        r"\b(tardis|dalek|time lord|doctor who)\b",
        r"\b(batman|superman|spider[- ]?man|gotham|kryptonite|avengers|thanos|deadpool|x[- ]men)\b",
        r"\b(starfleet|klingon|vulcan|uss enterprise)\b",
        r"\b(pok[eé]mon|pikachu)\b",
        r"\b(westeros|winterfell|khaleesi|jon snow|targaryen)\b",
        r"\b(hyrule|master chief|azeroth|skyrim)\b",
    )
]


def _source_tier(ans):
    """0 = no source named, 1 = generic category only, 2 = specific source."""
    a = (ans or "").strip().lower()
    if not a or _NONE_PAT.match(a):
        return 0
    for g in _GENERIC_MARKERS:
        if g in a:
            return 1
    return 2


# "fresh"/"invent"/"stock" are bare fragments that collide with unrelated
# words as substrings (REFRESHMENT(S) contain fresh; INVENTORY contains
# invent; STOCKPILES/STOCKROOMS contain stock) -- anchor to word boundaries
# with an explicit inflection whitelist so only the intended concept fires.
_FRESH_RE = re.compile(r"\bfresh(?:ly|ness|er|est)?\b")
_INVENT_RE = re.compile(r"\binvent(?:ive|ed|ion|ions|or|ors)?\b")
_STOCK_RE = re.compile(r"\bstock\b")


def _freshness(ans):
    """1.0 fresh, 0.0 stock, 0.5 unknown/empty."""
    a = (ans or "").strip().lower()
    if not a:
        return 0.5
    if _FRESH_RE.search(a) or _INVENT_RE.search(a) or "subver" in a or "original" in a:
        return 1.0
    if _STOCK_RE.search(a) or "common" in a or "trope" in a or "clich" in a or "derivative" in a:
        return 0.0
    return 0.5


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = text or ""
        try:
            t = ops.normalize(t)
        except Exception:
            pass
        ex = extracted if isinstance(extracted, dict) else {}

        tier = _source_tier(ex.get("borrowed_source", ""))
        fw = _freshness(ex.get("premise_novelty", ""))

        # Code backup: obvious fandom markers in the text override a missed field.
        hits = sum(1 for g in _FRANCHISE_GROUPS if g.search(t))
        if hits and tier < 2:
            tier = 2

        # Tier ladder (base + freshness lift). Borrowing that is genuinely
        # transformed still recovers part of the scale.
        if tier == 2:
            base = 0.15 + 0.25 * fw
        elif tier == 1:
            base = 0.28 + 0.28 * fw
        else:
            base = 0.35 + 0.42 * fw

        # Heavier saturation of one universe's markers -> more imitation.
        base -= 0.05 * min(hits, 3)

        # Tiny corpus-evidence nudge: derivative pieces share vocabulary with
        # their TF-IDF neighbors; originals sit farther away. Bounded +-0.05.
        try:
            sims = [s for (s, _id) in ops.retrieve_similar(t, k=5) if s < 0.95]
            if sims:
                top = sorted(sims, reverse=True)[:3]
                sbar = sum(top) / len(top)
                base -= 0.05 * math.tanh((sbar - 0.30) / 0.20)
        except Exception:
            pass

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
