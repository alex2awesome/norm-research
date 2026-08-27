"""a18: Topicality, subject breadth, and durability (hybrid v0).

Criterion: choice and range of contemporary targets; leverage timeliness
while avoiding shallow riffs; manage perishability vs evergreen value.

Design: code cannot reliably judge WHO/WHAT a joke's subject is or whether
it leans on a fresh, specific real-world referent -- that needs semantic
grounding, so two short LLM extractions carry the "what is this joke about"
and "does it lean on something topical" judgments. The PREDICATE (how those
grounded facts map to a score) stays entirely in code: a fixed list of
well-worn stock subjects / formulaic setups penalizes shallow riffs, a
non-empty topical hook (plus any explicit dates found) rewards leveraging
timeliness, a non-stock specific subject rewards range/breadth, and TF-IDF
corpus-similarity (evidence op) acts as a structural proxy for "how common/
overused is this premise in the corpus" -- high redundancy suggests a
well-worn riff, low redundancy suggests a more distinctive choice of target.
Evergreen jokes are not penalized by default; they are only penalized when
they ALSO look like a stock/formulaic riff, which is the intended
perishability-vs-evergreen tradeoff in the criterion.
"""
import re

LLM_FIELDS = {
    "target_type": (
        "Name the joke's comedic subject/target in <=6 words (e.g. a stock "
        "trope like 'blonde jokes' or 'lawyer jokes', an ethnic/religious "
        "group, a specific real public figure/brand, or 'original scenario'); "
        "answer NONE if unclear."
    ),
    "topical_hook": (
        "Name any specific real-world contemporary event, public figure, "
        "technology, meme, or trend this joke leans on, in <=10 words; "
        "answer NONE if the joke is generic or timeless."
    ),
}

# Well-worn stock subjects: checked against the LLM's target_type judgment
# (not raw text), since a passing mention of e.g. "lawyer" inside a joke
# about something else shouldn't trigger this -- the predicate needs the
# LLM's grounding of what the joke is actually ABOUT.
STOCK_SUBJECTS = [
    "blonde", "lawyer", "engineer", "redneck", "irish", "priest", "rabbi",
    "imam", "nun", "programmer", "genie", "chuck norris", "yo mama",
    "yo momma", "traveling salesman", "mother-in-law", "drunk", "lightbulb",
]

# Formulaic joke-format markers: literal fixed phrases, checked against the
# normalized raw text directly since these are high-precision surface cues.
FORMULAIC_SETUPS = [
    "walks into a bar", "knock knock", "why did the chicken",
    "in soviet russia", "that's what she said", "said no one ever",
    "i see what you did there",
]


def _norm_field(s):
    s = (s or "").strip().lower()
    return s


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        tl = t.lower()

        ex = extracted or {}
        target_type = _norm_field(ex.get("target_type", ""))
        topical_hook = _norm_field(ex.get("topical_hook", ""))

        def _is_empty(v):
            return v in ("", "none", "n/a", "na", "unclear")

        val = 0.5

        # --- shallow-riff penalties ---
        stock_hit = any(kw in target_type for kw in STOCK_SUBJECTS)
        if stock_hit:
            val -= 0.15

        formulaic_hit = any(kw in tl for kw in FORMULAIC_SETUPS)
        if formulaic_hit:
            val -= 0.10

        # --- leverage-timeliness bonus (LLM-grounded) ---
        if not _is_empty(topical_hook):
            val += 0.20

        try:
            dates = ops.extract_dates(t)
        except Exception:
            dates = []
        if dates:
            val += 0.05

        # --- subject breadth / originality ---
        if not _is_empty(target_type) and not stock_hit:
            val += 0.08

        # --- corpus-redundancy proxy for well-worn riffs (evidence op) ---
        try:
            neighbors = ops.retrieve_similar(t, k=5)
        except Exception:
            neighbors = []
        sims = []
        for item in neighbors or []:
            try:
                a, b = item
            except Exception:
                continue
            a_num = isinstance(a, (int, float))
            b_num = isinstance(b, (int, float))
            if a_num and not b_num:
                sims.append(float(a))
            elif b_num and not a_num:
                sims.append(float(b))
            elif a_num and b_num:
                # ambiguous tuple order; pack contract lists similarity first
                sims.append(float(a))
        if sims:
            avg_sim = sum(sims) / len(sims)
            if avg_sim >= 0.55:
                val -= 0.10
            elif avg_sim <= 0.15:
                val += 0.05

        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
