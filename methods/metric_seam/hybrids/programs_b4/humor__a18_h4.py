"""a18: Topicality, subject breadth, and durability (BUDGET-4 extension).

Criterion: choice and range of contemporary targets; leverage timeliness
while avoiding shallow riffs; manage perishability vs evergreen value.

Base design (unchanged from h0): code cannot reliably judge WHO/WHAT a
joke's subject is or whether it leans on a fresh, specific real-world
referent -- that needs semantic grounding, so two short LLM extractions
(target_type, topical_hook) carry the "what is this joke about" and "does
it lean on something topical" judgments. The PREDICATE stays in code: a
fixed list of well-worn stock subjects / formulaic setups penalizes
shallow riffs, a non-empty topical hook (plus any explicit dates found)
rewards leveraging timeliness, a non-stock specific subject rewards
range/breadth, and TF-IDF corpus-similarity acts as a structural proxy for
"how common/overused is this premise in the corpus."

Budget-4 extension (blind design -- no eval signal used, chosen on
construct grounds): h0's STOCK_SUBJECTS list is a fixed, necessarily
incomplete lexicon (it cannot cover every hackneyed comedy trope -- e.g.
"cats vs. dogs," "airline food," "IKEA furniture," "millennials and
avocado toast" are all well-worn premises absent from the list), and h0
rewards ANY non-empty topical_hook with a flat +0.20 regardless of whether
the joke actually DOES anything with that reference or merely name-drops
it -- which is precisely the "leverage timeliness while AVOIDING SHALLOW
RIFFS" distinction the criterion draws and h0 cannot make. Two new thin
LLM fields target exactly those two gaps:
  - cliche_premise: a direct, open-domain judgment of whether the joke's
    premise is a well-known comedy cliche, generalizing past the fixed
    STOCK_SUBJECTS lexicon (subject breadth / avoiding shallow riffs).
  - hook_depth: whether the topical reference is actually built on/
    subverted by the punchline (genuine leverage) vs merely name-dropped
    with no comedic payoff tied to it (shallow riff) -- this refines,
    rather than duplicates, the flat topical_hook bonus.
Both fields degrade to a no-op when absent from `extracted` (budget < 4),
so this program reproduces h0's exact score whenever only the original 2
fields are served.
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
    "cliche_premise": (
        "Is this joke's core premise a well-known comedy cliche/trope, even if "
        "not one of blonde/lawyer/knock-knock style? Answer YES, NO, or UNSURE."
    ),
    "hook_depth": (
        "Does the punchline build on or subvert the topical reference named "
        "above, or just name-drop it with no comedic payoff? Answer BUILDS, "
        "NAMEDROP, or NA."
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

        # --- budget-4 nudges: open-domain cliche check + hook-depth refinement ---
        # Missing fields (budget < 4) leave these as "" -> no branch fires -> val
        # unchanged, so this program reproduces h0 exactly at budget 2.
        cliche = _norm_field(ex.get("cliche_premise", "")).upper()
        hook_depth = _norm_field(ex.get("hook_depth", "")).upper()

        if cliche == "YES":
            val -= 0.12

        if hook_depth == "NAMEDROP":
            val -= 0.10
        elif hook_depth == "BUILDS":
            val += 0.10

        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
