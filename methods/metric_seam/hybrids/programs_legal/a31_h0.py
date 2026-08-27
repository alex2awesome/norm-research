"""a31 hybrid: Majority-group plaintiff flag (reverse discrimination).

Criterion: plaintiff is a majority-group member (white, male, heterosexual,
etc.) asserting reverse discrimination.

Baseline (v0_keyword) scans raw text for a fixed phrase list ('reverse
discrimination', 'white male', 'caucasian', ...) and normalizes hit-count.
That phrase list still catches the clean explicit-term cases (an affirmative-
action challenge that literally says "reverse discrimination"), so we KEEP it
as a code-side backstop. But it misses the doctrinal pattern when the case
narrative never uses those words — e.g. a male applicant excluded by a
female-only hiring policy (BFOQ case), or a white employee replaced by
younger Black hires after termination. The actual predicate the criterion
wants is compositional: (a) plaintiff belongs to a traditionally-majority
demographic, AND (b) the grievance is that a different (minority-relative)
group was favored instead. Regex can see (a) sometimes but essentially never
sees (b) reliably from a noisy scraped narrative — that's thick-input
grounding, so it goes to an LLM field. The AND/OR combination logic (the
actual predicate) stays in code, not the LLM.
"""
import re

LLM_FIELDS = {
    "plaintiff_group": (
        "State the plaintiff's own race, sex, religion, or sexual orientation "
        "as described in the text (e.g. 'white male', 'heterosexual woman'); "
        "answer NONE if no such identity is stated."
    ),
    "favored_group": (
        "If the plaintiff alleges being passed over, excluded, or disadvantaged "
        "in favor of a person or policy favoring a DIFFERENT demographic group, "
        "name that other group (e.g. 'Black candidate', 'female applicants'); "
        "answer NONE if no such favored-other-group comparison is alleged."
    ),
}

# --- code-side backstop: explicit legal-term / phrase hits (baseline signal) ---
_BASELINE_KWS = [
    "reverse discrimination", "majority group", "majority-group", "white male",
    "white employee", "white plaintiff", "caucasian", "heterosexual",
    "non-minority", "straight employee",
]

# --- code-side predicate: demographic-marker lexicons applied to the LLM's
# SHORT distilled answers (much less noisy than scanning raw scraped text) ---
_MAJORITY_PATTERNS = [
    r"\bwhite\b", r"\bcaucasian\b", r"\bmale\b", r"\bman\b", r"\bmen\b",
    r"\bheterosexual\b", r"\bstraight\b", r"\bchristian\b",
    r"\bnon-disabled\b", r"\bnon-minority\b", r"\bmajority[- ]group\b",
]
_MINORITY_PATTERNS = [
    r"\bblack\b", r"\bafrican[- ]american\b", r"\bfemale\b", r"\bwoman\b",
    r"\bwomen\b", r"\bhispanic\b", r"\blatino\b", r"\blatina\b", r"\basian\b",
    r"\bgay\b", r"\blesbian\b", r"\bmuslim\b", r"\bdisabled\b",
    r"\btransgender\b", r"\bnative american\b",
]
_NONE_ANSWERS = {"", "none", "n/a", "na", "not stated", "not mentioned", "unknown"}


def _has_marker(s, patterns):
    """Word-boundary marker match with a crude negation guard (excludes
    'non-white' / 'not white' style negated mentions from counting)."""
    if not s:
        return False
    low = s.lower()
    for pat in patterns:
        for m in re.finditer(pat, low):
            prefix = low[max(0, m.start() - 4):m.start()]
            if prefix in ("non-", "not "):
                continue
            return True
    return False


def _is_none_answer(s):
    if not s:
        return True
    return s.strip().lower() in _NONE_ANSWERS


def score(text: str, extracted: dict, ops) -> float:
    try:
        norm = ops.normalize(text) if text else ""
        t = norm.lower()

        # code layer 1: explicit-term backstop, catches clean cases directly
        hits = sum(1 for k in _BASELINE_KWS if k in t)
        kw_score = min(1.0, hits / 3.0)

        # LLM layer: thick-input grounding for the two halves of the predicate
        ext = extracted or {}
        plaintiff_group = ext.get("plaintiff_group", "")
        favored_group = ext.get("favored_group", "")

        plaintiff_is_majority = (not _is_none_answer(plaintiff_group)
                                  and _has_marker(plaintiff_group, _MAJORITY_PATTERNS))
        favored_is_minority = (not _is_none_answer(favored_group)
                                and _has_marker(favored_group, _MINORITY_PATTERNS))

        # code layer 2: the actual predicate — both halves present is the
        # doctrinal "reverse discrimination" pattern; only one half is a
        # weaker partial signal; neither is a plain, non-reverse claim.
        if plaintiff_is_majority and favored_is_minority:
            construct_score = 1.0
        elif plaintiff_is_majority or favored_is_minority:
            construct_score = 0.5
        else:
            construct_score = 0.0

        val = 0.3 * kw_score + 0.7 * construct_score
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
