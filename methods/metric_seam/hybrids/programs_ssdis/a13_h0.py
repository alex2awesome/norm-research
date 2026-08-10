"""a13 hybrid: code runs a narrow Appointments-Clause/Lucia keyword predicate through a small decision table; one LLM field confirms the phrase (if present) is actually raised as the claimant's own argument."""

# Criterion is a narrow, well-defined factual predicate: does the claimant
# explicitly challenge the ALJ's constitutional appointment (Appointments
# Clause / Lucia v. SEC / SSR 19-1p)? Higher score = stronger evidence the
# challenge is raised.
#
# Design: unlike broad doctrinal constructs (pretext, logical bridge), this
# doctrine's vocabulary is narrow enough that keyword presence is a
# reasonably strong signal on its own -- but a case-facts narrative can cite
# "Lucia" while describing an unrelated prior case, or use the phrase
# "properly appointed" in a wholly different context, so the LLM field is
# used to confirm the challenge is actually raised. Code's decision table
# trusts the field's yes/no over the raw keyword hit when the two disagree,
# and falls back to the keyword predicate alone when the field is
# missing/unclear.
import re

LLM_FIELDS = {
    "appointments_challenge": (
        "In <=10 words, does the claimant argue the ALJ was NOT "
        "constitutionally/properly appointed (Appointments Clause/Lucia)? "
        "Answer yes or no."
    ),
}

_KEYWORD = re.compile(
    r"appointments clause|\blucia\b|unconstitutionally appointed|"
    r"(?:properly|validly) appointed|ssr\s?19-1p|appointment of (?:the )?alj",
    re.I,
)

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""

        kw_present = bool(_KEYWORD.search(t))

        ex = extracted if isinstance(extracted, dict) else {}
        answer = str(ex.get("appointments_challenge") or "").strip().lower()

        if "yes" in answer:
            final = 0.95 if kw_present else 0.75
        elif "no" in answer:
            final = 0.3 if kw_present else 0.05
        else:
            # field missing/unclear: fall back to the keyword predicate alone
            final = 0.8 if kw_present else 0.05

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
