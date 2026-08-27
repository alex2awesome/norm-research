"""a11 hybrid: code counts regulatory-citation density and distinct doctrinal-topic breadth as the base predicate; two LLM fields carry the specific-vs-vague verdict and an argument-count estimate."""

# Criterion: whether the claimant's stated arguments identify concrete,
# doctrinally-cognizable errors (pinpoint a finding, cite the controlling
# reg/SSR, name the record evidence ignored) versus generalized disagreement
# or a bare request to reweigh evidence. Higher score = more specific,
# better-anchored arguments (a strong empirical predictor of remand).
#
# Design: per the corpus notes, statutory/citation quantities are genuinely
# load-bearing and extractable by code, so citation density and topic
# breadth are computed directly from text as a code-owned predicate. Whether
# the surrounding prose is actually "specific" (names the record fact and
# rule) rather than a conclusory label wearing a citation is a reading-
# comprehension judgment, so it's routed to an LLM field; code still owns
# the final weighting and caps the result low on an explicit "vague" verdict
# regardless of citation count (a plaintiff can cite a reg once and still
# argue generally).
import re

LLM_FIELDS = {
    "specificity_verdict": (
        "In <=12 words, are the claimant's arguments SPECIFIC (name a "
        "finding+rule+evidence) or VAGUE (general disagreement/reweighing)? "
        "Answer specific or vague."
    ),
    "distinct_error_count": (
        "How many distinct legal or evidentiary errors does the claimant "
        "argue in these facts? Give just a number."
    ),
}

_CITATION = re.compile(r"20\s?C\.?F\.?R\.?|SSR\s?\d{2}-\d|42\s?U\.?S\.?C\.?|§\s?\d|\bSSA\b", re.I)
_TOPIC_PATTERNS = {
    "rfc": r"\brfc\b|residual functional capacity",
    "credibility": r"credibility|symptom",
    "step5": r"step[- ]?5|job numbers?|vocational|\bdot\b",
    "step2": r"step[- ]?2|severity|severe",
    "opinion": r"supportability|consistency|treating|examining|opinion",
    "duty_develop": r"duty to develop|consultative exam|record gap",
    "legal_error": r"legal error|misapplied|wrong standard|de novo",
    "appointments": r"appointments clause|lucia",
}
_TOPIC_RE = {k: re.compile(v, re.I) for k, v in _TOPIC_PATTERNS.items()}

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

        cite_hits = len(_CITATION.findall(t))
        cite_score = min(1.0, cite_hits / 3.0)

        distinct_topics = sum(1 for pat in _TOPIC_RE.values() if pat.search(t))
        topic_score = min(1.0, distinct_topics / 4.0)

        code_score = 0.5 * cite_score + 0.5 * topic_score

        ex = extracted if isinstance(extracted, dict) else {}
        verdict = str(ex.get("specificity_verdict") or "").strip().lower()
        count_raw = ex.get("distinct_error_count")

        m = re.search(r"\d+", str(count_raw or ""))
        argn = int(m.group()) if m else 0
        argn_score = min(1.0, argn / 3.0)

        final = 0.4 * code_score + 0.25 * argn_score
        if "vague" in verdict:
            final = min(final, 0.25)
        elif "specific" in verdict:
            final = max(final, 0.55) + 0.2 * code_score
        else:
            final += 0.15  # field missing/unclear: fall back mostly on code_score

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
