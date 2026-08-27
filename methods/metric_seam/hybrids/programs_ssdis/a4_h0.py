"""a4 hybrid: Medical opinion supportability/consistency articulation (20 C.F.R. § 404.1520c).

Criterion asks whether the ALJ, for post-3/27/2017 claims, explicitly
articulated BOTH the supportability and consistency factors for material
medical opinions rather than giving a conclusory "not persuasive" label. A
strong remand basis is a mandatory-factor articulation failure; failure to
discuss the other, non-mandatory factors is generally not reversible.
Higher score = stronger remand basis (missing/bare articulation).

Design: code owns a structural PREDICATE -- regex counts of the
"supportab-" and "consisten-" word roots as a cheap coverage signal for
whether the decision even uses the two mandatory-factor vocabulary at all.
Regex cannot tell whether that vocabulary was actually applied WITH
record-grounded reasoning for the key opinion, versus dropped in as a bare
conclusory label -- that is a reading-comprehension task, so it is routed
to two LLM fields. Code owns the predicate: it only rewards low coverage
when combined with an LLM-confirmed missing-reasoning signal, and applies
a separate, additive penalty when the LLM finds an actual bare label
(e.g., "not persuasive" with nothing after it).
"""
import re

LLM_FIELDS = {
    "persuasiveness_reasoning": (
        "Quote the ALJ's supportability or consistency reasoning given "
        "for the key medical opinion, else NONE."
    ),
    "bare_label": (
        "Quote a persuasiveness conclusion like 'not persuasive' given "
        "without supporting explanation, else NONE."
    ),
}

_SUPPORT_KW = re.compile(r"\bsupportab", re.I)
_CONSIST_KW = re.compile(r"\bconsisten", re.I)

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

        support_hits = len(_SUPPORT_KW.findall(t))
        consist_hits = len(_CONSIST_KW.findall(t))
        both_covered = 1.0 if (support_hits > 0 and consist_hits > 0) else 0.0
        coverage_term = min((support_hits + consist_hits) / 6.0, 1.0)
        low_coverage = max(0.0, 1.0 - max(both_covered, coverage_term))

        ex = extracted if isinstance(extracted, dict) else {}
        reasoning_missing = 1.0 if _is_none(ex.get("persuasiveness_reasoning")) else 0.0
        bare_present = 0.0 if _is_none(ex.get("bare_label")) else 1.0

        final = 0.3 * reasoning_missing + 0.4 * bare_present + 0.3 * low_coverage
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
