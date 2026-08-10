"""a17 hybrid: code regex flags a numbered Medical-Vocational Guideline rule citation as a fallback; two LLM fields quote the actual rule number and state whether VE testimony was also used, feeding a decision table."""

# Criterion is a factual predicate: did the ALJ's disability finding rest on
# a SPECIFIC NUMBERED grid rule (e.g. "Rule 202.14") directing the outcome,
# as opposed to (or in addition to) Vocational Expert testimony identifying
# jobs. Higher score = stronger evidence the conclusion was grid-directed,
# with an extra boost when the grid was used ALONE (the cleanest form of
# the criterion) and a penalty when VE testimony did the real work instead.
#
# Design: the rule-number citation format (###.## following "Rule"/
# "Guideline"/"Table") is rigid, standard boilerplate a regex catches
# reliably, so code owns that predicate directly. What regex can't safely
# resolve is (a) whether a matched rule number is genuinely DIRECTING the
# outcome versus being cited in a background/regulatory-framework recital,
# and (b) whether VE testimony was used alongside it -- both need a read of
# surrounding context, so two LLM fields carry those distinctions. Code
# trusts an LLM-quoted rule number over a bare regex hit, and falls back to
# the regex predicate alone (with a VE-presence regex hedge) when a field is
# missing.
import re

LLM_FIELDS = {
    "grid_rule_cited": (
        "Quote the specific numbered Medical-Vocational Guideline Rule "
        "(e.g. 'Rule 202.14') the ALJ cited to direct the disability "
        "finding; NONE if no specific numbered rule is cited."
    ),
    "ve_testimony_used": (
        "State whether the step-5 finding also relied on separate "
        "Vocational Expert (VE) testimony about specific jobs, or on the "
        "grid rule ALONE: answer 'grid alone', 'VE used', or NONE."
    ),
}

_GRID_RULE = re.compile(
    r"\b(?:medical-?vocational\s+)?(?:rule|guideline)\s*(?:no\.?\s*)?"
    r"\d{3}\.\d{2}\b", re.I)
_TABLE_RULE = re.compile(
    r"\btable\s+(?:no\.?\s*)?\d+\b.{0,30}\brule\s*\d{3}\.\d{2}\b", re.I)
_VE_PRESENT = re.compile(
    r"vocational expert|\bVE\b\W{0,10}(?:testified|testimony)", re.I)
_RULE_NUM_IN_STR = re.compile(r"\d{2,3}\.\d{2}")

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "unclear", ""}


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

        code_hit = bool(_GRID_RULE.search(t)) or bool(_TABLE_RULE.search(t))
        code_ve = bool(_VE_PRESENT.search(t))

        ex = extracted if isinstance(extracted, dict) else {}
        rule_raw = str(ex.get("grid_rule_cited") or "").strip()
        ve_raw = str(ex.get("ve_testimony_used") or "").strip().lower()

        if _is_none(rule_raw):
            base = 0.1 if code_hit else 0.0
        elif _RULE_NUM_IN_STR.search(rule_raw):
            base = 1.0
        else:
            base = 0.7 if code_hit else 0.5

        if _is_none(ve_raw):
            mult = 0.7 if code_ve else 1.0
        elif "alone" in ve_raw:
            mult = 1.15
        elif "ve" in ve_raw or "vocational expert" in ve_raw:
            mult = 0.55
        else:
            mult = 1.0

        final = base * mult
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
