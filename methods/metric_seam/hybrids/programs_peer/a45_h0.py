"""a45 Scalability & Feasibility: code scores scale/feasibility lexicon density plus a regex
for quantitative scale markers (e.g. "1M users"); LLM fields ground whether a scalability
claim is evidence-backed (numbers/systems) vs. bare assertion, and whether operational
constraints/costs are acknowledged, since keyword presence alone is a weak quality proxy."""
import re

LLM_FIELDS = {
    "scale_evidence": "State the concrete evidence given for scalability (numbers, systems tested, deployments), in <=15 words, or NONE if only asserted.",
    "feasibility_constraint": "State any operational constraint or cost (compute/latency/resources) mentioned, in <=12 words, or NONE if none.",
}

_NONE_STRINGS = {"", "none", "n/a", "na", "unknown", "not stated", "not present",
                 "no evidence", "not applicable", "not specified", "not mentioned", "unclear"}

_SCALE_LEX = re.compile(
    r"\b(scalab\w+|scales? to|deploy\w* at scale|production[- ]?ready|real[- ]time|"
    r"throughput|latency|resource[- ]constrained|operationally feasible|computational cost)\b", re.I)
_NUM_SCALE = re.compile(
    r"\b\d[\d,]*\s*(?:million|billion|thousand|k|m)?\s*"
    r"(?:users|requests|queries|records|nodes|gpus?|documents|samples)\b", re.I)


def _clean(v):
    if not v:
        return ""
    v = str(v).strip()
    if v.lower().strip(".") in _NONE_STRINGS:
        return ""
    return v


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        if len(t) < 20:
            return 0.5
        tl = t.lower()
        n_words = max(1, len(t.split()))
        scale = 500.0 / n_words

        lex_d = len(_SCALE_LEX.findall(tl)) * scale
        num_hit = bool(_NUM_SCALE.search(tl))

        base = 0.3 + 0.25 * min(1.2, lex_d) + (0.15 if num_hit else 0.0)

        extracted = extracted or {}
        ev_field = _clean(extracted.get("scale_evidence", ""))
        ev_raw = str(extracted.get("scale_evidence", "") or "").strip()
        con_field = _clean(extracted.get("feasibility_constraint", ""))

        field_adj = 0.0
        if ev_field:
            field_adj += 0.25
        elif ev_raw and lex_d > 0:
            # field explicitly returned NONE while lexicon fired: bare assertion, penalize.
            field_adj -= 0.15
        if con_field:
            field_adj += 0.1

        s = base + field_adj
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
