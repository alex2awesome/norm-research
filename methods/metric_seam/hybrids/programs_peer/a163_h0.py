"""a163 Reasoning Quality: code scores hedge/justification-vs-overclaim lexicon density from
regex; LLM fields ground one concrete overclaiming phrase and one concrete justification for
the paper's main claim, since code alone can't tell an asserted claim from a supported one."""
import re

LLM_FIELDS = {
    "overclaim_phrase": "Quote the single most unsupported/overclaiming superlative phrase in the text, or NONE if none exists.",
    "claim_justification": "In <=15 words, state the reason or evidence given for the paper's main claim, or NONE if unjustified.",
}

_NONE_STRINGS = {"", "none", "n/a", "na", "unknown", "not stated", "not present",
                 "no evidence", "not applicable", "not specified", "not mentioned", "unclear"}

_OVERCLAIM_PAT = re.compile(
    r"\b(state[- ]of[- ]the[- ]art|unprecedented|first ever|far surpasses|dramatically|"
    r"substantially outperforms|always|never fails|guarantees?|proves? that|"
    r"indisputabl\w*|undeniably|revolutionary|best (?:known|possible)|clearly superior)\b", re.I)
_HEDGE_PAT = re.compile(
    r"\b(suggests?|indicates?|may|might|appears? to|hypothesiz\w+|we argue|preliminary|"
    r"to our knowledge|in this setting|under these conditions|tends? to)\b", re.I)
_REASON_PAT = re.compile(
    r"\b(because|therefore|since|given that|as a result|due to|this is because|which explains)\b", re.I)


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

        overclaim_d = len(_OVERCLAIM_PAT.findall(tl)) * scale
        hedge_d = len(_HEDGE_PAT.findall(tl)) * scale
        reason_d = len(_REASON_PAT.findall(tl)) * scale

        lex = (0.5
               + 0.15 * min(2.0, hedge_d)
               + 0.15 * min(2.0, reason_d)
               - 0.25 * min(2.0, overclaim_d))
        lex = max(0.0, min(1.0, lex))

        extracted = extracted or {}
        overclaim_field = _clean(extracted.get("overclaim_phrase", ""))
        justified_field = _clean(extracted.get("claim_justification", ""))

        field_adj = 0.0
        if overclaim_field:
            field_adj -= 0.2
        if justified_field:
            field_adj += 0.2

        s = lex + field_adj
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
