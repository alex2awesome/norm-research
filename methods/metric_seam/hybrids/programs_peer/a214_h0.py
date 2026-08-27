"""a214 Quotation Usage: code checks straight-quote balance (after ops.normalize collapses
curly quotes) and regex-detects defining-quote patterns ("X" refers to/means/etc.); LLM fields
ground one concrete quoted-term/definition pair and flag semantic misuse (unmatched quotes,
scare-quote/emphasis use) that balance-checking alone can't distinguish from correct usage."""
import re

LLM_FIELDS = {
    "quoted_definition": "Quote one term-in-quotes together with its defining phrase from the text, or NONE if no term is defined via quotes.",
    "quote_misused": "Answer YES if quotation marks are used incorrectly (unmatched, or as emphasis not quotation) anywhere in the text, else NO.",
}

_NONE_STRINGS = {"", "none", "n/a", "na", "unknown", "not stated", "not present",
                 "no evidence", "not applicable", "not specified", "not mentioned", "unclear"}

_DEFINE_PAT = re.compile(
    r'"[^"]{2,60}"\s+(?:refers to|is defined as|means|denotes|is called|we (?:call|term|refer to))\b'
    r'|(?:refers to|is defined as|means|denotes|is called|we (?:call|term|refer to))\s+"[^"]{2,60}"',
    re.I)


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

        n_quotes = t.count('"')
        balanced = (n_quotes % 2 == 0)
        define_hits = len(_DEFINE_PAT.findall(t))

        extracted = extracted or {}
        qd_field = _clean(extracted.get("quoted_definition", ""))
        misuse_field = _clean(extracted.get("quote_misused", ""))

        if n_quotes == 0:
            base = 0.5  # no quotation activity at all: neutral, not penalized
        else:
            base = 0.45
            if not balanced:
                base -= 0.15

        define_bonus = min(0.3, 0.15 * define_hits)

        field_adj = 0.0
        if qd_field:
            field_adj += 0.2
        if misuse_field.upper().startswith("YES"):
            field_adj -= 0.25

        s = base + define_bonus + field_adj
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
