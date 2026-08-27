"""a130 Title/Abstract/Keywords: code checks for an abstract heading, keyword-list regex, and
abstract sentence-structure balance via ops.sent_stats; LLM fields ground the actual title text
and keyword list to catch cases hidden by scraped-chrome mojibake or non-standard formatting."""
import re

LLM_FIELDS = {
    "title_text": "State the paper's title exactly as printed, in <=12 words, or NONE if no clear title appears.",
    "keyword_list": "List the paper's stated keywords or index terms comma-separated, or NONE if none are given.",
}

_NONE_STRINGS = {"", "none", "n/a", "na", "unknown", "not stated", "not present",
                 "no evidence", "not applicable", "not specified", "not mentioned", "unclear"}

_KW_HEADING = re.compile(r"\b(keywords?|index terms?)\s*[:\-]", re.I)
_ABSTRACT_HEADING = re.compile(r"\babstract\b", re.I)


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
        head = t[:1500]

        has_abstract_heading = bool(_ABSTRACT_HEADING.search(head))
        has_kw_regex = bool(_KW_HEADING.search(t[:3000]))

        try:
            n_sent, mean_wps, _frac_long = ops.sent_stats(head)
        except Exception:
            n_sent, mean_wps = 0, 0.0

        if n_sent <= 0:
            struct_score = 0.0
        else:
            struct_score = max(0.0, 1.0 - abs(n_sent - 4.0) / 8.0)
            if mean_wps > 45 or mean_wps < 6:
                struct_score *= 0.6

        extracted = extracted or {}
        title_field = _clean(extracted.get("title_text", ""))
        kw_field = _clean(extracted.get("keyword_list", ""))

        title_bonus = 0.15 if title_field else 0.0
        kw_bonus = 0.2 if (has_kw_regex or kw_field) else 0.0
        abstract_bonus = 0.15 if has_abstract_heading else 0.05

        s = 0.25 + 0.25 * struct_score + title_bonus + kw_bonus + abstract_bonus
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
