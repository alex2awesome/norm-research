"""a15 hybrid: code regex/date-window scans for Title-II-DIB and "date last insured" phrasing as a fallback; two LLM fields name the stated claim type and quote the actual DLI text for a conjunctive decision table."""

# Criterion is a conjunction of two factual predicates: (a) the claim is/
# includes a Title II DIB claim (as opposed to Title XVI SSI-only), and (b)
# a Date Last Insured is actually stated somewhere in the narrative. Higher
# score = stronger evidence BOTH predicates hold.
#
# Design: claim-type framing ("Title II", "DIB", "Sec. 404" vs "Title XVI",
# "SSI", "Sec. 416") and the DLI phrase itself are both short, standard-form
# legal vocabulary a regex can catch reliably, but two things need thicker
# reads: (1) some narratives discuss BOTH Title II and Title XVI (concurrent
# claims, or a Title XVI-only case that merely cites Title II boilerplate in
# a procedural-history recital) where naive keyword presence over-credits
# the DIB signal; (2) the DLI is sometimes phrased without the literal
# "date last insured" string (e.g. "insured status expired March 2019").
# Two LLM fields carry the disambiguated claim type and the DLI text itself;
# code falls back to the regex/date-window predicate alone when a field is
# missing, and always requires both sub-signals (min-style conjunction) so
# a strong DIB signal with no DLI evidence (or vice versa) cannot score high.
import re

LLM_FIELDS = {
    "claim_type": (
        "State the claimant's benefit claim type as written: Title II DIB, "
        "Title XVI SSI, both/concurrent, or unclear."
    ),
    "dli_stated": (
        "Quote the claimant's stated Date Last Insured (DLI) as written, "
        "else NONE if no DLI is stated anywhere in the text."
    ),
}

_TITLE_II = re.compile(
    r"\btitle\s*ii\b|\bdib\b|disability insurance benefits|"
    r"§\s*404\.|\bsection\s*404\b", re.I)
_TITLE_XVI = re.compile(
    r"\btitle\s*xvi\b|\bssi\b|supplemental security income|"
    r"§\s*416\.|\bsection\s*416\b", re.I)
_DLI_PHRASE = re.compile(
    r"date\s+last\s+insured|\bdli\b|insured\s+status\s+(?:expired|through)",
    re.I)
_DATE_NEAR = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2}",
    re.I)

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "unclear", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


def _code_dib(t):
    has_ii = bool(_TITLE_II.search(t))
    has_xvi_only = bool(_TITLE_XVI.search(t)) and not has_ii
    if has_ii:
        return 1.0
    if has_xvi_only:
        return 0.0
    return 0.3  # neither cue found: weak default, not a confident negative


def _code_dli(t):
    found_phrase = False
    for m in _DLI_PHRASE.finditer(t):
        found_phrase = True
        if _DATE_NEAR.search(t[m.end(): m.end() + 40]):
            return 1.0
    return 0.4 if found_phrase else 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""

        ex = extracted if isinstance(extracted, dict) else {}
        claim_type = str(ex.get("claim_type") or "").strip().lower()
        dli_raw = str(ex.get("dli_stated") or "").strip()

        code_dib = _code_dib(t)
        code_dli = _code_dli(t)

        if _is_none(claim_type):
            dib = code_dib
        else:
            has_ii = ("title ii" in claim_type) or ("dib" in claim_type) or ("disability insurance" in claim_type)
            has_xvi = ("title xvi" in claim_type) or ("ssi" in claim_type) or ("supplemental security" in claim_type)
            both = ("both" in claim_type) or ("concurrent" in claim_type)
            if has_ii or both:
                llm_dib = 1.0
            elif has_xvi:
                llm_dib = 0.0
            else:
                llm_dib = code_dib
            dib = 0.3 * code_dib + 0.7 * llm_dib

        if _is_none(dli_raw):
            dli = 0.3 * code_dli  # field explicitly says no DLI: trust it, keep a small code hedge
        else:
            llm_dli = 1.0 if any(c.isdigit() for c in dli_raw) else 0.5
            dli = 0.3 * code_dli + 0.7 * llm_dli

        final = min(dib, dli)
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
