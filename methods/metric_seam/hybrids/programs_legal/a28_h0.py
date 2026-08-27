"""a28 hybrid: employer meets 15-employee threshold (Title VII coverage).

Criterion: does the document establish that the defendant employer has 15+
employees (the statutory Title VII coverage floor)?

Corpus hazard notes flag this as a MECHANICAL/quantity criterion (unlike
doctrinal constructs such as pretext or hostile environment): real employee
counts, when stated, are genuinely load-bearing, and date/count arithmetic
beats keyword scanning. The baseline (v0_structure, train rho=0.109) already
tries a count-regex + sigmoid-around-15, which is the right idea but the
regex is too brittle: it only matches "N employees/workers/staff/personnel"
or "employs/employed [approximately] N" verbatim, so it misses very common
paraphrases like "employs over 42,000 individuals" (filler word "over"
breaks the second pattern; noun "individuals" isn't in the first pattern's
noun list) -- exactly the kind of miss that tanks rho on cases where the
count IS stated but phrased loosely.

The baseline's fallback for the (much more common) no-explicit-count case is
a fixed corporation/small-business KEYWORD list. That is where it actively
misfires: a substring hit on "corporation" fires on ANY company with
"Corporation" in its legal name (e.g. "National Can Corporation"), regardless
of whether the case narrative gives any real signal about org size -- a false
positive the training data confirms (baseline 0.85 vs judge 0.0 on that
example). Judging employer scale from context (multi-site operations,
franchise/chain structure, government/institutional identity, vs. sole
proprietorship / single storefront / independent-contractor framing) is a
semantic read of the whole narrative that a keyword substring match cannot
reliably make -- that's thick-input grounding, so it goes to an LLM field.
We keep code responsible for: (a) all numeric arithmetic (the actual
predicate, sigmoid around n=15), and (b) a narrow keyword backstop so we
never regress below the baseline's clean explicit-term cases.
"""
import re
import math

LLM_FIELDS = {
    "explicit_employee_count": (
        "The exact employee/staff/headcount number stated for the defendant "
        "employer in the text (e.g. '42,000', '15', '3'), in any phrasing "
        "(including 'over', 'about', spelled-out numbers); answer NONE if no "
        "such number is given."
    ),
    "employer_scale_class": (
        "Based on how the defendant employer is described (operations, "
        "locations, structure), classify its scale as one word: 'large' "
        "(multi-location company, government agency, university, hospital "
        "system, national/regional chain), 'small' (sole proprietor, single "
        "storefront, family business, independent-contractor arrangement), "
        "or 'unclear' if the text gives no real basis to judge."
    ),
}

# code-side numeric backstop: same noun list as baseline, extended with a
# couple of common paraphrase nouns regex was missing.
_COUNT_NOUN = r'(?:employees|workers|staff(?:\s+members)?|personnel|individuals|people)'
_COUNT_PAT1 = re.compile(
    r'(\d[\d,]{0,7})\s*(?:\+|or more)?\s*(?:full[- ]time\s+)?' + _COUNT_NOUN
)
# filler-tolerant "employs/employed ... N" -- baseline only allowed
# "approximately"; real text also says "over", "more than", "about",
# "roughly", "at least", "a total of".
_FILLER = (
    r'(?:a\s+total\s+of\s+|approximately\s+|about\s+|roughly\s+|over\s+|'
    r'more\s+than\s+|at\s+least\s+|some\s+)?'
)
_COUNT_PAT2 = re.compile(r'(?:employs|employed|employing)\s+' + _FILLER + r'(\d[\d,]{0,7})')
_DIGITS = re.compile(r'(\d[\d,]{0,7})')

_SMALL_TERMS = [
    'sole proprietor', 'family-owned', 'family owned', 'small business',
    'mom-and-pop', 'independent contractor only', 'single location',
    'one location', 'local business',
]
_LARGE_TERMS = [
    'national chain', 'multinational', 'fortune 500', 'publicly traded',
    'nationwide retailer', 'large employer', 'thousands of employees',
    'multiple locations', 'branches nationwide',
]
_NONE_ANSWERS = {"", "none", "n/a", "na", "not stated", "not mentioned", "unknown"}


def _is_none_answer(s):
    return (not s) or s.strip().lower() in _NONE_ANSWERS


def _parse_int(s):
    try:
        return int(s.replace(',', ''))
    except (ValueError, AttributeError):
        return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        ext = extracted or {}
        try:
            norm = ops.normalize(text) if text else ""
        except Exception:
            norm = text or ""
        tl = norm.lower()

        # --- 1. numeric predicate: quantities are the load-bearing signal here ---
        counts = []
        for m in _COUNT_PAT1.findall(tl):
            n = _parse_int(m)
            if n is not None:
                counts.append(n)
        for m in _COUNT_PAT2.findall(tl):
            n = _parse_int(m)
            if n is not None:
                counts.append(n)

        llm_count_raw = ext.get('explicit_employee_count', '')
        if not _is_none_answer(llm_count_raw):
            m = _DIGITS.search(llm_count_raw)
            if m:
                n = _parse_int(m.group(1))
                if n is not None:
                    counts.append(n)

        if counts:
            n = max(counts)
            x = n - 15
            conf = 1.0 / (1.0 + math.exp(-x / 5.0))
            return max(0.0, min(1.0, conf))

        # --- 2. no explicit count anywhere: fall back to qualitative scale read ---
        small_hit = any(k in tl for k in _SMALL_TERMS)
        large_hit = any(k in tl for k in _LARGE_TERMS)
        scale = (ext.get('employer_scale_class', '') or '').strip().lower()

        if scale.startswith('large'):
            conf = 0.65 if small_hit else 0.8
        elif scale.startswith('small'):
            conf = 0.35 if large_hit else 0.2
        elif large_hit and not small_hit:
            conf = 0.7
        elif small_hit and not large_hit:
            conf = 0.25
        else:
            conf = 0.5

        return max(0.0, min(1.0, conf))
    except Exception:
        return 0.5
