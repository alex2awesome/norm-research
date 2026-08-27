"""a333: Rule of three / triples (hybrid, 4-field).

Criterion: use of triads to establish a minimal rhythmic pattern that primes
a surprising or escalated third beat (and can imply plurality visually).

Design (original 2 fields):
  - Keyword-only signals (e.g. bare "three", "and", "finally") are WEAK
    proxies: many texts mention "three" as a plain fact/quantity, or use
    "first"/"second" without ever completing a triad, and the v0 baseline
    over-rewards these coincidental hits.
  - The real signal is STRUCTURAL: three parallel beats (dialogue turns,
    repeated actions, list items) where the third (or an inserted 4th
    element that displaces the expected third) delivers an escalation,
    subversion, or surprise. That semantic judgment (is beat 3 different/
    escalated vs. beats 1-2?) is a THICK-INPUT construct code cannot
    reliably reach on its own, so it is delegated to the LLM fields below.
  - Code keeps the PREDICATE: it parses/validates the LLM's short answers,
    and independently corroborates with cheap structural regex evidence
    (ordinal-word sequencing "first...second...third", the bare numeral/
    "triple" family, and repeated-content-word counts in the 2-4 range,
    which catches non-ordinal triads like a phrase repeated three times
    with escalating punctuation, e.g. "Ouch! Ouch!! OUCH!!!").
  - Corroboration is weighted lightly when the LLM already found a triad
    (nudge only) and more heavily when the LLM found nothing (partial
    rescue for extractor misses), but can never alone reach the top of
    the range - genuine top scores require LLM-confirmed escalation.

GAP the original 2 fields plausibly miss (construct grounds only -- no
judge/eval signal consulted): `triad_items` is instructed to answer NONE
unless the text specifically reads as "the" rule-of-three, which conflates
two different failures once it says NONE -- "no sequential-beat structure at
all" vs. "there IS a beat structure, but it's 2, 4, or 5 beats, not 3" (a
near-miss the code-side fallback can only weakly infer from raw regex). A
raw, unfiltered `beat_count` field gives code an independent extractive
count regardless of the first field's own triad-or-not filtering, letting
the structural bonus be graduated by literal distance-from-3 rather than a
blunt keyword fallback. Second, "rhythmic pattern" in the criterion names
PARALLEL FORM specifically (matching phrasing/cadence across the beats) as
the mechanism that produces the rhythm -- neither existing field asks
whether the beats are phrased in parallel (e.g. "The first Eskimo says
X... The second Eskimo says Y...") vs. merely thematically related but
free-form. `parallel_form` probes that construct directly. Both new fields
gate on the original `triad_present`/absence logic and degrade gracefully:
if absent from `extracted` (not yet extracted), score() reduces exactly to
the original 2-field formula.
"""
import re
from collections import Counter

LLM_FIELDS = {
    "triad_items": "List the 3 parallel beats/items (comma or semicolon separated) if this text uses a rule-of-three structure, else NONE.",
    "third_twist": "In up to 8 words, say how the third/final beat escalates or subverts the first two beats, else NONE.",
    "beat_count": "State the number of sequential parallel beats/list items in the text (an integer), or 0 if none present.",
    "parallel_form": "In <=8 words, do the beats repeat the same sentence pattern/phrasing, or vary freely, or 'no beats'?",
}

_NONE_TOKENS = {"", "none", "n/a", "na", "no", "nothing", "not applicable"}

_NEGATIVE_TWIST_MARKERS = (
    "same as", "no differ", "identical", "no escalat", "no surpris",
    "nothing differ", "no twist", "no change", "not differ",
)

_STOPWORDS = {
    "that", "this", "with", "from", "have", "those", "these", "would",
    "could", "should", "there", "their", "when", "what", "your", "were",
    "been", "into", "just", "like", "some", "them", "then", "also",
    "because", "after", "before", "during", "which", "about", "where",
    "being", "doing", "having", "only", "very", "such", "most", "more",
    "than", "over", "under", "again", "further", "once", "here", "other",
    "same", "each", "both", "against", "between", "through", "above",
    "below", "down", "near", "upon", "within", "without", "will", "shall",
    "went", "walk", "walks", "walked", "walking",
}

_ORDINAL_WORDS = ("first", "second", "third")
_NUMERAL_RE = re.compile(r"\b(three|trio|triple|triad)\b")
_CONTENT_WORD_RE = re.compile(r"\b[a-zA-Z]{4,}\b")

# --- new-field parsing vocab --------------------------------------------

_INT_RE = re.compile(r"\d+")
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_PARALLEL_YES_RE = re.compile(
    r"\b(same|match\w*|parallel|repeat\w*|mirror\w*|identical)\b", re.IGNORECASE
)
_PARALLEL_NO_RE = re.compile(
    r"\b(vary|varies|varied|differ\w*|no beats|different|freely)\b", re.IGNORECASE
)


def _clean_field(value) -> str:
    try:
        return (value or "").strip()
    except Exception:
        return ""


def _is_none_token(s: str) -> bool:
    return s.strip().lower() in _NONE_TOKENS


def _parse_beat_count(value):
    """Extract an integer beat count from a short free-text answer, or
    None if unparseable/absent. Independent of the triad_items filter --
    this can be non-None even when triad_items answered NONE.
    """
    s = _clean_field(value)
    if not s or _is_none_token(s):
        return None
    m = _INT_RE.search(s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    sl = s.lower()
    for w, n in _NUMBER_WORDS.items():
        if w in sl:
            return n
    return None


def _parallel_form_bonus(value) -> float:
    s = _clean_field(value)
    if not s or _is_none_token(s):
        return 0.0
    has_yes = bool(_PARALLEL_YES_RE.search(s))
    has_no = bool(_PARALLEL_NO_RE.search(s))
    if has_yes and not has_no:
        return 0.15
    return 0.0


def _max_repeated_content_word_count(t: str) -> int:
    counts = Counter()
    for m in _CONTENT_WORD_RE.finditer(t):
        w = m.group(0).lower()
        if w in _STOPWORDS:
            continue
        counts[w] += 1
    if not counts:
        return 0
    return max(counts.values())


def _ordinal_sequence_score(t: str) -> float:
    positions = {}
    for w in _ORDINAL_WORDS:
        m = re.search(r"\b" + w + r"\b", t)
        positions[w] = m.start() if m else None
    if all(positions[w] is not None for w in _ORDINAL_WORDS):
        if positions["first"] < positions["second"] < positions["third"]:
            return 0.20
        return 0.12
    if positions["third"] is not None and (
        positions["first"] is not None or positions["second"] is not None
    ):
        return 0.08
    return 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            normed = ops.normalize(raw)
        except Exception:
            normed = raw
        t = (normed or raw).lower()
        if not t.strip():
            return 0.5

        extracted = extracted or {}
        triad_field = _clean_field(extracted.get("triad_items", ""))
        twist_field = _clean_field(extracted.get("third_twist", ""))
        triad_present = not _is_none_token(triad_field)

        # --- LLM-grounded component: did the extractor find a real triad? ---
        if triad_present:
            items = [x.strip() for x in re.split(r"[;,]", triad_field) if x.strip()]
            n = len(items)
            if n == 3:
                a = 0.65
            elif n in (2, 4):
                a = 0.45
            else:
                a = 0.35

            if twist_field and not _is_none_token(twist_field):
                twist_l = twist_field.lower()
                if any(marker in twist_l for marker in _NEGATIVE_TWIST_MARKERS):
                    twist_bonus = 0.0
                else:
                    twist_bonus = 0.25
            else:
                twist_bonus = 0.0
            a = min(1.0, a + twist_bonus)

            # NEW: parallel phrasing/cadence across the beats -- the
            # "rhythmic pattern" mechanism the criterion names, which
            # neither triad_items nor third_twist directly probes. Only
            # meaningful once a triad has actually been identified.
            if "parallel_form" in extracted:
                a = min(1.0, a + _parallel_form_bonus(extracted.get("parallel_form", "")))
        else:
            a = 0.15

        # --- code-side structural corroboration (cheap, deterministic) ---
        ordinal_score = _ordinal_sequence_score(t)
        numeral_score = 0.05 if _NUMERAL_RE.search(t) else 0.0
        rep_count = _max_repeated_content_word_count(t)
        if rep_count == 3:
            repeat_score = 0.15
        elif rep_count == 4:
            repeat_score = 0.10
        elif rep_count == 2:
            repeat_score = 0.03
        else:
            repeat_score = 0.0
        b = min(0.35, ordinal_score + numeral_score + repeat_score)

        # Best-effort sentence-count gate: a single run-on "sentence" is a
        # weaker candidate for a multi-beat structure. Never fatal.
        n_sent = None
        try:
            stats = ops.sent_stats(raw)
            if isinstance(stats, dict):
                for k in ("n_sent", "num_sentences", "n_sentences", "sentence_count"):
                    if k in stats:
                        n_sent = stats[k]
                        break
            elif stats:
                n_sent = stats[0]
        except Exception:
            n_sent = None
        if isinstance(n_sent, (int, float)) and n_sent <= 1:
            b *= 0.5

        # If the LLM already confirmed a triad, corroboration only nudges;
        # if the LLM found nothing, allow structural evidence to partially
        # rescue (extractor miss) but never reach the top of the range.
        weight = 0.25 if triad_present else 0.6
        final = a + weight * b

        # NEW: raw beat_count, independent of the triad_items filter. This
        # rescues the case triad_items conflates -- "no structure" vs. "a
        # near-miss 2/4/5-beat structure" -- by directly grading distance
        # from the minimal-triad target of 3, whatever triad_items said.
        if "beat_count" in extracted:
            beat_count = _parse_beat_count(extracted.get("beat_count", ""))
            if beat_count is not None:
                if beat_count == 3:
                    c = 0.20
                elif beat_count in (2, 4):
                    c = 0.10
                else:
                    c = 0.0
                c_weight = 0.5 if triad_present else 0.7
                final += c_weight * c

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
