"""a216 hybrid channel: Cultural authenticity, inclusivity, and specificity.

Train-residual reading (30 stratified examples):
  * Judge mass sits ~0.20-0.25: stories that simply never engage cultural/
    identity material land in a flat default band.
  * Low tail (0.0-0.1) is dominated by CARICATURE: joke accents / eye-dialect
    ("zis ISIS", "y'all ... I's tells ya"), slurs and derogatory group labels
    ("homo", "illegals", "wanna be Jamaican stoner"), and sexual
    objectification ("cleavage", "perky", "perfect breasts", "space titties").
  * High tail (0.3-0.55) portrays an othered identity or community with
    DIGNITY plus CONCRETE cultural texture (lycan "Alignment Day" customs,
    a homeless man rendered with interiority, a sympathetic female Lucifer).
  Keyword presence is a weak proxy (baseline rho=0.15); the respect-vs-mockery
  judgment needs thick input -> two LLM fields. The predicate stays in code:
  high-precision caricature regexes push down, LLM mockery verdict pushes
  down, LLM dignity+specificity verdict pushes up (gated if mockery fired),
  and a small TF-IDF kNN nudge toward labeled train neighbors (EVIDENCE op)
  exploits same-prompt siblings in the WritingPrompts corpus.
"""

import math
import re

LLM_FIELDS = {
    "group_mockery": (
        "Does the story mock, stereotype, or sexually objectify a real-world "
        "group (ethnicity, nationality, religion, gender, sexuality, class), "
        "e.g. joke accents, slurs, caricature? Answer 'group: how' or NONE."
    ),
    "respectful_specificity": (
        "Name one community or identity (real or invented) the story portrays "
        "with dignity AND concrete cultural detail (named practices, customs, "
        "idioms, lived texture). Answer 'who: detail' or NONE."
    ),
}

# ---------------------------------------------------------------- negatives
_EYE_DIALECT = re.compile(r"\b(zis|zat|zees|zey|vot|vhat|vell|ze)\b", re.I)
_DIALECT_MOCK = re.compile(
    r"\b(y'?all|yer|i'?s\s+tells?|som'?uh|tells\s+ya|hain'?t|gots\s+ta)\b", re.I
)
_DEROG = re.compile(
    r"\b(homos?|fags?|faggots?|trann(?:y|ies)|retard(?:s|ed)?|illegals|"
    r"wetbacks?|towelheads?|gypp?(?:ed|sy|sies))\b",
    re.I,
)
_DEROG_SOFT = re.compile(
    r"\b(stoners?|douchebags?|rednecks?|hillbill(?:y|ies)|white\s+trash|"
    r"savages?)\b",
    re.I,
)
_OBJECTIFY = re.compile(
    r"\b(titt(?:y|ies)|tits|cleavage|perky|busty|skimp\w*)\b"
    r"|\bperfect\s+breasts?\b"
    r"|\b(?:lacy|see-?through)\s+\w*\s*(?:bra|panties)\b",
    re.I,
)
_WANNABE_NAT = re.compile(r"\bwanna\s*-?\s*be\s+[A-Z][a-z]+")

# ---------------------------------------------------------------- positives
_PRACTICE = re.compile(
    r"\b(rituals?|customs?|traditions?|festivals?|ceremon(?:y|ies)|prayers?|"
    r"idioms?|dialects?|heritage|ancestral|elders?|blessings?|pilgrimage|"
    r"sabbath|recipes?)\b",
    re.I,
)

_NONE_RE = re.compile(r"^\s*(none|no|n/?a|nothing|not)\b", re.I)

# datapoint_id -> judge_score_0_1 for the 30 train examples (kNN evidence)
_TRAIN_JUDGE = {
    "d00009": 0.0, "d04046": 0.0, "d01355": 0.0, "d02316": 0.0, "d03858": 0.0,
    "d01736": 0.05, "d01872": 0.1, "d01448": 0.1, "d04860": 0.1, "d04082": 0.1,
    "d01801": 0.2, "d04809": 0.2, "d01219": 0.2, "d02067": 0.2, "d03302": 0.2,
    "d01258": 0.2, "d01598": 0.2, "d04236": 0.2, "d02871": 0.2, "d00524": 0.25,
    "d00223": 0.25, "d03791": 0.25, "d00930": 0.25, "d04167": 0.25,
    "d00610": 0.25, "d02045": 0.3, "d00700": 0.35, "d00858": 0.35,
    "d03584": 0.4, "d04174": 0.55,
}


def _field(extracted, key):
    """Return a cleaned LLM answer, or '' if absent / NONE-like."""
    try:
        val = (extracted or {}).get(key, "")
    except Exception:
        return ""
    if not isinstance(val, str):
        return ""
    val = val.strip()
    if not val or _NONE_RE.match(val):
        return ""
    return val


def _verse_like(t):
    """Comedy-verse detector: many bare-ended short lines (one train zero)."""
    if "[Poem]" in t:
        return True
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if len(lines) < 10:
        return False
    bare = sum(1 for ln in lines if not re.search(r"[.!?,;:\"'”…]$", ln))
    words = [len(ln.split()) for ln in lines]
    mean_words = sum(words) / float(len(words))
    return (bare / float(len(lines))) >= 0.7 and mean_words <= 12.0


def _knn_nudge(text, ops, s):
    """Blend lightly toward judge scores of TF-IDF-close labeled train docs."""
    try:
        nbrs = ops.retrieve_similar(text, k=5) or []
    except Exception:
        return s
    hits = []
    for item in nbrs:
        try:
            sim, did = float(item[0]), str(item[1])
        except Exception:
            continue
        if sim > 0.95:  # top hit may be the story itself
            continue
        if sim >= 0.30 and did in _TRAIN_JUDGE:
            hits.append((sim, _TRAIN_JUDGE[did]))
    if not hits:
        return s
    est = sum(sim * j for sim, j in hits) / sum(sim for sim, _ in hits)
    w = min(0.20, 0.35 * max(sim for sim, _ in hits))
    return (1.0 - w) * s + w * est


def score(text, extracted, ops):
    try:
        raw = text or ""
        try:
            t = ops.normalize(raw) or raw
        except Exception:
            t = raw

        # ---- negative evidence: caricature / mockery / objectification
        w_neg = (
            0.7 * len(_EYE_DIALECT.findall(t))
            + 0.6 * len(_DIALECT_MOCK.findall(t))
            + 1.0 * len(_DEROG.findall(t))
            + 0.5 * len(_DEROG_SOFT.findall(t))
            + 0.8 * len(_OBJECTIFY.findall(t))
            + 1.0 * len(_WANNABE_NAT.findall(t))
        )
        neg_code = 0.16 * math.tanh(w_neg / 2.5)

        mock = _field(extracted, "group_mockery")
        neg_llm = 0.10 if mock else 0.0
        mock_fired = bool(mock) or neg_code > 0.08

        # ---- positive evidence: dignity + concrete cultural texture
        spec = _field(extracted, "respectful_specificity")
        pos_llm = 0.0
        if spec:
            pos_llm = 0.15 if len(spec) >= 12 else 0.08
            if mock_fired:
                pos_llm *= 0.4  # caricature elsewhere outweighs texture
        pos_code = min(0.04, 0.015 * len(_PRACTICE.findall(t)))

        s = 0.22 - neg_code - neg_llm + pos_llm + pos_code
        if _verse_like(t):
            s -= 0.05

        s = _knn_nudge(t, ops, s)
        return max(0.02, min(0.9, s))
    except Exception:
        return 0.5
