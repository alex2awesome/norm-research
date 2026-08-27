# Hybrid module for aspect a45: "Surprise, incongruity, and misdirection mechanics"
#
# Design idea: the criterion rewards jokes that build an expectation path and then
# deliver a "reconstructible" flip -- a reveal grounded in what came before, not just
# a pile of unrelated puns or a crude non-sequitur tacked on. From the corpus notes and
# contract, code can cheaply detect surface structure (quoted reveal at the end,
# multi-turn dialogue vs. one giant monologue of puns, explicit rule-of-three markers,
# lexical callback between setup and punchline, verbose/ornate phrasing). What code
# cannot reliably judge is *whether the ending is actually a meaningful subversion* of
# the setup, or whether the joke sets up and breaks a repeated pattern -- both require
# reading comprehension, so those are delegated to two LLM_FIELDS.
import re
import math
import statistics
from collections import Counter

LLM_FIELDS = {
    "twist_move": "In up to 6 words, name how the ending subverts/recontextualizes the setup; NONE if there is no real twist.",
    "pattern_break": "In up to 6 words, name a repeated pattern (list/rule-of-three/callback) the joke sets up and breaks; NONE if none.",
}

_STOPWORDS = set("""
a an the and or but if then so to of in on at for with without by from as is are was
were be been being this that these those it its it's he she they them his her their
i you we my your our not no yes do does did doing have has had having can could will
would should may might just about into over under again further once here there when
where why how all any both each few more most other some such only own same than too
very s t can will just don now up out down off him her me us it's im ive youre didnt
doesnt wasnt werent isnt arent wouldnt couldnt shouldnt cant won't
""".split())

_QUOTE_RE = re.compile(r'["“]([^"”]{2,400})["”]')
_PATTERN_KW_RE = re.compile(r'\b(first|second|third|firstly|secondly|thirdly|1\.|2\.|3\.)\b', re.I)
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(s):
    return _WORD_RE.findall(s.lower())


def _content_words(tokens):
    return [w for w in tokens if w not in _STOPWORDS and len(w) > 2]


def _safe_normalize(text, ops):
    try:
        norm = ops.normalize(text)
        if isinstance(norm, str) and norm.strip():
            return norm
    except Exception:
        pass
    return text or ""


def _safe_sent_stats(norm, ops):
    """Contract text describes sent_stats both as a dict and as a
    (n_sent, mean_words_per_sent, frac_long_words) tuple in different places --
    handle either shape defensively."""
    n_sent, mean_words, frac_long = None, None, None
    try:
        stats = ops.sent_stats(norm)
        if isinstance(stats, dict):
            n_sent = stats.get("n_sent")
            mean_words = stats.get("mean_words_per_sent")
            frac_long = stats.get("frac_long_words")
        elif isinstance(stats, (list, tuple)) and len(stats) >= 3:
            n_sent, mean_words, frac_long = stats[0], stats[1], stats[2]
    except Exception:
        pass
    return n_sent, mean_words, frac_long


def _safe_retrieve_mean_sim(norm, ops):
    try:
        neighbors = ops.retrieve_similar(norm, k=5)
        sims = []
        for item in neighbors or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            a, b = item[0], item[1]
            # contract text shows (similarity, datapoint_id); be defensive either way.
            if isinstance(a, str) and not isinstance(b, str):
                sims.append(float(b))
            elif isinstance(b, str) and not isinstance(a, str):
                sims.append(float(a))
            elif isinstance(a, (int, float)):
                sims.append(float(a))
        if sims:
            return sum(sims) / len(sims)
    except Exception:
        pass
    return None


def _clean_field(val):
    if not isinstance(val, str):
        return ""
    v = val.strip()
    if not v or v.lower() in ("none", "n/a", "na", "no", "no twist", "no pattern"):
        return ""
    return v


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        norm = _safe_normalize(raw, ops)
        n = len(norm)

        # --- sentence split (simple, robust) ---
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', norm.strip()) if s.strip()]
        if not sentences:
            sentences = [norm.strip()]
        punchline = sentences[-1]
        setup_text = norm[: max(0, n - len(punchline))] if len(sentences) > 1 else norm

        # --- quote structure ---
        quotes = _QUOTE_RE.findall(norm)
        dialogue_count = len(quotes)
        longest_quote_len = max((len(q) for q in quotes), default=0)
        monologue_ratio = (longest_quote_len / n) if n > 0 else 0.0

        ends_in_quote = bool(re.search(r'["”][^a-zA-Z0-9]{0,3}$', norm.strip())) or ('"' in punchline)

        # --- explicit rule-of-three / list markers ---
        pattern_hits = set(m.lower() for m in _PATTERN_KW_RE.findall(norm))

        # --- callback / grounded novelty between setup and punchline ---
        setup_words = set(_content_words(_tokenize(setup_text)))
        punch_words = _content_words(_tokenize(punchline))
        punch_word_set = set(punch_words)
        overlap = setup_words & punch_word_set
        novel = punch_word_set - setup_words
        novelty_ratio = (len(novel) / len(punch_word_set)) if punch_word_set else 0.0
        grounded = len(overlap) >= 1

        # --- sentence stats (economy / ornateness) ---
        _, _, frac_long = _safe_sent_stats(norm, ops)

        # --- evidence op: corpus-hazard shrinkage ---
        mean_sim = _safe_retrieve_mean_sim(norm, ops)

        # --- LLM-grounded constructs ---
        twist = _clean_field(extracted.get("twist_move", "")) if isinstance(extracted, dict) else ""
        pattern_break = _clean_field(extracted.get("pattern_break", "")) if isinstance(extracted, dict) else ""

        s = 0.28

        if ends_in_quote:
            s += 0.12
        if dialogue_count >= 3:
            s += 0.08
        if dialogue_count <= 1 and monologue_ratio > 0.55:
            # one giant quoted monologue crammed with wordplay, not a real exchange
            s -= 0.10
        if grounded:
            s += 0.14 * novelty_ratio
        if len(pattern_hits) >= 2:
            s += 0.06
        if isinstance(frac_long, (int, float)) and frac_long > 0.35:
            s -= 0.06
        if twist:
            s += 0.18
        if pattern_break:
            s += 0.10

        if mean_sim is not None and mean_sim < 0.05:
            # looks like a corpus-hazard document (nav chrome / boilerplate / outlier);
            # shrink toward neutral rather than trust the structural heuristics fully.
            s = 0.5 + 0.5 * (s - 0.5)

        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
