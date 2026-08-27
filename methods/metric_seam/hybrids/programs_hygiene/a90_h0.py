# Hybrid module for humor aspect a90: "Storytelling and personal material"
#
# Criterion: shape stories (argument vs. laugh modules), mining lived
# experience -- including hard events -- with specificity and emotion that
# universalizes without pleading.
#
# Design rationale (see improver pack a90.json):
#   - The baseline (first-person pronoun density) is a weak, topic-level
#     proxy: it scores stock jokes highly whenever dialogue happens to use
#     "I" (e.g. a bar-joke goose saying "I'd like to buy some peanuts").
#   - The real construct is CRAFT: is this a genuinely developed personal
#     narrative (scene, sequence, concrete grounding detail, earned emotion)
#     or a generic joke template (bar joke, blonde joke, riddle, one-line
#     quip, "rule of three" escalation gag) that merely borrows first-person
#     dialogue?
#   - Surface structure (length, temporal/sequential connectives, numeric
#     and proper-noun specificity, family/relation terms, emotion
#     vocabulary, dialogue density, known stock-joke openers, pleading
#     language) is reachable by regex/stdlib and is coded directly.
#   - Distinguishing "stock joke template" from "genuine personal anecdote",
#     and judging whether a concrete grounding detail is actually present,
#     is a semantic call regex cannot reliably make (e.g. an escalating
#     "three days in a row" structure can be either a real timeline or a
#     classic joke device). Those two constructs are pushed to the LLM
#     fields; the code keeps the combination rule (predicate stays in code).

import re, math

LLM_FIELDS = {
    "story_frame": (
        "In <=6 words, name this text's form: 'personal anecdote', "
        "'stock joke template', 'one-line quip', 'poem/list', or "
        "'riddle/Q&A'; else NONE."
    ),
    "concrete_detail": (
        "Quote the single most specific concrete detail (a name, number, "
        "place, or timeframe) that grounds this as a real lived scene, or "
        "answer NONE if everything is generic."
    ),
}

_STOCK_OPENERS = [
    r"\bwalks? into a bar\b",
    r"\bwalked into a bar\b",
    r"\bknock,?\s*knock\b",
    r"\bwhy did the\b",
    r"\bwhat do you call\b",
    r"\b(a|an|two|three)\s+(blonde|priest|rabbi|lawyer|doctor|nun|irishman|"
    r"scotsman|englishman|genie)\b.{0,40}\b(walks|walked|goes|went|says|"
    r"said)\b",
    r"\bthere (was|once was) a\b",
    r"\bso a\b.{0,30}\bwalks?\b",
    r"\byo mama\b",
]

_TEMPORAL_MARKERS = [
    "the next day", "next day", "years later", "months later", "when i was",
    "growing up", "eventually", "finally", "suddenly", "meanwhile",
    "after a while", "a few days", "a few years", "one day", "that day",
    "the following", "later that", "back then", "as a kid", "as a child",
]

_FAMILY_TERMS = [
    "mom", "mommy", "mother", "dad", "daddy", "father", "grandfather", "grandmother",
    "grandson", "granddaughter", "brother", "sister", "wife", "husband",
    "daughter", "son", "boyfriend", "girlfriend",
]

_EMOTION_TERMS = [
    "surprised", "shocked", "embarrassed", "disappointed", "livid",
    "furious", "pissed", "proud", "terrified", "heartbroken", "devastated",
    "relieved", "ashamed", "humiliated", "grateful", "nervous", "anxious",
]

_PLEADING_MARKERS = [
    "please don't judge", "sorry if this", "trigger warning",
    "not sure if this belongs", "just needed to share", "please be nice",
    "hope this doesn't", "i know this is sad but",
]


# HYGIENE PATCH: `t.count(term)` is a bare substring count over the whole
# document string, so "son" (in _FAMILY_TERMS) fired on "poisonous"/"reasons"/
# "seasoning"/"personality"/"reasoned" and "mom" fired on "moment"/"moments" --
# all unrelated to the family/relation concept. Word-bounded, with a small
# inflection whitelist so real forms ("son's", "moms") still count.
_TERM_PATTERN_CACHE = {}


def _term_pattern(term):
    cached = _TERM_PATTERN_CACHE.get(term)
    if cached is not None:
        return cached
    esc = re.escape(term)
    if " " in term or "-" in term:
        pat = r"\b" + esc + r"\b"
    else:
        pat = r"\b" + esc + r"(?:'s|s|es|ed|ing)?\b"
    compiled = re.compile(pat)
    _TERM_PATTERN_CACHE[term] = compiled
    return compiled


def _count_hits(t, terms):
    return sum(len(_term_pattern(term).findall(t)) for term in terms)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        t = ops.normalize(raw)
        tl = t.lower()

        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(t)
        except Exception:
            n_sent, mean_wps, frac_long = (1, 0.0, 0.0)

        words = re.findall(r"[a-zA-Z']+", tl)
        nw = max(1, len(words))

        # --- stock joke template detection ---
        stock_hit = any(re.search(pat, tl) for pat in _STOCK_OPENERS)

        # --- narrative sequencing (temporal/sequential connectives) ---
        temporal_hits = _count_hits(tl, _TEMPORAL_MARKERS)
        temporal_score = 1.0 - math.exp(-temporal_hits * 0.9)

        # --- specificity: numbers, dates, and mid-sentence proper nouns ---
        num_hits = len(re.findall(r"\b\d+([.,]\d+)?\b", t))
        try:
            dates = ops.extract_dates(t) or []
        except Exception:
            dates = []
        sentences = re.split(r"(?<=[.!?])\s+", t)
        proper_hits = 0
        for s in sentences:
            toks = s.split()
            for tok in toks[1:]:
                w = re.sub(r"[^A-Za-z]", "", tok)
                if len(w) > 1 and w[0].isupper() and w.lower() != "i":
                    proper_hits += 1
        specificity_raw = num_hits + 0.5 * proper_hits + 1.5 * len(dates)
        specificity_score = 1.0 - math.exp(-specificity_raw * 0.25)

        # --- personal ownership: first-person density + family/relation terms ---
        fp = ["i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"]
        fp_hits = sum(1 for w in words if w in fp)
        fp_ratio_score = 1.0 - math.exp(-(fp_hits / nw) * 12)
        family_hits = _count_hits(tl, _FAMILY_TERMS)
        family_score = 1.0 - math.exp(-family_hits * 0.6)
        personal_score = 0.6 * fp_ratio_score + 0.4 * family_score

        # --- length / development ---
        length_score = min(1.0, nw / 180.0)

        # --- dialogue density (heavy quotes + short overall = quip exchange) ---
        quote_chars = len(re.findall(r'["“”]', t))
        dialogue_density = quote_chars / max(1, len(t))
        dialogue_penalty = max(0.0, dialogue_density - 0.03) * 3.0

        # --- emotion vocabulary ---
        emo_hits = _count_hits(tl, _EMOTION_TERMS)
        emo_score = 1.0 - math.exp(-emo_hits * 0.8)

        # --- pleading language penalty ---
        pleading_hit = any(m in tl for m in _PLEADING_MARKERS)

        # --- structural shape: multi-sentence development, not a single zinger ---
        struct_score = min(1.0, max(0.0, (n_sent - 3) / 8.0))

        core = (
            0.22 * length_score
            + 0.16 * temporal_score
            + 0.16 * specificity_score
            + 0.18 * personal_score
            + 0.10 * emo_score
            + 0.18 * struct_score
        )
        core = max(0.0, min(1.0, core - dialogue_penalty))

        if stock_hit:
            core = min(core, 0.18)
        if pleading_hit:
            core *= 0.5

        # --- LLM field: story_frame (genre/formula recognition) ---
        frame = (extracted.get("story_frame") or "").strip().lower()
        if frame and frame != "none":
            if any(
                k in frame
                for k in ("stock", "template", "riddle", "quip", "pun", "q&a",
                          "one-line", "joke format", "poem", "list")
            ):
                core = min(core, 0.2)
            elif any(
                k in frame for k in ("personal", "anecdote", "narrative",
                                      "memoir", "story")
            ):
                core = min(1.0, core + 0.15)

        # --- LLM field: concrete_detail (grounding-detail presence) ---
        detail = (extracted.get("concrete_detail") or "").strip().lower()
        if detail and detail != "none":
            core = min(1.0, core + 0.12)
        else:
            core = max(0.0, core - 0.05)

        # --- evidence op: near-duplicate stock content dampener ---
        try:
            neighbors = ops.retrieve_similar(text, k=5)
            if neighbors:
                sims = [sim for (sim, _did) in neighbors]
                mean_sim = sum(sims) / len(sims)
                if mean_sim > 0.55:
                    core = max(0.0, core - 0.1)
        except Exception:
            pass

        return max(0.0, min(1.0, core))
    except Exception:
        return 0.5
