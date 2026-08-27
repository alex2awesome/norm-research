"""u3 Laugh Frequency: code scores textual "beat density" (a pacing/length proxy for laughs-per-minute) from structural cues; LLM fields carry the semantic count of distinct comedic beats and their hit-rate that code cannot judge."""

import re, math

LLM_FIELDS = {
    "beat_count": (
        "Count the distinct comedic beats or laugh-worthy lines in this "
        "text (not just the final punchline). Answer 'one', 'two', or "
        "'three-or-more'."
    ),
    "hit_rate": (
        "Of the comedic attempts in this text, do most land successfully, "
        "or are several flat/unfunny lines mixed in? Answer 'mostly land', "
        "'mixed', or 'mostly flat'."
    ),
}

# structural proxy for comedic "turns": dialogue-exchange quote marks and
# exclamation points, used as a textual analog to laughs-per-minute when the
# LLM field is unavailable
_TURN_RE = re.compile(r'["“”]|!')

_BEAT_MAP = [
    ("three-or-more", 3), ("three or more", 3), ("3+", 3),
    ("two", 2), ("one", 1),
]


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        try:
            n_sent, mean_wps, _frac_long = ops.sent_stats(t)
        except Exception:
            n_sent, mean_wps = 1, 20.0

        words = re.findall(r"[A-Za-z']+", t)
        n_words = max(1, len(words))

        # code-level structural pacing proxy: dialogue turns / exclamations
        # and short punchy sentences, density per 100 words
        turn_hits = len(_TURN_RE.findall(t))
        punchy = 1 if (n_sent >= 2 and 0 < mean_wps <= 12) else 0
        struct_density = (turn_hits + punchy) / (n_words / 100.0)
        struct_score = 1.0 - math.exp(-1.2 * struct_density)

        extracted = extracted or {}
        beat_raw = str(extracted.get("beat_count", "") or "").strip().lower()
        hit_raw = str(extracted.get("hit_rate", "") or "").strip().lower()

        beat_n = None
        for key, val in _BEAT_MAP:
            if key in beat_raw:
                beat_n = val
                break

        if beat_n is not None:
            llm_density = beat_n / (n_words / 100.0)
            llm_score = 1.0 - math.exp(-1.0 * llm_density)
        else:
            llm_score = struct_score

        if "mostly land" in hit_raw:
            hit_adj = 0.15
        elif "mostly flat" in hit_raw:
            hit_adj = -0.25
        elif "mixed" in hit_raw:
            hit_adj = -0.05
        else:
            hit_adj = 0.0

        base = 0.4 * struct_score + 0.6 * llm_score
        s = base + hit_adj
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
