"""
Hybrid channel for aspect a126: "Cringe/awkwardness calibrated"
Criterion: use social discomfort as the engine while calibrating sensitivity
to avoid audience alienation.

Design rationale (from the pack, not the 30 examples):
The baseline keyword scorer (matches on 'awkward', 'silence', 'stared', ...)
is a weak proxy: literal cringe-vocabulary can appear in jokes whose actual
comedic ENGINE is wordplay/absurdity/slapstick (discomfort merely decorates
the scene), and such jokes should NOT score high. Conversely, jokes whose
punchline genuinely hinges on an embarrassing exposure, taboo confession, or
dark/awkward tension -- resolved via a reveal -- should score high, even when
they contain no literal "cringe" vocabulary at all. That distinction (is
discomfort the MECHANISM vs merely present in the setting) is a semantic
judgment call code cannot make reliably, so it is delegated to an LLM field.
A second LLM field captures the "calibrated ... avoid alienation" half of the
criterion: whether the discomfort reads as relatable/earned or as gratuitous
shock content that would alienate an audience. Code supplies only structural
corroboration (dialogue/exchange density, buildup-then-short-punch shape) and
a light corpus-typicality guard via the evidence op.
"""

import re

LLM_FIELDS = {
    "discomfort_engine": (
        "Does the punchline's humor come mainly FROM embarrassment, taboo, "
        "or awkward social tension, or from wordplay/logic/absurdity instead? "
        "Answer discomfort or wordplay."
    ),
    "alienation_risk": (
        "Is the awkward/taboo content relatable and tasteful, or gratuitously "
        "graphic/offensive? Answer tasteful, gratuitous, or none."
    ),
}


def _classify_engine(raw):
    if not raw:
        return None
    s = raw.lower()
    neg_kw = ("wordplay", "pun", "logic", "absurd", "gotcha", "clever", "none", "n/a")
    pos_kw = ("discomfort", "cringe", "embarrass", "awkward", "taboo", "dark",
              "uncomfortable", "shame", "humiliat")
    for kw in neg_kw:
        if kw in s:
            return False
    for kw in pos_kw:
        if kw in s:
            return True
    return None


def _classify_calibration(raw):
    if not raw:
        return 0.0
    s = raw.lower()
    if any(k in s for k in ("gratuitous", "graphic", "extreme", "offensive",
                             "shock", "crude", "alienat", "disturb")):
        return -0.15
    if any(k in s for k in ("tasteful", "mild", "relatable", "earned",
                             "calibrated", "appropriate")):
        return 0.05
    return 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5

        extracted = extracted if isinstance(extracted, dict) else {}

        try:
            norm = ops.normalize(text)
        except Exception:
            norm = text

        # --- LLM-grounded semantic core -------------------------------
        engine_raw = extracted.get("discomfort_engine", "") or ""
        calib_raw = extracted.get("alienation_risk", "") or ""

        is_discomfort = _classify_engine(engine_raw)
        if is_discomfort is None:
            # Conservative default: without positive evidence that discomfort
            # is the driving mechanism, treat as a non-cringe (wordplay/other)
            # joke -- matches the corpus base rate of plain-pun jokes.
            is_discomfort = False

        base = 0.82 if is_discomfort else 0.16
        calib_adj = _classify_calibration(calib_raw)

        # --- Structural corroboration (code-only) ----------------------
        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(norm)
        except Exception:
            n_sent, mean_wps, frac_long = 1, 10.0, 0.0

        try:
            quotes = re.findall(r'"([^"]{2,200})"', norm)
            dialogue_turns = len(quotes)
        except Exception:
            dialogue_turns = 0

        try:
            sentences = [s for s in re.split(r"[.!?]+", norm) if s.strip()]
            last_words = len(sentences[-1].split()) if sentences else 0
        except Exception:
            sentences, last_words = [], 0

        struct_bonus = 0.0
        struct_bonus += min(0.08, 0.02 * dialogue_turns)
        if mean_wps and last_words and last_words < 0.7 * mean_wps:
            struct_bonus += 0.05  # short punch relative to buildup
        if n_sent >= 4:
            struct_bonus += 0.03  # real narrative buildup before the reveal
        struct_bonus = min(0.15, struct_bonus)

        raw = base + struct_bonus + calib_adj
        raw = max(0.0, min(1.0, raw))

        # --- Evidence-op guard against degenerate/atypical documents ----
        try:
            sims = ops.retrieve_similar(norm, k=5)
            if sims:
                vals = [s[0] for s in sims if isinstance(s, (list, tuple)) and len(s) > 0]
                avg_sim = sum(vals) / len(vals) if vals else None
            else:
                avg_sim = None
        except Exception:
            avg_sim = None

        if avg_sim is not None and avg_sim < 0.02:
            # Near-zero similarity to anything in the corpus suggests this
            # may be boilerplate/chrome rather than a real joke; pull the
            # estimate toward neutral uncertainty rather than trusting the
            # extreme.
            raw = raw + (0.5 - raw) * 0.3

        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
