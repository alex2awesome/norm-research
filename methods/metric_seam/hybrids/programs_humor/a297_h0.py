"""
Hybrid metric channel for aspect a297: "Timing and delivery (micro-timing
and paralinguistics)".

Criterion: precision and intentional anti-timing in performance/on-page --
pauses, tempo, linger/clip choices, voice/pitch shifts, and diegetic
interrupts that shape tension/release and land laughs.

Design idea
-----------
For text, the performance analog of "timing" is structural: does the
punchline arrive as a short, clipped beat after a longer setup (economy /
clip), are there explicit paralinguistic pause markers (ellipses, em-dashes,
"pause"/"beat"/"click"/"hesitat-" style words), is there tempo contrast
across beats (short-long-short cadence rather than one monotone register),
and are there vocal/pitch-shift proxies (caps-emphasis, exclamations)?
Rambling, single-register prose with long undifferentiated sentences and no
clip at the end is the textual analog of flat/undifferentiated delivery.

Two constructs that code cannot reliably reach are handed to an LLM
extractor: (1) a verbatim pause/interrupt/tonal cue right before the punch
(grounded back into the text in code -- we don't just trust the LLM), and
(2) a short named "delivery device" (e.g. deadpan reveal, rapid interrupt)
that lets us credit deadpan/anti-timing deliveries that carry no textual
pause marker at all (e.g. a flat, matter-of-fact reveal of something
absurd), which pure regex structure cannot detect.
"""

import re
import statistics

LLM_FIELDS = {
    "pause_cue": (
        "Quote the shortest exact phrase (<=8 words) that marks a deliberate "
        "pause, trailing-off, interruption, or diegetic sound/tonal shift "
        "right before the punchline lands; answer NONE if no such cue is "
        "present."
    ),
    "delivery_device": (
        "In <=5 words, name the timing/delivery device driving the piece "
        "(e.g. 'deadpan reveal', 'rapid interrupt', 'long buildup then "
        "clip', 'monotone list'); answer NONE if none is evident."
    ),
}

_PAUSE_WORD_RE = re.compile(
    r"\b(pause[sd]?|silen(?:ce|t)|beat|wait(?:ed|ing)?|click(?:ed)?|"
    r"trail(?:ed|ing)?\s+off|hesitat\w*|stammer\w*|stutter\w*)\b",
    re.IGNORECASE,
)
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")
_DASH_RE = re.compile(r"—|--")
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")
_WORD_RE = re.compile(r"[A-Za-z']+")

_DEVICE_WEIGHTS = (
    ("deadpan", 1.0),
    ("interrupt", 0.9),
    ("clip", 0.9),
    ("pause", 0.8),
    ("escalat", 0.7),
    ("build then", 0.75),
    ("buildup", 0.75),
    ("tonal", 0.6),
    ("timing", 0.6),
    ("shift", 0.55),
)


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _word_count(s):
    return len(_WORD_RE.findall(s))


def _split_beats(text):
    """Flatten paragraph breaks, sentence enders, and em-dash pauses into a
    sequence of "delivery beats" -- the textual analog of a spoken unit."""
    raw = re.split(r"\n\s*\n|(?<=[.!?…])\s+|—+", text)
    beats = [b.strip(" \t\"'“”‘’") for b in raw]
    return [b for b in beats if re.search(r"\w", b)]


def _fallback_sent_stats(text):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sents:
        return 0, 0.0, 0.0
    lens = [len(s.split()) for s in sents]
    n_sent = len(sents)
    mean_wps = sum(lens) / n_sent
    total_words = sum(lens)
    long_words = sum(1 for s in sents for w in s.split() if len(w) >= 8)
    frac_long = (long_words / total_words) if total_words else 0.0
    return n_sent, mean_wps, frac_long


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not isinstance(text, str) or not text.strip():
            return 0.0

        try:
            norm = ops.normalize(text)
            if not isinstance(norm, str) or not norm.strip():
                norm = text
        except Exception:
            norm = text

        # --- sentence stats: prefer ops, fall back to our own computation ---
        n_sent, mean_wps, frac_long = _fallback_sent_stats(norm)
        try:
            raw_stats = ops.sent_stats(norm)
            if isinstance(raw_stats, dict):
                n_sent = raw_stats.get("n_sent", n_sent)
                mean_wps = raw_stats.get("mean_words_per_sent", mean_wps)
                frac_long = raw_stats.get("frac_long_words", frac_long)
            elif isinstance(raw_stats, (list, tuple)) and len(raw_stats) >= 3:
                n_sent, mean_wps, frac_long = (
                    raw_stats[0], raw_stats[1], raw_stats[2]
                )
        except Exception:
            pass

        beats = _split_beats(norm)

        # --- S1: punchline clip -- final beat short relative to the setup ---
        if len(beats) < 2:
            s1 = 0.0
        else:
            last_len = _word_count(beats[-1])
            prior_lens = [_word_count(b) for b in beats[:-1] if _word_count(b) > 0]
            if not prior_lens or last_len == 0:
                s1 = 0.0
            else:
                mean_prior = statistics.mean(prior_lens)
                ratio = (last_len / mean_prior) if mean_prior else 1.0
                s1 = _clamp(1.0 - ratio)

        # --- S2: explicit pause / paralinguistic markers ---
        pause_hits = (
            len(_ELLIPSIS_RE.findall(norm))
            + len(_DASH_RE.findall(norm))
            + len(_PAUSE_WORD_RE.findall(norm))
        )
        s2 = _clamp(pause_hits / 3.0)

        # --- S3: tempo contrast across beats (short-long-short cadence) ---
        beat_lens = [_word_count(b) for b in beats if _word_count(b) > 0]
        if len(beat_lens) >= 2 and statistics.mean(beat_lens) > 0:
            cv = statistics.pstdev(beat_lens) / statistics.mean(beat_lens)
            s3 = _clamp(cv / 1.5)
        else:
            s3 = 0.0

        # --- S4: vocal/pitch-shift proxies (caps emphasis, exclamations) ---
        caps_hits = len(_CAPS_WORD_RE.findall(norm))
        excl_hits = norm.count("!")
        s4 = _clamp((caps_hits + 0.5 * excl_hits) / 3.0)

        # --- L1: LLM pause cue, grounded back into the text (not trusted blindly) ---
        pause_cue = ((extracted or {}).get("pause_cue") or "").strip()
        if pause_cue and pause_cue.upper() != "NONE":
            cue_norm = re.sub(r"[^a-z0-9 ]", "", pause_cue.lower()).strip()
            text_norm = re.sub(r"[^a-z0-9 ]", "", norm.lower())
            l1 = 1.0 if (cue_norm and cue_norm in text_norm) else 0.3
        else:
            l1 = 0.0

        # --- L2: LLM-named delivery device (catches deadpan/anti-timing
        # deliveries with no textual pause marker at all) ---
        device = ((extracted or {}).get("delivery_device") or "").strip().lower()
        l2 = 0.0
        if device and device != "none":
            for kw, w in _DEVICE_WEIGHTS:
                if kw in device:
                    l2 = max(l2, w)

        raw = (
            0.28 * s1
            + 0.16 * s2
            + 0.12 * s3
            + 0.12 * s4
            + 0.16 * l1
            + 0.16 * l2
        )

        # Rambling penalty: long, undifferentiated sentences (no clip/pause
        # at all) are the textual analog of flat, un-timed delivery.
        if mean_wps and mean_wps > 30:
            raw -= min(0.2, (mean_wps - 30) / 100.0)

        return _clamp(raw)
    except Exception:
        return 0.5
