"""u6 Comedic Timing and Pacing: code measures sentence-length rhythm variance, a crisp-final-beat check, and explicit pause markers as structural proxies; LLM fields carry the pacing read and whether a deliberate beat precedes the punchline."""

import re
import statistics

LLM_FIELDS = {
    "pacing_read": (
        "Does this joke's pacing feel well-timed, rushed, or dragged? "
        "Answer: well-timed, rushed, dragged, or unclear."
    ),
    "pause_before_punch": (
        "Is there a deliberate pause or beat (ellipsis, line break, "
        "dramatic delay) right before the final punchline? Answer: yes or no."
    ),
}

_PAUSE_MARK_RE = re.compile(r"\.\.\.|…|--|—|\n\s*\n")


def _classify_pacing(raw):
    if not raw:
        return None
    s = raw.lower()
    if "well" in s and "timed" in s:
        return "well-timed"
    if "rushed" in s:
        return "rushed"
    if "dragged" in s or "drag" in s:
        return "dragged"
    return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        extracted = extracted if isinstance(extracted, dict) else {}

        try:
            norm = ops.normalize(text)
        except Exception:
            norm = text

        # --- primary construct: LLM-grounded pacing read; code owns the
        # mapping to a score. ---
        pacing = _classify_pacing(str(extracted.get("pacing_read", "") or ""))
        if pacing == "well-timed":
            base = 0.78
        elif pacing in ("rushed", "dragged"):
            base = 0.28
        else:
            base = 0.5

        # --- code-only structural corroboration: alternating short/long
        # sentence rhythm (a build-then-punch shape) reads as well-paced;
        # a shorter, punchier final beat reads as good delivery timing. ---
        sentences = [s for s in re.split(r"[.!?]+", norm) if s.strip()]
        counts = [len(s.split()) for s in sentences]
        struct = 0.0
        if len(counts) >= 3:
            mean_c = statistics.mean(counts)
            if mean_c > 0:
                cv = statistics.pstdev(counts) / mean_c
                struct += max(0.0, 0.1 - abs(cv - 0.5) * 0.15)
        if len(counts) >= 2:
            other_mean = sum(counts[:-1]) / max(1, len(counts[:-1]))
            if other_mean > 0 and counts[-1] < 0.75 * other_mean:
                struct += 0.06

        # --- code-only structural corroboration: explicit pause markers ---
        pause_hits = len(_PAUSE_MARK_RE.findall(norm))
        struct += min(0.06, 0.03 * pause_hits)
        struct = min(0.2, struct)

        # --- secondary: LLM-grounded deliberate pause right before the
        # punchline (delivery matched to structure). ---
        pause_field = str(extracted.get("pause_before_punch", "") or "").strip().lower()
        pause_bonus = 0.08 if pause_field.startswith("y") else 0.0

        s = base + struct + pause_bonus
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
