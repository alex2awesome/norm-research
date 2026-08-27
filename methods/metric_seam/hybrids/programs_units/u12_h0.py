"""u12 hybrid: code measures build/release structure directly (ops.sent_stats sentence count, pause punctuation, final-sentence brevity relative to setup); LLM fields carry the semantic distinct-punchline judgment and the secondary image-dependence check code cannot see."""

import re

LLM_FIELDS = {
    "punchline_distinct": (
        "Does the final line/clause deliver a clear comedic turn distinct "
        "from the setup? Answer yes, no, or NONE."
    ),
    "visual_reference": (
        "Does the text reference or depend on an accompanying image, "
        "caption, or visual element? Answer yes, no, or NONE."
    ),
}

_NONE_VALUES = {"", "none", "n/a", "na", "unclear", "unknown"}
_PAUSE_PAT = re.compile(r"\.\.\.|…|—|--")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _norm_field(v):
    return (v or "").strip().lower().strip(". ")


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        ex = extracted or {}
        distinct = _norm_field(ex.get("punchline_distinct", ""))
        visual = _norm_field(ex.get("visual_reference", ""))

        sents = [s for s in _SENT_SPLIT.split(t) if s.strip()]
        n = len(sents)
        pause_hit = bool(_PAUSE_PAT.search(t))

        base = 0.3
        if n >= 2:
            base += 0.15  # setup/punchline are structurally distinguishable
        if pause_hit:
            base += 0.1  # explicit pause/beat before a release

        if n >= 2:
            word_counts = [len(re.findall(r"[A-Za-z']+", s)) for s in sents]
            prior_mean = sum(word_counts[:-1]) / max(1, len(word_counts) - 1)
            if word_counts[-1] < prior_mean:
                base += 0.1  # snappy release shorter than the build

        if distinct == "yes":
            base += 0.2
        elif distinct == "no":
            base -= 0.15

        base = max(0.0, min(0.9, base))  # leave room for the secondary bump

        # --- secondary (small weight): image-text integration, only when present ---
        if visual == "yes":
            base += 0.1

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
