# Hybrid module for humor unit u10: "Tone and Play-Frame Consistency"
# Construct division: code detects the textual signals that establish and
# hold a "play-frame" (markers that tell the reader "this is a joke": a
# canonical setup/question opener, dialogue quotes, exclamation-driven
# punch delivery) versus signals that a piece has stepped outside play into
# genuine distress/PSA register, plus a lightweight sentiment-swing check for
# gross tonal whiplash. The two LLM fields carry what code cannot classify:
# the piece's overall tone label, and whether the play-frame is explicitly
# broken (the reader is left unsure this was ever "just a joke").

import re

LLM_FIELDS = {
    "tone_label": (
        "In <=6 words, classify the overall tone: lighthearted, sincere/warm, "
        "dark/shocking, or mixed/inconsistent."
    ),
    "frame_break": (
        "In <=10 words: does the piece break its own playful joke-frame (e.g. "
        "genuine distress, real PSA, no signal it was meant as a joke)? "
        "Answer 'yes' or 'no'."
    ),
}

_SETUP_OPENER_RE = re.compile(
    r"^\s*(why (?:did|do|does|is|are)|what (?:did|do)|how (?:many|do)|"
    r"a\s+\w+\s+(?:walks|walked|goes|went)\s+into|knock knock)",
    re.IGNORECASE,
)
_QA_JOKE_RE = re.compile(r"\?\s*\n|\?\.\.\.|\?\s*$", re.MULTILINE)
_PLAYFUL_MARKER_RE = re.compile(r"[!]|lol\b|lmao\b|haha+\b|jk\b", re.IGNORECASE)
_DARK_SERIOUS_MARKER_RE = re.compile(
    r"\b(please (?:call|reach out|seek help)|hotline|if you or someone|"
    r"this is not a joke|in all seriousness|trigger warning)\b",
    re.IGNORECASE,
)
_ROAST_DIALOGUE_RE = re.compile(r'["“][^"”]{2,80}["”]\s*\n?["“][^"”]{2,80}["”]')

_POS_WORDS = {"love", "great", "happy", "funny", "wonderful", "best", "laugh", "smile", "fun"}
_NEG_WORDS = {"die", "dead", "kill", "cry", "sad", "hate", "suicide", "hurt", "pain", "alone"}


def _sentiment_swing(sents):
    scores = []
    for s in sents:
        toks = re.findall(r"[a-z']+", s.lower())
        p = sum(1 for w in toks if w in _POS_WORDS)
        neg = sum(1 for w in toks if w in _NEG_WORDS)
        if p or neg:
            scores.append(p - neg)
    if len(scores) < 2:
        return 0.0
    swings = [abs(scores[i] - scores[i - 1]) for i in range(1, len(scores))]
    return sum(swings) / len(swings)


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]

        # --- code: dominant criterion, play-frame markers (mutual recognition
        # this is "play") ---
        setup_hit = 1.0 if _SETUP_OPENER_RE.search(t.strip()) else 0.0
        qa_hit = 1.0 if _QA_JOKE_RE.search(t) else 0.0
        playful_hits = len(_PLAYFUL_MARKER_RE.findall(t))
        frame_signal = min(1.0, 0.4 * setup_hit + 0.3 * qa_hit + 0.1 * playful_hits)
        frame_signal = max(frame_signal, 0.3)  # a bare joke with no markers isn't automatically zero

        # --- code: serious/PSA register competing with the play frame ---
        serious_hits = len(_DARK_SERIOUS_MARKER_RE.findall(t))
        serious_penalty = min(0.5, 0.25 * serious_hits)

        # --- code: sentiment-swing as a rough tonal-whiplash proxy ---
        swing = _sentiment_swing(sents)
        swing_penalty = min(0.25, swing * 0.12)

        # --- code (secondary, lower weight): structural-format adherence,
        # e.g. one-liner or roast-style alternating dialogue ---
        is_one_liner = 1.0 if (len(sents) <= 2 and len(t) < 220) else 0.0
        is_roast_dialogue = 1.0 if _ROAST_DIALOGUE_RE.search(t) else 0.0
        format_bonus = 0.08 * max(is_one_liner, is_roast_dialogue)

        # --- LLM-field grounding ---
        extracted = extracted or {}
        tone_label = str(extracted.get("tone_label", "") or "").strip().lower()
        frame_break = str(extracted.get("frame_break", "") or "").strip().lower()

        tone_penalty = 0.35 if "mixed" in tone_label or "inconsistent" in tone_label else 0.0
        break_penalty = 0.5 if frame_break.startswith("yes") else 0.0

        base = 0.5 + 0.35 * frame_signal + format_bonus
        out = base - serious_penalty - swing_penalty - tone_penalty - break_penalty
        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5
