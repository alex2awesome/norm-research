"""u11 Punchline Placement & Wordplay: code owns the placement PREDICATE, locating where the LLM-identified wordplay lands in the text and checking for a setup/punchline split; LLM fields carry the pun word itself and the setup's misdirection, which code cannot identify."""

import re

LLM_FIELDS = {
    "pun_word": (
        "Copy the single pun, malapropism, or wordplay word/phrase in "
        "this joke, in <=6 words, or say 'none'."
    ),
    "misdirection": (
        "In <=12 words, describe the false assumption the setup plants "
        "that the punchline reveals or subverts, or say 'none'."
    ),
}

_WORD_RE = re.compile(r"[A-Za-z']+")
_NONE_VALUES = {"", "none", "n/a", "na"}


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        extracted = extracted if isinstance(extracted, dict) else {}

        try:
            norm = ops.normalize(text)
        except Exception:
            norm = text

        pun = str(extracted.get("pun_word", "") or "").strip().lower()
        has_pun = bool(pun) and pun not in _NONE_VALUES
        misdirection = str(extracted.get("misdirection", "") or "").strip().lower()
        has_misdirection = bool(misdirection) and misdirection not in _NONE_VALUES

        # --- primary construct: code computes WHERE the LLM-identified pun
        # lands (precision of placement), since that positional predicate is
        # exactly what code can determine once the field grounds the pun
        # word for it. Precision peaks in the closing ~25% of the piece. ---
        words = _WORD_RE.findall(norm.lower())
        n = len(words)
        if has_pun and n > 0:
            pun_tokens = [w for w in _WORD_RE.findall(pun) if len(w) >= 3]
            positions = [i for i, w in enumerate(words) if w in pun_tokens]
            if positions:
                rel_pos = max(positions) / n
                base = 0.35 + 0.55 * max(0.0, 1.0 - max(0.0, 0.75 - rel_pos) * 1.6)
                base = min(0.9, base)
            else:
                base = 0.55  # LLM found a pun paraphrased, not literal in text
        elif has_pun:
            base = 0.55
        else:
            base = 0.3  # no wordplay identified at all

        # --- code-only structural corroboration: a joke needs a
        # distinguishable setup/punchline split for "placement" to mean
        # anything in the first place. ---
        sentences = [s for s in re.split(r"[.!?]+", norm) if s.strip()]
        struct = 0.1 if len(sentences) >= 2 else 0.0

        # --- secondary competing criterion: setup plants a misdirection
        # the punchline reveals (thick semantic read, LLM-grounded). ---
        misdirect_bonus = 0.15 if has_misdirection else 0.0

        s = base + struct + misdirect_bonus
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
