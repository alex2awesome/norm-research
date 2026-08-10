"""u28 Truth-Revealing Satire: code detects concrete proper-noun targets (a proxy for a "clear and well-chosen satirical target") via mid-sentence capitalization; LLM fields carry the semantic judgment of what the satirical target actually is and whether the joke surfaces a genuine uncomfortable truth about it."""

import re

LLM_FIELDS = {
    "satirical_target": (
        "In <=8 words, name the specific real-world target (person, "
        "institution, group, or social norm) this joke satirizes, or say "
        "'none'."
    ),
    "reveals_truth": (
        "Does the joke surface a genuine, uncomfortable truth about its "
        "target, or is it just absurdity/wordplay with no real-world "
        "point? Answer 'reveals truth' or 'no real point'."
    ),
}

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_COMMON_CAP_RE = re.compile(r"^(I|I'm|I'll|I've|I'd)$")


def _midsentence_propn_hits(t):
    hits = 0
    for sent in _SENT_SPLIT_RE.split(t):
        words = sent.strip().split()
        for w in words[1:]:
            core = re.sub(r"[^A-Za-z']", "", w)
            if len(core) >= 2 and core[0].isupper() and not core.isupper() and not _COMMON_CAP_RE.match(core):
                hits += 1
    return hits


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        propn_hits = _midsentence_propn_hits(t)

        extracted = extracted or {}
        target = str(extracted.get("satirical_target", "") or "").strip().lower()
        truth = str(extracted.get("reveals_truth", "") or "").strip().lower()

        s = 0.5
        if "reveals truth" in truth:
            s += 0.35
        elif "no real point" in truth:
            s -= 0.35

        if target and target != "none":
            s += 0.15
        if propn_hits >= 1:
            s += 0.1

        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
