"""u13 Credits and Pre-Taped Segments: code detects the textual analog of a broadcast "segment" (bracketed stage directions, script-style dialogue tags, headers, lists) since credits/pre-taped-segment/musical-number craft has no direct plain-text equivalent; LLM fields carry whether such a segment device is present and whether satirical purpose is legible enough to mitigate offense (the competing sub-criterion)."""

import re

LLM_FIELDS = {
    "segment_device": (
        "Does the text use a distinct formatted segment -- a song/verse, "
        "list, mock-script with stage directions, or scene break -- "
        "rather than one plain paragraph? Answer 'segment device' or "
        "'plain paragraph'."
    ),
    "satire_purpose": (
        "If the joke pokes fun at a sensitive topic, is its satirical "
        "point/target clear and purposeful, or unclear/gratuitous? Answer "
        "'clear purpose', 'unclear/gratuitous', or 'not sensitive'."
    ),
}

_STAGE_DIR_RE = re.compile(r"\[[^\]]{1,40}\]|\([^)]{1,10}(?:laughs?|sighs?|pause)[^)]{0,10}\)", re.IGNORECASE)
_SCRIPT_LINE_RE = re.compile(r"(?m)^\s*[A-Z][A-Za-z .'\-]{0,20}:\s")
_HEADER_RE = re.compile(r"(?m)^[A-Z][A-Z ]{3,30}$")
_LIST_RE = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        code_signal = 0.0
        if _STAGE_DIR_RE.search(t):
            code_signal += 0.3
        if len(_SCRIPT_LINE_RE.findall(t)) >= 2:
            code_signal += 0.3
        if _HEADER_RE.search(t):
            code_signal += 0.2
        if len(_LIST_RE.findall(t)) >= 2:
            code_signal += 0.3
        code_signal = min(code_signal, 1.0)

        extracted = extracted or {}
        segdev = str(extracted.get("segment_device", "") or "").strip().lower()
        satire = str(extracted.get("satire_purpose", "") or "").strip().lower()

        llm_signal = 0.3 if "segment device" in segdev else 0.0

        if "clear purpose" in satire:
            satire_adj = 0.15
        elif "unclear" in satire or "gratuitous" in satire:
            satire_adj = -0.25
        else:
            satire_adj = 0.0

        # dominant construct has little-to-no plain-text analog for most
        # short reddit jokes: anchor near neutral, reward whichever signal
        # (code-detected or LLM-detected) found an explicit segment device
        s = 0.5 + 0.3 * max(code_signal, llm_signal) + satire_adj
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
