"""u7 hybrid: code owns the predicate (known-device credit x landing multiplier, corroborated by litotes regex and shaggy-dog sentence-count via ops.sent_stats); LLM fields carry which device is deployed and whether it lands, which regex cannot judge."""

import re

LLM_FIELDS = {
    "device_name": (
        "Name the primary rhetorical/structural comedic device used: irony, "
        "litotes, parody, shaggy-dog, wordplay, or none."
    ),
    "device_landed": (
        "Does this device create the comedic effect it aims for? Answer "
        "yes, partial, no, or NONE."
    ),
}

_NONE_VALUES = {"", "none", "n/a", "na", "unclear", "unknown"}
_KNOWN_DEVICES = {"irony", "litotes", "parody", "shaggy-dog", "shaggy dog", "wordplay"}

_LITOTES_PAT = re.compile(
    r"\bnot\s+(?:un\w+|bad|without|impossible|the\s+worst|nothing)\b|"
    r"\bisn't\s+unlike\b|\bno\s+small\b", re.I)


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
        device = _norm_field(ex.get("device_name", ""))
        landed = _norm_field(ex.get("device_landed", ""))

        if device in _NONE_VALUES:
            base = 0.2
        elif device in _KNOWN_DEVICES:
            base = 0.5
        else:
            base = 0.35  # something named, not a canonical device -- partial credit

        if landed == "yes":
            base += 0.25
        elif landed == "partial":
            base += 0.05
        elif landed == "no":
            base -= 0.25

        # --- code corroboration for the two devices code can partially see ---
        if device == "litotes":
            base += 0.1 if _LITOTES_PAT.search(t) else -0.05
        elif device in ("shaggy-dog", "shaggy dog"):
            try:
                n_sent, _, _ = ops.sent_stats(t)
            except Exception:
                n_sent = 0
            base += 0.1 if n_sent >= 4 else -0.1

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
