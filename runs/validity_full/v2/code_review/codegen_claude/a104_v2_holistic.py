import re
from math import tanh

# (regex, weight) signals; weights need not sum to 1 — we normalize.
SIGNALS = [["unit test", 0.9], ["integration test", 0.8], ["\\btest case", 0.7], ["\\bassert", 0.6], ["coverage", 0.6], ["regression test", 0.8], ["add(ing)? (a )?test", 0.9]]

def score(text: str) -> float:
    """v2 holistic for a104: blend multiple weak signals."""
    try:
        t = (text or "")
        if not t.strip():
            return 0.5
        tl = t.lower()
        wc = max(1, len(re.findall(r"\w+", tl)))
        total_w = sum(abs(w) for _, w in SIGNALS) or 1.0
        weighted = 0.0
        any_hit = False
        for pat, w in SIGNALS:
            hits = len(re.findall(pat, tl))
            if hits:
                any_hit = True
            # saturating contribution
            contrib = tanh(hits / 2.0)
            weighted += w * contrib
        base = weighted / total_w  # in roughly [-1, 1]
        # also density of code/identifier fragments suggests concrete review
        code_frags = len(re.findall(r"`[^`]+`", t)) + len(re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", t))
        cf_norm = tanh(code_frags / 12.0)
        # combine
        if not any_hit:
            # no aspect signal -> neutral
            return 0.5
        # map base from [-1,1] to [0,1]
        mapped = (base + 1.0) / 2.0
        out = 0.7 * mapped + 0.3 * cf_norm
        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5
