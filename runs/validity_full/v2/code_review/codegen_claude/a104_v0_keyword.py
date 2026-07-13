import re

KW = ["\\btest(s|ing|case|suite)?\\b", "\\bunit test\\b", "\\bintegration test\\b", "\\bassert", "\\bmock\\b", "\\bfixture\\b", "\\bcoverage\\b", "\\bregression test\\b"]
NEG = []

def score(text: str) -> float:
    """v0 keyword detector for a104."""
    try:
        t = (text or "").lower()
        if not t:
            return 0.5
        hits = sum(1 for k in KW if re.search(k, t))
        neg = sum(1 for k in NEG if re.search(k, t))
        if hits == 0 and neg == 0:
            return 0.5
        denom = max(1, hits + neg)
        raw = hits / denom
        # saturating; more hits -> higher confidence the aspect is being addressed positively
        boost = min(0.25, 0.05 * hits)
        return max(0.0, min(1.0, raw + boost))
    except Exception:
        return 0.5
