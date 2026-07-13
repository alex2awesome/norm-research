import re
from collections import Counter

TERMS = ["\\btest", "unit", "integration", "assert", "mock", "fixture", "coverage"]

def score(text: str) -> float:
    """v1 structure: density of topic-relevant terms for a104."""
    try:
        t = (text or "").lower()
        if not t.strip():
            return 0.5
        # Split into review chunks separated by '---'
        chunks = [c.strip() for c in re.split(r"-{3,}", t) if c.strip()]
        n_chunks = max(1, len(chunks))
        # total topic term hits
        hits = 0
        for term in TERMS:
            hits += len(re.findall(term, t))
        # words
        words = re.findall(r"\w+", t)
        wc = max(1, len(words))
        density = hits / wc * 1000.0  # hits per 1000 words
        # chunks that mention topic
        chunk_hits = sum(1 for c in chunks if any(re.search(term, c) for term in TERMS))
        coverage = chunk_hits / n_chunks
        # blend density (saturating) + coverage
        d_norm = density / (density + 5.0)
        score_v = 0.5 * d_norm + 0.5 * coverage
        return max(0.0, min(1.0, score_v))
    except Exception:
        return 0.5
