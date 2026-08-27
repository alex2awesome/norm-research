"""a172 hybrid: code detects model-documentation keyword register plus sentence-level conciseness via ops.sent_stats; LLM fields ground whether a design motivation and an explicit tradeoff are actually stated (regex sees the word "tradeoff", not whether one is genuinely argued)."""
import re

LLM_FIELDS = {
    "motivation": (
        "In <=15 words, the stated motivation/reason for the model's specific design "
        "choice, or NONE if absent."
    ),
    "tradeoff_stated": (
        "Answer yes or no: does the text explicitly discuss a design tradeoff "
        "(something sacrificed for a gain)?"
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_MODEL_DOC_PATS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bmodel card\b", r"\bdatasheet\b", r"\bintended use\b",
        r"\btrade-?off\w*\b", r"\bdesign choice\w*\b",
        r"\bwe (?:design|choose|opt|select)\w*\b", r"\barchitectur\w*\b",
        r"\bmotivat\w*\b",
    ]
]


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}

        n_kw = sum(1 for pat in _MODEL_DOC_PATS if pat.search(t))

        motivation = extracted.get("motivation")
        has_motivation = not _is_none(motivation)

        tradeoff_ans = str(extracted.get("tradeoff_stated") or "").strip().lower()

        if n_kw == 0 and not has_motivation and not tradeoff_ans.startswith("y"):
            return 0.15  # no model-documentation content in this excerpt

        mot_score = 1.0 if has_motivation else 0.2
        if tradeoff_ans.startswith("y"):
            trade_score = 1.0
        elif tradeoff_ans.startswith("n"):
            trade_score = 0.3
        else:
            trade_score = 0.5

        kw_score = min(1.0, n_kw / 3.0)

        try:
            n_sent, mean_wps, frac_long = ops.sent_stats(t)
        except Exception:
            mean_wps = 20.0
        # brevity/scannability: penalize long, run-on sentences (mean words/sentence);
        # 15 words/sentence or fewer -> full credit, 40+ -> no credit
        conciseness = max(0.0, min(1.0, (40.0 - mean_wps) / 25.0))

        s = 0.35 * mot_score + 0.30 * trade_score + 0.20 * kw_score + 0.15 * conciseness
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5


if __name__ == "__main__":
    class MockOps:
        def normalize(self, t):
            return t

        def sent_stats(self, t):
            sents = [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
            n = len(sents) or 1
            words = t.split()
            return (n, len(words) / n, 0.1)

    mock = MockOps()
    doc_case = "We chose a shallow architecture to reduce latency. This is a tradeoff: we sacrifice some accuracy for a 3x speedup, which matters for on-device deployment."
    plain_case = "Our model achieves state-of-the-art results on the benchmark."
    print("doc, no fields:", score(doc_case, {}, mock))
    print("doc, with fields:", score(doc_case, {"motivation": "reduce latency for on-device deployment", "tradeoff_stated": "yes"}, mock))
    print("plain, no fields:", score(plain_case, {}, mock))
    print("empty text:", score("", {}, mock))
