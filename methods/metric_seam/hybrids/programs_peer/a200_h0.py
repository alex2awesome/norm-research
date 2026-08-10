"""a200 hybrid: code regex-detects misconduct/retraction vocabulary in raw text; LLM fields ground whether the text IS actually about a retraction and which SPECIFIC problematic items are named (regex "retract" fires on both a genuine notice and an unrelated mention of "retracted funding")."""
import re

LLM_FIELDS = {
    "retraction_related": (
        "Answer yes or no: does the text describe a retraction, correction, or "
        "misconduct concern about a publication?"
    ),
    "issues_named": (
        "Comma-separated SPECIFIC misconduct/error types named (e.g. fabrication, "
        "plagiarism, duplicate publication, image manipulation), or NONE."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_MISCONDUCT_PATS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bretract\w*\b", r"\bcorrigendum\b", r"\berratum\b", r"\bwithdrawn\b",
        r"\bmisconduct\b", r"\bfabricat\w*\b", r"\bplagiaris\w*\b",
        r"\bduplicate publication\b", r"\bimage manipulation\b",
        r"\bdata manipulation\b", r"\bfalsif\w*\b", r"\bauthorship dispute\b",
    ]
]


def _split_names(val):
    if not isinstance(val, str):
        return []
    v = val.strip()
    if v.lower().strip(". ") in _NONE:
        return []
    return [p.strip() for p in re.split(r"[,;]", v) if p.strip() and p.strip().lower() not in _NONE]


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        tl = t.lower()

        n_kw = sum(1 for pat in _MISCONDUCT_PATS if pat.search(t))
        retraction_ans = str(extracted.get("retraction_related") or "").strip().lower()
        issues = _split_names(extracted.get("issues_named"))
        grounded_issues = [i for i in issues if i.lower() in tl]

        if n_kw == 0 and not retraction_ans.startswith("y") and not grounded_issues:
            return 0.10  # not a retraction/misconduct context -- aspect not engaged

        if retraction_ans.startswith("y"):
            relatedness = 1.0
        elif n_kw > 0:
            relatedness = 0.6
        else:
            relatedness = 0.3

        specificity = min(1.0, len(grounded_issues) / 2.0)
        kw_density = min(1.0, n_kw / 3.0)

        s = 0.35 * relatedness + 0.45 * specificity + 0.20 * kw_density
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5


if __name__ == "__main__":
    class MockOps:
        def normalize(self, t):
            return t

    mock = MockOps()
    specific_case = "This article has been retracted due to data fabrication in Figure 2 and undisclosed image manipulation identified during post-publication review."
    vague_case = "This article has been retracted. Concerns were raised about the results."
    unrelated_case = "We propose a new optimization method for deep neural networks."
    print("specific, no fields:", score(specific_case, {}, mock))
    print("specific, with fields:", score(specific_case, {"retraction_related": "yes", "issues_named": "data fabrication, image manipulation"}, mock))
    print("vague, with fields:", score(vague_case, {"retraction_related": "yes", "issues_named": "NONE"}, mock))
    print("unrelated:", score(unrelated_case, {}, mock))
    print("empty text:", score("", {}, mock))
