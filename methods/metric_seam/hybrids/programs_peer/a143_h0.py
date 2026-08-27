"""a143 hybrid: code cross-checks arXiv-vs-venue citation patterns in raw text; LLM fields ground which prior-work refs are named and whether a venue is actually stated (paraphrase presence code can't detect)."""
import re

LLM_FIELDS = {
    "prior_work_refs": (
        "Comma-separated specific prior works/papers referenced by name or citation "
        "in the text, or NONE if none."
    ),
    "venue_or_arxiv": (
        "For the prior work referenced, state whether venue/journal is given (yes), "
        "only arXiv/preprint (arxiv), or no detail (no)."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_ARXIV_PAT = re.compile(r"\barxiv\b|\b\d{4}\.\d{4,5}\b", re.IGNORECASE)
_VENUE_PATS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(NeurIPS|NIPS|ICML|ICLR|ACL|EMNLP|NAACL|CVPR|ECCV|ICCV|AAAI|IJCAI|"
        r"KDD|SIGIR|WWW|TACL|JMLR|COLING|EACL)\b",
        r"\bNature\b|\bScience\b|\bPNAS\b|\bCell\b|\bLancet\b",
        r"\bJournal of\b|\bProceedings of\b|\bConference on\b|\bTransactions on\b",
        r"\bvol\.\s*\d+|\bpp\.\s*\d+",
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

        n_arxiv_code = len(_ARXIV_PAT.findall(t))
        n_venue_code = sum(1 for pat in _VENUE_PATS if pat.search(t))

        refs = _split_names(extracted.get("prior_work_refs"))
        grounded_refs = [r for r in refs if r.lower() in tl]

        venue_ans = str(extracted.get("venue_or_arxiv") or "").strip().lower()

        if not grounded_refs and n_arxiv_code == 0 and n_venue_code == 0:
            # no prior-work provenance discussion at all in this excerpt -- aspect
            # not engaged, can't credit or penalize sourcing practice
            return 0.15

        total_code = n_arxiv_code + n_venue_code
        code_component = (n_venue_code / total_code) if total_code else 0.5

        if venue_ans.startswith("yes"):
            llm_component = 0.9
        elif venue_ans.startswith("arxiv"):
            llm_component = 0.35
        elif venue_ans.startswith("no"):
            llm_component = 0.15
        else:
            llm_component = 0.5

        specificity = min(1.0, len(grounded_refs) / 2.0)

        s = 0.35 * code_component + 0.40 * llm_component + 0.25 * specificity
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5


if __name__ == "__main__":
    class MockOps:
        def normalize(self, t):
            return t

    mock = MockOps()
    venue_case = "We build on Smith et al. (2022, NeurIPS) and Lee et al. (2021, ACL)."
    arxiv_case = "We build on Smith et al. (arXiv:2201.01234) and Lee et al. (arXiv preprint)."
    no_case = "We propose a new method for image classification using deep learning."
    print("venue, no fields:", score(venue_case, {}, mock))
    print("venue, with fields:", score(venue_case, {"prior_work_refs": "Smith et al., Lee et al.", "venue_or_arxiv": "yes"}, mock))
    print("arxiv, with fields:", score(arxiv_case, {"prior_work_refs": "Smith et al., Lee et al.", "venue_or_arxiv": "arxiv"}, mock))
    print("no provenance:", score(no_case, {}, mock))
    print("empty text:", score("", {}, mock))
