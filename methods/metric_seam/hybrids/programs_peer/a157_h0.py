"""a157 hybrid: code detects community/collaboration keyword register in raw text; LLM fields ground named collaborators and whether they show genuine involvement vs mere acknowledgment (a distinction keyword regex conflates)."""
import re

LLM_FIELDS = {
    "collaborators_named": (
        "Comma-separated external collaborators, organizations, or community/patient "
        "groups named as contributing, or NONE."
    ),
    "involvement_type": (
        "One word: genuine collaborator INVOLVEMENT in the work (involvement), mere "
        "CONTEXT/acknowledgment (context), or NEITHER (none)."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_COMMUNITY_PATS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcommunity\b", r"\bcollaborat\w*", r"\bcrowdsourc\w*",
        r"\bpatient advisory\b", r"\bpublic involvement\b", r"\bstakeholder\w*",
        r"\bco-design\w*", r"\bpartnership\b", r"\bvolunteer\w*",
        r"\bcitizen scien\w*", r"\bparticipatory\b",
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

        n_kw = sum(1 for pat in _COMMUNITY_PATS if pat.search(t))
        refs = _split_names(extracted.get("collaborators_named"))
        grounded = [r for r in refs if r.lower() in tl]

        involvement = str(extracted.get("involvement_type") or "").strip().lower()

        if n_kw == 0 and not grounded and involvement in ("", "none"):
            return 0.10  # no community/collaboration content in this excerpt

        if involvement.startswith("involv"):
            base = 0.9
        elif involvement.startswith("context"):
            base = 0.45
        elif involvement.startswith("none"):
            base = 0.15
        else:
            base = 0.4  # unanswered -- fall back to code-only evidence

        specificity = min(1.0, len(grounded) / 2.0)
        kw_density = min(1.0, n_kw / 3.0)

        s = 0.55 * base + 0.25 * specificity + 0.20 * kw_density
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5


if __name__ == "__main__":
    class MockOps:
        def normalize(self, t):
            return t

    mock = MockOps()
    involved_case = "We co-designed the study protocol with the Patient Advisory Board and three community organizations who shaped the research questions."
    context_case = "We thank the broader research community for prior work in this area."
    none_case = "We propose a new transformer architecture for sequence modeling."
    print("involved, no fields:", score(involved_case, {}, mock))
    print("involved, with fields:", score(involved_case, {"collaborators_named": "Patient Advisory Board", "involvement_type": "involvement"}, mock))
    print("context, with fields:", score(context_case, {"collaborators_named": "NONE", "involvement_type": "context"}, mock))
    print("none:", score(none_case, {}, mock))
    print("empty text:", score("", {}, mock))
