"""a186 hybrid: code regex-detects named study-design types and quantitative-threshold patterns in raw text; LLM fields ground whether the paper's OWN design is actually named and whether a threshold is genuinely stated as an evidence standard (regex can't tell a threshold used FROM one merely mentioned)."""
import re

LLM_FIELDS = {
    "design_named": (
        "In <=10 words, the specific study/experiment design type explicitly named "
        "in the text, or NONE."
    ),
    "threshold_stated": (
        "Answer yes or no: does the text state a specific quantitative evidence "
        "threshold or significance standard used to judge results?"
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_DESIGN_PATS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\brandomi[sz]ed controlled trial\b|\bRCT\b", r"\bcohort study\b",
        r"\bcase-control\b", r"\bablation stud\w*\b", r"\bobservational stud\w*\b",
        r"\bretrospective\b", r"\bprospective\b", r"\bcrossover\b",
        r"\bsimulation stud\w*\b", r"\bcase stud\w*\b", r"\bdouble-blind\b",
        r"\bquasi-experimental\b", r"\bbenchmark evaluation\b",
    ]
]
_THRESHOLD_PATS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bp\s*[<>=]\s*0?\.\d+", r"\bsignificance level\b", r"\bconfidence interval\b",
        r"\bCI\b", r"\beffect size\b", r"\bCohen'?s d\b", r"\bthreshold of\b",
        r"\balpha\s*=\s*0?\.\d+|\bα\s*=\s*0?\.\d+",
    ]
]


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        tl = t.lower()

        n_design_code = sum(1 for pat in _DESIGN_PATS if pat.search(t))
        n_thresh_code = sum(1 for pat in _THRESHOLD_PATS if pat.search(t))

        design_ans = extracted.get("design_named")
        design_grounded = (not _is_none(design_ans)) and (str(design_ans).strip().lower()[:12] in tl or n_design_code > 0)

        thresh_ans = str(extracted.get("threshold_stated") or "").strip().lower()

        if n_design_code == 0 and n_thresh_code == 0 and _is_none(design_ans) and not thresh_ans.startswith("y"):
            return 0.15  # no design taxonomy / evidence-threshold content in this excerpt

        # design classification component: LLM+code agreement -> high confidence
        if design_grounded and n_design_code > 0:
            design_component = 1.0
        elif design_grounded or n_design_code > 0:
            design_component = 0.6
        else:
            design_component = 0.15

        # evidence-threshold component
        if thresh_ans.startswith("y") and n_thresh_code > 0:
            thresh_component = 1.0
        elif thresh_ans.startswith("y") or n_thresh_code > 0:
            thresh_component = 0.6
        elif thresh_ans.startswith("n"):
            thresh_component = 0.15
        else:
            thresh_component = 0.3

        s = 0.5 * design_component + 0.5 * thresh_component
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5


if __name__ == "__main__":
    class MockOps:
        def normalize(self, t):
            return t

    mock = MockOps()
    strong_case = "We conduct a randomized controlled trial with pre-registered significance level alpha = 0.05 and report effect sizes (Cohen's d) for all comparisons."
    weak_case = "We ran some experiments and found improvements over the baseline."
    print("strong, no fields:", score(strong_case, {}, mock))
    print("strong, with fields:", score(strong_case, {"design_named": "randomized controlled trial", "threshold_stated": "yes"}, mock))
    print("weak, no fields:", score(weak_case, {}, mock))
    print("empty text:", score("", {}, mock))
