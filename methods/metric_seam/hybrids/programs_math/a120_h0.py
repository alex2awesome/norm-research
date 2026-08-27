"""
Hybrid metric channel for a120: "Multiple representations and translation"
(switching among algebraic / geometric / analytic / combinatorial / algorithmic
/ verbal / probabilistic / logical viewpoints on the SAME result, to deepen insight).

Design rationale (see pack a120.json known_failure_pattern): raw keyword presence
("geometric", "perspective", "alternatively", ...) is only a weak proxy for this
criterion (baseline_train_rho = 0.088) because keyword hits fire on off-topic or
single-register text (e.g. a LaTeX-software recommendation full of \\begin{align})
just as easily as on genuine cross-representation translation. Detecting whether
an answer *actually* re-expresses one result in a second register is a tacit,
semantic judgment a regex/parser cannot make reliably -- so we delegate that core
judgment to a single LLM_FIELDS extractor (constrained to a short, closed-category
answer so it stays cheaply parseable in code), and keep the predicate itself,
plus two purely structural corroborating signals, in code:
  - explicit case/branch structure (proof_skeleton 'case' markers) -- answers that
    walk through >=2 distinct sub-cases/approaches for the same question tend to
    be doing representation-switching work even when the LLM field is noisy.
  - a TIGHT list of explicit bridging connectives ("equivalently", "corresponds
    to", "same as", "can be viewed as", ...) -- narrower than the baseline's broad
    keyword list, so it does not just re-fire the known failure mode.
"""

import re

LLM_FIELDS = {
    "representations": (
        "From {algebraic, geometric, analytic, combinatorial, algorithmic, "
        "verbal, probabilistic, logical, numeric}, list ALL registers this "
        "answer actually uses to explain the SAME result (comma-separated); "
        "say NONE if only one register is used."
    ),
}

_CANON_PREFIXES = [
    ("algeb", "algebraic"),
    ("geom", "geometric"),
    ("visual", "geometric"),
    ("graph", "geometric"),
    ("analy", "analytic"),
    ("combinat", "combinatorial"),
    ("algorithm", "algorithmic"),
    ("comput", "algorithmic"),
    ("verbal", "verbal"),
    ("probab", "probabilistic"),
    ("logic", "logical"),
    ("numer", "numeric"),
]

_CONNECTIVE_RE = re.compile(
    r"\b("
    r"equivalently|in other words|another way (?:to|of)|"
    r"different (?:view|perspective|representation|approach)|"
    r"corresponds? to|same as|can be (?:seen|written|viewed|expressed) as|"
    r"think of (?:it|this) as|translat\w*|conceptually|two (?:ways|views)"
    r")\b",
    re.IGNORECASE,
)

_NONE_RE = re.compile(r"^\s*(none|n/?a|no(ne)?\.?)\s*$", re.IGNORECASE)


def _reps_component(extracted_val: str) -> float:
    if not extracted_val:
        return 0.0
    val = extracted_val.strip()
    if not val or _NONE_RE.match(val):
        return 0.0
    low = val.lower()
    found = set()
    for prefix, canon in _CANON_PREFIXES:
        if prefix in low:
            found.add(canon)
    n = len(found)
    if n <= 1:
        return 0.0
    return min(1.0, (n - 1) / 2.0)


def _case_component(text: str, ops) -> float:
    try:
        skeleton = ops.proof_skeleton(text)
        if not isinstance(skeleton, dict):
            return 0.0
        hits = len(skeleton.get("case", []) or [])
    except Exception:
        hits = 0
    return min(1.0, hits / 2.0)


def _connective_component(text: str, ops) -> float:
    try:
        norm = ops.normalize(text) if text else ""
    except Exception:
        norm = text or ""
    try:
        hits = len(_CONNECTIVE_RE.findall(norm))
    except Exception:
        hits = 0
    return min(1.0, hits / 2.0)


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text:
            return 0.5

        reps_val = ""
        try:
            reps_val = (extracted or {}).get("representations", "") or ""
        except Exception:
            reps_val = ""

        reps_c = _reps_component(reps_val)
        case_c = _case_component(text, ops)
        conn_c = _connective_component(text, ops)

        s = 0.6 * reps_c + 0.25 * case_c + 0.15 * conn_c
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
