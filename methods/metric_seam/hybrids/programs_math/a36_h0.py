"""Hybrid metric channel for a36: Structure and surveyability (modular proofs).

Rationale (see accompanying report): the description-compiled baseline (train
rho 0.094) counts \\begin{theorem}-style environments and Step/Case/Part
keywords, which almost never fire on Math StackExchange answers -- so it is
nearly flat. Reading the train examples, what the judge actually rewards is
scaffolding that makes an argument point-at-able: numbered/tagged equations
that get referenced again later, explicit (ad hoc or formal) stage labels,
and a real density of logical connectives (therefore/thus/since/...), not raw
LaTeX-keyword presence (several equation-dense, connective-free answers are
judged 0.0-0.1 despite heavy markup). Long answers with zero such scaffolding
are penalized regardless of prose quality (unsurveyable wall of text).
"""

import re
from collections import Counter

LLM_FIELDS = {
    "STAGE_LABELS": (
        "List any ad hoc section/step/case labels the answer uses to organize its "
        "argument (e.g. 'Step 1, Step 2' or 'First, Second, Finally'), else say NONE."
    ),
}


def _sat(x, k=1.0):
    try:
        return x / (x + k) if x > 0 else 0.0
    except Exception:
        return 0.0


def _safe_call(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception:
        return default


def _count_labels(s):
    if not s:
        return 0
    parts = [p.strip() for p in re.split(r"[,;]", s) if p.strip()]
    parts = [p for p in parts if len(p) <= 24]
    return min(len(parts), 6)


_CONNECTIVE_WORDS = (
    r"therefore|thus|hence|consequently|so\b|since|because|it follows that|"
    r"this means|by definition|we (?:get|have|obtain)|implies|"
    r"if and only if|\biff\b|contradiction|conclude|observe that|recall that|"
    r"note that"
)
_CONNECTIVE_RE = re.compile(_CONNECTIVE_WORDS, re.I)


def _connective_fallback(t):
    # supplementary, generic connective count -- hedges against ops'
    # proof_skeleton using a narrower vocabulary than expected.
    try:
        return len(_CONNECTIVE_RE.findall(t))
    except Exception:
        return 0


def _eq_backrefs(t):
    # numbered/tagged equations that get pointed back at later in the prose
    # are the clearest checkable signal of a modular, surveyable argument.
    # Handles both \tag{1} and bare \tag1 conventions seen in scraped MathJax.
    try:
        tags = re.findall(r"\\tag\{\s*([^\}]{1,12})\s*\}", t)
        tags += re.findall(r"\\tag\*?(\d{1,2}[a-z]?)\b", t)
        parens = re.findall(r"\((\d{1,2}[a-z]?)\)", t)
        c = Counter(tags)
        c.update(parens)
        return sum(1 for v in c.values() if v >= 2)
    except Exception:
        return 0


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5
        t = _safe_call(ops.normalize, text, default=text) or text
        nw = max(len(re.findall(r"[A-Za-z]+", t)), 1)

        eqstats = _safe_call(ops.equation_stats, t, default=(0, 0, 0, 0.0))
        try:
            n_display, n_inline, n_numbered, avg_tokens = eqstats
        except Exception:
            n_display, n_inline, n_numbered, avg_tokens = 0, 0, 0, 0.0

        skeleton = _safe_call(ops.proof_skeleton, t, default={}) or {}
        n_case_code = len(skeleton.get("case", []) or [])
        n_connective = max(len(skeleton.get("connective", []) or []), _connective_fallback(t))
        n_formal = sum(
            len(skeleton.get(c, []) or []) for c in ("theorem", "definition", "proof", "qed")
        )

        dhealth = _safe_call(ops.delimiter_health, t, default={}) or {}
        try:
            n_issues = sum(int(v) for v in dhealth.values())
        except Exception:
            n_issues = 0

        sstats = _safe_call(ops.sent_stats, t, default=(1, float(nw), 0.0))
        try:
            n_sent, mean_wps, frac_long = sstats
        except Exception:
            n_sent, mean_wps, frac_long = 1, float(nw), 0.0

        # ad hoc stage labels (e.g. "First Problem / Second Problem") that a
        # fixed regex vocabulary can miss -- thick-input grounding only.
        n_case_llm = _count_labels(extracted.get("STAGE_LABELS", "") if extracted else "")
        n_case = max(n_case_code, n_case_llm)

        reused_refs = _eq_backrefs(t)

        numbering_score = _sat(n_numbered, 1.0)
        backref_score = _sat(reused_refs, 1.0)
        case_score = _sat(n_case, 1.5)
        formal_score = _sat(n_formal, 1.0)
        connective_density = n_connective / nw * 100.0
        connective_score = _sat(connective_density, 3.0)

        # "lowest-level steps are short and transparent"
        sent_transparency = 1.0 - _sat(max(mean_wps - 15.0, 0.0), 15.0)
        if (n_display + n_inline) > 0 and avg_tokens:
            eq_transparency = 1.0 - _sat(max(avg_tokens - 18.0, 0.0), 18.0)
        else:
            eq_transparency = 0.6  # neutral: no math spans to judge

        delimiter_ok = 1.0 - _sat(n_issues, 3.0)

        sigs = [
            (1.0, numbering_score),
            (0.6, backref_score),
            (1.0, case_score),
            (0.5, formal_score),
            (1.0, connective_score),
            (0.6, sent_transparency),
            (0.4, eq_transparency),
            (0.4, delimiter_ok),
        ]
        wsum = sum(w for w, _ in sigs) or 1.0
        agg = sum(w * s for w, s in sigs) / wsum

        # long + zero scaffolding == unsurveyable regardless of prose quality
        structure_total = n_numbered + reused_refs + n_case + n_formal
        if nw > 350 and structure_total == 0:
            length_pen = max(0.55, 1.0 - (nw - 350) / 2000.0)
        else:
            length_pen = 1.0

        raw = agg * length_pen
        return max(0.0, min(1.0, 0.03 + 0.94 * raw))
    except Exception:
        return 0.5
