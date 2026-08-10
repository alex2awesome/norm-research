"""a42 h1 - Use of visuals and concrete examples (Math StackExchange).

Two general failure modes read out of h0's training residuals (not any single
excerpt):

1. h0's LLM_FIELDS only ever asked about "a concrete numeric example / named
   object / toy case." Nothing in h0 could detect the *other* half of the
   criterion name -- genuinely visual or diagrammatic reasoning (describing a
   graph, a spatial/geometric configuration, a picture) even when no numeric
   example is present. That produced a one-sided instrument: answers that
   reason visually but never say "for example" got scored near zero no
   matter how well they satisfied the criterion.

2. h0's grounding test excluded only one narrow case of "not grounding" --
   reusing the exact numbers already given in the question. But the same
   underlying problem shows up in several other shapes on the residuals: a
   concrete construction that *is* the entire requested answer (a direct
   proof-by-construction, a counterexample offered to disprove a claim, a
   numeric verification table checking an already-derived formula) with no
   separately-stated general idea that the instance is illustrating. In all
   of these the instance IS the substance of what was asked, not an added
   pedagogical aid layered on top of a general argument -- so h0's binary
   YES/NO tier rewarded them as if they were illustrative grounding.

h1 restructures the two LLM fields around this broader distinction (added
illustrative example vs. merely fulfilling the ask; genuine visual/spatial
reasoning as its own channel), and moves the code-side proxy from a flat
richness bonus to (a) a bounded elaboration measure that only scales credit
an LLM-confirmed signal already earned -- so a three-word passing aside no
longer collects the same tier credit as a worked-out illustration -- and (b)
a generic down-weight for numeric-verification-table answers (many
high-precision decimal comparisons), a recognizable genre distinct from
small illustrative toy cases. These are general, code-only predicates, not
rules keyed to particular documents.
"""
import re

LLM_FIELDS = {
    "adds_grounding_example": (
        "Beyond directly fulfilling what the question asked (its literal request, "
        "computation, construction, or proof), does the answer ALSO give a separate "
        "small concrete example (specific numbers/object not required by the question) "
        "illustrating a general idea? Answer YES, NO, or ONLY-ANSWERS-THE-ASK."
    ),
    "visual_or_diagram_reasoning": (
        "Does the answer reason using a picture, graph, diagram, or spatial/geometric "
        "configuration -- not just symbolic algebra -- to support its argument? "
        "Answer YES or NO."
    ),
}

_MARKER_RE = re.compile(
    r"\b(e\.?g\.?|for example|for instance|such as|consider the (?:case|example))\b",
    re.IGNORECASE,
)
_CASE_RE = re.compile(r"\bcase\s*\d", re.IGNORECASE)
_HIGH_PRECISION_RE = re.compile(r"\d+\.\d{4,}")


def _has_signal(value) -> bool:
    v = (value or "").strip().upper()
    return v.startswith("YES")


def _elaboration_score(t, ops) -> float:
    """Bounded 0..1 read on how WORKED-OUT the example/visual content is, as
    opposed to a bare passing mention. Reuses generic surface cues (marker
    phrases, enumerated 'Case N' splits, numeric density inside parsed math
    spans) purely to SCALE credit an LLM-confirmed signal already earned --
    never to manufacture credit by itself."""
    try:
        markers = len(_MARKER_RE.findall(t))
    except Exception:
        markers = 0
    try:
        cases = len(_CASE_RE.findall(t))
    except Exception:
        cases = 0
    density = 0.0
    try:
        spans = ops.extract_math_spans(t)
        total = len(spans)
        if total:
            numeric = sum(1 for _kind, content in spans if any(c.isdigit() for c in content))
            density = numeric / total
    except Exception:
        pass
    raw = 0.25 * min(markers, 3) + 0.35 * min(cases, 3) + 0.9 * density
    return max(0.0, min(1.0, raw / 2.2))


def _verification_heavy(t) -> bool:
    """A recognizable general pattern in math answers: a run/table of
    high-precision decimal comparisons validating an already-derived formula
    numerically. That is technical verification for the asker's benefit, not
    a pedagogical illustration that grounds a general idea for a reader, so
    it should not collect the same credit as a small worked toy case."""
    try:
        return len(_HIGH_PRECISION_RE.findall(t)) >= 4
    except Exception:
        return False


def _malformed_penalty(text, ops):
    """Small penalty for badly broken LaTeX -- an example/diagram that isn't
    'well-integrated' (contract's phrase) shouldn't get full credit."""
    try:
        dh = ops.delimiter_health(text) or {}
    except Exception:
        return 0.0
    issues = 0
    for v in dh.values():
        try:
            if int(v) > 0:
                issues += 1
        except Exception:
            continue
    if issues >= 4:
        return 0.08
    if issues >= 2:
        return 0.04
    return 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if len(t) < 30:
            return 0.0

        extracted = extracted or {}
        has_example = _has_signal(extracted.get("adds_grounding_example"))
        has_visual = _has_signal(extracted.get("visual_or_diagram_reasoning"))

        if has_example and has_visual:
            base = 0.55
        elif has_example or has_visual:
            base = 0.28
        else:
            base = 0.05

        if has_example or has_visual:
            elaboration = _elaboration_score(t, ops)
            base += (elaboration - 0.5) * 0.30

        if _verification_heavy(t):
            base -= 0.15

        penalty = _malformed_penalty(t, ops)

        s = base - penalty
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
