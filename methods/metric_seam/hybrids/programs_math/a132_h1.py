"""a132 h1 -- Unification and cross-area synthesis (Math StackExchange answers).

Criterion: reveals/leverages conceptual unity across math areas (or to other
fields), grouping results under shared structures/perspectives.

Why h0 over-fires (train rho=0.475 but fails held-out): h0 scores an answer
as "unifying" whenever (a) >=2 of its 11 vocabulary buckets appear ANYWHERE
in the answer, or (b) a generic relational phrase ("in terms of", "reduces
to", ...) appears ANYWHERE, or (c) the LLM names ANY field at all -- and
crucially the LLM-field alone floors the score at 0.5 regardless of how weak
the code evidence is. This conflates three distinct things with genuine
cross-area synthesis:
  1. Two domains' vocabulary merely co-occurring in a long answer even though
     neither sentence actually ties them together (buckets counted over the
     WHOLE document, unrelated to where any connective phrase sits).
  2. Borrowing a single routine lemma from an adjacent area to prove a
     same-area claim (e.g. citing "matrix multiplication is continuous" to
     finish a linear-algebra limit proof, or citing the Fundamental Theorem
     of Arithmetic to finish a number-theory divisibility proof) -- using
     one tool from area B is not "grouping results under a shared
     structure."
  3. The LLM naming the answer's own native field as if it were a "bridge"
     (a quantum-mechanics question tagged "Quantum mechanics", a topology
     question tagged "Topology") -- there is no second area here at all.
  4. Enumerating several ad hoc proof cases within ONE area (e.g. a
     probability/combinatorics case split) counted as if it were structural
     unification a la the Twelvefold Way.

h1's fix targets the general mechanism, not the specific excerpts:
  - Vocabulary-bucket and connective-phrase evidence is now required to
    co-occur inside the SAME local window of text, not merely anywhere in
    the document -- this is what actually distinguishes "the argument
    states a link here" from "two domains' words happen to both appear."
  - A cited named theorem/correspondence only counts if it sits in a window
    that also carries cross-bucket or connective evidence (so a routine
    single-area theorem citation gets no credit).
  - Multi-case proof structure is only a (small) bonus on top of existing
    cross-area evidence, never a standalone contributor.
  - The single LLM field is rewritten to demand a concrete two-sided
    correspondence ("A = B" / "A corresponds to B") instead of "any field
    beyond the question's own topic" -- this stops the LLM from crediting
    the answer's own native subject or a passing aside as a bridge, and it
    can no longer alone floor the score at 0.5: an LLM claim with zero code
    corroboration is capped below the "both agree" tier.
"""

import re

LLM_FIELDS = {
    "cross_area_link": (
        "In <=12 words, state the SPECIFIC identity/correspondence the ANSWER "
        "draws between two manifestly DIFFERENT areas, fields, or disciplines, "
        "in the form 'A = B' or 'A corresponds to B' (e.g. 'this ODE = "
        "Schrodinger equation', 'Beta and Binomial = same model', 'adjoint "
        "functors = discrete/indiscrete topology'). Answer NONE if the answer "
        "only applies one area's own standard tools/vocabulary, merely cites a "
        "theorem native to its own topic, or only mentions a second field's "
        "term in passing without asserting an explicit correspondence between "
        "two sides."
    ),
}

_BUCKETS = {
    "algebra": [
        r"\bgroup\b", r"\bring\b", r"\bfield\b", r"\bmodule\b",
        r"\bhomomorphism", r"\bisomorphism", r"\bideal\b", r"\balgebra\b",
        r"\bmatri(?:x|ces)\b", r"\bvector space\b", r"\bfree product\b",
    ],
    "topology": [
        r"\btopolog", r"\bcompact\b", r"\bhomotopy", r"\bhomeomorphism",
        r"\bmanifold\b", r"\bopen (?:set|cover)\b", r"\bclosed set\b",
        r"\bcontinuous\b", r"\bfundamental group\b", r"\beuler characteristic\b",
        r"\bclosed surface\b",
    ],
    "analysis": [
        r"\bderivative\b", r"\bintegral\b", r"\bconverg", r"\blimit\b",
        r"\bdifferentiable\b", r"\basymptotic", r"\bdifferential equation\b",
        r"\bode\b",
    ],
    "discrete_combinatorics": [
        r"\bcombinat", r"\bbinomial\b", r"\bpartition", r"\bstirling\b",
        r"\bbijection\b", r"\bpermutation", r"\bstars and bars\b",
        r"\binduction\b",
    ],
    "continuous_prob_stats": [
        r"\bdistribution\b", r"\bexpectation\b", r"\bvariance\b",
        r"\bgaussian\b", r"\brandom variable\b", r"\bdensity\b",
        r"\bcovariance\b",
    ],
    "number_theory": [
        r"\bprime\b", r"\bzeta\b", r"\briemann\b", r"\bmodular\b",
        r"\bcongruence\b", r"\bdivisib",
    ],
    "special_functions": [
        r"\bbessel\b", r"\bgamma function\b", r"\bdigamma\b",
        r"\bhypergeometric\b",
    ],
    "geometry": [
        r"\bcurvature\b", r"\bmetric tensor\b", r"\briemannian\b",
        r"\beuclidean\b", r"\btensor\b", r"\bgeodesic\b", r"\bcoordinate",
    ],
    "category_logic": [
        r"\bfunctor\b", r"\badjoint\b", r"\bmorphism\b", r"\bcategory\b",
        r"\bnatural transformation\b", r"\bhom-set", r"\byoneda\b",
    ],
    "physics_applied": [
        r"\bquantum\b", r"\bhamiltonian\b", r"\bschr[oö]dinger\b",
        r"\bmechanics\b", r"\bentropy\b", r"\bwave function\b",
        r"\brelativity\b",
    ],
    "cs_applied": [
        r"\balgorithm\b", r"\bcomplexity class\b", r"\bturing\b",
        r"\bautomaton\b", r"\bencrypt",
    ],
}

_NAMED_RE = re.compile(
    r"\b[A-Z][a-zA-Z]{2,}(?:-[A-Z][a-zA-Z]{2,})?(?:\s+[A-Z][a-zA-Z]{2,})?"
    r"\s+(?:[Tt]heorem|[Ll]emma|[Cc]orrespondence|[Dd]uality|[Pp]rinciple|"
    r"[Cc]onjecture|[Hh]ypothesis|Way)\b"
)

_CONNECTIVE_PATS = [
    r"\bdifferent views? of the same\b",
    r"\bequivalent(?:ly)? to\b",
    r"\bcorresponds? to\b",
    r"\bin terms of\b",
    r"\bspecial case of\b",
    r"\bgeneraliz\w*\b",
    r"\breduces? to\b",
    r"\bturns? out to be\b",
    r"\barises? in\b",
    r"\brelations? between\b",
    r"\bsame model\b",
    r"\bcan be (?:viewed|seen|thought of) as\b",
    r"\bexactly the\b",
    r"\breformulat\w*\b",
    r"\btranslat\w*\s+(?:into|to)\b",
    r"\banalogous to\b",
    r"\brelate[sd]?\s+to\b",
    r"\bconnect(?:s|ion|ed)?\s+(?:to|between|with)\b",
    r"\bidentified with\b",
    r"\bamounts to\b",
    r"\bboils down to\b",
    r"\bis precisely\b",
    r"\bnothing but\b",
    r"\bunifies?\b",
]
_CONNECTIVE_RE = [re.compile(p) for p in _CONNECTIVE_PATS]

_NEGATIVE_FIELD_TOKENS = {
    "none", "n/a", "na", "no", "none.", "not applicable", "no bridge",
    "no link", "n/a.", "nothing",
}


def _answer_body(t):
    m = re.search(r"\bAnswer\s*:", t)
    return t[m.end():] if m else t


def _windows(s, size=420, step=220):
    if not s:
        return []
    n = len(s)
    if n <= size:
        return [s]
    out = []
    i = 0
    while i < n:
        out.append(s[i:i + size])
        if i + size >= n:
            break
        i += step
    return out


def _buckets_in(w_low):
    hit = set()
    for name, pats in _BUCKETS.items():
        for p in pats:
            if re.search(p, w_low):
                hit.add(name)
                break
    return hit


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if len(raw) < 20:
            return 0.5
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        body = _answer_body(t)

        link_windows = 0
        named_and_linked = 0
        for w in _windows(body):
            w_low = w.lower()
            buckets_here = _buckets_in(w_low)
            has_connective = any(r.search(w_low) for r in _CONNECTIVE_RE)
            has_named = bool(_NAMED_RE.search(w))
            if len(buckets_here) >= 2 and has_connective:
                link_windows += 1
            if has_named and (len(buckets_here) >= 2 or has_connective):
                named_and_linked += 1

        n_cases = 0
        try:
            skel = ops.proof_skeleton(t)
            n_cases = len(skel.get("case", []) or [])
        except Exception:
            n_cases = 0

        code_component = 0.0
        if link_windows >= 2:
            code_component += 0.5
        elif link_windows == 1:
            code_component += 0.3
        if named_and_linked >= 1:
            code_component += 0.3
        if n_cases >= 3 and (link_windows >= 1 or named_and_linked >= 1):
            code_component += 0.1
        code_component = max(0.0, min(1.0, code_component))

        field = (extracted or {}).get("cross_area_link", "") or ""
        field = field.strip()
        field_low = field.lower()
        has_link = (
            bool(field)
            and len(field) >= 3
            and field_low not in _NEGATIVE_FIELD_TOKENS
        )

        if has_link and code_component > 0.0:
            base = 0.5 + 0.5 * code_component
        elif has_link:
            base = 0.4
        else:
            base = 0.3 * code_component

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
