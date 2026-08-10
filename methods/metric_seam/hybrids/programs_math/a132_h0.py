"""a132 -- Unification and cross-area synthesis (Math StackExchange answers).

Criterion: reveals/leverages conceptual unity across math areas (or to other
fields), grouping results under shared structures/perspectives.

Why not the v0_keyword baseline (train rho=0.099): its trigger words
("connect", "geometric", "topological", ...) are ordinary technical
vocabulary in math prose (e.g. "connection 1-form", "geometric
distribution", "disconnected metric space", "connected" as a topology
predicate) and fire constantly with zero relation to genuine cross-area
synthesis. Reading the 30 train examples: judge=0 items are single-topic
computations/proofs (however LaTeX-heavy); judge>=0.5 items explicitly
carry the argument INTO a second area/field and say so ("this differential
equation ... is Schrodinger's equation", "different views of the same
model" for Beta/Binomial, adjoint-functor characterization of
discrete/indiscrete topology, Gauss-Bonnet tying curvature to Euler
characteristic, Bessel zeros vs Riemann zeta zeros, the Twelvefold Way
enumerating 4 counting regimes). So the predicate needs: (a) evidence of
>=2 genuinely distinct area/field vocabularies actually used in the ANSWER
(not just the question setup), (b) generic relational/equivalence phrasing
("in terms of", "equivalent to", "relations between", "arises in", ...)
that is NOT the overloaded baseline words, (c) explicit named-theorem
citation or multi-case enumeration (proxy for "grouping under one
structure"). Regex/parsing can flag *candidate* synthesis but cannot
reliably tell "incidental co-occurring vocabulary" from "the answer is
actually using field B to explain claim in field A" -- that's exactly the
kind of tacit judgment call the contract says an LLM should ground. Hence
one LLM field that names the bridged-to field (or NONE), used as a gate on
top of the code-side richness score so stray vocabulary overlap alone
(the same failure mode as the keyword baseline) cannot inflate the score.
"""

import re

LLM_FIELDS = {
    "bridge_field": (
        "Name ONE other math subfield, science, or discipline (e.g. physics, "
        "category theory, statistics, computer science, number theory) that "
        "this ANSWER explicitly draws on or connects to beyond the question's "
        "own narrow topic, in <=6 words. Answer NONE if the answer stays "
        "entirely within one narrow topic or technique."
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
]
_CONNECTIVE_RE = [re.compile(p) for p in _CONNECTIVE_PATS]


def _answer_body(t):
    m = re.search(r"\bAnswer\s*:", t)
    return t[m.end():] if m else t


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
        body_low = body.lower()

        hit_buckets = set()
        for name, pats in _BUCKETS.items():
            for p in pats:
                if re.search(p, body_low):
                    hit_buckets.add(name)
                    break
        n_buckets = len(hit_buckets)

        named_hit = bool(_NAMED_RE.search(body))
        connective_hit = any(r.search(body_low) for r in _CONNECTIVE_RE)

        n_cases = 0
        try:
            skel = ops.proof_skeleton(t)
            n_cases = len(skel.get("case", []) or [])
        except Exception:
            n_cases = 0

        code_component = 0.0
        if n_buckets >= 3:
            code_component += 0.6
        elif n_buckets == 2:
            code_component += 0.35
        if named_hit:
            code_component += 0.2
        if connective_hit:
            code_component += 0.2
        if n_cases >= 4:
            code_component += 0.3
        elif n_cases >= 2:
            code_component += 0.15
        code_component = max(0.0, min(1.0, code_component))

        field = (extracted or {}).get("bridge_field", "") or ""
        field = field.strip()
        field_low = field.lower()
        has_field = bool(field) and field_low not in (
            "none", "n/a", "na", "no", "none.", "not applicable",
        )

        if has_field:
            base = 0.5 + 0.5 * code_component
        else:
            base = 0.35 * code_component

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
