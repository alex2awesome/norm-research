"""Hybrid metric for a78: 'Axiomatic presentation and commitment'
(build from clearly stated axioms/definitions and derive results deductively,
in an abstract, axiomatic style when appropriate).

Rationale (see improver pack a78.json): the baseline regex (\\begin{theorem}
etc.) barely fires on this corpus -- these are StackExchange prose answers,
not formatted papers -- so its 0.332 train rho is mostly the leftover keyword
term. The real judge signal, read off the 30 examples, is (a) density of
explicit proof/definition scaffolding and formal quantified language
(evidence a claim is being unwound from a stated definition rather than
asserted), tempered by (b) a semantic distinction the code cannot see:
whether the passage is reasoning ABOUT a definition/structure abstractly, or
just using precise notation to grind through a concrete numeric/geometric
instance (e.g. Erdos-prime-style counting, digit-count algebra, Bessel-zero
tables all score low despite heavy, correct LaTeX). One LLM field targets
exactly that abstract-vs-concrete distinction.
"""

import re

LLM_FIELDS = {
    "abstraction_move": (
        "Classify the answer's core move with exactly one label: "
        "'abstract-definitional' if it reasons from a stated definition/axiom/"
        "structure in general terms; 'concrete-computational' if it mainly "
        "grinds through a specific numeric/geometric computation or worked "
        "example. Reply with just one of those two labels, or NONE if unclear."
    ),
}

_FORMAL_PAT = re.compile(
    r"(\\forall|\\exists|\\iff|\\Longleftrightarrow|\\Leftrightarrow|\\implies|\\Rightarrow|"
    r"\bfor all\b|\bfor any\b|\bthere exists\b|\bif and only if\b|"
    r"\bis defined as\b|\bby definition\b|\bis called\b|\bwe define\b|"
    r"\bwe say\b|\bwe call\b|\bsuch that\b|\bin the form\b|\blet .{1,40}? be\b)",
    re.IGNORECASE,
)

_ABSTRACT_KW = ("abstract", "axiom", "definition", "formal", "structure", "deduc")
_CONCRETE_KW = ("concrete", "comput", "numeric", "specific instance", "geometr", "example only")


def _n(skel, cat):
    try:
        v = skel.get(cat) or []
        return len(v)
    except Exception:
        return 0


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t = ops.normalize(text)
        if not t or not t.strip():
            return 0.0

        n_words = max(20, len(re.findall(r"[A-Za-z]+", t)))
        scale = max(0.4, n_words / 150.0)  # density normalizer, ~150-word unit

        try:
            skel = ops.proof_skeleton(t) or {}
        except Exception:
            skel = {}

        n_def = _n(skel, "definition")
        n_thm = _n(skel, "theorem")
        n_proof = _n(skel, "proof")
        n_qed = _n(skel, "qed")
        n_case = _n(skel, "case")
        n_conn = _n(skel, "connective")

        def_score = min(1.0, (n_def / scale) / 1.2)
        conn_score = min(1.0, (n_conn / scale) / 3.0)
        thm_score = min(1.0, (n_thm / scale) / 1.0)
        proof_score = min(1.0, (n_proof / scale) / 1.0)
        case_score = min(1.0, (n_case / scale) / 2.0)
        qed_score = min(1.0, n_qed / 1.0)  # rare; any hit counts fully

        n_formal = len(_FORMAL_PAT.findall(t))
        formal_score = min(1.0, (n_formal / scale) / 2.0)

        try:
            _, _, n_numbered, _ = ops.equation_stats(t)
        except Exception:
            n_numbered = 0
        numbered_score = min(1.0, n_numbered / 2.0)

        struct_core = (
            0.27 * def_score
            + 0.20 * conn_score
            + 0.12 * thm_score
            + 0.08 * proof_score
            + 0.06 * case_score
            + 0.04 * qed_score
            + 0.04 * numbered_score
            + 0.19 * formal_score
        )
        struct_core = max(0.0, min(1.0, struct_core))

        # Semantic gate: abstract structural reasoning vs. concrete grinding.
        # This is the tacit half the code can't see -- many low- and mid-judged
        # examples are LaTeX-dense but concretely computational (numeric
        # integrals, digit-counting algebra), which the structural proxy alone
        # over-rewards; conversely some of the highest-judged examples are
        # structurally "thin" (a single cleanly-stated definition, e.g. "a
        # rational number is a/b where a,b in Z, b!=0") yet score very high
        # because the move IS abstract/definitional. So when the extractor can
        # classify the move, let it dominate the blend; otherwise fall back to
        # the structural proxy alone.
        note = (extracted.get("abstraction_move", "") or "").strip().lower()
        # Concrete-marker check takes priority: a stray "abstract" substring
        # inside a negated phrase (e.g. "concrete ..., not abstract") must not
        # flip this to the abstract branch.
        is_concrete = any(k in note for k in _CONCRETE_KW)
        is_abstract = (not is_concrete) and any(k in note for k in _ABSTRACT_KW)
        if is_abstract and not is_concrete:
            raw = 0.40 * struct_core + 0.60 * 0.88
        elif is_concrete and not is_abstract:
            raw = 0.40 * struct_core + 0.60 * 0.15
        else:
            raw = struct_core

        # Mild penalty for badly malformed LaTeX -- a proxy for careless
        # exposition, never a hard gate.
        try:
            health = ops.delimiter_health(t) or {}
            broken = sum(v for v in health.values() if isinstance(v, (int, float)))
        except Exception:
            broken = 0
        if broken >= 6:
            raw *= 0.85

        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
