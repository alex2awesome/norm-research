"""a132 -- Unification and cross-area synthesis (Math StackExchange answers).
Iterating on the a132_h0 reference (TRAIN rho 0.4753) via the agentic loop.

DIAGNOSIS (full-TRAIN, n=150 -- not just the 15 worst residuals): h0's own
code_component (bucket-vocabulary count / named-theorem / connective-phrase /
proof-case count) correlates with the judge alone at rho=0.38. h0's combining
formula (`base = 0.5 + 0.5*cc` when the LLM's bridge_field fires, else
`0.35*cc`) reaches rho=0.475 -- but this specific combination has a hard
ceiling: the two branches NEVER overlap in value (max of the "no field" branch
is 0.35 < 0.5, the floor of the "field" branch), so *every* has_field=True item
outranks *every* has_field=False item regardless of code richness. Binning the
actual judge scores by (n_buckets, has_field) shows this is the wrong
structure:

  has_field=False: nb=0 mean=.049  nb=1 mean=.064  nb=2 mean=.067   (FLAT)
  has_field=True:  nb=0 mean=.091  nb=1 mean=.182  nb=2 mean=.386  nb=3 mean=.43

I.e. bucket-vocabulary co-occurrence is *worthless on its own* (flat ~0.05-0.07
no matter how many area-buckets fire) -- exactly the corpus-note warning that
stray co-occurring vocabulary isn't synthesis. It only becomes informative
*gated by* the LLM's bridge_field confirmation, where it climbs cleanly and
monotonically with bucket count. Same story for the other code signals:
  named_hit:       (False,has_field=T)=.172  (True,has_field=T)=.355; never
                   fires at all when has_field=False in TRAIN.
  connective_hit:  (True,has_field=F)=.050 (same as False,False -- pure noise
                   without the field) vs (True,has_field=True)=.485.
So h0's own comment ("field... used as a gate on top of the code-side
richness") describes the right idea but the *implementation* inverts it (field
acts as a flat +0.5 floor, richness is a small modifier on top -- backwards).

Also: proof_skeleton "case" count has near-zero support here and a *negative*
OLS weight once the other features are controlled (routine epsilon-delta /
multi-case elementary proofs, not synthesis) -- dropped as a signal.

CHANGES vs h0 (structural, not per-item):
(1) Restructure combination: has_field now GATES how much bucket/named/
    connective signal counts (multiplicative-by-branch), instead of being an
    additive floor. Without the field, richness is capped near-flat/low
    (matches the empirical flat band); with the field, richness drives a
    monotone climb.
(2) Drop the case-count bonus (unsupported / anti-correlated in TRAIN).
(3) Broaden named-theorem detection (case-insensitive; add lowercase
    "fundamental theorem", "inclusion-exclusion principle", "pigeonhole
    principle", and a generic NAME's theorem/lemma/inequality/identity/
    principle form) -- these only matter when gated by has_field, so
    broadening recall here is safe (can't inflate has_field=False scores).
(4) Broaden connective set with structure-naming/invariance phrasing
    ("is called"/"is known as", "same idea/trick/technique/method/structure",
    "invariant"/"invariance", "symmetric"/"symmetry") -- again gated, so only
    sharpens the has_field=True climb; does not touch the no-field floor.

LLM_FIELDS is UNCHANGED from h0 (same single field, same instruction).
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
        r"\bfundamental group\b", r"\beuler characteristic\b",
        r"\bclosed surface\b",
    ],
    "analysis": [
        r"\bderivative\b", r"\bintegral\b", r"\bconverg", r"\blimit\b",
        r"\bdifferentiable\b", r"\basymptotic", r"\bdifferential equation\b",
        r"\bode\b",
    ],
    "discrete_combinatorics": [
        r"\bcombinat", r"\bbinomial\b",
        r"\binteger partitions?\b", r"\bset partitions?\b", r"\bpartition function\b",
        r"\bstirling\b", r"\bbijection\b", r"\bpermutation", r"\bstars and bars\b",
        r"\binduction\b",
    ],
    "continuous_prob_stats": [
        r"\bdistribution\b", r"\bexpectation\b", r"\bvariance\b",
        r"\bgaussian\b", r"\brandom variable\b", r"\bdensity\b",
        r"\bcovariance\b",
    ],
    "number_theory": [
        r"\bprime\b", r"\bzeta\b", r"\briemann (?:zeta|hypothesis)\b", r"\bmodular\b",
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

# ROUND 2 FIX: a blanket re.IGNORECASE on the h0 "<Name> Way" pattern (round 1)
# made bare "way" phrases ("one possible way", "Another way to think about
# this", "a shorter way") match as a false "named theorem/way" hit -- these
# are ordinary narrative connectives, not a citation of a specific named
# result (e.g. the Twelvefold Way). Keep the ORIGINAL h0 pattern
# case-SENSITIVE (requires an actual Capitalized proper-name token), and add
# the new lowercase-friendly forms as separate, narrowly-scoped alternatives
# (specific multi-word phrases naming a well-known general principle, not a
# generic word before "theorem").
_NAMED_RE = re.compile(
    r"\b[A-Z][a-zA-Z]{2,}(?:-[A-Z][a-zA-Z]{2,})?(?:\s+[A-Z][a-zA-Z]{2,})?"
    r"\s+(?:[Tt]heorem|[Ll]emma|[Cc]orrespondence|[Dd]uality|[Pp]rinciple|"
    r"[Cc]onjecture|[Hh]ypothesis|[Ii]nequality|[Ii]dentity|Way)\b"
)
_NAMED_RE2 = re.compile(
    r"\b[a-zA-Z][a-zA-Z]{2,}(?:'s|’s)\s+(?:theorem|lemma|inequality|identity|principle)\b"
    r"|\bfundamental theorem\b"
    r"|\binclusion[\s-]exclusion principle\b"
    r"|\bpigeonhole principle\b",
    re.IGNORECASE,
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
    # new: structure-naming / invariance phrasing -- gated by has_field below,
    # so broadened recall here cannot inflate the no-field branch.
    r"\bis (?:also |essentially |precisely |basically )?(?:called|known as)\b",
    r"\bsame (?:idea|trick|technique|method|argument|structure|principle|framework)\b",
    r"\b(?:translation|scaling|rotation)(?:-|\s)invarian(?:t|ce)\b",
    r"\bby (?:symmetry|invariance)\b",
    r"\bsymmetry (?:argument|properties?|of the (?:problem|integral))\b",
]
_CONNECTIVE_RE = [re.compile(p) for p in _CONNECTIVE_PATS]


def _answer_body(t):
    m = re.search(r"\bAnswer\s*:", t)
    return t[m.end():] if m else t


def _clean_field(v):
    v = (v or "").strip()
    if not v or v.lower() in ("none", "n/a", "na", "no", "none.", "not applicable"):
        return ""
    return v


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if len(raw) < 20:
            return 0.5
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw

        # ROUND 3: scope widened to the WHOLE document (question+answer), not
        # answer-only -- re-measuring on full TRAIN showed this alone lifts
        # rho (e.g. .513->.520), the same scope fix a198 needed: some
        # genuinely-credited docs set up the cross-area framing in the
        # QUESTION (e.g. an order-theory/function-theory analogy request)
        # that the answer resolves without repeating the vocabulary.
        full_low = t.lower()

        hit_buckets = set()
        for name, pats in _BUCKETS.items():
            for p in pats:
                if re.search(p, full_low):
                    hit_buckets.add(name)
                    break
        n_buckets = len(hit_buckets)

        named_hit = bool(_NAMED_RE.search(t)) or bool(_NAMED_RE2.search(full_low))
        connective_hit = any(r.search(full_low) for r in _CONNECTIVE_RE)

        field = _clean_field((extracted or {}).get("bridge_field", ""))
        has_field = bool(field)

        if has_field:
            # bucket count only informative once corroborated by the field --
            # monotone step ladder calibrated (via TRAIN-only grid search over
            # this exact discrete feature set, not per-item) to the observed
            # (nb, field=True) bin means.
            bucket_base = {0: 0.08, 1: 0.25, 2: 0.35}.get(n_buckets, 0.46)
            gated = bucket_base
            if named_hit:
                gated += 0.18
            # connective bonus only counts once at least one area-bucket is
            # also present -- a bare unifying phrase with ZERO detected
            # domain vocabulary anywhere is weaker corroboration.
            if connective_hit and n_buckets >= 1:
                gated += 0.15
            out = min(1.0, gated)
        else:
            # ROUND 5 FIX: h0's original assumption (and my own round-1
            # carry-over) was that connective/named language is worthless
            # without the LLM's field confirmation -- true for h0's NARROW
            # connective set, but re-measured on the round-3 BROADENED set
            # (naming/invariance phrasing) this bin is no longer flat:
            #   (connective,no field): False->.038  True->.123   (n=13)
            #   (named,     no field): False->.052  True->.075
            # I.e. the LLM sometimes fails to answer NONE-vs-name correctly
            # (missing a genuine bridge, esp. within-area unification the
            # field's own instruction wasn't designed to catch), but the
            # code-side unifying-language signal still carries some of that
            # missed evidence on its own. Small, capped bonuses (well below
            # the has_field branch) recover part of this without letting
            # bare vocabulary co-occurrence alone drive the score.
            out = 0.05 + 0.02 * min(n_buckets, 2)
            if connective_hit:
                out += 0.15
            if named_hit:
                out += 0.08

        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5
