# Where meaning lives: binding, provenance, and the coded–articulated boundary

*2026-07-04. Theory note. Companion to `notes/2026-07-02__two-faces-theory.md` (Face-1 census /
Face-2 decompression), `notes/2026-07-03__seam-certificate-lemmas.md` (certificate semantics),
and the CAM definition in `methods/metric_seam/pilot/cam_profile.py`. Prompted by two boundary
probes: (1) code that imports large libraries "indexes a TON of code" — isn't that
meaning-retrieval too? (2) code that runs a stochastic, data-driven procedure (LDA) seems to
create meaning in the learning or the randomness, not in the literal mechanism.*

---

## 0. The claim under pressure

The clean statement was: **a coded metric contains its own application; an articulated metric
points at an application living in the interpreter.** Code = meaning specified in mechanism;
articulation = meaning retrieved from enculturated competence.

Both probes attack the same joint: they exhibit *code* whose meaning is not "in the mechanism"
in any naive sense. `import spacy` is fifty characters indexing millions of lines plus learned
weights. `lda.fit(corpus)` is a fully specified mechanism whose output semantics ("topic 7")
nobody specified. If the dichotomy were *self-contained vs. indexed*, both would land on the
articulated side — which is absurd, since both are certifiable, replicable, and
judge-independent in exactly the way our code arm requires.

The resolution: the dichotomy was never really about self-containment. **All symbols index**
— even `+` defers to the CPU's adder; a "self-contained" program is one whose deferrals bottom
out in a conformance-checkable substrate. What separates the coded from the articulated pole is
not *whether* meaning is referenced but **(i) how the reference is bound** and **(ii) where the
referent's extension came from**. These are two independent axes, and the classic binary is
just the two extreme corners of their product. Stochasticity, examined below (§4), turns out
to load on neither axis — it lives in the reliability layer our ceiling machinery already
normalizes away.

---

## 1. Setup and notation

An **implementation** is a pair (π, e): program-plus-payload text π ∈ Π and executor e ∈ E,
inducing a verdict function m_{π,e}: X → [0,1]. The target is a judge criterion with
realizations M_ω and expected verdict M̄_E(x); channel quality is Spearman ρ(m, M̄) with the
attenuation ceiling c(rel₁, K) of the seam pipeline, giving the ceiling-normalized certified
floor r̃ = clip₀₁(ρ/c). Factor every implementation as

  m = F(P, D)

where **P** is the *procedure*: the human-followable text (your program, plus the transitively
imported program DAG), and **D** is the *payload*: inert data the procedure consults (weights,
topic matrices, retrieval indices, checkpoints). Either part may be empty. K(·) is description
length; I(·;·) is mutual information; ops/evidence-op notation follows the seam proposal §3.3.

---

## 2. Probe 1 — libraries. Answer: what matters is *binding*, not size

`import spacy` and "score this the way a fiction editor would" are both references. They differ
in **binding time** and **binding rigidity**.

**Definition 2.1 (conformance class).** Let S(π) be the specification content carried by π —
version pins, artifact hashes, API contracts, the test suites of every node in the import DAG.
The conformance class is C(π) = { e ∈ E : e ⊨ S(π) }. The implementation is **rigid** iff
m_{π,e} = m_{π,e'} pointwise for all e, e' ∈ C(π).

**Proposition 2.2 (certificate transport).** Any statistical certificate of m_{π,e} against
M̄_E (a gate, an r̃, a bootstrap posterior) is a statement about the function m_{π,e}; if the
implementation is rigid, the certificate extends verbatim to every executor in C(π). ∎

The epistemic size of a certificate is therefore not just its ρ — it is ρ *times the breadth
of the class it transports across*:

- **Pure code:** C = every standards-conforming interpreter, ever. Maximal transport.
- **Code + pinned libraries:** C = every platform that runs those artifact hashes. Slightly
  smaller, still enormous, and — the crucial property — *conformance is itself checkable by
  mechanism* (hashes, test suites). The reference is **early-bound and frozen**: resolved to a
  fixed artifact before any verdict is issued.
- **Prompt + frozen LM, temp 0:** rigid! — but C ≈ {this checkpoint × this decode stack}, a
  near-singleton. The certificate is *artifact-bound*: valid, replicable, and it transports
  almost nowhere. (This is the formal content of the same-family-scaling discipline: swapping
  the interpreter voids the certificate, so families are never pooled.)
- **Prompt + live endpoint / human judge:** the reference is **late-bound** — resolved at each
  application by whatever interpreter shows up. C has no nontrivial rigid subset; even
  self-swap fails (rel₁ < 1). There is no artifact to hash because the referent is not an
  artifact: it is a *practice*, maintained by ongoing community calibration.

So the library probe resolves cleanly: importing a ton of code moves you along a **cost**
dimension (auditing spacy is expensive) but barely along the **binding** dimension (the
reference is as rigid as your own code, and its rigidity is machine-checkable). What the
naive dichotomy was tracking all along is binding: *coded = the verdict function is fixed
before use, by artifacts, checkably; articulated = the verdict function is fixed at use, by
whoever interprets.* Size of the indexed mass is a red herring — rigid references compose
into arbitrarily large rigid references.

One honest residue survives: audit cost. A certificate over a huge P is transportable but its
*explanation* is only as accessible as P is readable. That residue is better carved at the
next joint.

---

## 3. Probe 2 — learned components. Answer: *provenance* of the extension

LDA is fully specified as procedure, yet "topic 7" means nothing you can read off the Gibbs
sampler. The meaning sits in D — and, unlike a library's payload, **nobody put it there by an
act of specification**. It was *found* by an optimizer. This is a genuinely different way for
an extension to come into existence, and it deserves its own coordinate.

**Definition 3.1 (specification residual).** σ(π) = K( m_{π,e} | P ): the description length
of the verdict function given the human-followable procedure alone — how much of the mapping
lives in payload that the procedure text does not determine or explain.

**Definition 3.2 (provenance ladder).** Classify where an implementation's extension was fixed:

1. **Specified.** Every component's mapping traces to intentional articulation acts — someone
   wrote down what it should do, and conformance is checkable against that writing-down. Pure
   code, formulas, rule systems. Libraries too: an import DAG is a *DAG of specification acts*,
   each node authored, reviewed, contract-tested. σ ≈ 0 (extension derivable from P; audit
   cost may be large but is bounded and parallelizes).
2. **Selected.** The mapping was found by optimization against a frozen dataset: LDA topics,
   a trained cross-encoder, fitted regression weights. The *procedure* is specified; the
   *semantics* is fixed only **ostensively by the data** — "topic 7" is defined the way "that
   color" is defined by pointing. σ ≈ K(D): you cannot say what topic 7 is without exhibiting
   the corpus and the run. In Two-faces terms this is **ostension at the artifact level**, and
   ostension is already a census violation (Face-1): the extension's basis cannot be
   enumerated, only sampled.
3. **Enculturated.** The mapping is a compressed image of a community's ongoing normative
   practice — human judgment, and derivatively an LLM checkpoint, which is best understood as
   a **frozen ethnographic snapshot** of that practice. σ ≈ K(training distribution + RLHF
   practice): no finite exhibited corpus fixes the extension the community would enforce,
   because the community adjudicates novel cases in ways no snapshot determines (Face-2
   decompression; rule-following). Enculturated meaning is *selected meaning whose selecting
   data is itself a practice, not a dataset*.

The ladder is ordered by **who fixed the extension**: an author / an optimizer over frozen
data / a community over ongoing practice. It is *not* ordered by binding — an LDA artifact and
an LLM checkpoint can both be perfectly frozen (Definition 2.1 rigid). That is exactly why the
two axes are independent, and why the LDA probe felt paradoxical under the one-axis story:
LDA-code is **(rigid, selected)** — coded by binding, articulated-ish by provenance.

Our own pipeline already crossed this line knowingly: `Ops.retrieve_similar` (TF-IDF over the
task corpus) is a *selected* component inside the "code" arm, and the evidence-op taxonomy
already marks it — Z touching corpus state is an evidence op whether the touch happens at
**call time** (live retrieval) or at **build time** (a fitted payload). Learning is
build-time evidence. The op taxonomy and the provenance ladder are the same distinction seen
from two sides:

  computation op = specified, Z = f(X)
  evidence op, call-time = late-bound world state
  evidence op, build-time = **selected payload** (LDA, CE, indices)
  LLM field = **enculturated payload**, bounded borrowing (≤ 2 fields)

**Measurability.** σ is not directly computable, but the project already estimates it from
the outside: the **recovery/reconstruction experiments** (recon-R; metric rediscovery from
(x, verdict) patterns) ask whether a strong recoverer can compress a metric's behavior back
into a short articulation. Specified components re-articulate (that is what "description-
compiled floor" means); selected components re-articulate only up to their ostensive core;
enculturated components resist precisely to the degree the snapshot's practice outruns any
short prompt. Recon-R is a behavioral upper bound on 1/σ, with T = I(M_ω; X) capping what any
recovery can extract.

---

## 4. Stochasticity — orthogonal to both axes

Two different things hide under "stochastic procedure":

**Run-time draws.** m_ω, ω ∈ Ω_run: the metric is a *random function*; its identity is the
distribution. This is not a boundary problem for the theory because it is already the
**general case**: our judge channel is exactly a stochastic metric (pass1/pass2 are two draws;
rel₁; attenuation ceiling; M̄_E the expectation), and deterministic code is the rel = 1
corner. Seeding is a *binding-time operation on ω*: it moves a draw from run time into the
artifact, and adds nothing semantic. The tvd_mi tie-breaking fix is the clean example —
seeded jitter, mechanism-specified, documented role, deterministic given the seed: randomness
fully on the *specified* rung.

**Proposition 4.1 (stochastic normalization).** For implementations with reliabilities
rel(k), the ceiling-normalized r̃ compares channels on equal footing under the same
exchangeability-of-draws assumption the judge's own attenuation ceiling already carries
(lemma-note A1 territory). Stochasticity therefore introduces no new epistemic *category* —
it loads on the reliability layer, which the ceiling divides out. ∎

**Build-time draws** (LDA initialization, SGD data order) are subtler: a distribution over
*artifacts*. The learned meaning is "whichever equilibrium seed 42 found," and its identity
across refits is an empirical stability question — the split-half / orbit-stability
measurements are exactly this shape. Refit instability is a *provenance defect*, not a
randomness defect: it says the selected meaning is not even ostensively stable (the pointing
finger wobbles). So build-time stochasticity matters only through axis 2, as variance in what
got selected.

Punchline: **stochasticity ⟂ articulacy.** What the LDA probe detects is not the randomness;
it is the learnedness.

---

## 5. The replacement formalism: an executor lattice with two coordinates

Drop the binary. An implementation carries coordinates

  ( |C(π)| , σ(π) )  —  transport breadth × specification residual,

with reliability as a third, ceiling-normalized dimension. The familiar objects:

| implementation | binding (|C|) | provenance (σ) |
|---|---|---|
| pure code | maximal | ≈ 0 (specified) |
| code + pinned libs | maximal | ≈ 0, high audit cost |
| code + frozen LDA/CE | artifact-hash class | medium (selected, ostensive) |
| code + ≤2 LLM fields, pinned ckpt | near-singleton | large (enculturated snapshot), **bounded** |
| prompt + live LM / human | none | total |

"Coded metric" = top-left corner; "articulated metric" = bottom-right. Everything interesting
we build lives on the diagonal, and **hybrids are engineered points**: freeze the binding
(keep certifiability and replication) while *borrowing a bounded quantity of enculturated
meaning*. The ≤2-field contract is literally a **budget on borrowed meaning**, and the
Null-ops ablation twins (NullExecOps, NullPriorArtOps) measure the certified marginal value
of each borrowed unit.

**Executor lattice.** Order executor classes by inclusion of ops and payload rungs:
E₁ = {code + computation ops} ⊂ E₂ = E₁ ∪ {≤2 enculturated fields} ⊂ E₃ = E₂ ∪ {evidence
ops} ⊂ … Each criterion has an **articulation spectrum** E ↦ r̃_E, and:

**Proposition 5.1 (monotone CAM, one-sided).** E ⊆ E′ ⇒ r̃_E ≤ r̃_{E′} for the *true*
optima; hence CAM_E ≤ CAM_{E′}. Empirical (searched) CAM estimates are lower bounds, so an
observed violation of monotonicity is a *search-shortfall diagnostic*, not a contradiction. ∎

This is the same shape as V-information's monotonicity in the predictive family V — read CAM
as a **ceiling-normalized, search-bounded usable-information statement with E playing V's
role**. "Is this criterion coded or articulated?" becomes a well-posed quantitative question:

  **seam(criterion) = the first lattice level E at which r̃_E enters its ceiling band**
  — a first-passage object, not a type.

Humor a153 (cross-cultural translatability) is the textbook instance: r̃_{E₁} ≈ 0 (raw ρ
*negative* — the specified proxy inverts), r̃_{E₂} = .62 — the seam sits exactly between the
computation-op level and two borrowed fields. Conversely PR's mechanical band saturates at E₁.
Evidence ops shift the *information bound itself* (I(M; X, Z) ≥ I(M; X)): they don't just
climb the lattice, they move what any level can reach — which is why the op-type diagnoses in
the seam tables (evidence-starved / computation / evidence-dominant) are coordinates, not
excuses.

**1 − CAM, re-read.** The uncertified residual decomposes conceptually into (i) search
shortfall at the current level, (ii) executor shortfall — mass reachable only at higher E,
(iii) genuinely enculturation-only mass. Only (iii) is "taste," and the lattice makes the
one-sidedness precise: you can push (i) down with more search and (ii) down by widening E,
but no finite experiment certifies that what remains is (iii). "Belongs as code" is provable;
"belongs as prompt" is only ever *not-yet-compiled at level E* — now with E explicit.

---

## 6. What this buys the paper

1. **Certificates get coordinates.** A gate-certified r̃ should be read as
   (r̃, transport class, borrowed-payload budget, op types used). Two channels with equal ρ
   are not epistemically equal if one transports across all conforming executors with σ ≈ 0
   and the other is bound to a checkpoint.
2. **LLM-at-temp-0 is not a counterexample to the dichotomy — it is its explanation.** Frozen
   weights make articulation *rigid* without making it *specified*: binding solved, provenance
   untouched. The certificate is real and artifact-bound.
3. **The hybrid program contract is a theory object**: bounded borrowing of enculturated
   payload inside a rigid binding, with ablation-measurable marginals. R4's
   construct-replacement failures (and humor's a297/a90/a306 regressions) are what
   *unbounded* borrowing looks like: replacing a specified construct with a vague
   enculturated pointer loses to a sharp selected/specified correlate.
4. **Anthropological framing, sharpened**: the enculturated rung is why this is a study of
   human preference articulability at all — the LM is a frozen informant; CAM_E measures how
   much of the community's norm can be re-housed in specified+selected form at executor
   level E, and the survival curves are the ethnography.

## 7. Measurable predictions (mostly already in flight)

- **Transport test (new, cheap):** re-extract a certified task's LLM fields with a different
  family (Llama instead of Gemma), same programs; certified r̃ should degrade in proportion to
  each program's field weight. This turns "artifact-boundness" into a number per criterion.
- **σ from behavior:** recon-R should order channels code > code+selected > hybrid at matched
  ρ (re-articulability tracks specification residual inversely). Partially visible already in
  the recon sweep medians.
- **Refit stability for selected ops:** re-fit TF-IDF/CE payloads on split halves; verdict
  agreement bounds the ostensive stability of build-time-selected meaning (mirrors split-half
  OPT_Ω stability, which ran ρ .70–.96).
- **Evidence-op marginals:** pr_exec (done) and patents_pa (in flight) Null-ablations put a
  certified number on call-time vs build-time world-state binding for evidence-dominant
  criteria.

*Discipline notes: all quantities here are reconstruction-side (judge-relative), never
label-aware; r̃ readings inherit the Gap-3 Spearman rule (Pearson companion near ceiling);
CAM claims remain one-sided lower bounds under search.*

**Measured worked example (2026-07-04, patents_pa):** the lattice claim has a converse the
prior-art experiment demonstrates. Evidence ops widen the *channel's* executor class, but the
*target* M̄_E(x) is a function of X alone, so I(M̄(X); Z | X) = 0: no op that binds
X-orthogonal world state can improve reconstruction of a doc-only judge. Measured op-marginals:
a26 P=.03 (the op actively hurts), a60 P=.24, a35 P=.96 only by attenuating an anticorrelated
text arm toward zero. Corollary for experiment design: evidence-op value is well-posed only
against targets at the same lattice level (a Z-aware judge M̄(x, Z), or M*) — channels and
targets must be level-matched, else DPI forces the null.
