# The two faces of articulability — Face 1 (census ceiling) and Face 2 (decompression), unified

*Written 2026-07-02. Purpose: give the Face-1/Face-2 framing a single formal home. Until now
"Face 1 / Face 2" appeared only as LABELS in the operational docs (cw-unified-grid-roadmap,
cw-grid-v1-results, r3cw-data-catalog); the machinery of Face 1 is fully formalized in
`2026-06-18__prompt-optimality-theory.md` (+ the whitepaper) but never under that name, and Face 2
had NO formal setup anywhere — only the grid spec and the conceptual content worked out in
session. This doc defines both, states the estimands, and — the load-bearing part — connects Face 2
to the Face-1 census through the species basis. Proofs for Face 1 live in the theory doc; this doc
does not reproduce them.*

## 0. One object, two directions

The quantity of interest is the **articulability of a concept `M` to an executor class `E`**: how
much of a human community's preference `M` (a target over texts `X`) can be transmitted through an
*articulated* channel that `E` reads. `E` (a model family) is the standardized instrument; the
subject is the human preference and the language it does or does not fit into.

We read this one quantity from two directions.

- **Face 1 — the census ceiling (supremum over articulations).** Fix the *reader* at full strength;
  search the *space of articulations* for the best one and certify how much better any unarticulated-
  yet criterion could do. Deliverable: an upper bound `OPT_Ω + ε` on what ANY checklist can transmit
  to `E`, and the certified residual below the behavioral floor. Answers **"what is the most a
  telling can carry?"**
- **Face 2 — decompression (shape over message richness).** Fix a small ladder of *message types* of
  increasing richness for the SAME concept; vary the *reader* strength; measure recovery at each
  rung. Deliverable: a curve, not a point — where recovery jumps, where it plateaus, who closes the
  gap. Answers **"which KINDS of telling carry this concept, and to whom?"**

Face 1 collapses the message axis (it takes a sup) to isolate the ceiling; Face 2 collapses the
optimization (it uses designated, interpretable points) to isolate the *kind* of knowledge. Two
concepts with the same Face-1 ceiling can have completely different Face-2 shapes — that difference
is what "tacit" vs "codifiable-but-verbose" vs "indexical" names.

## 1. Face 1 — formal recap (see theory doc for proofs)

Per metric `M_i`, over a held-out split, with `E` fixed at full strength:

    R(p) ≤ T(m_ω) ≤ I(M_i*; X) ≤ U_{B_E}          [the bracket]

- `R` = recovery of a prompt `p`; `T(m_ω) = I_f(M_ω; X)` transmission of the orbit-averaged readout
  (Fact 2 / DPI: `R ≤ T`; Fact 1: `T` convex in behavior). `T` is a **floor on the ideal** `I(M_i*;X)`
  — the operationalized readout can only under-transmit the ideal metric.
- `OPT_Ω := sup_{p∈Ω} R(p)` over the articulated-criteria pool `Ω`; the **reachable ceiling** is
  `OPT_Ω + ε`, where `ε` is a capture–recapture / missing-mass (Good–Toulmin/OSW) bound on the value
  that criteria NOT yet in `Ω` could still add, made anytime-valid by Theorem T1 (§12.8). This is the
  Face-1 number: *no checklist of this kind exceeds `OPT_Ω + ε`, at stated confidence.*
- **Certified residual (the tacitness read):** `Δ(E) := lowerCI(C_dense) − [OPT_Ω + ε]`, where
  `C_dense` is a dense model trained directly on the metric's OWN verdicts `(x, M̄_E(x))` — a
  reconstruction ceiling (proof the signal is in the judgment behavior). **Doctrine (user-affirmed
  2026-07-03): metrics are reconstruction-aware ONLY.** No quantity in either Face is anchored to
  task outcome labels; wherever this doc says "labels" it means the metric's own verdicts.
  `Δ(E) > 0` = value demonstrably present that no articulation in the census transmits to `E`.
- **Scaling reading (defends `Δ` against "weak instrument"):** recompute the bracket along a
  same-family ladder. `Δ` shrinks with scale ⇒ instrument weakness; `Δ` flat while `C` high ⇒ the
  words are the bottleneck (tacit relative to `E`). Human-target extension:
  `A_H ≥ lowerCI(C) − [OPT_Ω + ε]`.

Verdict cells (per metric): CODIFIABLE / DEEP / UNDERSAMPLED (`ε` too wide) / FORM-DOMINATED (readout
not stable under paraphrase — the gate the form-effects work is about).

> **Gate redesign adopted + IMPLEMENTED (2026-07-03, user delegated the call).** FORM-DOMINATED as
> a *verdict* (fixed 10% median-flip bar) is retired: the bar was arbitrary and binary, and it ate
> 36/43 CW + 36/60 humor metrics out of the residual analysis. Two decoupled pieces:
> 1. **Categorical verdict** keeps its census meaning — `alpha_probe.decide(..., form_mode='band')`
>    no longer preempts with FORM-DOMINATED; the metric gets its CODIFIABLE/DEEP/UNDERSAMPLED
>    verdict and the form boolean survives as a reported diagnostic. (`form_mode='verdict'` still
>    default, so the cert-builder path is byte-unchanged; 30/30 tests pass.)
> 2. **Residual** widens the ceiling: `Δ(E) = lowerCI(C_dense) − [OPT_Ω + ε + ε_form]`, with
>    `ε_form := max_form OPT_Ω(form) − OPT_Ω(canonical)` (exact, needs a per-form re-census — GPU,
>    rides with the forminv passes). Until that lands, `alpha_probe.eps_form_bits` returns a
>    clearly-labeled CPU proxy `q·OPT_Ω` (`source ∈ {exact, proxy, none}` so a proxy is never read
>    as the certified number). ε_form is deliberately kept OUT of the CODIFIABLE gate — form
>    stability is an orthogonal axis, not a census quantity.
>
> **Honest re-read finding** (`notebooks/data/two_faces_20260702/band_mode_reread.json`): when not
> ejected, 34/36 CW and 33/36 humor ex-FORM-DOMINATED metrics land in **UNDERSAMPLED**, only 2–3
> per domain are genuinely CODIFIABLE-but-form-fragile. So the form cliff was mostly *masking
> undersampling* at n=300, not certifying a distinct "form-dominated" population — the real
> bottleneck is probe count. (ε_form source='none' on Day-0 certs: they saved only the boolean, not
> the flip rate; band verdicts above use ε_form=0, the most generous-to-CODIFIABLE case.)

## 2. Face 2 — formal setup (new)

### 2.1 The rung set (articulation TYPES, not lengths)

For concept `M_i`, a writer `W` produces messages `μ_r` at rungs `r` of increasing richness — TYPES
of telling, not token budgets (length was tested and is a poor predictor; length is a nuisance
covariate, and `definition`/`explanation` are length-matched):

    name → definition → explanation → full_rubric → exemplars → dossier

`name` = the concept label (a pure index); `definition` = intension; `explanation` =
mechanism/recognition; `full_rubric` = the Face-1 anchor channel; `exemplars` = k held-out labeled
instances SHOWN, no words; `dossier` = definition+explanation+exemplars (telling+showing). The rung
set is a registry — new axes (contrast/near-miss pairs, worked exemplars, program form, emic
phrasing, dialogic) enter as new rungs without changing the structure.

### 2.2 The estimands

For reader `E` (a specific executor on the family ladder) and rung `r`, let `μ_r` be orbit-averaged
over form reformulations (so the message is Φ-invariant by construction — form control is a
PREREQUISITE, not a nicety: uncontrolled, a rung's recovery confounds transmission with paraphrase
noise, ~12% flips on CW). Define

    R_E(r) := recovery of the executor-indexed target M under message μ_r    [decompression profile]
    G(r)   := R_{strong}(r) − R_{weak}(r)                                   [reader gap at rung r]

The DELIVERABLE is the profile `{R_E(r)}_r` per reader and the gap `{G(r)}_r`, not a scalar.

**⚠ AMENDED 2026-07-03 (user-flagged; the sentence that stood here — "cross-reader claims use
R_E(r) against the external reference" — was a 2026-07-02 drift written alongside the grid driver
and is NOT derivable from the original theory, which is anchor-free within-executor throughout).**
The estimand is EXECUTOR-CONSISTENT: each reader's target is ITS OWN orbit-averaged rubric readout
`M̄_E` (the metric's own verdict by that reader — the same anchor-free reconstruction-target
doctrine as the census), so `R_E(r) := i_binary(M̄_E, verdicts_E(μ_r))` in bits, with a per-reader
degeneracy gate (`H_self ≥ 0.15`). Cross-reader claims compare self-decompression PROFILES and the
`H_self` ladder (judgment richness/stability) — mirroring how the Face-1 scaling ladder compares
self-contained per-tier brackets, never one model scored against another's readout. Scoring reader
E′ against a FIXED other-executor judgment is a legitimate but DIFFERENT quantity —
**cross-executor transmission (anchored)** — which requires an explicit, signed-off anchor choice
and must never be reported as "decompression".

**Operationalization pin (added 2026-07-03 after a user-flagged misalignment).** `R_E(r)` MUST be
measured in Shannon bits with the census's own functional, or the bracket and the §3 ostension test
are dimensionally broken: for a single rung judge the census's atomic estimator is `i_binary` — the
exact closed-form MI between binarized readouts (`M̄>0.5`, signature`>0.5`, NaN→0.5, exemplar
probes masked) — the same functional as the census's per-species additive values `v_add`. So
`R_bits(E, r) := i_binary(M̄, verdicts_E(μ_r))`, directly comparable to `H_M`, `OPT_Ω`, `ε`.
Balanced accuracy is a SECONDARY behavioral readout only (chance = .5 is interpretable; never place
it in the bracket). The v1 grid initially reported bal_acc as primary — corrected; first bits
results: cross-reader single-message recovery is ~0.03–0.10 bits of ~0.5–0.9-bit targets (≈5–15% of
H_M), far below census heads (OPT_Ω ≈ 0.7·H_M), and the ostension test returns **zero certified
exceedances** anywhere (all rungs deep in-span — no certified census violation in CW or humor v1).
**Resolved 2026-07-03 (see the §2.2 amendment above): the anchor question dissolves.** The primary
Face-2 estimand is executor-consistent (each reader vs its OWN `M̄_E`; `self_bits`/`H_self` now
emitted by the grid report), so no external anchor is needed and D2's self-referentiality problem
disappears (every reader, including the 8B, is a valid subject of its own profile). The
8B-anchored numbers throughout the v1 analyses are retained under their correct name —
cross-executor transmission (anchored) — a secondary instrument whose anchor choice, if it is ever
promoted, needs explicit sign-off. First executor-consistent results (notebook §3d): capacity
lives in the `H_self` ladder (CW medians 0.50→0.81→0.83→0.92 bits for 1B→3B→8B→70B; on humor the
3B holds a stable own-judgment on only 34/60 metrics), and self-recovery fraction alone is NOT a
capacity measure (the 1B reconstructs 65% of its own low-entropy CW judgment from the bare name —
simple judgments are trivially self-recoverable; the validity gates carry that weight).

### 2.3 What each rung-to-rung jump identifies

| transition | gain measures | tradition |
|---|---|---|
| name → definition | lexical / indexical content (does the label under-determine the intension?) | Frege sense/reference |
| definition → explanation | knowing-THAT → knowing-HOW (mechanism a definition omits) | Ryle |
| explanation → exemplars | OSTENSIVE content (what demonstration carries that words don't) | Polanyi, Wittgenstein |
| exemplars/dossier plateau **below** the behavioral floor, census saturated | TACIT-WITHIN-FRAME (L4) | Collins STK |

The **strong−weak gap at `name`** measures the reader's enculturated STOCK: a strong reader nailing
a two-word name is a measurement of its prior, not of the message's information (a short message
INDEXES rather than CARRIES; MDL: message = pointer into prior + residual specification).

### 2.4 The horizontal reading — iso-performance expansion cost (added 2026-07-02 PM)

Vertical gaps `G(r)` are confounded by reader main effects (a weaker reader may be generically
worse at using ANY instruction). The horizontal quantity is the clean one.

> **Anchor resolution (2026-07-03, user-approved).** Iso-performance intrinsically requires a
> common yardstick — matching across readers is undefined without one — so this section is the ONE
> place an external reference is licensed. The adopted anchor is **family-top**: the reference
> judgment is the LARGEST same-family member's own orbit `M̄`, fixed a priori (never data-chosen),
> within-family only (per [[same-family scaling]]). This readout is labeled *family-top
> transmission*; the PRIMARY decompression estimand stays executor-consistent (§2.2). Second
> families (e.g. Gemma-4) replicate with their own family-top — panels, never pooled.

**Definition (expansion cost).** Fix a concept `M_i`, a fixed external reference judgment, and a
NESTED message chain `p = p_0 ⊑ p_1 ⊑ … ⊑ p_L` (each level strictly appends content, so the
expansion dial is a scalar and composition is well-defined). For readers `B` (weaker) and `A`
(stronger), define

    x*(B→A, p_ℓ) := min { ℓ′ : R_B(p_{ℓ′}) ≥ R_A(p_ℓ) − δ }     (right-censored at L if none)

— the minimal expansion level at which `B`'s recovery from the richer message matches `A`'s
recovery from the poorer one, both reconstructing the SAME fixed reference, at δ-tolerance. Write
`d(B→A) := x*(B→A, p_0)` for the name-level cost, and `h_{B→A}(ℓ) := x*(B→A, p_ℓ)` for the full
level-shift function.

**Triangle inequality.** With exact matching (δ=0) and nested chains, the composed path is always
*feasible* for the direct problem: if `B` matches `A@p_0` at level `h_{B→A}(0)`, and `C` matches
`B` at that level, then `C` has reached `A@p_0`'s performance. Minimality of the direct cost gives

    d(C→A) ≤ h_{C→B}(h_{B→A}(0))        [composition can only overshoot]

Under noisy δ-matching the inequality picks up δ-accumulation per hop; the estimator therefore
uses paired bootstrap match-probabilities and censoring-aware Kaplan–Meier medians rather than
point matches.

**Transitivity = tightness = potential structure.** "Transitivity" is this inequality holding with
equality along the capacity ladder. If tight, articulation debt behaves like a **potential**: there
exists a scalar background-knowledge level `K_E` per reader such that `d(B→A)` depends only on the
gap `K_A − K_B` — message richness is a currency with a fixed exchange rate against capacity, and
the concept's knowledge peels off in uniform layers. Strict sub-additivity (slack > 0) localizes
**non-nested background**: what `C` lacks relative to `B` is a different KIND of prerequisite than
what `B` lacks relative to `A` — the composed path carries `B`-specific scaffolding useless to `C`.
The slack is not a nuisance statistic; its location maps the prerequisite structure of the concept,
and the increment TYPES spanning the gap name it.

**Relation to the bridge (§3).** In species terms: additivity is what a FLAT bag-of-species concept
predicts (each hop supplies the missing species); non-nested slack is the signature of HIERARCHICAL
packing (procedures presuppose mechanisms presuppose vocabulary). The span_R2 classifier applies
per level: an expansion that rescues a weak reader while its induced judge stays in-span is better
addressing; one that rescues while out-of-span is building new species capacity in the reader.

**Controls.** Planted mechanically-checkable items (programmatic gold) measure each reader's
instruction-following floor separately from knowledge content — "the small model just can't follow
instructions" is measured, not assumed away. A reversed-schedule arm controls type-position
confounding in the marginal-gain-by-type matrix. The writer remains label-blind (telling, not
fitting, per §3).

**First horizontal read (v1 grid, coarse type-rungs as the dial, 3B-vs-1B):** humor — expansion
substitutes for capacity (cumulative 1B-matches-3B@name reaches 62% by dossier; 38% right-censored);
creative writing — a capacity floor (74% censored at every verbal rung; 1B best-over-rungs median
.579 never reaches 3B@name .634). Implementation: `methods/codability/run_expansion_chain.py`
(nested chains, δ-grid + bootstrap matching, KM censoring, planted floors, type-tagged marginal
gains, triangle test; selection = ~10 rescued + 10 censored metrics per domain).

## 3. The bridge — why Face 2 is interpretable through the Face-1 census

This is the theoretical content that makes the two faces one project rather than two experiments.

**Units are species, not texts.** Elements of `Ω` are criteria (articulated prompts); their
behavioral equivalence classes under the executor are **species** (the form-quotient merges
paraphrases). `B_E` counts species; `OPT_Ω` is the best head assembled from them. A criterion string
is an *address*; the species is the *function* the executor instantiates.

**A message selects and combines species.** A decompression rung `μ_r` is a single address of
increasing richness aimed at making `E` instantiate the intended function for `M_i`. So a rung gain
is classified — not assumed — by WHERE the induced judge lands relative to the census basis:

    span_R2(r) := CV-R² of regressing the rung-r judge signature on the metric's species basis

- **HIGH span_R2** ⇒ the rung is an ASSEMBLY of census-known species — "better addressing," more
  units activated, no new inventory. (The coefficients literally name which subconcepts the message
  switched on.)
- **LOW span_R2 WITH high recovery** ⇒ the rung carries predictive value NO combination of census
  species explains — a genuinely new unit; content the words did not carry.

**Ostension formalized as a certified census violation.** The sharp Face-1↔Face-2 link:
`OPT_Ω + ε` is the certified ceiling of the WORD channel. A rung that EXCEEDS `OPT_Ω + ε` — with the
census saturated (adv_saturated) and controls green — is certified out-of-span: it transmitted a
distinction that no articulated criterion in `Ω` induces. That is the formal cash value of
"ostensive": extensional specification pinning a function with no reachable intensional description
in `E`'s language. Polanyi as a census violation. (Symmetrically, a rung gain that stays UNDER
`OPT_Ω + ε` is re-addressing of in-span content — a better pointer, not new knowledge.)

**Telling vs fitting (why this is not GEPA).** The Face-2 writer sees only the concept materials
(name + rubric), never probe labels; the exemplar rung carries exactly k held-out labeled instances,
so its information content is measured and small. Thus `R_E(r)` measures **what an act of telling
transmits**, not what a prompt slot can be made to encode under supervision. GEPA optimizes `R`
against labels — it estimates a sup (a lower bound on channel capacity) and its gains are confounded
across type/phrasing/length/content and can harvest the ~52%-calibratable form-shift. The grid holds
type fixed, controls form, and reads a SHAPE. (This is the same principle the census already
enforces: gepa-tagged criteria are excluded from the capture–recapture iid sets — optimizer output
is not an iid sample of the articulation distribution.) GEPA's legitimate roles: sharpen the Face-1
lower bound, ADVERSARIALLY probe the ceiling (beating `OPT_Ω + ε` with a saturated census breaks the
certificate), and optionally serve as a "best-fitted message" reference rung whose gap to the
best-TOLD rung (dossier) measures adaptation-vs-articulation.

## 4. Honest limits (carry on every slide)

- Everything is **executor-class-relative**: "tacit relative to `E`-family readers of English
  rubrics." The human-reader bracket `A_H` is the planned extension, not a current claim.
- `T` is a LOWER bound on the ideal, `B_E` an UPPER bound — they do NOT collapse into one ceiling.
- The census cannot certify its own SUPPORT completeness (an LM-proposed `Ω` can share blind spots —
  the planted tail-XOR breaker demonstrates exactly this). Certificates carry an explicit assumption
  ledger; they do not pretend otherwise.
- Face-2 exemplar rungs at small k and truncated excerpts UNDER-state ostension by construction (see
  v1 results); the k-curve and longer excerpts are the honest test, and the dossier must be rebuilt
  without the failing exemplar block before its plateau is read.

## 5. First empirical instance

Both domains complete 2026-07-02, AUDITED (D1 degenerate filter, D2 reference-executor exclusion —
earlier 8B-inclusive "+0.245 on rubric" RETRACTED as self-referential):

- **CW Face 1** (8B_v2 orbit + gate, 45 kept of 46): 38 FORM-DOMINATED / 5 UNDERSAMPLED /
  2 CODIFIABLE; form-gate pass 16%; fragility staircase ~flat 3B/8B/70B (calibrated ~6%).
- **Humor Face 1** (54 kept of 60): 35 FD / 15 US / 4 COD; form-gate pass **35%** — form fragility
  is DOMAIN-SPECIFIC (~2.2×), the words hold still for humor.
- **Face 2 clean gaps (3B−1B)**: CW name +.060 → definition/explanation +.133/+.139 → rubric +.121,
  exemplars −.006; humor +.045 → +.104/+.091 → rubric **+.019**, exemplars **+.091**. Shared
  signature: the gap opens at the index→content transition in both. k=2 exemplars transmit humor
  but not CW — ostension works where mechanics are visible in short excerpts.
- **70B ladder (real dynamic range, 8B=ref excluded)**: name-rung recovery climbs monotonically
  1B→3B→70B in BOTH domains (humor .523→.565→.676, CW .580→.635→.675) — the compressed-pointer /
  "strong reads short, weak needs unpacked" hypothesis CONFIRMED; 70B unpacking benefit tiny
  (+.046/+.005). CW SATURATES at 3B (70B−3B ≈0/neg on verbal rungs); humor stays graded to 70B.
  ⚠ Humor's full-rubric "collapse" (small 3B−1B gap) is a SMALL-READER artifact: the 70B reads
  humor's full rubric at .742 (highest cell) — the checklist is the richest channel, capacity-gated,
  NOT diluting. Two regimes: CW all-or-nothing at a low capacity bar; humor rewards capacity all the
  way up.
- **Horizontal read** (§2.4): humor rescued-by-expansion (62% cumulative), CW capacity-floored
  (74% censored) — the iso-performance expansion-chain experiment (`run_expansion_chain.py`) is the
  designed follow-up; first full run on the 1B/3B/70B ladder queued behind the 70B reader pass.

Full numbers + caveats: `2026-07-02__cw-grid-v1-results.md`, `2026-07-02__humor-vs-cw-crossdomain.md`,
notebook `notebooks/2026-07-02__two-faces-results-summary.ipynb`.

Related: `2026-06-18__prompt-optimality-theory.md` (§12.8 provable core, Face-1 proofs),
`2026-07-01__articulability-anthropology-reframe.md` (human-target framing),
`2026-07-01__cw-unified-grid-roadmap.md` (operational plan), `2026-07-02__cw-grid-v1-results.md`
(first numbers), [[decompression-rungs-are-types]], [[project_cw_grid_v1_results]].
