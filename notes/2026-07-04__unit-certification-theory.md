# Certified Unit Framework (CUF): a mathematical specification of Ω-elements
**2026-07-04** · status: v1 formal spec (plan-approved) · implements: `unit_certificate.py`
· supersedes the informal gloss "minimum span of text that changes the behavior of an outcome"

## 0. Position in the theory

The prompt-optimality doc already commits to a behavioral ontology: an element of Ω is a *behavioral
partition operator* (PO :746-752), its identity a *species* under the executor's form-quotient
(two-faces §BRIDGE: "a criterion string is an *address*; the species is the *function* the executor
instantiates"; PO :2065-2069). What has been missing is **certification**: unithood today is assumed at
construction (mining + CMI screening) and never tested. The census's own validity analysis singles out
**merge-precision as "the binding validity gate"** (PO :2091-2099, over-merge is the one
anti-conservative direction) — precisely the object this framework estimates instead of assumes.

Design lineage (house idiom): temp-0 forced-logprob readout (`vllm_backend.score_binary`); Φ-orbit
averaging (`orbit_metric_verdict`); noise floor τ₀ (`noise_floor_tau`); band-not-gate fragility charges
(the ε_form redesign that retired FORM-DOMINATED); adverse-end reporting; Bonferroni with
n_perm ≥ max(999, m/α) (the ctree lesson); planted-control shakedowns; split-half reliability of the
certificate itself.

User-fixed decisions (2026-07-04): **(D1)** U2/U3 fragility = bits/effect charges ε_id, ε_ctx (band
mode), categorical verdicts only at extremes; **(D2)** pilot runs the full Llama 3B/8B/70B ladder;
**(D3)** BOTH effect targets are certified — target-free δ^free and metric-relative δ^M — with the
explicit prediction to test that the M-arm certifies more readily; units are executor-specific
everywhere.

---

## 1. The declared tuple

Every unit claim is a functional of an explicitly declared tuple

**𝔗 = (E, d, P_X, Φ, Λ, 𝒞, 𝒩, Π, α, thresholds)**

— executor, decoding, probe measure, form measure, position measure, company measure, null ensemble,
partition functional, test level. The three user confounds are *components of the definition*:
LLM inconsistency → (d, ξ, Φ); executor difference → E-indexing (Def 1, 9); position/context → (Λ, 𝒞).
Changing any component changes the unit inventory — that is a feature (executor-relativity is doctrine),
and the tuple makes it explicit rather than implicit.

## 2. Definitions

**Def 1 (Executor and readout; stochasticity source i).**
An executor is E = (weights θ, decoding d, template family T). The readout r_E : Σ* → [0,1] is
P(YES)/(P(YES)+P(NO)) from a 1-token forced-logprob call at temperature 0 — deterministic given θ.
Observed replicate variation (batching nondeterminism; Monte-Carlo error if d samples) is modeled as
additive noise ξ with E[ξ] = 0 and scale estimated by R re-render replicates:
τ₀ = c · mean_i std_r[r_E^{(r)}(q_i)] (existing `noise_floor_tau`, c = 2.5).
*Remark (temp > 0):* if d samples, define r_E(q) = E_d[verdict] and estimate by s samples; ξ then
includes sampling variance and all definitions below are unchanged — stochastic decoding is absorbed
into the noise model, never into the unit's identity.

**Def 2 (Probe measure; stochasticity source ii).**
X₁,…,X_n i.i.d. ∼ P_X, frozen by hash-split (never reshuffled). The **behavior** of a prompt h is
σ_E(h) = (r_E(render(h, x_i)))_{i≤n} ∈ [0,1]^n, the empirical version of the P_X-functional
σ*_E(h) = law of r_E(render(h, X)). Since every coordinate is bounded in [0,1], any Lipschitz statistic
of σ_E concentrates at rate O(√(log(1/δ)/n)) (Hoeffding/McDiarmid); split-half reliability of each
downstream statistic is reported as the empirical check on this.

**Def 3 (Variance measures; confounds 1 and 3 as measures).**
Declare three probability measures:
- **Φ** on meaning-preserving host transformations (paraphrase, clause/template reorder, boilerplate) —
  "natural variance in prompting";
- **Λ** on insertion slots for the address (begin / interior slots / end) — "where the unit falls";
- **𝒞** on co-present address subsets (which sibling units accompany it).
The **context measure** is H = Φ ⊗ Λ ⊗ 𝒞. All unit-level quantities are H-expectations, and because H
is sampled factorially, the variance of the effect decomposes identifiably into form-, position- and
company-components (Prop 1) — this is what licenses statements like "this unit's effect is
position-stable but company-sensitive."

**Def 4 (Address, installation, ablation; the mechanical-artifact control).**
An **address** a is a node of the segmentation lattice L(h) of host h (level 0 = host; level 1 =
items/sentences; level 2 = clauses; deterministic segmentation). The **installation map**
ι_{φ,λ,C}(a) renders host φ(h) with a placed at slot λ among company C. The **ablated twin**
ι_{φ,λ,C}(∅) is identical except a is replaced by inert filler f with |f| ≈ |a| (length-matched
neutral-replace). Filler inertness is a certified side condition: δ^free(f) ≤ τ₀^abl (placebo gate).
Length and position are thereby held fixed across the (with, without) pair — ablation effects cannot be
mechanical artifacts of prompt length or token-position shift. `delete` and `paraphrase_swap` are
secondary operators (robustness check; identity probe respectively).

**Def 5 (Fingerprint; two effect functionals).**
The **ablation fingerprint** of a in h is the H-averaged signed behavior shift
  φ_E(a) := E_{(φ,λ,C)∼H}[ σ_E(ι_{φ,λ,C}(a)) − σ_E(ι_{φ,λ,C}(∅)) ] ∈ ℝ^n.
It records *which probes moved and in which direction* — the unit's behavioral signature.
Two effect scalars, one per certificate arm:
- **target-free:** δ^free(a) = ‖φ_E(a)‖₁ / n — does the address move E's behavior at all?
- **metric-relative:** δ^M(a) = E_H[ ρ(σ_E(ι(a)), m̄_ω) − ρ(σ_E(ι(∅)), m̄_ω) ], where m̄_ω is the
  metric's Φ-averaged own verdict (existing target) and ρ = Pearson correlation over probes — does the
  address move the host's alignment *toward the metric*? Signed and directional.
Auxiliary statistics for both arms: context dispersion Var_H(δ), sign-stability
s(a) = E_H[𝟙{per-draw fingerprint has same dominant sign pattern as φ_E(a)}], dispersion ratio
κ(a) = Var_H(δ) / δ̄².

**Def 6 (Null ensemble 𝒩; detectability, both arms).**
𝒩 is the distribution of (fingerprint, δ^free, δ^M) under **certified-inert edits** sampled under the
same H: filler↔filler swaps, function-word synonym edits, whitespace/punctuation edits, re-render
replicates. For a lattice with m tested nodes at level α:
  a is **U1-free detectable** iff P_𝒩[‖φ_null‖₁ ≥ ‖φ̂(a)‖₁] ≤ α/m,
  a is **U1-M detectable** iff P_𝒩[δ^M_null ≥ δ̂^M(a)] ≤ α/m (one-sided toward the metric),
with permutation/null-sample count n_null ≥ max(999, m/α) (Bonferroni; house standard). "Changes the
behavior of an outcome" is thereby replaced by two calibrated hypothesis tests.

**Def 7 (Identity; species; non-transitivity honesty clause).**
Similarity kernel ρ_id(a,b) = corr(φ_E(a), φ_E(b)) (computed on the shared probe set; both fingerprints
from their own hosts/contexts). Thresholded similarity is not transitive, so "same unit" is **defined
relative to a declared partition functional Π** applied to the ρ_id-matrix — v1: single-linkage at r*,
with the option of the judge-merge/behavior-split `quotient_species` refinement. The threshold
r*(metric, E) is *calibrated, not chosen*: the q-quantile (q = 0.05) of within-paraphrase-orbit
self-similarities {ρ_id(a, para(a))} — two addresses are the same species when they cohere at least as
well as a single address coheres with its own paraphrases. Π ships with measured **merge-precision and
split-recall** on planted synthetics and orbit-holdout (the census's binding validity gate, estimated).

**Def 8 (Unit; atomicity; interaction).**
A **unit** is a pair u = ([a]_Π, cert(u)): a species together with its certificate record, such that
U1 (either arm — the arm is part of the record), U2 and U3 pass at declared thresholds/charges.
On the sub-address lattice, a detectable a is an **ATOM** iff its proper parts b₁,…,b_k either
(i) are individually non-detectable, or (ii) fail additive reconstruction:
  ‖φ(a) − Σ_j φ(b_j)‖₁ / ‖φ(a)‖₁ > η.
It is **COMPOSITE(parts)** iff the parts detect and reconstruct within η — then the parts, not a, enter
Ω. Deviation from additive reconstruction is the *interaction* of the parts; at the pool level the
pairwise interaction index I(X_i;X_j|M) − I(X_i;X_j) (PO :796-803) is the information-theoretic
counterpart. This operationalizes the "green and round" doctrine (one signal ⇒ one unit) as a testable
outcome rather than an assumption.

**Def 9 (Executor scope).**
Units are indexed u^{(E)}; there is no executor-free unit. Cross-executor identity is tested with the
same kernel ρ_id computed on the shared probe set between fingerprints under E and E′, within one model
family (Llama 3B/8B/70B; Gemma/Qwen are replication panels, never pooled). Scope verdicts:
**E-SHARED** (identity holds along the ladder), **E-SPECIFIC(E)** (detectable only at E),
**E-EMERGENT(≥E)** (detectable from rung E upward).

**The superseding statement.** *"Minimum span of text that changes the behavior of an outcome" →
"minimal ATOM under Def 8, certified U1–U3 at level α with charges (ε_id, ε_ctx), at declared tuple 𝔗."*
Every term now has a measure, an estimator, and an error bar.

## 3. Certificates U1–U5 and the decision rule

| cert | claim | test / output |
|---|---|---|
| U1 | detectable above noise | Def 6, both arms; Bonferroni over lattice; outputs p_bonf per arm |
| U2 | form-stable identity | self-similarity r_self = median ρ_id(a, para(a)); charge **ε_id = (1 − max(r_self,0)) · δ̄** ; extreme (r_self < 0 or below planted-null band) ⇒ FORM-FRAGILE |
| U3 | context-robust | sign-stability s(a) ≥ 1−β and κ ≤ κ*; charge **ε_ctx = κ̂ · δ̄** capped at δ̄; sign flip across H-factor ⇒ CONTEXT-CONDITIONAL(factor) with the factor named by the Prop-1 variance decomposition |
| U4 | minimal granularity | ATOM / COMPOSITE(parts) per Def 8 with reconstruction deficiency η |
| U5 | executor scope | E-SHARED / E-SPECIFIC / E-EMERGENT along the declared ladder |

**decide_unit (band mode, default):** verdict ∈ {CERTIFIED-UNIT, COMPOSITE(parts), SUBTHRESHOLD,
UNDERSAMPLED, FORM-FRAGILE, CONTEXT-CONDITIONAL(factor)} × scope tag; certified effect is reported as
the interval **[δ̄ − ε_id − ε_ctx − CI, δ̄ + CI]** (adverse end first, house convention). UNDERSAMPLED
fires when the CI half-width exceeds δ̄ (cannot resolve at n, n_ctx) — mirroring the census lesson that
fragility cliffs often mask undersampling.

Charges are stated in effect units (ΔP); when a certified unit enters the value pipeline its charges
convert to bits alongside v(s|S_g) (the value_certificate integration is post-pilot work). Certification
never reads value — valuation and unithood remain separable.

## 4. Propositions (statements; proofs sketched, full versions as needed)

**Prop 1 (Identifiability of variance components).** Under factorial sampling from H = Φ ⊗ Λ ⊗ 𝒞 with
n_φ × n_λ × n_c draws, the ANOVA decomposition of δ(a; φ, λ, C) into main effects and residual is
identifiable, with each component's estimator concentrating at O(√(log(1/δ)/(n·n_ctx))) (bounded
responses). *Consequence:* CONTEXT-CONDITIONAL verdicts can name their conditioning factor.

**Prop 2 (FWER control).** With n_null ≥ max(999, m/α) exchangeable null draws and the α/m rule, the
probability that any truly-inert lattice node is declared detectable is ≤ α (standard permutation-test
validity + Bonferroni). *Empirical check:* pure-null synthetic lattices in the CPU test suite.

**Prop 3 (Asymmetric robustness, inherited).** Over-splitting (one species counted as two) inflates the
unit count but adds ≈0 to any conditional-gain head (duplicate gains ≈ 0); over-merging hides genuine
units — the anti-conservative direction. Hence Π's *merge-precision* is the audited quantity (Def 7),
and single-linkage at a calibrated-low r* errs toward over-splitting by construction.

**Prop 4 (Arm nesting).** If per-probe scores are bounded and sd(m̄_ω) ≥ s_min > 0 and sd(σ) ≥ s_min on
the probe set, then |δ^M(a)| ≤ L · δ^free(a) with L = L(n, s_min) an explicit constant (perturbation
bound for Pearson correlation under an ℓ₁-bounded shift of one argument). *Consequence:* U1-M certified
⊂ U1-free detectable — the metric-relative arm cannot manufacture units invisible to the target-free
arm; it can only *select among* them (directionally, hence typically at better SNR — the D3 conjecture
the pilot tests).

## 5. Relations to existing objects

- **§12.6 quotient / B_E census:** Def 7's Π on ablation fingerprints is a *certified* refinement of
  `conditional_species`/`quotient_species` on criterion signatures; its measured merge-precision is the
  census's binding validity gate. Certified units are the census species with error bars.
- **OPT_Ω / gains:** unchanged — but post-pilot, Ω can be *restricted to certified units*
  (`--quotient certified`), giving OPT over a certified basis and a cleaner ε story.
- **two-faces span_R2:** the species basis used in span_R2 regressions becomes the certified basis;
  "new species" claims (low span_R2 + high recovery) inherit U1's null calibration.
- **ε_form:** ε_id/ε_ctx are its unit-level siblings; all three are non-negative charges that widen
  intervals rather than categorical cliffs.

## 6. Pilot readouts (v1, CW#24 + CW#29 × {description, GEPA-final, checklist} hosts × 3B/8B/70B)

1. Verdict distribution per host type and per arm (free vs M) — including the D3 conjecture test:
   fraction certified under U1-M vs U1-free at matched α.
2. Cross-host unit identity: does the "same" span in description vs GEPA prompt vs checklist certify as
   the same species (Def 7 across hosts)? The strongest practical form of confound (3).
3. Ladder typing: E-SHARED / E-SPECIFIC / E-EMERGENT inventory across 3B/8B/70B.
4. Trust gates (must pass before 1–3 are read): planted positive certifies; placebo lands SUBTHRESHOLD;
   CPU FWER calibration ≤ α; split-half verdict flip-rate.
