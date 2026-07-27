# Seam certificate lemmas — formal statements, proofs, and gaps

*2026-07-03. Draft mathematical appendix for the metric-seam proposal
(`notes/2026-07-01__metric-seam-proposal.md`, §5bis/§5ter). Formalizes the two lemmas stated
informally there: (A) the **headroom lemma** (§5.1, §5ter row 1 — `T(m_ω) − R` "survives
executor-agnostically") and (B) the **matroid-U₂ / tightening lemma** (§5.2–§5.3, §5bis S3–S4 —
the within-class bound extends over `Ω × {code, llm}` and gated migration confines the
uncertified residual to the LLM share). Each lemma is stated with every assumption explicit,
checked against what `methods/metric_seam/certificates.py` actually computes, and followed by a
numbered list of places where the informal §5bis claim is NOT derivable as stated. Descriptive
draft for review — nothing here is a claim about empirical adequacy, only about what the math
licenses. Notation follows the prompt-optimality theory note (PO, `2026-06-18__prompt-optimality-theory.md`).*

---

## 0. Notation and standing conventions

- `X` — held-out item content, `X ~ D_test`; all information quantities live on the **same**
  held-out split (PO §2.2 same-distribution caveat — load-bearing throughout).
- `m_ω` — the operationalized metric: criterion words + target executor + template + decoding
  parameters. **The target executor is part of `m_ω`'s identity**: the same criterion words run
  on a different executor is a *different* `m_ω` (PO §12.7, §5ter "executor-relativity"). This is
  why "executor-agnostic" below must be scoped to the *implementation/reconstruction* side.
- `M_ω = m_ω(X)` — the target verdict random variable on a held-out item (possibly stochastic:
  the LLM judge channel).
- `T(m_ω) := I_f(M_ω; X)` — transmission, an f-mutual-information (Shannon or TVD; §2.2/§3.1 of
  PO: never mix `f`'s across the two sides of an inequality).
- `M̂` — an *implementation's* verdict on the same item: a reconstruction/program/hybrid channel
  executed by some executor `E'` on some input channel `V` (which may be `X` alone, `(X, f(X))`
  after a computation op, or `(X, Z)` after an evidence op).
- `R := I_f(M_ω; M̂)` — recovery / reconstruction fidelity, held-out.
- `B_E` — behaviors realizable by executor `E` (PO §1); `B_code,L` — the formal program class at
  complexity ≤ L (§5bis).
- Judge-channel (seam) model: `M_p(i) = τ(i) + ε_p(i)` for pass `p`; `rel₁ = corr(M_1, M_2)`;
  Spearman–Brown `rel_K = K·rel₁ / (1 + (K−1)·rel₁)`; attenuation ceiling `√(rel_K · rel_f)`.
- `Ω = {c_1, …, c_n}` — the metric's criterion set (per-metric, PO §1); channels as in
  proposal §3.1.
- All population statements; estimation enters only where flagged. `κ` = agreement-type fidelity
  (accuracy vs a binary target); `ρ̂` = correlation-type fidelity; `ρ̃ = ρ̂/√rel_K` its
  ceiling-normalized form.

---

## 1. Lemma A — the headroom lemma (executor-agnostic ceiling)

### 1.1 What "executor-agnostic" must mean (and what it must not)

Two executors appear in any seam quantity: the **target-side** executor inside `m_ω`, and the
**implementation-side** executor `E'` that produces `M̂`. The informal §5.1 phrase "DPI is
executor-agnostic" is derivable only for the implementation side: `T(m_ω)` is a functional of the
joint law of `(M_ω, X)` alone, so it does not mention `E'` at all — *that* is the invariance.
Changing the target-side executor changes `M_ω`, hence changes `T` itself; nothing transfers
across that change. So the precise claim is:

> The **ceiling** `T(m_ω)` is invariant under implementation-executor change; the **achieved**
> `R(E')` and hence the headroom *number* `h(E') = T(m_ω) − R(E')` are executor-indexed. What
> transfers when the executor changes is the *bound* `R(E') ≤ T(m_ω)` (equivalently
> `h(E') ≥ 0`), not the previously measured headroom value.

### 1.2 Admissible implementations

**Definition (admissible implementation).** An implementation is a pair `(V, A)`: an input
channel `V` and a (possibly stochastic) map `M̂ = A(V, U)` with internal randomness `U`. It is
**admissible relative to `X`** if:

- **(C1) same-split:** `R` and `T` are both evaluated on the same held-out distribution `D_test`,
  disjoint from anything the implementation was fit on (PO §2.2: DPI does not chain across
  distributions; `T_train` never bounds held-out `R`).
- **(C2) no target leakage:** `M_ω ⟂ V | X` and `U ⟂ M_ω | (X, V)`. Jointly these give
  `M_ω ⟂ M̂ | X`. Concretely violated by: training/fitting the implementation on the held-out
  target verdicts; *reusing cached realizations* of the target judge's passes; sharing the
  target's sampling noise (same seed/batch). Note that a *fresh* draw from the same judge model
  is admissible — it is a new channel from `X`; the exact cached realization is not.
- **(C3) same `f`:** both sides of the inequality use the same f-divergence (Shannon with
  Shannon, TVD with TVD).

### 1.3 Lemma A1 (information form)

> **Lemma A1.** Fix `m_ω` (target side frozen). For every admissible implementation `(V, A)`
> with `V = X`:
> `R = I_f(M_ω; M̂) ≤ I_f(M_ω; X) = T(m_ω)`,
> for any f-divergence, any implementation-executor `E'`, any program/prompt/hybrid `A`, and any
> amount of computation inside `A`.

*Proof.* By (C2), `M_ω ⟂ M̂ | X`, i.e. `M_ω → X → M̂` is a Markov chain on the held-out
distribution (C1). Apply the f-DPI with the channel `K : (m, x) ↦ (m, M̂)` exactly as in PO §2.2:
`K∘P_{M_ω,X} = P_{M_ω,M̂}` and `K∘(P_{M_ω}⊗P_X) = P_{M_ω}⊗P_{M̂}`, so
`I_f(M_ω; M̂) = D_f(K∘P ‖ K∘Q) ≤ D_f(P ‖ Q) = I_f(M_ω; X)`. No property of `E'` was used other
than (C2); the Shannon chain-rule proof is not needed (and would not cover TVD-MI). ∎

**Corollary A1.1 (computation ops are free).** If `V = (X, f(X))` for deterministic `f` (execute
embedded code, sympy-check, recompute a statistic — proposal §3.3), then
`I_f(M_ω; V) = I_f(M_ω; X)`: the map `X ↦ (X, f(X))` is invertible (projection recovers `X`), so
DPI applies in both directions. Hence Lemma A1 holds verbatim for any computation-op-augmented
executor: **tools that are functions of the document cannot pierce the ceiling**; they only move
the *achieved* `R` toward it (they widen `B_{E'}`, not the channel).

**Corollary A1.2 (evidence ops re-base the ceiling).** If `V = (X, Z)` with `Z = g(q(X), W)`
touching external world state `W`:
- If `M_ω ⟂ Z | X` (the target verdict process never consulted `W`), admissibility relative to
  `X` still holds and Lemma A1 applies unchanged — moreover `I(M_ω; X, Z) = I(M_ω; X)`, so
  there is nothing for the op to add at the channel level.
- If `M_ω ⟂̸ Z | X` (the construct genuinely has signal outside the document — the patents-§102
  case), then (C2) fails relative to `X` and **the old ceiling `T(m_ω; X)` is not a valid bound**
  for `Z`-sighted implementations. The correct invariant is the augmented-channel transmission:
  under `M_ω ⟂ M̂ | (X, Z)`, `R ≤ I_f(M_ω; X, Z) =: T(m_ω; X, Z)`, and by DPI on the projection
  `(X,Z) ↦ X`, `T(m_ω; X, Z) ≥ T(m_ω; X)` (monotonicity under enlarging the conditioning
  channel — a DPI fact, valid for every f-MI; the Shannon chain-rule phrasing in §3.3/S5 is a
  special case, and TVD-MI, which has *no* chain rule, is still covered by this DPI form).
  This is §5.1's "re-measure `T` on the augmented channel," now with its precise trigger:
  re-measurement is *required* exactly when the target–evidence dependence `I(M_ω; Z | X) > 0`.

**Corollary A1.3 (transfer statement).** Let `E_1, E_2` be two implementation executors, both
admissible with the same input channel `V`. Then `R(E_2) ≤ T(m_ω)` holds with no reference to
`E_1`: the ceiling measured once (a property of the target channel and the item distribution)
certifies `h(E_2) ≥ 0` for every future admissible swap. Nothing about the *value* `h(E_1)`
constrains `h(E_2)`; a stronger `E_2` shrinks headroom, a weaker one inflates it, and the
certified content in both cases is only the shared ceiling.

**Attainability caveat (imported PO §11.1 correction, stated so the lemma is not over-read).**
`T(m_ω)` is attained only by the soft posterior readout `M̂ = η(X) = P(M_ω = 1 | X)`; a *sampled*
binary readout is strictly below `T` whenever the target is stochastic given `X`. So
`h = T − R` is an upper bound on the articulation gap that also contains readout-rounding unless
both-readouts is used; the lemma licenses the *bound*, never the interpretation of `h`'s value as
pure articulability. Similarly `sup_{A ∈ B_{E'}} R` may sit strictly below `T` for a weak
executor class — that shortfall is executor limitation (PO §5.5 component (c)), inside `h`, not a
violation.

**Finite-sample caveat.** Lemma A1 is a population inequality. The plug-in estimates `R̂`, `T̂`
on the same held-out items can order-invert by estimation noise; `R̂ > T̂` beyond CI width is a
bug/leak flag (this is what `vinfo.tvd_guardrail`'s both-legs computation is for), not a
refutation. There is no uniformity over executor swaps: each new `E'` requires re-evaluating
`R̂(E')` on the same split with its own CI. `T̂` itself needs a CI before `h ≥ 0` claims are made
at any stated confidence.

### 1.4 Lemma A2 (second-moment / attenuation form — the S1 ceiling)

The seam pilot does not compute f-MI; it computes correlations against a noisy judge. The
second-moment analog of Lemma A1 is §5bis S1, formalized here with its assumptions separated.

**Model assumptions.**
- **(N1) additive noise:** `M_p(i) = τ(i) + ε_p(i)` with `E[ε_p | X] = 0` (mean-zero noise given
  the item; implies `Cov(g(X), ε_p) = 0` for every deterministic `g`, which is how heteroscedastic
  *item*-dependent noise is tolerated — only covariances are used).
- **(N2) pass exchangeability (second-moment):** `Cov(ε_p, ε_q) = 0` for `p ≠ q` and
  `Var(ε_p) = σ²` equal across passes. (Needed for the Spearman–Brown step; NOT stated in §5bis.
  Cross-item heteroscedasticity is fine — tested T4 — but pass-asymmetric noise is not covered.)
- **(N3) implementation orthogonality:** the implementation `f` satisfies `Cov(f, ε_p) = 0` for
  all passes `p`. Automatic for deterministic `f(X)` under (N1); for stochastic implementations,
  requires the implementation's randomness independent of the judge's. **Violated** by any
  implementation that shares realized noise with the target passes (cached judge outputs, same
  batch/seed, or an LLM channel scored by re-reading the target pass's rationale).

> **Lemma A2.** Under (N1)–(N3), for the K-pass mean `M̄_K` and any implementation `f` with
> reliability `rel_f` (deterministic code: `rel_f = 1`):
> `corr(f, M̄_K) = corr(f, τ) · √(rel_K · rel_f) ≤ √(rel_K · rel_f)`,
> with `rel_K = K·rel₁/(1+(K−1)rel₁)` and `rel₁ = corr(M_1, M_2)`. Equality on the right iff
> `f` is an affine function of `τ` (plus, for stochastic `f`, its stable part is).

*Proof.* Deterministic case. `Cov(f, M̄_K) = Cov(f, τ) + Cov(f, ε̄) = Cov(f, τ)` by (N3).
`rel₁ = Var(τ)/(Var(τ)+σ²)` by (N1)–(N2), so `σ² = Var(τ)(1−rel₁)/rel₁` and
`Var(M̄_K) = Var(τ) + σ²/K = Var(τ)·(1+(K−1)rel₁)/(K·rel₁) = Var(τ)/rel_K`. Hence
`corr(f, M̄_K) = Cov(f,τ)/(σ_f·σ_τ/√rel_K) = corr(f,τ)·√rel_K`, and `corr(f,τ) ≤ 1`
(Cauchy–Schwarz) gives the bound. Stochastic case: write `f_p = ψ + ν_p` with the same structure;
`Cov(f̄, M̄) = Cov(ψ, τ)` and `Var(f̄) = Var(ψ)/rel_f` gives the extra `√rel_f` factor. ∎

**Multi-template passes — where §5bis S1 consequence (3) needs an extra condition.** The two
passes use *different templates*, so decompose `M_p = τ + φ_p(X) + ε_p` with `φ_p` the
template-specific stable component (a fixed function of `X` per template, uncorrelated across
templates and with `τ`). Then:
- If additionally **(N4)** `Cov(f, φ_p) = 0` for each template, the SB ceiling remains **exact**
  for `corr(f, M̄_K)` (computing as above with `φ` absorbed:
  `rel₁ = V_τ/(V_τ+V_φ+V_ε)` and `Var(M̄_K)` picks up `V_φ/K`, and the two effects cancel — the
  bound is unchanged), and `ρ̃` estimates `corr(f, τ)`, the template-common stable score. This is
  the sense in which the measured ceiling is "conservative": it caps tracking of `τ`, the smaller
  common part.
- **Without (N4)** the ceiling can be *exceeded*: the true ceiling for arbitrary `f(X)` is
  `√((K·V_τ + V_φ)/(K·V_τ + V_φ + V_ε)) ≥ √(SB(rel₁, K))`, with the gap carried by `V_φ`. An
  implementation tuned to one template's quirks (e.g. fit against pass-1 outputs) can sit in that
  gap. So §5bis's "the ceiling is conservative (still valid)" is derivable only under (N4) —
  see Gap 2.

**Correspondence A1 ↔ A2.** A2 is the second-moment shadow of A1: (N3)/(C2) both say "the
implementation shares nothing with the target but the item"; `√(rel_K·rel_f)` plays `T`;
`ρ̃ = ρ̂/√rel_K` plays `R/T`; "code pins `rel_f = 1`" is the second-moment face of "code pins
`T_norm = 1`" (PO §5.5 C2). There is **no exact bridge** between the two stacks (correlation is
not MI); the §5ter caution-2 objective mismatch applies — cross-stack comparisons are directional
only (Gap 9).

**Rank-statistic caveat.** Lemma A2 is a Pearson statement. The Spearman version used in the
pilot is exact only under monotone-transform models and is otherwise an approximation with
empirically bounded slack (~0.01–0.03, test T5); it is *not* a theorem (Gap 3).

---

## 2. Lemma B — matroid-U₂ and the tightening decomposition

### 2.1 Setup: the implementation-augmented ground set

Ground set `G = ⋃_{c∈Ω} {c} × I_c`, where `I_c` is the set of implementations of criterion `c`
(a code program, an LLM prompt channel, optionally op-configurations). Partition matroid
`𝕄 = {S ⊆ G : |S ∩ ({c}×I_c)| ≤ 1 for all c}` ("≤ 1 implementation per criterion"); budget `k`.
The feasible family is `𝔽 = {S ∈ 𝕄 : |S| ≤ k}` — the rank-`k` truncation of a partition
matroid, which is itself a matroid.

**Value-function extension (needed, not free).** The proofs below require the value `R(·)` to be
defined on **all of `2^G`**, including *infeasible* sets containing several implementations of
one criterion (the union `S_g ∪ S*` in step 1 is generally infeasible). Standing assumption:

- **(B0) extension:** the frozen aggregator executes arbitrary channel multisets, so `R(S)` is
  defined for every `S ⊆ G` (e.g. both implementations of `c` enter as two channels). If the
  execution semantics for both-implementation sets is undefined, the lemma as stated has no
  proof (Gap 6).
- **(B1) monotonicity:** `R` is monotone on `2^G` — or the argument is read against the
  free-disposal monotonization `R↑(S) = max_{T⊆S} R(T)`, exactly as PO §3.2's caveat (raw `R` is
  non-monotone; PRUNE). All marginals below are then marginals of the same monotone object.
- **(B2) weak submodularity:** submodularity ratio `γ ∈ (0, 1]` for that monotone object **over
  the enlarged ground set `2^G`** — the PO-measured `γ` over criterion subsets does not
  automatically transfer to implementation-augmented sets (Gap 5b).

### 2.2 Lemma B1 (matroid-U₂ instance bound — §5bis S3)

> **Lemma B1.** Under (B0)–(B2), let `S_g ∈ 𝔽` be any feasible current set (greedy or partial
> greedy), `δ(e|S_g) = R(S_g ∪ {e}) − R(S_g)` for `e ∈ G∖S_g`, and
> `OPT_𝔽 = max_{S ∈ 𝔽} R(S)`. Then
> `OPT_𝔽 ≤ R(S_g) + (1/γ) · Σ_{j=1}^{k} δ_{(j)}`,
> where `δ_{(1)} ≥ … ≥ δ_{(k)}` are the `k` largest **nonnegative** marginals over **all**
> `e ∈ G∖S_g` (implementation variants included, feasibility ignored).

*Proof.* Let `S* ∈ 𝔽` attain `OPT_𝔽`.
1. *(monotonicity, uses (B0)+(B1)):* `R(S*) ≤ R(S* ∪ S_g) = R(S_g) + [R(S_g ∪ S*) − R(S_g)]` —
   note `S_g ∪ S*` may be infeasible; (B0) makes the value defined, (B1) makes the inequality
   hold.
2. *(the `γ` step is the definition of the ratio):* at `(S, Ω) = (S_g, S*∖S_g)`,
   `Σ_{e ∈ S*∖S_g} δ(e|S_g) ≥ γ · [R(S_g ∪ S*) − R(S_g)]`, i.e.
   `R(S_g ∪ S*) − R(S_g) ≤ (1/γ) Σ_{e∈S*∖S_g} δ(e|S_g)`.
3. *(top-k over the superset):* `|S*∖S_g| ≤ |S*| ≤ k`; dropping any negative marginals from the
   sum only increases it (under (B1) they are ≥ 0 anyway); and the sum of ≤ k nonnegative
   marginals over `S*∖S_g ⊆ G∖S_g` is at most the sum of the `k` largest nonnegative marginals
   over `G∖S_g`. Chain the three. ∎

**Remarks.**
- **(R1: where the matroid actually enters.)** The partition constraint is used only through
  `S* ∈ 𝔽 ⇒ |S*| ≤ k`; the proof never exploits "≤ 1 per criterion." That is exactly why the
  bound is "valid, slightly loose" (§5bis S3): the top-k may include two implementations of the
  same criterion, which no feasible `S*` can.
- **(R2: a free tightening the implementation does not take.)** Since `S*∖S_g` contains at most
  one element per part, the top-k may validly be taken over the *per-part maxima* first (best
  marginal within each `{c}×I_c`), then the `k` largest across parts. This is a strictly tighter,
  still-valid bound; `u2_matroid_bound` implements the looser all-elements version (see
  reconciliation table). §5.2's phrase "top-k restricted to *feasible* marginals" describes this
  tighter variant, i.e. proposal text and code differ — both valid, code looser.
- **(R3: worst-case multiplicative companion.)** §5.2 asserts "worst-case multiplicative
  guarantees for weakly-submodular + matroid exist in the literature to cite." Guarantees of this
  type exist (e.g. residual-random-greedy-style results for γ-weakly submodular objectives under
  matroid constraints), but the constant is **not** the cardinality-case `(1−e^{−γ})`, and plain
  greedy under a general matroid loses even the submodular case's `1−1/e` (it gets `1/2`). The
  specific citation and constant must be verified before the appendix ships — flagged as Gap 5c,
  not derived here. Lemma B1 (the *instance* bound) is what is implemented and is
  self-contained.
- **(R4: estimation.)** All `R`-evaluations are noisy. Noise in the selected top-k order
  statistics biases the sum *upward* (selection effect), which is the conservative direction for
  an upper bound; noise in `R(S_g)` is symmetric. But a *plug-in* `γ̂` is anti-conservative
  (Gap 5a) — the one estimation direction that can invalidate the certificate.

### 2.3 Lemma B2 (tightening decomposition — §5bis S4, "the seam is where the certificate is tight")

Fix the metric as a **frozen linear aggregator** over channels: `m(x) = Σ_{e} w_e · v_e(x)`,
`w_e ≥ 0`, `Σ w_e = 1`, verdicts `v_e ∈ [0,1]`. Each channel is implemented as `v̂_e`, of type
`t_e ∈ {code, llm}`. Define per-channel residual terms exactly as `tightening_decomposition`
does:

- code channel: `g_e = 1 − κ̂_e` — labeled **certified** (CI-only);
- llm channel: `g_e = 1 − min(1, ρ̂_e / √rel_{K,e})` — labeled **uncertified**
  (articulation headroom);

and the bookkeeping totals `G_cert = Σ_{code} w_e g_e`, `G_uncert = Σ_{llm} w_e g_e`.

The lemma has three parts of genuinely different strength; the informal §5.3 statement blends
them.

> **Lemma B2(i) — per-channel status upgrade (from PO §5.5).** If channel `e`'s verdict process
> after migration is a **fixed deterministic program executed purely** (determinism audit:
> `H(p_i) = 0` per item ⇒ `T_norm = 1`; no timestamps/hash-order/float nondeterminism/IO), then
> for that channel: tacitness component (a) `= 0` (a fitting rule exists by construction —
> realizability), executor-limitation component (c) `= 0` (a compiler applies any rule
> faithfully), `T_e = 1`, and the articulation gap `A_e = 1 − R_e` is **pure learnability** —
> a single estimable quantity whose ceiling (1) is known and reachable, hence estimable with a
> CI and nothing unidentified inside it.

*Proof sketch.* This is PO §5.5 verbatim, applied per channel: the generator axis (code ⇒
realizability, `R_e = 1` reachable) and the executor axis (compiler ⇒ zero transmission noise)
are the two independent "code" entries; with both pinned, the three-way decomposition
`A = (a) + (b) + (c)` collapses to (b). The determinism audit is the *empirical precondition*
for the executor-axis pin — a failed audit means data bug, and the upgrade does not apply. ∎

> **Lemma B2(ii) — confinement + monotone decrease of the uncertified share.** Under: frozen
> weights `w`; an accepted migration of channel `e` (gate §4.2: `κ̂_e ≥ κ_min`, metric-level
> `ΔR ≥ −ε`, CF validity, `T = 1` audit) changing only channel `e`; all other channels' measured
> quantities unchanged — the relabeling moves `w_e·(1 − ρ̃_e) ≥ 0` out of `G_uncert` and adds
> `w_e·(1 − κ̂_e)` to `G_cert`. Hence `G_uncert` is non-increasing in the number of accepted
> migrations, strictly decreasing whenever `w_e > 0` and `ρ̃_e < 1`, and every term remaining in
> `G_uncert` carries an llm index — the uncertified residual is *confined to the LLM share by
> construction of the bookkeeping*.

*Proof.* Immediate from the definitions: with `w` frozen and other channels untouched, the only
change to `G_uncert` is deletion of a nonnegative term. ∎

*This part is near-definitional; its content is entirely in (i) — WHY the code-side term
deserves the label "certified" (every component identified, ceiling known) while the llm-side
term does not (an unidentified mixture of tacitness, learnability, executor limitation, judge-
noise estimation, and readout rounding).*

> **Lemma B2(iii) — metric-level additivity (the only part that bounds a metric-level
> quantity).** With frozen linear `w` and per-channel targets `v_e`, implementations `v̂_e`:
> `E|m̂(X) − m(X)| ≤ Σ_e w_e · E|v̂_e(X) − v_e(X)|` (triangle inequality + linearity). For
> **binary** channels `E|v̂_e − v_e| = P(v̂_e ≠ v_e) = 1 − κ_e`, so the weighted disagreement
> sum is a genuine upper bound on the metric-level L1 error. For a code channel whose target is
> itself deterministic, the observed disagreement is the true one. For an llm channel the target
> is the noisy judge: the observed disagreement conflates judge noise with real infidelity, and
> the ceiling-normalized correlation residual `1 − ρ̃_e` is a *different functional* that does
> **not** plug into this L1 additivity.

*Consequence, stated bluntly:* the implemented total `G_cert + G_uncert` (mixing `1 − κ̂` on the
code side with `1 − ρ̃` on the llm side) is a **diagnostic index with correctly labeled epistemic
classes**, not a bound on any single metric-level functional. The rigorous bound (iii) exists on
the all-binary / all-L1 restriction; the correlation-scale version would need either an
attenuation-corrected disagreement identity (available for symmetric flip noise; not derived
here) or channel-orthogonality assumptions to control cross-channel covariance terms in a
`1 − corr(m̂, m)` decomposition. See Gaps 7–8.

**What monotone tightening is and is not a theorem.** B2(ii) makes "the uncertified share
shrinks monotonically" a theorem *given frozen weights and per-event relabeling*. The stronger
reading in §8 prediction 3 — the **total** certified gap `U − R̂` decreases with each accepted
migration — is *not* derivable: it additionally needs `1 − κ̂_e ≤ 1 − ρ̃_e` (the incoming
certified term no larger than the outgoing uncertified one), and the §4.2 gate enforces
`κ̂_e ≥ κ_min` and metric-level `ΔR ≥ −ε`, neither of which implies `κ̂_e ≥ ρ̃_e` per channel.
Total-gap decrease is an empirical prediction, gate-`ε`-bounded, correctly listed under
falsifiable predictions rather than under lemmas (Gap 7).

---

## 3. Reconciliation: what `certificates.py` computes vs what the lemmas license

| `certificates.py` object | lemma clause | what is licensed | what is NOT licensed / caveat |
|---|---|---|---|
| `spearman_brown`, `attenuation_ceiling`, `ceiling_normalized` | A2 | ceiling on Pearson `corr(f, M̄_K)` under (N1)–(N3); `ρ̃` estimates `corr(f, τ)` under (N4) | not an MI bound; Spearman use approximate (Gap 3); multi-template validity needs (N4) (Gap 2); unstable as `rel₁ → 0` (S6 guard 0 handles) |
| `bootstrap_gate` | Rung 3 (PO §3.3) | within-set statistical best-vs-baseline at stated `B, δ`, item-bootstrap | nothing global; constant-baseline caveat (§5bis); percentile bootstrap coverage is approximate |
| `enumerate_stump_class` | B-side S2 upper edge | sample max over the **tested threshold grid** (`vals[::step]`) | docstring overclaims "no stump over these features" — only the gridded class is certified (Gap 4); `class_size` fed to `hoeffding_term` must match the grid (it does in tests) |
| `hoeffding_term` | S2 uniform convergence | finite-class / VC half-width `ε_n` for 0-1 agreement | applies to accuracy-type `κ`, not to correlation-type `ρ̂`; witness lower edge instead uses bootstrap CI |
| `codability_bracket` | S2 = A2 ∧ enum edge | two-sided bracket on `κ*(C)` for materialized `C`; upper edge min(ceiling, enum+`ε_n`) | mixes scales if witness edge is Spearman while enum edge is accuracy — caller must keep one fidelity scale; open-ended classes get no upper edge (by design) |
| `u2_matroid_bound` | B1 | instance upper bound on `OPT_𝔽` given (B0)–(B2) and caller-supplied `γ`, marginals for ALL `e ∉ S_g` | uses the looser all-elements top-k, not the per-part tightening (R2); `γ` is trusted input — a lattice-measured `γ̂` is anti-conservative (Gap 5a); (B0) extension assumed, unstated (Gap 6) |
| `tightening_decomposition` | B2(i)+(ii) bookkeeping | residual split by epistemic class; monotone `G_uncert` under frozen `w` | not a bound on a metric-level functional (B2(iii) restriction; Gaps 7–8); code-channel `rel_f=1` presumes the `T=1` audit passed |
| `shapley_2` | S5 mixedness readout | exact 2-player Shapley on the given value dictionary | attribution on `ρ`, not variance decomposition (S5 caveat verbatim); values are noisy estimates, no CI propagated |
| `op_monotonicity_violations` | S5 computation-op theory | population `κ*(C;O)` is monotone in `O` (feasible-set growth — trivial) | measured `κ̂*` from heuristic search can violate; a violation diagnoses **search shortfall**, not theory failure |
| `op_submodularity_ratio` | S5 op-lattice `γ̂` for U₂-over-ops | empirical ratio on the observed lattice | min over observed pairs ≥ population `γ` ⇒ plugging `γ̂` into U₂ can under-cover; "certified prunes" inherit this (Gap 5a) |

Cross-cutting: nothing in `certificates.py` computes `T(m_ω) = I_f(M_ω; X)` or a TVD-MI `R` —
Lemma A1 governs the PO stack (`vinfo`/`recon_channel`), Lemma A2 governs the pilot's correlation
stack; the bridge is the §5ter caution-2 open item (Gap 9).

---

## 4. Gaps and counterexample risks (numbered, required reading before citing §5bis)

1. **Shared-noise implementations break both ceilings.** (C2)/(N3) fail for any implementation
   that reuses realized target-judge outputs (caching, shared seeds, scoring by re-reading the
   target pass's rationale, fitting on held-out target verdicts). Then `R > T` and
   `ρ̂ > √rel_K` are *reachable*, and a violation is a leak indicator, not a paradox. §5bis
   states S1 as if `Cov(f, ε) = 0` were automatic; it is an audit obligation.
2. **Multi-template passes: the "conservative ceiling" claim needs (N4).** With template-specific
   stable components `φ_p(X)`, the SB ceiling caps only implementations orthogonal to every
   `φ_p`; an implementation tracking one template's quirks can exceed `√(SB(rel₁,K))` by up to
   the `V_φ` gap (derivation in §1.4). §5bis S1 consequence (3) ("folds form-variance into noise
   ⇒ conservative, still valid") is not derivable without the extra orthogonality condition.
   The observed 0 violations at 116-metric scale is evidence that `V_φ` is small or untracked —
   evidence, not a theorem.
3. **Spearman is not covered by Lemma A2.** The lemma is Pearson-only; the pilot's Spearman usage
   carries empirically-bounded slack (~0.01–0.03, T5) with no finite-sample guarantee. Metrics
   near the ceiling must be read with the Pearson companion (as §5bis already instructs) — but
   note the *gates* (`bootstrap_gate` default) run on Spearman.
   **RESOLVED-BY-WORDING + MEASURED AT SCALE (2026-07-04, `bridge_calibration.json`, 609
   channel pairs across 10 surveys):** median |ρ_S − r_P| = .023 (consistent with T5) but the
   tail is much fatter than T5 suggested — p90 = .080, p99 = .198, max = .286 (p90 = .111 even
   restricted to |ρ| ≥ .5). Adopted wording: (i) GATE certificates are *empirical bootstrap
   statements about Spearman* and cite Rung-3 only — they never invoke A2, so they need no
   rank-version of the lemma; (ii) any CEILING-normalized reading (ρ̃, "x% of ceiling") must
   quote the per-channel Pearson companion, because tail slack ≫ gate margins. A2 stays
   Pearson-only; no rank extension is attempted.
4. **`enumerate_stump_class` threshold thinning voids the exact-upper-edge claim as documented.**
   With `step > 1`, thresholds between grid points are untested, and a stump there can exceed the
   returned `best_acc`; the certificate is valid only for the **gridded** class (which is what
   `class_size` counts in the tests — internally consistent, but the docstring's "no stump over
   these features exceeds best_acc on this sample" overclaims). Fix: enumerate all breakpoints
   (`n_thresholds ≥ #distinct values`) or state the gridded-class scope in the docstring and
   any prose using S2 upper edges.
5. **The `γ` inputs are the certificate's soft underbelly.** (a) `op_submodularity_ratio`'s `γ̂`
   is a minimum over *observed* pairs, hence `γ̂ ≥ γ_pop`; plugging it into `1/γ` makes U₂
   potentially too small — an **invalid** upper bound. Mitigations: planted-γ calibration (PO
   §5.5 C6), a lower confidence bound on `γ`, or reporting U₂ as a curve in `γ`. (b) The `γ`
   needed for Lemma B1 lives on the implementation-augmented ground set `2^G`; PO-era `γ`
   measurements on criterion subsets do not transfer automatically. (c) The §5.2 worst-case
   multiplicative companion for weakly-submodular-plus-matroid exists in the literature but with
   different constants than the cardinality case; the specific theorem/constant is
   **citation-to-verify**, not derived here.
6. **Lemma B1 needs `R` defined and monotone on infeasible sets.** Step 1 evaluates
   `R(S_g ∪ S*)`, which generally contains two implementations of one criterion. If the frozen
   aggregator has no execution semantics for such multisets, the proof has an undefined term.
   §5.2/S3 never state this extension assumption (B0). Either define the both-implementations
   semantics (e.g. channel concatenation under the linear aggregator) or restate the lemma with
   a value oracle on `2^G` as an explicit input.
7. **Monotone tightening of the TOTAL gap is not a lemma.** Only `G_uncert` is provably
   monotone (B2(ii), near-definitional under frozen `w`). Total-gap decrease per migration needs
   `κ̂_e ≥ ρ̃_e`, which the §4.2 gate does not enforce (it enforces `κ̂_e ≥ κ_min` and
   metric-level `ΔR ≥ −ε`). §8 prediction 3 correctly treats the slope as falsifiable; any prose
   reading §5.3 as "the certified gap provably shrinks" overstates.
8. **The S4 decomposition is not a bound on a metric-level functional as implemented.** Additivity
   is rigorous in L1/disagreement units for binary channels under the frozen linear aggregator
   (B2(iii)); the implemented mixture of `1 − κ̂` (agreement scale) and `1 − ρ̃` (ceilinged
   correlation scale) is a labeled index. A correlation-scale decomposition of `1 − corr(m̂, m)`
   has uncontrolled cross-channel covariance terms without further assumptions. Also, "monotone"
   in B2(ii) refers to population bookkeeping — re-estimated fidelities of *untouched* channels
   drift across rounds, so the empirical `G_uncert` series can tick up without contradicting the
   lemma.
9. **Two-stack objective mismatch (standing, §5ter caution 2).** Lemma A1 (f-MI, DPI, headroom)
   and Lemma A2 (correlation, attenuation) govern different measured stacks; no inequality
   connects a ceilinged Spearman fidelity to a TVD-MI headroom. Until the TVD-MI bridge runs,
   any sentence moving a number from one stack to the other is directional prose, not math.
   **BRIDGE RUN (2026-07-04, `methods/metric_seam/pilot/bridge_calibration.py` →
   `outputs/metric_seam_pilot/bridge_calibration.json`):** 609 (aspect × channel) pairs, every
   survey source (PR v1/v2/v3 incl. hybrids, 7 task surveys), TVD-MI (vinfo, 2-bin,
   perm-debiased) vs Spearman against the same 2-pass Gemma target. The correspondence is
   monotone in the mean (decile means .015 → .666 as |ρ| goes 0→.9; Spearman(|ρ|, TVD-MI) = .71
   across pairs) but per-pair spread is wide (e.g. |ρ| .6–.7 → TVD .21–.55). Licensed use: the
   decile table as a directional lookup; per-channel inversion remains unlicensed. Cross-stack
   sentences stay directional — now with an empirical curve behind the direction.
   ★ The bridge run also EXPOSED AND FIXED an estimator bug in `vinfo._binize` (2-bin path):
   stable-sort tie-breaking made bin membership position-driven for heavily-tied vectors, so two
   INDEPENDENT ~90%-tied vectors read TVD-MI ≈ .7–.85 (permutation debias cannot see it — the
   permutation destroys exactly the order-coupling it should calibrate). Fixed 2026-07-04 by
   independent seeded random tie-breaking per side (≡ independent infinitesimal jitter);
   19/19 tests green. Semantics note: identical heavily-tied vectors now read dependence-after-
   jitter (a lower bound), not ~max. `tvd_recovery`/`tvd_transmission` (closed-form, no binning)
   were never affected — headline R̂/T̂ numbers are safe; `measures.py` scorecard diagnostics
   (reliability/invariance/applicability `tvd_mi` fields) computed on heavily-tied metrics
   BEFORE the fix are suspect and need recompute where they were used.
10. **Evidence ops void the old ceiling silently if untracked.** "Headroom survives verbatim"
    (§0 of the proposal) is scoped: for a `Z`-sighted implementation with `I(M_ω; Z | X) > 0`,
    `T(m_ω; X)` is simply not an upper bound on `R` (Corollary A1.2). The certificate machinery
    must record, per implementation, *which input channel* its ceiling was measured on; a hybrid
    that quietly gains a retrieval op while citing the pre-op ceiling produces an invalid
    certificate. Conversely, when `M_ω ⟂ Z | X` the ceiling literally cannot move —
    a measured "`T` rise" after an evidence op on such a channel is estimation noise or leakage,
    and the S5 acceptance test should treat it as such (CI required, as S6 step 3 states).
11. **Attainability / readout rounding inside `h`.** `T` is attained only by the soft-posterior
    readout (PO §11.1 correction); under sampled binary readouts, part of measured headroom is
    rounding, not articulability. Lemma A1 licenses `h ≥ 0` and transfer of the ceiling —
    interpretations of `h`'s *magnitude* as pure articulation gap additionally require
    both-readouts. Same caution applies to `1 − ρ̃` in B2 (it also contains estimation error in
    `rel₁`, nonlinearly amplified through the square root at small `rel₁`).
12. **All statements are population-level; the certificate discipline is per-swap re-evaluation.**
    No lemma gives uniformity over executor swaps or over the op lattice: each admissible new
    implementation needs its own held-out `R̂` (same split, same canonical text — validity guards
    4/7) with CI, against a `T̂`/ceiling that itself carries a CI. Winner's-curse over many
    candidate implementations is handled by Rung 3 (`bootstrap_gate`), not by Lemmas A/B — citing
    a lemma does not substitute for the gate.

---

*Status: draft for review. B1/B2(i)–(ii) and A1/A2 are proved under the stated assumptions;
B2(iii) is proved on its restriction; R3's multiplicative companion and Gap 5c are
citation-to-verify; Gaps 2, 6, 7 identify §5bis statements that need either an added
assumption or a wording downgrade before the appendix is folded into the PO whitepaper.*

*2026-07-04 update (roadmap-v2 R2): Gap 3 resolved by wording + measured at survey scale
(see inline); Gap 4 resolved (docstring restated to gridded-class scope,
`certificates.py`); Gap 5a mitigated (`u2_matroid_bound` now warns, takes optional per-part
tightening per remark R2, and `u2_matroid_bound_curve` reports U₂ as a curve in γ — quote the
curve, never a plug-in γ̂ point); Gap 9 measured (bridge calibration + the `_binize` tie-break
fix). All 14 planted certificate tests + new-function checks green.*
