# Metric-count certificates — upper and lower bounds on how many articulated metrics a preference needs (and how many a model family can collect)

*2026-07-01. Companion to `2026-06-18__prompt-optimality-theory.md` (hereafter **PO**). PO answers the
PER-METRIC question with **no label**: given one latent metric `M_i`, how close is a found prompt to the
best possible articulation of it (`R ≤ T`, the ladder, B_E-ATLAS/§12.6). This doc answers the **BANK-level**
question **with labels**: given the revealed-preference record `Y` (decades of accept/reject decisions — the
anthropological estimand of PO §12.6-scope), certify **upper and lower bounds on (a) the total label-signal
any set of articulated metrics can carry on executor family `E`, and (b) the NUMBER of metrics needed /
collectible**. Assumption throughout: each metric is implemented **optimally as a prompt** — the PO pipeline
supplies, per criterion, a prompt with certified per-metric headroom `T_i − R_i ≤ ε_i` (PO §11.1). Every
claim is tagged **[holds] / [derived] / [design] / [conjectural]**.*

*Updated 2026-07-02: §2a generator abstraction (certificates are algorithm-agnostic), §4.4 gate-validity
conditions (both empirically bitten), §4.5 the certification chain as implemented, §10 explicit PO↔MCC
crosswalk, §11 first end-to-end empirical run (20 proposals / 0 survivors).*

---

## 0. The answer in one paragraph

With labels, the theory gets **stronger in three places and degenerate in one new place**. Stronger: (i) the
bank objective `V(S) = I(Y; M_S)` is **monotone** (information never hurts), so the Minoux/`U₂` stopping
certificate applies to the raw objective — PO's monotonization caveat (PO §3.2/§6.1) dissolves; (ii) the
assumption-free ceiling is no longer the vacuous readout cap `cap_f` but the **task-intrinsic**
`I(Y;X)`, estimable from below by the dense ceiling `C` — so the value bracket is *meaningfully tight*;
(iii) the capture–recapture value measure `v(s)` becomes the **supervised conditional gain**
`I(Y; s | M_S)` — the estimand itself, not a proxy — and the "adversarial saturation probe" becomes a
concrete, runnable **three-part residual battery** (MOB instability / gap nodes / dense-residual probe), of
which the ctree is part 1. Degenerate: with labels, a **single mega-metric** — the Bayes posterior
`η(x) = P(Y=1|x)` asked as one prompt — achieves the whole ceiling with `N = 1`, so the count question is
meaningless **unless metrics are constrained to the articulable class** (each member passes PO's recovery
certificate `R ≈ T` and is atomic under the §12.6.1 quotient). Under that constraint the two headline
certificates are: **necessity** `N ≥ V_bits / log₂K` (assumption-free; tightens to `V/T_max` under
per-metric-call scoring — no bank of `N` noisy `K`-ary judges can carry more than `Σ min(T_j, log K)` bits
of label-signal, by DPI + entropy counting) and **sufficiency /
collectibility** `N_δ ≤ (U − V(S_g))/δ` (at most this many *further* metrics each worth ≥ δ exist, where `U`
is any certified upper wrap on the articulable ceiling — the dense-stack ceiling under dense-dominance, or
the process-relative flux bridge without it). The count `N` is the size of the community's **evaluative
lexicon** — the anthropological quantity — and it is certified per model family `E_t`, reported as a
staircase, never a fitted asymptote.

---

## 1. Setup — objects, the two estimands, and the degeneracy guard

**Data.** Items `x` with revealed labels `Y ∈ {0,1}` (archival practice; no realtime elicitation). Splits:
`discover / guard / test` (three-way, grouped by source id where ids cluster). `n` items; readout arity `K`
(binary canonical, `K=2`).

**Metric.** An articulated criterion `c` implemented as a prompt on executor `E`, with the PO per-metric
certificate: orbit-averaged soft readout `m̄_c` (PO §12.6.2), per-metric transmission `T_c = I(m̄_c; X)`,
recovery `R_c`, headroom `T_c − R_c ≤ ε_c`. "Optimally prompted" = `ε_c` small and certified; a bank built
from under-optimized prompts **undercounts** articulable signal (see §7 direction-of-error).

**Bank.** `S = {m_1, …, m_k}`; `M_S(x)` the vector of verdicts. Bank value
> `V(S) := I(Y; M_S)` — measured on held-out items as **log-loss reduction** (cross-entropy of the bank
> model minus `H(Y)`), the plug-in MI in bits. AUC is retained as the secondary, familiar readout; the
> certificates compose only in bits. **[design — switch the bank readout to report both]**

**The articulable class (the degeneracy guard — new, has no PO analog).** With labels, `V` is trivially
maximized by ONE metric: ask the executor "will this community accept this item?" — `m = η(X)` is a single
`K`-ary readout achieving `I(Y; η_K(X)) → I(Y;X)` as `K` grows. The count question is well-posed only over:
> `𝒜_E := { m : m passes the PO recovery certificate (R_m ≈ T_m, held-out) and is ATOMIC under the
> §12.6.1 quotient (semantic-merge + behavioral-split; not decomposable into existing bank members by the
> redundancy/decomposition guard) }`.

This is not an anti-cheat kludge; it is the substance. The mega-metric is exactly the thing a taste-heavy
task makes **non-articulable**: no reconstructor can re-derive "the community's whole taste" from examples
(`R ≪ T` for the mega-prompt), so `𝒜_E` excludes it *by measurement, not by fiat*. The count `N` then counts
**atomic, reconstructible, distinct criteria** — the community's evaluative lexicon. **[design; the
constraint is operational and already implemented: recovery loop + quotient + redundancy guard]**

**The two estimands.**
| symbol | definition | anthropological reading |
|---|---|---|
| `A*_E` | `sup_{S ⊆ 𝒜_E} V(S)` — the articulable ceiling for family `E` | how much of the practice is communicable at all |
| `N_E(ε)` | `min{ |S| : V(S) ≥ A*_E − ε }` | the size of the lexicon needed to communicate it |

Levels, kept straight (standing correction, [[feedback_alpha_probe_is_metric_level]]): PO's `B_E`/α census
counts **criteria available to express ONE metric**; `N_E` counts **metrics needed to express THE
PREFERENCE**. One level up. The machinery is shared; the population is different.

---

## 2. What having `Y` changes relative to PO — four upgrades, one new failure mode

| # | PO (no `Y`) | here (with `Y`) | consequence |
|---|---|---|---|
| U1 | `R(S)` non-monotone (PRUNE), certificates need monotonization `R↑` | `V(S) = I(Y;M_S)` **monotone** (chain rule, CMI ≥ 0) **[holds]** | Minoux/`U₂` stopping bound applies to the raw objective; the §6.1-style caveat is gone. (Finite-sample plug-ins can still dip — use CIs.) |
| U2 | assumption-free global ceiling = `cap_f` (vacuous, a readout constant) | assumption-free ceiling = `I(Y;X)`, task-intrinsic; **estimable from below** by the dense ceiling `C` | the value bracket (§3) is tight enough to be scientific; PO §3.1's looseness complaint is answered by the label |
| U3 | capture–recapture value `v(s)` = recovery gain (proxy) | `v(s | S_g) = I(Y; s | M_{S_g})` — supervised conditional gain, **the estimand itself** | D1–D3 flux machinery (PO §12.6.3) transfers verbatim with a better value measure; the ε-gap bridge (§12.6.4) now bounds *label-signal*, not proxy-recovery |
| U4 | saturation probe = `I(X_probe; M | X_Ω) ≈ 0`, expensive and indirect | the **residual battery** (§5): MOB instability + gap nodes + dense-residual probe — all directly runnable | ctree/MOB acquires its formal role: part 1 of the saturation certificate |
| D1 | no degeneracy: the target is the metric itself | **mega-metric degeneracy**: one Bayes-posterior prompt collapses `N` to 1 | the articulable-class constraint (§1) is mandatory; PO's recovery certificate becomes the *membership test* |

Also unchanged, inherited wholesale from PO §12.6: the quotient species definition, form-orbit adversarial
reporting, the singleton-degeneracy lemma (report `f_1/N` beside every exponent), process-relativity (§12.5
— every "collectible" claim is relative to a named generator set), and the family-scaling staircase rules
(no fitted asymptotes).

---

## 2a. The generator abstraction — the certificates are algorithm-agnostic. [derived + implemented]

Nothing in §§3–4 depends on HOW candidate metrics are produced. Formalize the producer as a **generator
arm**:

> `𝒢 : (D_discover, S, aux) ↦ {c₁, c₂, …}` — any algorithm emitting candidate criteria from the discover
> split, the current bank `S`, and arm-specific auxiliaries (residuals, tree gap nodes, raw labeled
> examples, or nothing at all).

**The interface contract** (all four required for the certificates below to remain sound for that arm):

- **G-1 (split hygiene).** Proposals are a function of `discover` only; guard/test are never touched at
  generation time.
- **G-2 (materializability).** Each candidate is executable as a per-item `K`-ary verdict by the frozen
  judge, so its gate statistics and ledger tracks are measurable on any split.
- **G-3 (membership is post-hoc and uniform).** `𝒜_E`-membership — recovery `R ≈ T`, quotient atomicity,
  redundancy — is tested *downstream, identically for every arm*. Where a candidate came from carries
  **zero evidential weight**; a human-authored criterion and an autorubric hallucination face the same
  gate.
- **G-4 (ledger).** Every proposal emits the three tracks (data-to-develop, applicability,
  reconstruction `R`) plus its δ-gate outcome in bits, tagged with its arm.

Under the contract there is a clean division of labor **[derived]**:

- The **validity** of every certificate — floor, wraps, `N_lower`, `N_upper`, battery verdicts — is
  **generator-independent**. It flows from the splits, the gate, and the membership tests, never from the
  proposal mechanism. Any new generation algorithm (evolutionary refinement à la autometrics, GEPA-refined
  proposers, retrieval from rubric pools, future methods) plugs in without touching the theory.
- The **completeness** of the discovered set — how much of `𝒜_E` the process reaches — is entirely
  **generator-relative** (PO §12.5). Each arm is one **capture–recapture list** over `𝒜_E`; the flux and
  coverage reads (§3 Wrap 2, D1–D3) are defined over the generator **set** `G = {𝒢₁, …, 𝒢_A}`, and a
  single-arm flux read is anti-conservative (false saturation). `|G| ≥ 2` is mandatory for any saturation
  claim, and **the certified artifact is the UNION ledger across arms, never a per-arm ledger** (a per-arm
  certificate must carry the single-arm honesty note).

Arms already running through the one gate (`generators.py`, `loop.py`):

| arm | reads | targets |
|---|---|---|
| global residual contrast (WRONG/RIGHT) | `S` + residuals | corpus-uniform misses (shape iii-ish) |
| ctree/MOB gap-node contrast | `S` + tree partition | moderation/region misses (shapes i–ii) |
| label_contrast | raw `(x, y)` only, blind to `S` | naive baseline; the redundancy guard does the work |
| unconditional (autorubric-style) | task description only, data-free | the prior; how far zero-data articulation gets |
| (extensible) GEPA-refined, evolutionary, pool-retrieval, human lists | any | new recapture lists for the same census |

This places the ctree of §5 twice, coherently: as an **instrument** (residual-shape detector) and as a
**generator arm** (its gap-node contrasts emit candidates). Both roles obey the same contract; neither is
privileged by the certificates.

---

## 3. The value bracket — certifying `A*_E`

```
V(S_g)   ≤   A*_E   ≤   min(  U_dense ,  U_flux ,  I(Y;X)  )
(achieved,                (each an upper wrap, different assumptions)
 floor)
```

**Floor. [holds]** `V(S_g)` on the untouched test split, with CIs. Valid only if each member carries its
per-metric PO certificate — otherwise the floor is *also* an undercount of `A*` and the bracket silently
widens (§7).

**Wrap 1 — the dense-stack ceiling `U_dense`. [holds, conditional on dense-dominance]**
`A*_E ≤ Bayes(Y;X)` always (any bank predictor is a function of `X`). The dense model estimates Bayes from
below, so `C` alone does **not** bound `A*_E` — if the dense model is data-starved, an articulated bank can
beat it (CW's dense curve was *still climbing* in the 2026 sweeps — so for CW, `C` is NOT a valid wrap
today). Two repairs:
- **Stacking:** `U_dense := upperCI( V(dense ⊕ M_{S_g}) )` — the dense model given the bank's outputs as
  extra features. `≥ max(C, V(S_g))` by construction, still `≤ Bayes`. Using the stack as the wrap makes the
  dominance check *internal*: if the bank adds nothing to dense, dense already realizes the bank.
- **Dominance check (gate):** dense scaling curve plateaued **and** `V(S_g) ≤ C − margin`. If either fails,
  report `A*` as **right-censored above** — do not publish `U_dense` (the CW case).

**Wrap 2 — the process-relative flux bridge `U_flux`. [derived, conditional — PO §12.6.4 with supervised
value]** Run the discovery process — the generator set `G` of §2a, `|G| ≥ 2`, frozen, iid — to `N_draws`; quotient;
compute the value spectrum `w_j = Σ_{s: n_s=j} v(s|S_g)` with the **supervised** `v`. Then with probability
≥ 1−δ, at the horizon `(1+c)N_draws` and the adverse orbit end:
> `A*_{E, process, horizon} ≤ V(S_g) + ε_flux`,  `ε_flux = (1/γ̂)[ Ĝ(c) + B√(2 log(1/δ)/N) + B/N ]`
with `γ̂` the measured tail submodularity ratio and the residual battery (§5) as the γ-free backstop. All
PO §12.6 walls carry over: zero-mass tail invisible, merge-precision is the binding gate, claims are
process-relative. What is *better* here: `Ĝ(c)` bounds unseen **label-signal** directly.

**Wrap 3 — `I(Y;X)`.** Not directly estimable; listed for logical completeness. `U_dense` is its estimator.

**Task-level corollary (the headline PO §12.6.4 already states, now with the bank in it):**
`Taste ≥ lowerCI(C or C_stack) − upperCI(V(S_g) + ε_flux)` — the certified codification gap. This doc adds
the **second axis**: not just *how much* is inarticulable, but *how many words the articulable part takes*.

---

## 4. The count bracket — the headline certificates

### 4.1 Necessity — a lower bound on `N`. [derived, assumption-free]

Judges see only `X` (never `Y`), so `Y — X — M_S` is Markov and, chaining conditional MI:
> `V(S) = I(Y; M_S) ≤ I(X; M_S) = Σ_j I(X; m_j | m_{<j}) ≤ Σ_j min( T_j , log₂ K )`  bits.
The `log₂ K` leg is assumption-free (`≤ H(m_j)`). The tighter `T_j` leg uses one structural fact:
verdicts are **conditionally independent given `X`** (each metric scored in a separate judge call),
so `I(X; m_j | m_{<j}) = H(m_j | m_{<j}) − H(m_j | X) ≤ H(m_j) − H(m_j|X) = T_j`. **Implementation
flag:** the current materializer (`make_vllm_judge_scorer`) scores ALL metrics for an item in ONE
judge response — verdicts share the judge's forward pass, conditional independence fails, and the
`T_j` leg is not licensed. Under shared-call scoring only the `log₂ K` leg (and hence
`N ≥ V/log₂ K`) is valid; certifying the tighter `N ≥ V/max_j T_j` requires per-metric scoring
calls (or measuring the residual inter-verdict dependence `I(m_j; m_{<j} | X)` and adding it back).

> **Count-necessity certificate.** Any bank of `K`-ary articulated metrics achieving `V` bits satisfies
> `N ≥ V / log₂ K`; a *specific measured* bank additionally satisfies `N ≥ V / max_j T_j` (judge noise
> shrinks the per-metric budget below `log K`). Scoping: to bound the class-level `N_E(ε)` (over ALL
> banks, not the one in hand) the denominator must be bank-independent — `log₂ K`, or
> `T_max := sup_{m ∈ 𝒜_E} T_m` if the articulable class's transmissions are uniformly profiled (the
> per-metric census supplies an empirical `T_max`; it is process-relative like everything else).

Reportable today: with the achieved `V(S_g)` as a floor on `A*`, **"explaining this community's practice to
the level we have already demonstrated requires at least `⌈V(S_g)_bits / log₂K⌉` atomic `K`-ary criteria
(binary: `⌈V_bits⌉`), and at least `⌈V_bits / T_max⌉` under the measured transmission profile."**
This is the safe direction for the human-exceptionalism thesis (a big `N` is the claim; the bound is
assumption-free). The granularity lever is explicit and honest: the count is stated **at fixed `K`** — a
lexicon of 40 binary questions may compress to ~13 five-point questions (`log₂5 ≈ 2.3`); binary is the
canonical report.

### 4.2 Sufficiency / collectibility — an upper bound on `N`. [derived, conditional]

Two certificates, different strengths:

**(a) Value packing — the strong one.** `V` is monotone and bounded by any §3 wrap `U`. Telescoping:
> the number of *further* metrics each adding ≥ δ (the acceptance gate's `min_auc_gain`, stated in bits) is
> **`N_δ ≤ (U − V(S_g)) / δ`** — and hence the family can *collect* at most `|S_g| + (U − V(S_g))/δ`
> δ-distinct metrics, ever, relative to the wrap's assumptions.
This is the user-facing "certificate for an upper bound on how many metrics we can collect with this model
family." Its honesty conditions: (i) `U` valid (dense-dominance for `U_dense`, process horizon for
`U_flux`); (ii) δ-distinctness enforced by the quotient — infinitely many paraphrase-"metrics" are always
collectible, they just aren't species; (iii) each counted metric optimally prompted, else a real criterion
can be miscounted as < δ.

**(b) Greedy stopping — the within-class ε-certificate.** With monotone `V` (U1) the Minoux/`U₂` bound is
clean: after greedy over the discovered pool, if `(1/γ̂) Σ_{top-k} δ_j ≤ ε` then `S_g` is ε-optimal within
the pool, and D1–D3 extend the statement to the process horizon. **The count at stop, `|S_g|`, is a
certified ε-sufficient lexicon size** (process-relative). The classical worst-case complement: greedy needs
`k ≈ (k*/γ) ln(V_OPT/ε)` steps to match the best size-`k*` bank — reported as context, never as the
certificate (k* unobserved).

### 4.3 The two counts to publish, per task × family

| certificate | statement | assumptions |
|---|---|---|
| `N_lower = ⌈V(S_g)/T_max⌉` (assumption-free version: `⌈V/log₂K⌉`) | "no lexicon smaller than this explains what we already explained" | DPI + entropy (+ per-metric independence & `T_max` profile for the tight version) |
| `N_upper = |S_g| + (U − V(S_g))/δ` | "no more than this many δ-useful criteria exist for this family" | wrap `U` + quotient + per-metric optimality |

A **thin/codified** preference: small `N_upper`, bracket tight, residual battery silent. A **deep** one:
`N_lower` grows with every discovery round, flux `Ĝ(c)` stays heavy, dense-residual stays positive — the
lexicon doesn't close. That trichotomy (plus FORM-DOMINATED) is PO §12.6.6's decision rule lifted to the
bank level.

### 4.4 The acceptance gate δ is a certified quantity, not a knob. [measured 2026-07-02]

`N_upper` divides by δ and `|S_g|` counts what cleared δ, so both count certificates inherit the
statistical validity of the gate. Two conditions, **both empirically bitten** in the first arm-comparison
runs:

1. **Instrument-noise condition: δ must exceed the acceptance instrument's noise floor.** A single guard
   split at `n_g = 84` has Hanley–McNeil AUC SE ≈ 0.06 — a 0.02 AUC gate on one split is *below noise*, and
   nothing real at plausible per-metric effect sizes can pass it reliably (observed: 0/12 across three arms,
   including proposals with true small positive gains). Repair: **paired K-fold CV** over pooled
   discover+guard — per-fold pairing cancels the split noise, supporting a 0.01 gate at `n ≈ 420`. The
   gate's measured noise floor is reported next to δ in any certificate-bearing run.
2. **Selection condition (winner's curse): the gate is a max over `J` proposals.** Accepting whatever
   clears δ among `J` candidates induces selection bias; the accepted gain regresses toward 0 out of sample
   (observed: CV +0.018 → untouched-test −0.016 ± 0.042 at `J = 20`). Repair (either): a **fresh-seed CV
   confirmation stage** between gate and test, or **deflate the gate by the selection multiplicity**
   (accept only above the `1 − α/J` quantile of the permuted-null gain).

Direction of error if violated (§7 discipline): a sub-noise or un-deflated δ admits noise metrics —
`|S_g|` is polluted and `N_upper = |S_g| + (U−V)/δ` inflates in both terms, so the collectible-lexicon
bound is **overcounted**; the floor stays honest only because `V(S_g)` is read on the untouched test. The
gate conditions are therefore load-bearing for §4.2, not hygiene.

### 4.5 The certification chain, concretely. [implemented 2026-07-02]

What "certifying the metric space" means operationally — eight steps, each with its statistic, its
assumptions, and the artifact that carries it. The bank-level certificate is a **conjunction**: it is
valid iff every step below it holds, and each step's failure prints as an explicit honesty note rather
than silently degrading (implemented: right-censoring note, single-arm note, `T_max`-refusal note).

| # | object certified | statistic | assumptions | artifact |
|---|---|---|---|---|
| 1 | `𝒜_E` membership (per candidate) | reconstruction `R` from `(x, verdict)` pairs, re-executed, balanced agreement + AUC; quotient atomicity; redundancy `R² ≤ τ` | reconstructor never sees the rubric | ledger `reconstruction_agreement/auc`, `redundancy_r2` |
| 2 | per-metric optimality | `T_j − R_j ≤ ε_j`, orbit-averaged | PO §11.1 pipeline | `reconstructor_fn` = GEPA plug-point |
| 3 | acceptance (per proposal) | paired-CV Δbits ≥ δ, δ calibrated per §4.4 | fold-pairing valid | `acceptance_eval="cv"`, `min_bits_gain`; ledger `bits_gain` |
| 4 | confirmation (winner's-curse control) | fresh-seed CV of the accepted max | §4.4.2 | [design — next run] |
| 5 | the floor | `V(S_g)` on untouched test, with CI | test touched once | `validate_kept_metric.py`-style read |
| 6 | the wrap | dense-stack (dominance-gated) or flux over the generator set `G` | §3 per wrap | `certificates.dense_stack_wrap` / `flux_wrap`; right-censoring note |
| 7 | the counts | `N_lower` (log₂K leg only under shared-call scoring), `N_upper` | §4.1–4.2 | `certificates.count_certificates` (refuses the `T_max` leg without per-call scoring) |
| 8 | saturation | residual battery: (i) MOB, (ii) gap nodes, (iii) dense-residual + flux `Ĝ(c)` | §5 power conditions | both engines + playbook Phase-2 gates |

Steps 1–4 are per-metric and generator-agnostic (§2a G-3); steps 5–8 are bank-level and computed from the
**union ledger** across arms. Code: `methods/metrics_tree_infilling/{certificates,generators,global_infill}.py`;
protocol: `AGENT_PLAYBOOK.md`.

---

## 5. The residual-structure trichotomy — where ctree/MOB formally sits. [derived + measured]

The remaining signal `I(Y; X | M_S)` is what new metrics could still capture. It manifests in exactly three
detectable shapes, each with its own instrument:

| shape | definition | instrument | what fires |
|---|---|---|---|
| **(i) moderation-shaped** | bank→label coefficients vary across an observed partition axis `z` | **MOB / ctree** M-fluctuation test, permutation-calibrated, Bonferroni over `z`'s | a split ⇒ a missing *moderator* metric localizes the gap |
| **(ii) region-shaped main effect** | a subpopulation where the bank predicts poorly (deviance/AUC), even if coefficients are stable | **gap-node flags** on terminal nodes + within-node WRONG/RIGHT contrast → proposer | a flagged node ⇒ a missing *local* metric |
| **(iii) uniform / fine-grained** | residual spread evenly across the corpus — no `z`-projection above the permutation null | **dense-residual probe** (fit dense model on bank residuals; its held-out gain estimates `I(Y;X|M_S)` realizable by `E_dense`) + the discovery **flux** `Ĝ(c)` | positive dense-residual with silent (i)+(ii) ⇒ the gap exists but is *partition-invisible* |

**The MOB power condition [derived].** The instability test detects a missing metric `m*` only if the
heterogeneity it induces **projects onto an observed `z`** with signal exceeding the permutation-Bonferroni
threshold at `(n, m_z, B)` — mechanically, `B > m_z/α − 1` permutations (the `n_perm ≥ 999` rule for
`m_z = 26`), and the projection effect size above the sup-LM null at `n`. Two corollaries, both bitten this
session: (a) a missing metric independent of every `z` axis is **invisible to partitioning by
construction** — it lands in shape (iii); (b) **the multiplicity tax is a design variable** — every column
offered as `z` divides α; offering the whole bank (levels + NA indicators, `m_z ≈ 48`) demands raw
`p < 0.001` (the permutation floor at `B = 999`) of any real moderator, while curating `z` to a few
*hypothesized item-level axes* (source, genre/topic cluster, length; `m_z ≈ 2–5`) — partykit's intended
design — leaves the same moderator detectable at `p ≈ 0.003`. A stump under bank-wide `z` is therefore
evidence of *no residual strong enough to survive an m_z≈48 Bonferroni*, a much weaker statement than
saturation. The battery is a **necessary-condition ladder**: silence on (i)+(ii)+(iii) + light flux = the
saturation read; silence on (i) alone = almost nothing.

**The CW case, corrected twice [measured, this session].** First correction: the algorithm is sound
(planted sign-flip break under `m_z = 26` detected at `n_perm = 999`, adj-p 0.026; correct stump on stable
data), so the CW stump was real, robust to `min_node_size ∈ {10..30}`, and the borderline adj-p 0.047 was a
3–4-item NA-indicator artifact. Second correction (the known-moderator control): CW's training file is a
concat of two halves with label rates 0.44/0.18 — a REAL moderator. With bank-wide `z` (`m_z = 48`) it
ranks 7th, raw p = 0.003, adj-p = 0.144: **invisible, purely from the multiplicity tax** (the permutation
floor itself is adj-p 0.048). With curated `z = {source_half, text_cluster}` the tree splits on
source_half (adj-p 0.014) and then twice on the 6-way text-embedding cluster inside the low-rate half —
depth-2 structure on real CW data. And the bank does NOT absorb the axis (raw label gap +0.26 → +0.21
after residualizing on all 26 metric levels): it is genuine (i)/(ii)-shaped residual the metrics miss. So
the earlier "CW has no partition-visible structure" verdict was **partly a z-design artifact**, and the
default design (all metric levels + NA indicators as `z`) is wrong: metrics belong in `X`; `z` should be a
small curated set of item-level axes. What remains true and substantive: in *rubric-covariate* space the
relationships are homogeneous (the pool's 73,702 rubrics cluster richly in criterion space while their
effect-on-Y is uniform — many near-synonyms, one shared relationship), so CW's remaining articulable gap is
carried by **item-population axes** (source, genre) plus a possible **many-small-`v` flux tail**, not by
rubric-level moderation. The holdout test (fresh 500 items) settled it: the tree **generalizes** — per-leaf
label rates transfer, global bank-GLM 0.588 vs tree-routed 0.706 (+0.118) — but the decomposition control
shows the gain is mostly the axes' *main effects* (axes-only 0.690; bank+axes additive 0.679), leaving a
**moderation-specific +0.027**. And `source_half` is partly an identity/mixture confound (two concatenated
datasets with different labeling processes — the publisher-id lesson), so it is a *nuisance to stratify by*,
not a discovered metric. The honest residual worth infilling: the text_cluster axis and the WRONG/RIGHT
contrast inside the low-rate-half leaves, under `curated_z_only` with source_half deconfounded.

**Why unprincipled deficit-hunting worked on code [measured, prior sessions].** Code's residual is
**region-shaped along axes humans already name** (language, domain, test-type — the F2P/P2F work found
signal exactly by conditioning on such axes), so eyeballing deficits = informally running instrument (ii)
with a well-chosen `z`. The formal pipeline's value is symmetric: it says when the eyeballing missed axes
(the multiplicity-tax lesson cuts both ways — informal search has no false-positive control at all).

**Why the CW ceilings are low [measured, prior sessions].** Three compounding attenuations, none of them
articulation failures: (1) **noisy gold** — spot-checks found ~67% top-10 sensible with few true misses;
label noise multiplies *both* `C` and `V(S)` down; (2) **base-population mixture** — the training file is a
concat of two halves with label rates 0.47 vs 0.15 full-file (0.44/0.18 in the 500-item sample); any
feature correlated with the half boundary carries spurious "signal," and the halves' different labeling
processes add noise — the §5 known-moderator control confirmed this axis IS tree-visible under curated `z`
and NOT absorbed by the bank; (3) **R2 over-splitting** — near-dup
rubrics fragment per-metric evidence (the CW taste-matching finding), deflating per-metric `T_j` and hence
inflating `N_lower` while deflating measured `V`. Under the direction-of-error discipline (§7), (1)–(3) all
*shrink the certified articulable side* — conservative for the taste headline, but they must be named,
because they also make "CW needs few metrics" partly an artifact of a small measurable ceiling rather than a
small lexicon.

---

## 6. The model-family axis. [holds — inherits PO §12.6.5 rules]

Per executor tier `E_t` (3B → 8B → 70B → 122B → API frontier): report the quadruple
`( V_t(S_g^t), U_t, N_lower^t, N_upper^t )` — each tier gets its own bank (metrics re-prompt-optimized per
tier; a fixed bank under-serves stronger executors).
- **Staircase, not fit:** the achieved frontier `max_t V_t` and the per-tier count brackets are points;
  fitted saturation asymptotes are never bounds (standing rule).
- **The verdicts:** `N_upper^t` flat across ≥ 3 tiers while the taste gap `lowerCI(C) − V_t` stays positive
  = the strongest process-relative form of "the lexicon is bounded and the practice exceeds it" (the
  human-exceptionalism read, correctly scoped). `N_lower^t` still growing with tier = right-censored: "not
  yet articulated at capability `E_L`," never "inarticulable, period."
- **Cross-tier hygiene:** per-metric rankings across tiers are not trusted below rank-agreement ≈ 0.4
  (capture–recapture memo); trends only. `B_E`-census growth with discrimination-plateau flags
  readout-capacity artifacts.

---

## 7. Direction-of-error and positive controls. [design — mandatory companions]

The anthropological framing flips the usual skepticism: **every instrument weakness inflates the gap the
thesis wants**, so credibility = positive controls (standing memo). Per failure:

| weakness | direction | control |
|---|---|---|
| under-optimized metric prompts | `V(S)` ↓ ⇒ taste gap ↑, `N_lower` ↓ | per-metric PO certificate (`T_i − R_i ≤ ε_i`) required for bank membership |
| judge score collapse (all-min structured output) | `T_j → 0`, metric falsely worthless | score-distribution check before any bank fit (standing memo) |
| MOB under-powered (`n_perm`, `m_z`, `n`) | stump ⇒ false "(i) silent" | planted-break control (passes: split at adj-p 0.026 under `m_z=26`); p-floor printed with every run |
| bank-wide `z` multiplicity tax | real moderator invisible (CW: p=0.003 → adj-p 0.144 at m_z=48) | curate `z` to few item-level axes; known-moderator control must PASS under the shipped z-design (CW: source_half splits at adj-p 0.014 under curated z) |
| missing `z` axes | (i)+(ii) silent ⇒ residual misread as uniform | known-moderator control: plant a REAL axis (e.g. `source_half`, base rates 0.44/0.18) in `z`; the tree must split on it |
| over-merge in the quotient | flux `w_1` ↓ ⇒ `N_upper` too small (anti-conservative!) | merge-precision audit is the binding gate (PO §12.6.2); `w_1` at two probe sizes |
| noisy gold / label mixture | `C` and `V` both ↓; gap direction ambiguous | deconfound + denoise first; report bracket under a label-noise sensitivity band |
| dense under-trained | `U_dense` invalid (bank can beat dense) | dominance gate: dense scaling plateau + `V(S_g) ≤ C − margin`; else right-censor (CW today) |
| planted-bank calibration | whole pipeline | a bank of CODE metrics (V-layer) must recover `V →` its known value and the battery must stay silent on residuals it provably lacks |

---

## 8. Scorecard

| # | claim | status | measure |
|---|---|---|---|
| C1 | `V(S)=I(Y;M_S)` monotone ⇒ Minoux/`U₂` clean on raw objective | **holds** | held-out log-loss reduction + CIs |
| C2 | value bracket `V(S_g) ≤ A*_E ≤ min(U_dense, U_flux)`; `U_dense` needs dominance gate (CW fails it today) | **holds / conditional** | stack model; dense scaling curve; flux D1–D3 with supervised `v` |
| C3 | count necessity `N ≥ V_bits / log₂K` (assumption-free); `N ≥ V_bits / T_max` (needs per-metric-call scoring + `T_max` census) | **derived** | per-metric `T_j` (orbit-averaged), bank `V` in bits |
| C4 | count sufficiency / collectibility `N_δ ≤ (U − V(S_g))/δ` on the quotient | **derived, conditional (wrap + merge-precision + per-metric optimality)** | acceptance-gate δ in bits; quotient census |
| C5 | residual trichotomy: MOB ⇒ (i) only; stump ≠ saturation; battery = (i)+(ii)+(iii)+flux | **derived + measured** | ctree, gap flags, dense-residual probe, `Ĝ(c)` |
| C6 | CW: algorithm sound; earlier "no structure" verdict partly a z-design artifact (bank-wide z's multiplicity tax hid a real moderator: p=0.003→adj-p 0.144 at m_z=48; curated z splits at 0.014, depth 2); holdout: tree generalizes (+0.118 AUC) but mostly axes main effects (moderation-specific +0.027) and source_half is a mixture confound → nuisance, not metric | **measured (this session + priors)** | diag scripts; `source_half` control; holdout AUC + additive control; spot-checks |
| C7 | mega-metric degeneracy ⇒ counts well-posed only over the articulable class (R≈T + atomic quotient) | **derived** | recovery certificate as membership test |
| C8 | family axis: per-tier staircase of `(V_t, U_t, N^t_lower, N^t_upper)`; flat `N_upper` + persistent gap = the bounded-lexicon finding | **design** | ≥3 tiers; slope CIs |
| C9 | certificates generator-agnostic (validity) / completeness generator-relative; union ledger over `|G| ≥ 2` arms is the certified artifact | **derived + implemented (§2a)** | `generators.py` arms through one gate; `report_from_ledgers` union certificate |
| C10 | gate validity: δ > instrument noise floor AND selection-deflated (fresh-confirm or `α/J`) — else `N_upper` overcounted | **measured (§4.4; both violations observed then repaired)** | paired-CV gate; winner's-curse test read (CV +0.018 → test −0.016 at J=20) |

---

## 9. Honest limitations

1. **Everything beyond `U_dense` is process-relative** (PO §12.5): `N_upper` counts what the *named*
   proposer/discovery process can reach at the stated horizon. The zero-mass tail — a criterion no family
   can emit — is invisible at any horizon; a human/expert elicitation list is the only ceiling-raiser, and
   is out of scope by design (no realtime elicitation).
2. **`U_dense` inherits every confound of `C`** — publisher-id, topic, position (press-release lesson).
   An inflated `C` inflates `N_upper` and the taste gap together; deconfounding is prior to certification.
3. **Bits vs AUC:** the count certificates only compose in bits; AUC gaps do not divide by δ meaningfully.
   The bank readout must add log-loss reduction. Until then the packing bound is heuristic.
4. **γ̂ synergy wall, softened not removed:** with labels, pairwise synergy is *detectable* (residual
   interaction scan; the composite path with inverted-polarity recovery) and composites can be added to the
   ground set, but there is no theorem that composition closes the class; ctree's §9 limitation (XOR of two
   *absent* features) still holds at discovery level. The γ-free backstop is the residual battery.
5. **`N` is executor-relative and `K`-relative by construction.** There is no "the" number of metrics for a
   preference; there is `N_E(ε)` at arity `K` — state all three indices every time.
6. **Finite-sample:** every MI/AUC is a plug-in; Miller–Madow + bootstrap CIs; the acceptance gate δ must
   dominate estimation noise on the guard split, else "collected" metrics are noise (the current CW δ was
   set for exactly this reason — and §4.4 now records the two ways this bit us empirically).

---

## 10. PO ↔ MCC crosswalk — every tie-in, explicit

The two frameworks are one theory at two levels. Object-by-object:

| PO object (per-metric, no `Y`) | MCC object (bank, with `Y`) | relation |
|---|---|---|
| latent metric `M_i` (the target) | revealed-preference record `Y` | the estimand moves one level up |
| `R ≤ T` bracket (recovery ≤ transmission) | `V(S_g) ≤ A*_E ≤ U` | same shape: achieved floor ≤ estimand ≤ instrument-relative ceiling |
| `T_i = I(m̄_i; X)` per-metric transmission | `Σ_j min(T_j, log₂K)` — the necessity denominator (§4.1) | PO's `T` feeds MCC's `N_lower` directly; PO certifies the per-metric leaves MCC composes |
| recovery certificate `R ≈ T` (PO §11) | `𝒜_E` membership test (§1) | PO's *quality* certificate becomes MCC's *degeneracy guard* — the single deepest tie-in |
| `B_E` census: criteria to express ONE metric | `N_E(ε)`: metrics to express THE PREFERENCE | same census machinery, one level up ([[feedback_alpha_probe_is_metric_level]] — never conflate) |
| quotient partition (PO §12.6.1: judge may merge, behavior may split) | δ-distinctness in `N_upper`; atomicity in `𝒜_E` | inherited verbatim; merge-precision stays the binding gate |
| capture–recapture flux D1–D3, Good–Toulmin `Ĝ(c)`, value = recovery gain (proxy) | same estimators with supervised `v(s\|S_g) = I(Y; s \| M_S)` | U3: the value measure stops being a proxy |
| orbit-averaged readout `m̄`, adverse-orbit reporting (PO §12.6.2) | inherited unchanged (§2) | — |
| process-relativity (PO §12.5, named generator set) | generator abstraction §2a: interface contract, per-arm recapture lists, union ledger | MCC *formalizes* what PO states as a scoping rule |
| singleton-degeneracy lemma (`f₁/N → 1`) | reported beside every flux read | inherited |
| monotonization caveat (`R(S)` non-monotone, PRUNE) | **dissolves**: `V` monotone (U1) → Minoux/`U₂` clean | the one place MCC is *cleaner* than PO |
| `cap_f` (vacuous readout ceiling) | `I(Y;X)`, task-intrinsic, dense-estimable (U2) | the one place labels buy real tightness |
| per-metric headroom `ε_i = T_i − R_i` (GEPA, PO §11.1) | enters floor validity (§3) and the §4.5 chain at step 2 | an under-optimized bank *undercounts* articulable signal (§7 row 1) |
| — (no analog) | mega-metric degeneracy (D1) | the one genuinely new failure mode labels introduce |
| PO §12.6.6 verdicts (CODIFIABLE / DEEP / UNDERSAMPLED / FORM-DOMINATED) | §4.3 thin-vs-deep bank read + §5 battery verdicts | the decision rule, lifted to the bank level |

Reading the table columnwise gives the composition law: **MCC's bank certificate is a conjunction whose
leaves are PO per-metric certificates** (§4.5). A bank claim is never stronger than its weakest member's
PO certificate.

---

## 11. Empirical status — first end-to-end run of the chain (2026-07-02)

The §4.5 chain has now executed once, end to end, on two tasks (CW WritingPrompts, ICLR peer-review;
medoid banks of 40 from 73,702 / 75,649 rubric pools; `n ≈ 420` discover+guard, 180 test; glm-5.2 judge +
proposer; three arms × two tasks; artifacts under `outputs/ctree/arm_comparison/`).

- **Result: 20 proposals, 3 arms, 2 tasks, 0 survivors of the full chain.** CW 0/6 at the CV gate;
  peer-review 1/8 (label_contrast arm, "Explicit Analytical or Mathematical Contribution": CV +0.018 AUC /
  +0.0065 bits, applicability 0.98, reconstruction `R = 0.952`) — which then **failed step 5**, the
  untouched test (−0.016 AUC ± 0.042): winner's curse at `J = 20`, exactly the §4.4.2 failure mode
  (observed before the confirmation stage existed; step 4 is the repair).
- **Substantive read:** at this design's ~0.01-bit detection floor, *no* arm — residual-targeted, naive
  label-contrast, or data-free autorubric — finds a single new articulable criterion with detectable
  held-out label-signal on either task. Consistent with a thin/flat flux tail (many small-`v` species or
  none): the §4.3 "deep" signature, pending the bigger-`n` / joint-set-acceptance designs below.
- **Certificates emitted correctly:** wraps right-censored on both tasks (dense not plateaued — no
  `N_upper` published), `T_max` leg refused (shared-call scoring), per-arm reports carry the single-arm
  honesty note. Gap found in review: no **union-ledger** certificate was emitted (§2a requires it) —
  fixed; `report_from_ledgers` now produces the multi-arm artifact.
- **What raises detection power next:** larger `n` (test SE ∝ 1/√n); **joint set acceptance** (accept a
  SET of small metrics whose combined Δbits clears δ — single-metric gating structurally cannot see the
  §4.3 tail); per-metric `T_j` certificates in place of bank-marginal gains; and the step-4 confirmation
  stage before any test read.

Full run log: `running-research-notes.md` 2026-07-02 entries.
