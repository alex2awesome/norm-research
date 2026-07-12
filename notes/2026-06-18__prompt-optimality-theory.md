# Global optimality of an unsupervised-metric prompt — what can be certified about `p*`

*2026-06-18, rewritten 2026-06-19. The question this file answers: we search for a prompt `p` that
maximizes an **unsupervised** recovery metric `R(p)` (no label `Y`, no strong-LLM anchor). The
candidate set is **infinite** — all strings. Given a found prompt `p̂`, **how close is it to the
global optimum `p* = argmax_p R(p)`, and what can we actually certify?***

*Every claim is tagged **[holds] / [corrected] / [conjectural]** with a falsification column. The lead
(§1–§4) is the global-optimality ladder, focused on `p*`. The articulation gap `A = T−R` (§5) and the
greedy rubric-composition certificate (§6) are demoted: `A` is a measurement *of the metric*, not a
certificate *for the prompt*, and greedy is only the within-class rung restated in full.*

*Companion (2026-07-01): `2026-07-01__metric-count-certificates.md` lifts this machinery one level — from
"how good is the prompt for ONE metric (no label)" to "how much label-signal can a BANK of optimally-prompted
metrics carry, and how many metrics does a model family need / can it collect (labels `Y` available)."*

> **Validation correction (2026-07-10).** Three objects in this note must be kept separate.
> 1. For a **fixed target** `M_omega`, the DPI statement `R(p) <= I_f(M_omega;X)` is valid over every
>    candidate prompt satisfying the held-out Markov condition. Equality is conditional on the allowed
>    prompt/readout class realizing a sufficient posterior statistic; the experiments have not established
>    that condition. Thus `T(M_omega)` is a global upper bound, not automatically the exact optimum.
> 2. `OPT_Omega` from a greedy head is an achieved checklist value (or, under exhaustive enumeration, an
>    exact empirical optimum only within the declared finite class). It is not an upper bound over prompts.
> 3. The §12.6 `OPT_Omega + epsilon` bridge is **not presently certified**. Its code uses an adaptively
>    selected head, a horizon point predictor without a one-sided horizon deviation bound, and a measured
>    `gamma_hat` that is not a lower confidence bound for unseen synergy. The implemented tail-XOR breaker
>    demonstrates under-coverage. Until those gaps are closed, `epsilon` is a process-horizon diagnostic and
>    must not be plotted or reported as a ceiling. The matching code now emits `upper_bound_valid=False`.
> 4. **CR-2 (2026-07-10, `experiments/cr_horizon.py`) is the successor to the §12.6 bridge** — the same
>    capture-recapture idea rebuilt at ESTIMATE grade with a measured credential instead of a claimed bound:
>    probe-split freeze, permutation-null gates on every statistic, pair-synergy chain, REFUSED unseen-pair
>    extrapolation (it targets the declared hidden-partner blind spot), and a (kappa, lambda) calibration fit
>    on the train split of a planted battery against the analytic POOL truth `I(M; noisy units)` — reported
>    on the disjoint test split (out-of-sample). Two truths are now distinguished: POOL truth (what the
>    mining process can ever deliver — cr2's target) vs CONCEPT truth `I(M;R)` (what a better instantiation
>    could deliver — `concept_horizon`'s target, EVT over T across instantiations). Scope is generator-side
>    (reachability within ~10x the stream); unreachable species, parity depth >= 3, and hidden-partner
>    synergy are declared, measured blind spots. See `notes/2026-07-10__cr2-horizon-rehabilitation.md`.
> 5. **Current correction (2026-07-12): the authoritative bound-grade capture result is §12.6b.** CR-2
>    remains the historical planted-calibrated estimate described above. CR-3 targets best-single-prompt
>    recovery directly: a fresh stratified audit gives a finite-horizon expected-best bound, while exact-
>    pattern missing mass yields a support result only under an external minimum-mass assumption.

---

## 0. The answer in one paragraph

A global optimality certificate for a found prompt `p̂` is a **pair of bounds on
`OPT := sup_p R(p)`**: the lower bound `R̂(p̂) ≤ OPT` is free; the work is a *verified upper bound*
`U ≥ OPT`, and the **certified gap** `U − R̂(p̂)` is an *upper bound* on the true gap `OPT − R̂(p̂)`
(which may be much smaller). The whole theory is the ladder of available `U`'s.
The result is sobering and clean: **without additional structure on `B_E`, the strongest global upper
bound on `OPT` over the full infinite string set is the information cap** (`1 − 1/min(N,K)` for our
TVD-MI objective — binary **½**; `log min(N,K)` in Shannon bits) — proven via "transmission is convex" +
"recovery ≤ transmission", and it is loose: a **channel-capacity sanity check**, not a
proximity-to-optimum KPI (§3.1). (Other *structural* assumptions — model-capacity,
grammar, Lipschitz — could yield other global bounds; the cap is the strongest *assumption-free* one.) Every *tighter* certificate buys tightness by **shrinking the scope**
— either restrict prompts to a criterion class (`U₂`, a tight per-instance submodular bound, but only
global *within the class*) or to a finite tested set (`U₃`, statistical best-in-set). Below those sit
local necessary conditions (`U₄`). The reason there is no free tight global certificate is **structural,
not a lack of effort**: it is our ignorance of `B_E`, the set of behaviors any prompt can induce on
executor `E`, which we can only sample, never enumerate.

---

## 1. Setup, objective, and what "global optimality" means

**The unit of analysis is ONE metric — read this first.** Everything below scopes to a SINGLE latent
evaluation metric `M_i`, one per R2 cluster (e.g. "novelty of contribution" in peer review). A task has `j`
such metrics `M_1,…,M_j`; the observed outcome `Y = f(M_1(x),…,M_j(x))` (e.g. accept/reject) is their
AGGREGATE and is **not** the target here — the recovery loop below is anchor-free and label-free (**no `Y`**).
`M_i` is itself unlabeled; we recover it from its OWN `(item, verdict)` pattern. Every symbol — `R`, `T`,
`Ω`, `B_E`, `α`, `γ`, `A` — is **per-metric** (`R_i`, `T_i`, `α_i`, …); repeat the whole analysis for each
metric. **This is load-bearing and recurs as a bug:** at the TASK level (pooling every metric's criteria) the
criterion universe is always vast, so `α ≈ 0.9` *everywhere* — non-discriminating. The structure (thin vs
thick, coverable vs inexhaustible) lives only at the PER-METRIC level: `α_i < 0.5` for a simple metric,
`α_i → 1` for a complex one. §12's ALPHA-PROBE and VALUE-CENSUS run **once per metric `M_i`** (per R2
cluster), never once per task.

| symbol | meaning | scope |
|---|---|---|
| `M_i` (≡ `M` when context is clear) | ONE latent evaluation metric (e.g. "novelty"), **unlabeled** | per-metric — repeat for each |
| `Y = f(M_1,…,M_j)` | observed AGGREGATE outcome (accept/reject) | task-level; **not** the recovery target |
| `Ω_i` | the atomic criteria that compose/express `M_i` (mined per-metric, §6.5) | per-metric (each `M_i` has its own) |
| `M_ω` | `M_i` recompiled from its own criteria `Ω_i` (§6.7a′) | per-metric |
| `R_i`, `T_i` | recovery `I(M_i;M′_i)` / transmission ceiling `I(M_i;X)` | per-metric |
| `α_i`, `γ_i`, `A_i` | Heaps exponent / submodularity ratio / articulation gap `T_i−R_i` of `M_i` | **per-metric** (task-level `α≈0.9`) |
| `B_E` | the behaviors a prompt can induce **for `M_i`** | per-metric |

> **Notation correction (2026-07-12; original setup retained).** The historical text overloads `M_i`.
> Going forward, use `M_i*` for the prompt-independent ideal and `M_{i,b}` for a frozen operational target;
> `M_ω` is one possible `M_{i,b}`. Executable recovery sections target `M_{i,b}` unless they separately
> identify `M_i*`. Every inequality must name which target it uses, and an observed `Y` is only a proxy view
> after a measurement relation is declared.

**Items.** `X_1,…,X_N` i.i.d.; item index `I ~ Uniform[N]` (fixed — this matters for the convexity).

**Behavior.** A prompt `p` makes executor `E` a stochastic map item→verdict. Binary:
`s_p = (s_1,…,s_N) ∈ [0,1]^N`, `s_i = P(verdict=1 | item i, p)`. K-ary: `s_p ∈ (Δ_K)^N`. `s_p` is the
**induced behavior**; write the behavior space `C := (Δ_K)^N` (a product of simplices, convex,
compact).

**The achievable-behavior set.** `B_E := { s_p : p ∈ Σ* }`, where `Σ*` = all finite strings. Three
facts about `B_E` drive everything:
- it is a **countable** subset of `C` — countable *because the prompt domain `Σ*` is countable* (finite
  strings over a finite alphabet); so although `C` is a continuous cube, the image `B_E` of `p↦s_p` is
  countable (one behavior per string, modulo ties),
- it is **not convex**,
- **we have no explicit description of it** — only a *sampling oracle*: hand it a prompt `p`, observe
  `s_p` up to estimation noise. We never get to enumerate or even bound `B_E` analytically.

**The objective (unsupervised) — the mechanical recovery loop.** Split items into a *train* split
`X_train` and a disjoint *held-out* split `X_test`. The loop that *defines* recovery (no anchor `M`, no
label `Y`):
- **In-sample observation** `M_train = m(X_train)` — run the candidate prompt `p` on the train items.
- **Reconstruction** `ŝ = Reconstruct(X_train, M_train)` — a reconstructor reads the `(item, verdict)`
  pairs and writes a *new* prompt `p̂`; it never sees how `M` was generated (§5.5).
- **Out-of-sample execution** `M_test = m(X_test)` (run `p` on held-out) and `M̂_test = exec(ŝ, X_test)`
  (run the reconstructed `p̂` on the **same** held-out items).
- **Recovery** `R(p) = I_TVD(M_test ; M̂_test)` — the TVD-mutual-information between the original and the
  reconstructed verdicts, **on the held-out split**.
`R` is label-free and anchor-free; it is the flagship "articulability via articulate-then-re-execute"
quantity, and it reads high only when `p` executes a **generalizable** rule (`R≈cap`) rather than
memorizing train surface features (`R→0`). We optimize `R` over prompts.

**The transmission shadow — and the distribution it must live on.** Alongside `R` define transmission
`T(p) = I_f(I; V)`, the f-mutual-information between item index and verdict under a *single* behavior
`s_p`. **Two transmissions must not be conflated** (a real slip in the prior draft, corrected per the
2026-06-22 critique):
- **In-sample consistency** `T_train` — `p` re-applied to its *own* train items across `K` passes; the
  code's `I_V = H(p̄) − (1/N)Σ_i H(p_i)`, `p̄=(1/N)Σ_i p_i` (`channel_consistency`). A reliability
  number — **not** a ceiling on the held-out `R`.
- **Held-out transmission** `T_test = I_f(M̂_test ; X_test)` (and `I_f(M_test ; X_test)`) — the
  transmission of the verdicts **on the same held-out split `R` is evaluated on**.
The data-processing inequality `R ≤ T` is a statement *within one distribution*: it bounds the held-out
`R` only by the **held-out** `T_test`, never by the train-split consistency `T_train` (DPI does not chain
across two distributions — §2.2). **`T` is not the objective** — it is the quantity whose convexity we
can prove, and via `R ≤ T_test` it is the tool that bounds `R`.

**The optimization problem.** By the reduction "a prompt matters only through `s_p`" (two prompts with
the same behavior are interchangeable),
`OPT := sup_{p∈Σ*} R(p) = sup_{s ∈ B_E} F(s)`, where `F(s)` is recovery as a function of behavior.
`p*` is any (near-)maximizer.

**Definition (global optimality certificate).** For a found `p̂`, an **ε-global certificate** is a
verified `U ≥ OPT` together with the evaluated `R̂(p̂)`, such that `U − R̂(p̂) ≤ ε` (up to the CI on
`R̂(p̂)`). `p̂` is then **ε-globally optimal**: no prompt anywhere in `Σ*` beats it by more than `ε`.
*The lower bound is trivial; the entire difficulty is producing `U`.* The ladder of §3 is the ladder
of available `U`'s.

**A note on "V-information."** Our `I_V` is **Shannon** MI with a plug-in, *not* the Xu-et-al.
predictive-V-information functional; the "V" (what `E` can express) lives in the **channel** `s` (E can
only realize `B_E`), not in the entropy. So the Shannon/f-divergence convexity used below applies
cleanly, consistent with the standing memo to use Shannon/TVD transmission rather than raw `I_V` and to
avoid V-information's non-Lipschitz pathologies.

---

## 2. The two facts the whole ladder rests on

### 2.1 Fact 1 — transmission is convex in behavior. **[holds, general f]**

> Let `I ~ Uniform[N]` be fixed, channel `s = (p_1,…,p_N) ∈ (Δ_K)^N`. For any convex `f` with
> `f(1)=0`, `I_f(I;V) = D_f(P_{I,V} ‖ P_I⊗P_V)` is **convex in `s`**. In particular Shannon `T`
> (`f=t log t`) and TVD-MI (`f=½|t−1|`) are convex in `s`.

*Proof.* `P_{I,V}(i,v)=(1/N)p_i(v)` is affine in `s`; the product marginal `(1/N)p̄(v)` is affine in
`s`. `(P,Q)↦D_f(P‖Q)` is **jointly convex** (perspective `g(x,t)=t·f(x/t)`). Jointly convex ∘ affine =
convex. ∎

Cleaner and stronger than citing Cover–Thomas 2.7.4 (the `f=KL` case): convexity is the f-divergence
fact, so it survives the switch to TVD-MI, where it is visible by eye —
`I_TVD(s) = (1/2N)Σ_i ‖p_i − p̄‖_1` is a sum of norms of affine maps, hence convex (binary:
`(1/N)Σ_i|s_i − s̄|`).

### 2.2 Fact 2 — recovery is upper-bounded by transmission (DPI), on the held-out split. **[holds, general f]**

**All quantities here live on the held-out split `X_test`** — the *same* split `R` is evaluated on (this
is load-bearing; see the caveat below). Held-out item `i`: `X_i → (M_i, M̂_i)`, `M_i = m(X_i)`,
`M̂_i = exec(ŝ, X_i)` with `ŝ` learned on a *disjoint* train split, so `ŝ ⟂ X_i^{held-out}`. Hence the
common-cause structure `M_i ⟂ M̂_i | X_i` (the item content is the only shared parent).

> For **any** f-divergence, **both legs on `X_test`**: `R = I_f(M_test; M̂_test) ≤ I_f(M_test; X_test) =
> T_test(m)`, and symmetrically `R ≤ T_test(m̂)`.

*Proof.* The f-DPI: `D_f(K∘P ‖ K∘Q) ≤ D_f(P‖Q)` for any channel `K`. Take `K:(m,x)↦(m,M̂)` — keep `m`,
replace `x` by `M̂ ~ exec(·|x)` (legitimate because `M̂ ⟂ M | X`). Then `K∘P_{M,X}=P_{M,M̂}` and
`K∘(P_M⊗P_X)=P_M⊗P_{M̂}`, so `I_f(M;M̂) = D_f(K∘P_{M,X}‖K∘(P_M⊗P_X)) ≤ D_f(P_{M,X}‖P_M⊗P_X) =
I_f(M;X)`. ∎
*(The intuitive Shannon proof — `I(M;M̂)≤I(M;M̂,X)=I(M;X)` via `I(M;M̂|X)=0` and the chain rule — does
**not** transfer to TVD-MI, which has no chain rule. The f-DPI above is the one that does.)*

**Same-distribution caveat (the load-bearing correction, 2026-06-22).** DPI is an inequality *within one
distribution*: the channel argument requires `M_test`, `M̂_test`, `X_test` to be the **same** held-out
items. It does **not** license bounding the out-of-sample `R` by the *train-split* in-sample consistency
`T_train` of §1 — they live on different distributions and DPI says nothing across the two. So the
certificate and the cell-level guardrail (§5, §9) must read `R` against the **held-out**
`T_test = I_f(M̂_test; X_test)`, which is exactly what `vinfo.tvd_guardrail` computes (both legs from the
held-out recovered verdicts). `T_train` is a separate reliability readout, never a DPI ceiling on `R`.

*(Notation: `X` is `X_I`, the content of the random held-out item. On distinct-content held-out samples
`X_I ↔ I` is a bijection, so `I_f(M;X_I) = I_f(M;I) = T_test(m)`. The identification with the measurable
`I_V` needs distinct content — and the held-out split.)*

### 2.3 The asymmetry that shapes everything: `T` is convex, `R` is not (known to be). **[holds]**

This is the crux and the old draft glossed it. **`T` is provably convex in `s` (Fact 1).** **`R` is
*not* known to be convex in `s`**: `R = I(M;M̂)` depends on `s_p` through a *nonlinear, learned*
reconstruction map (the reconstructor reads `m`'s behavior and induces `ŝ`), so none of Fact 1's
machinery applies to `R` directly. Consequence:

> We cannot do clean convex-maximization on the objective `R` itself. The convexity is only available
> on its **upper bound** `T`. So the global story is forced into the shape: *use `T`'s convexity to cap
> `R`, and search for `R` directly with only local/statistical certificates.* Every structural claim
> below (extreme point, ½/½ shape, the cap) is proven **for `T`**; for `R` it transfers only through
> `R ≤ T`, never by convexity of `R`.

---

## 3. The ladder — certifying `p*`, step by step

Each rung is an upper bound `U ≥ OPT` (or, for rung 3, a within-set statement). They run from
**most global / loosest** to **most local / tightest scope**. Tightening always costs scope.

### 3.1 Rung 1 — the universal information cap. The strongest *assumption-free* global `U`. **[holds]**

> **`U₁ = cap_f := sup_{s∈C} I_f(I;V)`**, the maximum transmission any behavior can carry. For every
> prompt `p`: `R(p) ≤ T(p) ≤ cap_f`, hence `OPT ≤ cap_f`. The value is **measure-dependent** (DPI
> chains only within a fixed `f` — you may not put Shannon on one side and TVD on the other):
> - **TVD-MI (our objective):** `cap_TVD = 1 − 1/min(N,K)`  (binary: **½**).
> - **Shannon bits:** `cap_Shannon = log min(N,K)`  (binary: **1 bit**).
>
> **Use `cap_f` as a channel-capacity *sanity check*, not a proximity-to-optimum KPI** (downgraded
> 2026-06-22). `cap_f` is the maximum a `K`-symbol verdict channel can carry (½ for binary) — a constant
> of the *readout*, not of the task. So `cap_f − R̂(p̂)` is a mathematically valid upper bound on
> `OPT − R̂(p̂)` but **scientifically vacuous**: at `R̂=0.48` the "certified gap" `0.02` says nothing
> about nearness to the *task* optimum (the task's own ceiling `OPT` is typically `≪ cap_f`). Its only
> legitimate jobs: (i) verify `R̂ ≤ cap_f` — a violation flags an estimator/readout/leak bug; (ii) detect
> that the binary readout is compressing headroom (`R̂` near ½ ⇒ switch to a `K`-ary scale, which lifts
> the cap to `1−1/K`). **Do not report `cap_f − R̂` as "distance to optimum."**

*Proof.* `R ≤ T` is Fact 2 (both sides the same `f`). `T = I_f(I;V)` is convex in `s` (Fact 1), so its
max over the behavior cube `C = (Δ_K)^N` is at an extreme point — a **deterministic per-item labeling**
(§4.2) — and among those the balanced labeling maximizes it. Evaluating at the balanced deterministic
labeling: Shannon gives `H(p̄) = log min(N,K)`; TVD gives the Gini index `1 − Σ_m P_M(m)²`, maximized at
`1 − 1/min(N,K)`. Taking sup over `p` preserves `OPT ≤ cap_f`. ∎

*The clean fact behind the TVD case:* for any deterministic labeling, **`I_TVD(I;V) = 1 − Σ_v P(V=v)²`** —
the TVD-mutual-information of a labeling *is* the Gini impurity of its verdict histogram (the item index
collapses out exactly). Gini is maxed by a uniform histogram over as many distinct values as can be
populated, `min(N,K)`, giving `1 − 1/min(N,K)`.

**Intuition — `K`, not `N`, sets the ceiling (and why N→∞ does not lift it).** The verdict is a *channel*;
a binary verdict is a **2-symbol channel**, and `I_f(I;V)` measures the *fraction* by which knowing `V`
narrows *which item* — a 2-valued verdict halves the candidate set **at most**, for *any* `N`. Growing
`N→∞` pours more inputs into the same narrow channel without widening it: you sort `∞` items into 2 bins,
and the bin still cuts candidates in half — a fixed *fraction*, which is what MI is. `N` enters **only**
through `min(N,K)`, biting solely when `N<K` (too few items to populate the classes). For recovery
`R=I_TVD(M;M̂)` with binary `M` this is a `2×2` MI whose ceiling is ½ by `M`'s support alone — **`N` is the
*sample size* for estimating it, not a dimension of it** (more items ⇒ tighter CI on `R̂`, *same* ½ ceiling;
*total* information scales with `N`, the *intensive* per-comparison MI we report does not). **The lever for
headroom is granularity `K`** (`1−1/K`: 5-point→0.80, 10-point→0.90, continuous→1), not `N`. *Consequence
for our pipeline:* the **median-split-to-binary readout caps every recovery number at ½ by construction**;
a `K`-ary scale or a ranking lifts the cap to `1−1/K` and exposes top-end discrimination that binary
compresses away — worth switching the readout to when we need room to separate strong prompts near the top.

**Why this is the best label-free *global* bound, and why it's loose — the structural frontier.**
`U₁` is `sup_{s∈C} T(s)` over the **full cube** `C ⊇ B_E`, attained at a balanced deterministic vertex
(§4.2). The honest bound we'd *want* is `sup_{s∈B_E} T(s)`, which can be strictly smaller — but
**computing it requires knowing `B_E`, which we cannot.** Concretely:

- *Branch-and-bound can't help for free.* Because `T` is convex (Fact 1), convex-maximization
  branch-and-bound over `C` gives a valid, converging upper bound on `sup_C T` — but it converges to
  `U₁` itself, because `B_E` is invisible to it; B&B over the relaxation `C` cannot exploit `B_E ⊊ C`.
  To beat `U₁` you must *model* `B_E` (which behaviors a string can actually induce on `E`) — and
  label-free, with only a sampling oracle, we have no such model.
- *Therefore:* **`U₁` is the tightest global certificate obtainable without either (a) a model of
  `B_E`, (b) a class restriction (rung 2), or (c) a finite candidate set (rung 3).** This is the
  fundamental ceiling on global prompt optimality for an unsupervised metric. It is loose precisely
  because most real metrics have `OPT ≪ cap_f` (their target attribute simply does not discriminate
  every item) — and label-free we cannot certify how far below the cap `OPT` sits.
- *A model-aware cap (e.g. via the output-embedding spectrum `σ(W_out)`) is the right frontier but the
  wrong fix for **this** looseness.* Bounding `B_E` by the model's logit geometry gives a **model-capacity**
  ceiling — but a balanced binary split is trivially inside any real model's capacity, so for the binary
  verdict the spectral bound is **slack** and won't move `cap_f = ½`. The binding looseness is not the
  model's expressivity; it is the **target attribute's recoverability**, which is *task/label-dependent*
  (it is the construct gap `C−B`, not a logit-geometry quantity). So `σ(W_out)`-style bounds tighten a
  *different* ceiling than the one that matters here. Modeling `B_E` is genuinely the way to beat `U₁` for
  rich K-ary readouts; for the binary recovery objective it is not the lever.
- *The right object to tighten toward is the **task data distribution's intrinsic recoverable information**.*
  `cap_f` is a distribution-free upper bound; the operative ceiling is how much the *attribute itself*
  varies across the data — a property of the task's semantic complexity, not of logits or prompts. We do
  not claim global certification is "impossible" or "illegal": Rung 1 **is** an ironclad, valid certificate
  over all of `Σ*` — it is simply *unaligned with model behavior* and therefore slack. A tight, actionable
  certificate is bought by **structurally downscoping** to a finite ground set (rung 2) or candidate list
  (rung 3). This is a **utility trade-off (validity-vs-tightness)**, not a legality question.

**What rung 1 *does* deliver for `p*`:** if you find `p̂` with `R̂(p̂)` near `U₁`, that *is* a global
optimality certificate — `p̂` is within `U₁ − R̂(p̂)` of the best prompt that could ever exist. This is
the **only** rung that says anything about the infinite set. It is also the rung most likely to be
vacuous. Both are true at once.

### 3.2 Rung 2 — restrict the class → a tight per-instance bound (global *within the class*). **[holds]**

Give up "all strings"; restrict to rubrics expressible as a **criterion subset** `S ⊆ U`, where
`U = {x_1,…,x_n}` is a fixed ground set of candidate criteria, budget `|S| ≤ k`. Let
`R(S)` = recovery of the canonical rubric realizing exactly `S`, and
`OPT_class := max_{|S|≤k} R(S) ≤ OPT`.

> **Instance-specific upper bound.** Run greedy → `S_g`. With marginal gains
> `δ(e) := R(S_g∪{e}) − R(S_g)` for `e∉S_g` and submodularity ratio `γ∈(0,1]`:
> **`OPT_class ≤ U₂ := R(S_g) + (1/γ) · Σ_{j=1}^{k} δ_{(j)}`**, the `k` largest `δ`'s.
> At `γ=1` this is the classical **Minoux/Lagrangian** online bound; the `1/γ` form is a **corollary of
> the submodularity-ratio definition** (derived below), *not* a separately-named theorem — the famous
> weakly-submodular result is the *multiplicative* `(1−e^{−γ})` (§6.2), a different bound.

*Proof (3 lines; verified — this `1/γ` additive form is the highest-risk claim in the file, so the
derivation is given in full rather than cited).* Let `S*` be the optimum, `|S*|≤k`.
1. **monotonicity:** `R(S*) ≤ R(S_g∪S*) = R(S_g) + [R(S_g∪S*) − R(S_g)]`;
2. **the `γ` step is the *definition* of the submodularity ratio** — `γ` is the largest constant with
   `Σ_{e∈Ω}[R(S∪e)−R(S)] ≥ γ·[R(S∪Ω)−R(S)]` for all `S,Ω`; at `S=S_g, Ω=S*∖S_g` this rearranges to
   `R(S_g∪S*) − R(S_g) ≤ (1/γ)·Σ_{e∈S*∖S_g} δ(e∣S_g)`;
3. **top-k:** `|S*∖S_g| ≤ k` and each `δ≥0`, so `Σ_{e∈S*∖S_g} δ ≤ Σ_{top-k} δ`.
Chaining: `OPT_class = R(S*) ≤ R(S_g) + (1/γ)·Σ_{top-k} δ`. ∎ (Sanity: `γ≤1 ⇒ 1/γ≥1 ⇒` looser bound, as
a less-submodular function should give. Requires `γ>0`; degenerates as `γ→0`, the synergy limit of §6.6.)

**Caveat (monotonicity).** The proof uses monotonicity in the step `R(S*) ≤ R(S*∪S_g)`. But raw `R(S)`
is **non-monotone** (§6.1, PRUNE). So this instance bound applies cleanly only to the free-disposal
monotonization `R↑(S) = max_{T⊆S} R(T)` (an *upper* idealization — see §6.1), or to raw `R` only where
monotonicity has been checked on the corpus. For genuinely non-monotone `R`, the certified factors are
the worst-case **½** (unconstrained) / **1/e** (cardinality) of §6.1, and the tight instance bound below
is read against `R↑`, not `R`.

This is **far tighter than the worst-case multiplicative guarantee** and is *computed from the marginal
gains you already have* — it routinely certifies `R(S_g) ≥ 0.9·OPT_class` on the actual instance even
when the worst-case factor (below) is `0.6`. **Implemented** in `experiments/large_omega.py:U2_bound`
(operates on any `f(S)`; self-check: γ=1 ⇒ `U₂` is the Minoux bound and holds `≥ OPT`; γ<1 ⇒ `1/γ`
loosens it as predicted), reached via the entry point `experiments/omega_certificate.py:OmegaCertificate`
(auto-dispatches: exact `small_omega_brute_force` at K ≤ `--large-k` 15, this U₂ fallback at K > 15) and
cross-checked vs exact OPT on real non-monotone R at small K.

> **Worst-case multiplicative form (the a-priori one).** `R(S_g) ≥ (1/α)(1−e^{−αγ})·OPT_class`
> (Bian et al. 2017), with curvature `α∈[0,1]`. Recovers `(1−1/e)` at `α=γ=1`. Use it for *reporting
> the metric's thickness* (§6/§8); use `U₂` for *certifying this run*.

**Scope, stated bluntly.** `U₂` bounds `OPT_class`, **not** `OPT`. The gap `OPT − OPT_class` is
whatever a prompt *outside the criterion-set class* could add (free-form wording, holistic framing) —
**uncontrolled**, and exactly the thing rung 1's `U₁` covers loosely. So rung 2 certifies "`S_g` is
near the best rubric *in the class spanned by `U`*," a genuinely global statement within that class and
nothing more. Enlarging `U` raises `OPT_class` toward `OPT` but never certifies the closure.

### 3.3 Rung 3 — finite candidate set → statistical best-in-set. **[holds]**

This is the literal "is `p_i` best among `p_1,…,p_k`?" question. No optimization, no upper bound on
`OPT` — you **measure** `R̂(p_j)` for each with Miller–Madow debiasing + bootstrap CIs (resample items
*and* passes; the `vinfo.py` estimator).

> `p_i` is **certified best-in-set at confidence `1−δ`** iff `LB_{1−δ}(R̂(p_i)) > UB_{1−δ}(R̂(p_j))`
> for all `j≠i`.

"Within bounds" here = within statistical confidence. Scope: best among **the `k` tested only** — says
nothing about untested prompts. This is the most defensible everyday certificate and was missing from
the prior draft. When CIs overlap, the honest output is a *partial order* (a top group), not a winner.

### 3.4 Rung 4 — local necessary conditions. **[holds, caveated]**

For a given `p̂`, cheap checks that are **necessary but not sufficient** for global optimality:

- **Edit-stationarity.** `R(p̂) ≥ R(p')` for all `p'` in the one-edit neighborhood. Where
  GEPA/Evo/ProTeGi/APE halt. Local = global only if the edit graph is benign (large neighborhoods, no
  deep basins) — **no guarantee**.
- **Improvability witness.** If reconstructing from `p̂`'s verdicts yields a rubric `p'` with strictly
  higher transmission `T(p') > T(p̂)`, then `p̂` was not transmission-optimal — swap to `p'`. (Valid
  because `R ≤ min(T(m),T(m̂))`: a more-transmissive reconstruction cannot lower the ceiling.)
- **Saturation witness.** `R(p̂) = T(p̂)` certifies `p̂` extracts *all of its own* transmission
  (recovery-optimal **for its behavior**). It does **not** certify that behavior is the best behavior —
  a different `p` with higher `T` could have higher `R`.
- **Optimizer-agreement (empirical global proxy).** Independent optimizers converging to the same
  `R̂` is evidence the local certificates are global — the actual "real ceiling vs hill-climbing" test.

### 3.5 The ladder, summarized

| rung | upper bound `U` | scope of "optimal" | tightness | cost / assumption |
|---|---|---|---|---|
| 1 | `cap_f` (TVD `1−1/min(N,K)`) | **all strings (global)** | loose | none (Facts 1+2) |
| 2 | `R(S_g)+(1/γ)Σ_{top-k}δ` | best in criterion class `U` | **tight (per-instance)** | set model + measure `γ` |
| 3 | — (statistical) | best among `{p_1..p_k}` | CI-tight | finite tested set + CIs |
| 4 | — (necessary only) | local optimum | n/a | one-edit neighborhood |

**The honest frontier.** Global optimality over the infinite set is certifiable only to within `U₁`,
and that gap is governed by ignorance of `B_E`, not by insufficient effort. Convexity (Fact 1) buys two
things and no more: the cap `U₁`, and the *structure* of where the `T`-optimum sits (§4). Everything
tighter trades the infinite set for a class or a finite list. **There is no free, tight, global,
all-strings certificate — and that is a theorem about `B_E`, not a TODO.**

---

## 4. Structural facts about `p*` that convexity buys

These support the ladder; all are proven for `T` (Fact 1) and transfer to `R` only via `R ≤ T`.

### 4.1 `p*` is a pure prompt, not a randomized mixture (for `T`). **[holds, scoped]**

A convex function on a compact convex set attains its max at an **extreme point** (Bauer). Every extreme
point of `conv(B_E)` is a pure behavior — `ext(conv(B_E)) ⊆ B_E` (the inclusion, *not* equality: not
every pure behavior is extreme, since some `s_p` lie inside the hull of others; but the proof needs only
`⊆`, so the maximizer lands on some single prompt). A **score-averaging** randomized prompt
(sample `p~π`, average the verdict probabilities) induces `E_π[s_p] ∈ conv(B_E)`. So:

> Probability-averaging across prompts **never increases `T`**, and strictly decreases it unless `T`
> is affine on the mixing segment.

**Scope (a hole in the chat version):** this is the **score-averaging** ensemble only. A **vote / max /
nonlinear** aggregation is *not* a point of `conv(B_E)` and is **not** forbidden — majority-vote sharpens
`p_i` toward {0,1}, lowers `(1/N)ΣH(p_i)`, can *raise* `T`. Two checkable signs: prob-averaged
ensembles ≤ best single prompt; vote ensembles may exceed it. **For `R`:** since `R` is not known
convex (§2.3), the no-mixing claim is *plausible but unproven* for the objective itself — flag, don't
assert.

### 4.2 Shape of the transmission optimum. **[holds]**

By Fact 1, `T` is convex in `s`, so its max over the cube `C=(Δ_K)^N` sits at a vertex — a
**deterministic per-item labeling** — for *any* `f`. (Shannon view: subtracting `(1/N)Σ_i H(p_i) ≥ 0`
drives every `p_i` to a vertex, zeroing the second term, leaving `T=H(p̄)`.) Among deterministic
labelings the balanced one maximizes `T`:
- **Binary:** optimum = **balanced deterministic split** (½ items →1, ½ →0), `p̄=½`. Shannon `T=1 bit`;
  TVD `I_TVD=2q(1−q)`, max at `q=½` → **½**. *Corrects the chat's "item-distinct," impossible with two
  values.*
- **K-ary:** deterministic labels spread evenly. Shannon `T = log min(N,K)`; TVD = Gini
  `1 − Σ_m P_M(m)²` → `1 − 1/min(N,K)`. "item-distinct" is the `K≥N` case.

Prediction: optimal rubrics push the judge toward **confident, low-hedge, balanced** verdicts, coupling
to label balance (½/½) — lines up with the 50/50 planted-scary design.

### 4.3 `T ≠ R`: the cap is `T`'s max; spurious spread games `T`. **[holds — the key fix]**

§4.2's "optimum" is a trap as an objective. A judge that splits items ½/½ by an **irrelevant** feature
(e.g. length parity) hits `T=1 bit` while transmitting nothing about the target. **`T` is maximized by
spread, including spurious spread.** That is why `T` is only the *cap/shadow* and `R` is the objective:
- `R` (held-out) **rescues from non-articulable / non-generalizing spread** — a memorized idiosyncratic
  ½/½ labeling has high in-sample `T` but no rule reproduces it out-of-sample, so `R` is low.
- `R` does **not** rescue from **articulable-but-wrong-attribute** spread — a length-parity rule is
  articulable and reconstructible, so it scores high `R` too. Separating "right attribute" from
  "articulable wrong attribute" is **correctness**, which provably needs a label `Y` (bracketed, §10).

### 4.4 Hardness — the honest version. **[corrected framing]**

Maximizing convex `T` over `conv(B_E)` is **convex maximization**: NP-hard in general, local optima at
any of exponentially many vertices. We did *not* reduce a known-hard problem to our instance, so we do
not claim "NP-hard for our metric." The honest statement: at the raw-string level `B_E` is an arbitrary
finite set with **no exploitable structure**, so a global certificate must come from rung 1 (the cap)
or rung 2 (imposed structure); rung 4 (local) is all that's free.

**Where this sits relative to the literature (citations verified 2026-06-19).** Three strands circle the
pieces but none unifies them into a *global optimality certificate for a prompt*:
1. **Metric foundation.** Robertson–Koyejo (TVD-MI) build the no-ground-truth evaluator and prove TVD-MI's
   polynomial gaming-robustness. We turn their metric into an **objective to optimize the prompt against**
   (convexity + DPI over the behavior space), not just an evaluator — that is the departure.
2. **Submodularity *is* in prompting — but for selection sub-problems as a compute shortcut, never as a
   certificate.** SESS (Nian et al. 2026, arXiv 2601.03493) proves the **evaluation-subset** objective
   monotone-submodular for greedy *data* selection; "Select Smarter, Not More" (arXiv 2604.11328) does
   prompt-aware eval *scheduling* with submodular guarantees; submodular **demonstration/ICL selection**
   (Query-Focused Submodular MI; InSQuAD) picks few-shots. All use submodularity to be *fast under a
   compute constraint* — none builds a containment zone or certifies ε-optimality of the prompt. §6 (and
   the §6.7 brute-force enclosure) use it for exactly that.
3. **Discrete APO ignores global guarantees.** OPRO, TextGrad, APE, ProTeGi, and **Local Prompt
   Optimization** (Jain–Chowdhary, NAACL 2025 — *explicitly local*) are empirical local searches over
   tokens (LLM-feedback / bandit / gradient-proxy); they report "achieved X% accuracy" but cannot answer
   "does a different phrasing yield more?" — exactly the adversarial-reviewer question §3 / §6.7 targets.
   Rigorous-convergence APO instead abandons strings for **soft-prompt tuning** (continuous embeddings,
   gradient, convergence to stationary points) — a different object than the discrete string space.

The distinctive contribution: an *in-sample information shadow* `T` bounds recovery `R ≤ cap_f` over the
**whole** of `Σ*` (rung 1), with the within-class certificate (rung 2 / §6.7a enclosure) and
discovery-coverage (§6.7c) pushing toward **existential certification** — "we mapped the achievable space
and certified its optimum," not "we optimized as best we could."

### 4.4a GEPA is a generator, not a certifier — and its bias threatens *coverage*, not the `T`-ceiling. **[new, 2026-06-25]**

GEPA (Agrawal et al. 2025, reflective genetic-Pareto prompt evolution) is **population-based** — it maintains
a Pareto frontier specifically to resist the local-optima collapse of pure hill-climbing. So it is *not*
"explicitly local" in the LPO sense (§4.4 strand 3); calling it that understates the mechanism and a
GEPA-literate reviewer will catch it. The operative bias is different and sharper: GEPA explores only via the
**proposer LLM's edit-distribution**, so its reachable region is bounded by that proposer's blind spots (§6.9,
"LLM generators share gaps"); path-dependence (`e₂(e₁(p)) ≠ e₁(e₂(p))`, §6.5) compounds it.

What that bias does and does not threaten:

| question | GEPA bias threatens it? | why |
|---|---|---|
| "Is my found prompt near the fixed-target upper bound?" | **Yes, for attainment** | `R ≤ T(m_ω)` is method-agnostic, but `R<T` does not distinguish search failure from non-realizability of the sufficient readout. GEPA bias cannot invalidate DPI; it can invalidate an empirical claim that its plateau estimates the prompt optimum. |
| "Have I covered `B_E`?" (saturation, §6.9 / §11.3) | **Yes, badly** | GEPA is adaptive + value-tilted + (usually) single-family — exactly the three things that break Good–Turing and bias the discovery curve. "No new criteria" reflects the reachable subspace, not `B_E`. |

So GEPA is suspect as both a saturation estimator and an estimator of the achieved prompt asymptote. The
validity of the fixed-target DPI bound itself is independent of GEPA. Precise placement: **GEPA *is* the §12 depth stream** (adaptive,
value-tilted, hunts high-`R`), which structurally **disqualifies it from the §12 breadth/mass estimator** (its
draws are not iid from a frozen π). "Single-family GEPA saturation ≠ `B_E` saturation" is therefore not a
fixable flaw but GEPA being the wrong stream for that job; the cross-family remedy (§11.3) is "add a breadth
stream," not "de-bias GEPA." Legitimate uses: optimizing the *discriminative* metric prompt `M_ω` (need
discriminative, not optimal — local tendency tolerable) and as the depth-stream magic-word hunter. Illegitimate
use: as the sole estimator of Ω-coverage.

---

## 5. (Less central) The articulation gap `A = T − R`

*Demoted on purpose: `A` measures **the metric**, not the prompt. It is not an optimality certificate
for `p*` — it does not appear in any rung of §3. It is a useful by-product of having both `T` and `R`.*

`A(m) := T_test(m) − R(m) ≥ 0` is the discrimination that does **not** survive articulate→re-execute —
and it must be the **same-distribution** gap to be `≥ 0`: both legs on the **held-out** split, the held-out
transmission `T_test = I_f(M̂_test;X_test)` minus the held-out recovery `R` (the `A_tvd` of
`vinfo.tvd_guardrail`). This is `≥ 0` by Fact 2's DPI (Jensen, termwise). *Do not* form `A` as
`T_train − R_test` (in-sample consistency minus out-of-sample recovery): that mixes two distributions, is
not guaranteed `≥ 0`, and is a *different* quantity — the **generalization gap** `T_train − T_test`, which
is a separate measurement that *feeds* component (b) below. With `A` pinned to the held-out same-`f` gap, it
is label-free and anchor-free, but it still **bundles three things a single scalar cannot separate**:

- **(a) genuinely tacit** — no rule in language reproduces it (the on-thesis residual);
- **(b) non-generalizing** — overfit/memorized spread; no stable rule generalizes (read off the *separate*
  generalization gap `T_train − T_test`, not from `A` itself);
- **(c) executor-limited** — a rule exists and is articulable but *this* `E` can't apply it; a stronger
  `E'` would recover it.

Therefore **`A` is an *upper bound* on the tacit residual, not equal to it** — also inflated by a weak
reconstructor (fails to *find* the rule) or weak executor (fails to *apply* it). The honest target is
`A* = inf over reconstructors, with a capable executor`; we only ever estimate `A` from above. A high
`A` is never proof of tacitness — only "we couldn't articulate it *yet*." The **E-axis** splits the
components: the part of `A(m;E)` that shrinks as `E` grows is (c); what persists is (a)+(b), and the
held-out split separates those. `A` is **not** an error rate: `A=0` means "fully articulable," not
"correct" (§4.3).

**Quantifying component (b) with statistical learning theory.** Component (b) — non-generalizing /
overfit spread — is a *generalization gap* and can be bounded as one. If the reconstructor searches a
rubric family `𝒱` of complexity `d` (VC dimension / Rademacher), then the held-out recovery `R̂` differs
from its population value by `O(√(d/N_held))`, so the part of measured `A` attributable to (b) is bounded
by the reconstruction family's generalization gap, and the sample size to drive (b) below `ε` is
`N_held = Ω(d/ε²)`. This is the **right** use of an SLT bound here. **It is *not* a ceiling on recovery
and does *not* license a V-information ceiling** (see §6): the DPI failure of V-information is a
*population* fact (computation can create usable information even at `N→∞`), so adding `O(√(d/N))` to a
V-info transmission does **not** restore `R ≤ T` — that residual is non-vanishing, not a generalization
term. The clean ceiling stays the f-divergence DPI on Shannon/TVD (Fact 2); SLT only sharpens *which part
of `A` is overfit vs. genuinely tacit*.

*Three uses, descending weight:* (1) `A` as the label-free articulation-loss scalar above; (2) the
guardrail `R̂ ≤ T̂` cell-by-cell (a violation flags a leak or estimator bug) — **both sides must be the
same `f` AND the same (held-out) distribution**: the shipped `I_V` is *Shannon*, so a TVD guardrail needs
`T` recomputed as TVD-MI, the bounding `T` is the held-out `T_test` (not train consistency, §2.2), and `A`
is only a difference when `T` and `R` share `f` and split; (3) saturation `R=T(m)` as
a corner-case lossless-articulation witness (measure-zero, loose bound around it). None of these certify
prompt optimality.

---

## 5.5 The verifiable corner: code metrics as the instrument's calibrated zero. **[new]**

*Companion to §5. The one corner of the recovery loop where `R`'s ceiling is **known** to be 1 — so it
calibrates the whole pipeline, pins down which component of `A` we're measuring, and operationalizes
V/A/Taste as differences of one quantity.*

**The reconstructor is source-blind; the source sets the ceiling, not the process.** `R = I_TVD(M; M̂)`
is built by a reconstructor that reads only the pairs `(X_i, m(X_i))` and induces a rule — it never sees
*how* `M` was generated. So whether `m(x)` came from code or from another LLM is invisible to the
*recovery process*. What the source changes is the **realizability ceiling**: if `M` was produced by a
deterministic program, a rule that perfectly fits the pairs *provably exists*, so `R = 1` is
**reachable**; if `M` came from holistic LLM judgment, there may be *no* compact rule and `R = 1` may be
unreachable (genuine tacitness). "`R` should be 1" means *the reachable ceiling is 1* — the gap
`1 − R̂` is then diagnostic of the **pipeline**, not the metric.

**Two independent axes where "code" enters** (do not conflate):

| axis | controls | "code" value ⇒ |
|---|---|---|
| generator of `M` | realizability — does a fitting rule exist? | code-generated ⇒ `R=1` **reachable** |
| executor of `M̂` (`E`) | transmission fidelity `T` | `E`=compiler ⇒ `T=1` (zero executor noise) |

A **V metric** is *both*: `M` is code-expressible **and** `E` is a compiler.

**`T = 1`, precisely.** With `E` a deterministic compiler running **fixed** code, within-item entropy
`H(p_i)=0` for every item, so normalized reliability `T_norm = I_V/H(p̄) = 1` — *no consistency loss*,
only the metric's across-item range survives. Two caveats: (i) this is the **fixed-code** reading; if
each recovery pass **re-extracts** the program, `T<1` and that gap is **articulation variance** (does the
LLM converge on one program?); (ii) it assumes **pure** execution — timestamps, hash-ordering, float
nondeterminism, or I/O break `T=1` and signal a *data bug*, not articulation.

**DPI goes vacuous, so `A` collapses to one clean component.** `R ≤ T = 1` is uninformative; all the
action is the *direct* `R̂`. Onto §5's decomposition `A = (a) tacit + (b) non-generalizing + (c)
executor-limited`:
- code-generated `M` ⇒ **(a) = 0** (a rule exists by construction);
- `E`=compiler ⇒ **(c) = 0** (the executor applies any rule faithfully);
- so for a V metric `A = 1 − R` is **pure (b): learnability** — did the reconstructor *find* the existing
  rule from finite `(X, m(X))` pairs? This is the only setting where `A` is a *single interpretable*
  quantity, not a three-way bundle: it measures sample-complexity / articulation difficulty with nothing
  else mixed in.

**The checks code metrics give the pipeline** (run *before* trusting any `R̂` on an opaque metric):

| # | check | plant | pass condition | isolates |
|---|---|---|---|---|
| C1 | **calibration / kill-switch** | a real code metric (word-count threshold, regex, AST-node count) | `R̂ → 1` within CI through the *full* articulate→re-execute loop | realizability+learnability+executor jointly |
| C2 | **`T=1` audit** | fixed code, `E`=compiler | `H(p_i)=0` ⇒ `T_norm=1` | executor purity (catches nondeterminism) |
| C3 | **articulation-variance** | re-sample the code extraction per pass | `T`≈1 ⇒ stable program; low ⇒ more extraction draws needed | how many reconstruction samples we need |
| C4 | **realizability vs learnability** | same code-`M`, `E`=compiler vs `E`=LLM | compiler `R=1` & LLM `R<1` ⇒ gap is *executor expressiveness*; compiler `R<1` ⇒ learnability loss | separates (b) from executor-class limits |
| C5 | **cap / TVD-scale sanity** | balanced binary code metric, recovered exactly | `R̂ → cap_TVD = ½`, never exceeds it | the cap computation + TVD estimator scale |
| C6 | **planted-γ submodularity** | `M = f(X_1..X_K)` of known predicates (AND ⇒ redundant/γ≈1; XOR/majority ⇒ synergy/γ<1) | `γ̂` and greedy-vs-OPT recover the *planted* structure | validates §6.6 / `small_omega_brute_force` γ on real items with a known answer |

**How to use them — gating, not decoration.**
- **C1–C2 are a hard gate.** If a planted code metric doesn't recover `R̂≈1` (C1) or `T` isn't pinned at
  1 under a compiler (C2), *stop* — no `R̂` on a real metric is trustworthy. This is the plan's E0
  kill-switch, now on **real items** instead of only synthetic planted metrics.
- **C6 gates the submodularity certificate.** Validate `γ̂`/greedy on a planted-structure code metric
  *before* running the certificate on an opaque GEPA-mined `Ω` — otherwise a wrong `γ̂` is unfalsifiable.
- **C4 is the empirical definition of the V→A boundary.** The compiler-vs-LLM-executor recovery gap *is*
  what the A-layer adds over deterministic verification — measured, not asserted.

**The executor-class ladder = operational V/A/Taste.** Restricting the *executor class* on the *same* `R`
turns the decomposition into differences of one quantity:

| layer | `R_L`, `E` | recovers |
|---|---|---|
| **V** | code, compiler | `sup R` over deterministic programs — the verifiable floor |
| **V+A** | NL rubric, LLM | + what articulable rules & an LLM judge add |
| **V+A+Taste** | dense model | the ceiling `C` |

`A = (V+A) − V`, `Taste = C − (V+A)` — same `R`, widening `E`. The V corner is the *anchored, zero-noise
reference* (known ceiling, `T=1`) that makes the gaps above it measurable. Its limit: it covers only
code-expressible `M`, so it pins the **floor**, never the tacit ceiling.

---

## 6. (Less central) Greedy rubric composition — rung 2's machinery in full

*This is the engine behind rung 2 (§3.2). It is the within-class certificate; it does not extend to the
infinite string set.*

**Per-metric (§1).** Everything here is for ONE metric `M_i`: the ground set `U = Ω_i` is `M_i`'s OWN atomic
criteria (mined per-metric, §6.5), and `R`, `α`, `γ` are `R_i`, `α_i`, `γ_i`. The task-level pool (union of
all `Ω_i`) is vast → `α≈0.9` → non-discriminating; the structure is per-metric.

**Model.** Rubric = set `S ⊆ U` of articulated criteria; `R(S)` = recovery of the canonical rubric for
`S`, measurable by ablation. **The value function here *is* the V-information staircase:** the marginal
gain `ΔR_x = R(S∪{x}) − R(S)` is exactly the staircase increment `ΔI_x^E = I_{V_{S+x},E} − I_{V_S,E}`,
the marginal V-usable information that articulating `x` adds. So `α` and `γ` below are computed from
these increments, and **V-information is the natural value function for this layer** — its DPI violation
(fatal for the §2.2 ceiling) is irrelevant here, because attribution needs no DPI. Division of labor:
**V-information for §6 (attribution / the M,K axes); Shannon/TVD-MI for §2–§3 (the ceiling and the
cap).** Different layers, not competitors.

*Why we do **not** "commit to V-information everywhere" (a tempting but wrong unification).* One could try
to use V-information for the ceiling too and "restore" `R_V ≤ T_V` with a statistical-learning term,
`R_V ≤ T_V + O(√(d_𝒱/N))`. **This is false as a ceiling.** V-information's DPI failure is a *population*
property (Xu et al.: processing can *create* usable information); the SLT term vanishes as `N→∞` while
the population gap does not, so `R_V` can exceed `T_V` no matter how large `N` is. The correct split is
the one above: TVD/Shannon (which *do* obey DPI) carry the ceiling; V-information carries attribution; and
SLT sharpens the *non-generalizing component of `A`* (§5), not the ceiling. Using two `f`'s for two jobs
is coherent, not a conflict to be resolved by force-collapsing to one.

### 6.1 Monotonicity fails — so the PRIMARY rung-2 certificate is non-monotone USM, not the monotone bound. **[corrected]**
PRUNE evidence (dropping criteria *raises* recovery) ⇒ a distracting criterion is read by the judge and
shifts behavior ⇒ `R(S)` is **non-monotone**. NWF `(1−1/e)` and the monotone Bian/Minoux bounds (§6.2,
§3.2) require monotonicity and do **not** apply to raw `R`. **Lead with Unconstrained Submodular
Maximization (Buchbinder–Feldman–Naor–Schwartz 2015):**

- **double-greedy** — maintain `A=∅` and `B=Ω`; at each element decide include-in-`A` vs exclude-from-`B`
  by marginal gain. **Randomized double-greedy = ½-approximation** (tight); **deterministic = 1/3**
  (*not* ½ — a common mis-citation). Holds for **non-monotone submodular `R`**, no monotonicity, no
  exponential subset sweep.
- **random-greedy = 1/e** under a cardinality constraint (non-monotone).

The monotone-only objects — the Minoux instance bound (§3.2) and the Bian `(1/α)(1−e^{−αγ})` (§6.2) — apply
only to the free-disposal monotonization `R↑(S):=max_{T⊆S}R(T)` (an *upper idealization*; PRUNE says
disposal isn't free, so `R↑` is not the operating object) or to an `R` empirically verified monotone on
the corpus. Default to double-greedy. The PRUNE down-steps and the staircase-non-monotonicity are the
**same phenomenon**. **Implemented** in `experiments/large_omega.py:double_greedy` (both the deterministic
`1/3` and randomized `½` variants, on any `f(S)`; self-check: recovers OPT on known submodular `f`),
the large-Ω fallback algorithm reached via `experiments/omega_certificate.py:OmegaCertificate` (auto-
dispatched at K > `--large-k`) and cross-checked vs exact at small K.

**The honest hard case both fixes miss — non-monotone *and* weakly-submodular at once.** The two clean
constants patch *different* axes: Bian `(1/α)(1−e^{−αγ})` tolerates `γ<1` but **requires monotone**;
double-greedy `½` tolerates non-monotone but **requires true submodularity (`γ=1`)** — its analysis uses
the submodular exchange inequality. Our `R` is realistically **both** non-monotone (PRUNE) **and**
weakly-submodular (`γ<1`, complementary cues, §6.2), the *intersection* neither result covers. There is
no clean off-the-shelf constant here: non-monotone γ-weakly-submodular USM has only weaker, partly-open
guarantees that degrade with `γ`, and I will **not** invent one (the critique's "`1/(1+γ)` or similar"
is not a constant I can verify for this regime). The honest protocol: **measure both** the monotonicity
violation and `γ`; if `γ≈1` use double-greedy `½`; if effectively monotone use Bian on `R↑`; in the
genuine intersection, report the degraded regime as a measured fact, not a guaranteed ratio. This
intersection is the true frontier of Rung 2.

### 6.2 Submodularity is not automatic. **[corrected]**
MI-of-a-feature-set is neither sub- nor super-modular in general. Krause–Guestrin (2005): `I(X_S;Y)` is
submodular **when** the criteria are conditionally independent given the target (a *sufficient*
condition — not generally necessary, and not our case). Two
**distinct measurable** constants grade the departure:
- **submodularity ratio `γ∈[0,1]`** (Das–Kempe): `γ=1` ⇔ submodular; `γ<1` ⇔ super-modular/complementary
  (XOR cues — A or B alone uninformative, together discriminating). Greedy `≥(1−e^{−γ})·OPT` (monotone).
- **generalized curvature `α∈[0,1]`** (Conforti–Cornuéjols `κ`; Bian et al. for non-submodular): `α→0` ⇔
  modular (greedy exact); `α=1` ⇔ maximally saturating.

**Bian et al. 2017 (monotone case / `R↑`):** `R(greedy) ≥ (1/α)(1−e^{−αγ})·OPT_class` — `(1−1/e)` at
`α=γ=1`; `(1−e^{−γ})` at `α=1`; `→γ` as `α→0` (exact only if *also* `γ=1`). **Measure `(α,γ)` → those two
numbers are the worst-case certificate; the per-instance `U₂` of §3.2 is the tighter run-time one.**

**Estimating `γ` (it is empirical/structural, never analytic for a black-box judge).** For an arbitrary
prompt value function on an opaque LLM, submodularity **cannot be proven analytically** — LLMs show
non-monotonicity and strong semantic interaction (a rule added to a long prompt can trigger a different
attention pattern / context collapse than the same rule added to a short one). The honest routes, in order
of preference:
- **Brute-force is the default for `|Ω| ≤ 15` (§6.7a) — it bypasses `γ` entirely.** Enumerate the
  budget-feasible subset lattice and read the *exact* within-class optimum; no `(α,γ)`, no monotonicity.
  The submodular machinery below is needed **only** when `|Ω| > 15` makes enumeration infeasible
  (enumeration stays tractable to `~25`).
- **`γ` is trustworthy only on an *orthogonalized* `Ω`.** The submodularity-ratio bound is only as good as
  the conditional-independence structure it assumes; on a raw GEPA-mined pool of near-paraphrases `γ` is
  corrupted by redundancy. Run the **Shannon-CMI orthogonalization filter** (§6.5) first so the surviving
  units are behaviorally near-orthogonal — *then* the §6.6 conditional theorem's CI-given-`M` premise
  approximately holds and the measured `γ` means something.
- **Empirical estimate (when `|Ω| > 15`):** sample a representative set of `(S,Ω)` pairs, compute their
  marginal-gain ratios, take the **lower tail** as a conservative proxy for `γ` — reported with its
  sampling CI, never as a certificate.
- **Exact `γ` is `O(2^M)`** (all `(S,Ω)` pairs over `M` criteria) — infeasible at scale; the brute-force
  enclosure is the tractable substitute below `|Ω|=15`.

*Dropped (2026-06-22): the Das–Kempe spectral lower bound `γ ≥ λ_min(Gram)`.* It treats the value function
as a **linear regression** on the criteria, so the eigenvalue certificate is valid only when the executor
combines criteria *linearly*. An LLM judge does not — it mixes criteria through attention, which is exactly
the non-linear interaction `γ<1` is meant to capture — so `λ_min` is **not** a sound lower bound on `γ`
here (it can sit *above* the true `γ`, making the "certificate" unsafe). Keep it, if at all, as a cheap
*diagnostic of pairwise redundancy*, never as a guarantee; the orthogonalization filter + brute-force
enclosure replace it as the route to a trustworthy within-class bound.

### 6.3 Worst case is real. **[holds]**
`γ→0` (strong complementarity) ⇒ greedy can be arbitrarily bad; the multiplicative bound vanishes. The
certificate is *honest about thick metrics*: it degrades exactly where the metric is holistic and *tells
you so* (small `γ`). Not a guarantee for thick metrics — rung 1's cap is the always-available fallback.

### 6.4 Relaxation note. **[corrected]**
The chat's Frank–Wolfe "duality gap on behavior" was wrong (FW gaps certify *concave* max; §2 proved the
behavior objective *convex*). The correct relaxation certificate lives in the **set model**: the
multilinear extension is concave along non-negative directions, which **continuous-greedy**
(Calinescu–Chekuri–Pál–Vondrák 2011) exploits for `(1−1/e)` under a matroid, with pipage rounding. It
applies to criterion selection, **not** to the behavior-space convex program.

### 6.5 Discovery-to-Selection: turning a GEPA run into a set problem (where `Ω` comes from). **[new]**

Rung 2 assumed a ground set `U` of criteria. **Where does `U` come from?** Not from running an optimizer —
and here a **category error must be avoided**: a live GEPA/Evo/ProTeGi/APE run is **not** a submodular
selection. Its edit operators (`PRUNE`, `ADD`, `REPHRASE`) are *ordered, non-commutative morphisms* on a
string state-graph — `e₂(e₁(p)) ≠ e₁(e₂(p))` (PRUNE-then-clarify ≠ clarify-then-PRUNE) — applied to a
ground set that **does not exist a priori** (the supervisor invents each edit from the current error
trace). Submodularity is a property of a set function `f:2^Ω→ℝ` on an *unordered, fixed, exchangeable*
`Ω`. A live edit trajectory is **local hill-climbing / an MDP** (a supervisor policy `π(e∣p)` over edit
states — formally tree-search/MCTS, not set selection; rung 4, §5), not set optimization. Conflating the
two is the single most tempting mistake here.

**The bridge (Discovery-to-Selection), four steps (revised 2026-06-22).** Use the optimizer as a *miner*,
then **orthogonalize**, canonicalize, and run submodular selection on top:
1. **Mine the *diffs*, not the winners.** Run GEPA `K` steps; harvest the **semantic diffs** — the
   criteria each *accepted mutation added* — across the whole Pareto lineage, **not** just the surviving
   final prompts. This forms the candidate pool `R_pool`. (Freezing only the winners discards most of what
   was discovered; the diffs *are* the discoveries.)
2. **Orthogonalize → `Ω` (the new load-bearing step).** `R_pool` is full of near-paraphrases that induce
   the *same* per-item behavior. Build `Ω` by a behavioral filter: for each candidate `e`, extract its
   per-item signal `X_e ∈ {0,1}^N` (does `e` fire on item `i`) and compute the conditional information
   `I(X_e ; X_Ω)` of `e` *given the units already in `Ω`*. **High ⇒** `e`'s behavior is already explained
   by `Ω` — a redundant paraphrase, **discard**; **low ⇒** `e` induces a new, orthogonal partition,
   **add it to `Ω`**. The result is a behaviorally near-orthogonal ground set — which is exactly what makes
   the §6.6 conditional theorem's CI-given-`M` premise approximately hold, hence `γ` trustworthy
   (`experiments/orthogonalize.orthogonalization_filter`).
3. **Canonicalize.** Define a deterministic **compiler** `C(S)` that assembles any subset `S⊆Ω` into a
   prompt by a *fixed* template with a **fixed section order — Format → Semantics → Negative Constraints**
   — criteria slotted in a fixed index order within their section. This **kills path-dependency**: `C(S)`
   is unique per subset, so `f(S) := R(C(S))` is a genuine set function. Selection order no longer matters.
4. **Select with a certificate.** Now Rung 2 applies verbatim: **brute-force for `|Ω| ≤ 15`** (§6.7a, the
   default — bypasses `γ`), else double-greedy (½, §6.1) for non-monotone `f` / the monotone bounds on
   `R↑`, with `γ` estimated per §6.2.

**The atomic unit is *behavioral*, not logical.** An element of `Ω` is a **behavioral partition
operator**: a text fragment that induces a measurable, orthogonal shift in the executor `E`'s behavior
space. "Green and round" is *one* atomic unit if `E` processes it as a single per-item signal — we do
**not** split it into "green" + "round", because if the LLM emits one verdict for the conjunction the two
sub-predicates are *mathematically identical* in behavior space (same `X_e`). Atomicity is defined by the
executor's induced partition, not by logical irreducibility; the orthogonalization filter (step 2) is the
operational test — a "split" the filter discards as redundant was never a separate unit.

**Why the filter uses Shannon CMI, not TVD (the proxy split).** Step 2 needs *high-dimensional*
conditional mutual information `I(X_e ; X_Ω)` with `|Ω|` growing — and **TVD-MI has no chain rule**, so a
plug-in high-dimensional TVD-CMI is intractable (the plug-in estimator outputs noise). So the **filter**
uses **Shannon CMI**, estimated as the **cross-entropy reduction of a surrogate classifier** (predict
`X_e` from `X_Ω`; the drop in held-out cross-entropy is the usable conditional information —
`orthogonalize.shannon_cmi_surrogate`). Once `Ω` is frozen, the **certificates revert to the
gaming-robust TVD-MI**: the within-class bound (Rung 2) and the global cap (Rung 1) are evaluated in TVD.
Shannon for *constructing* `Ω` (needs the chain rule), TVD for *certifying* on it (needs gaming-robustness)
— the same division of labor as §2–§3 vs §6, now made operational at the filter.

**Do NOT claim `f` is "natively submodular."** The seductive argument — *attention dilutes as the prompt
grows, so marginal returns diminish → submodular* — is a **story, not a proof**, and it is contradicted by
two facts this very framework establishes: (i) **complementary criteria give *increasing* returns** (rule
"emit JSON" + rule "add a confidence field": the second is worthless without the first → super-modular,
`γ<1`); (ii) **`R` is non-monotone** (PRUNE — a rule can *hurt* in a big prompt via context collapse).
You cannot simultaneously "measure `γ` because rules interact" and "assert `γ=1` from attention dilution."
Attention-dilution and semantic-redundancy are at best *heuristic pressures* toward diminishing returns;
**`γ` is measured (§6.2) or established by the conditional theorem of §6.6, never assumed from a story.**
Canonicalization (step 3) buys the *set-function* structure for free, but buys **nothing** about
submodularity; the **orthogonalization filter (step 2)** is what approximately buys the CI-given-`M`
structure submodularity needs — by *removing redundant paraphrases* until the survivors are near-orthogonal
(§6.6), measured, not assumed.

**Scope — this does not recover global optimality.** The double-greedy/Bian guarantee is relative to
`OPT_Ω` = the best subset of the *mined* `Ω`, **not** the global `OPT` over all strings. A richer miner
(more GEPA steps, more seeds) grows `Ω` toward `OPT` but never certifies the closure. So Discovery-to-
Selection upgrades a heuristic GEPA run into a **within-`Ω` certified** prompt — exactly Rung 2's "global
within the class," with `Ω` now a concrete, optimizer-mined class. It is the operational realization of
Rung 2 on our actual optimizer (the GEPA lineage we already log), not a fifth rung.

### 6.6 Is submodularity *ever* analytical here, or only measured? A conditional theorem. **[new]**

§6.5 says "measure `γ`, never assume it" — correct as the *default*, but it undersells what is provable.
Submodularity of the mined-set recovery is a **conditional theorem** with an **exact failure signature**,
not a pure empirical shrug. (Triple-checked below; the failure direction is a clean identity, not
"attention dilution" hand-waving.)

**Reduce.** Model each criterion `i` as a per-item signal `X_i` (does the rubric-item fire on the
example); the *ideal* recovery is `Ǐ(S) = I(M; X_S)`, information the criteria jointly carry about the
original verdict `M`.

1. **Monotone, always:** `Ǐ(S∪{i}) − Ǐ(S) = I(M;X_i∣X_S) ≥ 0`.
2. **Exact submodularity signature (co-information).** Adding criterion `j` shifts `i`'s marginal gain by
   `I(M;X_i∣X_j) − I(M;X_i) = I(X_i;X_j∣M) − I(X_i;X_j)`. So **redundancy** (correlated *through* `M`,
   `I(X_i;X_j∣M) < I(X_i;X_j)`) ⇒ gain shrinks ⇒ **submodular** (`γ→1`); **synergy/XOR**
   (`I(X_i;X_j∣M) > I(X_i;X_j)`) ⇒ gain grows ⇒ **super-modular** (`γ<1`). `γ<1` has a computable
   information-theoretic fingerprint per criterion pair.
3. **Sufficient condition (Krause–Guestrin, derived):** if criteria are **conditionally independent given
   `M`**, then `I(X_i;X_j∣M)=0`, every marginal gain is non-increasing, so `Ǐ` is **monotone submodular**
   ⇒ greedy `(1−1/e)`. A theorem, conditional on a *checkable* structure.
4. **Why the *real* `R(S)` still isn't clean — the executor bottleneck.** `R(S)=I(M;M̂_S) ≤ Ǐ(S)`: the
   executor compresses the whole signal vector `X_S` into one verdict. *That compression* (not "attention
   dilution") is what injects the **non-monotonicity** (`M̂_{S∪e}` can be a worse function of `M` than
   `M̂_S` — the PRUNE effect). So `R` inherits submodularity only to the extent the executor is a faithful,
   saturating aggregator of (near-)CI criteria; the measured non-monotonicity and `γ<1` quantify the gap.
5. **Computable certificates (not just point estimates):** (a) the **Shannon-CMI orthogonalization
   filter** (§6.5) enforces the CI-given-`M` premise *by construction* — it drops any unit whose behavior
   `X_e` is already explained by `Ω`, so the surviving set is behaviorally near-orthogonal and the
   conditional theorem applies; (b) estimate the **pairwise co-informations** `I(X_i;X_j∣M) − I(X_i;X_j)`
   directly to localize *which* surviving pairs still push `γ` below 1 (residual synergy the filter did not
   remove). *(The former Das–Kempe spectral `λ_min` certificate is **dropped** — it assumes a
   linear-regression value function, false for an attention-mixing LLM executor; §6.2.)*

**Net.** Submodularity for the GEPA-mined set is a conditional theorem (CI-given-`M` + faithful executor),
`γ<1` is exactly synergy (co-information), and `γ` has a computable lower bound. "Measure `γ`" stays the
operating default — but it is *backed by* a theorem and a certificate, and the things that break it
(synergy; the executor's compression → non-monotonicity) are exactly named. This is the theoretical
handle on GEPA submodularity, with its assumptions stated rather than assumed away.

*Empirical confirmation (`experiments/submod_conditional.py`, zero-GPU).* Planting CI-given-`M` criteria
(redundant noisy copies) gives **exact `γ=0.94`, greedy/`OPT_Ω`=1.00, no synergy flagged**; planting an
XOR pair among distractors gives **`γ=0.00`, greedy/`OPT_Ω`=0.24** (greedy chases the distractors, blind
to the zero-marginal pair) and the **co-information flags exactly that pair (+1.0 bit)**. The conditional
theorem and its failure signature both reproduce on ground truth.

*Real-criteria run (`experiments/real_gamma.py`, code-review, 90 items, 10 LLM-scored item-level
criteria; `real_gamma` takes `M` = the full-rubric verdict by construction — note this is the tool's own
choice and distinct from the certificate's prose-`M` target of §6.7a′; 2026-06-22/23).* `γ=0.16`, curvature
`α=1.0`, **greedy/`OPT_Ω`=0.818 (exact)**, no pairwise synergy. *(NB: the number reported here was first
measured under a now-removed `--holistic-m` target `M`; the full-rubric-M re-run is pending a GPU —
treat `γ=0.16` as a placeholder pending that re-scoring, not a final result.)* The `α=1`+no-synergy+
greedy-near-OPT picture says the criteria are **redundant proxies for the target** (the submodular-
friendly regime); the low `γ=0.16` is **not** clean complementarity — it is **finite-sample** (joint
`I(M;X_S)` over 5 criteria ≈ 64 cells estimated from 90 items → heavy plug-in bias; `γ_exact`=min over
`2^K` subset-pairs picks up the noisiest). This is the empirical motivation for §6.7 (brute-force the
*real* `R(C(S))` — clean 2×2 MI per subset, no joint-MI under-sampling).

### 6.7 High-compute regime: enumerate the class, and certify its coverage. **[new]**

With compute to spare, two upgrades make the within-class guarantee *ironclad* and push toward the global one.

**(a) Brute-force the class — skip the approximation, and fix the sampling problem.** When `Ω` is small
enough to enumerate the budget-feasible subset lattice (`|Ω|≲25`, `Σ_{j≤k}\binom{|Ω|}{j}` evaluations),
**score the *real* `R(C(S))` for every subset** and take the max. This is strictly better than the
submodular bound: it returns the **exact within-class optimum** (Rung 2 collapses into a Rung-3
best-in-set over the *entire* class), needs no `(γ,α)` and no monotonicity, and — crucially in our data —
**avoids the joint-MI under-sampling that made `γ` unreliable** (each `R(C(S))` is a clean 2×2 MI
`I(M;M̂_S)`, well-sampled), while **capturing the executor bottleneck for free** (no `Ǐ` idealization;
non-monotonicity/PRUNE is simply *in* the enumerated values). The deliverable to a skeptic: *"`p̂` is not a
greedy local optimum; it is the exact global optimum of the criterion class."* Cost = `#subsets × scoring`;
parallelizable. *(Implementation: the entry point is `experiments/omega_certificate.py:OmegaCertificate`,
which auto-detects K and runs `small_omega_brute_force` exact mode for K ≤ `--large-k` (15) — or `large_omega`
fallback above it. See §6.7a′ for the prose-`M` / two-quantity design that governs the target. 2026-06-22.)*
(Submodularity, §6.1–§6.6, is the fallback **only** when `|Ω|` is too large to enumerate — `large_omega.py`
supplies `U₂` and double-greedy, the two rung-2 objects brute force bypasses.)

### 6.7a′ The prose-`M` target and the two-quantity decomposition. **[new, 2026-06-23]**

*Where the target `M`, the criterion set `Ω`, and the certificate's two measured quantities come from.
The earlier draft set `M = C(Ω)` (all criteria re-compiled); that conflated two gaps. This corrects it.*

**The prompt is prose; `Ω` is a post-hoc decomposition.** Phase A (GEPA, `optimizer.improve`) optimizes
a **natural-language prose prompt** `p̂` — the object the executor `E` actually reads and scores with.
Forcing structured `{criteria:[...]}` JSON into the *executed* prompt would be unnatural and would change
what we measure. `Ω` is therefore not the prompt: it is a **separate pass** in which a strong model (GLM,
`real_gamma._decompose`) decomposes the prose `p̂` into atomic criteria **after** optimization. Division of
labor: prose `p̂` is the prompt; `Ω` is its decomposition, built only for the certificate.

**Two distinct behavioral quantities, not one.** With `M` = `E`'s verdict on prose `p̂`, `M_ω` = `E`'s verdict
on `C(Ω)` (all decomposed criteria re-compiled), and `M_s` = `E`'s verdict on `C(S)` (a criterion subset):

> **`I(M, M_ω)`** — the **decomposition gap**: how much of the prose prompt's behavior survives
> atomization into criteria. Expected large if the decomposition is (nearly) lossless; if so, prose and
> `C(Ω)` are behaviorally the same object and the selection certificate (below) reads against a faithful
> stand-in for the prompt. **`I(M_ω, M_s)`** — the **selection gap**: how much of the all-criteria
> behavior a subset `S` recovers. This is the within-class subset certificate of §6.7a (Rung 2), with
> `M_ω` as the target.

The earlier single-quantity `R(C(S))` against `M = C(Ω)` measured `I(M_ω, M_s)` but **hid the
decomposition gap** inside it (a low recovery could mean either a bad subset *or* a lossy decomposition).
Splitting them makes each loss diagnosable on its own. Note these are distinct **pairwise** MIs that
diagnose distinct gaps — MI does not telescope into a clean sum, so do **not** read them additively;
the direct quantity `I(M, M_s)` would conflate both, the two-step chain separates them.

**Each quantity carries its own reliability ceiling.** Per the same-distribution DPI of §2.2, each `I(·,·)`
is read against its own `T = I(\text{prompt},\text{prompt}')` (two passes of the same prompt): `T_prose`
for `I(M, M_ω)`, `T_ω` for `I(M_ω, M_s)`. The gap `A = T − R` quantifies each quantity's
non-articulable / non-generalizing residual on its own distribution. Both legs use the deterministic
logprob `P(YES)` readout (`_signal`) rather than sampled binary, to keep the execution-noise level at the
model's predictive uncertainty rather than adding sampling stochasticity.

**Scope — same caveat as the rest of §6.** Both are **reconstructibility** measures (behavioral agreement),
not correctness: two prompts agreeing on a spurious feature read high for the wrong reason (§4.3).
Separating the right attribute from an articulable wrong one needs a label `Y`, out of scope here.

**(b) Large `Ω` — the relaxation, honestly.** The multilinear-extension **continuous-greedy** (§6.4, *not*
the Lovász extension — that is for submodular *minimization*) is the right tool, but it **inherits the
§6.1 caveats**: its `(1−1/e)` needs monotone-submodular, which `R` is not, and free-disposal `R↑` is an
*idealization* (optimizing it ≠ optimizing `R`) that is itself exponential to evaluate exactly. So it is
not a free fix; report `(α,γ)` and the realized ratio, not a clean `(1−1/e)`.

**(c) Certifying `Ω` is exhaustive — the "magic words" defense (missing *impact*, not missing *mass*;
revised 2026-06-22).** The within-class optimum still leaves the cynic's question: *"what if a rule outside
`Ω` changes everything?"* The earlier **Good–Turing missing-mass** estimate (`≈ singletons/N`) is
**rejected**: it bounds the probability of *seeing a new concept under the generators' sampling
distribution*, **not** its *impact*. A "magic word" is defined by **disproportionate impact** and is
rare-to-sample *by definition* — exactly what missing-mass cannot catch. Replace it with a two-pronged
bound on **missing impact**:

1. **Submodular tail-bound (the mathematical cap).** Because the orthogonalization filter (§6.5) enforces
   near-submodularity, greedy marginal gains are non-increasing. Greedy chose its last unit `e_last` over
   every other candidate, so by submodularity **no unseen unit beats the smallest greedy marginal gain**:
   > `max_{e∉Ω} Δ(e ∣ Ω) ≤ min_i [greedy gain]_i = Δ(e_last ∣ Ω∖{e_last})` (the **certified** bound).
   The leave-one-out minimum `min_{e_i∈Ω} Δ(e_i ∣ Ω∖{e_i})` is `≤` this (since `Ω∖{e_i} ⊇ Ω_{<i}`) — a
   *tighter diagnostic*, not itself a guaranteed bound; both decay together as the tail saturates
   (`orthogonalize.submodular_tail_bound` returns `certified_bound` and the loo `tail_bound`). When they
   reach near zero, even an unseen magic word cannot lift recovery by more than that. *Honest caveat:* the
   inequality is **conditional on tail-submodularity** (`γ≈1` in the discovered tail — exactly §6.9's
   condition); a genuine tail-synergy (`γ<1`) undiscovered in `Ω` can still make `OPT` jump, which is why
   the tail-bound is paired with an empirical probe that does **not** rely on submodularity:
2. **Adversarial behavioral saturation (the empirical cap).** Generate **adversarial probes** — rare,
   out-of-distribution, or deliberately odd fragments (`"Brevirostrate"`, `"Use XML schema"`, `"Ignore
   previous instructions"`). Extract each probe's per-item signal `X_probe` and compute the conditional
   information `I(X_probe ; M ∣ X_Ω)` (`orthogonalize.adversarial_saturation`). If this is `≈ 0` for **all**
   probes, even extreme rare words cannot induce a new `M`-relevant behavior partition that `Ω` does not
   already capture — the *behavior space* (not just the string space) is saturated.

**Mining halts only when both fire:** the submodular tail-bound is below `δ` **and** the adversarial probes
show behavioral saturation. This upgrades Rung 1's loose cap into a *measured missing-impact statement* —
stronger than the old missing-mass evidence, and honest about its one condition (tail-submodularity for
leg 1, covered by leg 2's assumption-free probe).

> **Correction (2026-07-12; historical argument retained).** The displayed inequality does not follow for
> `e∉Ω`: submodularity on a known ground set constrains elements in that ground set, not prompts that were
> never represented. A finite adversarial battery can falsify saturation but cannot prove every untested
> prompt redundant. Consequently this two-pronged procedure remains a useful diagnostic and historical
> design argument, but it is not a closure certificate. Bound-grade replacements are §12.6b's fresh
> all-draw finite-horizon gain bound and, under external positivity, exact-pattern support exhaustion.

---

### 6.8 The form axis — does the within-class bound need to incorporate prompt *style*? **[new]**

*An anticipated criticism, met head-on. The within-class certificate (Rung 2) ranges over content-subsets
`S⊆Ω` at a **fixed form** `φ` — few-shot count `k`, persona, output format, ordering. A reviewer will
object: "you optimized one coordinate and froze another that demonstrably moves `R`, so `OPT_Ω` is a
**conditional** optimum, not a stationary point of the full prompt space — your tight bound doesn't bound
the true sup over prompts." The objection is fair to raise; here is the honest accounting.*

**First, what is *not* a defense.** The Rung-1 cap is form-invariant — but **vacuously**: `cap_TVD = 1−1/K`
is a constant set by verdict granularity (½ for binary), independent of *every* prompt, content or form
(§3.1, "most likely to be vacuous"). "Form can't break the cap" is true and says nothing. Do not offer it
as the answer.

**Form is an orthogonal coordinate, not a missing criterion.** A few-shot block adds no per-item check —
it has no signal `X_e` about `M`, so it **cannot** enter the §6.6 co-information machinery (which needs
each element to be a per-item signal). What it does is modulate **executor fidelity**: it makes `E` apply a
*fixed* rubric more reliably, raising `T`, which lifts the `R`-ceiling (`R≤T`). So form is the *continuous
knob within* the "`E`=LLM" rung of the §5.5 executor-class ladder. Factor recovery into two arguments:
`R(S, φ)`, `S⊆Ω` = content (what is checked), `φ` = form (how reliably `E` checks it). Freezing `φ` while
certifying `S` is a legitimate **block-coordinate** decomposition — *provided the `φ` coordinate is itself
bounded*, which is the real obligation.

**The `φ` coordinate gets its own bound — two routes, no separability assumed.**
- *Curvature.* `R` is typically **concave in `k`** (few-shot saturates: each extra example helps less), so a
  small `k` is near-optimal with a bounded gap. Same *flavour* as submodularity (diminishing returns) on a
  different axis — a separate curvature argument, not a free lunch from `Ω`.
- *Finite-set (Rung 3), the rigorous version.* Materialize a small set `Φ` of form configs
  (`k∈{0,1,2,4,8}` × a few personas/formats), run the **content certificate at each `φ∈Φ`**, and take
  best-in-set over the **product** `{S⊆Ω} × Φ` with a bootstrap CI. This certifies the joint optimum over
  the product **without assuming `R(S,φ)` separates** — content×form interaction (a criterion that only
  fires with the right exemplars) is handled directly by re-certifying content *per form* and comparing.
  The nesting is `max_{φ∈Φ}[ max_{S⊆Ω} R(S,φ) ]`: inner submodular-certified, outer finite-set-certified.

**The honest residual — form does not introduce a *new* kind of gap.** The product certificate is over
*materialized* `Ω` × *materialized* `Φ`, not the infinite space of all exemplar sets / phrasings. Form has
its **own Discovery/coverage gap**, exactly parallel to `Ω`'s (§6.5): we certify selection within what we
enumerated, never closure over the infinite universe. So the criticism, fully met, reduces to the *same*
conditional-on-materialization caveat that already governs content — form **doubles** the existing
Discovery gap (content-discovery + form-discovery), both bounded loosely by Rung 1 and **measured** by the
coverage residual `Δ_cover = R(actual full prompt) − OPT over subsets of Ω` (a positive residual attributes
to form configs outside `Φ` *and* criteria outside `Ω`; the two are separated by re-running selection with
`Φ` widened). We do **not** claim the within-class bound is a global prompt optimum over style; we claim it
is a certified optimum over `{materialized content} × {materialized form}`, with form as an explicit outer
finite-set coordinate carrying its own concavity — *not* an omitted variable.

**Ordering is the same kind of coordinate — and it is why the combinatorics is over *sets*.** A prompt is
a *subset* `S ⊆ Ω`; the canonical compiler `C(S)` lays the chosen criteria in a **fixed index order**, so
`f(S)=R(C(S))` has one value per subset (`2^|Ω|` of them). This is *required*, not incidental:
submodularity is defined on **sets**, so we do **not** permute inside the combinatorics. Order does matter
in prompting — but ordering is a **form coordinate** (no per-item `X_e` signal; it modulates
presentation/executor fidelity), so it gets the treatment above verbatim: fix a canonical order to make `f`
a set function, then handle order as an outer finite-set coordinate (sample a few orderings per subset,
best-in-set with a CI). Folding order *into* the combinatorics would blow the space from `2^|Ω|` subsets to
`Σ_k k!·C(|Ω|,k)` ordered sequences and destroy the lattice (no diminishing-returns on sequences) — the raw
string-optimization we abstracted away to *obtain* a certificate. The set-abstraction's error is
**measurable**: it equals the executor's **order-sensitivity** — permute a subset's criteria, observe how
much `R` moves; small ⇒ checklist-like executor, abstraction justified; large ⇒ lift order to a form
coordinate. Empirically order is second-order vs. *which* criteria, so canonical-order `f` is a good proxy
— a bet we can check with a permutation test, not an assumption we smuggle.

**The Permutation Test, made explicit (formalizing the order-residual bound).** The set-abstraction
`f(S)=R(C(S))` is justified as a proxy iff *which criteria* dominates over *in what order*. Measure both
variances directly:
- `σ²_subset` = variance of `R` across **distinct subsets** `S⊆Ω` (the signal the set-function carries);
- `σ²_perm` = variance of `R` across **permutations of one fixed subset** (the order residual the canonical
  compiler discards).
**Pass condition: `σ²_subset ≫ σ²_perm`** — then subset-selection is the dominant factor and order is a
*bounded second-order residual*, so the set-abstraction is a sound proxy and the certified `OPT_Ω` is a
real bound up to an order term of size `√σ²_perm`. If `σ²_perm` is comparable to `σ²_subset`, the executor
is order-sensitive and order must be lifted to a form coordinate (sample orderings per subset, best-in-set
with a CI, above). This is the empirical contract behind "order is second-order," not an assumption —
implemented as `orthogonalize.permutation_order_test` (planted check: `σ²_subset/σ²_perm ≈ 900`).

---

### 6.9 Ω→∞: the discovery scaling law (conditional on tail-γ and saturation). **[new]**

*GEPA proposes no scaling law for its genetic search. We can — for the discovery limit — because the
within-Ω optimum is a **bounded monotone** sequence.*

**The limit exists, safely.** `OPT_Ω = max_{S⊆Ω} R(C(S))` is **monotone non-decreasing in `Ω`** (a larger
pool retains every smaller pool's subsets) and **bounded by the cap** (`≤ ½`). A bounded monotone sequence
converges: `OPT_{Ω_t} → OPT_∞ ≤ ½` as the generator discovers criteria over steps `t`. This *sidesteps* the
standing "no scaling-law extrapolation / V-info is non-Lipschitz" rule — that warned against extrapolating
an *unbounded non-Lipschitz functional*; here the object is a bounded monotone convergent sequence, and only
its tail *rate* is extrapolated, under the explicit conditions below.

**The scaling law — two measurable ingredients.**
1. **Submodular curvature over the universe (the same `γ`).** If `R` is submodular over the criterion
   universe, newly discovered criteria have diminishing marginal value — high-impact criteria are found
   early (greedy picks high-marginal first), late ones are low-marginal *by submodularity*. This caps the
   coverage gap.
2. **The generator's discovery curve.** GEPA's rate of producing *new distinct* criteria per step is
   typically a Heaps'/Zipf law. Combined with (1), the residual `OPT_∞ − OPT_{Ω_t}` decays as a **power law
   in `t`**.

**Importable backbone.** This is *subsampled / random-ground-set submodular maximization*: a random ground
set of size `~(n/k)·log(1/ε)` retains `(1 − 1/e − ε)` of the population optimum. That converts "how many
criteria discovered" into an approximation guarantee on `OPT_∞` — the Ω→∞ result — **provided the generator
samples criteria representatively.**

**The wall (it is conditional).**
- **Tail synergy breaks it.** The scaling holds only where `γ ≈ 1`. A rare *synergistic* criterion
  undiscovered in the tail (γ<1) can make `OPT` *jump* on discovery — the **missing-impact** problem. The
  test is now the **submodular tail-bound + adversarial behavioral saturation** of §6.7c (the tail-bound
  certifies missing impact *given* tail-`γ≈1`; the adversarial probe `I(X_probe;M∣X_Ω)≈0` catches the
  tail-synergy case the bound's condition excludes). The old Good–Turing missing-*mass* estimate is
  dropped — it bounds the probability of an unseen criterion, not its impact.
- **Correlated blind spots.** LLM generators share gaps, so the discovery curve is generator-specific and
  only empirically estimable.

**So we certify the limit only conditional on** (a) measured `γ ≈ 1` in the *discovered tail*, and (b) the
discovery curve *saturating*. Both hold ⇒ extrapolation to `OPT_∞` is trustworthy; either fails ⇒ the limit
is not yet bounded, keep mining. **Testable now:** we already log the full GEPA lineage, so fit `OPT_{Ω_t}`
vs `t`, watch tail-`γ`, and check saturation — the scaling law GEPA omits, and the principled bridge from
the within-Ω certificate (Rung 2) toward the global one (Rung 1).

> **Correction (2026-07-12; historical scaling proposal retained).** Measured `γ` and a flattening discovery
> curve can guide compute allocation, but they do not certify an `Ω→∞` limit: neither statistic constrains
> an element outside the sampled ground set. The prospective CR-3 confirmation audit supplies a valid
> finite-horizon result; full proposer-support exhaustion additionally requires external positivity.

---

### 6.10 Validity: side-channels, v-information, and the PRUNE-as-leakage reframe. **[new, 2026-06-23]**

*Crystallizes what is actually load-bearing, prompted by "does the theory break without submodularity?"*

**Two orthogonal axes — do not conflate.** Submodularity (`γ`, §6.2/§6.6) is the *combinatorial* axis (does recovery have diminishing returns?). **Validity** is separate: is the recovered `I(M;M̂)` *causal/semantic*, or a *confounded side-channel* (item length, formatting, a surface marker the executor latches onto)? TVD-MI is a correlation measure — high `I` means `M̂` *predicts* `M`, not that it captures the intended signal (§4.3, §10 item 7 already flag label-free ≠ correctness). The contribution here is making validity *operational*.

**PRUNE-help is a leakage alarm, not a structural claim.** §6.1 treats PRUNE (adding a criterion lowers `R`) as non-monotonicity. Reframe: if removing criterion `e` *raises* recovery, `e` carried *confounded/spurious* signal — it helped alone (via the side-channel's strong marginal) but *interfered in combination* (double-counting/contradiction). So **PRUNE-help ⟹ suspect `e` of side-channel leakage.** (The structural reading — "metric construction is non-monotone" — is true but unsurprising from prompt-craft; the leakage reading is actionable.)

**No heavy ignorability machinery — v-information gives it by construction.** The causal-inference reflex (back-door adjustment, IV) is unneeded. Xu et al.'s **v-information** `I_V(Y→X)` filters information through a *constrained predictor class* `V`: it excludes any signal `V` cannot represent. Our predictor class is `V = {criterion-prompt → E's verdict}`; constrain the criteria to be causally-semantic and `I_V` is clean — *no causal model of the data-generating process required.* (This is exactly why v-information was introduced — to sidestep the unidentifiability of natural-language entropy; cf. `project_irreducible_E_estimation_2026_06_13`.) Ignorability here is a property of the *channel class*, not the confounder graph.

**The channel-cleanliness gate (operationalizing the predictor-class constraint).** A criterion *stays* in `Ω` iff ALL hold — instruments already built:
1. `orthogonalize.adversarial_saturation`: `I(X_probe; M | X_Ω) ≈ 0` for side-channel probes (length/format/OOD) — a conditional-independence test that `Ω` *blocks* the channel (= ignorability given `Ω`; assumption-free, §6.7c leg 2);
2. **no PRUNE-help**: removing `e` does not raise `R(C(Ω))` (the leakage alarm, computable free from the certificate's per-subset `R`-dict);
3. `measures.counterfactual_validity`: `e` tracks a *planted* direction, not a confound.

**The convergence — why we never "prove γ" standalone.** Side-channel criteria are a *major source* of the non-monotonicity §6.1 measures (a confound helps alone, interferes in combination ⇒ PRUNE). Cleaning channels therefore removes the **spurious** (leakage-driven) non-monotonicity, leaving only genuine semantic interactions — small for well-designed criteria. So **clean channels ⇒ near-monotone ⇒ near-submodular as a *consequence*, not an assumption** — the cheap `γ` guarantees (§6.2/§6.6) are *re-enabled as a bonus* of validity work we'd do anyway. Honest caveat: "clean ⇒ near-monotone" is empirical-mostly, not a theorem (genuine semantic synergy, §6.6, can still be non-monotone); but it removes the *dangerous* kind of PRUNE — the leakage kind — leaving a small, named residual.

**Net.** Lead the optimality story with the submodularity-free core (DPI, held-out `R`, exact `|Ω|≤15` cert, `A`); treat submodularity as an *emergent* near-property of clean channels, established by the cleanliness gate, not assumed; for `|Ω|>15` report the approximation as measured-not-certified (already the code's behavior). The thing that can actually break a bound is a *side-channel*, and we block those by construction, not by adjustment.

> **Correction (2026-07-12; historical validity argument retained).** The finite battery is a falsification
> test, not an assumption-free proof of ignorability or prompt-space closure. A constrained predictor class
> defines what signal is measured; it does not remove the need for a causal/measurement argument. For bound
> semantics, use the target-indexed DPI and §12.6b; treat cleanliness and submodularity as diagnostics.

---

## 7. What the TVD-MI switch changes

We use **TVD-MI** `= TV(P_{M,M̂}, P_M⊗P_M̂)` for recovery, not Shannon. Three consequences:
1. **Convexity survives** (Fact 1 is an f-divergence statement): all of §2/§4 hold verbatim. ✓
2. **Gaming robustness is the *reason*.** TVD is the bounded f-divergence: Robertson–Koyejo show
   bounded-f-MI keeps **polynomial** guarantees under strategic manipulation while KL-MI degrades
   **exponentially** (and TVD is the unique f-divergence with both a bounded and a *binary* optimal
   critic, `I_TVD(Y_i;Y_j) ≥ TPR+TNR−1`). For a no-ground-truth measure we optimize prompts *against*,
   this is the property we want (Goodhart / collusion defense).
3. **The cap (rung 1) holds for TVD-MI — but only via the f-DPI proof** (Fact 2), since the Shannon
   chain-rule shortcut doesn't transfer. *Estimator caveat:* the proofs are about *population* f-MI;
   the shipped estimator is binned + permutation-debiased + finite-pass, so check `R̂ ≤ T̂` empirically
   (§5 guardrail) rather than assuming it pointwise. (Follow-up: TVD-MI IRT/additivity paper —
   additivity ↔ the modular `α→0` regime of §6.2.)

---

## 8. thin/thick ↔ `(α, γ, A)` — hypothesis + validation. **[conjectural]**

| coordinate | small ⇒ | large ⇒ | from |
|---|---|---|---|
| curvature `α` | modular, decomposable (greedy exact) | saturating/redundant | §6.2 |
| submodularity ratio `γ` (small) | — | complementary/holistic | §6.2 |
| articulation gap `A=T−R` | fully articulable | tacit residual survives | §5 |

**Hypothesis.** thin: `α≈0, γ≈1, A≈0`; thick: `γ≪1` and/or `A≫0`. These need **not** coincide (`α,γ` are
criterion-interaction; `A` is articulation loss). **Validation:** measure `(α,γ,A)` per metric; correlate
with independent thickness proxies (`v_struct` on legal slices; expert thin/thick ratings; the law 2×2).
Only a positive correlation upgrades this from hypothesis. Until then: conjectural.

---

## 9. Scorecard — claim → status → measurement

| # | claim | status | measure on corpus |
|---|---|---|---|
| F1 | f-MI (Shannon **and** TVD) convex in behavior | **holds** | — (analytic) |
| F2 | `R ≤ T_test` (f-DPI, common-cause) — bounding `T` is the **held-out** `I_f(M̂_test;X_test)`, NOT train consistency `T_train` (DPI within one distribution) | **holds (corrected)** | check `R̂ ≤ T̂_test` every cell (`tvd_guardrail`) |
| **R1** | `OPT ≤ cap_f` (TVD `1−1/min(N,K)`, Shannon `log min(N,K)`) — a **channel-capacity sanity check** (verify `R̂ ≤ cap_f`, detect binary-readout compression), **not** a proximity-to-optimum KPI; do not report `cap_f − R̂` as distance-to-optimum | **holds (downgraded label)** | `R̂(p̂) ≤ cap_f`? readout `K` |
| **R1′** | cap is the strongest *assumption-free* global `U`; other structure (model/grammar/Lipschitz) could give other global bounds | **holds** | — (argument) |
| **R2** | `OPT_class ≤ R(S_g)+(1/γ)Σ_{top-k}δ` (per-instance, within class) — **corollary of the γ-def + monotone, proof in §3.2** | **holds (derived)** | greedy marginal gains |
| **R2′** | `R(S_g) ≥ (1/α)(1−e^{−αγ})·OPT_class` (worst-case) | **holds (Bian'17)** | estimate `(α,γ)` |
| **R3** | best-in-set at `1−δ` via bootstrap CIs | **holds** | CI dominance among `{p_j}` |
| **R4** | edit-stationarity necessary; saturation `R=T` witness; local=global unproven | **holds (caveat)** | optimizer-agreement |
| S1 | `p*` (for `T`) at extreme point = pure prompt; score-avg never helps `T` | **holds (scoped)** | avg-prob vs vote ensemble `T` |
| S2 | binary `T`-optimum = balanced deterministic ½/½ | **holds** | optimal-rubric `p̄`, hedge rate |
| S3 | `T ≠ R`; `T` gameable by spurious spread | **holds (key fix)** | high-`T`/low-`R` cells |
| S4 | `R` not known convex ⇒ no convex-max on objective | **holds** | — (structural) |
| A1 | `A=T−R` = articulation loss (metric, **not** prompt cert) | **holds (def.)** | per-metric `A`; rank metrics |
| B1 | `R(S)` non-monotone ⇒ **double-greedy ½** (rand.) / 1/3 (det.) / 1/e (card.), not 1−1/e | **corrected** | ablate; test monotonicity |
| B2 | `γ` is empirical/structural (not analytic); **brute-force `|Ω|≤15` bypasses `γ`**; `γ` trustworthy only on an **orthogonalized `Ω`**; exact `γ` is `O(2^M)`; **spectral `λ_min` DROPPED** (linear-regression assumption false for LLM) | **holds (corrected)** | orthogonalization filter then lower-tail sampling |
| B3 | Discovery-to-Selection: GEPA-mine **semantic diffs** → **orthogonalization filter** (Shannon CMI) → canonical compiler (Format→Semantics→Negative) ⇒ behaviorally-orthogonal set fn; certifies within `Ω`, **not** global | **holds (scoped)** | mine diffs; `orthogonalization_filter`; `C(S)`; brute/double-greedy on `f=R∘C` |
| B4 | submodularity is a **conditional theorem**: `Ǐ(S)=I(M;X_S)` submodular if criteria CI given `M`; `γ<1` ⟺ synergy (co-information); executor compression breaks it; orthogonalization filter enforces CI by construction | **holds (conditional)** | pairwise `I(X_i;X_j∣M)−I(X_i;X_j)`; orthogonalization filter |
| B5 | small `Ω` (`|Ω|≤15`, default) ⇒ **brute-force real `R(C(S))`** = exact within-class optimum, **bypassing `γ`** (no approx, no monotone, no joint-MI undersampling, executor included) | **holds** | enumerate `Σ_{j≤k}\binom{|Ω|}{j}` subsets, score each |
| B6 | magic-word defense = **submodular tail-bound** (`max_{e∉Ω}Δ(e∣Ω) ≤ min_i[greedy gain]_i`, bounds missing *impact* given tail-`γ≈1`; loo-min is a tighter diagnostic) **+ adversarial behavioral saturation** (`I(X_probe;M∣X_Ω)≈0`); Good–Turing missing-*mass* DROPPED | **historical claim; corrected 2026-07-12** | `submodular_tail_bound`; `adversarial_saturation` |
| B6′ | correction: in-pool tail gains and finite adversarial batteries cannot bound an element outside `Ω`; use fresh all-draw gain marks for a finite horizon, or exact mass plus external positivity for support exhaustion | **holds in §12.6b scope** | `prompt_articulation_certificate` |
| B7 | set-abstraction `f(S)=R(C(S))` valid iff `σ²_subset ≫ σ²_perm` (order is bounded 2nd-order residual) | **holds** | `permutation_order_test` |
| B8 | atomic unit = **behavioral partition operator** (orthogonal shift in `E`'s behavior), not logical irreducibility; composites the executor reads as one signal are NOT split | **holds (def.)** | `X_e` partition equality |
| B9 | the certificate measures **two** pairwise quantities: `I(M, M_ω)` (decomposition gap, prose↔all-criteria) and `I(M_ω, M_s)` (selection gap); `M` = prose `p̂`, `Ω` = post-hoc GLM decomposition, not the prompt itself | **holds (def., 2026-06-23)** | `omega_certificate --prose-prompt`; each `I` read vs its own `T` |
| B10 | prose `p̂` is the prompt; structured `Ω` is a *separate* decomposition pass (`real_gamma._decompose`), never forced into the executed prompt | **holds (design)** | Phase A prose → GLM decompose → orthogonalize |
| A2 | SLT bounds component (b) of `A` (`O(√(d/N))`); does **not** restore a V-info ceiling | **holds** | reconstruction-family complexity vs `N_held` |
| H1 | `(α,γ,A)` ↔ thin/thick | **conjectural** | correlate `v_struct`/expert |
| V1 | **validity ≠ submodularity**: side-channel leakage is the load-bearing risk (TVD-MI = correlation, not causal); submodularity is an *optional* top layer (γ/U₂/tail-bound only for \|Ω\|>15 fallback) — core (DPI, held-out R, exact cert ≤15, A) is submodularity-FREE; theory does NOT break without it | **holds (framing)** | cleanliness gate (V3) |
| V2 | **PRUNE-help = leakage alarm**: removing criterion `e` raising `R` ⟹ `e` confounded (helped alone via side-channel, interfered in combination); reframe of §6.1, free from the per-subset R-dict | **holds (reframe)** | `omega_certificate` R-dict |
| V3 | **v-information predictor-class = ignorability by construction** (no back-door/IV); channel-cleanliness gate (`adversarial_saturation` I(X_probe;M\|X_Ω)≈0 + no-PRUNE-help + `counterfactual_validity`); clean channels ⇒ near-monotone ⇒ near-submodular *emerges*, never proven standalone | **historical claim; corrected 2026-07-12** | `adversarial_saturation`, `counterfactual_validity`, R-dict |
| V3′ | correction: constrained predictor classes and cleanliness batteries define/diagnose a channel; they do not prove causal ignorability or prompt-space closure | **descriptive/falsification only** | same instruments |

---

## 10. Honest limitations

> **Scope constraint (2026-06-22): every bound in this file certifies *pure* `R(p)`.** Rungs 1–4, the cap,
> the within-class certificate, the DPI guardrail (§2.2), and the scaling law (§6.9) are all statements
> about the label-free recovery objective `R = I_TVD(M_test; M̂_test)` and nothing else. They do **not**
> automatically transfer to the shipped fidelity *scalarization*
> (`w_cf·CF + w_recon·recon + w_reliability·rel + …`); item 6 gives the only routes by which a certificate
> on the blend can be recovered (component-wise convexity + a composition bound on the blend's `γ` + the
> budget caps as a knapsack). Read every "certified" claim as "certified for `R`," not "certified for the
> production objective."

1. **The global gap is governed by `B_E`-ignorance, not effort.** Rung 1's `U₁` is the cube sup; the
   achievable `sup_{B_E}T` is unknowable label-free. This is the structural ceiling, not a TODO.
2. **`R` not known convex (§2.3).** All clean structure (extreme point, ½/½, cap) is proven for `T`;
   `R` inherits only `R ≤ T`. Do not assert convexity-based claims about the objective `R`.
3. **`R(S)` as a clean set function is an idealization.** Real rubrics aren't disjoint criterion unions
   — wording/order/few-shots interact. `(α,γ)` are on a *canonical realization*; measure
   realization-variance before trusting them.
4. **Estimator ≠ functional.** §2–§3 are population statements; estimators are binned +
   permutation-debiased + finite-pass. `R̂ ≤ T̂` is the cheapest reality check.
5. **`B_E` is executor-specific — and so is the certified optimum (the single-LLM scope).** Every result
   is relative to `E`; the certified-optimal prompt is optimal *for that executor*, not universally.
   The target `M` is a **behavioral** standard — a bank metric's own verdict (`recon_channel`), or, for
   the certificate path, the **prose prompt `p̂`'s** verdict (`small_omega_brute_force` / `omega_certificate`,
   §6.7a′; 2026-06-23) — **not** a strong-LLM "holistic quality" anchor. (The criterion set `Ω` is a
   post-hoc decomposition of `p̂`, and `M_ω` = `E`'s verdict on `C(Ω)` is a *separate* measured quantity,
   not the target.) So the certificate is reconstruction of that prose standard, not self-recovery of `E`'s
   taste. Residual single-model scope remains when `M` and `M̂` come from the *same* executor; the §6.7a
   brute-force still gives the exact optimum for that pairing.
   Three extensions restore cross-model breadth (all keep the §6.7a brute-force; submodularity transfers only for the
   average):
   - **Cross-model target `M`.** Make `M` a **consensus** (majority, or mean-then-median-split over
     models, or a strong anchor) so the optimum certifies recovery of a *shared* standard.
   - **Average objective** `R_avg(S) = (1/m)Σ_i R(C(S);E_i)` — a non-negative combination of per-model
     recoveries, so **both brute-force *and* submodularity transfer** (sum of submodular is submodular):
     certifies the prompt optimal *in expectation over the model family*.
   - **Worst-case objective** `R_min(S) = min_i R(C(S);E_i)` — certifies optimality for the *weakest*
     model; **brute-force transfers, submodular bounds do not** (a min of submodular functions is not
     submodular).
   The apparatus is an `E`-indexed *family*; the scientifically interesting object is the **`E`-dependence
   itself** — capability–articulation substitution: does a stronger `E` reach the same recovery with
   *fewer* explicit criteria? — not a single model-collapsed number.
6. **Objective mismatch with the shipped system — and how to make the certificate transfer.** The
   optimizer maximizes the fidelity *scalarization* (`config.py`:
   `w_cf·CF + w_recon·recon + w_reliability·rel + w_consistency·cons + w_oracle·oracle`), not pure `R`.
   Two transfer rules: **(convexity, §2)** the sum is convex in behavior iff *every weighted component* is
   — `T`-type terms (reliability/consistency) are f-MI and convex, but `CF`/`recon` are not obviously the
   same form and must be checked. **(submodularity, §6)** the certificate needs the *blend* to be (weakly)
   submodular; the right tool is a **composition bound** on `γ` of the weighted sum (not the tempting
   "`R + modular cost`" shortcut — our objective is **not** `R` plus a length penalty; it is a sum of
   *several non-modular* measures, so there is no free modular term to lean on). What *does* map cleanly:
   the **budget caps** (`instruction_tokens`/`n_fewshots`/`data_budget`) are **knapsack/cardinality
   constraints**, so once the blend's `γ` is bounded, *submodular-maximization-under-a-knapsack* results
   (Sviridenko) give the constrained certificate. Certificates are clean for `T` and `R` individually;
   for the running objective, bound `γ` of the blend and treat the caps as the knapsack.
7. **Bounds articulability, never correctness.** Both `T` and `R` are label-free; neither separates the
   right attribute from an articulable wrong one (§4.3). Any "this metric is *correct*" claim needs `Y`
   and is outside this theory by construction.

---

### One-paragraph takeaway

Global optimality of an unsupervised-metric prompt over the infinite string set is — *absent further
structure on `B_E`* — certifiable only to within the **information cap** (rung 1: `1 − 1/min(N,K)` for
the TVD-MI objective, binary ½; Shannon `log min(N,K)`) — proven because transmission `T` is convex and
recovery `R ≤ T`, but loose, and loose for a *structural* reason: we can never see `B_E`, the set of
behaviors a prompt can induce. Tightening means shrinking scope: **rung 2** gives a tight per-instance
bound `R(S_g)+(1/γ)Σ_{top-k}δ`, but only for the best *criterion-set* rubric; **rung 3** gives
best-among-`{p_1..p_k}` with bootstrap CIs; **rung 4** gives local necessary conditions and an
optimizer-agreement proxy for "local = global." The chat sketch's convexity and submodularity instincts
were right; the load-bearing corrections were (a) `T` is convex but **`R` is not**, so we cap `R` via
`T` and search the objective directly; (b) the cap is the strongest *assumption-free* global certificate
(other structure could give others), its looseness being `B_E`-ignorance rather than a missing theorem;
(c) `A = T − R` is a measurement of the metric, not
a certificate for the prompt, and belongs after the optimality story, not in front of it.

---

#### References (verified 2026-06-18)

- Cover & Thomas, *Elements of Information Theory*, Thm 2.7.4 — `I(X;Y)` convex in `p(y|x)` for fixed
  `p(x)` (the `f=KL` case of Fact 1).
- Csiszár — f-divergence joint convexity (perspective-function argument).
- f-divergence data-processing inequality (standard; Csiszár–Shields) — Fact 2.
- Minoux (1978) — accelerated greedy / the online (Lagrangian) submodular upper bound (rung 2 `U₂`).
- Krause & Guestrin (2005), *Near-optimal Nonmyopic Value of Information* — MI submodular under
  conditional independence given the target.
- Das & Kempe (2011 / JMLR 2018) — submodularity ratio `γ`, greedy `≥ (1−e^{−γ})`. *(Their spectral
  `γ ≥ λ_min` bound is **dropped** here — §6.2 — because it assumes a linear-regression value function,
  false for an attention-mixing LLM executor; the orthogonalization filter replaces it.)*
- Conforti & Cornuéjols (1984) — total curvature `κ`.
- Bian, Buhmann, Krause, Tschiatschek (ICML 2017) — `(1/α)(1−e^{−αγ})` for non-submodular greedy (monotone).
- Nemhauser, Wolsey, Fisher (1978) — monotone submodular greedy `(1−1/e)`.
- Buchbinder, Feldman, Naor, Schwartz (2015) — unconstrained non-monotone submodular: **deterministic
  double-greedy `1/3`, randomized `½`** (tight); random-greedy `1/e` under cardinality.
- Sviridenko (2004) — submodular maximization under a **knapsack** constraint (the budget-cap case, §10 item 6).
- Calinescu, Chekuri, Pál, Vondrák (2011) — multilinear extension / continuous greedy.
- Mohri, Rostamizadeh, Talwalkar, *Foundations of ML* — Rademacher/VC generalization bounds (§5,
  bounding component (b) of `A`; **not** a recovery ceiling).
- Robertson & Koyejo (2025), *Let's Measure Information Step-by-Step* (arXiv 2508.05469) — TVD-MI,
  bounded-f-divergence gaming robustness; IRT/additivity follow-up (arXiv 2510.14966).
- Xu et al. (2020), *A Theory of Usable Information* — predictive V-information (contrast: our `I_V` is
  Shannon; the V-restriction is in the channel).
- Good (1953) — missing-mass / Good–Turing estimation. *(**Superseded** for the magic-word defense — §6.7c
  — by the submodular tail-bound + adversarial behavioral saturation, which bound missing *impact*; Good–
  Turing bounds only missing *mass*.)*

*Related work — prompt optimization & submodularity (verified 2026-06-19):*
- Robertson & Koyejo (2025), *Let's Measure Information Step-by-Step* (arXiv 2508.05469) — TVD-MI as a
  no-ground-truth evaluator (we optimize the prompt *against* it).
- Nian et al. (2026), *Submodular Evaluation Subset Selection in Automatic Prompt Optimization* (SESS,
  arXiv 2601.03493) — submodular *evaluation-data* selection; greedy speed, not prompt certification.
- *Select Smarter, Not More: Prompt-Aware Evaluation Scheduling with Submodular Guarantees* (arXiv
  2604.11328) — submodular eval scheduling for APO.
- Query-Focused Submodular Demonstration Selection (IEEE); InSQuAD (arXiv 2508.21003) — submodular
  few-shot/ICL selection. (NB: a specific "SMILE, ICML 2026" title was cited to us but not verified;
  these are the verified submodular-ICL-selection works.)
- Yang et al. (2023), *Large Language Models as Optimizers* (OPRO); Yuksekgonul et al. (2024), *TextGrad*;
  Jain & Chowdhary (NAACL 2025), *Local Prompt Optimization* (arXiv 2504.20355, explicitly local);
  Zhou et al. (APE); Pryzant et al. (ProTeGi) — empirical local APO, no global guarantee.

---

## 11. Recovery vs. within-class — which path gives global optimality? **[new, 2026-06-25, post-MCQ-pivot]**

*Reconciles the §6 within-class ladder with the recovery objective actually shipped
(`experiments/run_r2_recovery.py`, `recon_channel.run_metric`). Written after the 2026-06-25 pivot from the
within-class certificate to recovery = C(R(Ω)) = I(M_ω; M′) ([[feedback_report_recovery_metric_only]]).
Answers: for the shipped objective, what IS the global-optimality certificate, and is the §6 subset/Ω-coverage
machinery still needed for it?*

### 11.0 Formal definitions — "within-class" vs "out-of-class"

Let `Σ*` = all finite strings (prompts); `Ω = {c₁,…,c_K}` = a finite mined ground set of atomic criteria;
`C : 2^Ω → Σ*` = a fixed canonical compiler (template + fixed section order, §6.5 step 3).

> **Criterion class:** `𝒫_Ω := { C(S) : S ⊆ Ω } ⊆ Σ*` — every prompt the compiler can assemble from subsets
> of the mined criteria. Its induced behaviors `ℬ_Ω := { s_p : p ∈ 𝒫_Ω } ⊆ B_E`.
>
> **Within-class:** any `p ∈ 𝒫_Ω` (i.e. `p = C(S)` for some `S ⊆ Ω`). Within-class optimum
> `OPT_Ω := max_{S⊆Ω} R(C(S))`.
>
> **Out-of-class:** any `p ∈ Σ* \ 𝒫_Ω` — *not* expressible as `C(S)`. Includes: criteria never mined
> (incomplete `Ω`); holistic / non-decomposable phrasings; forms/orderings/personas/few-shot configs outside
> `C`'s template; arbitrary free-form prose.
>
> **Magic-words gap:** `OPT − OPT_Ω`, `OPT = sup_{p∈Σ*} R(p)` — the most an out-of-class prompt could exceed
> the best in-class one.

*Two distinct "class" notions — do not conflate:* (1) the **syntactic criterion class** `𝒫_Ω` (finite,
enumerable — what brute-force / submodular search ranges over; THE class for the within-class certificate);
(2) **behavioral species** — equivalence classes of prompts under behavioral equivalence (`s_p = s_{p′}`, or
near-equal per-item signal `X_p`) — the capture–recapture / orthogonalization-filter notion (§6.5, §11.3),
used to estimate `|B_E|` coverage. Different objects.

**Why the boundary matters.** The within-class certificate (brute-force exact / submodular `U₂`) is valid
ONLY over `𝒫_Ω` — "no `S ⊆ Ω` beats `S*`." It is provably silent about `Σ* \ 𝒫_Ω`. Every tightening (§6.5
enlarge `Ω`; §6.8 enrich `C` with form/ordering) grows `𝒫_Ω → Σ*` but never closes it (`B_E` unknowable).

### 11.1 The recovery objective has a global all-Σ* ceiling the within-class objective lacks

The within-class certificate (§6, `OmegaCertificate`) measures `OPT_Ω` — exact *within the pool*, never
global; its global gap is bounded only conditionally (tail-`γ≈1`, §6.7c) and never tightly, because `B_E` is
unknowable label-free (§10 item 1).

The **recovery** objective `R = I(M_ω; M′)` has a stronger structure: `M` and `M′` share **only** the
held-out item `X` — `M ⟂ M′ | X` (reconstructor trained on a disjoint split never sees `X`). So
`M → X → M′` is a Markov chain, and the f-DPI (Fact 2, §2.2) gives, for **every** prompt the reconstructor
could produce:

> **`R = I_f(M_ω; M′) ≤ I_f(M_ω; X) = T(m_ω)`** — a bound over **all** of `Σ*`, not just `Ω`.

The upper bound is global, but tightness is conditional:
- **Attainability in the unrestricted readout space.** If an arbitrary measurable readout is allowed, the
  soft posterior `η(X) = P(M_omega=1|X)` is sufficient and attains Shannon equality
  `I(M_omega;η(X)) = I(M_omega;X)`. This proves tightness over the unrestricted readout space.
- **Realizability by prompts is an additional assumption.** The prompt behavior set `B_E` need not contain
  `η`, and a binary sampled verdict is not the posterior statistic. Therefore
  `sup_{p in Sigma*} R(p) <= T(m_omega)` always, but equality holds only if the declared prompt/readout
  interface can realize a sufficient statistic (or approach it in the relevant divergence). Putting the
  target option in an MCQ establishes this only when re-executing that option exposes the same sufficient
  continuous readout; an independent sampled copy does not attain `T` for a stochastic target.
- **What headroom certifies.** `T(m_omega)-R(p_hat)` is a valid *upper bound* on the candidate's true
  suboptimality for the fixed-target problem. It is not necessarily the exact gap, and a large headroom does
  not show that a better prompt exists. Population claims additionally require a one-sided upper confidence
  bound on `T` and a lower confidence bound on `R`, evaluated on iid items after the candidate is frozen.

This is Rung 1 (§3.1) instantiated at the fixed target's transmission `T(m_omega)`, not merely the
distribution-free channel cap. It can be much tighter than `cap_f`, but it remains an upper bound unless
prompt-class realizability is established. Consequently, `headroom` is a certified upper bound on the
fixed-target optimization gap, not an identified decomposition of that gap into articulability alone.
**Scope — do NOT over-read `T` as the ideal's bound:** `T(m_ω)` is the ceiling on recovering the
OPERATIONALIZED metric `m_ω`, which is the **FLOOR** on the *ideal* `M_i*` (§11.2 intent row) — a richer pool
reaches more of the ideal. The ideal's UPPER bound is a different object, lives beyond `Ω`, and (with **no
`Y`**) is approached unsupervised in **§12.4**, not here. The
shipped `missing_impact` (= 0 on the first run) is a *within-pool* diagnostic, already silent; the global
statement lives in `headroom` alone.

> **Correction (2026-07-12).** The floor claim requires a measurement/garbling relation that was not
> established. `T(m_ω)` bounds only the fixed operational target. It has no automatic ordering with
> `T(M_i*)`; see the corrected target-indexed statement immediately below and §12.6b-F.

> **Reconciliation.** §6's exhaustive subset search can certify the empirical optimum *within a declared
> finite Omega class*. The fixed-target DPI bound is global over all candidate prompts and therefore
> supersedes Omega machinery for producing an all-prompt upper bound. It does **not** dissolve the class
> boundary for attainability: `B_E` still determines whether any prompt can approach `T(m_omega)`. Omega
> diagnostics can show achieved structure or supply candidates, but neither Omega coverage nor free
> generation proves that the posterior readout is prompt-realizable.

### 11.1a Reconstruction measures anchor-free annotation fidelity; it is not an upper-bound method. **[clarified 2026-07-12]**

Reconstruction was introduced as a **measurement technique** for a setting with no anchor labels or
ground-truth quality variable. Freeze a metric operationalization `M`, draw a training sample, and ask a
reconstructor `W` to infer what criterion could have produced the observed `(text, M)` annotations. A
fresh executor then applies the recovered metric on held-out items:

```
D_train --W--> P_W,          Y = binarize(E(P_W, X_test)).
```

The scientific read is annotation fidelity/recoverability: do the metric's annotations contain enough
stable, metric-specific structure for another system to infer the annotating criterion and reproduce its
held-out decisions? High recovery supports coherent annotation; low recovery can arise from noisy or
underspecified annotations, reconstructor noise, executor noise, or an inadequate reconstruction interface.
It does **not** establish correctness, because there is no external `Y`, and it was never intended to issue
a prompt-space upper bound.

Mathematically, each recovered prompt is also an achieved prompt channel, so its held-out value lies below
the corresponding prompt optimum. That inequality is a useful consistency fact, not the experiment's
measurement purpose and not a reason to fold reconstruction into the CR-3 certificate.

The target-indexed DPI still supplies the upper side:

```
I(M;Y) <= A_E(M;P) <= I(M;X) <= H(M).
```

When the implemented target is the deterministic threshold of its executor `P(YES)` vector,
`I(M;X)=H(M)`. That is a separate, usually loose all-prompt cap; reconstruction does not tighten it.

**Why MCQ is appropriate here.** Free end-to-end reconstruction entangles annotation informativeness with
free-form hypothesis writing and prompt application, producing high variance and a low empirical
measurement ceiling. MCQ deliberately supplies a closed codebook containing the target. The reconstructor
identifies a metric description from demonstrations, and the selected identity maps to its canonical rubric
body for re-execution. That mapping is part of the declared measurement design, not a hidden side channel.
MCQ therefore asks a narrower and cleaner question: *do these annotations distinguish their metric from
plausible alternatives?* Identification accuracy is the direct readout; held-out behavioral recovery is a
complement that gives partial credit when a wrong identity is behaviorally close.

> **Implementation correction (2026-07-12): behavior designs the lesson; it is not shown to the
> reconstructor.** The earlier `hard` mode selected the highest-kappa distractors and balanced examples only
> against the target. That can deliberately construct an observationally unidentified MCQ: two concepts can
> be different but nearly collinear on a random sample. Low selection accuracy would then measure a poor
> teaching set, not poor articulation. The bound-grade successor uses behavior in a separate design stage.

Let option `j` have design-split hard verdict `m_j(x)`. For target `t`, first exclude empirical clones and
options with fewer than a declared number of target disagreements. Among the remaining related near-misses,
choose a teaching set by the exact max-min design

```
S_t* = argmax_{|S|=n} min_{j != t} sum_{x in S} 1[m_t(x) != m_j(x)],
```

subject to showing both target labels. The implementation solves this integer program exactly, then
lexicographically maximizes target-label balance, total separation, and executor confidence. Thus, for
`strong female lead` versus `more female characters`, the shown set should contain the available cases that
separate those concepts rather than merely sampling their common positive region. If no such cases exist in
the declared design population, the instrument fails closed or treats the options as an empirical
equivalence class; it does not report the target as inarticulable.

This defines an **active-teaching reconstruction estimand**. A passive-iid example experiment answers a
different question and must be reported under a different name. The active protocol is:

1. On a calibration/design split, score the target and candidate options, select non-clone distractors, and
   select contrastive demonstrations. These behavioral vectors are never placed in the MCQ prompt.
2. Freeze option descriptions, canonical-body hashes, demonstrations, and a counterbalanced option-order
   schedule. The reconstructor sees only target `(text, score)` demonstrations plus the option descriptions.
3. Read the normalized target-option probability from choice logits when the backend exposes them; otherwise
   estimate it with counterbalanced stochastic choices. Invalid choices count as failures rather than being
   dropped.
4. Repeat the identical option orders with no demonstrations and with permuted demonstration labels. Report
   the target probability and its lift over both controls; chance correction alone does not remove semantic
   or position priors.
5. Only after choices are frozen, optionally execute the selected canonical body on an untouched item
   lockbox. `I(M_t; M_selected)` is a secondary behavioral-equivalence readout, not the MCQ selection rule.

**Which quantity is MI.** For one fixed target `t`, target-option probability/accuracy is a conditional
recovery score; target identity has zero entropy, so calling that number mutual information is a category
error. Let `J` instead be an equal-weight randomized target metric across a declared panel, and let `Jhat` be
the canonical option selected by the reconstructor. The panel-level identity measure is `I(J;Jhat)`, computed
from the complete stored choice-probability channel. Report the annotation, no-demonstration, and
shuffled-label channels separately. This identity MI uses no anchor or silver labels: `J` is known by
experimental construction. The per-metric target probability remains the primary metric-level readout.

The implementation is `recon_channel.run_metric(mode="mcq", distractor="contrastive")`; raw split IDs,
teaching-set diagnostics, option/body hashes, permutations, condition probabilities, choices, and held-out
behavioral vectors are persisted. Prompt optimization must happen before this measurement or in an outer
training loop; a design/lockbox used to select prompt revisions is consumed and cannot serve as its final
confirmation. These are validity improvements to an anchor-free measurement, not ingredients of an upper
bound; prompt bounds remain separate objects.

### 11.2 Three nested ceilings (be precise about which gap is which)

| ceiling | value | what limits it | raise it by |
|---|---|---|---|
| `cap_f` | `1−1/min(N,K)` (binary ½) | readout granularity | K-ary scale |
| `T(m_ω)` | fixed-target DPI upper bound (≤ `cap_f`) | **this executor + target channel** | stronger/less noisy target channel |
| `R` | achieved recovery (≤ `T`) | reconstructor articulability (+executor) | better reconstructor, both-readouts |
| `I(M_i*; X)` (the *ideal* intent) | ≥ `T(m_ω)` — so `T` is the **floor** on the ideal, not its ceiling | how faithfully `m_ω` reflects the ideal | **§12.4** unsupervised consensus (NO `Y` exists), up to shared-proposer-bias |

`R ≤ T(m_ω) ≤ cap_f`, and `T(m_ω) ≤ I(M_i*;X)` — `T(m_ω)` is at once the executor's ceiling on recovering the
OPERATIONALIZED `m_ω` **and** the FLOOR on the *ideal* `M_i*`. The ideal's upper bound is NOT `T(m_ω)`; there
is **no ground-truth `Y`**, so it is approached *unsupervised* via the §12.4 consensus-stabilization
(assumption-bounded, up to shared-proposer-bias), never via a label. A found `R/T(m_omega)` is the fraction
of the DPI upper bound reached, not necessarily the fraction of the true prompt optimum. `headroom = T-R`
upper-bounds the fixed-target prompt gap; `missing_impact` is within-pool; the ideal-beyond-Omega gap is
§12.4. Do not blur them.

> **Correction (2026-07-12; historical ladder retained).** The cross-target inequality
> `T(m_ω) ≤ I(M_i*;X)` is not automatic. An operationalization can omit intended distinctions, add
> executor-specific artifacts, or both. The only automatic chain is target-indexed:
> `R_E(p;M) ≤ A_E(M;P) ≤ I(M;X) ≤ H(M)`. Instantiate it separately for `M_b` and, after identification,
> `M*`; do not order the two ladders without an explicit measurement relation. Consequently `H(M_b)` and
> `T(M_b)` are neither floors nor ceilings for the ideal.

### 11.3 Can we *estimate* `|B_E|`? Capture–recapture, Chao, and the discovery curve — and why it isn't the bound

*The "establish the whole prompt space" question, met head-on. There is a genuine population-estimation
flavor to sampling `B_E`, but it estimates **coverage/saturation**, not **impact** — and recovery doesn't
need it at all.*

**What holds.** `B_E` is a countable subset of the behavior cube `C`. Sampling it under *independent*
generators and counting **behavioral collisions** is a capture–recapture procedure: high recapture (collision)
rate ⇒ the behavior space is nearly saturated; many singletons ⇒ much is unseen. Concretely:
- **"Species" = behaviorally-distinct unit** (the §11.0 *behavioral* notion, NOT the syntactic class), defined
  by the §6.5 orthogonalization filter: two prompts collide iff their per-item signals `X_p` are equal, or one
  is conditionally redundant given the rest (Shannon-CMI, `orthogonalize.orthogonalization_filter`).
- **Independent samplers of `B_E`:** GEPA lineage diffs (§6.5 a), failure-informed free-gen (d),
  prose-decomposition (b), and — for the independence the estimator needs — *different proposer model
  families* (GLM vs Sonnet vs …). Same-family samplers share blind spots and violate the assumption.
- **Estimators:** **Chao1** `|B_E|_eff ≈ S_obs + f₁²/(2f₂)` (richness from singletons `f₁`/doubletons `f₂`);
  the **discovery/Heaps curve** of `S_obs` vs cumulative samples and its asymptote (§6.9); **Good–Turing
  coverage** `C = 1 − f₁/n`. All measure *how much of `B_E`-as-your-generators-see-it* has been seen.

**What this buys, and the wall.**
- **Buys: a saturation/coverage statement** — "are we still discovering new behaviors, or have we plateaued?"
  Evidence for the §6.9 halt condition; pairs with the orthogonalization filter (which *is* the collision
  detector).
- **Wall 1 — mass ≠ impact (§6.7c, restated).** These estimators bound the *mass* of unseen behaviors, **not**
  their *impact* on `R`. A rare magic word has low mass but disproportionate impact — exactly what
  missing-mass cannot catch. **Impact must be bounded separately:** the submodular tail-bound (§6.7c leg 1,
  conditional on tail-`γ≈1`) and the assumption-free `adversarial_saturation` probe `I(X_probe; M | X_Ω) ≈ 0`
  (leg 2). Coverage feeds the *saturation* halt; the tail-bound + probe do the *certifying*.
- **Wall 2 — generator-specificity (§6.9).** The discovery curve is a property of *your generators*; LLM
  generators share gaps, so saturation under one family ≠ saturation of `B_E`. Only cross-family generators
  make the recapture rate meaningful.
- **Wall 3 — continuity.** Behaviors are real-valued `P(YES)` vectors, so "distinct species" needs a
  behavioral-distance threshold; the richness estimate is sensitive to it.

**Why recovery makes this moot.** All of §11.3 addresses the within-class worry "could a prompt outside `Ω`
beat `OPT_Ω`?" For **recovery**, that worry is already closed by `T(m_ω)` (§11.1): no prompt anywhere exceeds
`T(m_ω)`, in-pool or out. So estimating `|B_E|` is **unnecessary for the recovery certificate** — `T(m_ω)`
bounds all of `B_E` without enumerating or estimating it. Capture–recapture remains the right tool only if one
returns to the within-class `OPT_Ω` question, where it supplies the *saturation* half of the §6.9 halt.

> **Correction (2026-07-12; historical discussion retained).** The submodular/adversarial pair above does
> not certify unseen prompts, and capture is not moot if the goal is to tighten the universal DPI cap over a
> declared search process. §12.6b retains capture in two valid scopes: direct all-draw gain marks bound a
> predeclared finite mining horizon; exact-pattern missing mass plus external `p_min` can certify proposer-
> support exhaustion. Fuzzy missing mass alone still does not imply impact.

### 11.3a Multi-source capture–recapture — the four independence axes and what they buy **[new, 2026-06-25]**

*What assumptions a population-style `missing-impact` bound needs, and what it can actually conclude. The four
axes below — (1) the prompt space, (2) strong-LM independence, (3) data-slice independence, (4)
Ω-discovery-algorithm independence — are not separate nuisances: they are the **lists of a multiple-recapture
design**. Each is an independent mechanism that "captures" a subset of `B_E`; the unseen cell ("captured by no
source") = the missing impact, inferred from cross-list overlap (census-undercount / Fienberg machinery).*

**Three jobs, three *different* assumptions — do not merge them.** Capture–recapture natively bounds *mass*;
the value-weighting is a separate assumption; DPI (`R ≤ T`) caps whatever both miss.

| job | tool | assumption it needs |
|---|---|---|
| bound missing **mass** of `supp(π)` | Good–Turing `f₁/n` | closed pop., **stationary** π, stable equivalence |
| convert mass → **impact** | smoothness / EVT | value-Lipschitz in a behavior metric, or near-optimal mass `β(ε) > 0` |
| extend support past one π | multi-list overlap | independence (or *fitted* interaction) across sources |

**Per-axis.**

| axis | role | assumption | realistic violation (direction) | testable? |
|---|---|---|---|---|
| prompt space `≤ V^L` | population size | irrelevant raw; needs **finite effective class**: discovery exponent `α<1` on probe set | `α→1` ⇒ inexhaustible ⇒ only a *lower* bound on richness, never an upper bound on impact | yes — fit `S_obs(n)` (§6.9) |
| strong LMs | capture lists; also consensus denoisers | `k`-way capture interaction `≈ 0` (NOT full independence) | shared pretraining ⇒ **positive** correlation ⇒ overlap inflated ⇒ **underestimate** missing ⇒ **false saturation (anti-conservative)** | partially — pairwise corr. on *seen* behaviors |
| data slices | items defining behavior; iid trials for `R,T` | items iid from deployment dist; probes **disjoint from GEPA-train**; probes **span** input variation | clustered items ⇒ effective `n` collapses; overfit probes ⇒ behaviors collapse ⇒ undercount (anti-conservative) | yes — cluster-robust var; held-out split |
| Ω-discovery algos | capture lists | distinct search operators ⇒ low capture corr. | same base-LM + same objective `R` ⇒ redundant lists, no coverage gain | yes — overlap of discovered sets |

**Three structural facts about the multi-list inference.**

1. **The top interaction is non-identifiable (Fienberg).** With `k` lists the data is a `2^k` table with one
   missing cell; a saturated log-linear model has `2^k` params but only `2^k − 1` observed cells, so the
   `k`-way interaction — exactly the one controlling the unseen cell — must be *assumed* (set to 0). Adding a
   source weakens this: "`(k+1)`-way `= 0`" ≺ "`k`-way `= 0`". Source *diversity* doesn't just add coverage;
   it pushes the irreducible assumption to higher order.
2. **Every realistic correlation is positive, and positive dependence is anti-conservative.** Shared biases ⇒
   inflated overlap ⇒ underestimated unseen ⇒ *false* saturation. So independence buys an **optimistic point
   estimate, not a certificate**. For a one-sided certificate either (a) test-and-pass pairwise independence on
   *seen* behaviors, or (b) take the conservative **max-dependence** correction on the unseen cell. Seen-cell
   correlations are observable, so independence is *rejectable* — but never *provable* (the unseen cell can
   always differ; the §11.4-style residual).
3. **Independence compounds coverage multiplicatively.** If behavior `b` has capture prob `p_i(b)` on source
   `i`, independent given `b`, then `Pr[b unseen] = ∏_i (1 − p_i(b))` and missing mass
   `= Σ_b μ(b) ∏_i (1 − p_i(b))` — shrinks geometrically per independent source. *That* is the leverage in
   "strong independence up and down": `L` LMs × `S` slices × `G` algos act like `L·S·G` lists. **But** the
   behaviors that matter (magic words) have `p_i → 0`, so `(1 − p_min)^{LSG} ≈ e^{−LSG·p_min}` needs
   `LSG ≳ 1/p_min → ∞`: independence covers *moderately*-rare behaviors and **nothing in the zero-mass tail**.
   DPI is the only guard there. *(Does independence EVER certify support-completeness — i.e. that the union
   support equals `B_E`? No: the gap is a shared-support ceiling, not a correlation, so no amount of LM-list
   independence crosses it; only positivity / Lipschitz-impact / a non-LM expert list can, none of them a
   certificate. Full treatment: **§12.2.4**.)*

**Other assumptions (beyond the four axes).**
- **Closed population** — freeze executor, temperature, probe set, readout during the capture phase.
- **Stationary proposal — the sleeper.** Good–Turing / Good–Toulmin assume iid draws from a *fixed* π;
  **GEPA is adaptive** (rich-get-richer), which breaks exchangeability. Run capture–recapture from a **frozen
  checkpoint**, or use adaptive-data missing-mass (weaker). Easy to miss; invalidates the estimator silently.
- **Stable behavioral equivalence** — enough readout samples per prompt that clustering at tolerance `τ` is
  reproducible (else readout noise overcounts richness — safe direction, just loose).
- **EVT domain-of-attraction** — value-tail extrapolation assumes unseen high-value behaviors share the
  observed tail's mechanism; a *different kind* of magic word breaks it, untestable from below threshold.
- **`T̂` validity** — `T̂` is a sound ceiling only if the posterior estimator computing `H(M_ω|X)` is **≥ as
  strong as the reconstructor**; a reconstructor stronger than the `T`-estimator can *violate* `R ≤ T̂`.
  Estimate `T` with the strongest available model, not the executor.

**What you can conclude — a ladder of conditional global claims** (each rung a real all-`Σ*` statement; each
costs strictly more assumption):

| rung | assumptions added | global claim |
|---|---|---|
| 1 | DPI + valid chain only | `OPT_global ≤ T` (behavioral, this executor, this dist.) — *already shipped* |
| 2 | + closed + stationary + stable equiv. | `ε`-optimal **relative to π** (mass coverage of `supp π`) |
| 3 | + smoothness (value-Lipschitz / `β(ε) > 0`) | `OPT_{supp π} ≤ OPT_Ω + Δ̂_π` (mass→impact) |
| 4 | + `k` sources, `k`-way interaction `≈ 0` | extend to `∪_i supp(π_i)`; `Δ̂` shrinks geometrically in independent-source count (non-tail) |
| 5 | + consensus target over independent LMs | escape single-executor; claim over a model *family*; `T` rises toward `I(intersubjective intent; X)` |

Two hard stops no independence can cross (= §11.4): behavioral→intent needs a label `Y`; sub-smoothness
zero-mass spikes are guarded only by `R ≤ T`. Independence converts the *loose* ceiling `T` into a *tight*
estimate `OPT_Ω + Δ̂` — a real upgrade — but cannot make the certificate about *intent* vs *behavior*, or about
*all models* vs the family you ran.

**Refinement to §11.3's "recovery makes this moot."** Recovery moots the *target-pool* `|B_E|` (MCQ target
in-pool; `T` caps all). But the headroom splits
`T − R(p̂) = [T − sup_p R(p)]_realizability + [sup_p R(p) − R(p̂)]_{search over reconstructor prompts}`, and the
second term has the *same* magic-words structure over **reconstructor** prompts ("is there an undiscovered
reconstructor prompt that elicits more of the posterior?"). Capture–recapture, with the assumptions above, is
the tool that certifies *that* search is done — it relocated from target-pool to reconstructor-prompt, it did
not vanish. Bites exactly in the current `R/T ≈ 0.4` regime: it says whether the headroom is realizability
(→ both-readouts, bigger executor) or unfound reconstructor prompts (→ keep searching).

> **Correction (2026-07-12; historical ladder retained).** The `T̂` direction cannot be repaired by using a
> stronger achieved estimator: an upper bound needs a proved one-sided procedure. Nor do source independence,
> fitted smoothness, or an in-pool tail statistic establish the displayed global rungs without their full
> identifying assumptions. The current bound-grade ladder is: target-indexed DPI; fresh-audit finite-horizon
> gain; exact support exhaustion only with external `p_min`; and a separate identified-target analysis for
> `M*`. See §12.6b.

**Design corollary.** Don't chase many *independent* discovery algorithms (they correlate through the shared
objective `R`). Run **one adversarial, novelty-tilted proposer** whose objective is to find behaviors *unlike*
`Ω` — a deliberately opposite-tilted second list brackets the missing cell far better than `k` near-duplicate
lists and directly lowers the unseen-cell variance. Highest-value single addition to the current setup (= the
`adversarial_saturation` probe, repurposed as a capture list).

### 11.4 The two honest residuals (what global recovery does *not* certify)

1. **Behavioral optimality ≠ intent-optimality** (§10 item 7, §4.3). `R ≈ T(m_ω)` means "p̂ recovers all of
   `X`'s *behavior* on the metric," not "p̂ is *correct* for what you meant." The gap `I(intent; X) − T(m_ω)`
   is unmeasurable without a label `Y`. The one place "ε-optimal for the original metric question" escapes us
   label-free.
2. **Single-executor scope** (§10 item 5). The cert is optimal *for executor X*; `T(m_ω)` and the optimum are
   executor-specific. Cross-model consensus `M` or an `E`-family average objective `R_avg = (1/m)Σ_i R(·;E_i)`
   (submodular; brute-force transfers) broadens scope.

### 11.5 Scorecard additions

| # | claim | status | measure |
|---|---|---|---|
| **R5** | for fixed-target recovery, `OPT_prompt ≤ T(m_ω)` over all `Σ*`; equality requires a sufficient readout to be realizable by the prompt interface. `T−R` upper-bounds the found prompt's true suboptimality; it need not equal it. | **holds as upper bound; prompt tightness unproved (corrected 2026-07-10)** | same-heldout Shannon/TVD `R`, `T_target`, `dpi_ok`, `target_headroom_upper` |
| **R6** | `R ≤ T(m_ω) ≤ cap_f`; `headroom` is an upper bound on the fixed-target prompt gap, while `missing_impact` is a separate within-pool diagnostic. Neither quantity identifies articulability or the ideal-intent gap. | **holds with corrected interpretation** | per-metric `R`, `T`, `cap_f`, `missing_impact` |
| **R7** | capture–recapture/Chao/Good–Turing estimate `B_E` **coverage/saturation** (behavioral species via orthogonalization filter; needs cross-family generators), NOT **impact**; impact bounded by submodular tail-bound (cond. `γ`) + `adversarial_saturation`; **recovery does not need `|B_E|`** (`T(m_ω)` bounds all of `B_E`) | **historical claim; corrected 2026-07-12** | Chao1/discovery-curve on multi-generator samples; `submodular_tail_bound`; `adversarial_saturation` |
| **R7′** | current result: historical count curves are descriptive; fresh per-family all-draw gains certify a fixed future horizon, and exact-pattern mass plus external `p_min` can certify proposer-support exhaustion | **holds in §12.6b scope** | `prompt_articulation_certificate`; immutable confirmation audit |
| **R8** | the soft posterior `η(X)=E[M_ω|X]` attains Shannon `T` in the unrestricted measurable-readout space. This proves readout-space tightness, not prompt-space tightness; continuous readouts are necessary but not sufficient for prompt realizability. | **readout theorem holds; prompt conclusion withdrawn (2026-07-10)** | report soft vs sampled readout and an explicit realizability assumption |
| **R9** | the four axes (prompt space, LMs, slices, Ω-algos) can be analyzed as recapture lists only under a frozen sampling design. Positive dependence and zero-mass support make saturation anti-conservative; the resulting estimates are descriptive and do not tighten `T` into `OPT_Ω+Δ̂`. | **historical/descriptive** | pairwise seen-behavior capture corr.; Chao/Good–Turing on frozen multi-family samples; `Δ̂_π`; discovery exponent `α` |
| **R9′** | current design treats proposer families as strata, not independent capture lists; each receives its own exact/empirical-process bound, and zero-mass support remains unidentifiable without positivity | **holds in CR-3 design** | per-family CP/EB/DKW; external `p_min` only |

### 11.6 The exact decoder-capacity ceiling and its non-certifying spectral approximation. **[corrected 2026-07-10]**

*An exact but generally intractable upper-bound object on `R* = OPT`, obtained by working backwards from the
recovery objective. The finite spectral search proposed below does not compute that maximum.*

**Construction.** A prompt induces a labeling `s_p ∈ B_E ⊆ {0,1}^N`, but recovery `R(m) = I(m; m̂)` is defined
for ANY labeling `m` (`m̂` = a strong decoder's reconstruction of `m` from held-out `X`). Maximize over all
labelings:
```
OPT′  :=  max_{m ∈ {0,1}^N}  R(m)                 (the most-decodable labeling)
B_E ⊆ {0,1}^N   ⇒   R* = max_{s∈B_E} R(s)  ≤  OPT′  ≤  cap_f
```
So `OPT′` is a **prompt-free** upper bound on the optimal recovery, between `R*` and the info cap.

**What it IS — the strong decoder's *capacity*, NOT `I(X;M*)` (do not over-read).** Maximizing recoverability
finds the *easiest-to-decode* function of `X` — usually a trivial surface feature (length, topic, language) —
which is **metric-agnostic**: the exact maximum over-bounds *every* metric at once. Exact `OPT′` is a
**conservative (safe, loose) sanity ceiling** — "no metric beats this on this decoder" — not the metric-specific bound, and emphatically not
the ideal `M*` (meaning-defined, §11.2, unreachable: maximizing `R` finds decodability, never meaning).

**Computation caveat (corrected 2026-07-10).** The exact maximum over all `2^N` labelings is an upper bound,
but it is intractable. Searching a decoder's putative low-dimensional learnable subspace evaluates only a
subset of labelings:
```
1. features    φ(X) ∈ R^{N×d}            the decoder's representation of the items (frozen strong embedding)
2. top-k dirs  PCA / kernel-PCA on φ(X)   the most-learnable directions (decodable labelings ⊂ span φ)
3. candidates  m_j = threshold(dir j) ;   evaluate R(m_j) on the recovery loop  (k cheap evals)
4. L_spectral := max_j R(m_j)              achieved lower bound on OPT′
```
(A relaxed search over `m∈[0,1]^N` has the same direction: without a proof that the search region contains
the global maximizer and a certified optimizer upper bound, it does not compute a ceiling.)

**No-free-lunch caveat.** A decoder's ability to recover a labeling does not imply that a prompt can induce
that labeling, even when the decoder and executor are from the same model family. Claims that the two spaces
coincide require a separate expressivity theorem. A stronger probe may raise achieved spectral values, but that
still supplies lower-bound examples, not a certified upper bound on the exact `OPT′`.

**Distinct from `c_∞` (§12.4) — same spectral machinery, different input.** `OPT′` = top direction of *raw
learnability* (metric-agnostic, often trivial). `c_∞` = top direction of the *criteria's agreement*
(meaning-relevant, the process-reachable ideal estimate). They can point in completely different directions:
`OPT′` is a conservative ceiling on `R*`; `c_∞` is an estimate of the ideal.

**Where it sits.** Mathematically, the exact `R* ≤ OPT′ ≤ cap_f` relation is valid. Operationally, the shipped
spectral search computes `L_spectral ≤ OPT′`, so it must not be called a ceiling or used to form certified
headroom. The operative fixed-target upper-gap certificate remains `T(m_omega)−R` (§11.1).

## 12. Algorithm `B_E-ATLAS` — approximating the reachable behavior space **[new, 2026-06-25]**

*The constructive payoff of §11.3a. What an algorithm to "approximate `B_E`" should be — and the regime split
that decides whether you build it at all.*

### 12.0 The regime split — for *recovery* you do **not** approximate `B_E`

The posterior `η(X)` characterizes the unrestricted readout optimum. For any deterministic readout
`φ(X)`, the Shannon chain rule gives

> **Headroom = sufficiency deficit.** `T − R(φ) = I(M_ω; X) − I(M_ω; φ(X)) = I(M_ω; X | φ(X)) ≥ 0`, with
> equality (`R = T`) **iff** `φ(X)` is sufficient for `M_ω`, i.e. `φ` determines the posterior `η(X)`.

This does not make the value surface on the prompt behavior set `B_E` known: `η` may lie outside that set, and
the restriction of the objective to `B_E` can still have hidden peaks. For fixed-target certification one need
not census `B_E` to obtain the valid upper bound `T`; one still must search for high-`R` prompts, and only a
small `T-R` certifies near-optimality. Estimating `η` is useful for diagnosing the sufficient readout, not for
proving that any prompt can elicit it.

**Build the atlas only in the unknown-target / within-class regime (§6)** — "map the full space of behaviors a
community's metrics could induce," where there is *no* single `η` to project onto and value = coverage/lift.
There the full machinery below earns its keep.

### 12.1 The algorithm (frozen · multi-list · value-annotated · active)

Represent a behavior as its soft signature `σ(p) = (P(YES|X_1), …, P(YES|X_n)) ∈ [0,1]^n` on a frozen probe
set. `B_E` (on the probe set) is a finite subset of `[0,1]^n`; the atlas samples its reachable region and
annotates observed cells. Recapture statistics provide stopping diagnostics, not a support-completeness proof.

| stage | operation | assumption it discharges |
|---|---|---|
| **0 · Freeze** | fix executor, temperature, probe set (iid from deployment dist, **disjoint from any GEPA-train**), soft readout; snapshot proposers | closed population + stationary π |
| **1 · Capture** | each of `K` diverse proposers (different model families + **1 adversarial novelty-tilted**) draws prompts; compute `σ(p)` (few samples to denoise); record which list(s) captured it | multi-list design |
| **2 · Collide** | dedup signatures via the existing `orthogonalization_filter` (Shannon-CMI: collide iff per-item signal equal within `τ`, or conditionally redundant given the cluster) → behavior-species + the `K`-way capture contingency table | stable behavioral equivalence |
| **3 · Coverage** | discovery curve `S_obs(m)` → Heaps exponent **`α`** (`S ~ K·m^α`); Good–Turing `M_0 = f_1/N`; Chao1/ACE richness floor; Fienberg log-linear up to `(K−1)`-way **with the conservative max-dependence interval**; Good–Toulmin extrapolation to `c·N` | missing **mass**; anti-conservative correction |
| **4 · Value** | annotate each species with value `V` (recovery: sufficiency deficit; else predictive lift / CE); value-restricted missing mass `f_1^(ε)` over species with `V ≥ OPT_Ω − ε`; (no single target ⇒ GPD endpoint on the value tail) | mass → **impact** |
| **5 · Active bracket** | steer the adversarial proposer toward species *unlike* `Ω` **and** near the value frontier; re-enter Stage 1 as a new list (importance-sample the unseen cell) | lower unseen-cell variance |
| **6 · Diagnose/stop** | halt operationally when the coverage diagnostics stabilize and the adversarial list is dry for a fixed budget; report the named process and resource horizon. Separately report the fixed-target DPI cap `T`. | heuristic stopping rule, not a coverage certificate |

**Output:** the atlas (species + signatures + values + capture table), the coverage **interval**, the
missing-impact bound, and `α`.

### 12.1a `ALPHA-PROBE` — run-first instance + implementation status (hand-off spec). **[new, 2026-06-25]**

*Run **once per metric `M_i`** (per R2 cluster — see §1): the minimal front half of §12.1 — freeze →
breadth-sample (criteria for `M_i`'s OWN `Ω_i`, proposer seeded by the cluster's description) → collide →
estimate `α_i` + coverage, to decide **GO** (`M_i` has low-dim `Ω_i`, run full ATLAS) / **NO-GO** (`M_i`'s
behavior space is inexhaustible, only `T_i` bounds it). One `α_i`, one decision, per metric. **Task-level `α`
(pooling all metrics' criteria) is always ≈0.9 → non-discriminating — that is the WRONG level** (the
2026-06-26 task-level sweep). Written as an implementation hand-off — the math, what to reuse (file:line,
audited 2026-06-25), what to build. Value/active/atlas stages (§12.1 stages 4–6) are out of scope for the probe.*

> **IMPLEMENTED 2026-06-25.** Species-accounting layer B1–B7 live in
> `methods/metric_implementer/experiments/alpha_probe.py`; the driver (K-family breadth proposers, frozen
> vLLM executor, GEPA-disjoint probe set, decision print) in `experiments/run_alpha_probe.py`; planted
> ground-truth tests in `tests/test_alpha_probe.py` (11 tests, all green — low-dim→GO/AMBIGUOUS,
> high-dim→NO-GO). Verified reuse map (symbols under `experiments/`): `_pyes` recon_channel.py:98,
> `_free_generate` run_real_test.py:150, `orthogonalization_filter`/`shannon_cmi_surrogate`
> orthogonalize.py:91/63, `make_judge_backend` vllm_backend.py:273. Dry-run (FakeVLLM + mock proposers)
> runs end-to-end. **One honest deviation from §C below:** the multi-list coverage lower bound is the
> **assumption-free Berend–Kontorovich** `C_lo = 1 − (f_1/N + √(log(1/δ)/N))` (valid under ANY family
> dependence — cannot be invalidated by shared-LM-bias correlation) plus a pairwise-Petersen sensitivity
> point, NOT the Fienberg log-linear + Fréchet/LP max-positive-dependence bound. Reason: the all-zero cell
> is not point-identified without an external interaction-strength bound (max positive interaction inflates
> it without limit), so the assumption-free bound is the safe certificate and the log-linear value is a
> reported sensitivity, not the headline. **UPDATE 2026-06-25:** the Fienberg multi-list log-linear POINT
> estimators ARE now implemented (`coverage_fienberg`: Poisson-GLM IRLS on the 2^K−1 observed cells, empty
> cell = exp(β_0); independence = K-list Petersen-generalization, pairwise = dependence-corrected,
> identified only for K≥4). The §C Fréchet/LP **max-positive-dependence** bound is provably **unbounded**
> (n_empty→∞ as the interaction→∞ — no finite LP lower bound exists), which is exactly why the
> assumption-free `C_lo` remains the only VALID-under-any-dependence certificate; the Fienberg values are a
> dependence-sensitivity ladder above it.
> **Cost note:** assumption-free `C_lo` needs N≈1200 draws to clear 0.95 (it subtracts the concentration
> term even at f_1=0); a smaller `--M` lands honestly in AMBIGUOUS (scale M, re-run).
> **Constant verified (2026-06-25, Opus):** the `√(log(1/δ)/N)` half-width is the EXACT Berend–Kontorovich
> upper-tail constant — Thm 1 of *On the Concentration of the Missing Mass* (ECP 2013, arXiv:1210.3248)
> gives `P(M_0 > E[M_0] + ε) ≤ e^{−N·ε²}` (variance proxy `1/N`, `c=1`), so `ε = √(log(1/δ)/N)` is tight and
> N≈1200 for 0.95 stands. Two riders: **(i)** Thm 1 concentrates `M_0` around its MEAN while the code plugs
> in the observed `f_1/N`; the Good–Turing bias `|E[M_0]−E[f_1/N]| ≤ 1/N` is unaccounted — add `+1/N` to
> `C_lo` for full rigor (≈0.0008 at N=1200; numerically negligible, but makes it provably one-sided
> conservative). **(ii)** It is one-sided (upper tail of `M_0` = lower bound on coverage), exactly our
> direction; the lower-tail constant is looser (`c≈1.92`) but irrelevant.
> **Novelty fallback wired (2026-06-25):** when the base K-family pass returns AMBIGUOUS, the driver
> auto-adds ONE tail-tilted novelty list (`_novelty_generate`: fixed, iid, `existing=[]`, temp 0.9, biased
> to rarely-stated criteria) and re-decides on the 5-list sample BEFORE suggesting "crank M" — the R9
> corollary (one novelty-tilted list ≫ k correlated ones), and the tilt gives a different support that
> breaks some shared-LM positive dependence. `--no-novelty-fallback` disables it. The adaptive
> residual-hunting proposer (conditioned on found species) is a STRONGER but non-iid follow-on and is
> deliberately NOT wired into the capture accounting.
> **Report the floor↔point GAP, not just the floor:** `C_lo` (assumption-free certificate) and the Fienberg
> independence/pairwise POINTS print side by side; the gap `C_Fienberg − C_lo` is itself the quantity of
> interest — the *price of the independence assumption* (how much coverage you'd claim on faith). Never
> headline a Fienberg point; headline `C_lo`, show the gap.

> **Correction / current status (2026-07-12; historical implementation record retained).** `alpha_probe.py` still computes
> descriptive richness, rarefaction, Chao/Fienberg sensitivities, and singleton diagnostics. The old
> `C_lo` claim was over-scoped: missing-mass concentration requires iid draws from one fixed distribution,
> and substituting observed Good-Turing flux for its expectation requires its own bias term. A stratified
> multi-family stream is handled family by family, not declared valid under arbitrary dependence. No
> ALPHA-PROBE decision is a prompt ceiling. Bound-grade capture now uses a frozen pool, independent
> per-family audit draws, exact Clopper-Pearson components, and direct all-draw gain marks (§12.6b).

#### A. Objects and estimators (ASCII, implementable as written)

```
probe set     X = {X_1..X_n}   frozen, iid from deployment dist, DISJOINT from GEPA-train; n ~ 300
signature     sigma(p) = ( P(YES|X_1), ..., P(YES|X_n) ) in [0,1]^n           (SOFT readout, not sampled bit)
noise floor   tau0 = c * mean_i std_r[ P(YES|X_i) over r rescorings of the SAME p ]      (c ~ 2..3)
species       p ~ p'  iff  (1/n) sum_i |sigma(p)_i - sigma(p')_i| <= tau      (single-linkage)
              [alt / cross-check: orthogonalization_filter CMI test, cmi_thresh as the tau-analog]

counts        N = total breadth prompts ;  D = # distinct species
spectrum      f_j = #{ species seen exactly j times }      (j = 1,2,3,...)
K-way table   capture[species] = subset of families {1..K} that produced it

coverage pt   C_hat  = 1 - f_1 / N                               Good-Turing (independence-OPTIMISTIC)
missing mass  M0_hat = f_1 / N   +/-  sqrt( log(1/delta) / N )   (Berend-Kontorovich concentration)
richness floor Chao1 = D + f_1^2 / (2 f_2)        (f_2>0 ; else D + f_1 (f_1 - 1) / 2)
rarefaction   E[S(m)] = D - sum_j f_j * C(N-j, m) / C(N, m)      m = 1..N   (DE-NOISED curve)
Heaps / dim   alpha(m) = d log E[S(m)] / d log m       report the TRAJECTORY, not a single number
diversity gap Delta_div = D_pooled|_M  -  max_k D_family_k|_M    (rarefy pooled DOWN to M for fairness)
coverage cert C_lo  = 1 - (max-positive-dependence unseen-cell bound from the K-way table)   <-- CERTIFICATE
```

#### B. Reuse map — DO NOT rebuild (shipped, verified)

| primitive | reuse | file:line |
|---|---|---|
| soft P(YES) readout | `recon_channel._pyes` | recon_channel.py:98 |
| `R = I(M_ω;M′)`, `T = I(M_ω;X)` + bootstrap CI | `vinfo.iv_transmission`; `recon_channel.run_metric` | vinfo.py:573; recon_channel.py:575 |
| value of a behavior = `T − R(σ)` (sufficiency deficit) | `iv_transmission` (different inputs) | vinfo.py:573 |
| CMI collision / behavioral equivalence | `orthogonalize.orthogonalization_filter`, `shannon_cmi_surrogate` | orthogonalize.py:91, 63 |
| impact cert (full ATLAS, not the probe) | `submodular_tail_bound`, `adversarial_saturation`, `OmegaCertificate`, `_missing_impact` | orthogonalize.py:144, 219; omega_certificate.py:96; run_real_test.py:251 |
| GEPA depth stream | `recon_channel.induce_gepa`, `m_omega_gepa.py` | recon_channel.py:305 |
| multi-family backends (Llama/Qwen/GLM/Sonnet) | `backends.py`, `config.py`, `vllm_backend.py` | — |
| (M,X) signal-matrix scaffold | `real_gamma.py` (saves .npz) | real_gamma.py:9 |

#### C. Build list (MISSING / PARTIAL — concrete specs)

```
B1  signature(prompt, probe_set, backend) -> np.ndarray[n]              [PARTIAL: thin wrap of _pyes]
B2  breadth_sample(families, M, probe_set) -> [(prompt, family, sig)]   [MISSING: PLAIN sampling, NO gepa loop]
B3  noise_floor_tau(prompt, probe_set, backend, reps) -> float          [MISSING: rescore-based tau0]
B4  collide(sigs, tau) -> species_labels                                [PARTIAL: single-linkage; or reuse CMI]
B5  spectrum(species_labels, family_tags) -> ({f_j}, K_way_table)       [MISSING]
B6  estimators: good_turing, chao1, rarefaction, heaps_alpha,
       diversity_gap, coverage_interval_conservative                    [MISSING ~150 LOC]
B7  alpha_probe(...) orchestrator + decision rule + report              [MISSING]
```

`coverage_interval_conservative` (in B6): fit a Fienberg log-linear model on the K-way `capture` table up to
`(K−1)`-way interactions; `C_hat` uses the no-`K`-way-interaction (independence) fit; **`C_lo` takes the
unseen-cell value under the maximum positive interaction consistent with the observed margins** (Fréchet
bound / LP for general K; for K=2 it is the pairwise-odds-ratio-inflated `n_00`). `C_lo` is the reported
certificate; `C_hat` is for intuition only.

#### D. The flaw NOT to inherit

`discovery_scaling.py:8,99` already plots a discovery curve — but it accumulates distinct criteria **in
GEPA-version order**, i.e. on the *adaptive depth stream*. GEPA front-loads easy/high-value criteria, so that
curve saturates faster than iid and yields **`α` too small → false "coverable"** (anti-conservative, the
dangerous direction). `B6.heaps_alpha` MUST consume the **frozen breadth sample (B2)** with **rarefaction
averaging (the `E[S(m)]` formula above)**, never the GEPA trajectory. Treat `discovery_scaling.py`'s `α` as
invalid for this purpose; it can stay as a *depth-stream* exploration diagnostic but must not feed the
coverage certificate.

#### E. Reliability guards (acceptance criteria — each is a real failure mode already seen on this project)

```
G1  breadth stream is iid from a FROZEN proposal (no GEPA, no reflection)  -- else B6 statistically invalid
G2  tau tied to noise floor (B3); report alpha(tau) over a small sweep     -- species count is tau-sensitive
G3  recovery on vLLM SOFT readout (_pyes), NOT API _sampled_binary         -- else R<T is hard-round artifact (R8)
G4  T estimated with a posterior model >= reconstructor in strength        -- else R<=T can be VIOLATED
G5  iv_transmission bootstrap-CI spread is checked, not collapsed          -- MI under-bias / judge-collapse
G6  alpha stabilizes as n (probe count) grows                             -- too-few probes => false coverable
```

#### F. Output + decision rule *(GO/NO-GO superseded by §12.6.6 — these count-axis gates remain the behavior-census/vocabulary read, not the depth verdict; see Lemma 12.6.0 for why `α≈1` under singleton-dominated spectra is mechanical)*

```
report = { alpha_trajectory, alpha(tau)-sweep, [C_lo, C_hat], Chao1, D, Delta_div, per-family curves }

GO     (run full §12 ATLAS; trust saturation-based stopping):
         terminal slope alpha(N) < ~0.3  AND  C_lo >= 1-eps  AND  D/Chao1 -> 1  AND  Delta_div closing
NO-GO  (B_E inexhaustible under these proposers; `T` is the ONLY global statement):
         alpha(N) ~ 1 (not decreasing)  OR  Delta_div large and still growing
AMBIGUOUS: add families / probes, re-run

Wiring note: the breadth families (B2) and GEPA (depth) share backends but run on SEPARATE paths. GEPA output
may be appended to the eventual atlas WITH its values, but is excluded from {f_j}, alpha, C_lo, Delta_div.
```

**Net for the implementer:** the expensive, validated machinery (scoring, `R`, `T`, the CMI collision test,
the impact certificate, multi-family backends) is shipped and reused as-is via §B. The build is the
species-accounting layer §C (≈ a few days, no new model runs, no new research) plus the discipline of §D/§E —
run it on a *frozen iid breadth stream*, not GEPA's trajectory.

### 12.2 Four non-obvious consequences

1. **`α_i` is an intrinsic-dimension estimate of metric `M_i`'s behavior manifold (PER-METRIC).** The Heaps
   exponent measures how fast new behaviors appear in `M_i`'s OWN criterion space `Ω_i` — not the task's. The
   metric-decomposition hypothesis predicts: a SIMPLE metric (few orthogonal criteria) → small `α_i < 0.5` →
   fast saturation, atlas cheap; a COMPLEX/holistic metric (many complementary criteria) → `α_i → 1` →
   inexhaustible, `T_i` is the only global statement. It is the **go/no-go to run first, per metric**.
   Task-level `α` (pooling every metric's criteria) is always ≈0.9 and tells you nothing — run it per R2
   cluster. Ties directly to the metric-tree work.
2. **Two streams, never one.** A *frozen* breadth stream (multi-family, novelty-tilted) for valid Good–Turing
   mass estimates, **plus** an *adaptive* adversarial depth stream for finding/bracketing magic words. Keep them
   separate in the accounting — the adaptive stream's draws are **not** iid and must not enter the mass
   estimator (the stationarity sleeper, §11.3a).
3. **The certificate is an interval; the point estimate is optimistic.** Positive capture correlation (shared
   LM biases) over-claims saturation. Report the max-dependence-corrected coverage lower bound; the point
   estimate is for intuition only.
4. **No independence among the K lists can certify support-completeness — only a different, un-testable
   assumption can.** [new, 2026-06-25] The capture–recapture machinery (incl. the Fienberg independence
   fit) speaks ONLY about the support of the proposal distribution = the union support of the K families.
   A criterion in NO family's support has zero capture probability under every list, so list-independence
   — which only point-identifies the all-zero cell of *capturable* species — cannot see it (the
   DPI-guarded zero-mass tail). The gap is a **shared-support ceiling, not a correlation**, so adding more
   diverse LM lists narrows it but never closes it. Three things *could* close it, each trading the
   un-verifiable support claim for a different un-verifiable one: **(a) positivity/overlap** — assume every
   articulable criterion has proposal probability ≥ `p_min>0`; then coverage-of-support = coverage-of-`B_E`
   and `C_lo` certifies the whole thing — but positivity is untestable (a zero count is consistent with
   `p=0` OR `p` arbitrarily small). **(b) Lipschitz-impact** — assume value is `L`-smooth in a criterion
   embedding; this does not recover unseen species but BOUNDS their impact by (embedding-distance to the
   covered region)·`L`, converting "unseen support" into "bounded impact of the residual" (the more useful
   target — impact, not species count, is what we care about; the job-2 mass→impact bridge of §11.3a).
   **(c) complexity bound** — assume `B_E` has finite VC / Heaps `α<1`; explicitly distrusted (non-Lipschitz
   extrapolation, §11.3a). The only thing that raises the LM-support ceiling *itself* is a sampler whose
   support is NOT a subset of the LMs' — a **non-LM (human/expert) elicitation list**; if it too surfaces
   nothing new, that is the strongest available (still non-certifying) evidence of exhaustion. The
   novelty-tilted "+1" list is the cheap in-distribution proxy; an expert arm is the principled
   ceiling-raiser. Net: report `C_lo` as "coverage WITHIN union support," and treat support-completeness as
   an *assumption* (positivity) or a *bounded-impact claim* (Lipschitz), never a certificate.

| # | claim | status | measure |
|---|---|---|---|
| **R10** | `B_E-ATLAS`: for fixed-target recovery, `T` supplies an all-prompt upper bound without a census, but `eta` need not be prompt-realizable. For unknown-target mapping, the atlas and its Heaps/coverage outputs are descriptive process diagnostics; `alpha` is not identified as manifold dimension. | **corrected scope** | `alpha`; coverage diagnostics; achieved values; fixed-target `T` separately |

### 12.3 The VALUE-CENSUS — population dynamics on *improvement*, per metric. **[new 2026-06-26; reframed metric-level + reconstruction-based 2026-06-26]**

> **Status correction (2026-07-12): descriptive only.** The species partition and values in the shipped
> census are estimated from the same sample, the greedy conditional-MI sequence is an achieved diagnostic,
> and mutual information is not submodular without additional distributional assumptions. Therefore
> `alpha_V`, `MV_0`, conditional-greedy traces, and Hill estimates do not certify saturation or an unseen
> optimum gap. The bound-grade replacement is the independent all-draw gain audit in §12.6b.

*Per metric `M_i` (§1). The behavior census (§12.1a) puts the **counting measure** on `M_i`'s criteria and
asks how fast new ones appear (`α_i`). It is SILENT on value: a NO-GO (`α_i≈1`) says `M_i` has inexhaustibly
many EXPRESSIBLE criteria, NOT that adding them keeps recovering `M_i`. The value-census swaps in the **value
measure** — mass = each criterion's contribution to **recovering `M_i`** — and re-runs the same machinery.
Where `α_i≈1` destroyed the behavior map, `α_{V,i}` can be small (few criteria actually pin down `M_i`) and
the interpretable decomposition comes back. UNSUPERVISED and anchor-free, **consistent with §1**: value is
measured against `M_i`'s OWN verdict pattern, **never the aggregate `Y`** (the original §12.3 used `I(Y;σ)`
— a slip that conflated all `j` metrics and contradicted §1's label-free loop; corrected here).*

**TASK-level smoke is the WRONG level — do not read it as the answer.** peer-review×llama-8b *pooled over all
metrics* gave `α≈0.975`, long-tail `f`={1:78,2:1,10:1} — exactly the non-discriminating task-level `α≈1`
(§1, §12.1a). Re-run PER R2 CLUSTER: each metric `M_i` gets its own `α_i` and `α_{V,i}`; simple metrics
`α_i<0.5`, complex ones `α_i→1`, and the `α_i ≫ α_{V,i}` gap is the per-metric "many expressible criteria,
few recover it" signal.

**Where it sits relative to `R_i` and `T_i`.** `R_i = I(M_i;M′_i)` (achieved recovery of `M_i`, a FLOOR) and
`T_i = I(M_i;X)` (the DPI ceiling — **the upper bound on how well `M_i` can be measured from `X` at all**, the
quantity we ultimately want) bracket the per-metric optimum: `R_i ≤ OPT_i ≤ T_i`. `α_i` is NOT in those units
— it is the intrinsic dimension of `M_i`'s criterion space; it bounds whether the `R_i→T_i` gap is closable by
**enumeration** and the **description length** of a faithful rubric for `M_i` (`α_i≈1 ⇒ no short rubric`), not
the gap's size. The quantity that climbs `R_i→T_i` is `α_{V,i}`. Read together: **`α_i≈1 ∧ α_{V,i}≪1` = "`T_i`
is reachable but not by enumeration — a few criteria recover most of `M_i`; no finite rubric recovers all."**

#### A. Objects (ASCII; per metric `M_i`; parallels §12.1a-A)

```
metric verdict M_i = executor's verdict using M_i's rubric (the cluster merged_description) on the probe items
                     — the reconstruction TARGET. ANCHOR-FREE: M_i's OWN pattern, NEVER the aggregate Y.
species s        a distinct criterion-signature σ(s) for M_i (as in §12.1a); n_s = times captured ; N = draws
marginal value v(s)= I(M_i ; bin σ(s))            additive   (standalone recovery of M_i; OVER-counts redundancy)
                   = I(M_i ; σ(s) | σ(selected))  submodular (greedy; redundancy-correct)
                   [the SAME machinery as §1's R, restricted to one criterion; Y NEVER enters]
value rarefy  E[V(m)] = Σ_s v_s · ( 1 − C(N−n_s, m)/C(N, m) )   recovery a random m-criterion subset captures
value Heaps   α_{V,i}(m) = d log E[V(m)] / d log m  → terminal α_{V,i} ≪ 1 ⇔ M_i's recovery saturates fast
BREADTH GAP   α_i − α_{V,i}                        "many expressible criteria, few recover M_i" — per metric
value miss-mass MV_0 = ( Σ_{s: n_s=1} v_s ) / N   expected recovery-gain of the NEXT unseen criterion for M_i
value cert    MV_0 ≤ (Σ_{singleton} v_s)/N + B·√(log(1/δ)/N)   B = max per-species v ; N ~ B² for fixed prec.
              STOP when MV_0 ≤ ε (recovery bits), NOT when behavior coverage ≥ 1−ε
```

> **Correction (2026-07-12).** These historical `MV_0`/STOP lines require a species/value map and cap frozen
> independently of iid capture draws. The shipped same-sample census does not meet that premise and issues no
> stopping rule; its singleton-weighted value is descriptive.

#### B. The one genuinely new ingredient: ordering / diminishing returns

```
Behavior counting is EXCHANGEABLE → Good–Turing is clean. Recovery value is ORDERED: M_i's criteria are
REDUNDANT (many express the same sub-distinction), so marginal recovery DIMINISHES with what is already
selected. The right population model is extreme-value + submodular, NOT pure species-sampling:
  - greedy marginal recovery gains g_1 ≥ g_2 ≥ … ; "keep recovering M_i?" = does the tail Σ_{k>K} g_k → 0
  - Hill tail-index on the top per-criterion recoveries = the value-axis analog of α_i (light tail ⇒ done)
  - the iid BREAK (v depends on selection) is the obstacle; SUBMODULARITY bounds the conditional marginal of
    ALL unseen criteria by an unconditional missing-mass quantity (the missing-RECOVERY certificate)
  - additive v(s) vs submodular v(s|sel): the GAP is M_i's criterion REDUNDANCY
  - Good–Toulmin extrapolates NEW recovery in c·N — "how much more of M_i in 10× the criteria"
```

> **Correction (2026-07-12).** Mutual information is not submodular without additional distributional
> assumptions, and an in-pool conditional-gain sequence cannot bound unseen prompts. The listed curves remain
> useful diagnostics; the bound-grade value object is §12.6b's direct gain on every independent audit draw.

#### C. Reuse map — the per-metric recovery loop already IS the value census (file:line audited 2026-06-26)

| primitive | reuse | file:line |
|---|---|---|
| metric scope (`M_i` = one R2 cluster): name, description, atomic criteria `Ω_i` | `mine_clusters.r2_groups` / `r2_children` → `outputs/hierarchy/{task}_{bucket}_r2_expanded.json` | mine_clusters.py:77,100 |
| `M_i`'s verdict (the reconstruction TARGET) | `recon_channel._pyes` on the cluster `merged_description` | recon_channel.py:98 |
| per-metric recovery `R_i = I(M_i;M′_i)` (full §1 loop; a criterion or a set) | `recon_channel.run_metric` → `iv_transmission` | recon_channel.py:575 |
| `v(s)` standalone / conditional (against `M_i`, NOT `Y`) | `vinfo._h_bits` (closed-form binary MI) ; `orthogonalize.shannon_cmi_surrogate` | vinfo.py ; orthogonalize.py:63 |
| missing-RECOVERY certificate (submodular tail) | `orthogonalize.submodular_tail_bound` ; `run_real_test._missing_impact` | orthogonalize.py:144 ; run_real_test.py:254 |
| metric-scoped breadth sample | `alpha_probe.breadth_sample_metric` / `_free_generate_metric` | alpha_probe.py |
| persisted per-metric sample | `run_alpha_probe` (scoped) → `…_sigs.npz` (`--from-checkpoint`) | run_alpha_probe.py |
| rarefaction / `α` (weight by `v_s` not count) | `alpha_probe._rarefy` / `heaps_alpha` | alpha_probe.py |

#### D. The cheap experiment — per metric, CPU-only, anchor-free (no `Y`, no new draws)

For ONE metric `M_i` (one R2 cluster), on its persisted *scoped* sample:

```python
z = np.load(f"{task}_metric{idx}_sigs.npz", allow_pickle=True)
sigs, tags = z["sigs"], list(z["tags"])                       # (n_crit × n_probes) criterion signatures for M_i
labels = ap.collide(sigs, tau)                                # species — SAME τ as M_i's behavior run (GV4)
M_i = score_metric_verdict(executor, merged_description, probe_texts)  # the metric's OWN verdict — NOT Y

species = sorted(set(map(int, labels)))
n_s = np.array([list(labels).count(s) for s in species])
v_s = np.array([ I_binary(M_i, np.nanmean(sigs[[i for i,l in enumerate(labels) if int(l)==s]],0) > 0.5)
                 for s in species ])                          # v(s) = I(M_i ; σ(s)) — recovery, anchor-free
ms, V   = value_rarefaction(n_s, v_s, N=len(labels))
alpha_V = ap.heaps_alpha(ms, V)[-k:].mean()
out = {"alpha_i": "<from M_i's behavior run>", "alpha_V_i": alpha_V, "breadth_gap": alpha_i - alpha_V,
       "MV0": v_s[n_s==1].sum()/len(labels),
       "top_value_criteria": sorted(zip(v_s, species))[-20:]} # ← the criteria that actually recover M_i
```

The ONLY change from the (wrong) `Y`-version: the target column is `M_i` (the metric's own verdict), not the
aggregate `Y`. Everything downstream is identical. Full-strength `v(s)` swaps the closed-form `I(M_i;σ(s))`
for the `run_metric` reconstruction `iv_transmission` (the §1 loop, per criterion).

#### E. Decision rule + guards (per metric `M_i`) *(the `α_V` Heaps readout degenerates when `f_1/N→1` — Lemma 12.6.0; the certificate now reads the conditional/greedy value stream, §12.6.3–12.6.6, with `α_V` valid only once the quotient partition yields real multiplicities)*

```
α_{V,i} ≪ 1 ∧ MV_0 ≤ ε  → M_i's recovery SATURATED: a short rubric recovers M_i; report top-v criteria as
                          M_i's decomposition. (A behavior NO-GO for M_i is then irrelevant.)
α_{V,i} ≈ α_i (gap≈0)   → recovery tracks breadth: M_i genuinely needs many criteria — irreducible to a short
                          rubric (the tacit/holistic metric; its A_i = T_i − R_i is large).
0 ≪ α_{V,i} < α_i       → long tail of redundant criteria with a value tail that keeps paying; read Hill ξ.

GV1  value is measured against M_i's OWN verdict (anchor-free, §1) — NEVER the aggregate Y. (The original
     §12.3 used I(Y;σ): it conflated all j metrics AND contradicted §1's label-free loop — corrected.)
GV2  additive v(s) OVER-counts redundancy ⇒ α_{V,i}/V(m)/MV_0 are UPPER reads; the submodular greedy +
     submodular_tail_bound is the redundancy-correct (lower) read. Report both; gap = M_i's redundancy.
GV3  certificate cost scales with the value range B (N ~ B²): a few high-recovery criteria ⇒ more draws.
GV4  SAME scoped sample, SAME τ as M_i's behavior census — α_i and α_{V,i} must read the SAME partition.
GV5  METRIC SCOPE (the recurring bug): the breadth sample MUST be scoped to M_i (proposer seeded by the
     cluster, criteria restricted to Ω_i) and M_i = that cluster's verdict. Pooling metrics → task-level
     α≈0.9, meaningless. ONE value census per R2 cluster.
GV6  RELEVANCE TARGET = M_i, NOT the items X and NOT Y. The test is I(s; M_i | Ω), not I(s; X | Ω) — X's
     entropy is far larger, so an items-target over-includes criteria informative about the items but not
     THIS metric (and loosens the ceiling from T_i = I(M_i;X) to H(X)). Membership is I(s; M_i | Ω) > ε,
     NOT > 0: a strictly-positive threshold readmits an infinite infinitesimal tail (unbounded again);
     ε is the resolution you certify at, and the sub-ε residual is what the missing-impact bound covers.
GV7  MARGINAL for the Heaps read, CONDITIONAL for the certificate — do not mix. α_{V,i} must use the MARGINAL
     I(s; M_i) (a fixed, stationary test → valid Good–Turing/Heaps). The CONDITIONAL I(s; M_i | Ω) with a
     GROWING Ω is non-iid (the §11.3a stationarity sleeper) and belongs in the submodular missing-impact
     certificate, never the Heaps stream. (The code does this right: α_{V,i} ← additive/marginal v; R_full ←
     submodular/conditional greedy.)
GV8  CERTIFICATE VALIDITY: the upper bound is T_i − R_i (the gap to the DPI ceiling), but only as good as the
     T_i estimate — the posterior/recoverer estimating T_i = I(M_i;X) must be ≥ the reconstructor in strength
     (guard G4), else T_i − R_i is anti-conservative. (Here T_i = H(M_i) because M_i is a deterministic soft
     readout of X — guard G3; a SAMPLED verdict would give H(M_i) > T_i and require I(M_i;X) as the ceiling.)
```

**Net:** the value census is the per-metric recovery loop (§1) viewed as population dynamics — value = each
criterion's contribution to recovering `M_i` (anchor-free, **never `Y`**), reported as `α_{V,i}`, the
**breadth gap `α_i − α_{V,i}`**, and the missing-recovery `MV_0`. The upper bound on measuring `M_i` is
`T_i = I(M_i;X)`; the census says, per metric, whether a short rubric reaches it. One run per R2 cluster.

> **Correction (2026-07-12).** The historical decision and Net paragraph are descriptive hypotheses, not
> saturation conclusions. Same-sample `alpha_V`, `MV_0`, Hill, and greedy CMI do not bound the prompt optimum.

### 12.4 Upper-bounding the ideal `M_i*`: what B_E can and can't give. **[new, 2026-06-26]**

*The goal that keeps getting answered at the wrong level. We are NOT asking the within-pool question
(`H(M_i_ω)` = the ceiling for the operationalized verdict — that is the FLOOR). We want the ceiling on the
IDEAL metric `M_i*` the cluster is reaching for, which lives BEYOND the current pool `Ω` — i.e. it requires
saying something about `B_E`. **There is no label `Y` — none exists.** The entire construction must be
unsupervised. Stop re-deriving `H(M_i_ω)`; stop invoking `Y`.*

**Bracket, correctly oriented.**
```
   within-pool recovery R(Ω)   ≤   measurability of the ideal M_i*   ≤   ??? (from B_E, beyond Ω)
   (FLOOR — a richer pool does                                          (CEILING — the open question)
    at least this well)
```

**Assumption-free, B_E gives only the loose cap.** With no structural assumption the sup over all reachable
behaviors is bounded by the information cap (§3.1), `1 − 1/min(N,K)` / `log min(N,K)`; and when the reachable
space is inexhaustible (`α≈1`) the B_E route *degenerates to exactly this cap* (`I(all-of-B_E; X) → H(X)`).
Sampling B_E harder cannot tighten it. The raw-B_E-*extent* route is a dead end for a tight ceiling.

**Constructive, fully-unsupervised route: R-weighted, cross-source consensus stabilization (no `Y`).** The only
ground truth available without labels is *self-consistency among reachable criteria.* Draw criteria coherent
with the cluster from `B_E` (scoped proposer, BEYOND `Ω`), weight each by its **reconstructability** `R(s)`
(its §1 recovery — `run_metric → iv_transmission`; this keeps the GENERALIZABLE agreement and strips
shared-but-non-generalizable artifacts — the value target is `R`, not the verdict `M_i_ω`; the cheap proxy
`I(M_i;σ)` of §12.3 is the within-pool shortcut, the wrong object here), and track the consensus as the pool
grows:
```
draw coherent criteria  s_1, s_2, …    (from B_E beyond Ω; NO labels), each with reconstructability R(s)
per-source consensus     c_k = top R-weighted factor of proposer-family k's columns   (k = a source/family)
CROSS-SOURCE consensus    c   = agreement ACROSS families {c_k} — NOT the pooled factor (that over-weights the
                         most prolific proposer; cross-source is what makes per-family biases CANCEL). The
                         independence that matters here is SOURCE-independence (§11.3a recapture axes), NOT
                         coverage-iid (Good–Turing C_lo, §12.1a) — a different requirement, do not conflate.
stabilization (split-half) align(m) = |cos(c_A, c_B)| over two DISJOINT m-criteria subsets, averaged
                         → 1 fast ⇒ ideal pinned ; plateau < 1 ⇒ no stable ideal (incoherent cluster).
                         The RIGHT signal — the criterion-COUNT α stays ≈1 (proposer creativity) while the
                         consensus can still converge.
UPPER BOUND              c_(m) → c_∞  ⇒  ceiling on the ideal ≈ H(c_∞) ;  beyond-pool gain = H(c_∞) − R(Ω)
beyond-pool signal       |cos(c_∞, M_i_ω)| and H(c_∞) − H(M_i_ω): does the consensus EQUAL the current rubric
                         (≈1, 0 → no gain) or DRIFT richer (<1, >0 → the ideal extends past Ω)? — exactly
                         what α_V cannot produce (its target is fixed at M_i_ω).
BIAS-FLOOR diagnostic    recompute c_∞ as you ADD diverse proposer families; when c_∞ STOPS shifting you have
                         hit the common-LM-bias floor (§12.2.4) — the irreducible unsupervised residual.
```
Estimating `c_∞` is a **Good–Toulmin-style extrapolation of the stabilization curve** — assumption-bounded by
smoothness/saturation (Lipschitz-impact flavored), a point estimate, NOT a certificate. (Mechanically it
reuses the value-census machinery, but the target is the cluster's OWN evolving cross-source consensus,
R-weighted — never a fixed verdict, never `Y`.)

**The one irreducible caveat — and it is NOT the `Y` problem (there is no `Y`).** `c_∞` is the
**proposer-reachable consensus**: it is `M_i*` *up to shared proposer bias*. If every proposer family shares
the LM's blind spots, `c_(m)` converges to the LM-biased consensus, not the true ideal — and no unsupervised
procedure can separate the two (§12.2.4, in unsupervised form: shared support looks like truth). The ONLY
unsupervised lever that widens past it is **proposer-support diversity** — genuinely different model families
(different training support), so the consensus is taken over a wider reach. Not labels; we have none. The
residual `M_i* − c_∞` is unmeasurable without an out-of-distribution sampler whose support is not a subset of
the current proposers'.

**Net.** Assumption-free → the loose cap. Assumption-bounded (unsupervised) → the consensus-stabilization
extrapolation `H(c_∞)` under a smoothness assumption, anti-conservative under shared proposer bias, tightened
only by proposer diversity. There is no labeled-target route — there is no label. The honest ceiling we can
report is **the entropy of the reachable coherent consensus**, scoped as `M_i*`-up-to-shared-proposer-bias.
The right thing to measure is the **consensus stabilization rate**, NOT the criterion-count `α` (which is
≈1 by proposer creativity and tells you nothing about the ideal).

### 12.5 The process-relative ceiling — a scoping note on the `B_E` claims. **[new, 2026-06-26]**

*One small point, kept deliberately separate from the `T`/`R`/headroom bracket (§11): it is about what KIND of
claim our coverage/consensus numbers are — not about their values, and not a new bound.*

**We never sample `B_E` iid.** There is no unbiased oracle over "all reachable criteria." Every generator we
have is a *non-IID, biased process*: LLM free-gen (creativity bias), GEPA (optimization-conditioned bias),
curated children (human/pipeline bias). Good–Turing coverage (§12.1a), the consensus `c_∞` (§12.4), and the
recapture axes (§11.3a) all *assume a sampling process* — and ours is irreducibly non-IID.

**So read every `B_E` claim as process-relative.** Coverage, `H(c_∞)`, "the ceiling on the ideal" are scoped to
*what the specific set {LLM, GEPA, children} reaches* — never `B_E` in the absolute. The honest phrasing is
always "…reachable by these generators," not "…full stop."

**This is the right scope, not a defect.** The generators are *diverse*, so cross-process agreement cancels
their *idiosyncratic* biases (the scoped claim beats any single process); their *common* bias is the boundary,
and since IID is unattainable that boundary is not removable — only *named* (the §12.2.4 floor, in process
form). The virtue is that the scope is **replicable and resource-invariant**: it bounds what anyone using the
same generators achieves at any compute. So our methods certify a property of **exploration by a specific,
diverse, non-IID set of criterion-generating processes**, not a property of `B_E` itself. State the
process-set; report the claim relative to it. That is the whole point — small, and orthogonal to the `T`/`R`
machinery above.

## 12.6 The capture-recapture diagnostic — family scaling × prompt permutation. **[certificate claim withdrawn 2026-07-10]**

*This section preserves the diagnostic construction and documents why it is not currently an upper-bound
certificate. The count-axis and value-Heaps reads remain descriptive.*

*Scope (the anthropological estimand).* The study object is **human preference as accumulated revealed
practice**: the decades of outcome decisions `Y` (accept/reject, contest wins, editorial selection) are the
human evidence — archival, unobtrusive, longitudinal. **No realtime elicitation is used, by design:** asking
people to articulate their taste measures confabulation (Nisbett–Wilson), not the practice; the practice is
in the record. The two levels are: task-level revealed-practice contrasts and metric-level anchor-free
instrument diagnostics. The current method does not certify a codification gap or exhaustion of the named
articulation process; all such differences must be labeled descriptive until a valid upper confidence bound
replaces the bridge below.

### 12.6.0 Why the observed `α ≈ α_V ≈ 1` was inevitable — the singleton-degeneracy lemma. **[derived]**

> **Lemma (degeneracy).** If every observed species is a singleton (`f_1 = D = N`), then for ALL `m`:
> `E[S(m)] = m` and `E[V(m)] = (m/N)·Σ_s v_s` — both exactly linear — so `α(m) ≡ 1` and `α_V(m) ≡ 1`
> **identically, regardless of the value profile `{v_s}`.** More generally both slopes → 1 as `f_1/N → 1`.

*Proof.* Rarefaction with `n_s = 1`: `C(N−1,m)/C(N,m) = (N−m)/N`, so `E[S(m)] = N − N·(N−m)/N = m`, and
`E[V(m)] = Σ_s v_s (1 − (N−m)/N) = (m/N)Σ_s v_s`. Log-log slope of a linear curve = 1. ∎

*Consequence.* In the singleton-dominated regime the count exponent and the value exponent are **pinned at 1
mechanically and carry zero information** — high `α_V` is NOT evidence of inexhaustible depth, and low `α_V`
is unreachable. The two exponents can only separate when the spectrum has real multiplicities (recapture),
i.e. after the partition actually collapses paraphrases. **Empirical check that must accompany every `α`:
report `f_1/N`.** (The observed task-level `f = {1:78, 2:1, 10:1}` ⇒ `α ≈ 0.975` is this lemma, exactly.)
Where recapture is absent, the informative value objects are the **multiplicity-free** ones: the greedy
conditional-gain sequence `g_1 ≥ g_2 ≥ …`, its tail sum, the Hill index on `{v_s}`, top-k value share.

### 12.6.1 Objects

```
quotient species   s ~ s'  iff SAME-content:  semantic-judge MERGE (paraphrase/form-orbit collapse)
                   + behavioral SPLIT within (distinct verdict pattern splits; judge never splits)
form orbit  Φ      sampled meaning-preserving maps: template order (rubric-first/text-first), clause
                   order, boilerplate, paraphrase — the measured group the content unit is quotiented by
target      m̄_ω    m̄_ω(x) = (1/|Φ|) Σ_{φ∈Φ} P(YES | x, φ(rubric))   — the Φ-averaged soft verdict
value       v(s)   v(s | S_g): marginal recovery gain of s given the FROZEN post-run greedy set S_g
                   (frozen ⇒ a fixed function ⇒ stationary stream; GV7 satisfied)
spectrum    w_j    Σ_{s: n_s = j} v(s)        (value-weighted multiplicity spectrum on the quotient)
lists              K frozen iid proposer families + 1 novelty-tilted (§12.1a); N total draws
ladder      ℰ      capability-ordered executor families E_1 … E_L (3B → 8B → 70B → 122B → …)
```

### 12.6.2 Leg 1 — prompt permutations (the form-orbit quantification). **[design + derived]**

**The target is made `Φ`-invariant by construction.** `m̄_ω` averages the soft readout over the sampled
orbit. It is still deterministic given `X` (an average of deterministic-given-`X` readouts), so `H(m̄|X)=0`,
the DPI chain and `T = I(m̄_ω; X)` survive unchanged (guards G3/GV8 intact). Cost: `|Φ| ≈ 3–5` extra scoring
passes **on the probe set only** (the #6 caching applies). `Var_φ[σ]` is reported as the instrument-error
bar on every `T`/`R`.

**Every partition-dependent statistic is reported at the ADVERSE end of the orbit.** For any statistic
`θ(partition)`: compute `θ` under each sampled `φ ∈ Φ` (template reorder, Ω-order permutation, probe
subsample) and report `[min, max]`; a certificate must hold at the adverse end. This replaces the pretense
of invariance with a randomization interval — the permutation gates (`order_stability`,
`prompt_ordering_check` ARI, `form_invariance` flip-rate) are the pass/fail summary of that interval's width.

**Asymmetric partition-robustness (which fragilities matter).** The certificate's value accounting is
- **over-split-SAFE:** a paraphrase mis-split into a fake new singleton carries conditional value
  `v(s|S_g) ≈ 0` (its signal is already in `S_g`), so it inflates `f_1` but adds ≈0 to `w_1` — the
  value-weighted missing mass is intrinsically robust to the failure mode that destroyed the count census;
- **over-merge-UNSAFE (the one anti-conservative direction):** a genuinely novel criterion absorbed into an
  existing species never becomes a singleton, so its value never enters `w_1` ⇒ flux under-estimated.
  **Merge-precision is therefore the binding validity gate** (the `semantic_behavioral` audit — currently
  0.28, must be fixed by the semantic-merge/behavioral-split quotient), and `w_1` is reported at two probe
  sizes (merge decisions are probe-limited).

**The head of the certificate is partition-FREE.** The greedy conditional-gain sequence `g_1 ≥ g_2 ≥ …` is
computed on the raw pool (no species needed): duplicates land in the tail with gain ≈ 0, so over-splitting
inserts zeros without moving the head, and no dedup decision can corrupt `OPT_Ω`. Species identity enters
ONLY through the tail flux `w_1` (12.6.3) — the fragile object is confined to the smallest role.

**`FORM-DOMINATED` is a finding, not a failure.** A metric whose gates fail (flip-rate > 10%, ARI low even
after the quotient) has content that does not survive rephrasing — meaning inseparable from wording. For a
linguistic study this is a category (ineffability-of-paraphrase), reported as such; its certificate is
stated only at the orbit-adverse end.

### 12.6.3 Leg 2 — population capture-recapture on the VALUE measure (derivations D1–D3). **[derived]**

**D1 — value-weighted Good–Turing (Robbins) flux.** The unseen-value flux (expected new-species value
carried by the next draw) is `Φ_V := Σ_{s unseen} p_s v_s`. Then
`E[Φ_V] = Σ_s v_s p_s (1−p_s)^N = (1/(N+1)) Σ_s v_s · P[n_s = 1 in N+1 draws] = E[w_1^{(N+1)}]/(N+1)`
(second equality: `P[n_s=1 in N+1] = (N+1) p_s (1−p_s)^N`). So the plug-in
> **`V̇ := w_1 / N`** estimates the unseen-value flux, with the Good–Turing bias rider `≤ B/N`
(`B = max_s v(s)`). NOTE the object: the **flux** (value per additional draw), NOT the total unseen value —
totals need positivity (untestable, §12.2.4) or a horizon (D3). ∎

**D2 — concentration (McDiarmid).** Changing one of the N iid draws moves at most two species across the
singleton boundary, each shifting `w_1/N` by ≤ `B/N` ⇒ bounded differences `c_i = 2B/N` ⇒
> `P( E[Φ_V] > V̇ + B√(2 log(1/δ)/N) + B/N ) ≤ δ` — one-sided, assumption-free given the frozen iid lists
(the value analog of the §12.1a Berend–Kontorovich `C_lo`). ∎
*Scope correction (2026-07-11, external review):* "valid under ANY family dependence" was too
broad — the statement requires **iid draws from one fixed distribution AND a species map / head /
value function frozen independently of those draws**. Our real streams are (a) family-STRATIFIED
(200/200/200 block design, not iid from a mixture) and (b) were used to build the species map and
head, so the premise fails twice. The repaired route is the CR-3 discovery/audit design (§12.6b):
per-family exact bounds + frozen-before-audit objects.
*Correction (2026-07-10):* a safe pre-selection value cap is `B=H(M)`, fixed before head selection. The
previous `H(M)-OPT_Ω` cap was data-dependent. Optional-stopping and order-statistic error spending can make
this next-draw flux band simultaneous; it does not cover the horizon estimator below.

**D3 — Good–Toulmin value horizon.** Expected NEW value discovered in the next `c·N` draws:
`Δ_V(c) = Σ_s v_s (1−p_s)^N (1 − (1−p_s)^{cN})`. Poissonizing (`λ_s = N p_s`):
`Σ_{j≥1} (−c)^j E[w_j] = Σ_s v_s e^{−λ_s}(e^{−cλ_s} − 1) = −E[Δ_V(c)]`, so
> **`Ĝ(c) = − Σ_{j≥1} (−c)^j w_j`** is the (Poisson-)unbiased horizon estimator — exact machinery for
as a formal infinite-series point estimator; for `1 < c ≲ log N` use Orlitsky–Suresh–Wu smoothing (the alternating series' variance blows up
raw). OSW supplies a normalized-MSE prediction result for unseen counts, not the one-sided weighted
horizon deviation bound needed here. A next-draw missing-mass radius cannot be added to this total-horizon
statistic without a separate sensitivity/bias derivation. ∎
*Implementation note (2026-07-01):* for `c ≤ 1` the code truncates the series at `j ≤ k₀ = 4`, introducing
a bias-variance tradeoff rather than preserving unbiasedness. A plug-in Poisson calculation suggests that a
species recaptured at least five times has small unseen probability, while its raw alternating term can inject
`O(w_j)` sign-oscillation (on a saturated spectrum such
as `w = {5,6,7}` the raw series returns `w_5 − w_6 + w_7 ≈ w_jmax`, manufacturing horizon mass out of
parity). `k0=None` recovers the raw series. For `1 < c ≤ ln N`, OSW-style binomial smoothing is
implemented (`osw_horizon_value`, §12.8.7 I1). Its horizon metadata marks the regime of the source point-
prediction result, not a one-sided weighted-value certificate.

The **counting** census is retained in its §12.1a/§12.2.4 role — `C_lo` = coverage of the union support
(secondary), Chao1/two-list LP = richness cross-checks — and gains a *linguistic* reading: the Heaps curve
of the quotient species is the **vocabulary-growth law of the community's evaluative lexicon** (`α` = lexical
productivity), now cleanly separated from value depth (`g_k` tail), which the Lemma shows the raw exponents
conflate.

### 12.6.4 Leg 2→bridge — proposed flux-to-gap diagnostic. **[not currently certified; 2026-07-10]**

Weak submodularity at the frozen `S_g` (the Das–Kempe definition applied to `(S_g, U)`, `U` = unseen):
`R↑(S_g ∪ U) − R↑(S_g) ≤ (1/γ̂) Σ_{e∈U} v(e | S_g)` — with `γ̂` the measured tail submodularity ratio
(lower-tail estimate on the discovered tail, §6.2; monotonization caveat §3.2 applies). Bounding the unseen
conditional-value total **at the process horizon** by D1–D3:

> **Historical proposed bridge.** Relative to the named process run to horizon `(1+c)N`:
> ```
>   CHECKLIST-OPT_{F, process,(1+c)N,Φ}  ≤  OPT_Ω  +  ε ,   ε = (1/γ̂)·[ Ĝ(c) + O(B√(log(1/δ)/N)) + B/N ]
> ```
> where **CHECKLIST-OPT_F** = the best selection/weighting of separately-executed criteria under a
> **named combiner class F** (implemented: F₁ = additive-logistic readout, `combiner='linear'`;
> F₂ adds pairwise-interaction columns on the head, `combiner='pairs'`). Backstopped by (i) the
> γ-free adversarial probe `I(X_probe; M_i | X_Ω) ≈ 0` — whose probe set MUST include
> holistic/composed prompts (`covers_composition`), else it certifies the checklist channel only —
> and (ii) the unconditional DPI ceiling `≤ T(m̄_ω)` (covers everything, composition included).

This display is **not a current probability certificate**. The implementation adds a next-draw flux
deviation radius to a horizon point estimator without a theorem connecting that radius to horizon error;
the hard-truncated/OSW estimator is a point predictor; the selected head and species values reuse the same
probe data; and measured `gamma_hat` is not a lower confidence bound for the unseen tail. The tail-XOR
breaker below shows that even a fully recaptured observed pool can hide higher-order conditional value.
Until a frozen selection/evaluation split, a one-sided horizon theorem, and a certified structural lower
bound on gamma are supplied, report `OPT_Omega` as achieved checklist value and `epsilon` as sensitivity only.

**Historical task-level proposal (withdrawn as a certificate).** The quantity
`lowerCI(C) − [OPT_Ω + ε]` was intended as a codification-gap lower bound. Because `OPT_Ω+ε` is not a valid
upper confidence bound, this difference is currently a descriptive contrast only and cannot support the
anthropological lower-bound claim.

**The three named escapes (what the certificate does NOT cover — each measured or scoped, never
assumed away):**
1. **Zero-mass tail (unchanged, §12.2.4):** a criterion NO proposer family can emit is invisible at
   any horizon; converting flux→total needs positivity (`p_s ≥ p_min` ⇒ total ≤ flux/`p_min`), which
   is untestable. The certificate claims the **process horizon**; the beyond-horizon residual is the
   named §12.5 scope.
2. **Composition channel (named 2026-07-01):** a single COMPOSED prompt is not a function of the
   unit verdicts — the executor's joint reading (order, phrasing, persona/gestalt framing) is a
   different channel: **no unit-verdict DPI applies and prompt-space performance is not monotone in criteria**, so
   unit-level accounting cannot bound prompt-space OPT. The fixed-target `M_omega-X-M_p` DPI still applies
   directly to the composed prompt. Response: the residual is MEASURED —
   `Δ_comp` (`composition_gap.delta_comp`: composed variants vs the certified head, with
   `Δ_comp_beyond = I(exec(composed); M | S_g)` as the sharp read) plus holistic/GEPA-whole-prompt
   probes inside `adversarial_saturation`. `Δ_comp ≈ 0` ⇒ the unit model is empirically adequate for
   that metric; `Δ_comp > 0` on taste metrics with ≈ 0 on craft metrics is itself the finding — "the
   tacit lives in the SAYING, not the said" — not an instrument failure.
3. **Combiner class:** the certificate is F-indexed (§12.7 executor-relativity biting our own
   instrument): F₁ is parity-blind — an XOR-of-two-criteria metric reads "tiny OPT, saturated" while
   a checklist stating the conjunction recovers a full bit (the XOR planted control documents this
   limitation). F₂ (`[x_i, x_j, x_i·x_j]` pair candidates) covers pairwise conjunctions; deeper
   conjunctions remain outside the stated F. Every certificate must name its F.

For the ineffability thesis this shape is *correct*: the claim is falsifiable by any future checklist
that beats `OPT_Ω + ε` under the stated F, by any composed prompt with large measured `Δ_comp`, and
by any richer combiner that closes the gap.

### 12.6.5 Leg 3 — model-family observational scaling (what curves may and may not certify). **[holds]**

Per executor tier `E_t`: compute `OPT_Ω(E_t)`, `ε(E_t)`, `T̂(E_t)`, and the gap
`Δ(E_t) = lowerCI(C) − [OPT_Ω(E_t) + ε(E_t)]`. Rules, inherited from the standing no-extrapolation memo
(V-info non-Lipschitz; Relative Scaling Laws heterogeneity):
- **The frontier is a monotone staircase, not a fit:** `OPT* = max_t OPT_Ω(E_t)` is the achieved lower
  frontier (always valid); a **fitted saturation asymptote is NEVER a bound** — report per-tier points.
- **The verdict is the trend of `Δ`:** flat across ≥3 tiers (slope CI ∋ 0) = the strongest available
  process-relative tacitness evidence ("no capability trend closes it"); still-shrinking = right-censored
  ("not yet articulated at capability `E_L`"), never "tacit, period."
- **Same-family ladders only, for any slope claim (2026-07-01, binding):** training scaling laws are
  within-family objects (Kaplan/Hoffmann hold architecture + data recipe fixed), and the observational
  variant that spans families (Ruan–Maddison–Hashimoto 2024) does so only through latent-capability
  axes — raw cross-family size-vs-metric curves are confounded by family offsets (data mix, tokenizer,
  post-training judge-bias) and the x-axis is ill-defined across dense/MoE (active vs total params).
  With 3–5-point staircases one family offset can flip flat↔shrinking. Protocol: the `Δ(E_t)` slope is
  computed WITHIN one model family; other families enter as family-indexed REPLICATION staircases/points
  ("does the verdict replicate?"), never pooled into one slope. Prior mixed-family `B_E` scaling
  (llama-3b/8b + qwen-122b) is descriptive only — it does not sanction mixed ladders.
- **Capacity-artifact cross-checks:** `B_E(E)` rising while executor discrimination (between-signature L1)
  plateaus flags census growth as readout-capacity artifact, not metric structure (the `pipeline_audit`
  read); cross-tier metric *rankings* are not trusted below rank-agreement ≈ 0.4 — trends only.
- **Proposer-side scaling (writer diversity):** `Δ_div → 0` and the §12.4 `c_∞` bias-floor (consensus stops
  moving as families are added) are the writer-axis saturation reads; both are census diagnostics, not
  certificates.

### 12.6.6 The historical decision rule and current safe statuses. **[corrected 2026-07-10]**

The old table treated the heuristic epsilon as a bound. Current interpretation:
```
UNDERSAMPLED    singleton-dominated or heuristic epsilon unresolved  → descriptive; draw more/fix quotient
FORM-DOMINATED  form gates fail                                      → empirical wording-instability finding
CODIFIABLE      unavailable from the current bridge                  → requires upper_bound_valid=True and
                                                                        explicit adversarial saturation
DEEP            heavy achieved greedy tail with recapture            → descriptive candidate depth, not a
                                                                        tacitness or inexhaustibility proof
```
The code now prevents `CODIFIABLE` when `upper_bound_valid=False` or `adv_saturated` is missing. It may still
emit `DEEP`/`UNDERSAMPLED` as descriptive diagnoses. A future CODIFIABLE rule needs a separately validated
upper-bound payload; the current process-horizon epsilon cannot supply one.
Count-`α`, Chao1, `B_E` remain as *descriptive* vocabulary statistics with orbit/probe/order error bars.

**Order-adverse reporting (2026-07-01).** Every partition-DEPENDENT certificate quantity — the
spectrum `w_j`, flux, `Ĝ(c)`, hence ε — is reported at its **adverse end over core-build order
permutations** (`n_orders` re-partitions of the strict behavioral quotient; the head and `S_g` are
order-free by construction, so only the flux leg moves): `eps_bits_adv = max_k ε_k`, and `decide`
reads the adverse end — the same convention as the orbit's adverse end. What permutations can and
cannot do: they **quantify** the partition's order-arbitrariness band (and tighten its estimate as
`n_orders` grows) and they convert the over-split-safety *argument* into a per-metric *measured*
bound; they **cannot shrink** a real non-identifiability — richness (Chao1/`B_E`) stays descriptive
because the §C max-dependence result makes it unbounded without positivity, no matter how many
permutations are run. The max over the `n_orders+1` statistics is δ-unioned and the whole band is
time-uniform over re-issuance checkpoints (Theorem T1, §12.8.0) — without that union the adverse-end
read was a level-`(n_orders+1)·δ` claim, and re-issuing after more draws voided it entirely. Judge-dependence of merges is NOT covered by order permutations — it is covered
by dual-quotient reporting (behavioral vs judge, adverse = max), without re-spending judge quota
per order.

| # | claim | status | measure |
|---|---|---|---|
| **R11** | singleton degeneracy: `f_1/N→1 ⇒ α ≡ α_V ≡ 1` mechanically (value profile irrelevant) — observed high `α, α_V` are this artifact, not depth | **derived (Lemma 12.6.0)** | report `f_1/N` beside every exponent |
| **R12** | value-flux diagnostic: `Vdot = w_1/N` and Good–Toulmin/OSW horizon point estimates are descriptive; the implemented radius and sampled `gamma_hat` do not prove `CHECKLIST-OPT_F ≤ OPT_Ω+epsilon`. | **upper-bound claim withdrawn (2026-07-10)** | `w_j`, heuristic `epsilon`, explicit `upper_bound_valid=False` |
| **R13** | partition-robustness is asymmetric: head (greedy `g_k`) partition-free; flux over-split-SAFE / over-merge-UNSAFE ⇒ merge-precision is the binding gate | **derived** | pool-level greedy; `semantic_behavioral` merge-precision; `w_1` at 2 probe sizes |
| **R14** | family-scaling verdicts are trend-descriptive only (flat ⇒ strongest process-relative evidence; shrinking ⇒ right-censor); fitted asymptotes forbidden | **holds (standing rule)** | per-tier `Δ(E_t)` + slope CI; discL1 plateau check |
| **R15** | `m̄_ω` (orbit-averaged soft target) keeps DPI/`T` intact and makes the target `Φ`-invariant; FORM-DOMINATED is a reportable category | **holds (design)** | `Var_φ` bar; form gates |
| **R16** | composition escape: a composed prompt is not a function of unit verdicts, so an Omega/checklist argument cannot bound it. The fixed-target DPI `R(p)≤T(M_omega)` still covers the composed prompt directly. | **named limitation + measured** | `Delta_comp`; holistic/GEPA probes; fixed-target DPI payload |
| **R17** | order-adverse ε: partition-dependent quantities reported at the max over core-build permutations; permutations quantify the band, they cannot rescue richness (non-identifiable without positivity) | **holds (design)** | `eps_bits_adv` + `order_band` in every certificate; `decide` reads the adverse end |
| **R18** | combiner-class indexing: F1 (linear) is parity-blind; F2 adds pair interactions. Every achieved/checklist analysis must name its F, but naming F does not certify pool coverage. | **derived limitation + design** | `combiner` field; XOR planted control |
| **R19** | optional-stopping and order-union bookkeeping can make the next-draw flux band time-uniform, but does not turn the horizon point estimate or synergy bridge into a confidence bound. | **bookkeeping holds; epsilon certificate invalid** | `stopping/checkpoint/delta_effective/n_union/B_flux_cap` plus `upper_bound_valid=False` |
| **R20** | assumption ledger: the valid lemmas and impossibility results remain useful, but designated controls can falsify assumptions, not prove that every untested escape is absent. | **diagnostic framework, not certificate** | ledger table; planted battery including tail-XOR breaker |

### 12.6.7 What this does not certify (unchanged walls, correctly scoped)

Support-completeness (§12.2.4), behavioral-vs-intent (§11.4), and the three named escapes of §12.6.4
(zero-mass tail / composition channel / combiner class) stand. Under the anthropological framing
they are not defects to hide but limits on interpretation. The current defensible claim is only an achieved
checklist value and a process-horizon sensitivity analysis; it is not "incommunicable as a checklist" in
certified form. A future checklist that beats `OPT_Ω+epsilon` would demonstrate undercoverage, as the existing
tail-XOR breaker already does structurally.
The direction-of-error discipline flips accordingly: every instrument weakness inflates the gap the
thesis wants, so the **mandatory planted controls** accompanying every DEEP verdict are: the C1
planted-rule positive control (`R → 1` through the FULL quotient pipeline), the **XOR control**
(documents the F₁ blind spot; recovered under F₂), the **composition control** (a planted
composed-channel bit must read `Δ_comp_beyond > 0`; a restated checklist must read ≈ 0), the
per-tier flatness read, and empirical planted-reference comparisons. The historical tail-XOR breaker
(§12.8.8 C2) shows why no CODIFIABLE verdict can be based on the current bridge. Adversarial saturation is a
useful falsification probe, not a proof that all untested higher-order escapes are absent.

## 12.6b-H Historical CR-3 discovery/audit proposal and M*/M_b reframe. **[written 2026-07-11; retained as research history]**

Two external reviews (2026-07-11) were accepted essentially in full. This section records (A) the
corrected notation separating the ideal metric from its operationalizations, (B) exactly what
capture–recapture can and cannot bound, (C) the certified CR-3 design that replaces CR-2's calibrated
heuristic for every bound-flavored claim, and (D) the paper-claim hierarchy.

### A. M* / M_b notation (the ideal-metric reframe)

- **M\*** — the ideal, prompt-independent metric (the concept itself).
- **M_b** — an operationalization produced at definition/refinement budget *b*. Our `M_ω` is one M_b.
- **Z_p** — the executor's output under prompt *p*; **R\*(p) = I(M\*; Z_p)**.
- **T\* = I(M\*; X) ≤ H(M\*)**, with equality iff M\* is deterministic given X. If ideal judgment
  carries legitimate evaluator disagreement, model `M*(x) ~ Bernoulli(q*(x))`; then
  **T\* = H(M\*) − E_X[H_b(q\*(X))]** — the disagreement term is *not* recoverable structure.
- Prompting cannot change H(M\*). The improvement target is **I(M\*; Z_p) ↑** (equivalently
  H(M\*|Z_p) ↓) — never raw output entropy (a constant executor has minimal entropy and zero use).
- **Garbling lemma (assumption-explicit):** if M_b is a garbling of M\* (M_b ⊥ X | M\*), then
  T(M_b) = I(M_b;X) ≤ T\*. This is the formal content of the standing "T lower-bounds the ideal"
  doctrine — and it is exactly as strong as the garbling assumption, no stronger.
- **Two distinct GEPA experiments, never to be conflated:** (1) *instruction optimization* — semantics
  fixed, GEPA improves execution of a fixed M_b; self-recovery is the right score. (2) *metric
  refinement* — GEPA changes the substantive metric, producing a new M_b; self-recovery CANNOT
  evaluate this (a rubric can converge beautifully to the wrong target); it needs M\*-side evidence.
- H(M_ω) is hereby demoted to a **self-fidelity diagnostic**. The paper's ceiling is **T\***, held via
  the identified-set route in (D).

### B. What capture–recapture can bound — three routes, no fourth

1. **Missing behavioral probability mass** — assumption-light, finite-sample valid (CR-3 U0).
2. **Discoveries within a fixed future budget m** — valid once mass is bounded (m·U0; quantile form).
3. **All remaining species / all remaining value** — requires a declared mass floor p_min, a finite
   universe, or a parametric tail model. **There is no fourth route**: population size is not
   identified without assumptions on unseen inclusion probabilities and list dependence
   (Aleshin-Guendel, Sadinle & Wakefield 2024; Valiant & Valiant for the support-size floor).

The "reachable within ~10×" scope language is now the **literal assumption** p_min = 1/(10·N₀), N₀
fixed in advance, reported with a sensitivity curve over p_min ∈ {1/5N₀, 1/10N₀, 1/20N₀, 1/50N₀}.
Note: 10× mining catches a boundary-p species w.p. ≈ 1−1/e ≈ 63% — the floor is an assumption about
scope, not a claim of exhaustion.

### C. CR-3 (historical 2026-07-11 implementation)

Predeclare (metric, executor, probe distribution, proposer families + frozen mixture, species rule,
combiner class, horizon m, p_min, α). Split the capture stream — not the probes — into DISCOVERY
(species map + head built, then frozen) and AUDIT (never touches them). Conditional on the frozen
pool, audit novelty is Binomial ⇒ exact one-sided Clopper–Pearson U0, computed **per family** at
α/2F and combined under the frozen balanced mixture (declared assumption: within-family
exchangeability only). Conversions as in (B). Value: frozen bounded marks
`Y_i = 1[new]·clip(CMI(y_B; col_i | frozen head), 0, H(M))` (cap predeclared — never max-observed),
one-sided empirical-Bernstein ⇒ U_φ; horizon m·U_φ; totals U_φ/p_min. **No recovery-optimum
conversion** without a certified submodularity ratio γ_L (MI is not submodular in general — Iyer et
al. 2021); the assumption-free fallback is `E[mining improvement] ≤ H(M)·min(1, m·U0)` — valid even
if a single new species unlocks the entire residual. α-allocation is explicit in the payload.

CR-2 (`cr_horizon.py`) is retained as a **descriptive diagnostic only** (saturation detection +
component decomposition): its G1 is a one-doubling flux, its slack's freezing premise was unmet, its
stream-iid premise fails under the stratified design, its battery test coverage (30/36 = .83, exact
95% CI ≈ .67–.94) does not support bound semantics, and additive per-species marks can exceed H(M)
outright (humor Production-design: G1 = 2.68 bits vs H = 0.27 — the H-cap did all the bounding).
Its honest conclusion form is: *"under the declared species rule, this mining process shows no
saturation — the experiment cannot rule out substantial additional value."* Clarification of a
phrasing bug: **2-unit XOR is in scope** via the pair chain; **parity depth ≥ 3 remains a declared
blind spot** — there was never a depth-3 claim.

### D. Identifying M* and the historical paper-claim hierarchy

Capture–recapture cannot identify M\* from prompt behaviors alone (mutually correlated prompts can
converge to the wrong target). Route: a Dawid–Skene-style latent-target model over **independent
views** — generically identifiable with three views under explicit conditional-independence + rank
conditions (Allman–Matias–Rhodes 2009) — or, when those assumptions are doubtful, a **confidence set
C(M\*)** and the robust bound `U* = sup_{M∈C} min{ I(M;X), R_M(S₀)+U_gap(M) }`
(partial-identification style, Finkelstein et al. 2021). **Project constraint note:** the review's
first view (human expert panels) is excluded by the standing no-human-annotation rule; our views are
(i) archival revealed-practice outcomes (merged PRs, acceptance, N&C rule-changes — evaluate-only),
(ii) independent high-budget operationalizations with model families and prompts separated. C(M\*)
is correspondingly wider; that price is stated, not hidden.

**Paper hierarchy (replaces every retired chain):**

    achieved recovery  ≤  process-relative CR-3 bounds (per-quantity, stated assumptions)
                       ≤  identified-set bound U* for M*  ≤  H(M*)

with T\* = I(M\*;X) as the scientific ceiling and H(M_ω) as self-fidelity context only.

**Planned tracks (review-accepted):** (1) behavioral saturation, 40–60 metrics × 600 discovery + 300
fresh audit draws with family randomized per draw (0 novelties in 300 ⇒ U0 ≈ 1%); (2) long
continuation, ~12 metrics, prospective bounds at N=600 → +600 → +4,800 with no retuning; (3)
ideal-target study, 6–10 central metrics × ≥2 independent views across a definition-budget ladder
(does a new M_b add conditional information about held-out target views?). Adaptive GEPA stays
OUTSIDE the formal iid stream (already the doctrine: STREAM_EXCLUDE) and is a separate stress-test
list. Any future calibrated estimator needs a fresh LOCKBOX battery (realistic 600/300 regimes,
rare high-value species, shared proposer bias, drift, over-merging, degree-3/4 synergy), frozen
before development ends.

> **Correction (2026-07-12).** The historical section above is retained to show the route by which CR-3
> emerged. Its novelty-multiplied and CMI-head value conversions are not the current certificate, and its
> paper hierarchy mixes quantities that do not share a target or guaranteed ordering. The authoritative
> replacement below targets best single prompts directly, marks every audit draw against the exact pool
> optimum, and keeps operational `M_b` and ideal `M*` on separate ladders.

## 12.6b The executor-indexed prompt-articulation ceiling. **[authoritative; rewritten 2026-07-12; implemented in `experiments/cr_audit.py`]**

This section supersedes every earlier CR-2/CR-3 ceiling statement. Historical count, Hill, Good-Toulmin,
checklist-tail, and design-effect quantities remain descriptive; none participates in the certificate below.

> **Primary-object correction (2026-07-12, reconstruction restored).** The earlier rewrite below made the
> fixed-target channel `I(M_fixed;E(p,X))` primary. That admits the target-copy identity and is mathematically
> exact, but it bypasses the reconstruction experiment. It is retained in §A as an auxiliary theorem and as
> the value used by the already-running legacy CR-3 behavioral audit. It is **not** the primary definition of
> `M*_{b,E}` going forward. No oriented error, silver label, human label, archival outcome, or other external
> target is introduced by this correction.

### P. Primary object: anchor-free Reconstruction-MCQ prompt optimality

Fix metric identity `b`, executor `E`, reconstructor `W`, item distribution, readout, a frozen option
codebook `C_b` containing `b`, and a teaching-set design rule. A candidate prompt `p` may be any finite string
of any length. It generates only its own annotations:

```
Z_{E,p}(x) = readout(E(p,x)),
D_p        = Design({(x,Z_{E,p}(x))}, C_b),
q_p(j)     = W(j | D_p, C_b).
```

`W` sees the selected target `(item, annotation)` demonstrations and option descriptions. It never sees
distractor behavior vectors. Those vectors are used only on a disjoint design/calibration pool to ensure
that the demonstrations can separate the frozen options. Define the raw and annotation-attributable
per-metric reconstruction values

```
qbar_c,b,E(p) = E[q_c,p(b)]                                  for each condition c,
V_raw,b,E(p)  = qbar_annotations,b,E(p),
V_ann,b,E(p)  = [qbar_annotations,b,E(p)
                  - max{qbar_no-demo,b,E(p), qbar_shuffled,b,E(p)}]_+.
```

The expectation is over the predeclared teaching-set/reconstructor randomness; normalized choice logits
remove sampling variance when available. `V_ann` is primary for prompt optimization because it cannot reward
a target option that the reconstructor already prefers without annotations. Both values lie in `[0,1]` and
use no external labels: `b` is known by experimental construction.

```
V*_{b,E}   := sup_{p in Sigma*} V_ann,b,E(p),
p*_{b,E}   := any attaining prompt, when one exists,
M*_{b,E}   := Z_{E,p*_{b,E}}.
```

For one fixed `b`, target-option probability is a conditional reconstruction value, not mutual information
because `b` has zero entropy. Across an equal-weight randomized panel `B` of metric identities, the primary
bank-level information quantity is the stored selection channel `I(B;Bhat)`. This is how the work continues
to maximize MI without inventing anchor labels. Per-metric `V_ann` and bank-level identity MI must both be
reported; neither is the secondary canonical-body replay `I(Z_{E,p};Z_selected)`.

There is a sharper anchor-free all-prompt cap than one. Let
`c_0,b = qbar_no-demo,b(b)`. Because the option codebook, reconstructor, option counterbalancing, and
no-demonstration query are frozen before prompt search, `c_0,b` is independent of candidate `p`. Therefore

```
V_ann,b,E(p) <= 1 - c_0,b =: C_b                         for every p in Sigma*.
```

> **Theorem P.1 (frozen-control global certificate).** If the no-demonstration channel is frozen and
> candidate-independent, `[V_best,C_b]` is an identified interval for `V*_{b,E}` over all finite prompts,
> and `C_b-V_best` is a certified global optimization-gap upper bound for the declared Reconstruction-MCQ
> functional. Attaining `C_b` proves global optimality.
>
> **Proof.** Target-option probability is at most one, while the strongest-control term is at least the
> fixed no-demonstration probability `c_0,b`. Positive clipping preserves the inequality. The achieved
> value supplies the lower endpoint. No external label, prompt-length bound, proposer, or capture premise
> is used. ∎

This cap is partly instrument-defined, so raw target-option probability and `c_0,b` must be reported beside
the lift. Without verified executor structure, finite black-box prompt observations cannot lower `C_b`
further: an unseen string can attain target probability one and shuffled-control probability at most
`c_0,b` while preserving all queried results. CR-3 supplies additional, potentially tighter bounds for
(i) a declared finite future horizon under frozen proposer families and, with additional assumptions,
(ii) their support. It does not turn capture-recapture into an all-strings theorem.

**Why the range cap and capture-recapture are complementary.** The frozen-control inequality is a
distribution-free **range bound**: it uses only that a normalized target-option probability cannot exceed
one. Consequently it covers every `p in Sigma*`, including prompts that no mining process can generate,
but by itself says nothing about how quickly search approaches the cap. CR-3 is a **discovery bound**: fresh
captures estimate unseen behavior/value-state mass and the gain distribution under declared proposers
`Q_f`. It can show a tightening finite-horizon ceiling and, under an external positivity floor, exact
support exhaustion. It is more informative about search, but its information is weighted by proposal
probability. A zero-probability prompt is invisible; even if `Q` has full support on `Sigma*`, an arbitrarily
valuable prompt may have arbitrarily small mass. No common positive mass floor exists on an infinite
support. Therefore the valid combined reports are

```
all strings:                 V_best <= V*_{b,E} <= C_b,
fixed future Q-horizon m:    E[V_best after m] <= min(C_b, U_CR3,m),
support(Q), with positivity: sup_{p in Omega union support(Q)} V(p) <= min(C_b, U_CR3,support).
```

The second and third bounds may be much tighter than `C_b`, but their prompt-class qualifiers cannot be
dropped. Conversely, attaining `C_b` certifies all-string optimality even when behavioral discovery remains
unsaturated: many distinct behaviors may map to the same maximal reconstruction value.

**Codebook scope is part of the estimand.** An easy four-option panel can yield a valid but scientifically
weak all-string certificate because `C_b` and `V_ann` are defined for that panel. Production codebooks are
therefore frozen from a broader task-level candidate bank before target prompt search. A bank-only metric
contributes only its canonical executor behavior on the design panel; its historical prompt pool is not
automatically admitted into another target's search. Separately, prompts previously generated **for the
same target metric** may enter `Omega_N` as candidate-only evidence after they are content-validated,
rescored by the current executor namespace, and revalued under the current frozen codebook. They can raise
the achieved lower endpoint but can never be relabeled as fresh audit or confirmation draws.

For each target, the implementation first enumerates behaviorally hard non-clone panels using only the
design split. Before any candidate prompt is valued, it evaluates each unlabeled menu with the exact blind
no-demonstration query and counterbalanced option positions. A panel is headline-eligible only if the
predeclared full-posterior gates pass (default maximum option probability `0.35`, target probability within
`0.10` of chance, and normalized entropy at least `0.90`). Among passing panels it freezes the behaviorally
hardest; if none pass, it retains the least-violating panel only for a formally valid, explicitly
`FORMAL_CERTIFICATE_ONLY` estimand. The teaching-set optimizer then maximizes demonstrated separation and
label balance. The reconstructor still sees only option descriptions and the target prompt's
self-annotations. Changing the bank or selected panel changes the reconstruction instrument, so easy-panel
and hard-bank values must be reported as different estimands rather than pooled.

Formal validity and scientific informativeness are separate. A frozen codebook may give the target option
nearly unit no-demo probability, leaving a tiny `C_b`; attaining that cap is a correct global theorem but a
trivial articulation result. Likewise, a target with no behaviorally close bank member has an easy MCQ
panel. Certificates therefore retain the formal interval for every valid frozen instrument but mark a
result headline-eligible only when the predeclared minimum value headroom and minimum selected-distractor
kappa both pass. Raw target probability, the complete no-demo option prior, normalized prior entropy,
shuffled control, selected kappas, and directional disagreements remain visible. This quality gate changes
interpretation, not the inequality in Theorem P.1.

### A. Auxiliary fixed-target behavioral channel (historical identity theorem retained)

Fix a text distribution `D_X`, a frozen metric operationalization `M_b`, an executor `E`, and a fixed
verdict readout. Here **`b` indexes the operational target; it is not a prompt-length budget**. Prompts may
be arbitrarily long. The unrestricted class is the countably infinite set `P_infty = Sigma*` of all finite
strings. Executing one prompt `p` produces `Z_{E,p}(X)`. Define

```
R_{b,E}(p)       := I(M_b ; Z_{E,p})
A*_{b,E}         := sup_{p in Sigma*} R_{b,E}(p)        unrestricted promptable articulation value
p*_{b,E}         := any argmax, when the supremum is attained
M*_{b,E}         := Z_{E,p*_{b,E}}                     optimal induced metric behavior
T_b              := I(M_b;X)                           target-indexed channel/DPI cap
```

`M*_{b,E}` is executor-specific induced behavior, not the prompt-independent ideal `M*` of §12.6b-F.
Keeping the value `A*`, the prompt `p*`, and the induced behavior `M*_{b,E}` distinct avoids using
"optimal metric" for three different objects.

Because `M_b -> X -> Z_{E,p}`, every finite prompt obeys

```
R_{b,E}(p) <= A*_{b,E} <= T_b <= H(M_b).                (12.6b.1)
```

> **Theorem 12.6b.0 (unrestricted epsilon-global certificate).** For any evaluated prompt `p_hat`,
> `A*_{b,E} - R_{b,E}(p_hat) <= T_b - R_{b,E}(p_hat)`. Therefore `[R_best,T_b]` is an identified
> interval for `A*_{b,E}` and `T_b-R_best` is a certified upper bound on the global optimization gap
> over **all** finite prompts. If a candidate attains `T_b`, it is globally optimal. If the gap is at
> most a predeclared `epsilon`, it is `epsilon`-globally optimal.
>
> **Proof.** The lower endpoint is achieved by an evaluated prompt. The upper endpoint is (12.6b.1).
> Subtract the achieved value. No enumeration, prompt-length cap, proposer, or capture assumption is
> involved. ∎

For the implemented hard-verdict target, `M_b` is a deterministic function of `X` on the declared
channel, so `T_b=H(M_b)`. This can be loose, but it is a genuine all-prompt upper bound. Tightening this
certificate means raising the achieved lower endpoint toward the fixed upper endpoint; finite search data
cannot lower the endpoint below `T_b` without additional structure on `E`.

> **Corollary 12.6b.0a (target-copy identity).** Suppose one rubric `r_b` and the same frozen executor and
> hard readout define the operational target itself:
> `M_b(X)=binarize(E(r_b,X))`. Because `r_b in Sigma*`, choosing `p=r_b` gives
> `Z_{E,r_b}=M_b`, hence `R_{b,E}(r_b)=H(M_b)=T_b=A*_{b,E}`. The target-defining prompt is a provably
> globally optimal identity witness, on the whole declared item distribution, regardless of its length.

This result is exact but scientifically narrow. It certifies **self-reproduction of an operationalization**;
it does not show that a learner can infer the rubric from annotations, that atomic units can express it,
or that `M_b` equals the prompt-independent ideal. It also does not automatically apply when `M_b` is a
stochastic verdict and the candidate is an independent sampled verdict, or when `M_b` is a thresholded
average of several form-orbit channels: no single constituent prompt need reproduce that aggregate.

**Polarity is part of annotation fidelity.** For a nonconstant binary hard target and binary candidate,
`I(M_b;Z)=H(M_b)` iff `H(M_b|Z)=0`, which means `Z=M_b` **or** `Z=1-M_b` almost surely. The second channel
is information-perfect but reverses YES and NO. The unrestricted MI optimum legitimately treats it as
sufficient; a metric-annotation claim must not. Certificate payloads therefore report `EXACT_MATCH`,
`EXACT_COMPLEMENT`, or `MIXED_ERRORS` separately. The structural target-copy corollary is an exact match.
Any optional polarity flip for another candidate must be chosen on calibration data and frozen before the
lockbox; choosing it from lockbox outcomes invalidates the population certificate.

> **Scope correction.** This polarity/oriented-error paragraph belongs only to the auxiliary fixed-target
> behavioral channel. The primary Reconstruction-MCQ objective in §P neither orients against an external
> target nor consumes anchor labels; it evaluates the planted metric identity from prompt self-annotations.

> **Proposition 12.6b.0b (finite black-box no-free-lunch).** Let an audit query any finite set of prompts
> `S subset Sigma*`. Unless the executor class imposes additional verified structure, no rule based only
> on those observations can issue a universally valid all-prompt upper bound strictly below `T_b`.
>
> **Proof.** Choose an unqueried prompt `p_0`. An alternative executor can agree with every observed
> response on `S` and make `Z_{E,p_0}` a sufficient readout of `M_b`, attaining `T_b`. The audit cannot
> distinguish the two executors. ∎

This is the precise boundary for capture-recapture. A frozen proposer `Q` can certify a finite future
mining horizon, and an external positivity floor can sometimes exhaust `support(Q)` on a fixed panel. It
cannot certify `Sigma*`: full support alone is insufficient because a target-perfect prompt may have
arbitrarily small proposal mass, and a common positive lower bound is incompatible with an infinite
support. CR-3 is therefore an auxiliary **search-process** certificate, not the unrestricted upper bound.

The code now reports these objects separately. `all_finite_prompt_dpi_certificate` issues (12.6b.1), the
achieved lower endpoint, and the exact gap status. `prompt_articulation_certificate` issues the prospective
`Omega_N union support(Q)` results in §§C-D. For one-form production targets, the canonical target-form is
evaluated as a separate identity witness even though the atomic mining pool excludes it; this preserves the
unit-decomposition experiment without misreporting it as the unrestricted optimum.

> **Theorem 12.6b.0c (fresh-lockbox population gap).** Let a hard binary target and one candidate prompt,
> including its YES/NO orientation, be frozen before observing `n` iid lockbox items. Let `p=P(M_b=1)`
> and `e=P(Z != M_b)`. Construct a simultaneous exact confidence interval `[p_L,p_U]` and one-sided upper
> bound `e_U`. Then, with the stated simultaneous confidence,
>
> ```
> T_b <= H_U := max_{p in [p_L,p_U]} h(p),
> R_{b,E}(p_hat) >= H_L - h_U(e),
> H_L := min_{p in [p_L,p_U]} h(p),
> h_U(e) := max_{0 <= q <= e_U} h(q),
> A*_{b,E} - R_{b,E}(p_hat) <= min{H_U,h_U(e)}.
> ```
>
> **Proof.** The deterministic target gives `T_b=H(M_b)=h(p)`. For binary variables, the error bit
> `D=1[Z != M_b]` and `Z` determine `M_b`, so
> `H(M_b|Z) <= H(D) = h(e)`. More directly,
> `A*_{b,E}-R(p_hat) <= T_b-R(p_hat) = H(M_b|Z) <= h(e)`. Optimize over the simultaneous confidence
> set and cap by `H_U`. The prevalence interval is needed for the value interval `[R_L,H_U]`, but it
> does **not** inflate the optimization-gap bound. ∎

This theorem turns lockbox size into a predictable tightening mechanism after prompt mining. It does not
pretend that a zero-error finite panel proves population equality: the exact binomial `e_U` remains positive.
Only the constructional one-form identity has a zero population gap without sampling. The implementation is
`all_finite_prompt_population_certificate`; it fails closed unless candidate, threshold, and orientation are
declared frozen.

For a 95% simultaneous certificate using the implementation's half-alpha error allocation, the prospective
zero-error requirements are: 283 lockbox items for `epsilon=0.10` bits, 657 for `0.05`, 1,931 for `0.02`,
4,287 for `0.01`, and 9,413 for `0.005`. These are minimums conditional on observing zero errors, not promised
outcomes. `zero_error_lockbox_plan` computes the exact design before data collection; observed errors are fed
back through Theorem 12.6b.0c rather than waved away.

**Search consequence for aggregate targets.** Mining remains useful, but its role is now exact: it raises
the achieved lower bound. For a hard target, optimize the candidate's oriented disagreement or a proper soft
loss on discovery/calibration items. Do not optimize MI alone, because an exact complement maximizes MI.
Freeze one prompt, readout threshold, and orientation, then invoke the untouched lockbox once. A failed
lockbox is a valid unresolved result, not reusable development data; continued optimization needs a new
lockbox (or a prospectively alpha-spent sequential design).

> **Primary-method correction.** The preceding oriented-disagreement recommendation is retained only with
> the auxiliary fixed-target theorem. It is not used to optimize `p*_{b,E}` in §P. The current primary
> optimizer sees only Reconstruction-MCQ values derived from prompt self-annotations and their no-demo /
> shuffled controls.

The implemented recovery is Shannon MI between the hard target verdict and `E`'s hard verdict on a frozen
empirical probe distribution. The functional is exact on that finite distribution. A non-identity numerical
gap requires a fresh iid probe lockbox and simultaneous confidence bounds before it is a population claim.
The identity corollary is different: when target and candidate are literally the same prompt/executor/readout
construction, equality holds wherever that operational definition is used, rather than by panel
generalization.

### B. Discovery pool and exact within-pool optimum

Let `Omega_N` be any frozen discovery pool, including prompts obtained adaptively before the audit.
For the primary reconstruction objective, every pool prompt receives `V_ann,b,E(p)` under the same frozen
MCQ protocol. For the auxiliary fixed-target objective, substitute `R_E(p;M)`. For a single-prompt class
there is no combinatorial `U_Omega` problem:

```
V_Omega := max_{p in Omega_N} V(p).
```

On the fixed panel this is the exact within-pool optimum. It is not an upper bound outside `Omega_N`.
This is why the certificate now targets single-prompt articulation directly rather than routing value
through a greedy atomic-unit checklist and an unproved submodularity bridge.

### C. Fresh audit and the finite-horizon theorem

The audit contains independent prompts within each predeclared proposer family `f`. Distinct requests
use distinct logged seeds; deterministic rejection sampling retains the first valid draws, so the
accepted stream is iid from the frozen family distribution conditional on validity. Before mining, the
target and every prompt in `Omega_N` are rescored on one ordered, hashed panel. Each prompt then has one
content-addressed executor signature, keyed by panel, executor revision, readout protocol, and text;
recaptures in later worker processes reuse that exact signature.

The family index includes both model and proposal mode. `atomic` draws one short checkable criterion and is
useful for behavior-unit discovery. `holistic` draws a standalone multi-sentence rubric intended to cover
the full metric, including tradeoffs, exclusions, and edge cases. These streams are never pooled under one
iid label: use tags such as `phi4_atomic` and `phi4_holistic`. This matters because an atomic-process plateau
with a large all-prompt gap is evidence about that restricted search distribution, not about unrestricted
`Sigma*`. The all-string theorem remains length-unrestricted; operational proposer modes have declared
generation limits and may be broadened with additional length/mode strata without changing Theorem P.1.

For an audit prompt define the bounded all-draw gain for either declared value functional

```
G(p) := max{0, V(p) - V_Omega}  in [0, B],
B    := value_cap - V_Omega.
```

Let `mu_f = E_Q[G | f]`. A one-sided empirical-Bernstein bound gives simultaneous `mu_f <= U_f`
after Bonferroni allocation across families. Independently within the same preallocated gain-claim
budget, Dvoretzky-Kiefer-Wolfowitz gives a simultaneous lower envelope
`L_f(t) = max{0, Fhat_f(t)-epsilon_f}` for every gain CDF. For a predeclared future design containing
`m_f` draws from each family, define

```
A_{E,Q,m}(V | Omega_N)
  := E[ max( V_Omega, V(P_{f,j}) for f,j ) ].
```

> **Theorem 12.6b.1 (finite-horizon expected prompt ceiling).** On the simultaneous gain-confidence
> event,
> `A_{E,Q,m}(V | Omega_N) <= V_Omega + min{B, U_sum, U_DKW}`, where
> `U_sum = sum_f m_f U_f` and
> `U_DKW = integral_0^B [1 - product_f L_f(t)^(m_f)] dt`.
>
> **Proof.** First, pointwise,
> `max_{f,j} G(P_{f,j}) <= sum_{f,j} G(P_{f,j})`. Taking expectations gives
> `E[max G] <= sum_f m_f mu_f <= U_sum`. Second, independence of future draws gives
> `P(max G <= t) = product_f F_f(t)^(m_f) >= product_f L_f(t)^(m_f)` uniformly in `t`.
> Integrating the complementary CDF gives `E[max G] <= U_DKW`. Take their simultaneous minimum and
> cap by the residual `B = value_cap-V_Omega`. ∎

This theorem has no species rule, substitutability premise, submodularity ratio, or synergy escape:
the object is the best **one** prompt. It is an upper bound on an expectation over a declared finite
future mining budget, not a high-probability bound on the realized maximum and not an all-support
supremum. Those are different estimands and are never conflated.

#### C.1 Value weighting: what is certified and what is not

The historical `alpha_V` curve weights discovered species by realized value and remains useful for asking
whether cumulative value grows more slowly than behavioral richness. It is not a ceiling: singleton
degeneracy, same-sample value assignment, redundancy, and synergy prevent any transformation or
"upweighting" of `alpha_V` from bounding an unseen optimum.

The bound-grade value weighting is instead the mark `G(p)` on **every** fresh audit draw. This produces the
desired asymmetry without a heuristic coefficient: one hundred novel behaviors with zero improvement add
zero gain, while one rare prompt with gain `g` contributes the full bounded mark `g` and prevents a false
plateau. Behavioral missing mass and the value-gain ceiling must therefore be plotted and classified as
separate axes. A metric may validly remain behaviorally `UNSATURATED` while becoming value-`PLATEAUED`.

Search may be made more value-sensitive by adding a separately named, prospectively frozen
**value-tilted proposer family** (for example, mutations or compositions of high-value discovery prompts).
Its model, conditioning data, sampling law, future quota, and family tag must be frozen after a design
stage and before its confirmation audit. The family then receives its own empirical-Bernstein and DKW
components. Post-hoc weights based on observed gains are forbidden. The value-tilted family can raise
`V_Omega` faster and, after improvements are absorbed and a new audit is drawn, shrink the residual
`B=C_b-V_Omega`; it does not broaden a process-relative theorem to all strings.

### D. Capture mass and exact-support exhaustion

Freeze a leader classifier from `Omega_N`. For each family, audit novelty is an iid Bernoulli event;
one-sided Clopper-Pearson bounds and the frozen family weights give a valid upper bound on
classifier-relative missing mass. This quantity answers only, "how often will this classifier call a
fresh draw novel?" Fuzzy leader balls do not form a partition and support-size algebra is forbidden.

Alongside it, map every prompt to its exact binarized executor pattern on the frozen panel. Exact
patterns form a genuine partition. Let `U_exact` be the simultaneous upper bound on the proposer-mixture
mass of exact patterns absent from `Omega_N`.

> **Corollary 12.6b.2 (support exhaustion under positivity).** Assume externally that every exact
> pattern in the support of `Q` has mixture probability at least `p_min > 0`. If
> `U_exact < p_min`, then no exact pattern is missing. More generally the number of missing patterns is
> at most `floor(U_exact/p_min)`. If, additionally, value is proven to be a function of the exact behavior
> pattern, then the support-wide value equals `V_Omega`.
>
> **Proof.** Any missing support pattern contributes at least `p_min` to missing mass. One would imply
> missing mass `>= p_min`, contradicting `U_exact < p_min`. The value conclusion follows only under the
> additional value-as-a-function-of-pattern premise. ∎

`p_min` is an identifying assumption, never estimated from the same capture data. If it is absent or
the inequality fails, the all-support ceiling remains the declared value cap. Low fuzzy novelty, zero observed
novelties, a small Hill estimate, or a flat monitor trajectory cannot replace positivity.

The full executor pattern is often much finer than the reconstruction functional. Define the exact
**value state** `S_V(p)` as the frozen teaching transcript actually shown to the reconstructor (ordered item
identities plus hard annotations) inside one codebook/readout/query-cache namespace. The repaired instrument
constructively enforces `V(p)=g(S_V(p))`. Apply the same fresh-audit Clopper-Pearson calculation to the event
`S_V(P) notin S_V(Omega_N)`, yielding `U_value`.

> **Corollary 12.6b.2a (value-state support exhaustion).** If every value state in `support(Q)` has
> proposer-mixture mass at least an externally declared `p_min,value>0`, then
> `U_value < p_min,value` certifies that every proposer-support value is represented in the pool and hence
> `sup_{p in Omega_N union support(Q)} V(p)=V_Omega`.
>
> This is not approximate fuzzy clustering. It is an exact quotient of the implemented value functional,
> uses a separate positivity assumption from behavior-pattern support, and fails closed if one transcript
> ever maps to two stored values.

For legacy fixed-target behavioral MI, exact executor behavior determines value. The repaired production
Reconstruction-MCQ logit path now has the same invariance on the frozen panel: candidate prompt text is not
shown to the reconstructor; teaching-set selection uses only hard annotation behavior and a fixed item-order
tie break; and example order is derived from the selected annotation transcript. GPU kernels can still vary
at the last floating-point bits across processes, so "temperature zero" alone is **not** treated as exact
determinism. The production path content-addresses every rendered MCQ query, seed, option vocabulary, and
reconstructor revision and reuses the first finite probability row; it also freezes the no-demonstration
channel once before mining. The implementation validates that repeated exact patterns have identical values
before asserting the premise. Thus exact behavior-support exhaustion can also imply the Reconstruction-MCQ
proposer-support value ceiling on this path. A sampled/nondeterministic fallback does **not** receive that
promotion unless it separately establishes the same invariance. The all-draw finite-horizon gain theorem
remains valid in either case.

### E. Simultaneous confidence, evolution status, and production protocol

The v5 payload first splits total `alpha` equally between two jointly reported bundles. The primary
upper bundle contains leader-classifier missing mass, exact-pattern missing mass, optional exact-value-state
missing mass, and gain (including its horizon consequence). Its gain share is split again between empirical-Bernstein and DKW
before taking their minimum. Each method then Bonferroni-allocates across proposer families. Lower mass
bounds and stricter-threshold variants are labeled marginal diagnostics and are not silently included in
that upper bundle.

The other half of total `alpha` constructs a simultaneous **status evidence** bundle containing a two-sided
behavioral-mass interval and lower/upper finite-horizon gain evidence. The horizon lower endpoint is the
maximum of a per-family mean-gain LCB and a DKW expected-maximum LCB, so repeated low-probability gains can
certify `RISING`; the upper endpoint is the minimum of the mean-sum and DKW bounds. Thus the primary upper claims and
the status labels are joint at the declared overall confidence, rather than each silently spending the
full error budget. On a never-absorbed confirmation only:

```
behavior = SATURATED   if U_mass <= delta
         = UNSATURATED if L_mass >  delta
         = UNRESOLVED  otherwise

value    = PLATEAUED   if U_gain <= epsilon
         = RISING      if L_gain >  epsilon
         = UNRESOLVED  otherwise.
```

Patience, a flat monitor, or `max_iter` never implies `PLATEAUED`. These labels are prompt evolution at one
fixed executor and declared proposer horizon; OSL executor-scaling labels have not begun.

The production protocol is part of the theorem's premise:

1. freeze the metric description, `E`, ordered probes, proposer configurations, family weights, validity
   predicate, horizon, thresholds, and alpha;
2. bootstrap `M` and all initial-pool prompts through the same seeded readout and immutable signature cache;
3. generate each proposal occasion with a distinct logged seed and require exact family quotas;
4. hash the actual ordered probes, source checkpoint, descriptions, model revisions, readout template, and code;
   fail closed if any namespace component changes or any score is nonfinite;
5. score duplicate prompt texts once and reuse the content-addressed signature across worker processes;
6. absorb monitor batches only through an ordered fsynced ledger;
7. after optional stopping, draw a separately seeded immutable confirmation audit that the ledger loader
   cannot read.

For a certified tightening curve, predeclare checkpoint pool sizes. Each checkpoint gets its own separately
seeded never-absorbed audit before further monitoring. Alpha is allocated across checkpoint plus final cells;
an optional study-level allocation additionally covers every declared metric. Adaptive monitor rows may be
plotted as diagnostics but never appear as certified trajectory points.

**Prospective 95%/90% reporting rule (declared 2026-07-12 before the v11 checkpoint and confirmation
audits).** The scientific certificate remains the 95% simultaneous interval. The identical frozen audit may
also be evaluated at 90% confidence as a secondary sensitivity analysis. Results passing only the latter are
`SUGGESTIVE`, not `CERTIFIED`; the report shows both intervals for every metric and may not choose the level
after seeing the metric. This tiering broadens the descriptive status spectrum without weakening the theorem
or spending additional GPU observations.

The July-11 live run violated item 2: a shared vLLM seed produced one prompt repeated 150 times in
almost every family file, with some empty files. It was terminated and none of its monitor or future
confirmation artifacts are evidence. Legacy pilot-v2 intervals remain arithmetic sensitivities
conditional on iid provenance that those artifacts do not establish.

### F. Fixed operational target versus ideal metric

> **Deferred scope.** This subsection concerns the later executor-independent / latent-ideal problem. It is
> not part of the current `M*_{b,E}` work, and no archival, silver, human, or other external view is assumed
> available. The current optimizer and every certificate above use reconstruction from prompt
> self-annotations only.

For the auxiliary fixed-target channel, `M=M_b` is a fixed operationalization and the theorem above answers
how much of that frozen behavior another prompt can reproduce. `M_omega` is one such `M_b`; its entropy is a
valid fixed-target DPI cap, not a claim about the primary Reconstruction-MCQ optimum or the ideal concept.

Metric refinement changes the target and cannot be evaluated by self-recovery. Let `M*` denote the
prompt-independent ideal. Then separately

```
A_E(M*;P) <= T*(M*) := I(M*;X) <= H(M*).
```

Capture-recapture over prompt behaviors cannot identify `M*`. Point identification by a latent-class
model needs at least three conditionally independent, rank-sufficient views; correlated model-family
operationalizations do not satisfy that premise merely by having different names. Otherwise construct
an explicit confidence set `C(M*)` under declared local-dependence constraints and report the robust
supremum over `M in C`. Archival outcomes are proxy views only after a measurement relation to the same
latent metric is specified. There is therefore no mixed ladder ordering CR-3 mass, `M_b` recovery, and
an `M*` identified-set bound: each inequality must keep one target, prompt class, executor, and unit.

### G. Status of older instruments

- `cr_horizon.py` / CR-2: descriptive saturation visualization only.
- `value_census.py`: descriptive alpha-value, singleton-flux, conditional-greedy, and Hill diagnostics;
  it issues no saturation verdict because its species/value map is same-sample and MI need not be
  submodular.
- `value_certificate.py`: historical checklist diagnostics with `upper_bound_valid=False`; no output
  enters the prompt-ceiling theorem.
- GEPA: an achieved prompt generator/stress test. A GEPA policy may become a proposer family only after
  it is trained, frozen, and given a fresh audit.

## 12.7 Do we need V-information? The executor-indexing note. **[new, 2026-07-01]**

*Consolidates the scattered V-information statements (§1 note, §6 division-of-labor, §6.10, the proof-core
companion) into one referenceable answer: **as a formalism, no — deliberately; as a concept, yes — and it is
implemented in the channel, not the entropy.***

**The rule.** Every quantity in this theory is **ordinary Shannon/TVD information computed on
executor-produced random variables**. Executor-boundedness — the thing V-information is *for* — enters
through *which variables exist*: `B_E` is the set of behaviors `E` can realize, `m_ω`/`m̄_ω` is `E`'s
verdict, `M̂` is `E` executing the reconstruction. Once those variables exist, the information measure on
them is plain Shannon/TVD — so every proof inherits DPI, convexity, and the chain rule, the three properties
the Xu-et-al. functional lacks. "The `V` lives in the channel `s`, not in the entropy" (§1).

**Why the functional would actively break the theory (not just complicate it).** The *defining feature* of
predictive V-information is that processing can CREATE usable information (decryption creates V-usable
information for a class that cannot read ciphertext). Our load-bearing inequality `R ≤ T` is exactly a
data-processing statement, and the reconstructor IS a processor — under V-information the
articulate-then-re-execute loop could legitimately exceed the "ceiling," and the certificate would be
vacuous. This is a population fact; no `O(√(d/N))` SLT term repairs it (§6). Secondary kills: no chain rule
⇒ the §6.5 orthogonalization filter must be Shannon-CMI (as coded); non-Lipschitz family-dependence ⇒ the
§12.6.5 no-extrapolation rule; noise-can-raise-`I_V` ⇒ uninterpretable gaps.

**Where the executor-boundedness actually lives (the concept, kept):**

| the V-info *idea* | structural implementation here | why better |
|---|---|---|
| "usable by class V" | `B_E` = image of prompt→behavior for `E` | downstream MI is Shannon ⇒ DPI holds |
| class-relative ceiling | `T(m_ω) = I(M_ω;X)` — *this* executor's transmission | measurable, certifiable, per-executor |
| grading by class strength | §5.5 ladder (compiler ⊂ LLM ⊂ dense), same `R`, widening `E` | V/A/Taste = differences of ONE Shannon quantity |
| class capacity | `OPT′` (§11.6) — decoder-class ceiling | explicit about which class; no functional |
| marginal usable info of a criterion | `v(s\|S_g)` (§12.6) — marginal recovery gain | a Shannon MI difference; feeds the ε-gap |
| sup over classes | §12.6.5 family-scaling flatness verdict | handles the class-sup without extrapolation |

For the anthropological estimand this indexing is not incidental: "articulable" is always "articulable **to
a reader class**," and the incommunicability claim is quantified over the `E`-ladder via the flatness
verdict.

**The three places the functional legitimately remains — all optional.** (1) The §6 attribution gloss
(staircase increments read as "marginal V-usable information") — interpretation only; the numbers are
Shannon MI differences. (2) The §6.10 ignorability-by-construction argument — a *definitional* exclusion
(the channel class cannot represent the confound), using no V-info theorem; keep. (3) Predictive complexity
`C_α` ("smallest rubric reaching agreement α") as the pathology-free reviewer flank for the L-axis.

**Naming (practical).** The code module `vinfo.py` and the metric `iv_transmission` compute **plug-in
Shannon MI** — the name is historical. In papers, write **executor-indexed (Shannon) information** /
class-indexed transmission, cite Xu et al. once as the conceptual ancestor of executor-relativity, and state
why the functional is unusable for a ceiling (this section). This prevents reviewers from pattern-matching
our quantities to the V-information pathologies and demanding a defense of theorems we never use.

## 12.8 The provable core — soundness theorem, lemmas with proofs, imports, conjectures, and the assumption ledger. **[new, 2026-07-01]**

*Purpose: upgrade the operating disciplines of §12.6 from convention to theorem wherever a short proof
exists, fix the soundness gaps found in the ε-band on the way, and name — as formal conjectures with
designated breakers — the assumptions that cannot be theorems. Each item states which standing rule it
converts. Notation as in §12.6: species partition `P` of the frozen-iid criterion stream, counts `n_s`,
spectrum `w_j = Σ_{n_s=j} v(s|S_g)`, `f_j` the count spectrum, head `S_g` (partition-free), combiner
class `F`, `V(S) = sup_{f∈F} I(f(u_S(X)); M)`, `OPT_F(Ω) = sup_S V(S)`.*

### 12.8.0 Theorem T1 — a conditional anytime flux band; current pipeline does not meet its freezing premise. **[corrected 2026-07-10]**

The pre-2026-07-01 band `flux_hi = w₁/N + B√(2·ln(1/δ)/N) + B/N` (D2) had three genuine invalidities —
not formalization gaps, *wrong coverage*:

- **H1 — optional continuation.** The §12.6.6 loop *invites* adaptive re-sampling (UNDERSAMPLED → draw
  more frozen-iid criteria → re-issue). A fixed-N McDiarmid band read at a data-chosen N is optional
  stopping; the realized level is not δ.
- **H2 — the order-max without simultaneity.** `eps_bits_adv = max_k ε_k` over the canonical partition
  + `n_orders` permutations is a claim about all `n_orders+1` bands AT ONCE; with per-statistic level δ
  it holds only at level `(n_orders+1)·δ`.
- **H3 — an anti-conservative McDiarmid constant.** The code used `B = max observed species value`. The
  bounded-differences constant must cover every *realizable* coordinate change: an UNSEEN species can
  carry conditional value up to the head's residual entropy `H(M|S_g)`. The empirical max is a valid
  constant only by luck.

> **Theorem T1 (time-uniform, simultaneous flux band under external freezing).** Independently freeze the
> species map, head, and species-value function before observing the iid capture stream. Fix the deterministic
> checkpoint grid `N_j = n0·2^{j−1}` and allocate `δ_j = δ/(j(j+1))` (so `Σ_j δ_j = δ`, telescoping).
> Let `B_cap = H(M)`, a cap fixed before selection.
> Under A1 (frozen-iid stream), with probability ≥ 1−δ, **simultaneously for every checkpoint `j` and
> every order statistic `k ∈ {0,…,n_orders}`**:
> `E[Φ_V^{(k)}(N_j−1)] ≤ w₁^{(k)}/N_j + B_cap·√(2·ln((n_orders+1)/δ_j)/N_j) + B_cap/N_j`.
> Consequently ANY adaptive rule that continues, stops, or re-issues at checkpoints — including reading
> the adverse max over `k` — inherits a valid 1−δ band.
>
> *Proof.* Each `(j,k)` cell is the D2 bound at level `δ_j/(n_orders+1)`: `w₁^{(k)}/N` is a function of
> `N` iid draws in which changing one draw moves at most two species across the singleton boundary, each
> shifting the statistic by ≤ `B_cap/N`, so McDiarmid applies with `c_i = 2·B_cap/N`; L3 links the mean to
> the expected flux one step back, which dominates the current flux. Union over the `(j,k)` grid spends
> `Σ_j δ_j = δ`. The checkpoint index `j(N) = ⌊log₂(max(N,n0)/n0)⌋+1` is a function of N ALONE, so the
> grid — not the stopping rule — determines which cells are read: adaptivity cannot select an uncovered
> cell. ∎

The implemented pipeline selects the head and estimates species values using the same realized pool. Changing
one capture draw can therefore change the head and many downstream values, not only two singleton indicators.
Thus T1 validates the error-spending pattern for a future split-sample implementation; it does not validate the
current epsilon band.
>
> **Cost.** `ln(1/δ) → ln(1/δ) + 2·ln(j+1) + ln(n_orders+1)` — the iterated-logarithm-style price;
> at `N = 1024`, `n0 = 16`, `n_orders = 8`, δ = .05 the slack multiplier is ≈ 1.6× vs the (invalid)
> naive band. **Scope note:** the band is stated at the MEAN level (estimator vs expected flux) plus L3;
> concentration of the *realized* missing value mass around its mean is standard (McAllester–Schapire
> 2000; Berend–Kontorovich 2013) and adds a same-order term — cited, not re-derived, and no constants
> are invented in code.

**Implementation status.** `anytime_delta()` correctly computes the intended error allocation
`δ_eff =
δ/(j(j+1)·(n_orders+1))`; `certificate(..., stopping='anytime')` is the DEFAULT and reports
`stopping / checkpoint / delta_effective / n_union / B_flux_cap / resid_cap`; `stopping='fixed'`
(declared-budget, single issuance) skips only the checkpoint union — the order union and the B-cap
apply in both regimes. This bookkeeping is necessary but not sufficient because the external-freezing premise,
the horizon deviation result, and the unseen-synergy lower bound are absent. Historical planted worlds were tuned
to the band, but passing them did not establish coverage:
(a) the codifiable positive control now keeps its i=0 rule copy noiseless — verdict noise of rate γ on
the best criterion puts `h₂(γ)` bits into `H(M|S_g) = B_cap`, and at small N nothing rules out an
unseen species carrying that much value, so 0.05·H certification is rightly impossible there (that
robustness question belongs to L6, not to this control); (b) distractors are recaptured (two copies) —
a *positive-control* world is a saturated process by definition. **Remark (the head is also the
variance reducer):** after a head is frozen independently, its residual entropy can reduce pointwise value
range. It cannot be substituted as an a-priori concentration cap when the same data selected that head.

### 12.8.1 Lemma L1 — refinement safety ("when in doubt, split" is a theorem). **[derived]**

> **(i) Count part (exact, unconditional).** If `P′` refines `P`, then on every realization
> `f₁(P′) ≥ f₁(P)` and `w₁(P′) ≥ w₁(P)` (values ≥ 0, head fixed), with N unchanged — so the Robbins
> flux estimate and the D2 band are nondecreasing under refinement.
> *Proof.* A `P`-species with count 1 sends its single occurrence to exactly one `P′`-part, which is
> then a singleton with the same representative row (majority of one member) and the same value —
> singletons are preserved injectively. A `P`-species with count ≥ 2 splits its occurrences into
> positive parts, creating ≥ 0 new singletons (each adding ≥ 0 value mass) and destroying none. ∎
>
> **(ii) Validity under over-split (conditional on part-subadditivity).** A certificate computed on a
> refinement remains a valid upper bound for the coarser estimand, provided a merged species' value is
> ≤ the sum of its parts' values (`v(merged|S_g) ≤ Σ v(part|S_g)` — parts are paraphrases of one
> concept; the merged representative is a garbling of the parts, so this fails only under synergy
> *among parts of the same concept*, which the paraphrase planted controls test directly).
>
> **(iii) Over-merge is the sole anti-conservative direction, by two independent mechanisms.**
> (a) Merging two count-1 species destroys two singletons → `w₁` strictly drops → the flux
> underestimates and ε can under-cover. (b) The target itself moves: a coarse unit's verdict is a
> function of its fine parts' verdicts, so any checklist over coarse units is simulable over fine ones
> — `OPT_F(fine) ≥ OPT_F(coarse)` — hence an over-merged certificate bounds a *smaller class* than the
> claim quantifies over. Both push the same way. ∎

This converts the §12.6.2 asymmetry argument and the standing "over-merge is the one anti-conservative
partition failure" from discipline to theorem, and grounds the merge-precision gate as the binding one.
(The horizon leg `Ĝ` is not monotone under refinement — alternating terms move both ways — so partition
ambiguity of the horizon is covered *empirically* by the §12.6.6 order-adverse band, not by this lemma.)

### 12.8.2 Lemma L2 — the certified target is well-posed. **[derived]**

> If `F` is closed under adding ignored inputs (all implemented combiners are: a weight-0 extension
> exists in F₁ and F₂), then `S ⊆ S′ ⇒ V(S′) ≥ V(S)`; hence `OPT_F(Ω) = sup_{finite S} V(S)` is a
> monotone, well-defined supremum bounded by `H(M)`, and the measured head value `V(S_g)` is always a
> valid LOWER bound on it. ∎

Trivial, but it is the statement that makes "OPT_Ω + ε ≥ OPT_F" a bound on a well-defined object —
worth one line in any writeup before ε is introduced.

### 12.8.3 Lemma L3 — the flux estimator over-covers by construction (Robbins direction). **[derived]**

> Under A1, `E[w₁(N)/N] = Σ_s v_s·p_s(1−p_s)^{N−1} = E[Φ_V(N−1)] ≥ E[Φ_V(N)]` (each term monotone in
> the exponent; counts: the same with `v ≡ 1`). The plug-in flux is *exactly unbiased one draw back*
> and *upward-biased for the current flux*. ∎

D1 derived the (N+1)-shifted identity; L3 states the direction that matters for a certificate: the
estimator's bias points the way an upper bound must point. Say it this way in papers — "conservative by
construction," not "approximately unbiased."

### 12.8.4 Lemma L4 — the model-free envelope chain (count flux is the fallback). **[derived]**

> With a head frozen independently and values in `[0, cap]`: `w₁ ≤ cap·f₁`. The universal pre-selection
> cap is `H(M)`; `H(M)-OPT_Ω` is not an a-priori cap when the same data selected `OPT_Ω`. Thus
> `H(M)·f₁/N` is the conservative envelope under the fixed-value premise;
> per-singleton plug-in value weighting (each singleton's value estimated from its one occurrence) is a
> sharpening *inside* the envelope. If the value model is in doubt, the envelope stands. ∎
> The cond-vs-add comparison (`w₁_cond ≤ w₁_add`) is NOT free — it is the submodularity direction,
> monitored by γ̂, and can fail under head-synergy (a unit worthless marginally, valuable given `S_g`).

Report both reads (the code does); cite the envelope when a reviewer questions the singleton value
estimates.

### 12.8.5 Lemma L5 — combiner separation is total in the worst case. **[derived]**

> Let `U₁,…,U_k` be iid uniform ±1 unit verdicts and `M = Π U_i` (parity). Any `f` measurable in a
> proper subset of the units is independent of `M` (parity of iid uniform bits ⊥ every proper subset),
> so `I(f;M) = 0`; the full parity, available one class up, achieves `H(M) = 1` bit. Hence for every
> `k`: `sup over verdict distributions [OPT_{F_k} − OPT_{F_{k−1}}] = H(M)`. ∎
> **Corollary.** For any FINITE combiner class `F` there exists a metric with `OPT_F = 0` and
> checklist-articulable value `H(M)` one class up — no F-free checklist ceiling exists; every
> certificate must name its F (the §12.6.4 re-scope is *forced*, not cautious). **Caveat:** this is
> worst-case existence — on correlated verdict pools partial parity can leak into lower classes, so the
> lemma licenses the F-indexing rule, not a prediction that F₁ misses everything conjunctive.

### 12.8.6 Lemma L6 — judge noise: attenuation is provable, deconvolution needs calibration. **[derived + design]**

> **(i) Attenuation (SDPI — the free direction).** Under the chain `f — M — Ĵ` (judge noise ⊥ checklist
> given truth), `I(f;Ĵ) ≤ η(P_{Ĵ|M})·I(f;M)` with contraction coefficient η ≤ 1: measured value
> UNDERSTATES true value, `I(f;M) ≥ I(f;Ĵ)/η`. Safe for head claims ("truly at least this much");
> it does NOT bound the ceiling against the true target.
> **(ii) No free reverse.** A constant judge (η = 0) gives `v_meas ≡ 0` for every `f` while `v_true`
> ranges over `[0, H(M)]` — no function of measured values upper-bounds true value. Generic contraction
> cannot close this; only a noise MODEL can.
> **(iii) Calibrated reverse (binary symmetric).** If `Ĵ = M ⊕ Z`, `Z ~ Bern(γ) ⊥ (f,M)`, then
> `P(f=Ĵ) − ½ = (1−2γ)·(P(f=M) − ½)` EXACTLY (*proof:* condition on `Z`), so with a calibration upper
> bound `γ ≤ γ̂ < ½` from planted/known-truth items, agreement-scale values deconvolve exactly and
> MI-scale values obey `I(f;M) ≲ I(f;Ĵ)/(1−2γ̂)²` (quadratic MI–correlation relation near
> independence). ∎
> **(iv) Escape E4 — uncalibrated judge noise.** Direction-of-error flip: unmodeled judge noise
> DEFLATES `OPT_Ω + ε` and so INFLATES `Δ(E) = lowerCI(C) − [OPT_Ω+ε]` — the thesis-friendly direction.
> The burden is therefore the calibration bound γ̂, carried by known-truth items in the probe set;
> `methods/codability/decompose.attenuation_correct` is the codability-layer instance of exactly this.

### 12.8.7 Imports I1–I3 — known theorems that slot in with a citation. **[imported]**

- **I1 (OSW horizon point predictor) — IMPLEMENTED, certificate interpretation corrected 2026-07-10.**
  Orlitsky–Suresh–Wu 2016 proves normalized-MSE guarantees for unseen-species prediction to horizons on
  the order of `log N`, with impossibility results farther out. The implemented binomial smoothing is a
  useful point predictor. That theorem does not by itself provide the one-sided, adaptively value-weighted
  horizon confidence bound required by D3; linearity of the estimator is not a deviation proof.
  *Implementation:* `value_certificate.osw_horizon_value` — coefficients `h_j = P(Bin(k, 2/(t+2)) ≥ j)`
  with `k` from the OSW bias–variance balance `E[t^L]² ≈ N·t²/(t−1)`; `certificate(..., c > 1)`
  routes all three spectra (cond/add/per-order) through it; `t` clamped at `ln N` with a `clamped`
  flag; the report carries `horizon_estimator`/`c_requested`/`osw` meta. Scope: horizon point
  estimate only; T1's next-draw flux band does not cover its horizon error. Stress-tested against analytic
  conditional truth on a uniform planted world (t = 4, N = 150:
  median error 42 vs raw-series 100, truth ≈ 157 — the raw series swings ±t^j per unit f_j
  fluctuation). ET truncation (c ≤ 1) remains the default; nothing changes at c = 1.
- **I2 (support-size impossibility — the count demotion is forced).** Estimating support size with no
  lower bound on species mass is impossible (mass below ~1/N is invisible to any estimator), and even
  with a mass floor `1/K` it costs ~`K/log K` samples (Raskhodnikova–Ron–Shpilka–Smith 2009;
  Valiant–Valiant 2011). Hence count-level `B_E` is non-identifiable without positivity — §12.2.4's
  in-house unboundedness argument is the estimation-literature standard, and demoting counts to
  descriptive is a THEOREM-backed necessity, not house style. The value-flux-at-horizon is precisely
  the functional that remains identifiable.
- **I3 (greedy near-optimality — conditional).** IF the metric admits a distinction-cover
  representation (units cover "atoms" of `H(M)`; `V(S)` = covered mass — monotone submodular), THEN
  greedy attains `V(S_g) ≥ (1−1/e)·max_{|S|=|S_g|}V(S)` (Nemhauser–Wolsey–Fisher). This bounds the
  head-selection suboptimality *multiplicatively* where the ε-gap bounds the tail *additively*. Not
  free: parity worlds violate coverage (the XOR control is the designated counterexample class);
  γ̂ ≈ 1 is the empirical coverage signal.

### 12.8.8 Named conjectures C1–C3 — falsifiable, with designated breakers. **[conjecture]**

- **C1 (low-degree articulability — the thesis in learning-theory form).** Relative to a NAMED unit
  basis and probe measure, expand the best predictor of `M` in the Fourier–Walsh basis of the unit
  verdicts. Conjecture: metrics with concentrated achieved heads carry ≥ (1−ε₀) of their explainable mass at degree ≤ 2 over
  the discovered basis; metrics that remain DEEP (or show `Δ_comp > 0`) carry irreducible degree-≥3 /
  synergy mass over EVERY basis reachable by the named process. *Falsifier:* a DEEP metric closed by an
  F₃ certificate (a degree-3 checklist articulates it) — the wall would relocate from "tacit" to
  "under-expressive combiner", a finding either way. *In-grid test:* the pair stage measures degree-2
  mass; sampled triple-products on the head estimate degree-3 mass. **Caveat (state it in the
  conjecture, not in the rebuttal):** Fourier degree is basis- and measure-dependent — there is no
  basis-free degree; the per-named-basis form is the only well-posed one and matches §12.5
  process-relativity.
- **C2 (tail-synergy decay — assumption A3, named).** The ε bridge sums SINGLETON-conditional values of
  unseen species (chain rule truncated at first order); it cannot see synergy among never-observed
  species. Conjecture: the unseen tail's synergy profile is dominated by the pairwise-synergy decay
  measured on the discovered tail (γ̂'s object). This is the ONE assumption of the bridge that is
  anti-conservative when it fails; this is why the current bridge cannot issue CODIFIABLE.
  **Designated breaker — IMPLEMENTED 2026-07-01 PM** (`test_planted_tail_xor_breaker_A3_is_anticonservative`,
  a documented-failure test like the F₁ XOR blindspot): `M = u_r ⊕ z` with `z = u_a ⊕ u_b ⊕ u_c`
  flipping ~1% of probes — the "judge noise" on a codifiable rule is secretly the parity of THREE
  pool criteria. The triple is **pairwise-independent with uniform marginals**, so every census
  instrument reads zero *by construction, not estimator weakness*: the species partition finds no
  merges (pairwise MI = 0), the head rejects each unit (`I(u_i; M | u_r) = 0` exactly), γ̂'s sampled
  blocks measure joint ≈ 0, the F₂ pair stage is blind (the needed product involves `u_r` —
  4-parity), and all units are RECAPTURED (no unseen mass, no singleton flux — sharper than
  suppressing them, which would be the zero-mass escape instead). Measured on the planted world:
  the historical implementation issued CODIFIABLE at ε = 0.031 while `H_emp(M|u_r) = 0.096` bits sat one
  conjunction away — a **3.1× under-cover of the proposed gap**. Permutations, more draws, and
  richer flux accounting cannot detect it; the two mitigations are exactly the mandated ones: the
  composition-covering adversarial probe (a composed prompt STATING the full rule executes it — the
  current test asserts that the invalid bridge cannot authorize CODIFIABLE.
- **C3 (bracket coherence — exploratory diagnostic).** The achieved recovery `R` and heuristic
  `OPT_F+epsilon` may be compared for instrument debugging, but neither must "pinch" absent a valid checklist upper bound. Divergence
  LOCALIZES the wall: `R ≪ OPT_F` = reconstructor weaker than checklist (instrument artifact);
  `OPT_F + ε ≪ lowerCI(C)` with `Δ_comp > 0` = value lives in the SAYING (composition channel);
  with `Δ_comp ≈ 0` = value beyond the named process (zero-mass tail or combiner class). *Testable
  now* on the planted battery (planted rule: both legs → `H(M)`); in-grid: per-metric scatter of `R`
  vs `OPT_F+epsilon`, status-colored. This is a diagnostic comparison, not mutual validation.

### 12.8.9 The assumption ledger — every escape is the negation of one row. **[design]**

| # | assumption | used by | testable? | designated control | failure direction |
|---|---|---|---|---|---|
| **A1** | frozen-iid capture occasions (heterogeneous `p_s` fine; no occasion drift) | D1–D3, T1 | partially — orbit/form stability, two-list agreement | C1 planted rule through the full pipeline | either direction ⇒ report two-list + order band |
| **A2** | positivity `p_s ≥ p_min` — OR claims scoped to the process horizon | totals (forbidden) vs flux (used) | **NO** (untestable; I2) | — | totals unbounded ⇒ horizon-scoping is mandatory, §12.5 |
| **A3** | tail-synergy decay (first-order truncation of the chain rule) | ε bridge (§12.6.4) | partially — γ̂ on the discovered tail | planted tail-XOR (breaker) | **anti-conservative** — the one assumption whose failure under-covers ε |
| **A4** | judge-noise calibration `γ ≤ γ̂` | any reading against the TRUE target | yes — known-truth calibration items | judge-calibration battery | unmodeled noise inflates `Δ(E)` (thesis-friendly) ⇒ burden on γ̂ |

**Impossibility ledger (what we provably cannot have, so reviewers see the hedges are forced):**
(i) no DPI from unit verdicts to an arbitrary composed prompt — composition is a different channel and is
never bounded by the unit accounting; the separate fixed-target `M_omega-X-M_p` DPI still holds for it;
(ii) no positivity-free count identifiability (I2) — counts stay descriptive
forever, no estimator upgrade changes this; (iii) no label-free unconditional ideal-intent ceiling. The ledger
records assumptions and failure modes; it is not itself a certificate.

## 13. The Certified Unit Framework (CUF) — unithood itself becomes a certificate. **[new, 2026-07-04/05; full spec in notes/2026-07-04__unit-certification-theory.md; implemented: unit_certificate.py + run_unit_certificate.py, 15/15 CPU tests]**

§6.5/§B8 defined the atomic unit doctrinally (behavioral partition operator; species under the
form-quotient; "a criterion string is an *address*, the species is the *function*") but unithood was
ASSUMED at construction (mining + CMI screen), never certified. §13 closes that gap: **membership in Ω
is now itself a certified claim**, with the same discipline as §12.6 (nulls, bands, adverse ends,
planted controls). This section is the doc-of-record summary; the standalone note carries full
definitions and estimator statements.

### 13.1 The declared tuple — every variance source is a component, not a caveat

A unit is a functional of **𝔗 = (E, d, P_X, Φ, Λ, 𝒞, 𝒩, Π, α, δ_min)**: executor+decoding (Def 1; no
executor-free unit exists — §12.7 executor-indexing extended to unithood), probe measure (Def 2), the
three variance measures **H = Φ⊗Λ⊗𝒞** (form-orbit × insertion-position × co-present company — Def 3;
"stochasticity of prompting" and "where the unit falls" are integrated over BY DEFINITION), sham-null
ensemble 𝒩, partition functional Π (identity — thresholded similarity is not transitive, so "same
unit" is Π-relative with measured merge-precision, the census's binding validity gate), test level α,
and materiality floor δ_min (§13.4).

### 13.2 The object: ablation fingerprints, dual effect arms

Effectors: length-matched **neutral-replace** ablation (mechanical length/position artifacts controlled
by construction). The **fingerprint** φ_E(a) = E_H[σ(ι(a)) − σ(ι(∅))] ∈ ℝ^n carries identity (WHICH
probes move); magnitude carries detectability. TWO certified effect arms (user decision, 2026-07-04):
- **δ^free = ‖φ‖₁/n** — target-free behavior shift;
- **δ^M = E_H[Δcorr(σ, m̄_ω)]** — directional shift of alignment toward the metric's own §12.6.2 target.
Magnitude nesting |δ^M| ≤ L·δ^free holds (Prop 4), **but the significance sets are NOT nested** (each
arm has its own null; pilot: M-certification without free-detection is common). The arms are
complementary instruments, not redundant.

### 13.3 Certificates U1–U5 (↔ the three user confounds + two more)

U1 DETECTABLE (sham-ablation null, Bonferroni over the lattice, n_null ≥ max(999, m/α), both arms) ·
U2 IDENTITY (paraphrase-orbit self-similarity ≥ r*; shortfall = **ε_id charge**, ε_form's sibling) ·
U3 CONTEXT-ROBUST (sign-stability + dispersion over H; shortfall = **ε_ctx**; extreme ⇒
CONTEXT-CONDITIONAL(factor) with the conditioning factor named by ANOVA over H's factors) ·
U4 MINIMAL (ATOM vs COMPOSITE by additive fingerprint reconstruction of parts — the "green and round"
doctrine operationalized as a testable outcome) · U5 EXECUTOR SCOPE (E-SHARED / E-SPECIFIC /
E-EMERGENT / E-DRIFT along the within-family ladder). Certified effect reported at the adverse end:
[δ̄ − ε_id − ε_ctx − CI, δ̄ + CI]. Certification never reads value — v(s|S_g) stays in §12.6.

### 13.4 Empirical amendments forced by the pilot (CW#24/#29 × hosts × 3B/8B/70B, 2026-07-05)

1. **Materiality floor δ_min** joins the tuple: at 70B the placebo (δ=0.003) was statistically
   detectable — a hyper-precise executor makes EVERYTHING significant, so detection is a TWO-gate
   (p ≤ α/m AND δ ≥ δ_min). The purely statistical reading of "changes the behavior" is a reductio.
2. **Inertness is executor-relative**: even the neutral-filler control must be certified per E (the
   70B trust gate refused the run — the certificate protected itself).
3. **The effect is a COMPANY PROFILE, not a scalar.** Full-host ablation measures the LAST marginal
   δ(a | Ω∖a) — under submodularity the smallest — and systematically understates units in redundant
   hosts (g24 GEPA host: free-arm 0/11 vs M-arm 6/11). Def 5 upgraded to the bracket **[δ_LOO,
   δ_solo]**; existence gated at the solo end; the bracket width IS the measured redundancy. This
   lands unithood exactly on the §12.6 conditional-gain machinery (LOO = v(a|Ω∖a); solo ≈ standalone).
4. **Low ablation-δ ≠ merge.** Four discriminable causes: duplication (same species, multi-address —
   merge ADDRESSES iff fingerprint ρ ≥ r*; ruled out on g24-GEPA, max ρ=.52), substitutability
   (distinct units, overlapping coverage — keep distinct, wide bracket; the observed case), ceiling
   (σ(host) saturated — host UNRESOLVABLE), dead weight (null at all company levels — drop).
   **Identity uses SOLO fingerprints** (intrinsic function), never LOO residuals: full substitutability
   with distinct fingerprints is impossible, so solo-identity separates all four cases.
5. **Units are E-relative in FUNCTION, not just detectability**: g29 "Ground claims in ethical
   research" is detected at 3B and 8B but with different fingerprints and OPPOSITE δ^M sign
   (−0.07 at 3B, +0.03 at 8B) — E-DRIFT: the same address installs different functions in different
   readers. Sharpens §5ter executor-relativity from "different m_ω" to "measurably different unit."

### 13.5 Superseding statement + downstream wiring

*"Minimum span of text that changes the behavior of an outcome"* → **"minimal ATOM under Def 8,
certified U1–U3 at level α with charges (ε_id, ε_ctx) and materiality δ_min, at declared tuple 𝔗."**
Every term has a measure, an estimator, and an error bar. Post-census wiring (separate step):
`orthogonalize --mode certified` (fingerprint same-species merge replaces raw CMI screen),
`value_certificate --quotient certified` (OPT over a certified basis), span_R2 on the certified basis
— B_E census species inherit error bars. Production state 2026-07-05: Tier-1 unit census running over
all certified domains' description hosts (~1,100+ metrics × Llama ladder), Tier-2 (company profiles)
validated on synthetics, live validation chained.
