# Critique + redesign: the UPPER-bound half of the prompt-optimality certificate

*2026-07-01. A deep-dive review of `notes/2026-06-18__prompt-optimality-theory.md` and the shipped
estimators (`methods/metric_implementer/experiments/alpha_probe.py`, `crc_analyze.py`, `be_report.py`,
`semantic_behavioral.py`), answering four questions: (1) critique the upper-bound theory/design;
(2) a more unified / proof-oriented alternative — an ε-gapped guarantee; (3) why we're not converging;
(4) validity under form/structure change. This is a review note, not an edit to the theory — the theory
doc is left intact.*

**Standing bracket (per `feedback_T_lower_bound_Mstar_be_upper`).**
```
   R(p̂)  ≤  T(m_ω)=I(M_ω;X)  ≤  I(M*;X)  ≤  U
   ────    ──────────────      ─────────    ──
   floor   floor on the ideal   the target   census (α / c_∞ / OPT′)
   (cert)  (cert, DPI)          (open)       (NOT a cert)
```
Recovery `R` is the LOWER bound. `T(m_ω)` is the ceiling on recovering the *operationalized* metric =
the FLOOR on the ideal. The UPPER bound on the *ideal* `I(M*;X)` is the whole open problem.

---

## 1. Critique — what's certified, what isn't, and where the design is mis-centered

**The lower half is sound and certified; the upper half is being carried by the wrong observable.**
Three precision points, in decreasing order of how much they matter.

### 1a. The middle inequality is monotonicity-in-pool, not DPI — so route it through `OPT_Ω`
`I(M_ω;X) ≤ I(M*;X)` is written as if automatic. It is **not** a data-processing inequality: there is no
Markov chain `X → M* → M_ω` (m_ω is an executor's verdict on a *mined* pool Ω, not a garbling of the
ideal). The justification is pool-*completeness* — "a richer pool reaches more of the ideal." But that
monotonicity holds for `OPT_Ω := max_{S⊆Ω} R(C(S))` (monotone by set inclusion, §6.9), **not** for the
transmission `T(m_ω)` of one particular compiled prompt — because adding criteria is non-monotone for
recovery (PRUNE, §6.1). The clean monotone-convergent chain is therefore:
```
   R(p̂)  ≤  OPT_Ω  ≤  OPT_∞  ≤  [ I(M*;X) under support-completeness ]  ≤  cap_f
            (cert)    (bounded-monotone limit — exists)   (named assumption)
```
`T(m_ω)` sits as an *instance floor*; the object that actually converges (and that the ε-gap argument in
§2 needs) is `OPT_Ω`. **Recommendation: state the bracket through `OPT_Ω`, not `T(m_ω)`.** Not pedantry —
it's what licenses a convergence guarantee (a bounded monotone sequence converges; a single non-monotone
`T` does not).

### 1b. The upper bound on the ideal is structurally NOT a certificate — and the theory says so
Every tool offered for `I(M*;X)` is a point estimate with a *named-sign bias*, never a certificate:

| tool | §  | status | bias direction |
|---|---|---|---|
| consensus `c_∞` = `H(c_∞)` | 12.4 | point estimate | anti-conservative under shared proposer bias |
| independence / Fienberg census | 11.3a | optimistic point | positive dependence → false saturation |
| spectral `OPT′` (decoder capacity) | 11.6 | conservative but **metric-agnostic** | loose (bounds every metric at once) |
| everything | 12.5 | **process-relative** | scoped to `{LLM, GEPA, children}` reach |

So the honest status is asymmetric: `R ≤ T(m_ω)` is a genuine certificate; `T(m_ω)` as the executor
ceiling is a genuine certificate; **`I(M*;X) ≤ U` is not, and by §12.2.4 cannot be, without an untestable
assumption (positivity / Lipschitz-impact / a non-LM expert list).** The brittleness the user feels is
this structural fact surfacing — not an estimator bug.

### 1c. The census answers the wrong question (mass, not impact) for the ideal
`|B_E|` (richness of reachable behaviors) and `I(M*;X)` (transmission of the ideal metric) are different
objects: a huge B_E does not imply a large `I(M*;X)`, because most reachable behaviors are irrelevant to
the metric (spurious spread, §4.3). Counting behaviors/criteria bounds **mass/richness**, not the
**ideal's transmission**. The theory knows this ("mass ≠ impact," §6.7c, §11.3) — but the *shipped
experiment* (`crc_analyze`, `be_report`, `alpha_probe.decide`) centers count-B_E + coverage + count-α,
i.e. it measures mass. That is why the numbers don't converge and don't obviously bound anything we care
about. **The design isn't wrong so much as mis-centered: the theory's own better observable (value /
consensus) is under-used in the experiments.**

**Keep (these are right):** `R ≤ T` DPI/convexity core (§2) — the *one* monotonicity that holds, not
V-info; assumption-free `C_lo` (Berend–Kontorovich) as a template; two-stream discipline (frozen breadth
vs adaptive depth); process-relative scoping (§12.5) — it's what makes the claim replicable.

---

## 2. A more unified, proof-oriented upper bound — the ε-gap guarantee

**Unifying observation: every certificate here is a missing-mass problem; only the *measure* differs.**

| certificate | measure | tool | status |
|---|---|---|---|
| coverage `C_lo = 1−(f₁/N+√(log(1/δ)/N))` | counting | Berend–Kontorovich | assumption-free, one-sided ✓ shipped |
| **value gap `MV₀ = (Σ_singleton v_s)/N + B√(log(1/δ)/N)`** | **value** `v_s=I(M_i;σ(s))` | **same B–K** | **assumption-free, one-sided (under tail-submod)** — in §12.3-A, not centered |

The value-measure version bounds the thing we actually care about. Assemble it into one statement.

### The theorem (ε-gapped upper bound on achievable recovery)
Fix metric `M_i`, frozen executor `E`, frozen iid probe set `X` (disjoint from optimizer-train), frozen
iid proposal `π` over criteria (N draws → pool Ω). Let `g_k` be the k-th greedy marginal *recovery* gain,
`v_s` the marginal value of criterion-species s, `B=max_s v_s`, `α_V` the value-Heaps exponent. Then with
probability ≥ 1−δ:
```
   OPT_∞ − OPT_Ω  =  Σ_{k>|Ω|} g_k  ≤  (Σ_{s:singleton} v_s)/N  +  B·√(log(1/δ)/N)      ... (★)
   PROVIDED   (i) tail-submodularity  γ≥1 for unseen criteria      [one-sidedness]
              (ii) light value tail   α_V < 1                      [summability of Σ g_k]
```
**Why (★) is a genuine ε-gap and not the count census:**
1. **RHS = value-weighted Good–Turing missing mass + B–K slack** → assumption-free given the frozen iid
   proposal (identical machinery to `C_lo`), concentrating at 1/√N.
2. **Submodularity gives the safe direction for free.** True (conditional) marginal `v(s|S_sel) ≤ v(s|∅) =
   v_s^marginal`, so the *conditional* missing value (what limits OPT) ≤ the *marginal* missing value
   (what (★) estimates). Marginal over-counts redundancy → (★) is an **upper** bound on the true gap.
3. **`α_V < 1` is the summability that makes `Σ_{k>|Ω|} g_k` finite** — this is exactly "measure
   convergence on `a_value`." Count-`α ≈ 1` is irrelevant here: paraphrase-criteria are count-singletons
   (inflating `f₁` so `C_lo` never certifies) but carry ~0 *marginal* value beyond what's seen, so they
   do **not** inflate `Σ_singleton v_s`. **This is why value converges when count doesn't.**

**The one honest hole, named not hidden.** (★) is one-sided only under tail-submodularity. Genuine tail
*synergy* (γ<1: a "magic word" whose value appears only in combination) makes `v(s|S) > v_s` and flips
(★). The assumption-free backstop is the adversarial probe `I(X_probe; M_i | X_Ω) ≈ 0` — it needs no
submodularity and directly tests "does any out-of-pool behavior carry residual M_i-information." So:
```
   ε-gap certificate  =  [ (★) value missing-mass, B–K ]  ∧  [ adversarial_saturation ≈ 0 ]
```
Both legs are already coded (`value cert` §12.3-A; `orthogonalize.adversarial_saturation`). **The theory
has the pieces; it has never assembled them into this single statement or centered the experiment on it.**

### What (★) does and does NOT give — the two-layer certificate
```
   R(p̂)  ≤_cert  OPT_Ω  ≤_(★)  OPT_∞ = I(M*_reachable ; X)  ≤_named  I(M* ; X)
   ────────────────────  ──────────────────────────────────  ─────────────────
   certified (DPI + B–K)                                       support-completeness
                                                               (§12.2.4 wall — NOT certifiable)
```
(★) certifies the gap to `OPT_∞` = best recovery over the **reachable support** `∪ supp(π_k)`. It does
**not** close `OPT_∞ → I(M*;X)`. That last inequality is the §12.2.4 wall; the *only* thing that puts a
bounded number on it is **Lipschitz-impact** (§12.2.4b): if value is `L`-smooth in a criterion embedding,
residual unseen impact ≤ `L · d(covered-region, frontier)`, and `L` is *partially checkable on-support*
(fit it on seen criteria). Promote this from a footnote to the headline sensitivity — it's the most
defensible of the three closers, and it converts "unbounded unseen support" into "bounded residual
impact" (impact is what we care about, not species count).

### Framing decision for the paper
Claim **`OPT_∞` (the reachable-ceiling)** as THE upper-bound result — it is fully certified via (★). Report
the ideal-gap `I(M*;X) − OPT_∞` as a *named, Lipschitz-bounded sensitivity*, not a certificate. This is
the honest and still-strong story: an ε-gap with confidence δ on everything the generators can reach,
plus one explicitly-labeled un-closeable inequality — instead of brittleness diffused across the census.

### A second, census-free upper bound worth promoting: the metric-specific strong-probe ceiling
The census tries to characterize B_E intrinsically (needs a stable species partition — the fragile part).
Sidestep it: train a **stronger decoder `E′`** (a supervised probe on full embeddings, feature access no
prompt has — §11.6 OPT′ machinery) and compute `T′(m_i) = I_{E′}(M_i; X)` on the metric's *own* soft
labels. `T′` upper-bounds the ideal's transmission *for this executor family*, it's a **scalar**, it's
**form-robust**, and it needs **no equivalence relation**. §11.6 keeps OPT′ "conservative/metric-agnostic";
the refinement is to make it metric-*specific* by probing on `M_i`'s labels — §11.6 already notes "same
spectral machinery, different input" (= `c_∞`). **Recommend: run `T′` head-to-head against the census as a
robustness cross-check — when they disagree, trust the scalar `T′` (no partition to destabilize it).**

---

## 3. Why we're not converging — root cause + the two proposed fixes + what's missing

**Root cause: the equivalence relation ("same species") is unstable, so every downstream population
statistic (Chao1, Good–Turing, Heaps α) is unstable — all are functionals of the `{f_j}` spectrum, which
is a functional of the partition.** The evidence (code + the parallel-agent findings):

| failure | evidence | direction |
|---|---|---|
| over-merge distinct criteria | `semantic_behavioral`: merge-precision **0.28** (72% of merges are semantically DIFFERENT), flat 8B→122B | count unstable *down* |
| threshold-defined, no plateau | cmi_thresh 0.10→0.30 → D_obs **9→88**, monotone | count has no fixed point |
| form-sensitivity | rubric-first vs text-first → B_E **~2×** at 81% per-item agreement | count unstable *sideways* |
| count-α ≈ 1 by construction | proposer creativity keeps emitting new *phrasings* | count never saturates |

So the count/richness axis is dominated by accidental collisions (over-merge) + paraphrase proliferation
(under-merge) and **never plateaus** — it is the wrong convergence target.

### Fix (a) — "accept paraphrases of Ω elements" — CORRECT and load-bearing
Formalize as: **define the atomic Ω unit as the SEMANTIC equivalence class (form-orbit), not the raw
behavioral signature.** Use semantic-SAME to *merge* paraphrases that behave slightly differently (fixes
form-fragility and the `beh-DIFF & sem-SAME` cell), and keep behavioral-difference to *split* distinct
criteria that behave the same (fixes over-merge, the `beh-SAME & sem-DIFF` cell). `semantic_behavioral.py`
already computes the semantic judgment — **promote it from a diagnostic to the partition.** Safe ordering:
use the semantic judge only to *merge* (never to split — the judge tracks meaning not surface, cf. the
`subtask_codability` retraction), keep behavioral splitting. This directly repairs merge-precision 0.28 and
makes the partition **form-invariant by construction** (→ §4). Highest-value single change; it's the user's
instinct, made rigorous.

### Fix (b) — "measure convergence on `a` vs `a_value`" — CORRECT, and it's the whole game
`α` (count) ≈ 1 is uninformative; `α_V` (value) can be ≪ 1 because redundant paraphrases carry ~0 marginal
value. Convergence should be declared on **`MV₀ → 0` (the (★) certificate) / value-rarefaction `E[V(m)]`
saturating (`α_V < 1`)** — NOT on `D/Chao1 → 1`, NOT on count-`α < 0.3`. **But note the current
`alpha_probe.decide()` gates GO on `alpha_terminal < 0.3 AND C_lo ≥ 1−ε AND D/Chao1 → 1` — all three are
count-axis gates**, which is why everything lands NO-GO/AMBIGUOUS. Replace with a value-axis gate:
```
   GO  ⇔  MV₀ ≤ ε  ∧  α_V < 1 (summable tail)  ∧  adversarial_saturation ≈ 0
```
Count-α, Chao1, B_E become *descriptive diagnostics with error bars*, not the decision. Concrete code
change in `decide()`.

### What else is missing (beyond the user's two)
- **(c) The probe set is the binding constraint, not the model.** `semantic_behavioral`: species count
  doubles 8B→122B but merge-precision stays 0.28 → bigger models find *more* species, not *cleaner* ones →
  the bottleneck is 300 probes failing to separate criteria. Fix (a) dissolves this: with a **semantic**
  merge, probe count only affects the *value estimate* `v_s=I(M_i;σ(s))` (needs far fewer probes than a
  stable *partition* does), not the partition. If you keep a behavioral partition, scale probes to the knee
  (~450–600 per the parallel agent).
- **(d) Stationarity on the value stream (the §11.3a sleeper).** GV7: `α_V` must use the MARGINAL `v(s)`
  (stationary → valid Heaps/GT); the CONDITIONAL `v(s|Ω)` with growing Ω is non-iid and belongs *only* in
  the submodular missing-impact certificate. Verify `run_value_census.py` respects this split (not read
  here — flag).
- **(e) Report distributions, not points** — already the parallel agent's conclusion (Ω-order + probe +
  form error bars). The value-axis version: bootstrap `MV₀` over probe-subsample × form-orbit; if the CI
  *upper* end ≤ ε, that's the certificate.

---

## 4. Form/structure — hope of validity, how to probe, what guarantee

**The tension, stated cleanly.** §6.8 treats form as an *orthogonal coordinate* that modulates executor
fidelity (raises T) but carries no per-item signal `X_e`, so "form can't enter the co-information
machinery." But the *measurement* is done on signatures `σ(p)` that form demonstrably moves (B_E ~2× under
a template reorder). **You cannot simultaneously claim form is orthogonal to content AND measure content
via a signature that form moves.** The user is right to be unconvinced.

**Resolution — same move as fix (a): quotient the content unit by the form group.** Two routes:
1. **Semantic quotient (preferred).** Paraphrases/form-variants collapse into one species via the semantic
   judge. Form-variation becomes *within-species* by definition; B_E counts form-invariant content units;
   form no longer moves the *count*, only the *within-species signature spread* (reported as an error bar).
   This makes "we measure only content Ω units" TRUE by construction, not by hope.
2. **Form-orbit marginalization.** Canonical signature `σ̄(c) = E_{φ∼Φ}[σ(compile(c,φ))]` — average the
   soft signature over a sampled form orbit Φ (paraphrase, reorder, ±few-shot). `form_invariance()` already
   scores reformulation drift; this repurposes it from *diagnostic* to the *definition* of the signature.
   Residual `Var_φ[σ]` is the measured form-sensitivity.

**The formal guarantee (a Lipschitz stability bound, not a proof of irrelevance).**
```
   | T(content, φ) − T(content, φ') |  ≤  L_form · d(φ, φ')
```
`L_form` is measured directly by `form_invariance` (median drift, p95, binary-flip rate). If
`L_form · diam(Φ) ≤ ε`, then optimizing content at fixed form loses ≤ ε to any form choice — a genuine
block-coordinate ε-guarantee (§6.8's curvature argument, now *measured* as a constant, not *assumed*
concave). **The honest content: form-invariance is BOUNDED by a measured constant, not proven. The content
certificate is explicitly conditional on `L_form · diam(Φ) ≤ ε`.** Currently the measured drift is large
(flip rates non-trivial, B_E 2×), so at the *present* orbit width the content-only measurement is **not**
valid — the honest move is (i) quotient it out (route 1) or (ii) widen to the `{content × form}` product
certificate (§6.8's rigorous route) and pay the form-discovery gap.

**Does form-as-lever break the enterprise? No — it relocates the claim.** If a better-phrased rubric
recovers more, that lever acts by raising `T` toward `cap_f` (executor fidelity), which is *inside* the
bracket — `R` and its ceiling `T` already absorb form's fidelity effect. What form must NOT be allowed to
do is *change which content is measured* — exactly the species-instability the quotient fixes. Defensible
framing: **form is a fidelity knob (in-bracket, fine to optimize); content is what we census; the two are
separated by quotienting the species partition by the form group, with the residual reported as a measured
Lipschitz constant.** A checkable contract, not a hope.

**One probe to add: the content×form interaction test.** §6.8's dangerous case is a criterion that fires
only with the right exemplars (non-separable). Probe directly with a 2-way ANOVA on the signature tensor
`σ(c,φ)`: compare the form main effect `Var_φ` against the interaction `Var_{c,φ} − Var_c − Var_φ`. Small
interaction relative to the content main effect ⇒ separability holds, content-only is sound; large ⇒ form
and content are entangled and the product certificate is mandatory. This is a clean, runnable
**falsification** of the separability assumption — currently assumed, never tested.

---

## What to change (concrete, in priority order)

1. **Partition → semantic/form quotient** (`semantic_behavioral` judge as the *merge* rule, behavioral as
   the *split* rule). Repairs merge-precision 0.28 + form-invariance *by construction*. Highest value.
2. **Re-center convergence on the value axis:** report `α_V`, `MV₀` (with B–K δ-CI), `adversarial_saturation`;
   demote count-B_E/Chao1/count-α to descriptive-with-error-bars.
3. **Rewrite `alpha_probe.decide()`** GO gate → `MV₀ ≤ ε ∧ α_V < 1 ∧ adversarial_saturation ≈ 0`.
4. **Assemble (★)** as one theorem in the doc (§12.3 already has the legs); state the two-layer certificate
   `R ≤ OPT_Ω ≤_(★) OPT_∞ ≤_named I(M*;X)`; elevate Lipschitz-impact as the ideal-gap sensitivity.
5. **Add the metric-specific strong-probe ceiling `T′`** as a scalar, form-robust upper bound; cross-check
   vs the census.
6. **Add the form Lipschitz constant `L_form` + the content×form interaction ANOVA**; make the content
   certificate explicitly conditional on `L_form·diam(Φ) ≤ ε`.

**Un-fixable, be honest:** support-completeness (`OPT_∞ → I(M*;X)`) cannot be certified (§12.2.4). Best
available is Lipschitz-impact (bounded residual, partially checkable) or a non-LM expert list (ceiling-
raiser, still not a certificate). Decide whether the paper claims the *reachable* ceiling `OPT_∞` (fully
certified) or the *ideal* ceiling (needs the named assumption). Recommendation: claim `OPT_∞`, report the
ideal-gap as a named sensitivity.
