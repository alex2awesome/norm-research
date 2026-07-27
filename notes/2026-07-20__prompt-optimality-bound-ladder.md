# The prompt-optimality bound ladder — what "nothing can beat", precisely

**Goal (user, 2026-07-20):** solid theoretical upper bounds from the M_ω exploration that nothing
— not M_ω, not GEPA, not any other prompt optimization — can beat.

The audited answer is a three-rung ladder. Each rung is a different *conditioning*, carries a
different *certificate strength*, and now has a *correctly-specified instrument*. The 2026-07-19
D4 negative result (best-of-m cannot produce a bound above best-achieved — structural, verified
twice) is what forced this decomposition: "an upper bound on prompt optimization" is not one
object; it is three, and only the top rung binds *every* method.

Notation: task input `X`, gold target `Y`, a prompt `p` turns a FIXED task LM into a predictor
`M_p(X)`. All bounds below are for the fixed task LM + fixed scoring pipeline (change either and
every rung moves).

---

## Rung 0 — certified, all-prompt, all-method (information-theoretic)

For ANY predictor computed from `X` alone — every prompt, every optimizer, M_ω, GEPA, anything
not consuming labels at inference:

    I(M_p; Y) <= min( I(X; Y), I(M_p; X) )        (DPI; chain Y — X — M_p)
    P(M_p(X) = Y) <= Bayes accuracy given X  =  E_x[ max_y P(Y=y | X=x) ]

- **Status: the only CERTIFIED all-prompt bounds in this project.** The DPI fixed-target cap is
  the surviving certified bound style (memory `project_momega_audit_bracket`; OPT_Ω+ε / Fano /
  CR-ε were retracted — do not resurrect Fano here).
- `I(X;Y)` / Bayes accuracy are properties of the TASK, not of any search: no prompt optimizer
  can beat them because no function of `X` can. This is the theorem-grade sense of "nothing can
  beat" — everything else on the ladder is conditional.
- Practical estimation of Bayes accuracy is itself a modeling problem (it is an upper bound you
  estimate from below), so rung 0 is a *ceiling you cite*, rarely a *number you compute*. Where
  labels' provenance gives structure (e.g. hotpotqa answers are functions of the provided
  context), `I(X;Y)` arguments can rule out perfect recovery when the context is truncated —
  that is a legitimate rung-0 argument.

## Rung 1 — process-conditional (the EVT endpoint; NEW instrument 2026-07-20)

Condition on the *generating process* (this proposer LM, this seed, this budget, this reflection
scheme) and ask: what is the upper endpoint `x*` of the score distribution the process draws
prompts from? Any run of THIS process — however long — is capped by `x*` in expectation of its
max; a different process has a different `x*`.

- **Instrument:** `datasets/prompt-optimality-test/analyze_bounds_evt.py` — GPD-MLE and Pickands
  endpoint estimators cross-checked over a k-sweep, candidate bootstrap CI, dequantization
  sensitivity, binomial SE beside every margin, i.i.d.-violation caveat stated up front
  (adaptive search ⇒ draws are neither independent nor identically distributed; treat as
  exploratory diagnostics, never certificates).
- **First numbers (contaminated 100/17-item pools, runs/bounds_evt_summary.md):** hover — both
  estimators agree, endpoint 0.8325 vs best-achieved 0.8300 (+0.0025, inside SE 0.038): the
  search has exhausted its own draw distribution; hotpotqa/aime — estimators disagree /
  tie-degenerate: no stable endpoint at this n and discreteness. The real version runs on the
  300-item paper-exact splits (SE ~.023) once T3 produces their rescore matrices.
- This is the rung that answers "can M_ω beat GEPA by more than noise?": both draw from
  closely related processes; if the endpoint sits at GEPA's score (hover), the honest statement
  is *no method drawing from this process family has meaningful headroom* — which is the
  precise form of the earlier "GLM saturates hover/hotpot" finding.

## Rung 2 — pool-conditional (what D4 established, with the structural negative)

Condition on the realized candidate pool. Two objects, neither an all-prompt bound:

- **union-of-all oracle** (fraction of items solved by ≥1 pool member): a hard cap for any
  SINGLE prompt *selected from this pool*, but the ceiling of a per-item oracle selector — a
  different object — and inflated by multiple comparisons over noisy binary scoring (aime's
  oracle gap is entirely within noise; hover's mostly).
- **best-of-m / y_inf:** CANNOT yield a bound above best-achieved, by construction —
  `sup_m E[max of m] = pool max` at `m = n`. Verified independently twice (exact combinatorial
  estimator matches MC; byte-identical re-run). Any "projected ceiling" from a monotone fit to a
  best-of-m curve is an *interpolation artifact*, never a bound. This closes the original D4
  framing permanently.

## The superset construction (where M_ω ≥ GEPA comes from)

M_ω v2 (benchmark `run_momega_v2.py`; reconstruction `unit_recombination_m_omega`) initializes
from official GEPA's shipped winner and only ships a compile that beats it on a disjoint
confirmation slice (no-regret guard). Hence **M_ω ≥ GEPA by construction, up to confirmation
noise** — a floor, not a bound. Combined with rung 1: GEPA's score ≤ M_ω's score ≤ process
endpoint `x*`. When `x*` ≈ GEPA's score (hover), the sandwich collapses: M_ω can only match, and
"no improvement" is the *predicted* outcome, not a failure of M_ω. When `x*` sits above (open
question for the 300-item splits), the gap is exactly the room M_ω's extra unit sources
(children metrics, LLM-suggested facets) are competing for against GEPA's own reflection.

## The reconstruction-side mirror (capacity caps, D6)

For the reconstruction program the same ladder appears one level up: recovery
`I(M_ω; M̂) <= I(M_ω; X) <= H(M_ω)` — the recovery readout is capped by the metric's capacity,
which is exactly why the discrimination-maximizing M_ω objective (a capacity objective, per the
D6 SHA-parity analysis) can inflate the cap without adding recoverable content
("variance-revival ≠ information-revival"). Any M_ω-generation upgrade — including v2's unit
recombination, which deliberately keeps the objective unchanged pending sign-off — moves the
*cap*, and only the recovery measurement says whether content moved with it. The open fix
(complexity-penalized recovery) must target executed-computation complexity, NOT description
length (which the hash minimizes) — see the corrected §5 of
notes/2026-07-19__reconstruction-optimality-theory.md.

## What to claim, per rung, in one line each

| Claim | Strength | Instrument |
|---|---|---|
| No prompt method beats `min(I(X;Y), I(M;X))` / Bayes accuracy | **Certified theorem, all methods** | cited, not computed (DPI fixed-target cap) |
| No run of THIS process beats its endpoint `x*` | Process-conditional estimate | analyze_bounds_evt.py (300-item redo pending) |
| No single prompt from THIS pool beats union-of-all | Pool-conditional, oracle-typed, noise-inflated | analyze_bounds.py |
| best-of-m extrapolation bounds anything | **FALSE — structural** | analyze_bounds.py (negative, verified) |
| M_ω ≥ GEPA | Floor by construction (superset + no-regret guard) | run_momega_v2.py / unit_recombination_m_omega |

**Honest headline for the paper:** prompt optimization has exactly one theorem-grade ceiling —
the information the input carries about the target. Below it, ceilings are conditional on the
search process or the realized pool, and we provide correctly-specified instruments for both;
our M_ω construction attains a guaranteed floor at GEPA's result, and the process-endpoint
estimate tells you when the space above that floor is already empty.

---

## Addendum (2026-07-21, user question to preserve): unit-level search as MACRO-ACTION
## factorization of the word-level space — and the RL comparison

**The question:** does unit-level search divide the cost and make brute force tractable vs
word-level search? Are we doing macro-step search through a word-level prompt-update space?

**The honest formalization.** Word/token space is |V|^L (~10^5^hundreds) — no search touches
it directly; even GEPA searches the space of LM-PROPOSED whole-prompt rewrites (a learned,
sampled macro-proposal distribution). Unit recombination replaces that with a FIXED, FINITE
macro basis (n=48-96 clauses mined from successful trajectories + suggestions). This does NOT
make global brute force tractable (2^n subsets), but it changes what IS tractable:
1. The FIRST-ORDER value spectrum becomes EXHAUSTIVELY measurable — every unit's paired
   marginal in O(n) panel evals (GEPA can never enumerate its own proposal space).
2. Under measured near-additivity, greedy prefix + drop-one approximates the subset optimum
   with the residual interaction error MEASURED as Δ_recomb (never assumed). Observed: hotpot
   units additive to k=7; hover redundant after k=1; Δ_recomb small (+.017-.05 panel-scale).
3. For small survivor sets (k<=5), exact enumeration (2^k) is affordable — optional exactness.
So: units convert an intractable open search into (exhaustive linear scan) + (low-order
interaction correction) + (measured slack). "Macro-step search through word-space" is exactly
right, with the crucial addition that the macro basis is EXTRACTED FROM the search's own past
trajectories — we search the low-dimensional manifold reflective search already carved out.

**The duality this buys the paper:** GEPA is bounded by its PROPOSAL DISTRIBUTION (EVT endpoint
= endpoint of sampled draws); M_ω is bounded by its BASIS (Chao/Good-Turing missing-value =
endpoint of the enumerated unit generator). Both are species-sampling bounds — the same
question at the sampled-macro vs enumerated-macro levels. Empirically the enumerated basis
finds residual value exactly where the proposal distribution has gone dry (hover/hotpot/aime:
GEPA 0 improvements, units +.04-.07).

**vs RL-based approaches (GRPO in the paper; Finn-style RL more broadly):**
- WHAT MOVES: RL updates weights (capability itself); prompt methods re-index existing
  capability (articulable-strategy headroom). The boundary case is AIME: prompt headroom ~0,
  and it is exactly the one bench where GRPO (38.0) beats GEPA (32.0) in the paper's own
  table — our vacuity/endpoint story PREDICTS which benches are RL-only territory.
- CREDIT ASSIGNMENT: RL gets high-variance scalar trajectory rewards; unit marginals get
  PAIRED per-item attribution (the variance win behind our sign tests). Units are options/
  skills in RL language: temporally-extended, reusable, composable actions — M_ω is to GEPA
  what offline skill-extraction + planning is to online policy search.
- SAMPLE ECONOMY (paper's table + ours): GRPO 24,000 rollouts/bench; GEPA 1.8-7k; M_ω
  3-12k with exhaustive first-order coverage and a no-regret floor (RL and GEPA have no
  analogous guard — cf. our 3 documented guardless regressions).
Positioning sentence: prompt-space methods buy cheap re-indexing up to an articulability
ceiling; RL buys capability past that ceiling at ~10-40x rollout cost; the unit factorization
is the cheapest complete sweep of the re-indexing regime, with measurable bounds on what
remains.

## Addendum (2026-07-22, user): the DENSE arm — articulation gap of the benchmarks themselves,
## and the OSL-for-prompts staircase (DESIGN ONLY, launch pending user sign-off + free GPUs)

**The question:** these datasets must have dense upper bounds — is the prompt-vs-dense gap its
own articulation gap? **Yes, with one precision:** rung 0 is vacuous here (deterministic
labels), so the meaningful dense bound is CHANNEL-CONDITIONAL — what weight-installation
(SFT/RL on train) achieves vs what articulated instructions can index. Gap = dense ceiling −
EVT prompt endpoint. The paper's own GRPO column IS the dense arm at one scale (Qwen3-8B,
24k rollouts): AIME 38.0 > GEPA 32.0 (positive articulation gap — RL installs what prompts
can't index; the math-as-orthogonal-floor pattern), while hover/hotpot/pupa show GEPA ≥ GRPO
= no DETECTED gap (GRPO-at-budget only lower-bounds the dense ceiling). Collins mapping:
articulation↔RTK, SFT↔STK, RL↔CTK (tacit-installation-channels stream).

**What existing OSL evidence says (honest inventory):** (a) prompts do NOT always improve
with executor scale — inverted-U in 4 families, falling limb = divergence-toward-truth; BUT
that was misspecified-metric execution; for truth-labeled tasks the same mechanism should
help → PREDICT more-monotone (prereg before confirmatory). (b) units-supported vs scale:
NEVER measured. (c) unit transfer up-scale: unmeasured at unit granularity (closest: H49
8B→70B rank transfer; transport 41/46). (d) existing OSL freezes the artifact across scale;
freeze-vs-remine for unit pools is an open design choice.

**Design v2 (2026-07-22 USER SIGN-OFF — Qwen3 staircase, same-family rule):
1.7B/4B/8B/14B/32B × {hover, aime, +ifbench?}**
- **Pool: FREEZE-AND-REMINE (user's choice)** — union of the frozen 8B-mined pool + per-scale
  re-mined units, source-tagged; every scale SELECTS over the union, so reuse-vs-scale-native
  choice is an observable, not an assumption. Pool construction is not the object: treat it
  as an approximation to the ideal huge-IID low-missing-mass pool and MEASURE per-scale
  Good-Turing missing mass (existing instrument) instead of assuming it away.
- Arms per scale: seed / GEPA-8B-winner transplant / M_ω select-over-union (primary) /
  LoRA-SFT-on-train (dense estimate; phase 2).
- **PREREGISTERED HYPOTHESES (user, verbatim intent; freeze before confirmatory):
  H-i bigger models absorb more articulation (n_compiled ↑ scale); H-ii they afford more
  SPECIFIC articulation (LLM-coded specificity of selected units ↑ scale; Sonnet+ coder);
  H-iii they locate the articulation ceiling (best-prompt value ↑ toward asymptote; per-scale
  EVT endpoints converge to it).
  H-iv (2026-07-22 user: "super super interesting"): ABSOLUTE SCORE vs scale and PROMPT LIFT
  (best − seed) vs scale are SEPARATE preregistered readouts — prediction: absolute monotone ↑,
  lift SHRINKS (the articulation gap closing from the capability side).
  H-v (separability of capacity vs specificity, user-approved 2026-07-22): the union-pool +
  specificity coding + small-pool-survival jointly distinguish (a) PURE CAPACITY (n_compiled ↑,
  selected-specificity flat), (b) SPECIFICITY-SUBSTITUTION (n flat/↓, specificity ↑), (c) both.
  Survival sub-hypothesis: among frozen-pool units, up-scale survival correlates POSITIVELY
  with tier-2 specificity (what small-model selections lack is specificity, not coverage).**
- **POOL POLICY (2026-07-22 decision): FROZEN 8B pool = PRIMARY** (one sampling distribution →
  coherent Good-Turing/Chao missing mass; up-scale unit survival well-defined); union with
  per-scale re-mined units = SECONDARY/exploratory (selection run twice per scale; frozen-unit
  marginals shared, so extra cost ≈ one prefix/drop-one pass). **UNIT IDENTITY = two tiers**:
  tier-1 exact hash (module+normalized string; conservative floor), tier-2 semantic (embedding
  NN candidates + Sonnet+ judge "same instructed behavior?", blinded anchor pairs per batch,
  per anchor-test rule). Cross-pool "same unit" claims use tier-2.
- **FAMILIES (v3, 2026-07-22 user: "full depth is probably necessary for the paper"):
  ALL THREE families run FULL DEPTH — every available rung × both benches × the complete arm
  set (fixed arms + per-scale frozen-pool M_ω + per-scale GEPA + phase-2 LoRA-SFT). Rungs are
  capped by each family's lineup: Qwen3 1.7/4/8/14/32B (5), Llama 1/3/8/70B (4), Gemma-4
  (3-4 per available sizes). Families NEVER pooled (same-family rule); "shape" language
  applies ONLY to cross-family comparisons (slope signs, asymptote existence,
  threshold-free readouts; z×a family-relative precedent). Est. ~6-7 days wall-clock;
  32B/70B rungs are the poles.**
- **GEPA RE-OPTIMIZED PER SCALE (user override of the 8B-only scoping, 2026-07-22: "may as
  well compare against GEPA all the way")** → per scale we get TWO lift curves (GEPA−seed,
  M_ω−seed) + the M_ω−GEPA gap vs scale (does the unit-recombination edge persist, or does
  reflective search catch up as models grow?). Budget: ~10 GEPA runs for the Qwen3 primary;
  families PHASED (Qwen3 complete → Llama → Gemma-4).
- **LETTER-ANCHOR definition (for the recovery-side y): "the metric's own labels" = the fixed
  metric text executed by its designated REFERENCE EXECUTOR M_ref (no platonic labels);
  fidelity = agreement(M_E, M_ref); divergence-toward-truth = on the M_ref≠gold disagreement
  subset, big-E outputs land disproportionately on gold. Verify-before-freeze must pin down
  WHICH reference executor anchored each task's staircase.**
- **VERIFY-BEFORE-FREEZE (2026-07-22): before confirmatory runs, re-pull the OSL notebooks and
  confirm which tasks had GOLD-ANCHORED disagreement-subset analysis (the actual
  divergence-toward-truth measurement: on metric-vs-gold disagreement items, large-executor
  outputs land disproportionately on the gold side) vs letter-only divergence — prereg cites
  only the anchored subset.**
- Two y-axes kept SEPARATE (mechanism prediction, 2026-07-22): supervised benches (y=truth) →
  absolute score monotone BUT prompt LIFT (best − seed) may shrink (gap closing from the
  capability side); unsupervised recovery (y=the articulated letter) → falling limb EXPECTED
  (big executors repair idiosyncratic metrics toward constructs = identity dilution at scale);
  recovery side gets a two-axis (executor × recoverer) staircase.
- Measures: score + lift vs scale, n_compiled vs scale, small-pool-unit survival in larger
  models' selections (overlap), specificity vs scale, per-scale missing mass, per-scale EVT
  endpoint, articulation gap (SFT − best-prompt) vs scale. Threshold-free readouts for any
  cross-family comparison; staircases one family only.

**CORRECTION (user, 2026-07-21): the intended Finn comparison is AUTO-SCAFFOLDING — RL
searching the space of available scaffolds** (agent architectures / workflows / component
compositions), not weight-space GRPO. That is a much closer cousin to unit recombination than
RL fine-tuning: both search DISCRETE COMPOSITIONAL MACRO SPACES (their macros = scaffold
components: tools, subagent patterns, control flow; ours = instruction clauses within a fixed
program). The real contrast is the SEARCH OPERATOR: RL-over-scaffolds learns a sampling policy
via rollout credit assignment (high-variance, amortized, can generalize across tasks);
unit recombination enumerates the first-order value spectrum exhaustively and composes greedily
with measured interaction slack (low-variance, per-task, no policy to train). Our bound ladder
transfers verbatim to their setting: an EVT endpoint conditional on the scaffold-proposal
process, and a Chao/Good-Turing missing-value bound over scaffold-component species — nothing
in the machinery cares whether macros are clauses or scaffold components. Positioning: the
bound framework is a general audit for ANY macro-space search; RL-scaffolding is a learned
proposal distribution whose endpoint our instruments can estimate but whose training cost buys
cross-task amortization we don't attempt.
