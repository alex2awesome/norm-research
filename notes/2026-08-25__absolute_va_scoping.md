# SCOPING — an absolute V+A estimate: missing mass × per-metric optimal-prompt caps

Date: 2026-08-25 (overnight charge, Addendum F-e). Status: SCOPING ONLY, no
numbers computed. Companion to the capacity-null battery (Addendum F-a/b/c).

## The estimator in one line

Absolute-VA upper estimate for a cell:

    UB_VA = A(S)  +  H_impl  +  H_cov

where A(S) = the certified bank AUC with the discovered criteria set S,
H_impl = headroom from implementing each discovered criterion OPTIMALLY
(Paper #2 machinery), and H_cov = headroom from criteria species never
discovered (Good-Turing / Chao-style). If UB_VA sits below the dense T with
margin, the articulability gap survives the strongest articulated-system
estimate we can construct — the ε-tail null is then quantitatively bounded,
not just definitionally excluded.

## Component 1 — H_impl (per-metric optimal-prompt caps, Paper #2)

For each criterion i we have a judge implementation with score column s_i.
Paper #2's certified object is the **DPI fixed-target cap** (the only bound
that survived that lineage's audit): for a fixed target construct, an
information-theoretic ceiling on what ANY prompt implementation of the
metric can extract. Per-criterion headroom Δ_i = cap_i − I(s_i; ·).
Conversion to bank-AUC units is the delicate step — proposal: use the F-a
SCALING CURVE's fitted marginal-value function dAUC/d(criterion-quality) at
the bank's operating point, applied to the top-m informative criteria only
(the F-a curve shows tail criteria contribute ~nothing, so implementation
headroom on the tail is immaterial — this is what makes the computation
affordable).

Cost: Paper-2 cap machinery per criterion. Feasible scope: ONE cell
(cw_community), top-24 criteria by local informativeness ≈ 24 cap runs.
Needs: the M_ω/bracketing harness pointed at our criterion definitions +
the cw corpus; sk2 is that lineage's box (sk2 model overrides per its
memory). Rough cost: days of GPU-judge time, not hours — this is the
expensive leg.

## Component 2 — H_cov (undiscovered criteria species)

We already compute Good-Turing missing mass M̂ = f1/N over criteria SPECIES
per campaign (strict species artifacts, two-judge merges). What M̂ does NOT
give is the VALUE of unseen species. Proposal — the discovery-order value
curve:
1. From the mining rounds we have each species' marginal bank-AUC
   contribution and its discovery round. Empirically the marginal value
   decays by round (the stopping rule fired). Fit value-vs-discovery-rank.
2. Estimate unseen-species COUNT (Chao1 from f1/f2, both recorded).
3. H_cov = Ŝ_unseen × E[value | rank > rank_max], with a conservative
   variant using the max late-round contribution instead of the mean.
   Assumption to state: species value is stochastically decreasing in
   discovery order (supported by the round-increment series; testable by
   permutation within rounds).
4. Correlated-criteria slack: summing per-species values overestimates the
   joint gain (criteria correlate) — which is the SAFE direction for an
   upper bound. State as assumption A2.

Cost: CPU only; all inputs exist in closure artifacts (roundN_species,
roundN routing, per-round ladder increments). ~1 day of analysis code.

## Component 3 — assembly + the two appendix artifacts

- UB_VA point + a band (vary the H_cov tail assumption between mean-tail
  and max-tail; vary H_impl between 0 and full cap).
- Appendix artifact 1: MISSING-MASS PLOTS (currently the paper quotes M̂ and
  draws whiskers but shows no derivation figure): per-cell species
  discovery curves (new species per round), f1/f2 histories, M̂ trajectory
  with the round-ahead backtest. All data on disk; build with matplotlib
  alongside the existing figure suite. ~half day.
- Appendix artifact 2: the F-a scaling curves with fitted asymptotes — the
  same argument made architecture-side.

## Risks / honesty notes

- The DPI caps were certified on Paper-2's tasks; porting to bank criteria
  means re-running, not reusing numbers (never re-quote across designs).
- H_impl conversion via the scaling curve is a linearization; quote as an
  estimate with the conversion assumption named.
- If the F-b distillation probe shows the criteria span already reconstructs
  the dense score (rho high), the whole absolute-VA exercise simplifies:
  the gap is estimation-side and UB_VA ≈ T. If rho is low, UB_VA's margin
  below T is the quantitative form of "the residual resists articulation."

## Recommended sequencing

1. F-a/b/c results land (tonight) — they determine whether H_impl matters
   (scaling-curve slope at k=144) and whether the exercise is even needed.
2. H_cov analysis (CPU, ~1 day) — no new judging.
3. Decision point with user: commission the 24 cap runs (sk2, days) or
   quote UB_VA with H_impl bracketed [0, literature-style bound].
