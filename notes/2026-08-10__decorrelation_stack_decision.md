# Decorrelation stack — production decision rationale (one page)

Date: 2026-08-08 (filed under the campaign's 08-10 slot). Decision-maker: Fable
(user directive: "make a decision on the right stack to do proper decorrelation
with spurious variables"). The FREEZE-READY stack itself is
`notes/2026-08-05__taste-decomposition-design.md` §13; this page is the why.

Poll status at decision time: GRL audit CLOSED (definitive negative, both
architectures); LEACE pilot CLOSED (adopted linear-scope); lit review LANDED
(draft-complete, adoption recs #1-3 identifiable in text); decorrelated-training
battery IN FLIGHT — gradcheck PASS, V1 PASS from stored artifacts, first training
arm (D02) just launched, full battery ~6-8h out (> the ~3h window). Decision made
on best-available; the decor leg is written as a conditional that executes itself
on the battery's own declared, BINDING gates — no re-litigation when it lands.

## Why this shape

**One stack, three scopes.** The campaign's core lesson is that "decorrelation"
is three different jobs with three different failure modes, and every instrument
disaster came from letting one instrument moonlight in another's scope:
- INFLUENCE (observational): stacked increment stays primary because it ran
  everywhere, split the grid into interpretable regimes, and its known bias
  (Westfall & Yarkoni, toward positive increments) is one-sided and patchable
  with a CPU-only reliability band. Matched sampling is demoted to
  sign/consistency duty: on the one channel where every instrument ran (length,
  N&C responded) it read 2-7× low under one protocol and flipped sign under a
  near-identical one — that is not a magnitude instrument. LEACE's erase-and-
  refit JOINS as the third leg because it agreed with stacked/stratified within
  1.2×/0.93× on that same channel while being immune to the two failure modes
  that kill the observational legs past spurious-alone .65 (conditioning-on-
  label for stratification, thin match support for matching).
- TRAINING-TIME (incentive removal): decorrelated reweighting is the only
  candidate left standing with a live battery, and its recipe already encodes the
  two literature landmines (stabilized weights → n_eff .90-.98 vs 32-43% loss
  unstabilized; 2-epoch early-stopping regime is where Byrd & Lipton say
  weighting can bite at all). T_decor is a ROBUSTNESS ARM, never a T
  replacement: it is design-conditional, and T_decor ≈ T is ambiguous between
  "no reliance" and "weights inert" — corroboration, not headline.
- REMOVAL CERTIFICATION: LEACE linear-scope only, rare by design. It is the one
  certificate GRL never delivered at any operating point, and it is honest only
  because our score head is linear in h — that architectural fact is now part of
  the freeze. The .96/.92 MLP residue and the y-tax placebo ship with every
  quote; certificates are void under any retraining or nonlinear consumer.

**The thresholds are inherited, not invented.** The .65 spurious-alone line is
the twice-documented point where stratified discounting became conditioning on
the label (Δ_adj +.1146/.110 at nuisance-alone .712/.713 — both NEVER-QUOTE) and
where the pilot's own freeze recommendation swapped matching in for
stratification. The .02 Δ_beyond line is the existing Layer-3 gate: below it
there is no taste claim to protect, so steps 2-5 of the per-cell procedure
switch off.

**The one new piece of machinery** is the §13.4 cross-instrument signature:
a model that does not rely on channel X should show LEACE channel-specific
erasure cost ≈ 0 (total cost ≈ the y-tax placebo alone). Running that on
T_decor vs T reconciles the reliance scope with the intervention scope using
only existing code.

## What was rejected, and why

- **GRL, permanently** — defeats its own adversary while keeping the channel;
  amplifies reliance under pressure (ablation Δ +.018 → +.123); root cause holds
  on both architectures. The lit review found the negative publishable (an ICML
  2026 paper still ships the mechanism) but no fix worth retrying — Elazar &
  Goldberg already ablated bigger/ensembled adversaries in 2018.
- **Decile stratification as discount past .65** — conditioning on the label;
  survives only as the ≤.65 descriptive appendix; Δ_adj bands or negative-form
  quotes only.
- **Matched sampling as a magnitude instrument** — sign instability documented
  above; kept for consistency duty with a |estimate| < .01 = indeterminate rule.
- **Counterfactual text editing (incl. RAZOR)** — standing user rejection, now
  backed by Joshi & He 2022 / Kaushik 2021 / Chandra Mouli UAI 2022. Not
  re-proposed. RRM-style pairing permutation is the one edit-free counterfactual
  form not covered by the rejection: PARKED pending user sign-off, not adopted.
- **ODIN / AFR / DFR / subsampling — not now.** No third training-time
  instrument while one is mid-battery (no-overengineering; check-before-new-
  approach). They are the NAMED successors in the decor fail branch, each behind
  its own planted battery (with the Bastings tic/op difficulty ladder) and user
  sign-off. AFR-style balanced head-refit is first in line: our frozen-backbone
  + small-head dense standard is literally the DFR precondition, and the
  nuisance-only model already supplies the weights.
- **INLP** — dominated by LEACE (no closed form, more collateral damage).
- **KRaM / Obliviator / kernel erasure** — watch-list; the score path is linear,
  so linear scope suffices and nonlinear-scope claims stay unquotable.
- **Probe-based removal evidence anywhere** — probes detect decodability, not
  use (Kumar et al. 2022; our own two false-PASS near-misses). Ablation/reliance
  and erasure-cost readouts gate; probes are scope notes.

## Standing quoting rules (the short list)

1. Positive stacked increments: quote with the reliability band + the Feng
   limitation sentence. Null/negative increments: quotable directly.
2. Matched sampling: full caliper sweep; |est| < .01 = indeterminate; sign duty.
3. Any LEACE number: MLP residue + y-tax placebo beside it; channel-specific
   cost = total − tax; ≈.55 eval transfer, never "provably zero"; linear-scope
   language verbatim; continuous channels erased as one-hot bins.
4. T_decor: beside T, never instead; named-design caveat; never in the same
   figure/staircase as T scorings.
5. Δ_adj: band or negative form only; never a point estimate from a strong
   nuisance model.

Per-cell procedure and full freeze text: design note §13.6 / §13.
