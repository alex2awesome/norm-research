# If v14 lands: the optimistic (but honest) conclusion set

**2026-07-15. Written before results, at the user's request, so we know in advance what a good
outcome licenses us to say — and what it doesn't. Companion to
`2026-07-14__ceiling-ladder-runbook.md` and `2026-07-13__v14-decoder-tuning-roadmap.md`.
Everything below is conditional on the CERT lane's gates passing (independent reference survives
anchors, planted suite certifies, shuffled ≈ 0, blind-menu prior subtracted). FAST rows are never
quoted.**

## 1. Certified upper bounds across metrics × tasks

**Best case:** a table of exact structural caps — one per (metric, channel) over 35 Tier-B metrics
× 7 tasks — each an *enumeration*, not an estimate: the maximum value ANY decoder could have
achieved given the frozen panel, executor, and H. Cap sentinels (8-demo, 65,536-state) anchor the
enumeration at two panel sizes.

**What it licenses:** "Under a frozen 6-demo instrument, the demonstration channel's capacity for
criterion X is at most B bits — no prompt engineering, no better decoder, no tuning can exceed it
without changing the instrument." This is the DPI fixed-target kind of bound — the only kind we
have ever certified — now metric-resolved and cross-task. If caps cluster low for whole metric
families while others sit high, that is a *structural articulability map*: the first quantitative,
certified statement of which evaluative criteria are even in principle transmissible through k
examples.

**Conclusive?** Yes, in the strongest sense available to us: caps are theorems given the frozen
instrument (exhaustive over states; no sampling error, no significance test needed). The relativity
is the instrument itself (§5).

## 2. Behavioral reconstruction, improved channel

**Best case:** median achieved behavioral value significantly > 0 by the 10K selection-preserving
permutation null, at |H| = 240 (vs v13's 60), bias-audited, with per-metric z-scores; a meaningful
fraction of metrics individually significant; planted mechanical criteria recover near their caps
(instrument works); worst-pool value no longer median-zero (pool sensitivity tamed or at least
characterized).

**What it licenses:** the v13 headline that was "unresolved pending permutation audit" becomes a
certified existence result: **human-mined evaluative criteria transmit measurable information
through demonstrations into executed behavior on held-out data** — not into judged explanation
quality, into verdicts. Paired with §1, the achieved/cap ratio per metric says *how much of the
attainable ceiling articulation actually reaches* — the paper's bounds-reframe (§III) gets its
central column. v13's ~36%-of-cap number, if it replicates with certification, becomes quotable.

**Conclusive?** Positive rows, yes (permutation-certified transmission is an existence proof).
Null rows, no — a zero under a frozen 8B executor cannot distinguish "tacit criterion" from
"executor too weak to express the rule" (the Codex diagnosis stands). Nulls only become
interpretable jointly with §1 caps (cap high + achieved zero = genuine transmission failure;
cap ≈ 0 = the instrument never gave it a chance) and with the Phase A/B executor decision.

## 3. MCQ reconstruction, trustworthy

**Best case:** MCQ identification reported as **lift over the blind-menu prior** (the standing rule
from the retro no-demo audit: id-acc minus blind baseline, counterbalanced menu orders,
centralness-balanced menus), positive with permutation support, shuffled-label control ≈ 0.

**What it licenses:** the first MCQ number we can quote at all — every previous MCQ id-acc was
retracted as prior artifact. The claim becomes: **demonstrations move menu recognition above what
the menu alone predicts.** Cross-referenced with §2, disagreement cases (metric 34 pattern: flat
MCQ, positive behavioral — or the reverse) become the recognition-vs-execution dissociation,
measured within one frozen design.

**Conclusive?** Yes for "recognition exceeds prior," provided the blind-menu arm is the baseline
in the same frozen design — that is exactly the control the v13-era numbers lacked.

## 4. C1 — menu selection executed behaviorally

**Best case:** the picked description, executed by the same frozen 8B on the same H, yields
positive certified value; C1 sits between C0 (own description) and C2 (free induction) per metric,
with the size-11 deterministic menu arm agreeing in sign.

**What it licenses:** this is the rung that makes MCQ *commensurable* — recognition cashed out in
bits on H, same units as C2, no more units-wall between channels. If C1 ≈ C0 for a metric family,
the lexical handle suffices (taste-as-index gets behavioral grounding); if C1 ≫ C2, criteria are
recognizable-but-not-inducible; C2 > C1 metrics are learnable-from-behavior-but-not-named. The
C0/C1/C2(/C3) profile per metric IS the articulability decomposition the whole project is after,
finally on one scale.

**Conclusive?** The ordering per metric, yes — same panels, same executor, same H, same controls,
so within-metric rung comparisons are clean. Cross-family decoder comparisons remain off the table
(same-family rule); Llama 8B→70B within-family (Phase C) is the only scaling claim.

## 5. What stays non-conclusive even in the best case

1. **Everything is instrument-relative.** Caps and values are properties of (criterion × frozen
   executor × panel × H). "At most B bits" means through THIS channel. The E2L lesson applies: the
   wall can recede with capability. Quote every bound with its instrument.
2. **The survivor set is selected.** Teaching-balance exclusions (near-constant 8B scorings) are
   recorded, not hidden — report the denominator. The exclusions are themselves a §2-null-style
   finding about the executor, not about the criteria.
3. **FAST is screening.** Nothing from the FAST lane is ever quoted; promotion bias is real and
   the CERT population is a fresh measurement for exactly that reason.
4. **No cross-channel unit mixing** except where v14 built commensurability (C1's execute step).
   Behavioral bits and MCQ lift stay in separate columns of any figure.
5. **Constructor scaling stays withdrawn** until Phase C produces the matched 8B→70B same-family
   rows. The 13.7× number does not come back under any outcome.
6. **This is an LM-instrument result about articulability of human-mined criteria** — not a human
   study, and not a claim that tacit-under-our-instrument = tacit-for-humans.

## 6. The one-paragraph optimistic abstract (if every gate passes)

> We freeze a demonstration-channel instrument (fixed executor, fixed panels, held-out
> re-execution) and certify, per criterion, an exact structural upper bound on how much of a
> human-derived evaluative criterion can be transmitted through k labeled examples — across 35
> criteria in 7 domains. Against these ceilings we measure achieved transmission through four
> articulation rungs (own description, menu recognition, free induction, quantized code), all
> bottoming out in executed verdicts on the same held-out set with matched blind and shuffled
> controls. Criteria differ by an order of magnitude in both ceiling and attainment; recognition
> and execution dissociate in both directions; and a measurable fraction of criteria are
> certified transmissible while others fail against high ceilings — a quantitative, per-criterion
> map of what preference language can and cannot carry.

That abstract needs: §1 caps + §2 permutation-positive median + §3 lift>0 + §4 orderings + all
gates. Any subset that lands still supports the corresponding subset of claims — the caps alone
(§1) are publishable structure even if every achieved value stays small, because "small against a
small certified ceiling" is itself the finding: the bottleneck is the channel, not the decoder.
