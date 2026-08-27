# EXP-EXAMPLES-BANK-1 — AMENDMENT 2 (selection simplification), 2026-08-17

Amends the frozen prereg (c42a8f54db2e6f54, Amendment 1 49dd908dacce36f9). User directive:
the crowd machinery inside SELECTION (crowd-confident seeds; candidate pool restricted to
crowd-ambivalent items, .3<consensus<.7) is an unablated heuristic — revert to the simpler
design: choose positive and negative examples solely by measured flip impact.

## Changes (selection stage only)
1. NO crowd anywhere in selection. Seeds removed: the example set starts EMPTY.
2. Candidate pool = 24 items drawn by seeded random (stable hash of metric name, seed 0) from
   train-A items with non-empty text — no consensus filter. Each item proposable with either
   polarity (unchanged: polarity chosen solely by measured impact).
3. Greedy loop unchanged: rounds of 8 trials, accept the best addition iff train-A balanced
   gain >= .01 AND train-B does not degrade; cap 12; exemplar items masked; selection key
   remains the 2-voter bank key (evaluation key remains LOFO primary per Amendment 1);
   null control every 3rd metric unchanged.
4. All 236 metrics re-run under this design (including those previously skipped for empty
   ambivalence pools — that skip class no longer exists). The crowd-era selection state is
   RETAINED on disk (renamed, never deleted) but is not evaluated and not reported beyond a
   count.
5. The LEGACY 38/56-metric rescore is unchanged (its stored sets are the object being
   corrected; labeled legacy-design in the artifact).

Evaluation, scoring keys, silver cross-check, gates, readouts: unchanged from Amendment 1.
