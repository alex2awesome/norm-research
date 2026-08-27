# RUNG 1 RESULTS (descriptive) — selection regret on real items

Date: 2026-08-22. Design + addenda: notes/2026-08-21__rung12_design_gap_consequences.md
(frozen before run). Artifacts: methods/taste_decomposition/results/rung1_*.json
(+ per-row score npz from the ties sweep). All 13 F2 cells ran (bbc_mostread
has no mac artifacts — excluded). Numbers below are the collated table
(fusion/rung1_collate.py).

## Per-cell table

| cell | mode | rel gap % | regret [CI95] | disagree d:b | ratio |
|---|---|---|---|---|---|
| mathse_accepted | grouped (992) | 8.3 | **+.068 [+.029,+.105]** | 229:162 | 1.41 |
| press_verdict | grouped (8) | 10.9 | +.000 [0,0] | 0:0 | — |
| hashtagwars | grouped (8) | 12.5 | +.125 [−.25,+.50] | 2:1 | 2.0 |
| nc_responded | grouped (20) | 13.3 | +.100 [−.05,+.30] | 3:1 | 3.0 |
| nc_outcome | grouped (69) | 14.5 | +.101 [−.04,+.25] | 16:9 | 1.78 |
| jokes | grouped (10) | 15.0 | +.100 [.00,+.30] | 1:0 | — |
| mathse_vote | grouped (1140) | 25.5 | +.014 [−.018,+.046] | 196:180 | 1.09 |
| peer_curation | pairwise 460k | 36.0 | **+.065 [+.019,+.107]** | 1.33:1 | |
| peer_verdict | pairwise 383k | 39.4 | **+.109 [+.077,+.140]** | 2.07:1 | |
| peer_revealed | pairwise 57k | 41.7 | **+.171 [+.125,+.217]** | 4.41:1 | |
| cw_community | grouped (523) | 43.5 | **+.061 [+.011,+.107]** | 101:69 | 1.46 |
| cap_finalist | grouped (46) | excl (NYer) | +.065 [−.09,+.24] | 9:6 | 1.5 |
| nc_agree | grouped (65) | excl (design-cond.) | +.092 [−.06,+.25] | 16:10 | 1.6 |

## What is established vs what is not

**Established (strong):** the dense pick beats the articulated pick on
disagreements in EVERY cell that has disagreements — 12/12 cells with
ratio ≥ 1.0, regret ≥ 0 in 13/13 (sign test across cells p ≈ .0005). Five
cells individually significant. The quotable sentence survives:
"when the two instruments disagree about which paper the community will
cite, the dense pick is right 4.4:1 — judged by actual citations"
(peer_revealed, 12,565:2,852 over 57k pairs).

**Weak / NOT established:** the preregistered cross-cell monotonicity.
Spearman gap↔win-ratio ρ = +.29 (p=.39, n=11); gap↔regret ρ = +.15 (n.s.);
grouped-only ρ ≈ 0. Signed direction is positive but far from significant.
Two rank-breakers: mathse_vote (gap 25.5, ratio 1.09 over 1,140 groups —
big-n, genuinely small) and mathse_accepted (gap 8.3, ratio 1.41 — bigger
consequence than its gap suggests). Caveats: regret magnitudes are not
comparable across grouped cells (group size / pos-rate differ); several
grouped cells have <25 decidable groups (win ratios like 3:1 are 4 events).
The pairwise peer cells + the two big grouped cells DO order with gap
(1.09 → 1.33 → 1.41/1.46 → 2.07 → 4.41 vs gaps 25.5/36.0/8.3/43.5/39.4 —
mixed at the low end, clean at the top end).

## Ties / indistinguishability (Addendum B, partial — sweep in flight)

cw_community (only cell with a V arm so far): abstention@1% rank
quantization = V 5%, VA 4%, dense 6% of decidable groups; groups where BOTH
articulated instruments abstain while dense speaks: 0.4% (2 groups, dense
right in both). mathse cells: VA 3-4%, dense 2%; dense-only-granularity mass
2.5-3.6% of groups with dense hit .61-.64. Early read for these cells: the
articulated instruments' failure mode is "speaks but wrong", not "cannot
rank" — the regret is not explained by tie-abstention. Peer pairwise cells
pending; full q-grid in results/rung1_ties_<cell>.json.

## Bugs hit (all failed loudly, no silent contamination)
- cells.py module collision when running two closure cells in one process →
  one cell per process now; the OOF alignment gates caught it.
- mathse artifacts live on sk3 (box=sk3), peer cells are ntitle-singleton →
  Addendum A pairwise mode.
