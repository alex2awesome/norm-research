# Unit-recombination bottleneck — why Arm C compiled 1 of ~30 units (all 3 datasets)

Diagnosis of `run_unit_recombination.py` (Arm C) against `runs/<ds>/unitrecomb/{proposals.jsonl,result.json,rescore.jsonl}`
for hover, hotpotqa, aime2025. Read-only. All numbers reproduced by `/tmp/unit_diag.py`.

**Headline:** the "1 unit compiled" is **overdetermined by the harness geometry, not by the units.**
Three structural causes stack (budget starves the greedy stage → the greedy examines only 4 of the 11
remaining units in round 2 on every dataset; the 15/8-item selection panel is noise-dominated so the one
unit it does pick is a coin-flip that fails to generalize; the pre-filter to "top 12" is near-random).
The two hypotheses the plan leaned on — **acceptance threshold too strict, and unit redundancy — are NOT
supported by the data.** The proposed **paired sign-test fix would make it strictly worse (0 units compile).**

Per-dataset facts (panel P = dev-panel size; val = final val-split size):

| ds | P | screen×n | base panel | final panel | **seed val100** | **final val100** | best 1-unit val100 | units compiled |
|---|---|---|---|---|---|---|---|---|
| hover | 15 | 8×30 | 0.667 | 0.867 | **0.760** | **0.730** ▼ | 0.820 | 1 |
| hotpotqa | 15 | 8×30 | 0.733 | 0.800 | **0.800** | **0.810** | 0.830 | 1 |
| aime2025 | 8 | 5×29 | 0.000 | 0.125 | **0.294** | **0.294** = | 0.471 | 1 |

▼ hover's one compiled unit made the val-100 score **worse than the untouched seed** (0.73 < 0.76): a false
positive selected off the 15-item panel. aime's compile added nothing on val (0.294 → 0.294) while a single
different unit in the same pool reaches **0.471** on val. So the arm is not just under-compiling — it is
**mis-selecting**, and real value (esp. aime +0.18, hover +0.06) is left on the table.

---

## Ranked by explanatory weight

### 1. Budget starves the greedy stage → round 2 is truncated at 4-of-11 on ALL three (deterministic) — DOMINANT

The reserve formula `greedy_reserve = panel*(1 + 12 + 4)` (line 120) literally plans for **one base eval + one
full 12-unit round + only 4 probes of a second round.** Replaying the budget confirms the second round dies
after exactly 4 units everywhere:

| ds | budget₀ | screen | base+k1 | **rem. for round 2** | round-2 units examined | of remaining |
|---|---|---|---|---|---|---|
| hover | 500 | 240 (48%) | 195 | 65 → 4 evals | **4** | of 11 |
| hotpotqa | 500 | 240 (48%) | 195 | 65 → 4 evals | **4** | of 11 |
| aime2025 | 283 | 145 (51%) | 104 | 34 → 4 evals | **4** | of 11 |

Screening burns ~half the entire budget to rank units that (see #3) it cannot rank. After round 1 picks its
single unit, only 4 of the 11 survivors are affordable before the budget hits the val reserve and stops. This
mechanism is **dataset-independent and deterministic** — it explains the otherwise-suspicious "exactly ~1 on
all three" pattern far better than any noise argument. Even with a perfect pool, this harness caps compiled
units at ≈1.

**Fix implied:** *larger declared budget* **and** *init-from-GEPA-best* (D2 — start the compile from a strong
prompt so it's a superset by construction, not a from-seed climb) **and** reallocate budget out of screening
into greedy. Strongly supported.

### 2. The 15/8-item selection panel is noise-dominated → the accepted unit doesn't generalize — STRONG

`MIN_GAIN=0.01` is not "too strict" — on a 15-item panel it is **0.15 items**, looser than a single item, so
*every* +1-item unit clears it (all 12 hover round-1 units cleared it). The real problem is the opposite: the
threshold is far below the panel's own noise floor.

- sd of a panel mean ≈ √(p(1−p)/P): **hover 0.122 (1.8 items), hotpot 0.114 (1.7 items)** — i.e. ~1.7–1.8
  items of pure sampling noise vs a 0.15-item acceptance bar. The single greedy comparison is noise, not signal.
- Consequence: hover's accepted unit scored +3 items on the panel but **0.73 on val-100, below the 0.76 seed.**
  The panel picked a loser. (The best-val unit, 0.82, *was* in the top-12 but the noisy panel didn't rank it first.)
- Panel **saturation** compounds it: after 1 unit hover sits at 13/15, hotpot 12/15 — almost no room for a
  second unit to show +1 item, so round 2 shows ties/losses (hover: 3 of 4 examined tie the incumbent exactly;
  hotpot: all 4 lose; aime: 3 lose, 1 ties).

**Requested sign-test simulation (binomial sign test on discordant panel items, one-sided, accept at p<0.1):**

| ds | accepted unit wins–losses | its sign-test p | **# units accepted at p<0.1 (k1+k2)** |
|---|---|---|---|
| hover | 4–1 | 0.188 | **0** |
| hotpotqa | 1–0 | 0.500 | **0** |
| aime2025 | 1–0 | 0.500 | **0** |

**Zero** units reach p<0.1 on any dataset — including the units actually accepted. With base accuracy 0.67–0.73
only ~4–5 items are ever available to "win," so wins cap at ~5 and the sign test is **structurally underpowered**
on a 15/8-item panel (you'd need ≥4–0 or ≥6–1). A paired sign-test acceptance rule would therefore compile
**0 units, strictly worse than the current 1.**

**Fix implied:** *bigger dev panel* (the binding constraint is P, not the acceptance rule) and validate
survivors on the val split, not the 15-item panel. Strongly supported. → **The plan's "paired per-item
sign-test acceptance" fix is NOT supported — it degrades the arm to 0 compiles unless the panel is enlarged
first; panel size is the lever, not the test.**

### 3. The behavioral screen is uninformative for greedy value → "top 12" is near-random — MODERATE

Requested Spearman(screen score, greedy round-1 score) over the 12 units:

| ds | Spearman(screen, k1) | perm p | screen distinct values (of ~30) | 12th-place (boundary) score | # units tied at boundary |
|---|---|---|---|---|---|
| hover | **−0.495** | 0.136 | 4 | 0.625 | 14 (24 at/above) |
| hotpotqa | +0.251 | 0.410 | 5 | 0.375 | 22 (26 at/above) |
| aime2025 | +0.378 | 0.471 | 2 | 0.200 | 22 (29 at/above) |

None significant; hover is negative. The 8/5-item screen quantizes to 2–5 distinct values, so the "top-12"
cut sits on a massive tie (14–22 units share the boundary score) — the 12 units handed to the greedy are
essentially a **random draw** from the 24–29 tied units. The screen *does* weakly track standalone val-100
(hotpot ρ=0.47 p=.008, aime ρ=0.43 p=.022; hover ρ=−0.08 n.s.), i.e. it has some *standalone*-value signal —
but standalone value ≠ conditional/greedy value, and the greedy panel scores themselves don't track val
(ρ=0.13/0.11/0.17, all n.s.). Screening spends ~half the budget (#1) to produce a near-random shortlist.

**Fix implied:** shrink or drop the screen and redirect that budget to the greedy stage (or screen on a much
larger panel). Supported. This is largely the same lever as #1–#2 (bigger panels / more greedy budget).

### 4. Headroom is REAL and mis-captured — this rules OUT the "GLM saturation" hypothesis — MODERATE

| ds | seed val100 | best single-unit val100 | # of ~30 single units beating seed on val100 | compiled-final val100 | **regret vs best 1-unit** |
|---|---|---|---|---|---|
| hover | 0.760 | 0.820 | 19/30 | 0.730 | **−0.090** |
| hotpotqa | 0.800 | 0.830 | 16/30 | 0.810 | −0.020 |
| aime2025 | 0.294 | 0.471 | 18/29 | 0.294 | **−0.177** |

16–19 of ~30 single units beat the seed on val-100, and the best single unit is in the screen-top-12 in all
three cases — **the value is present and survives screening; the greedy just fails to select it.** aime is
nowhere near saturated (a single unit lifts val by +0.18). hover/hotpot sit against a soft ceiling ~0.82–0.83,
so absolute headroom is smaller (~6 points), but it is non-zero and the arm captured none of it (hover went
backwards). **This is selection error, not GLM saturation.**

**Fix implied:** *init-from-GEPA-best* (start above the seed) + validate on val, so the compiled prompt is a
superset of a strong candidate rather than a noisy from-seed climb. Supported. → The plan's hypothesis (d)
"GLM saturation headroom on hover/hotpot" is **only weakly true (small absolute ceiling on those two) and
false for aime; it is not the bottleneck.**

### 5. Unit redundancy is NOT the cause — token-Jaccard of the top-12 is LOW — RULED OUT

Requested token-Jaccard similarity matrix of the top-12 unit texts:

| ds | mean pairwise J | max pairwise J | pairs ≥0.6 | **clusters @0.6 (of 12)** | full-pool mean J / clusters |
|---|---|---|---|---|---|
| hover | 0.075 | 0.667 | 1 | **11 / 12** | 0.059 / 28 of 30 |
| hotpotqa | 0.117 | 0.778 | 3 | **10 / 12** | 0.092 / 26 of 30 |
| aime2025 | 0.035 | 0.500 | 0 | **12 / 12** | 0.043 / 28 of 29 |

The top-12 units are **lexically diverse**, not clause-splits of one prompt: 10–12 distinct clusters at the
0.6 threshold, mean pairwise overlap 0.04–0.12. Only 1–3 near-duplicate pairs exist total. Deduping/clustering
would remove at most those 1–3 pairs and would **not** unlock additional compiles — the greedy stops for
budget/noise reasons (#1–#2), not because it exhausted a redundant pool.

**Fix implied:** none needed here. → The plan's hypothesis (c) "unit redundancy collapses conditional value"
and the candidate fix **"unit dedup/clustering" are NOT supported by the redundancy evidence.**

---

## Fixes: what the evidence supports vs. does not

**Supported (in priority order):**
1. **init-from-GEPA-best (D2)** — addresses #1 and #4: the compile starts as a superset of a strong prompt,
   so it clears the seed by construction and the greedy only needs to find *marginal* additions.
2. **Larger declared budget + reallocate budget out of screening into greedy** — addresses #1 directly (round
   2 currently sees 4 of 11 units on every dataset); shrinking/removing the near-random screen (#3) frees ~half.
3. **Bigger dev panel** — addresses #2: current sd ≈ 1.7–1.8 items dwarfs the 0.15-item threshold and produced
   a val-negative pick on hover. Validate greedy survivors on the val split, not the 15/8-item panel.

**NOT supported by the evidence:**
- **Paired per-item sign-test acceptance** — 0 units reach p<0.1 on any dataset (panel too small to reach
  significance); this rule compiles **0 units, worse than 1**. Panel *size* is the lever, not the acceptance test.
- **Unit dedup/clustering** — top-12 token-Jaccard is 0.04–0.12 with 10–12 clusters; the pool is already
  diverse, so dedup cannot unlock more compiles.
- **"GLM saturation" as the cause (hyp d)** — headroom is real (aime +0.18, hover +0.06 available and unused);
  the failure is mis-selection, not a ceiling.

**Weak / conditional:**
- **remove/swap moves** — could in principle swap hover's val-0.73 unit for the val-0.82 one, but *only* if the
  selection panel can tell them apart, which #2 shows it cannot. Swap helps **only combined with a bigger/val
  panel**; on the current 15-item panel it would swap on noise.

---

## Independent re-verification (2026-07-20 audit pass)

Every quantitative claim above was independently recomputed from the raw logs
(`proposals.jsonl`, `rescore.jsonl`, `result.json`) by a second agent: budget arithmetic
(48/48/51% screen burn, round-2 truncation at exactly 4-of-11 on all three datasets), panel-sd
(0.122/0.114 ≈ 1.7–1.8 items), all Spearman correlations WITH permutation p-values
(−0.495/+0.251/+0.378, p .137/.413/.470), tie counts, single-unit headroom (19/16/18 of ~30
units beat seed on val; regret −0.09/−0.02/−0.18), and the 0-compile sign test. ALL CONFIRMED.

**One framing caveat found:** the claim that the compiled prompt went *backwards* vs the seed
(hover 0.73 < 0.76) mixes two measurement passes — run-time `final_val` vs `rescore.jsonl`.
Rescored like-for-like in a single pass, the identical hover compiled prompt scores 0.760 (= the
seed exactly; aime rescores 0.353 > its 0.294 run-time number). The measurement-robust claim is
the weaker one: **the compiled prompt captures far less than the best available single unit**
(like-for-like regret −0.06 hover / −0.118 aime). Direction and every v2 design decision are
unaffected; do not quote the "went backwards" framing externally.
