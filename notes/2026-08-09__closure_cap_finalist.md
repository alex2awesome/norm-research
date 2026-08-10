# cap_finalist — Layer-3 articulation-closure campaign (LANE A, cell 2)

Cell: **New Yorker cartoon caption contest, EDITOR track** — y = 1 if the caption was
selected as one of the three finalists for its contest, 0 if it is a HARD NEGATIVE drawn
from the same contest. 5,218 captions, 227 contests, pos-rate .1299.

Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + FREEZE DECLARATION + Addenda 1–4.
Queue: `notes/2026-08-09__full_sweep_queue.md` (LANE A, GPU 5).
Reference campaign whose machinery and rulings this one inherits:
`notes/2026-08-09__closure_jokes.md` + `methods/taste_decomposition/closure/jokes_community/`.
Code + artifacts: `methods/taste_decomposition/closure/cap_finalist/`.
Rounds 1–2 (map-focused, pre-standard) live in
`methods/taste_decomposition/closure/maps_batch1/cap_finalist_r{1,2}_*`.

Abbreviations spelled out, per the standing rule. **V** = 16 deterministic label-blind
surface features. **A** = the Gemma-4-31B-judged criterion bank (364 raw columns).
**VA_lin / VA_nl** = grouped-OOF logistic / HistGradientBoosting aggregation of the same
V+A matrix, seed-mean over {0,1,2}. **T** = the dense standard. **Δ_beyond** = T − VA_nl.
**MONITOR** = the frozen decision population. **HONEST** = every dense-held-out row.
**Track A** = candidate-real criteria mined to close the residual. **Track B** =
suspected-spurious channels mined to discount it. **Good-Turing missing mass** = the
share of the concept pool represented by singletons. **LOPO** = leave-one-proposer-out.
**TIER S** = sealed independently-proposing fleet (the only tier that may feed
Good-Turing). **TIER D** = directed decomposition components. **TIER R** = re-measurement
of existing criteria (this campaign's view-repair pass; defined below).

---

## 0. What was already on disk (nothing rebuilt)

| artifact | value |
|---|---|
| Layer-1 ledger `results/cap_finalist_layer1.json` | n 5,218 · V_lin .6247 / A_lin .6299 / VA_lin .6508 / **VA_nl .6800** (seeds .6787/.6804/.6808) / T .6252 · Δ_beyond −.0548 |
| Master grid `results/vat_fullgrid_cap_finalist.json` | n_E 1,055 · **n_groups_E 46** · T .6124 · VA_nl E-refit .5806 · **VA_nl fullfit@E .6666** · VAT_nl .6077 · V3 k10 .6707 |
| Closure splits (frozen r1) | FIT+MINE 4,692 / 204 contests · MONITOR **526 / 23 contests / 66 positives** · mining slice M 529 · HONEST 1,055 / 46 contests |
| Round-0 context | bank 345 features · VA_nl MONITOR .6919 · VA_nl HONEST .6695 · T MONITOR .6497 · T HONEST .6124 |
| Rounds 1–2 | see §1 — they are **not at current standard** and §1 is the registered ruling on what that means |

**One T convention exists on this cell.** The dense arm is a single probability column
(`closure/samerows_preds/cap_finalist_dense_preds_slim.csv`) that reproduces the master
ledger's T = .6124 on the 1,055 dense-held-out rows exactly. There is no
meanAUC-versus-ensemble pair to keep apart, unlike jokes_community.

**The residual is NEGATIVE before any mining.** Δ_beyond = **−.0422 on MONITOR** and
**−.0571 on HONEST** at round 0, with a group-bootstrap band of [−.1074, −.0078],
p(>0) = .012. The articulated bank already beats the dense standard here. The closure
question for this cell is therefore not "can the residual be closed" — it is already
closed — but the two the coordinator posed: **does mining widen the bank's lead, and
what, if anything, is the dense model's remaining increment carried by.**

---

## 1. REGISTERED RULING on rounds 1–2 (written before round 3 ran)

Rounds 1 and 2 exist and produced a two-round dual-track map. Inventorying them against
the standards now in force turns up **three protocol gaps and two defects**, all verified
on disk. Every one is recorded here, with what follows from it, before any new round.

### 1.1 Protocol gaps versus the current standard

**(a) Fleet was P = 4 / 2 families, not P = 8 / 3 families.**
`cap_finalist_r{1,2}_proposals_fleet.json` list 12 slots; both GLM slots are `MISSING` in
both rounds, so the realised fleet was claude_opus, claude_sonnet, codex_luna_a,
codex_luna_b — four proposers in two families, 100 proposals per round instead of 200.
This sits exactly on the FREEZE DECLARATION's own degradation floor ("degrade gracefully
to P ≥ 4 / ≥ 2 families under GLM rate limits, recorded"), so the rounds are *within* the
frozen protocol, at its weakest admissible setting. It is worth noting that these rounds
ran during the 2026-07-25 → 08-06 GLM dead period; the Lite plan is live again now, so
rounds 3+ run at full P = 8.

**(b) Species identity by embedding-τ only (τ = .79); no strict two-judge blind merge.**
The freeze's identity rule is blind pairwise adjudication, never embedding-τ, and the
jokes campaign then measured that **τ fails in both directions** — over-merging Track A
and under-merging Track B, in all four of its fleet rounds. The r1/r2 missing-mass
numbers (A .400 / .383, B .600 / .575) are therefore **superseded and not of record**.
They are kept in the results files, never quoted.

**(c) Audit probes were not corpus-matched and not realised-draw-chained.**
`cap_finalist_r1_audit_key.json` plants **"Explicit statement of the limitation or failure
regime"** as the quality-relevant probe — a PEER-REVIEW probe, scored against one-line
cartoon captions. This is precisely the failure mode the freeze names (probes authored
for the wrong genre make the auditor fail through no fault of its own). The reported 4/4
probe passes and the .04 / .00 misrouting rates are therefore not evidence of a
well-calibrated auditor. `cap_finalist/audit.py` now carries four caption-matched pairs
with the chained draw.

### 1.2 DEFECT 1 (material, new) — ITEM-VIEW MISMATCH

The A bank scored every caption as

```
CARTOON: <description of the published drawing>

CAPTION: "<text>"
```

(`datasets/humor/caption_multiy/score_va_gemma_captions.py:190`). `maps_batch1`'s
`score_gemma_maps.py` scored the MINED criteria as

```
CAPTION:
<text>
```

with **no cartoon at all**. A New Yorker caption is close to ungradeable without the
drawing it captions — the round-1 slice's top disagreement rows are "you should see the
breasts", "they're still looking for votes", "where we disagree is that exposed skin
implies consent". Every round-1 and round-2 mined criterion was therefore measured on a
**strictly weaker view than the bank it was being added to**.

The measured signature is consistent with the defect: of the 30 mined Track-A criteria,
**21 have alone-AUC below .5 on HONEST** and the best is .575 — against a jokes-cell best
of .682 on a matched view, and against this cell's own bank, which was scored WITH the
cartoon and reaches VA_nl .6695 on the same rows.

### 1.3 DEFECT 2 — the PROPOSERS were also blind to the cartoon

`stage1_slice.py` emits `text` only, so the sealed fleet proposed caption-quality criteria
while unable to see what the captions were captioning. Both halves are fixed for round 3+:
`cap_finalist/stage1_slice.py` carries a `cartoon` field and `cap_finalist/harness_maps.py`
renders it above every slice row.

### 1.4 THE RULING

1. **Rounds 1–2 COUNT as rounds** against the cap of 5 (the jokes precedent counts round
   numbers, not proposing rounds, toward the cap). They consumed mining budget and their
   A-routed criteria are in the bank.
2. **Their Δ values do not advance the stopping clock**, because they were measured on a
   mismatched item view. This is moot in the permissive direction and is registered as
   such rather than being used to buy a round: both gains (+.0155 MONITOR r1, +.0085
   MONITOR r2) are **above** ε = .005 anyway, so the clock reads **0** under either
   reading.
3. **Their missing-mass numbers are superseded** (τ-only, P = 4, 2 families).
4. **Round 3 is a VIEW-REPAIR pass (TIER R)** that re-scores all 50 round-1/round-2
   criteria on the matched cartoon+caption view and recomputes the r1 and r2 Δ from the
   repaired scores. Like a TIER-D decomposition round it runs **no sealed fleet**, so by
   the registered 2026-08-08 rule it **cannot** advance the stopping clock, and by the
   two-tier rule it contributes **no** Good-Turing mass. It is a re-measurement of
   existing criteria, not a proposal of new ones.
5. **Routing is not re-decided.** The blind audit routed on criterion TEXT, which the view
   defect does not touch. The corpus-mismatched probe pool is recorded as a caveat on the
   reported probe-pass and misrouting rates for r1/r2; rounds 3+ use matched probes.
6. The pre-repair `_scores.npz` files are preserved as
   `cap_finalist_r{1,2}_scores.MISMATCHEDVIEW.npz` and never deleted.

---

## 2. Design decisions taken before round 3

**Splits are INHERITED UNCHANGED** from round 1 (`cap_finalist_splits.json`): stable-hash
on the contest key, `sha256(group)/2**256 >= .80 AND dense_split in {eval,test}` →
MONITOR, so MONITOR ⊂ dense-held-out as the freeze requires. Group overlap between
FIT+MINE and MONITOR: 0. Re-cutting them would break comparability with rounds 1–2 for no
gain.

**POWER CAVEAT, registered here rather than discovered later.** MONITOR holds **526 rows,
23 contests and 66 positives**. The two existing rounds produced group-bootstrap gain
bands of **[−.0102, +.0448]** (r1) and **[−.0154, +.0323]** (r2) — half-widths of ~.024 to
~.028, i.e. **5× the ε = .005 the stopping rule is applied at**. Worse, r2's MONITOR gain
(+.0085) and its HONEST gain (−.0151, p(>0) = .066) point in **opposite directions**. This
cell cannot resolve ε-scale movements, and that is a property of a 66-positive decision
population, not something a better estimator fixes. Three consequences, fixed now:
* the **stopping rule still reads MONITOR** (TIER 1) as the freeze requires, but every
  round reports HONEST beside it and a round is only described as a real increment when
  the two agree in sign;
* the **group band is the quoted band**; the item-level band is printed beside it and
  never substituted for it (23 contests is coarse by construction);
* no round-over-round movement under ~.025 is called a discovery on this cell, whatever
  the point estimate does.

**TIER 2 readout = within-contest AUC.** y is a *within-container* selection — exactly 3
finalists among ~23 entries — so the within-contest readout is the one that matches the
y-definition. Reported every round, never substituted for TIER 1.

**Roster tier.** Matched Δ_beyond is negative here, well below the FREEZE DECLARATION's
+.02 full-dual-track line, so this is a **map-focused dual-track cell**: both tracks run
at full budget (k_A = 15, k_B = 10, sealed fleet P = 8) and **the spurious map plus the
dense-increment question is the headline**, with the Track-A curve reported in full. Fixed
now, before any proposal.

---

## 3. Round 0 — FREEZE ADDENDUM 4 observed-covariate line (`cap_finalist_contest_line.json`)

Run before round 3 because two of its findings are structural and change what the fleet
should be asked for.

**The position-in-container family is STRUCTURALLY NULL on this cell.** The hard-negative
pool takes exactly 3 finalists and ~20 hard negatives per contest, so the label rate is a
constant .1304 in 225 of the 227 contests. A container-level covariate therefore *cannot*
predict y. Measured: contest number alone-AUC **.5022**. This is a design fact about the
pool, not a null result from a search, and it is recorded so no round spends budget
re-deriving it.

**No arrival-order ordinal exists in this corpus at all.** The raw scrape
(`datasets/humor/newyorker_caption_ratings.csv.gz`) carries a per-contest `rank`, and its
row index equals `rank` within every contest (verified), so file order is the CROWD
RANKING, not a submission order. There is no timestamp and no entry-stream index. Recorded
as ABSENT rather than searched-for-and-null. Addendum-4's MODE 3 is still asked of the
fleet — in the form that *can* carry signal here, the era of the contest series an entry's
idiom belongs to — and any fingerprint it produces is checked against the observed contest
ordinal exactly as the jokes cell checked its era fingerprint against `created_utc`.

**The covariate that does carry is WITHIN-CONTEST CROWD RANK, and it runs BACKWARDS.**
Every caption has a crowd rating from the NEXTML rating experiment (98.3% coverage) — an
observed, non-text measure of the same item by a different judge population. Its
within-contest percentile rank has alone-AUC **.335** against y, i.e. |Δ| = .165 and
**inverted**: crowd-popular captions in this pool are *less* likely to be editor
finalists. That is the hard-negative construction showing through — the negatives were
drawn to be crowd-plausible — and it means **any Track-B channel that tracks generic
crowd-pleasingness will read sub-chance here**. This is the empirical reason the round-3
fleet gets a fifth Track-B mode (§4).

Full table in `cap_finalist_contest_line.json`.

---

## 4. Cell-specific brief changes for round 3 onward (registered before the fleet ran)

Three, all in `cap_finalist/harness_maps.py`, all recorded rather than silent:

1. **The cartoon is shown** — to the fleet (slice rows) and to the judge (scoring view).
   The fix for Defects 1 and 2.
2. **MODE 3 is re-aimed** at the contest-series era rather than an entry ordinal, for the
   structural reason in §3.
3. **MODE 5, new and cell-specific: GENERIC-PRIOR-GUESSABLE QUALITY.** This cell is the
   T₀ arm's **sole exception in the programme** — an untrained prior contributes +.0374 to
   fusion here, about a third of the fusion gain. So "the sort of caption a competent
   writer or a general-purpose model with no knowledge of this contest would nominate as
   good" is a live *nuisance* channel, and the fleet is asked for its textual fingerprint.
   Combined with §3's inverted crowd rank, the hypothesis this mode tests is that the
   generic prior and the editors' choice come apart in a nameable way.

**What the fleet is never told:** that the negatives are hard negatives from the same
contests, and that within-contest crowd rank runs .335 against y. Both are facts about how
the population was built; telling a proposer would be a design steer outside the freeze.
They are used only when the map is interpreted.

---

## 5. Per-round ledger

| round | kind | fleet | VA_nl MONITOR | gain | sub-ε | counts toward clock | clock | probes | misroute | GT mass A / B |
|---|---|---|---:|---:|:--:|:--:|---:|---|---|---|
| 0 | baseline | — | .6919 | — | — | — | 0 | — | — | — |
| 1 | proposing (**degraded**, mismatched view) | P=4 / 2 fam | .7074 | +.0155 | no | **no (§1.4)** | 0 | 4/4 *(genre-mismatched probes)* | .04 | τ-only, superseded |
| 2 | proposing (**degraded**, mismatched view) | P=4 / 2 fam | .7159 | +.0085 | no | **no (§1.4)** | 0 | 4/4 *(genre-mismatched probes)* | .00 | τ-only, superseded |
| 3 | **VIEW REPAIR (TIER R, exempt)** | none | — | — | — | **no — exempt** | 0 | n/a | n/a | n/a |

Δ_beyond MONITOR: −.0422 (r0) → −.0577 (r1) → −.0661 (r2). Δ_beyond HONEST: −.0571 (r0)
→ −.0668 (r1) → −.0518 (r2). Stopping clock **0**; cap 5 rounds.
*(Those are the AS-REPORTED r1/r2 figures, under the old screen and the mismatched view.
§7 supersedes them.)*

---

## 7. Round 3 — VIEW-REPAIR PASS (TIER R). The reconstruction of record.

50 criteria (25 from r1, 25 from r2; 30 Track-A, 20 Track-B) re-scored corpus-wide on the
matched `CARTOON: <desc>\n\nCAPTION: "<text>"` view. Gemma-4-31B offline batch, GPU 5
lane-pinned, **268,400 prompts**, 28 min, released rc=0. No fleet ran; TIER R, so no
Good-Turing mass and no stopping-clock effect.
Artifacts: `cap_finalist_r3_scores.npz`, `cap_finalist_view_repair_report.json`,
`cap_finalist_r3_viewrepair_results.json`.

### 7.1 The repair moved the measurements, substantially

Median rank-correlation between the old-view and new-view score columns is **.678 (r1)**
and **.776 (r2)** — i.e. a quarter to a third of the rank information changes when the
judge is shown the cartoon. The biggest movers are exactly the criteria that need the
drawing: "Clashing interpretive frames" ρ = **.204** (mean 5.97 → 9.15), "Elliptical
inference" ρ = .252, "Deadpan audacity" ρ = .310, "Coherent causal twist" ρ = .311, "Vivid
scene specificity" ρ = .441 (mean 3.46 → 6.36). These are not the same measurements.

### 7.2 A second, independent correction: the ENFORCED COLLAPSE GATE

This campaign runs the coordinator's enforced gate (modal > .98 dropped inside
`clean_fit`), which rounds 1–2 did not. On this cell it removes **56 of 345 bank-0
features** — 55 A criteria and 1 V column — because a one-line caption gives most rare
criteria nothing to bite on and the historic `off-modal < 5` screen lets a criterion at
modal .99 through at n = 5,218. **Bank 0 is 289 features, not 345**, and VA_nl on MONITOR
is **.6767, not .6919**. Every number in §7.3 is computed under one gate and one view.

### 7.3 Reconstructed Δ curve — REPAIRED view beside PRE-REPAIR view

Both computed by the same code on the same rows; the only difference is which score
matrix is read. MONITOR n = 526 / 23 contests / 66 positives; HONEST n = 1,055 / 46
contests. T: MONITOR .6497, HONEST .6124.

| | | **REPAIRED (cartoon + caption)** | **PRE-REPAIR (caption only)** |
|---|---|---:|---:|
| bank 0 | VA_nl MONITOR | .6767 | .6767 |
| **r1** | VA_nl MONITOR | **.6861** | **.6982** |
| | gain MONITOR [group CI] | **+.0094** [−.0133, +.0334] p .77 | **+.0215** [+.0005, +.0434] p .98 |
| | gain HONEST | **−.0016** | −.0004 |
| | within-contest MONITOR gain | +.0123 | +.0293 |
| | swap dC₊ / dC₋ | +.0037 / **−.0099** → **adverse** | +.0052 / −.0091 → adverse |
| **r2** | VA_nl MONITOR | **.6907** | **.7111** |
| | gain MONITOR [group CI] | **+.0046** [−.0186, +.0263] p .67 | **+.0129** [−.0105, +.0344] p .87 |
| | gain HONEST | **+.0021** | **−.0107** |
| | swap dC₊ / dC₋ | +.0031 / +.0006 → not adverse | −.0095 / −.0127 → not adverse |
| **cumulative** | MONITOR | **+.0140** | +.0344 |
| | HONEST | **+.0005** | **−.0111** |
| | Δ_beyond MONITOR / HONEST | **−.0409 / −.0540** | −.0613 / −.0423 |

**The headline of the repair: the mismatched view was manufacturing MONITOR-only gain.**
Pre-repair, two rounds of mining looked like +.0344 on MONITOR — and delivered **−.0111**
on the four-times-larger HONEST population. Repaired, the two populations agree: +.0140
and +.0005, both inside the noise this cell can resolve. A 66-positive decision population
plus a defective instrument produced a closure curve that pointed the opposite way from
the honest one. This is the concrete cost of the item-view defect, and it is the reason
the repair was run before any new round rather than after the campaign.

### 7.4 The dense model's residual: there is nothing left for it to carry

Stratification-free stacked increment on the repaired scores, cumulative nuisance set
(both rounds' Track-B channels, 20 columns):

| population | joint-B | dense | bank | **dense increment over B + bank** | bank increment over B + dense |
|---|---:|---:|---:|---:|---:|
| HONEST (n 1,055) | .6437 | .6124 | .6664 | **+.0019** [−.0054, +.0090] p(>0) .70 | **+.0247** |
| MONITOR (n 526) | .6722 | .6497 | .6907 | **+.0020** [−.0140, +.0160] p(>0) .61 | **+.0194** |

The coordinator's question for this cell was what the dense model's residual is carried
by. The answer at this point in the campaign is that **there is no residual to carry**:
conditioned on the articulated bank plus every named nuisance channel, the dense model
adds +.002 with a CI straddling zero on both populations, while the bank adds an order of
magnitude more (+.019 to +.025) over dense-plus-nuisance. The direction of the Δ_beyond
sign is not an artifact of one estimator.

### 7.5 Anchor battery — and a registered amendment to the V9 pass rule

Anchors K = 50 per class, in the same batch, scored on the same cartoon+caption view
(the scrambled anchor is shown against the pos caption's real cartoon).

* pos 3.402 / neg 3.467 / **scrambled 0.698**
* **coherent-vs-scrambled AUC (item mean) = .9994** — the strongest coherence separation
  in the programme so far.
* coherent-vs-scrambled AUC (**non-NA count**) = **.5000**, and the code's `pass_scrambled`
  flag therefore reads **false**.
* pos-vs-neg AUC **.485**.
* NA rate **.00005**, 0 all-NA rows, 0 all-NA anchors, **1 collapsed** criterion
  (r2 A12 "No self-commentary on its own funniness", modal .993).

**Registered amendment.** The V9 recommendation — score coherence on the non-NA count,
because a mean over "whichever criteria answered" selects for scrambles the judge could
still score — is **inapplicable on this cell, for a checkable structural reason**: this
judge emits NA on 5 prompts in 100,000, so the non-NA count is exactly 50 for every
anchor of every class and carries zero information by construction. It cannot discriminate
and its .5000 is not a failure signal. The **item-mean reading is the operative one on
this cell** and it passes at .9994. The non-NA reading stays in the report as the
diagnostic that established the degeneracy. Both are printed every round; if a future
round's NA rate rises above ~.02 the non-NA rule becomes live again and takes precedence.

**Manual inspection of the scrambled anchors was done, as V9 requires**, and the
proper-noun-survival defect V9 named IS present in mild form — "linda,", "ted,", "harry,",
"ikea", "dane" survive intact into the shuffled blobs. It does not rescue them: the
full texts read as unambiguous word salad ("ready there isn't never there body engagement
was an ring, was summer my") and the .9994 separation confirms the judge treats them that
way. Every scrambled anchor is dumped to `cap_finalist_r3_score_report.json`
(`anchors.anchor_scram_texts`) so the inspection is reproducible.

**pos-vs-neg .485 is expected here and is not an anchor failure.** These are HARD
negatives from the same contests as the finalists; the mean of 50 criteria is not supposed
to separate them, and if it did, the cell would not have a residual worth studying. The
jokes cell's .58–.69 came from bottom-quartile negatives. The coherence axis is the one
the battery certifies, and it certifies it.

### 7.6 STOPPING CLOCK — the registered ruling does real work, in the costly direction

Repaired, **r2's gain is +.0046, i.e. BELOW ε = .005**. Under the plain reading of the
stopping rule that would be the campaign's first sub-ε round and the clock would advance
to 1.

**It does not.** §1.4 point 2 registered — before the repaired numbers existed — that
rounds 1 and 2 do not advance the stopping clock, because they were degraded-fleet
(P = 4 / 2 families) rounds audited against genre-mismatched probes. The repair fixes the
item view; it does not retroactively make those rounds sealed-P=8 proposing rounds. The
clock therefore stays at **0**, and the campaign owes two consecutive sub-ε rounds at
current standard before it can terminate on the rule.

Recorded plainly because the registration cost a round here rather than saving one: had
the ruling been written after the repair, the temptation would have run the other way.

**Round-3 state:** bank **318 features** (289 base + 15 r1 A-routed + 14 r2 A-routed after
the gate), VA_nl MONITOR **.6907**, HONEST **.6664**, Δ_beyond **−.0409 / −.0540**.

---

## 8. Round 4 — the first sealed PROPOSING round at current standard

**Fleet: 16/16 slots, P = 8, 3 families, 200 proposals (120 A / 80 B). No degradation.**
Both GLM Lite keys smoke-tested live before the round (2.3 s and 3.0 s, thinking budget
512, `stop_reason: end_turn`). Sixteen distinct slice orderings, one per (slot × track)
salt. Slice rebuilt on the repaired 318-feature bank (VA_nl OOF FIT+MINE .6829 entering)
and — for the first time on this cell — **every slice row carried its cartoon**.

### 8.1 Species — τ fails in both directions here too

Two sealed blind judges (Sonnet + Opus) adjudicated **240** cross-proposer shortlisted
pairs; a pair merges only if **both** say SAME. **Both planted identity anchors passed for
both judges** (SAME/SAME, DIFFERENT/DIFFERENT).

| track | τ-only S_obs | **merged S_obs** | direction | τ-only mass | **merged mass (record)** | LOPO band |
|---|---:|---:|---|---:|---:|---|
| A | 35 | **75** | τ **over**-merged (45 edges) | .258 | **.450** | [.448, .552] mean .487 sd .029 |
| B | 44 | **39** | τ **under**-merged (41 edges) | .412 | **.300** | [.257, .357] mean .323 sd .031 |

**The two-directional τ failure replicates on a fourth cell.** jokes_community found τ
over-merging Track A and under-merging Track B in all four of its fleet rounds; cap_finalist
reproduces the same signed pattern on the first try, with the same strict two-judge rule
and both anchors passing. Any τ-only mass number on this cell is biased in a
track-dependent direction, and the r1/r2 τ-only figures (§1.1b) are superseded for exactly
this reason. Cross-proposer recapture .280 (A) / .385 (B): the A-side concept space is
wide open, the B-side is the more consumed of the two — the same asymmetry the jokes cell
reported.

### 8.2 Blind routing audit — and this cell's version of the craft/circulation boundary

Fresh blind Sonnet-class auditor, 29 items (25 selected + 4 planted from the two
**caption-matched** pairs drawn by the realised-draw chain; r4 drew the word-count /
withheld-explanation and proper-noun / register-clash pairs, r3's draw was the other two,
so no auditor saw a repeat).

* **Planted probes 4/4.** "Number of words in the caption" and "Contains a proper noun or
  brand name" → incidental; "Register of the words clashes with the situation drawn" and
  "Withholds the explanation the drawing seems to demand" → quality-relevant. Exactly as
  planted. This is the first calibrated probe pass on this cell.
* **Misrouting 2/25 (.08)**, and both were **Track B → quality-relevant**, the opposite
  direction from the jokes cell's re-routes.
* Frontier arbiter **upheld both re-routes**. Final **A = 17, B = 8, of which 4 mixed**.

**Both disputes are one concept: OBVIOUSNESS.** "Obvious first-thought pun" (B09) and
"Canonical first-thought angle" (B06) were proposed as MODE-3 position-in-container
channels — a caption's position in a crowded entry stream, fingerprinted as "the reading
hundreds of other entrants also had". The blind auditor and the arbiter both read the same
instruction as a judgement about the *caption's originality*, i.e. craft. This is
cap_finalist's analogue of the jokes cell's craft/circulation boundary, and it lands in the
same place: *the nameable, judgeable thing in the neighbourhood of a container ordinal is
the craft that co-occurs with it.* Recorded as a substantive cell finding, not a fleet
failure. Note what it costs the Addendum-4 programme on this cell: the fleet DID reach for
position-in-container, and the routing protocol correctly moved its two best attempts out
of the nuisance set and into the bank.

### 8.3 Scoring

Gemma-4-31B offline batch, GPU 5 lane-pinned, 25 criteria × 5,218 rows + 150 anchors =
**134,200 prompts**, 16 min, released rc=0. Matched cartoon+caption view.
Anchors: pos 4.627 / neg 4.873 / **scrambled 0.577**; **coherent-vs-scrambled (item mean)
.9998**; non-NA-count reading .5000 again (the structural degeneracy of §7.5 — NA rate
.00034); pos-vs-neg .433 (hard negatives, as expected). **1 collapsed** criterion
(A13 "Meta-humor about the joke/genre itself", modal .984), dropped by the enforced gate.
0 all-NA rows, 0 all-NA anchors.

### 8.4 RESULT — the first sub-ε PROPOSING round

**TIER 1, MONITOR (n = 526, 23 contests): VA_nl .6907 → .6897, gain −.0010**
[group CI **−.0218, +.0178**], p(>0) = .46; item band [−.0200, +.0180], p = .45.
Bank 318 → 334 features. Seed spread .0221.
HONEST gain **+.0011** [−.0115, +.0136]. Within-contest gain +.0011 (MONITOR) / +.0069
(HONEST). Δ_beyond MONITOR −.0409 → **−.0399**; HONEST −.0540 → **−.0551**.

**−.0010 < ε = .005 and the CI straddles zero on both populations. This is the campaign's
first sub-ε round at current standard, so the stopping clock advances to 1 of 2.**

Best new Track-A criterion: **"Comic economy: punch lands with minimal, loaded wording"
alone-AUC .592** on HONEST — respectable in isolation and still not *additional* to a
318-feature bank. Then "Canonical first-thought angle" .545 (the arbiter-promoted one) and
"Conversational deadpan delivery" .533.

### 8.5 SWAP FLAG (coordinator standing request) — adverse

| round | dC₊ | dC₋ | dρ | adverse by registry rule (dC₊>0, dC₋≤0) |
|---|---:|---:|---:|:--:|
| 1 (repaired) | +.0037 | **−.0099** | +.0195 | **YES** |
| 2 (repaired) | +.0031 | +.0006 | +.0151 | no |
| **4** | **+.0024** | **−.0008** | −.0077 | **YES** |

Flagged as instructed. Round 4's point estimate is *negative*, so the reading is the same
one the jokes campaign gave for its terminal rounds: there is no real gain left, and the
small movement the aggregator does make is agreement with the dense model's ordering
rather than independent signal. The sub-ε call and the swap flag agree.

### 8.6 Track B — the map, and the dense increment turns negative

8 channels after routing (4 mixed). Joint spurious-alone **.677 MONITOR** (HistGB), above
the .65 matched-sampling trigger; STRICT no-mixed .616. Strongest channels: **"Generic
well-formed complete-sentence 'caption-shaped' joke" and "Terse fragment/staccato
delivery" .528**, "Topical news and era reference" .469, "Live-news political reference"
.478, "Uniform lowercase / no terminal punctuation transcription style" **.500 exactly**.

Two readings worth recording:
* **MODE 5 (generic-prior-guessable quality) was named by 7 of 8 sealed proposers** — the
  joint-top consensus channel of the round — which is the fleet independently confirming
  the T₀ observation that motivated the mode. Its alone-AUC is only .528, but see §3: on
  this pool a generic quality prior is *anti*-predictive by construction, so a channel
  that tracks it cannot show a large positive AUC and .528 is the wrong statistic to judge
  it by. Its value is as a discount column, and it is in one.
* **The transcription-convention channel reads exactly .500.** The compiled-entry-list
  typography that MODE 4 asked for carries no signal at all — a clean null on the one
  Track-B family that is unambiguously pure nuisance.

Stratification-free stacked increment (readout of record):

| population | joint-B | dense | bank | **dense increment over B + bank** | bank increment over B + dense |
|---|---:|---:|---:|---:|---:|
| HONEST | .6368 | .6124 | .6675 | **−.0019** | **+.0315** |
| MONITOR | .6770 | .6497 | .6897 | **−.0034** | **+.0229** |

The dense increment over everything nameable is now **negative on both populations**.
Combined with §7.4 (+.0019 / +.0020 at round 3), the honest statement is that it is
**zero to within the resolution of this cell**, and the bank's increment over
dense-plus-nuisance is 10-15× larger in the other direction.

Missing mass (strict merged, record): **A .450** [.448, .552], **B .300** [.257, .357].

---

## 6. Claim discipline for anything quoted from this cell

* Never quote a round-1 or round-2 alone-AUC without saying it was measured on the
  caption **without its cartoon** (§1.2).
* Never quote the r1/r2 missing-mass numbers: τ-only at P = 4 (§1.1b).
* MONITOR holds 23 contests and 66 positives; the group band is the quoted band and no
  movement under ~.025 is a discovery (§2).
* The Addendum-4 null on this cell is **structural**, not a search result (§3).
* Δ_beyond is negative before any mining: this cell tests whether mining *widens the
  bank's lead*, not whether it closes a residual (§0).

---

## 9. Round 5 — the CAP round. The stopping rule does NOT fire.

### 9.1 A recorded fleet degradation, and why it matters more than usual

**Fleet: 10/16 slots, P = 5, 2 families (openai ×3, glm ×2), 125 proposals.** The six
Claude slots could not run: this session's subagent budget (500) was exhausted after
round 4. P = 5 / 2 families sits on the FREEZE DECLARATION's own degradation floor
("degrade gracefully to P ≥ 4 / ≥ 2 families ... recorded") but below the P = 8 / 3
families current standard — i.e. **the same status as rounds 1–2**.

The judge legs moved with it, and are recorded the same way:
* **Merge judges**: gpt-5.6-luna (codex exec, effort high) + glm-5.2 (thinking, key B) —
  still two independent frontier models in two different families, still blind, still
  STRICT. **Both planted identity anchors passed for both judges.** (Their self-reported
  names inside the JSON — "GPT-5", "gpt-4o" — are wrong and are not the endpoints called;
  the authoritative identities are in each file's `_judge_meta`. Recorded because a
  future reader would otherwise take the self-report at face value.)
* **Blind auditor**: gpt-5.6-luna, fresh instance, caption-matched chained probes.

**Applying §1.4 symmetrically: a degraded round does not advance the stopping clock.**
That ruling was registered against rounds 1–2 before any repaired number existed, and it
binds here too. As it happens the question is moot again — see §9.4 — but the symmetry is
recorded rather than left to be noticed.

### 9.2 Species — τ's two-directional failure replicates a second time

| track | τ-only S_obs | **merged S_obs** | direction | **merged mass** | LOPO band |
|---|---:|---:|---|---:|---:|
| A | 22 | **51** | τ **over**-merged (24 edges) | **.507** | [.450, .633] mean .563 sd .078 |
| B | 35 | **30** | τ **under**-merged (20 edges) | **.380** | [.350, .525] mean .440 sd .068 |

Second consecutive round on this cell, same signed pattern, both anchors passing. The
masses are higher than round 4's because P dropped from 8 to 5 — which is the estimator
behaving correctly, and is exactly why **round 4 is the missing-mass figure of record**.

### 9.3 Audit — the cleanest of the campaign

Probes **4/4** on the rotated caption-matched pair set. **Misrouting 0/25 (.00)**, zero
disputes, no arbiter needed. Final **A = 15, B = 10 (4 mixed)**. Scoring: 134,200 prompts,
GPU 5 lane-pinned, released rc=0; anchors pos 5.461 / neg 5.593 / scrambled 0.917,
**coherent-vs-scrambled (item mean) 1.0000**, NA .00043, **0 collapsed**, 0 all-NA rows.

### 9.4 RESULT — the largest gain of the campaign, at the cap

**TIER 1, MONITOR: VA_nl .6897 → .7100, gain +.0204** [group CI **−.0006, +.0433**],
p(>0) = **.970**. Bank 334 → 349 features.
**HONEST: .6675 → .6803, gain +.0129** [group CI **+.0004, +.0252**] — **the only round in
the campaign whose HONEST band excludes zero.**
Within-contest gain +.0115 (MONITOR) / +.0041 (HONEST). Seed spread .0221.
Δ_beyond MONITOR −.0399 → **−.0603**; HONEST −.0551 → **−.0680**.

**Swap: dC₊ +.0149, dC₋ +.0097 — NOT adverse.** Both halves of the pair algebra rose by
comparable amounts, which is the uniform-improvement signature: the bank got better on
pairs the dense model orders correctly *and* on pairs it gets wrong. This is the clean
direction, and it is the opposite of round 4's shape.

Best new criteria: **"Surprise reinterpretation" .582**, "Familiar-phrase transformation"
.558, "Conceptual recontextualization" .531.

**The stopping clock RESETS to 0.** r4 was sub-ε (clock 1); r5 at +.0204 is four times ε
with a p(>0) of .97 and a HONEST band excluding zero. **The campaign therefore terminates
on the CAP of 5 rounds, with the closure curve still rising.** That is a materially
different terminal statement from jokes_community, which terminated on the rule itself.

### 9.5 Track B round 5 — the map replicates

| alone-AUC | mixed | ρ with V | channel | conjectured parent |
|---:|:--:|---:|---|---|
| .435 | — | .07 | List-normalization typography | audience familiarity |
| .453 | — | .07 | **Era-coded idiom** | position in the contest series |
| .531 | ✓ | .46 | Direct pun or wordplay reliance | entrant's practice |
| .529 | ✓ | .35 | **Consensus-angle pun** | entrant's practice |
| .526 | — | .09 | **Generic caption closure** | generic writer / LM prior |
| .509 | ✓ | .07 | Strictly lowercase orthography | magazine house style |
| .508 | — | .12 | Extreme brevity word-count | transcription convention |

Seven of ten families reproduce round 4's map from an independently re-ordered slice
against a different bank and a *different fleet composition*: the era/position family, the
consensus-angle family, the generic-prior family, the typography family, the brevity
family. **The spurious map for this cell is reproducible** — the property a map must have
before it is used to discount. **Zero channels are ALREADY-ARTICULATED at ρ_V ≥ .70**, so
the mined nuisance set is genuinely new information rather than a restatement of the
16-feature V block.

---

## 10. TERMINAL PACKAGE (`cap_finalist_TERMINAL_LEDGER.json`)

### 10.1 Stopping condition

| round | kind | fleet | VA_nl MONITOR | gain | sub-ε | counts | clock |
|---|---|---|---:|---:|:--:|:--:|---:|
| 0 | baseline (enforced gate) | — | .6767 | — | — | — | 0 |
| 1 | proposing, **degraded** (view-repaired) | P=4 / 2 fam | .6861 | +.0094 | no | **no** | 0 |
| 2 | proposing, **degraded** (view-repaired) | P=4 / 2 fam | .6907 | +.0046 | **yes** | **no (§1.4)** | 0 |
| 3 | **VIEW REPAIR (TIER R)** | none | — | — | — | **exempt** | 0 |
| 4 | proposing, **full standard** | **P=8 / 3 fam** | .6897 | **−.0010** | **yes** | **yes** | **1** |
| 5 | proposing, degraded (**CAP**) | P=5 / 2 fam | **.7100** | **+.0204** | no | no | **0 (reset)** |

**Terminated on the CAP, not on the rule.** The clock stands at 0 at termination because
the final round produced the campaign's largest and best-behaved gain. The honest terminal
claim is therefore **"the bank's lead over the dense standard was still widening when the
round budget ran out"** — not "the miner is exhausted".

### 10.2 Terminal ledger

| | MONITOR (n 526 / 23 contests / 66 pos) | HONEST (n 1,055 / 46 contests) |
|---|---:|---:|
| **T** (one convention) | **.6497** | **.6124** |
| VA_nl bank 0 (289 features, gate enforced) | .6767 | .6658 |
| **VA_nl TERMINAL** (349 features) | **.7100** | **.6803** |
| Δ_beyond bank 0 | −.0270 | −.0534 |
| **Δ_beyond TERMINAL** | **−.0603** | **−.0680** |
| **total closure gain** | **+.0333** | **+.0145** |

**The bank's lead over the dense standard more than doubled on MONITOR** (−.027 → −.060)
and grew ~27% on HONEST (−.053 → −.068). The coordinator's question — does mining widen
the bank's lead here — has a clear answer: **yes, and it had not stopped widening.**

### 10.3 Final dual-track map and discount

Terminal nuisance set **38 channels**; joint spurious-alone **.660 HONEST / .7005 MONITOR**
(above the .65 matched-sampling trigger, so both estimators ran).

| band | channels | pooled Δ | decile Δ_adj (HONEST / MONITOR) | matched Δ_adj (HONEST / MONITOR) | **stacked dense increment over B + bank** |
|---|---:|---:|---|---|---|
| ALL_B | 38 | −.0680 / −.0603 | **−.0979 / −.0348** | −.1074 / −.1591 | **−.0028 / −.0030** |
| STRICT no-mixed | 17 | −.0680 / −.0603 | **−.0839 / −.0667** | −.0630 / +.0303 | **−.0021 / −.0009** |

Per the registered readout of record, the **stratification-free stacked increment** is the
quoted discount: with 38 channels, decile stratification on 5 coarse MONITOR strata over
23 contests has lost resolution, and the matched estimator swings ±.16. The stacked figure
is stable across both bands and both populations at **−.003 to −.001**, while the **bank's**
increment over dense-plus-nuisance runs **+.024 to +.046**.

### 10.4 The dense model's residual: measured four ways, it is zero

| where | dense increment over B + bank |
|---|---:|
| round 3 (repaired, 20 channels) HONEST / MONITOR | +.0019 [−.0054,+.0090] / +.0020 [−.0140,+.0160] |
| round 4 (28 channels) HONEST / MONITOR | −.0019 / −.0034 |
| round 5 (38 channels) HONEST / MONITOR | −.0003 / +.0065 |
| **terminal cumulative, ALL_B / STRICT** | **−.0028 / −.0021 (HONEST)**, −.0030 / −.0009 (MONITOR) |

Every CI straddles zero and the point estimates change sign between rounds inside a band
of ±.007. **Conditioned on the articulated bank plus every named nuisance channel, the
dense model on this cell adds nothing.** There is no residual for a mechanism to carry —
which is the answer to the coordinator's second question, and it is stable across three
nuisance-set sizes and two populations.

### 10.5 Missing mass (strict two-judge merged, both tracks)

| round | fleet | A | B |
|---|---|---|---|
| **4 (figure of record)** | **P=8 / 3 families** | **.450** [.448, .552] | **.300** [.257, .357] |
| 5 | P=5 / 2 families | .507 [.450, .633] | .380 [.350, .525] |

Round 4 is the record because it is the only round at full fleet. The A-side space is
wide open (recapture .28) and the B side is the more consumed of the two (recapture .385)
— the same asymmetry jokes_community reported, on a cell with a completely different
construct. The r1/r2 τ-only numbers are superseded (§1.1b).

### 10.6 GEPA phrasing pass — stage 1 run, stages 2–4 NOT run

`cap_finalist_gepa_targets.json`, over all 62 mined Track-A criteria of rounds 1/2/4/5:
**35 quotable, 12 GEPA-targeted** (modal > .75), **2 collapsed** (already excluded by the
enforced gate), **16 sign-triggered**.

Worst phrasings, all high-modal rather than high-NA: "No self-commentary on its own
funniness" modal .993, "Meta-humor about the joke/genre itself" .984, "Oddly precise
number anchors the joke" .976, "Meaningful numerical specificity" .934, "Punctuation-driven
timing" .940.

**Stages 2–4 (sealed rephraser → probe-row rescore → corpus rescore → swap-in) were NOT
run**, and this is the campaign's one incomplete deliverable. Recorded plainly, with the
reason it does not change a conclusion: on jokes_community the pass was worth +.0041 to
the terminal bank, i.e. it moves the bank UP, and this cell's terminal bank is already
**above** the dense standard by .060–.068. A pass that can only widen that gap cannot flip
the sign of Δ_beyond, the dense-increment result, or the cap-not-rule termination. It
would, however, raise the terminal *level*, so the quoted VA_nl .7100 / .6803 should be
read as a **lower bound on the articulated instrument**, not a ceiling.

### 10.7 SIXTEEN sign-contradicting criteria — a substantive cell finding

The Hanley-McNeil two-sided band on FIT+MINE is ±.025 (612 positives). **16 of 62 mined
Track-A criteria are more than 2 SE BELOW .5** — significantly *anti*-predictive of editor
selection. They are not a random assortment; read the list:

> Cultural allusion repurposing · Distinctive speaker voice (twice) · Vivid scene
> specificity · Exact wordplay fit · Late twist or escalation · Idiomatic frame
> transformation · Directness on taboo subject matter · Obvious first-thought pun ·
> Metaphorical or satirical mapping · Phonetic pun precision · Adapted familiar phrase ·
> Off-screen narrative · Idiom-to-visual subversion · Layered double entendre ·
> Character-voiced perspective

**That is the standard comic-craft vocabulary, and on this pool it runs backwards.** The
mechanism is §3: the hard negatives were drawn to be crowd-plausible, and within-contest
crowd rank runs **.335** against y (i.e. .665 the other way). Anything that measures
generic comic appeal therefore separates in the wrong direction. This is not a routing
failure and the criteria are **kept** — the aggregator is sign-free, and a criterion that
reliably predicts *non*-selection is as informative as one that predicts selection. It is
recorded as this cell's counterpart to the jokes cell's inverted-polarity defect-naming
criteria, with a different cause: there the criterion named a flaw on purpose, here the
*population construction* inverts the sign of craft itself.

---

## 11. CROSS-CUTTING LINES

**1. A judge shown the wrong view manufactures MONITOR-only gain.** The item-view defect
(mined criteria scored on the caption without its cartoon, while the bank was scored with
it) turned a +.0005 HONEST reality into a +.0344 MONITOR / −.0111 HONEST closure curve.
The two populations disagreed in sign for two consecutive rounds and nothing in the
per-round readout flagged it, because each round's MONITOR gain looked plausible on its
own. The generalisable rule: **when a mined criterion is scored, the item view must be
asserted equal to the bank's item view**, and a persistent MONITOR/HONEST sign
disagreement is the symptom to look for when it is not.

**2. The enforced collapse gate removes 16% of this bank.** 55 of 364 A criteria sit at
modal > .98 on one-line captions. The historic `off-modal < 5` screen passes every one of
them at n = 5,218. Short-item corpora are where that screen fails hardest, and this cell
quantifies it: bank 0 goes 345 → 289 features and VA_nl MONITOR .6919 → .6767.

**3. τ-clustering fails in BOTH directions, on a fourth cell and a fifth/sixth round.**
Over-merged Track A (35→75, 22→51) and under-merged Track B (44→39, 35→30), with both
identity anchors passing both times. jokes_community found the same signed pattern in four
rounds. Two cells, six rounds, two different construct domains, one direction each: the
freeze's blind-pairwise identity rule is load-bearing, not a refinement.

**4. Position-in-container can be structurally null — and that is a design fact, not a
result.** This pool takes exactly 3 finalists per contest, so the label rate is constant
per container and no container-level covariate can predict y (contest ordinal .5022). The
corpus also has **no arrival-order ordinal at all** (the raw scrape's row index is the
crowd rank). Addendum-4 should distinguish "searched and found nothing" from "cannot carry
signal by construction"; this is the programme's first clean instance of the latter.

**5. Obviousness is this cell's craft/circulation boundary.** The fleet's two
position-in-container channels — "obvious first-thought pun", "canonical first-thought
angle" — were both re-routed to Track A by a blind auditor and upheld by a frontier
arbiter, on the reasoning that they measure the caption's originality. That is the jokes
cell's finding in a new domain: *the nameable, judgeable thing near a container ordinal is
the craft that co-occurs with it.* Two cells, two constructs, same conclusion.

**6. The generic prior is nameable and the fleet named it unanimously.** MODE 5 was added
because this cell is the T₀ arm's sole exception (untrained prior worth +.0374). Seven of
eight sealed proposers independently named "generic well-formed caption-shaped joke" — the
joint-top consensus channel of round 4 — and it reappeared in round 5 as "generic caption
closure". The construct the T₀ column measures is **articulable**, which is a
non-trivial thing to have shown, even though its alone-AUC is small for the reason in §3.

**7. Hard negatives invert the sign of craft.** Within-contest crowd rank predicts
*non*-selection at .665, and 16 of 62 mined craft criteria are significantly
anti-predictive. Any cross-cell comparison that treats "alone-AUC below .5" as a defect
signal will mis-read this cell wholesale.

**8. A cap-limited campaign is a different object from a saturated one.** jokes_community
terminated on two consecutive sub-ε rounds with A-side mass .56; cap_finalist terminated on
the cap with its largest gain in the final round and A-side mass .45. Both are "5 rounds,
terminal", and they license opposite claims. The terminal language for this cell must be
**"still gaining at the cap"**, never "not discoverable by this miner".

---

## 12. Claim discipline for anything quoted from this cell

* Never quote a round-1 or round-2 number from `maps_batch1` — those are the mismatched
  item view and the un-enforced collapse gate. §7.3 is the reconstruction of record.
* Never quote the r1/r2 missing-mass numbers (τ-only, P = 4). Round 4 is the record.
* MONITOR holds 23 contests and 66 positives. The group band is the quoted band; no
  movement under ~.025 is a discovery.
* The Addendum-4 null here is **structural** (constant per-contest label rate), not a
  search result.
* Terminal VA_nl (.7100 MONITOR / .6803 HONEST) is a **lower bound**: GEPA stages 2–4 were
  not run.
* Rounds 1, 2 and 5 ran degraded fleets (P = 4, 4, 5 / 2 families). Only round 4 is at
  P = 8 / 3 families. Any statement about fleet coverage on this cell must name which.
* The campaign terminated on the **cap**, with the stopping clock at 0. Do not describe
  this cell as saturated.
