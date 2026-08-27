# cap_crowd — Layer-3 articulation-closure campaign (LANE A, cell 3)

Cell: **New Yorker cartoon caption contest, COMMUNITY track** — y = 1 if the caption's
crowd rating is above its own contest's median, 0 if below (NEXTML crowd-rating
experiment, ≥100 votes). 10,893 captions, 223 contests, pos-rate **.5001**.

**Sister cell: `cap_finalist`** — same corpus, same contests, same A/V banks, EDITOR
label. The two terminal maps are the programme's humour **curation-vs-community**
contrast, so this campaign deliberately reuses cap_finalist's criterion names for the same
constructs and its machinery unchanged.

Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + FREEZE DECLARATION + Addenda 1–4.
Reference campaigns: `notes/2026-08-09__closure_jokes.md` (terminal, LANE A cell 1) and
`notes/2026-08-09__closure_cap_finalist.md` (terminal, LANE A cell 2 — the view-repair rule
originates there and is now programme-wide).
Code + artifacts: `methods/taste_decomposition/closure/cap_crowd/`.
Rounds 1–2 (map-focused, pre-standard): `closure/maps_batch1/cap_crowd_r{1,2}_*`.

Abbreviations spelled out. **V** = 16 deterministic label-blind surface features.
**A** = the Gemma-4-31B-judged criterion bank (364 raw columns, the same bank as
cap_finalist). **VA_lin / VA_nl** = grouped-OOF logistic / HistGradientBoosting over V+A,
seed-mean {0,1,2}. **T** = the dense standard (two arms — see §2). **Δ_beyond** = T − VA_nl.
**MONITOR** = the frozen decision population. **HONEST** = every dense-held-out row.
**Track A** = candidate-real criteria. **Track B** = suspected-spurious channels.
**TIER S** = sealed independently-proposing fleet (the only tier that feeds Good-Turing).
**TIER R** = re-measurement of existing criteria on a corrected item view (cap_finalist §1.4).

---

## 1. VIEW CHECK (coordinator's item 1) — the mismatch IS present

Checked before anything else. Rounds 1–2 came from the same `maps_batch1` machinery as
cap_finalist's, and they carry the same three defects plus a fourth that is worse here:

| | finding on cap_crowd | evidence |
|---|---|---|
| **item view** | mined criteria scored as `CAPTION:\n<text>`, **no cartoon**, while the A bank was scored as `CARTOON: <desc>\n\nCAPTION: "<text>"` | `maps_batch1/score_gemma_maps.py:151-154` vs `datasets/humor/caption_multiy/score_va_gemma_captions.py:190` |
| **proposer view** | `cap_crowd_r1_slice.json` rows carry `text` only — no `cartoon` field | slice keys |
| **fleet** | 12 slots, **both GLM slots MISSING in both rounds** → **P = 4 / 2 families** | `*_proposals_fleet.json` |
| **probes** | genre-mismatched (peer-review) probe pool, and here it **actually failed: r1 probe pass 2/4** | `cap_crowd_r1_results.json` routing |
| **species** | τ-only at τ = .79, no strict two-judge blind merge | `*_species.json` |

**The r1 probe failure is the direct evidence the cap_finalist note could only argue for.**
There the mismatched probes happened to pass 4/4 and the complaint was structural; here the
same pool put the blind auditor at chance on the planted items. A .00 misrouting rate in
r1 read against a 2/4 probe pass is not evidence of good routing.

**Ruling, registered before round 3 runs**, inheriting cap_finalist §1.4 verbatim:
1. Rounds 1–2 **count as rounds** against the cap of 5.
2. Their Δ values **do not advance the stopping clock** (degraded fleet, mismatched view,
   failed probes).
3. Their **missing-mass numbers are superseded** (τ-only, P = 4).
4. **Round 3 is a VIEW-REPAIR pass (TIER R)** — all 50 r1/r2 criteria re-scored on the
   matched cartoon+caption view; no fleet, so no Good-Turing mass and no clock effect.
5. Routing is not re-decided (the audit routed on criterion TEXT); the probe failure is
   recorded as a caveat on r1's reported routing.
6. Pre-repair matrices preserved as `cap_crowd_r{1,2}_scores.MISMATCHEDVIEW.npz`.

---

## 2. T CONVENTION (coordinator's item 2) — declared, with both arms

This cell is the one place in the programme where the dense standard is known to be
**trainer-dependent** (registry 2026-08-08):

| arm | source | AUC on the 2,190 dense-held-out rows |
|---|---|---:|
| **T_archived** | `closure/samerows_preds/cap_crowd_dense_preds_slim.csv` — the master-ledger arm, and what rounds 1–2 were read against | **.5554** |
| **T_matched_vanilla** | `debias/runs/D20_cap_vanilla/preds_slim.csv` — a fresh vanilla dense model (λ_adv = 0) from the debias pilot's trainer, **same rows, same train/eval/test split** (asserted in `cells.load()`) | **.6047** |

**CAMPAIGN CONVENTION: `T_matched_vanilla` is PRIMARY.** Reasons, registered: the registry
instruction is "RE-BASE before quoting"; the archived arm is a different pipeline from
every other cell's; and Δ_beyond computed against an undertrained dense model overstates
the bank. Re-basing moves the archived Δ_beyond of −.110 to about −.066.
**T_archived is reported beside it in every table and in the terminal ledger — never
dropped, never quoted alone, and the two are never differenced against each other.**
`cells.load()` carries both per-row and asserts the split assignments match.

---

## 3. Splits, power, and how this cell differs from its sister

Splits **inherited unchanged** from round 1 (`cap_crowd_splits.json`): stable hash on the
contest key, MONITOR ⊂ dense-held-out. Group overlap FIT+MINE / MONITOR: 0.

| split | rows | contests | pos |
|---|---:|---:|---:|
| FIT+MINE | 9,821 | 191 | .5002 |
| — mining slice M (dense-held-out) | 1,118 | — | — |
| **MONITOR** | **1,072** | **32** | **.500 (536 positives)** |
| HONEST (dense-held-out) | 2,190 | 64 | — |

**This cell is far better powered than cap_finalist: 536 MONITOR positives against 66.**
cap_finalist could not resolve ε-scale movement; here the r1 group band was ±.012, about
half cap_finalist's. The ε = .005 rule is applied to a quantity this cell can actually
measure, and that difference should be stated whenever the two cells' curves are compared.

**TIER 2 = within-contest AUC** (y is a within-contest median split, so the within-contest
readout matches the y-definition). Reported every round, never substituted for TIER 1.

**Roster tier: map-focused dual-track** — matched Δ_beyond is strongly negative (see §5),
far below the +.02 full-dual-track line, so the spurious map is the headline and the
Track-A curve is reported in full. Both tracks at full budget k_A = 15, k_B = 10.

**FLEET STANDARD FOR THIS CAMPAIGN, declared up front: P = 5 / 2 families**
(gpt-5.6-luna ×3 via codex, glm-5.2 ×2), because the session's Claude subagent budget is
exhausted and no Claude slot can run. This sits on the FREEZE DECLARATION's own degradation
floor ("degrade gracefully to P ≥ 4 / ≥ 2 families ... recorded").

**And a registered departure from cap_finalist's clock ruling, with its reasoning.** On
cap_finalist, degraded rounds did **not** advance the clock, because that campaign
*contained* a full-standard round (r4) and mixing standards inside one curve would have let
a weaker miner buy a termination. Here **every** round runs at one declared standard, so
"degraded rounds don't count" would make the stopping rule vacuous — it could only ever
terminate on the cap. The rule's purpose is to detect *miner exhaustion*, which is always
relative to the miner. So on this cell **sub-ε rounds at the declared P = 5 / 2-family
standard DO advance the clock**, and the terminal claim is qualified by the fleet it was
measured with: "not discoverable by a P = 5 two-family miner", never "no such criteria
exist". Registered now, before round 3, so it cannot be chosen after seeing a gain.

---

## 4. REGISTERED PREDICTION (coordinator's item 4) — written before the first fit

cap_finalist found that **within-contest crowd rank runs .335 against the EDITOR label**
(|Δ| = .165, inverted: crowd-popular captions are *less* likely to be editor finalists in
that hard-negative pool), and that **16 of 62 mined Track-A craft criteria were
significantly anti-predictive** there — the standard comic-craft vocabulary running
backwards. The named mechanism was the pool construction: hard negatives were drawn to be
crowd-plausible.

**If that mechanism is right, the same criteria must flip sign POSITIVE on cap_crowd**,
because here the crowd rating *is* the label. Registered predictions, in falsifiable form:

**P1 (directional, primary).** Of the criteria that fired the sign-contradiction trigger on
cap_finalist, a **majority** will have alone-AUC **> .5** on cap_crowd's FIT+MINE. The
named list to check, verbatim from cap_finalist §10.7: cultural allusion repurposing;
distinctive speaker voice; vivid scene specificity; exact wordplay fit; late twist or
escalation; idiomatic frame transformation; directness on taboo subject matter; obvious
first-thought pun; metaphorical or satirical mapping; phonetic pun precision; adapted
familiar phrase; off-screen narrative; idiom-to-visual subversion; layered double entendre;
character-voiced perspective.

**P2.** The **rank correlation across shared criteria** between cap_finalist alone-AUC and
cap_crowd alone-AUC will be **negative**. This is the sharper test: P1 could pass on a
level shift alone, P2 requires the ordering to invert.

**P3 (falsifier).** If instead the same criteria are anti-predictive on BOTH cells, the
"hard-negative construction inverts craft" explanation is **wrong** and the honest reading
becomes that the Gemma judge's craft scores are simply mis-signed on captions generally.
That would retract cap_finalist §10.7's mechanism, and it is the outcome this note commits
to reporting if it occurs.

Both cells are scored by the same judge on the same bank and (after round 3) the same item
view, so the comparison is clean. **Criterion names are held identical across the two maps
for the same constructs**, per the coordinator, so P2 can be computed by name join.

---

## 5. Round 0 / inherited state (AS REPORTED — superseded by §6)

Under the historic screen and the mismatched view, and against **T_archived**:

| | MONITOR (1,072) | HONEST (2,190) |
|---|---:|---:|
| T_archived | .5469 | .5554 |
| VA_nl bank 0 (362 features) | .6456 | .6235 |
| Δ_beyond bank 0 | **−.0987** | **−.0681** |
| r1 gain / VA_nl | +.0063 → .6520 | +.0067 → .6302 |
| r2 gain / VA_nl | **−.0023** → .6496 | +.0035 → .6337 |

r1 routing: A 15 / B 10, misrouting .00, **probes 2/4 (FAILED)**.
r2 routing: A 9 / B 16 (11 mixed), misrouting .24, probes 4/4.

**Addendum-4 note, structural as on the sister cell.** y is a *within-contest median
split*, so the label rate is .5 in every contest by construction and no container-level
covariate can predict y. The corpus also has no arrival-order ordinal (the raw scrape's row
index is the crowd rank). **And a leak guard specific to this cell:** crowd_mean,
crowd_votes and within-contest crowd rank ARE the definition of y here, so unlike on
cap_finalist they are excluded from the covariate line entirely (`contest_line.py`), never
carried as covariates. That asymmetry — the same three columns are legitimate observed
covariates on the editor cell and forbidden on the community cell — is itself part of the
curation-vs-community contrast.

**MODE 5 (generic-prior-guessable quality) stays in Track B** (coordinator's item 3). On
cap_finalist, 7 of 8 sealed proposers named it and the T₀ untrained prior contributed
+.0374; on this cell T₀ read **−18%**. The contrast is the point: if the generic prior
helps an editor label and hurts a crowd label, the channel should still be *nameable* here
while being useless-to-harmful as a predictor, and the fleet's consensus on it is the
measurement.

---

## 6. Round 3 — VIEW-REPAIR PASS (TIER R). The inherited curve, re-derived.

50 criteria (25 from r1, 25 from r2) re-scored corpus-wide on the matched
`CARTOON: <desc>\n\nCAPTION: "<text>"` view. Gemma-4-31B offline batch, GPU 5 lane-pinned,
**552,150 prompts**, ~58 min, released rc=0. No fleet ran; TIER R.

**The repair moved the measurements as much as it did on the sister cell**: median
rank-correlation old-view vs new-view **.728 (r1) / .709 (r2)**, with the cartoon-dependent
constructs moving most — "Concrete absurd escalation" ρ = .342 (mean 3.15 → 5.43),
"Productive ambiguity" ρ = .309 (6.92 → 4.59), "Deadpan underreaction" ρ = .387,
"Visualizable physical staging" ρ = .449 (4.87 → 2.91).

Anchors (K = 50/class, matched view): pos 3.512 / neg 3.714 / **scrambled 1.063**;
**coherent-vs-scrambled (item mean) .9994**; non-NA-count reading .495 — the same
structural degeneracy recorded on cap_finalist §7.5 (NA rate .00014, so the count is
~constant and carries no information); item-mean reading is operative here too.
pos-vs-neg .433. **1 collapsed** (r1 B09 "Specialist abbreviation / acronym shorthand",
modal .982). 0 all-NA rows.

### 6.1 Reconstructed Δ curve, both views, Δ_beyond against BOTH T arms

Bank 0 under the enforced collapse gate: VA_nl MONITOR **.6488**.

| | **REPAIRED (cartoon+caption)** | **PRE-REPAIR (caption only)** |
|---|---:|---:|
| r1 VA_nl MONITOR | .6424 | .6528 |
| r1 gain MONITOR [group CI] | **−.0065** [−.0180, +.0045] p .11 | +.0040 [−.0079, +.0169] p .74 |
| r1 gain HONEST | +.0021 | +.0049 |
| r2 VA_nl MONITOR | .6420 | .6502 |
| r2 gain MONITOR [group CI] | **−.0004** [−.0079, +.0071] p .48 | −.0026 [−.0081, +.0023] p .14 |
| r2 gain HONEST | −.0004 | +.0047 |
| **cumulative MONITOR / HONEST** | **−.0069 / +.0017** | +.0013 / +.0097 |
| dense increment over B + bank, HONEST | **+.0133** | +.0092 |

**Repaired, both inherited rounds are sub-ε and slightly negative on MONITOR.** As on
cap_finalist, the caption-only view was flattering the curve (+.0013/+.0097 pre-repair vs
−.0069/+.0017 repaired) — the second independent replication of the view-repair effect,
now on a cell with 8× the positives, so it is not a small-sample artifact.

Per §1's ruling those two rounds still do not advance the clock. **Clock 0** entering
round 4; cap 5.

**Δ_beyond after r2, against both arms** (matched-vanilla PRIMARY, archived beside):

| | MONITOR | HONEST |
|---|---:|---:|
| VA_nl (repaired, r2) | .6420 | .6252 |
| **T_matched_vanilla (PRIMARY)** | **.6212** | **.6047** |
| **Δ_beyond (primary)** | **−.0208** | **−.0205** |
| T_archived | .5469 | .5554 |
| Δ_beyond (archived) | −.0951 | −.0698 |

**Re-basing cuts the bank's apparent lead by more than a factor of three** (−.095 → −.021
on MONITOR). This is why the registry caveat exists and why the archived number must never
travel alone. The bank still leads on both arms — the sign is robust — but the magnitude
is entirely a statement about which dense model you compare to.

### 6.2 Addendum-4 covariate line — structurally null, and a leak guard

`cap_crowd_contest_line.json`. Contest ordinal alone-AUC = **.5000 exactly** on both
HONEST and MONITOR (and .4998 on the full population): y is a within-contest **median
split**, so the label rate is .5 in every contest by construction and a container-level
covariate cannot carry signal. This is the same structural null as cap_finalist, arrived
at by a different route (median split vs fixed 3-of-23), and it is a design fact, not a
search result. No arrival-order ordinal exists in the corpus.

**Leak guard, cell-specific and load-bearing:** crowd_mean, crowd_votes and within-contest
crowd rank ARE the definition of y here, so they are excluded from the covariate line
entirely. On the sister cell the identical three columns are legitimate observed
covariates carrying |AUC − .5| = .165. *The same measurement is a covariate on the editor
cell and the label on the community cell* — which is the cleanest statement of what the
curation/community contrast is.

---

## 7. CROSS-CELL TRANSPLANT (TIER D) — the registered P1/P2/P3 predictions are decided

**Why a transplant and not a name join.** §4's predictions were first tested by joining
criterion NAMES across the two cells' rounds 1–2. That join returned **zero shared names**
— those rounds were proposed independently on each cell, before the coordinator's
same-names rule existed — so a name join could not test anything. Recorded rather than
quietly dropped, and replaced with the controlled version: take cap_finalist's criterion
**TEXT verbatim** and score it on cap_crowd, same judge, same bank, same cartoon+caption
item view, so that **the label is the only thing that differs**.

25 criteria: the **12** of cap_finalist's 15 sign-contradicting criteria whose text is on
disk (3 were not recoverable from its species files and are recorded as such), plus its
**13** highest-alone-AUC criteria, so the correlation has both signed ends to work with.
**TIER D**: directed, excluded from Good-Turing, does **not** join this cell's bank or its
closure curve. 272,325 prompts, GPU 5, released rc=0; anchors coherent-vs-scrambled
**1.0000**, 0 collapsed, NA .00015.

### 7.1 P1 — PASSES, unanimously

**12 of 12** of cap_finalist's sign-contradicting criteria read **above .5** on cap_crowd,
and **all 12 are significantly above** (Hanley-McNeil 2 SE band = ±.019 at 5,451/4,370
FIT+MINE positives/negatives). Not one exception.

| criterion | cap_finalist | **cap_crowd** |
|---|---:|---:|
| Late twist or escalation in the final clause | .449 | **.579** |
| Off-screen narrative | .468 | **.561** |
| Metaphorical or satirical mapping | .464 | **.549** |
| Vivid scene specificity | .462 | **.544** |
| Exact wordplay fit | .461 | **.543** |
| Idiomatic frame transformation | .468 | **.537** |
| Obvious first-thought pun | .467 | **.536** |
| Adapted familiar phrase | .475 | **.535** |
| Cultural allusion repurposing | .456 | **.533** |
| Directness on taboo or transgressive subject matter | .469 | **.520** |
| Phonetic pun precision | .468 | **.515** |
| Distinctive speaker voice | .450 | **.513** |

### 7.2 P2 — FAILS, and the failure is the more interesting half

The rank correlation across all 25 transplanted criteria is **ρ = +.411 (p = .041)**,
Pearson **+.462 (p = .020)** — **positive**, where §4 predicted negative. P2 is recorded as
**FAILED**.

**What that means, stated carefully.** Both readings are true and they are not in tension:
* **Sign relative to chance flips** (P1): every one of those 12 criteria crosses from below
  .5 to above it.
* **The ordering does not invert** (P2): editors and crowds *agree on which craft
  properties matter more*. "Comic economy", "punchline compression", "surprise
  reinterpretation", "compactness with payload" sit near the top on both labels; the weak
  ones are weak on both.

So the effect is a **LEVEL SHIFT, not a SIGN INVERSION**. Mean alone-AUC over the same 25
criteria: **.514 on cap_finalist → .553 on cap_crowd**, and on cap_crowd **every one of the
25 is ≥ .501**. A uniform downward shift of ~.04 in AUC is enough to push the weaker half
of a craft bank below chance on the editor label while the same criteria stay positive on
the crowd label.

**This partially RETRACTS the mechanism cap_finalist §10.7 asserted.** That note said "the
standard comic-craft vocabulary runs backwards on this pool", which implies inversion. The
transplant shows it does not run backwards in the ordering sense — it runs **flat and
noisy** against the editor label (centred near .5) and **weakly positive** against the
crowd label. The *observation* (16 of 62 significantly anti-predictive on cap_finalist)
stands; the *interpretation* is corrected here, and cap_finalist's cross-cutting line 7
should be read against this paragraph. P3's falsifier did **not** trigger — the criteria
are not anti-predictive on both cells — so the hard-negative construction remains the
right explanation for the *shift*; it is the word "backwards" that was too strong.

---

## 8. APPENDIX — campaign STOPPED mid-round-4 (caption cells retired 2026-08-11)

The user retired the caption cells from the programme while round 4 was in its readout
stage. This appendix is the honest state of the campaign at that point. **No further
rounds were run, and this cell has NO terminal ledger** — do not quote one.

### 8.1 What is complete and trustworthy

| stage | state |
|---|---|
| view check (§1) | **DONE** — mismatch confirmed, plus r1 probes **failed 2/4** |
| round 3 view repair (TIER R) | **DONE** — 552,150 prompts, GPU 5, rc=0; curve re-derived (§6.1) |
| observed-covariate line (§6.2) | **DONE** — contest ordinal exactly .5000; crowd columns excluded as label-defining |
| cross-cell transplant (§7) | **DONE** — P1 passes 12/12, P2 fails (ρ = +.411), P3 not triggered |
| round 4 fleet + species + merge + audit | **DONE** — 10/16 slots (P=5 / 2 families), strict merge anchors 2/2 both judges, **probes 4/4, misrouting 0/21**, final A=11 / B=10 (4 mixed) |
| round 4 Gemma scoring | **DONE** — GPU 5, rc=0, 0 collapsed, NA .0006, coherence 1.0000 |
| **round 4 readout** | **launched detached; lands at `cap_crowd_r4_results.json`** — not read into this note |
| round 5, GEPA, terminal ledger | **NOT RUN** |

### 8.2 The Δ curve as it stands (repaired view, enforced collapse gate)

| round | kind | fleet | VA_nl MONITOR | gain | counts toward clock |
|---|---|---|---:|---:|:--:|
| 0 | baseline | — | .6488 | — | — |
| 1 | proposing (degraded, view-repaired) | P=4 / 2 fam | .6424 | −.0065 | no (§1) |
| 2 | proposing (degraded, view-repaired) | P=4 / 2 fam | .6420 | −.0004 | no (§1) |
| 3 | **VIEW REPAIR (TIER R)** | none | — | — | exempt |
| 4 | proposing | P=5 / 2 fam | *pending readout* | *pending* | yes (§3 ruling) |

Clock **0** entering round 4.

### 8.3 BOTH T conventions, as required — the numbers to quote from this cell

Δ_beyond after round 2, repaired view. **Matched-vanilla is the campaign PRIMARY; the
archived arm is reported beside it and the two are never differenced.**

| | MONITOR (1,072 / 32 contests / 536 pos) | HONEST (2,190 / 64 contests) |
|---|---:|---:|
| VA_nl (repaired, after r2) | .6420 | .6252 |
| **T_matched_vanilla (PRIMARY)** | **.6212** | **.6047** |
| **Δ_beyond (primary)** | **−.0208** | **−.0205** |
| T_archived | .5469 | .5554 |
| Δ_beyond (archived) | −.0951 | −.0698 |

**Re-basing cuts the bank's lead by more than 3× and that is the single most important
thing to carry from this cell.** The sign is robust (the bank leads on both arms); the
magnitude is a statement about which dense model you compare to, not about the bank.

### 8.4 Missing mass, round 4 (strict two-judge merged, the figure of record here)

**A .387** [LOPO .367–.533], **B .300** [.300–.425]; τ over-merged A (11 → 43) and
under-merged B (32 → 26) — the two-directional τ failure replicating on a **third** cell.
Round 4 is the only cap_crowd round at this campaign's declared standard; the r1/r2
τ-only numbers are superseded.

### 8.5 The result worth keeping from this cell

The **cross-cell transplant (§7)** is the deliverable that survives the cancellation, and
it does not depend on the unfinished rounds: cap_finalist's criterion text, scored on
cap_crowd with the label as the only difference, shows the curation/community contrast is
a **LEVEL SHIFT (mean alone-AUC .514 → .553, all 12 sign-contradicting criteria crossing
above .5, all significantly) and NOT a sign inversion (ρ = +.411, p = .041 — the two
labels agree on which craft properties matter more)**. That corrects cap_finalist §10.7's
"comic-craft vocabulary runs backwards" wording, and cap_finalist's cross-cutting line 7
should be read against §7.2 here.
