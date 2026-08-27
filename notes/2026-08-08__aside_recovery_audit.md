# A-side leave-out recovery audit: why sensitivity is .333/.556, and what would raise it

Date: 2026-08-08. Status: **exploratory audit, CPU + 2 proposer calls (gpt-5.6-luna via
codex exec), no judging, no GPU, `latex/` untouched.** Audits the M3 leave-out-recovery
arm of the robustified missing-mass battery
(`notes/2026-08-06__missing-mass-robustification.md`) on the peer-review verdict cell.

Terminology, spelled out per the standing rule: **A** = the articulated-criterion bank
(154 delivered criteria, 54 distinct effective concepts); **VA_nl** = the
HistGradientBoosting aggregation of the V+A score matrix; **M3** = the leave-out
recovery arm (3 replicates x 8 held-out concepts, sealed proposer fleets, full-recall
Opus adjudication); **alone-AUC** = a concept column's solo area under the ROC curve on
the FIT+MINE split; **strength** = |alone-AUC - .5|; **sensitivity** = fraction of
held-out concepts a fleet re-proposes (judge-matched); **retained control** = the
stratum-matched never-removed concepts judged with the same instrument (false-positive
floor); **lift** = sensitivity - control; **tau** = the bge-large embedding-cosine
threshold for calling two criteria the same species (defensible band .77-.81);
**P** = number of proposers in a fleet.

Headline inputs being audited: sensitivity **.333** overall, **.556** high-stratum,
lift **-.042** (zero). All artifacts for this audit:
`methods/taste_decomposition/closure/robust_mm/recovery_audit/` (scripts `q1_*` ... `q5_*`,
JSON outputs named alike).

---

## 1. Dose-response: rediscovery keeps RISING with concept strength — no plateau inside the bank's range

Per-concept table (all 48 recall targets = 24 held-out + 24 retained controls, from
`m3_recall.json`; full table in `recovery_audit/q1_dose_response.json` key `table`).
Because the measured depletion lift is zero, held-out and retained targets are the same
experiment (spontaneous naming) and can be pooled; fits are also reported separately.

| readout | value |
|---|---|
| logistic slope, matched ~ strength, pooled n=48 | **+.572 log-odds per .01 of alone-AUC** (z=2.93, p=.0034) |
| same, held-out only (n=24) | +.467 / .01 (p=.086) |
| same, control only (n=24) | +.677 / .01 (p=.025) |
| held-out-vs-control covariate in joint fit | +.076 log-odds, p=.92 (zero lift again, now conditional on strength) |
| rank statistic: Mann-Whitney AUC of strength, matched vs not | **.803** pooled (perm p=.0003); .742 held-out; .837 control |

Rates along the dose axis (pooled):

| strength quartile (alone-AUC range) | n | match rate |
|---|---|---|
| Q1 (.500-.503) | 12 | .083 |
| Q2 (.503-.518) | 11 | .364 |
| Q3 (.519-.533) | 13 | .231 |
| Q4 (.537-.607) | 12 | **.750** |

Top end: **top-10 strongest targets .80 matched, top-5 .80**; the three strongest
concepts in the study (alone .588-.604, all retained controls: "Novelty and
significance", "Data/code availability statement", "Methods/data transparency") are all
matched, two of them by 2 independent proposers. Catch DEPTH is graded the same way:
mean distinct proposers naming a high-stratum target = .78 (held-out) / 1.33 (control)
vs .44/.11 at mid and 0/.33 at low.

**Answer to the dose-response question: the curve is still rising at the top of the
observed range — it does not saturate below it.** The fitted logistic puts P(rediscover)
at .15 for alone-AUC .505, .30 at .52, .64 at .545, .81 at .56, and .98 at the bank
maximum .607. Two honest qualifications: (i) the bank's own range is compressed — the
strongest concept anywhere in it is alone .607, so nothing here says what happens for
genuinely strong concepts (none exist in this bank); (ii) the one repeated top-end miss
is "Abstract accuracy, completeness, and balance" (.555) — missed twice (as held-out in
rep3 AND as control in rep1), an adjacent proposal existed both times ("Honest Scope
Boundaries", cos .67) and the taxonomy-directed arm below produced a near-identical
criterion ("Accurate, balanced abstract claims", cos .752). Its misses are
match-strictness, not absence-from-prior.

So low overall sensitivity (.333) is mostly COMPOSITION: 15 of 24 held-out concepts sit
at alone-AUC <= .52, where spontaneous naming probability is ~.1-.3 for any concept,
removed or not.

## 2. Miss autopsy: 16 missed held-out concepts

### 2a. SLICE link — the gap never surfaces in what proposers read, and no slice rule can fix that

Two measurements (`q2a_perconcept_depletion.json` + `q2a_analysis.json`,
`q4ii_error_slice.json`); 24 single-concept depletion refits under the frozen spec:

* **Per-concept slice churn is graded and stratum-structured, not jitter.**
  Single-concept removal churns 0-13 of the 60 slice rows (median 6.5). **All six
  low-stratum concepts churn 0-1 rows — their removal is literally invisible in
  what the proposers read** — while mid/high concepts churn 4-13 (the 8-concept
  depletions churned 11/10/15, strongly sub-additive). Churn tracks strength
  (Spearman .58) and the per-concept honest-AUC drop (.76); drops themselves are
  tiny (median +.0010, max +.0084, 23/24 inside the +-.007 readout-noise band).
* **Churn correlates with rediscovery, but as a MARKER, not a mechanism.** Rank-AUC
  of churn for rediscovered-vs-missed = .777 (perm p=.026), drop .805 — but strength
  alone already gives .742, and the three are collinear. Two facts break the causal
  reading: (i) every REDISCOVERED concept's matched proposal also shows up in fleets
  whose slice contained no trace of its removal (Section 3, T1 — 82% recur in the
  round-5 full-bank fleet); (ii) among mid/high MISSES the slice did visibly move
  (4-13 rows: "Study design description" churned 13, "Data availability and sharing"
  10, "TRIPOD" 9 — all missed). Surfacing is therefore neither sufficient (moved
  slices still missed) nor necessary (unmoved banks still "rediscovered"); strength
  drives both the churn and the prior-naming probability.
* **The zero-churn six are the .000-sensitivity low stratum.** For them the audit
  charge's hypothesis ("proposers can't track a gap that doesn't surface in the 60
  rows they read") holds in its literal form — there was nothing to track.
* **Even an ORACLE slice cannot surface the hole.** The proposed error-conditioned
  slice (worst-predicted rows of the depleted stack — requires labels, so it BREAKS
  the label-blind mining protocol; computed as an oracle diagnostic only) shares just
  12-14/60 rows with the shown slice, yet surfaces the held-out concepts no better:
  mean non-null fraction of held-out concept columns on slice rows = **.565 shown vs
  .572 oracle vs .526 random-60** (all-mining base .537). The held-out columns are
  near-null and their activity is spread nearly uniformly over the corpus, so no row
  selection — not even label-aware selection — concentrates them. The "better
  slices" route is dead at the root, which together with T1/T2 is the strongest
  evidence for the prior-coverage mechanism in Section 3.

### 2b. PROPOSER link — something adjacent was proposed for 15 of 16 misses

Calibration first: the 27 judge-confirmed cross-register match pairs have cosine
.580-.717 (median .648) — the same band the misses' nearest candidates occupy, so
cosine has NO discriminative power across registers (true matches and true non-matches
are interleaved; this reconfirms Section 2.3 of the parent note from the opposite
direction). Using the band only to rank (`q2b_nearmiss.json`):

| failure class (analyst-read of nearest candidates) | n / 16 | examples |
|---|---|---|
| adjacent proposal exists, judged (correctly, strictly) different — PARTIAL overlap | 8 | "Outcome measures/precision" <- "Quantitative claim calibration"; "Theoretical framing" <- "Theory instantiated as usable method" (a JUDGE SPLIT); "Title/abstract quality" <- "Abstract reads as one coherent argument" |
| concept is out-of-register for the corpus: clinical/editorial reporting-guideline items with no ML-abstract realization | 7 | TRIPOD adherence, TIDieR-complete intervention description, review/synthesis design, ethics/human-data compliance, citation ethics, dataset stewardship, study-design arms/allocation |
| nothing adjacent proposed at all | 1 | "Accessibility and inclusive communication" (cos .52 to nearest) |

The second class is the structural one: a proposer reading 60 ML abstracts will never
propose "TIDieR-complete intervention description," and *should not* — the bank was
authored in general/clinical scientific-reporting language (CONSORT/PRISMA/STROBE
inheritance), and ~7-8 of the 24 held-out concepts are corpus-inapplicable in exactly
this sense. Their alone-AUCs are .500-.511: they carry no signal to lose, and the fleet
correctly ignores them. This is the low-stratum .000 row of the sensitivity table.

### 2c. DETECTOR link — the strict judges cost ~2-4 catches at the margin

The pairwise instrument's 9 judge-disagreement pairs re-read one by one
(`m3_adjudication_blind.json` x `m3_adjudicated.json`): all nine are genuine partial
overlaps ("Open data/code/models for reproduction" vs "Reusable research asset";
"ML experiment setup transparency" vs "Reproducibility detail is substantive";
"Outcome measures/precision reporting" vs "Quantitative claim calibration"). A lenient
human would call 3-4 of them same-concept; the primary rule resolves all
disagreements to "different". Consistently, either-judge sensitivity is .375/.417
(recall/pairwise) vs .333/.292 strict. The anchor battery already showed both judges
strict (both rejected both weak-SAME anchors). So the detector contributes a real but
bounded haircut: **+.04-.08 sensitivity** sits between the strict and either-judge
readouts — nowhere near enough to reach the .70 floor, and symmetric across held-out
and retained (it cannot manufacture lift).

Dominant failure link, quantified on the 16 misses: **prior/register composition (8
out-of-register or never-adjacent) > judge strictness at the match margin (~4 partial
overlaps a lenient rule would count) > detector range (mechanical tau unusable
cross-register, already excluded from the readout)**. The slice link (2a) is upstream
of all of them and unfixable by slice design.

## 3. Zero-lift mechanism: the fleet is a STATIC-PRIOR sampler — confirmed on three independent tests

The implied model: each proposer samples criteria from a stable personal prior over
plausible quality criteria, insensitive to the bank's actual gaps. Tests
(`q3_mechanism.json`):

* **T1 — rediscoveries recur where nothing was removed.** Of the 11 judge-matched
  proposals, **9/11 (82%) recur at tau >= .79 (within-register) in the round-5 fleet,
  which read the FULL bank's slice with nothing depleted** (mean best-cosine .823);
  10/11 recur in at least one other fleet. The fleet proposes these criteria because
  they are in its prior, not because they were removed.
* **T2 — proposals follow the proposer, not the slice.** Same model reading a
  DIFFERENT slice re-proposes its own species at **.444**; different models reading
  the SAME slice overlap at only **.205**. If slice content drove proposals the
  inequality would reverse. Per-slot self-recapture across slices: codex_luna_a .58,
  codex_luna_b .50, claude_sonnet .45, claude_opus .24 — stable personal dialects.
* **T3 — accumulation is rarefaction-shaped.** Marginal new species per added
  proposer at P=6 (round5): 11.6 / 8.7 / 8.1 / 7.3 / 6.7 / 6.5 — the slow near-linear
  decay of sampling a large heavy-tailed static pool, with only 6/49 species named by
  >= 2 families. Nothing concentrates, which is what gap-tracking would look like.

**Verdict: sensitivity is a property of PRIOR COVERAGE, not gap-tracking.** The model
predicts sensitivity rises with P (more draws from more priors) and with prior
WIDENING (directed prompting), but not with better slices — exactly the pattern
measured in Section 4 (subset curves rise with P; the oracle slice does nothing; the
directed arm moves most). It also predicts zero lift at any P, which is what the
battery found and why the M3 gate must always publish the control.

## 4. Interventions, ranked

Measured or projected effect on high-stratum held-out sensitivity (floor = .70):

| rank | intervention | expected gain | cost / trade |
|---|---|---|---|
| 1 | **(iii) taxonomy-directed prompting** — category-level sweep instruction | **measured below: mechanical target-cosine +.10 mean, analyst-read ~6-7/8 vs 1/8 on rep3** | breaks proposal-independence: directed rounds are OUT of the Good-Turing/Chao1 estimator; category-level bank visibility (weak unsealing, recorded) |
| 2 | **(i) raise P** | subset curves still rising at P=4 (.19 -> .35 -> .47 -> .56 held-out-high); beta-binomial fit: **70% at P~6, 80% at P~8**; zero-inflated fit (asymptote pi=.88): 70% at P~7, 80% at P~10 | ~2-6 more sealed calls/round; keeps the estimator valid — the only route that does |
| 3 | **(v) detector-recall improvements** — lenient/majority-of-3 match rule, or authored same-concept anchors per Section 4.3-5 of the parent note | +.04-.08 (bounded by the either-judge readout) | ~1 extra judge per replicate; must keep anchors authored, not cosine-derived |
| 4 | **(iv) iterative rounds with bank visibility** | untested here; T1/T2 predict it mostly redirects the same prior (the pilot ran this way and produced the mined-register recaptures) | breaks sealing entirely; Good-Turing degenerates again (the original M1 motivation) |
| 5 | **(ii) error-conditioned slices** | **~0 — measured dead.** Oracle worst-predicted slice surfaces held-out columns no better than the shown slice (.572 vs .565 non-null; random .526) | would also break label-blindness if done for real |

### The measured intervention (2 proposer calls, rep3 = the weakest replicate, sensitivity .125)

Two sealed gpt-5.6-luna calls on the SAME rep3 depleted slice (`q5_intervention.json`;
prompts and raw outputs in the scratchpad `robust_mm/audit_rep3/`):

* **Arm R (control, raise-P route):** original sealed prompt, fresh ordering salt.
* **Arm T (directed):** + an 8-category coverage-sweep block over general
  scientific-reporting quality domains (design reporting; outcomes/precision;
  data/code availability; ethics; abstract accuracy/balance; citation practice;
  software artifacts; accessibility) + an instruction to phrase criteria in
  reporting-checklist register. Category-level direction only — no concept names, no
  bank sight; held-out and retained targets come from the same taxonomy, so the
  direction is symmetric and the lift readout stays fair.

Readout without new judging (the sealed Opus instrument was not re-run):
mechanical max-cosine of each of rep3's 16 recall targets against each arm.

| arm (n proposals) | held-out mean max-cos | retained mean | global max on targets | tau>=.79 hits | non-target-38 mean (specificity) |
|---|---|---|---|---|---|
| original fleet P=4 (60) | .617 | .657 | .713 | 0 | .613 |
| Arm R redraw (15) | .591 | .588 | .670 | 0 | .574 |
| **Arm T directed (15)** | **.690** | .675 | **.803** | **1** | .632 |

The directed arm shifts held-out target cosine by **+.10 against a same-size undirected
draw** while the 38 non-target concepts move only +.06 — the shift is target-shaped,
not blanket register inflation — and it puts a target above the tau band for the first
time in the entire battery (every prior fleet's global max was .722). Analyst read of
the 15 directed criteria against rep3's 8 held-out concepts (NON-BLIND, descriptive,
not the sealed instrument): "Accurate, balanced abstract claims" ~ "Abstract accuracy,
completeness, and balance"; "Data and code availability" ~ "Data availability and
sharing"; "Reproducible methodological detail" ~ "ML experiment setup transparency";
"Outcome precision and uncertainty" ~ "Outcome measures/precision"; "Study design
transparency" ~ "Study design description"; "Research artifact usability" ~ "Research
software artifact quality"; "Accessible scientific communication" ~ "Accessibility and
inclusive communication" — **~6-7/8 plausible matches vs the 1/8 the sealed judges gave
the original P=4 fleet**, on one call. (That last pair sits at cosine .585 — one more
exhibit that cross-register cosine is broken as a detector.) Confirming this as a
sensitivity number requires one sealed-judge wave (~2 Opus calls), which this audit was
scoped not to run.

The estimator implication, stated plainly: a directed round is a non-independent draw
and can NEVER feed the missing-mass estimator. The clean protocol is two tiers —
sealed independent fleets for M-hat and the mass bound, plus a directed
taxonomy-sweep round for COVERAGE, reported separately and never pooled.

## 5. What this changes for the M3 gate

1. The .70 floor should be evaluated on a **taxonomy-directed tier at P >= 6**, with
   the sealed tier retained for the estimator and the lift/control published as the
   primary honesty statistic (it stays zero under every intervention above except
   directed prompting, whose lift is expected to stay zero TOO — direction widens the
   prior for held-out and retained alike; that is the correct behaviour, not a bug).
2. **Register-match the bank at authoring time** (parent note Section 4.3-3 stands):
   7-8/24 held-out concepts were unrediscoverable for register reasons alone. A
   register-matched bank would raise measured sensitivity without touching the fleet.
3. Judge-side: adopt majority-of-3 or report strict AND lenient, since the strict rule
   sits ~.04-.08 below a defensible lenient one, and all 9 disagreements were genuine
   partial overlaps.

## Caveats

1. Exploratory; no prereg; n=24+24 targets, stratum cells 9/9/6; all CIs wide.
   Nothing here changes Delta_plateau = +.081 or the parent note's quotable bound.
2. The dose-response is fit inside a compressed strength range (alone-AUC .48-.61);
   the top-end statement is "still rising at .607", not "rises forever".
3. The beta-binomial and zero-inflated P-extrapolations assume proposer
   exchangeability and independent catches; they disagree about the asymptote
   (1.0 vs .88) in exactly the region the P=4 data cannot resolve.
4. The intervention readout is mechanical-cosine + non-blind analyst read; the sealed
   instrument was deliberately not re-run (no-judging scope). The +.10 target-specific
   cosine shift and the 1 tau-hit are instrument-independent facts; the "~6-7/8" is not.
5. Arm T's category list was authored with knowledge of the bank's taxonomy (though
   not of the holdout assignment); a confirmatory version should derive categories
   mechanically from the bank (e.g., clustering concept names) to remove author
   discretion.
6. Per-concept depletion refits measure slice/AUC movement for SINGLE removals; the
   fleets saw 8-at-once slices, so per-concept churn is a counterfactual the fleets
   never directly read. The churn-rediscovery correlation (.777) is confounded with
   strength (.742) and cannot be separated at n=24.

## Artifacts

| file (under `methods/taste_decomposition/closure/robust_mm/recovery_audit/`) | what |
|---|---|
| `q1_dose_response.py/.json` | per-concept table, logistic/rank fits, top-end rates |
| `q2a_perconcept_deplete.py/.json`, `q2a_perconcept.log` | 24 single-concept depletion refits: AUC drop + slice churn + non-null surfacing |
| `q2b_nearmiss.py/.json` | true-match cosine calibration + nearest-candidate autopsy of all 16 misses |
| `q3_mechanism.py/.json` | T1 recurrence, T2 same-model-cross-slice recapture, subset curves, beta-binomial + zero-inflated P-extrapolations |
| `q4ii_error_slice.py/.json` | oracle error-conditioned slice diagnostic |
| `q5_intervention.py/.json` | two-arm rep3 intervention (redraw vs taxonomy-directed) |
| `recovery_audit_summary.json` | headline numbers for all four questions in one file |
| scratchpad `robust_mm/audit_rep3/` | sealed prompts, raw codex outputs, parsed criteria |
