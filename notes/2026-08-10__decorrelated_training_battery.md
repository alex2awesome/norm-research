# Decorrelated training — planted-check battery (debias instrument #3)

Date: 2026-08-08 (note filed under the campaign's 08-10 slot). Agent:
claude-decor-battery-fable. Status: **COMPLETE — V2' FAIL (utility legs) with
mechanism identified; reliance-removal leg PASSES decisively; cap bonus arm
run as diagnostic: shortcut-suppression hypothesis NOT supported (§5).**
Spec: `notes/2026-08-05__taste-decomposition-design.md` §12 — BINDING (gate
battery runs BEFORE any scaled decor arm). Prior instruments in the series:
GRL (adversarial gradient reversal) — RETIRED, definitive negative
(`notes/2026-08-07__debias_audit_fable.md`); LEACE (closed-form linear erasure)
— separate agent, own battery. This instrument is the third and last:
importance-REWEIGHTING of the dense training distribution so y ⊥ the named
nuisance score. No text edits, no row deletion, no adversarial dynamics.

Terms unpacked: **decorrelated training** = retrain the dense standard with
per-example weights that make the label independent of a named nuisance score
in the reweighted training distribution — it removes the model's INCENTIVE to
learn the shortcut, not the shortcut's decodability. **V1/V2'/V3'/V4'** = the
planted-check gates (EXPLOIT / REMOVAL-OF-RELIANCE / SPECIFICITY / CONSISTENCY);
primes mark that, unlike the GRL battery's V2, the removal gate here tests
RELIANCE (token-ablation ≈ 0), not representation-level decodability — a
reweighted model may still be able to READ the channel; it must not USE it.
**n_eff** = (Σw)²/Σw², the effective sample size after weighting. **T** =
dense-standard clean-eval AUC (Llama-3.1-8B LoRA reward model). **plant
⟦QX7⟧** = synthetic prefix token, P(plant|y=1)=.65 / P(plant|y=0)=.35.
**realtok ⟦RS4⟧** = the content-derived real-signal token from the GRL battery
(V3a corpus). **joint-B score** = grouped-OOF logistic P̂(y|B) on a cell's
Track-B (mined-spurious) criterion score matrix.

## 0. Reuse (nothing retrained that already existed)

- Corpora, splits, plants, 26-col nuisance matrix: `debias/build/` unchanged
  (planted rates verified .654/.351; docket-grouped stable-hash splits
  7,618/953/950).
- Vanilla baselines REUSED from the GRL pilot: R00 (unplanted vanilla, eval
  .7926), R01 (planted vanilla, eval .8088), R06 (realtok vanilla), R08 (v3b
  vanilla) — stored runs + probes on sk3 `debias_pilot/runs/`.
- Probe machinery (`probe_reps.py`), paired docket-bootstrap discipline
  (FREEZE CHANGE 3), ablation readout — all reused verbatim.
- New code: `methods/taste_decomposition/debias/decor/` (fit_weights.py,
  train_decor.py, analyze_decor.py, make_cap_assets.py, run_battery_decor.sh).

## 1. The weights (recipe to freeze if the battery passes)

w_i = P̂(y_i) / P̂(y_i|s_i) — **stabilized** inverse-propensity form.
DECLARED CHOICE: the design's literal w ∝ 1/P̂(y|s) forces the weighted
positive rate to .50 (from .783 on this cell), which would confound the V2'(c)
task-AUC gate with a class-rebalancing intervention the vanilla never received;
the stabilized form yields the identical independence P̃(y,s)=P(y)P(s) with
both marginals preserved. Both n_eff values reported below.

P̂(y|s): logistic (StandardScaler + LogisticRegression C=1), 5-fold
GroupKFold on the grouping unit (docket / contest), OOF predictions, TRAIN rows
only; eval/test rows get w=1 and are always scored unweighted. p floored at
1e-3; weights clipped at the 99th percentile then renormalized to mean 1.

Application: **per-example loss weights** (loss = mean_i w_i·BCE_i), NOT
WeightedRandomSampler — deterministic, keeps epoch composition / step count /
LR schedule identical to the vanilla arms (paired-comparison discipline), exact
reweighted expectation without duplicate-row variance. Verified numerically by
a 2-batch gradient-linearity check on the real model and real batches
(`--gradcheck`: grad(w-weighted batch) vs Σᵢ wᵢ·grad(example i alone); PASS
threshold cosine > .999, rel err < .02; dropout disabled for determinism):
**PASS** — batch 1 cosine .9999965 / rel err .0027, batch 2 cosine .9999975 /
rel err .0024, on 9 probe tensors (score head + first/last-layer LoRA pairs);
`runs/D02_decor_planted_plant/gradcheck_report.json`. The trainer demonstrably
honors the weights.

Weight-fit diagnostics (all four N&C targets + cap):

| weights (target) | n_eff (frac) | clip rate | w range | AUC(ŝ,y) unw→w | pos rate unw→w | spec-literal n_eff frac |
|---|---|---|---|---|---|---|
| plant (D02) | 7,152/7,618 (.939) | .0000 | .64–1.71 | .637 → .484 | .783 → .783 | .572 |
| standard 26-col (D07) | 6,859 (.900) | .0101 | .22–2.61 | .638 → .458 | .783 → .803 | .620 |
| length (D10) | 7,431 (.975) | .0101 | .49–1.42 | .634 → .539 | .783 → .799 | .683 |
| date, v3b subsample (D09) | 5,457/5,458 (1.000) | .0068 | .92–1.05 | .458 → .446 | .820 → .819 | .589 |
| cap joint-B (D21) | 8,480/8,703 (.974) | .0101 | .60–1.57 | .575 → .487 | .500 → .500 | — |

Reading: the plant decorrelation equalizes the weighted plant rates exactly
(.580/.580 across classes); the date weights on the v3b subsample are ≈1 by
construction (the channel carries nothing there — decorrelating an unused
channel is a near-no-op at the WEIGHT level, which is what V3'(b) then checks
at the MODEL level). Length's weighted AUC(ŝ,y) lands at .539 not .500 — OOF
propensities can't be perfectly calibrated; residual documented. The
spec-literal (unstabilized) weights would have cost 32–43% of n_eff vs 0–10%
stabilized.

## 2. Arms

| tag | corpus | weights target | role |
|---|---|---|---|
| R00/R01/R06/R08 | (stored GRL-pilot vanillas) | — | baselines, NOT retrained |
| D02 | planted | plant | V2' removal-of-reliance |
| D00_s1 / D00_s2 | real | — (seeds 1, 2) | vanilla seed band for V2'(c) |
| D07 | realtok | standard (26-col) | V3'(a) real-signal survival |
| D09 | real, v3b subsample | date | V3'(b) unused-channel cost |
| D10 | real | length | V4' consistency |
| D20 / D21 | cap_crowd | — / mined joint-B (25 cols) | bonus arm (payoff) |

Trainer: `train_decor.py` = the frozen dense standard (Llama-3.1-8B LoRA r16/
α32, BCE, lr 5e-5, batch 16, max_len 1024, 2 epochs, select-on-eval
UNWEIGHTED, seed 42 unless stated) with the weighted loss; no adversary, no
bottleneck.

## 3. Gates (declared before any D-arm finished; PASS rules verbatim in
`analyze_decor.py` docstring)

- **V1 EXPLOIT — verified from stored artifacts, PASS.** R01 ablation
  +.0275 [95% CI +.0161, +.0392] (recomputed from stored preds, matches the
  pilot exactly), plant probe 1.000 vs .540 unplanted control. Spec-literal
  between-run jump +.0162 [+.0011, +.0315] — marginal on a .02 bar, as both
  prior batteries documented; the ablation readout is the declared V1 basis
  (audit note §4 precedent).
- **V2' REMOVAL-OF-RELIANCE**: (a) |auc_eval(D02) − auc_eval(R00)| ≤ .005
  spec-literal, reported with CI (known noise-limited: independent-run CI
  half-width ≈.03 at n=953); (b) CAUSAL PRIMARY |ablation Δ_eval(D02)| ≤ .005;
  (c) D02 eval AUC ≥ min of the 3-seed vanilla band. Chain-gate = (b) AND (c).
  Probe on D02 reps reported as SCOPE ONLY (frozen substrate carries the plant
  at .955 before any training; non-decodability is not claimed and not
  required — the instrument certifies non-USE).
- **V3' SPECIFICITY**: (a) |ablΔ(D07) − ablΔ(R06)| < .005 (realtok
  contribution survives standard-nuisance decorrelation; difference-of-deltas,
  joint docket bootstrap); (b) |auc_eval(D09) − auc_eval(R08)| < .005 literal,
  PASS-within-resolution recorded if CI ∋ 0 and |diff| ≤ .02 (declared here,
  not post hoc).
- **V4' CONSISTENCY**: primary readout EVAL+TEST POOLED (n=1,903; declared —
  the pilot showed the three instruments disagree in sign on a single 953-row
  split). Implied length influence = stratified-drop(R00) − stratified-drop
  (D10) (the CHANGE in length reliance), plus within-stratum AUC advantage;
  compared to the two ADOPTED instruments on R00's own preds (matched
  sampling caliper .02; stacked increment). PASS = same sign, ratio in
  [.5, 2]; if the references are incoherent (opposite signs) or both < .005,
  recorded INDETERMINATE (non-blocking — a property of the cell, not the
  instrument). Pre-registered reference values on R00 (computed from stored
  preds before any D-arm landed): eval — stratified drop +.0374, matched(.02)
  −.0089, stacked +.0070 (pilot §3.1 reproduced exactly); evaltest —
  stratified +.0348, matched(.02) −.0212, stacked +.0182.
- **CAP bonus (no gate; runs only if V2'–V4' pass)**: D20 vanilla vs D21
  joint-B-decorrelated retrain on cap_crowd (10,893 rows, contest-disjoint
  dense split 8,703/1,098/1,092, pos rate .500). Mined nuisance set = all 25
  B-routed non-collapsed criteria from closure rounds 1+2. References:
  archived same-rows T .5554 (n=2,190), bank VA_nl .6656
  (`notes/2026-08-06__samerows_T_rescores.md`). D20 exists because the
  archived T came from a different training pipeline: the decor effect must be
  read as D21 − D20 (same trainer, same split, paired), not D21 − .5554.
  Tests the user's shortcut-suppression hypothesis: does removing the
  incentive to learn mined shortcuts free capacity for real signal on the
  cell where the bank beats the dense model by .110?

## 4. Results

### V2' REMOVAL-OF-RELIANCE — **FAIL** (mechanism works, utility gate fails)

Landed 2026-08-08T11:36Z (gate rc=3; chain stopped per spec; GPU released).

| sub-gate | statistic | value | verdict |
|---|---|---|---|
| (b) causal reliance (PRIMARY) | D02 plant-ablation Δ eval | **+.0023** [−.0049, +.0096] | **PASS** |
| — reliance change vs planted vanilla | ablΔ(R01) − ablΔ(D02), eval | +.0252 [+.0154, +.0354] | 92% of reliance removed |
| — evaltest legs | ablΔ(D02) evaltest / test | +.0094 [+.0007, +.0169] / +.0156 [−.0002, +.0279] | small residual, nonzero on pooled leg |
| (a) spec-literal | auc_eval(D02) − auc_eval(R00) | **−.0276** [−.0480, −.0067] | FAIL (in the UTILITY direction, not exploit) |
| (c) seed band | D02 .7650 vs band [.7752, .7926] (R00 s42 .7926 / s1 .7752 / s2 .7839) | −.0101 below band min | **FAIL** |
| probe (scope only) | plant probe on D02 reps | .977 | as predicted: decodable, not USED |

Reading: decorrelated training does exactly what it promises on the reliance
axis — the planted channel's causal contribution collapses from +.0275 to
+.0023 — and unlike GRL it never amplifies reliance. But the retrained model
pays a task-AUC cost (~.02–.03 on eval, ~.01–.03 on test) that the n_eff
arithmetic (.939) cannot explain. Seed spread on this recipe is real
(vanilla band width .017 over 3 seeds; SD ≈ .009), and D02 sits ~2 SD below
the vanilla mean — a marginal-to-real utility loss, in the direction that
matters for production.

**Consequence per binding spec: the gated chain stopped; no downstream number
from this battery is gate-certified.** Remaining arms below ran as
DIAGNOSTICS (coordinator-ordered full table), including a new attribution arm:
**D03 = plant-derived weights on the UNPLANTED corpus** (same weights, same
rows, no plant text) — D03−R00 isolates the pure cost of reweighting; D02−D03
isolates the planted-text interaction.

### D03 attribution arm — the cost is the REWEIGHTING, and the mechanism is found

D03 (plant-derived weights on the UNPLANTED corpus, same rows): eval **.7615**
/ test .6733 ≈ D02 (.7650/.6928) — both ~.03 below vanilla. The planted TEXT
contributes nothing to the cost (D02 − D03 = +.0035); the weighting itself
does.

Mechanism (verified from the weight files): pooled n_eff hides a
CLASS-CONDITIONAL collapse. The decorrelation weights concentrate mass on the
rare counter-correlated cell — (y=0, plant=1) gets w=1.66 while (y=0,
plant=0) gets .65 — so the effective NEGATIVE sample shrinks by ~20% on this
78%-positive cell, exactly where AUC resolution lives:

| weights | n_eff frac y=0 (n=1,654) | n_eff frac y=1 (n=5,964) | pooled (reported §1) |
|---|---|---|---|
| plant | **.809** | .982 | .939 |
| standard | **.796** | .929 | .900 |
| length | .941 | .985 | .975 |

In-battery prediction registered BEFORE D10 landed: length weights are gentle
on negatives (.941), so D10's utility cost should be much smaller than
D02/D03's. Production implication regardless of the remaining arms: report
and gate on CLASS-CONDITIONAL n_eff, not pooled; on imbalanced cells consider
decorrelating within the majority class only, or capping the minority-class
weight ratio.

### V3'/V4'/cap — DIAGNOSTIC arms (chain completed 2026-08-08T19:38Z; NOT gate-certified)

**V3'a real-signal survival (D07 realtok + standard-weights vs R06):** the
⟦RS4⟧ contribution did NOT erode — it GREW: ablΔ eval +.0369 (D07) vs +.0272
(R06), difference-of-deltas **+.0097 [+.0032, +.0168]**; evaltest +.0044
[−.0009, +.0119]. Literal |dod|<.005 FAILS on eval — but in the direction
V3'a does not guard against (erosion of real signal is absent; reliance on
the real channel increased, consistent with capacity shifting off the
suppressed nuisances onto content). Probe controls clean: plant probe .55 =
chance on this plant-free corpus (probe calibration holds); realtok probe 1.0
(decodable — scope, as everywhere). D07 utility: eval .7703 vs R06 .8061
(−.036) — the worst weight set by minority-class n_eff (.796), as predicted.

**V3'b unused-channel cost (D09 v3b date-weights vs R08): PASS literal.**
+.0033 [−.0148, +.0210] on eval (test −.0106, run noise). Weights ≈1 → no
cost, as the weight-level diagnostics predicted.

**V4' consistency (D10 length-weights vs R00), primary evaltest:
INDETERMINATE-REFS-DISAGREE (pre-declared rule).** On R00's own preds the two
ADOPTED reference instruments are incoherent on this cell/regime: matched
sampling (caliper .02) says length-control RAISES AUC (−.0117 drop) while the
stacked increment says length ADDS +.0255 and stratification says +.0348.
The decor readouts: between-run pooled R00−D10 = +.0114 [−.0007, +.0241];
implied influence (stratified-drop change) = −.0048 (D10's length-stratified
drop .0396 vs vanilla .0348 — decorrelated training did NOT reduce the
length-stratified drop); within-stratum advantage −.0162 (decor slightly
worse within strata). Honest reading: on this docket-disjoint cell length is
substantially REAL signal entangled with content (stacked +.0255); a model
trained under length-decorrelation still shows a stratified drop because
content proxies of length remain predictive, while it pays a small pooled
cost (−.005 eval). The consistency test cannot resolve against incoherent
references — recorded, non-blocking, with the incoherence itself flagged as a
finding about the adopted instruments at this n.

### CAP bonus arm — shortcut-suppression hypothesis NOT supported

D20 (fresh vanilla, this trainer) vs D21 (joint-B decorrelation, n_eff .974,
balanced classes so no minority-class collapse):

| split | T_van (D20) | T_decor (D21) | paired diff [95% CI] |
|---|---|---|---|
| eval | .6364 | .6142 | **−.0222 [−.0328, −.0116]** |
| test | .5721 | .5829 | +.0108 [−.0002, +.0214] |
| evaltest | .6047 | .5988 | −.0059 [−.0144, +.0025] |

- De-shortcutted training does NOT recover real signal on cap_crowd:
  T_decor ≤ T_van (eval leg significantly negative, test leg positive-noise,
  pooled null); neither arm approaches the bank (VA_nl .6656). The
  bank-over-dense gap on this cell is not explained by shortcut-incentive
  during dense training.
- B-channel reliance was ALREADY small in the vanilla: B-stratified drop
  .0066 (van) / .0044 (decor); stacked increment of joint-B over the score
  NEGATIVE for both (−.0094 / −.0059) — the mined B set adds nothing on top
  of either model, so there was little shortcut reliance to remove
  (consistent with the closure discount's T_adj ≈ T finding).
- SIDE FINDING worth its own follow-up: the fresh same-recipe vanilla D20
  lands at **.6047 evaltest on the same 2,190 held-out rows where the
  archived T is .5554** (+.049). The modern select-on-eval trainer beats the
  archived pipeline; the quoted bank-vs-dense gap for cap_crowd (−.110)
  shrinks to −.066 under a matched-trainer T. Any interpretation of
  bank>dense gaps should re-base on a fresh-vanilla same-rows T first.

## 5. Verdict + production recommendation

**Gate table (battery verdict: V2' FAIL → instrument NOT certified for
production retraining as specced; all post-V2' arms are diagnostics):**

| gate | verdict | basis |
|---|---|---|
| V1 EXPLOIT | **PASS** (stored) | R01 ablation +.0275 [+.0161,+.0392], probe 1.000 vs .540 |
| V2' (b) reliance removal | **PASS** | ablΔ +.0023 [−.0049,+.0096]; 92% of reliance removed |
| V2' (a) spec-literal AUC | FAIL | −.0276 [−.0480,−.0067] (utility direction) |
| V2' (c) seed band | **FAIL** | .7650 vs [.7752,.7926]; −.010 below min |
| V3'a specificity | survival CONFIRMED / literal fail | real-channel reliance GREW +.0097 [+.0032,+.0168] |
| V3'b unused channel | PASS | +.0033 [−.0148,+.0210] |
| V4' consistency | INDETERMINATE | adopted refs sign-incoherent on this cell (declared rule) |
| CAP hypothesis | NOT SUPPORTED | T_decor ≤ T_van; B-reliance was already ≈0 |

**What survives, cleanly:** decorrelated training is the only instrument in
the debias series whose REMOVAL MECHANISM actually works — GRL amplified
reliance under pressure (ablation +.028→+.123); reweighting collapses it
(+.0275→+.0023) with no adversarial pathology and no text edits. The failure
is a quantified UTILITY cost with an identified mechanism (minority-class
effective-sample collapse), a monotone dose-response across five arms:

| arm | y=0 n_eff frac | Δeval vs its vanilla |
|---|---|---|
| D09 (date, v3b) | 1.000 | +.003 |
| D10 (length) | .941 | −.005 |
| D02 (plant, planted text) | .809 | −.028 |
| D03 (plant weights, unplanted text) | .809 | −.031 |
| D07 (standard) | .796 | −.036 |

**Production recommendation:**
1. Do NOT ship T_decor arms into the full-grid ledger on the current recipe:
   the utility gate fails exactly where decorrelation bites (imbalanced cells
   × strong nuisances).
2. If a decor arm is ever fielded, freeze this recipe: stabilized IPW
   w = P̂(y)/P̂(y|s) (grouped-CV logistic OOF, train-only, p-floor 1e-3,
   99th-pct clip, mean-1 renorm; per-example loss weights; 2-batch gradcheck
   mandatory) PLUS the NEW gate this battery discovered: **class-conditional
   n_eff ≥ .95 in the minority class** — pooled n_eff is not a safe
   pre-check (.939 pooled hid .809 minority). Mild channels (length-like)
   pass; strong binary nuisances on imbalanced cells will not.
3. The instrument's valid production role is NOT wholesale retraining but
   (i) a reliance CERTIFICATE (train a decor arm, read the ablation/reliance
   change — it measures how much of T is shortcut-carried) and (ii) targeted
   decorrelation of mild channels where minority-class n_eff survives.
4. Spurious-influence control on readouts stays with the two ADOPTED
   instruments (stacked increment + matched sampling) — with the new caveat,
   documented by V4', that they can disagree in sign at n≈2K on
   docket-disjoint N&C; read them jointly, never singly.
5. cap_crowd: retire the shortcut-suppression explanation for bank>dense; and
   re-base that cell's Δ on a fresh same-recipe T (+.049 vs archived on the
   same rows) before quoting the gap.

## 6. Artifacts

- Code: `methods/taste_decomposition/debias/decor/` (fit_weights.py,
  train_decor.py, analyze_decor.py, make_cap_assets.py, run_battery_decor.sh,
  run_diagnostics_decor.sh); configs `debias/configs/D*.json`; weight
  npz+reports `debias/build/weights_*.npz(.report.json)`,
  `debias/build/cap/`.
- Results: `debias/results/battery_decor.json` (all gates + CIs + alignment
  gates); local run mirrors `debias/runs/D*/`
  (result.json/probe.json/preds_slim.csv;
  D02 also gradcheck_report.json).
- sk3: `datasets/notice-and-comment/debias_pilot/` (runs/D* incl. reps.npz,
  logs/decor_*, results/battery_decor.json). GPU ledger: GPU 7 claim
  09:22Z → RELEASE 11:36Z (gate=v2p rc=3 STOP); GPU 4 CLAIM-STACKED 16:50Z
  (co-tenant nntruong parked vLLM, never touched) → RELEASE 19:38Z (rc=0
  diagnostics complete). No orphaned processes; nothing killed.
- Process notes: (i) `run_diagnostics_decor.sh` reruns are idempotent
  (skips tags with result.json); (ii) the D-arm trainer is
  `train_decor.py` — `train_grl.py` was left untouched (LEACE agent shares
  the directory); (iii) cap assets live under `build/cap/` inside the N&C
  debias_pilot dir purely for tooling co-location.
- Alignment gates (coordinator directive 2026-08-08): this battery performs no
  positional-order joins — every cross-arm comparison hard-asserts doc_id
  equality; weights are doc_id-keyed with full-train-coverage asserts; the
  cap B-matrix was built under an exact row_id==population-id assert; VA_nl
  references are quoted constants, never recomputed via oof-array joins.
  Recorded in `results/battery_decor.json:alignment_gates`.
