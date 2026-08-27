# Adversarial debiasing (gradient reversal) — pilot + planted-check battery, N&C responded

Date: 2026-08-06 (completed 2026-08-07). Status: COMPLETE — V2 FAILED, GRL not validated.
Spec: `notes/2026-08-05__taste-decomposition-design.md` §9 — BINDING. User-approved
2026-08-06. **No real-cell number in this file is quotable until V1–V4 all pass**;
each section states its own gate verdict.

Terms unpacked on first use. **GRL** = gradient-reversal layer (the adversarial
debiasing mechanism: a second head predicts named nuisance channels from the
model's internal representation, and its gradient is multiplied by −λ on the way
back into the encoder, so the encoder is pushed to *stop encoding* those
channels). **λ_adv** = the reversal weight. **T** = dense-standard clean-eval AUC
(Llama-3.1-8B LoRA reward model on raw text). **VA_nl** = the nonlinear
(gradient-boosting) aggregation of the V (verifiable/surface) + A
(articulated-criterion, LLM-judged rubric) score matrix. **V1–V4** = the four
planted-check gates the design doc requires before any debiased number is
quoted: V1 EXPLOIT, V2 REMOVAL, V3 SPECIFICITY, V4 CONSISTENCY. **probe** = a
fresh 2-layer MLP trained on *frozen* representations to recover a channel — it
measures whether the channel is still readable, independent of whether the task
head uses it.

## 0. Setup that everything else depends on

### 0.1 Cell, population, splits
- Cell: N&C responded (regulatory comment → agency responded), 9,521 rows /
  1,814 dockets, positive rate .781. Population is the canonical Layer-1/closure
  one (`methods/taste_decomposition/closure/nc_responded/nc_responded_base.npz`).
- Splits: **docket-grouped stable hash**, dockets ordered by
  `sha256(docket)/2**256` and cut at 80% / 90% of *cumulative rows* →
  train 7,618 / eval 953 / test 950, docket-disjoint (asserted). Cutting on the
  hash value directly gives 80.1 / 6.6 / 13.3 by row, because docket sizes range
  from 1 to hundreds; the cumulative-row walk keeps the rule deterministic and
  group-disjoint while hitting 80/10/10.
- Positive rate by split: train .783 / eval .699 / test .852.

**This is a different regime from the archived N&C dense chain.** That chain used
a random row split, so dockets leak across train/eval; docket-identity-alone AUC
on this cell is **.916** (Layer 2 part (a)), and its same-rows held-out T is
**.8167**. A docket-disjoint split removes that channel by construction, so the
vanilla baseline here is expected to sit well below .8167 and the two numbers are
not interchangeable. Reported below.

### 0.2 Recipe (frozen dense standard, verified against the archived `train.log`)
Llama-3.1-8B, LoRA r16 / α32 / dropout .05 on q,k,v,o,gate,up,down;
`num_labels=1` + BCEWithLogits; lr 5e-5, wd .01, warmup .1, batch 16,
grad-accum 1, max_length 1024, 2 epochs, gradient checkpointing, seed 42,
5 validation checkpoints per epoch. Two declared departures: **select-on-eval**
(the archived chain selected on *test*) and the docket-grouped split above.

### 0.3 GRL implementation
Pooled representation h = last hidden state at the final non-pad position — the
exact vector `LlamaForSequenceClassification`'s scalar head reads (left padding,
asserted per batch). h → `GradReverse(λ)` → 2-layer MLP (4096→256→d) → the
standardized nuisance vector, MSE. The adversary head minimises normally
(lr 1e-3, wd 0 — a fresh head at 5e-5 never becomes a competent adversary, which
would make the reversal vacuous); the encoder receives −λ × that gradient.
λ ramps linearly over the first 10% of optimizer steps, identically in every GRL
run, so λ=1.0 does not destabilise step 1. Sweep: λ ∈ {0.1, 0.5, 1.0}.

### 0.4 Nuisance vector (26 columns, z-scored on TRAIN rows only)
The Track-B mined channels for this cell are **not available** — the closure
campaign's `nc_responded` round-1 Gemma scoring has not landed
(`round1_track_b_discount.json` does not exist), and the round-1 B routing list
contains no length channel anyway. So the Layer-2 nuisance set is used, as the
task's fallback directs:

| group | columns |
|---|---|
| length | `char_len`, `log_token` |
| format | `linebreak_rate`, `markdown_rate` (Layer-2 `_MARK_RE` definitions, reused verbatim) |
| date | `docket_year`, `year_missing` |
| topic | 20 one-hot k-means clusters on `BAAI/bge-large-en-v1.5` (Layer-2 embedding cache), **k-means fit on train rows only** |

**Date landmine.** Layer 2 dropped date for this cell because `year` in
`nc_vat_sample.jsonl` / `nc_unmatched_sample.jsonl` covers only 78% of rows —
and that 78% is *exactly the matched pool*: positive rate 1.000 among rows that
carry a year, .003 among rows that don't. Using that field would have made the
"date channel" a near-perfect label leak. The date channel here is instead
derived from the **docket ID** (`FSIS-2016-0021` → 2016): 94.7% coverage,
present for both classes, no leak.

Channel-alone AUC (docket-grouped OOF logistic regression, full population):
length **.621**, date **.582**, format **.516**, topic **.479**, all-standard
jointly **.654**.

### 0.5 The two synthetic plants
Both are mechanical prefixes — no model rewriting (the design doc REJECTED
counterfactual text editing). Prefix, not suffix: `max_length=1024` truncates
from the right and 25% of these comments hit the 4,000-char cap (≈1,000 tokens),
so an appended token is not guaranteed to survive tokenisation.

**V1/V2 plant `⟦QX7⟧`** — target P(plant|y=1)=.65 vs P(plant|y=0)=.35; realized
**.6541 / .3511** (overall .588). Plant-alone AUC **.6515** (eval .6579).
Per-split realized rates are in `debias/build/build_report.json`.

**V3a real-signal token `⟦RS4⟧`** — the design's suggested "top-quartile
evidence-specificity rubric score" is **not usable on this cell**: every
evidence/specificity criterion in the 198-rubric A bank has alone-AUC in
[.487, .522] here, so a single-criterion quartile token would carry no learnable
signal and V3a would pass vacuously. Substituted: each A criterion is linearly
residualised on the nuisance matrix (train-fit), a docket-grouped OOF gradient
boosting composite is fit on the residualised criteria (composite AUC .785), the
composite is residualised again, and the token marks rows above the **train
median**. Result: token-alone AUC **.617**, and only **4.3%** of the token's
variance is explained by the whole nuisance matrix — so any V3a erosion cannot be
blamed on legitimate length removal. (A token on the *raw* composite would be
14.8% nuisance-explained; a quartile rather than median cut costs channel
strength: .551 vs .617.) The composite's weights are y-supervised, so this is a
*content-derived* channel rather than a y-free one; what V3a tests is whether an
**unnamed** channel survives debiasing of **named** ones, which does not depend
on how the unnamed channel was built.

### 0.6 V3b year-balanced subsample
Constructed so date is verifiably unusable: within **each split separately**,
each docket-year is subsampled to a common positive rate (.82). n = 6,291
(train 5,458 / eval 503 / test 330), 1,285 dockets.
- date-alone AUC in the subsample: **.5002** (vs .582 on the full corpus)
- train→eval generalising readout (per-year positive rate fit on train, applied
  to eval): **.4948** in the subsample vs **.7313** on the full corpus.

Balancing the *pooled* subsample is not enough — it leaves a train→eval date-alone
AUC of .41, because train and eval draw on different dockets and dockets are
nested in years. Per-split balancing fixes the split the gate is actually read on.
The target rate .82 was chosen to maximise **eval comparable pairs**
(n_pos × n_neg = 37,170); maximising raw n instead lands on a 92%-positive
subsample with ~54 eval negatives, which cannot resolve a .005 gate.

## 1. Runs

12 trainings, chained on one ledger-claimed sk3 GPU (GPU 7, claimed
2026-08-07T04:08Z in `gpu_ledger.txt`). All arms share identical rows and
identical splits — only the text overlay and the named channels differ — so every
comparison is **paired** and every interval is a **docket-level** bootstrap
(resample dockets, not rows; FREEZE CHANGE 3 discipline).

| tag | corpus | named channels | λ | role |
|---|---|---|---|---|
| R00 | real | — | 0 | baseline for V1, V4, V3b-contrast, FINAL |
| R01 | planted | — | 0 | V1 EXPLOIT |
| R02/R03/R04 | planted | standard + plant | .1 / .5 / 1.0 | V2 REMOVAL sweep (literal spec) |
| R05 | planted | plant only | best | V2 diagnostic (isolates the plant) |
| R06 | real-signal-token | — | 0 | V3a reference |
| R07 | real-signal-token | standard | best | V3a SPECIFICITY |
| R08 | real, V3b subsample | — | 0 | V3b reference |
| R09 | real, V3b subsample | date | best | V3b SPECIFICITY |
| R10 | real | length | best | V4 CONSISTENCY |
| R11 | real | standard | best | FINAL full-nuisance debias |

---

## VERDICT (read this first)

**GRL (gradient-reversal adversarial debiasing) is NOT VALIDATED for this program.
V2 REMOVAL failed at every λ in the specced sweep {0.1, 0.5, 1.0} and again at
λ=5.0 with the adversarial pressure concentrated on a single channel. No debiased
number from this pilot is quotable, and none is reported. V3, V4 and the final
full-nuisance run were NOT executed as gates** (V2 failed; the spec orders each
gate to pass before the next).

The battery did its job in the strongest possible sense: **at λ=1.0 the AUC gate
alone would have declared a clean removal** (debiased eval AUC within +.0039 of
the unplanted baseline, inside the .005 tolerance) **while the planted channel was
still perfectly readable from the representation (probe AUC .999).** Reading only
the AUC gate would have shipped a silent failure into every downstream number.

Failure mode, in the coordinator's taxonomy:
- **(a) reversal live but insufficient — CONFIRMED at λ ≤ 1.0.** The adversary is
  defeated (its held-out R² on the plant collapses .853 → .795 → .110 as λ goes
  .1 → .5 → 1.0) while a *freshly trained* probe still recovers the plant at
  .997–.999. Only the co-trained adversary is beaten; the information is never
  removed.
- **(b) removal-by-destruction — CONFIRMED at λ=5.0, and it is worse than that:
  destruction WITHOUT removal.** Task AUC collapses (−.1141 [−.1715, −.0593] vs
  the unplanted baseline) while the plant stays *linearly* decodable at AUC
  **1.000**, and the model's reliance on the plant *quadruples* (ablation Δ .0275
  → .1023 on eval, .0689 → .1800 on test).
- **(c) plant leaking through correlated features — RULED OUT.** The plant is
  correlated with y by construction, so a pooled probe could in principle be
  reading the label direction. Within each label stratum — where the plant is
  independent of y by construction — a plain logistic probe still recovers it at
  **1.000 / .999**. The plant occupies its own linear direction.

**Recommendation: stick with the two ADOPTED instruments — stacked increment and
matched sampling.** Both are already computed for this cell's length channel and
agree with each other (§3); neither requires trusting a removal procedure that
just failed a ground-truth test we designed ourselves.

Mechanism, stated plainly: the reversal gradient rewards making the representation
*unpredictable to the current adversary*, and the encoder satisfies that by
rotating/degrading the representation globally rather than by deleting the planted
direction. Under enough pressure it destroys task-relevant structure first and the
nuisance direction last — and because the task head is then starved of other
signal, it leans on the surviving plant harder than before. This reproduces the
known adversarial-removal negative result (Elazar & Goldberg 2018; the INLP line)
on our own instrument, in a setting where we planted the channel and therefore
know ground truth exactly.

---

## 2. Battery results

All intervals are 95% docket-level bootstrap (2,000 resamples) on the **paired**
difference. Selection is on eval, so eval is mildly optimistic and test is the
clean leg; both are reported. Vanilla baseline R00: **eval .7926 / test .7037 /
eval+test .7624** (best checkpoint = last, step 954 of 954).

The docket-disjoint vanilla eval AUC (.793) lands close to the archived
random-split same-rows T (.8167) — so the docket-identity channel that Layer 2
flagged (docket-alone AUC .916) turns out **not** to be carrying the dense
model's advantage on this cell, even though it carries VA_nl's. Worth recording
independently of the battery.

### V1 EXPLOIT — PASS on the ablation readout, MARGINAL on the literal gate

| readout | value | 95% CI |
|---|---|---|
| **spec-literal**: eval AUC(planted vanilla) − eval AUC(unplanted vanilla) | **+.0162** | [+.0011, +.0315] |
| same, on test | +.0401 | [+.0109, +.0714] |
| same, on eval+test | +.0226 | — |
| **within-model token ablation** on the same eval rows (strip `⟦QX7⟧` at inference) | **+.0275** | [+.0168, +.0397] |
| within-model token ablation, test | +.0689 | [+.0269, +.1093] |
| plant probe on the planted vanilla representation | **1.000** | (unplanted control: .540) |

Verdict: the vanilla model exploits the plant **decisively** — a fresh probe
recovers the plant from its representation at AUC 1.000, and deleting the token
at inference costs .0275 AUC on eval / .0689 on test. The spec's literal
statistic — differencing two *independently trained* models — lands at +.0162,
below the .02 bar, with a CI that contains .02. The between-arm difference is the
noisier estimator of the same quantity (it carries run-to-run training variance
on top of the plant effect); the within-model ablation isolates the plant on
identical weights and identical rows and clears .02. **Recorded as PASS on the
substance, MARGINAL on the literal .02-on-eval threshold** — not silently
reinterpreted.

### V2 REMOVAL — **FAIL**, at every λ, on the probe gate

Master table. All eight completed runs; identical recipe (§0.2) except λ and the
named channels. `abl Δ` = within-model token-ablation contribution (strip the
token at inference on the same weights and the same rows). `MLP probe` = the
spec's post-hoc 2-layer probe (mean of seeds 0,1,2). `lin probe` = plain L2
logistic probe. `lin y0 / y1` = the same logistic probe fit and scored **within a
single label stratum**, where the plant is independent of y by construction.
`adv R²` = the *co-trained* adversary's own held-out R² on the plant.

| tag | λ | channels | eval | test | eval+test | best step | abl Δ eval | abl Δ test | MLP plant probe | lin plant probe | lin y0 / y1 | adv R² plant | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R00_vanilla_real | 0 | — | **.7926** | .7037 | .7624 | 954 | — | — | .540 | .527 | .465 / .502 | — | 41.6 |
| R01_vanilla_planted | 0 | — | .8088 | .7438 | .7850 | 954 | +.0275 | +.0689 | **1.000** | .9999 | 1.000 / .9999 | — | 51.2 |
| R02_grl_planted_full | 0.1 | standard+plant | .8024 | .7833 | .8007 | 859 | +.0325 | +.0529 | **.999** | .9986 | .9992 / .9984 | .853 | 50.2 |
| R03_grl_planted_full | 0.5 | standard+plant | .8122 | .8151 | .8210 | 859 | +.0363 | +.0531 | **.997** | .9970 | .9888 / .9976 | .795 | 47.2 |
| R04_grl_planted_full | 1.0 | standard+plant | .7966 | .8270 | .8154 | 573 | +.0500 | +.0698 | **.999** | .9982 | .9906 / .99996 | **.110** | 47.2 |
| R05B_grl_plantonly | 5.0 | plant only | **.6785** | .6862 | .6948 | 477 | **+.1023** | **+.1800** | **1.000** | **1.000** | **1.000 / 1.000** | .125 | 47.4 |
| R06_vanilla_realtok | 0 | — | .8061 | .6932 | .7624 | 954 | +.0272† | −.0126† | .557 | .525 | .447 / .496 | — | 47.2 |
| R08_vanilla_v3b | 0 | — | .7050 | .6640 | .6888 | 616 | — | — | .508 | .531 | .404 / .531 | — | 28.4 |

† R06's ablation is of the **real-signal** token `⟦RS4⟧`, not the plant. Its own
`realtok` probe is .99993 — the real-signal channel is fully encoded, confirming
the V3a corpus construction works even though V3a was never run as a gate.

Paired docket-level bootstrap (2,000 resamples), vs the unplanted baseline R00 on
eval:

| arm | eval − R00 | 95% CI | ablation Δ eval | 95% CI |
|---|---|---|---|---|
| R01 (planted vanilla) | +.0162 | [+.0011, +.0315] | +.0275 | [+.0168, +.0397] |
| R02 (λ=0.1) | +.0097 | [−.0173, +.0355] | +.0325 | [+.0206, +.0453] |
| R03 (λ=0.5) | +.0195 | [−.0154, +.0551] | +.0363 | [+.0222, +.0501] |
| R04 (λ=1.0) | **+.0039** | [−.0349, +.0399] | +.0500 | [+.0218, +.0772] |
| R05B (λ=5.0, plant-only) | **−.1141** | [−.1715, −.0593] | +.1023 | [+.0626, +.1446] |

**Gate 1 (AUC within .005 of the unplanted baseline).** Met only at λ=1.0
(+.0039) — and its CI is [−.035, +.040], i.e. **the AUC gate has no resolution at
this n**: it cannot distinguish "removed" from "unchanged". This is the false PASS.

**Gate 2 (post-hoc probe ≤ .55 debiased, ≥ .75 vanilla). FAILS at every λ.**
Vanilla side passes emphatically (1.000 vs the .540 unplanted control, so the
probe machinery is calibrated). Debiased side: **.997–1.000, unmoved.**

**Third signal, unrequested and decisive: the plant's causal contribution grows
monotonically with λ** — .0275 → .0325 → .0363 → .0500 → .1023. The "debiased"
models rely on the planted shortcut **more**, not less. At λ=5.0 stripping the
token at inference costs .1023 AUC on eval and .1800 on test.

### The three failure modes, discriminated

**(a) reversal live but insufficient — CONFIRMED for λ ≤ 1.0.** The reversal is
demonstrably active: adversarial loss rises with λ (≈.45 at λ=.1 → ≈.85 at λ=.5)
and the co-trained adversary's held-out R² on the plant collapses **.853 → .795 →
.110** across λ = .1 → .5 → 1.0. So the encoder *is* being pushed and the
adversary *is* being beaten — yet a **freshly trained** probe reads the plant at
.997–.999 throughout. Only the particular adversary co-trained with the encoder is
defeated; the information is never removed. Note the diagnostic trap this sets:
`adv R² = .110` at λ=1.0 looks like successful removal if you read the adversary's
own loss instead of an independent probe.

**(b) removal-by-destruction — CONFIRMED at λ=5.0, in its worst form.** The single
authorised remedial run (plant as the *only* target, so 27× the per-channel
gradient, plus λ escalated 5×) collapsed task AUC by −.1141 [−.1715, −.0593] while
the plant remained **linearly** decodable at AUC **1.000**. This is destruction
*without* removal — the failure mode is not a tunable trade-off between task
performance and debiasing, because the nuisance direction survives the very
pressure that destroys the task representation. Its eval-AUC trajectory across the
10 checkpoints (.549, .502, .465, .468, .679, .593, .563, .571, .573, .572) shows
the representation degrading, not converging.

**(c) plant leaking through correlated features — RULED OUT.** The plant is
correlated with y by construction (.65/.35) and h necessarily encodes y, so a
pooled probe could in principle be reading the label direction rather than a plant
direction. Within each label stratum — where the plant is **independent of y by
construction** — a plain logistic probe still recovers it at **1.000 (y=0) /
1.000 (y=1)** on R05B and .989–1.000 on R02–R04, against .40–.53 on the three
unplanted-corpus controls. The plant occupies its own linear direction and is not
being smuggled in via y.

### V3 SPECIFICITY, V4 CONSISTENCY, FINAL — NOT RUN AS GATES

The spec orders "each gate must PASS before the next"; V2 failed, so V3/V4 are
moot as gates. If the machinery removes nothing, V3 passes *vacuously* (nothing is
damaged because nothing is removed) and V4 reduces to restating V2 (the
removal-measured influence of any channel is ≈0 — or, at λ=5.0, is swamped by a
−.114 collapse that has nothing to do with the channel). The remaining budget went
to the failure diagnosis above, then stopped. Runs that were queued and
deliberately **not** executed: `R05_grl_planted_plantonly_l1.0`,
`R12_grl_planted_plantonly_l5.0` (superseded by R05B),
`R10_grl_length_real` (the only obtainable V4 number),
`R13_grl_planted_plantonly_l20.0`, `R07_grl_realtok_standard`,
`R09_grl_v3b_date`, `R11_grl_full_real`. Their configs are written and shipped
(`debias/configs/*.json`) should anyone want them.

R06 (`vanilla_realtok`, the V3a reference) and R08 (`vanilla_v3b`, the V3b
reference) were already in the chain when V2 landed and completed; both are clean
λ=0 vanilla arms and are reported above.

### What would have to be true for GRL to be revisited

Stated so the next agent does not repeat this: the pilot as run does **not**
establish that no GRL variant can work — it establishes that *this* variant
(single co-trained MLP adversary, MSE on standardized targets, reversal on the
pooled final-layer representation, λ ≤ 5) does not. Untested variants that the
literature suggests and that this harness could carry with small edits: iterative
nullspace projection (INLP) applied post-hoc to the frozen representation rather
than adversarially during training; multiple adversaries re-initialised on a
schedule; reversal applied at several layers rather than only the pooled output;
adversary trained to convergence between encoder steps (k>1 inner steps). Each
would have to clear the *same* V1–V4 battery before any number is quoted.

## 3. Frozen length-influence instruments (V4 reference set)

Computed on the archived same-rows dense predictions
(`closure/samerows_preds/nc_responded_dense_preds_slim.csv`, held-out
`dense_split ∈ {eval,test}`, n=1,904, pooled T .8167):

| instrument | value |
|---|---|
| Layer-2 length-stratified drop, **VA_nl** (frozen, `notes/2026-08-06__layer2_robustness.md` §3) | **.073** |
| Layer-2 length-stratified drop, **T** analogue (this note) | **.0195** (pooled .8167 → stratified .7973, 5/10 qualifying deciles) |
| matched sampling on the joint-length score (caliper .01 / .02 / .05) | drop **.0130 / .0075 / .0038** (1,386 / 1,426 / 1,454 pairs) |
| stacked increment: how much **length adds on top of T** | **+.0014** (T-alone .8079 → T+length .8093) |

Reading: on the archived dense model, T already absorbs essentially all of the
length channel's label information (+.0014 increment), and removing length by
matching or stratification costs .004–.020 AUC. The VA_nl figure (.073) is
5–19× larger because it is a *different model* — the V+A stack leans on length
much harder than the dense model does. **V4's factor-of-2 test is therefore run
against the T-side instruments, not against .073**, and is additionally recomputed
on this pilot's own vanilla model so that instrument and gate share one regime
(the archived numbers come from a docket-leaky random split; see §0.1).

Artifact: `methods/taste_decomposition/debias/results/frozen_length_instruments.json`.

### 3.1 The same instruments on this pilot's own vanilla eval split — a power warning

Recomputed on R00's own eval predictions (n=953), so instrument and (never-run)
gate would have shared one regime:

| instrument | value |
|---|---|
| length-alone OOF AUC (this eval split) | **.493** |
| length-stratified drop | +.0374 (only 6/10 deciles qualify) |
| matched sampling, caliper .01 / .02 / .05 | **−.0010 / −.0089 / −.0122** |
| stacked increment, length over T | +.0070 |

The three instruments **disagree in sign** at n=953 (stratification says length
costs .037, matching says it *adds* .001–.012). On the archived n=1,904 held-out
population they agree (§3: +.0195, +.004…+.013, +.0014). So V4's "same sign and
within a factor of ~2" test was **underpowered on a single 953-row eval split
regardless of the GRL result** — a fourth reason the pilot could not have
delivered a quotable V4 number, independent of the V2 failure. Any future
consistency check should be read on eval+test pooled at minimum, and preferably on
a cell with a larger held-out population.

Artifact: `methods/taste_decomposition/debias/results/same_regime_length_instruments.json`.

## 4. Artifacts — everything needed to pick this up cold

**Code** — `methods/taste_decomposition/debias/`:

| file | what it does |
|---|---|
| `build_corpora.py` | stage 0, CPU, local. Builds splits, the 26-column nuisance matrix, both planted corpora, the V3b subsample. Deterministic (seed 20260806); rerunning reproduces the corpora byte-identically (verified by md5). |
| `train_grl.py` | dense standard + optional GRL head. `--config <json>`. Writes `result.json`, `preds_slim.csv`, `reps.npz` (all 9,521 pooled representations, fp16), `best_state.pt`, `best_adv.pt`. |
| `probe_reps.py` | the spec's post-hoc 2-layer MLP probe on frozen reps, 3 seeds, fit on the run's own train rows (10% held back for early stopping), scored on eval. |
| `linear_probe.py` | the L2 logistic probe **and the within-label-stratum probes** that rule out failure mode (c). CPU only. |
| `analyze_battery.py` | gate arithmetic + docket-level paired bootstrap (`boot_diff`) + the same-regime length instruments. |
| `make_configs.py`, `run_chain.sh` | config emitter and the one-GPU serial chain runner (skips any tag that already has `result.json`). |

**Run configs**: `debias/configs/<tag>.json`, one per run, each carrying a `note`
field stating what that run was for. Both executed and not-executed tags are
present.

**Data**: `debias/build/` (local) and
`datasets/notice-and-comment/debias_pilot/build/` (sk3) —
`population.csv`, `corpus_planted.csv`, `corpus_realtok.csv`,
`v3b_population.csv`, `nuisance.npz` (Z, raw, names, group index map, plant,
realtok, split, y, docket, v3b_keep), `build_report.json` (every rate and
diagnostic quoted in §0).

**Per-run outputs**: sk3
`datasets/notice-and-comment/debias_pilot/runs/<tag>/` has the full set including
`reps.npz` (78 MB each — not copied down). Locally,
`debias/runs/<tag>/` has `result.json`, `probe.json`, `preds_slim.csv`.
`result.json` carries the full checkpoint history, the resolved recipe, the
adversary's per-channel held-out R², and the ablation block.

**Result JSONs**: `debias/results/` —
`linear_probe.json` (pooled + within-stratum linear probes, all runs),
`paired_bootstrap.json` (every paired difference quoted in §2 with its CI),
`frozen_length_instruments.json` (§3).

**Logs**: sk3 `datasets/notice-and-comment/debias_pilot/logs/` —
`<tag>.log` (per-run training), `<tag>_probe.log`, `chain_stage1.log`,
`chain_remedial.log`.

**GPU**: sk3 GPU 7 throughout; claimed 2026-08-07T04:08Z, released
2026-08-07T19:5xZ in `gpu_ledger.txt`. No co-tenant GPU was touched and nothing
was killed. Total ≈ 6.2 GPU-hours over 8 trainings (41–51 min each).

## 5. Process notes for the next agent

1. **`run_chain.sh` reads its order file incrementally through an open fd.**
   Rewriting that file mid-chain (e.g. via `rsync` of the configs dir) desyncs the
   read offset and silently drops queued tags — this cost R05/R10/R12/R13 on the
   first pass. Either append only, or relaunch the chain (it skips completed tags).
2. **The `year` field in `nc_vat_sample.jsonl` / `nc_unmatched_sample.jsonl` is a
   label leak on this cell** (positive rate 1.000 among rows that carry it, .003
   among rows that don't). Use the docket-ID year instead. See §0.4.
3. **Read an independent probe, never the adversary's own loss.** At λ=1.0 the
   co-trained adversary's R² on the plant is .110 — it looks like removal and
   isn't.
4. Eval n=953 means a .005 AUC tolerance is below the instrument's resolution
   (paired bootstrap CI half-widths are ≈.02–.04). Any future gate at that
   tolerance needs either a much larger eval split or a within-model estimator
   (the token-ablation readout, whose CIs are ≈3× tighter).
