# LEACE closed-form-projection pilot: planted battery V1-V4 + utility-distortion frontier — N&C responded

Date: 2026-08-10 (user-approved; battery landed 2026-08-08 and was RE-ISSUED
same day after the lit review's certificate-form fix (R1a): categorical
one-hot erasers throughout — 358.6 s local CPU rerun + one 335 s GPU pass on
sk3). Agent: claude-leace-pilot-fable. Status: **CLOSED — VERDICT §7:
linear-only erasure instrument, certified for linear-score-path use, with a
large quantified nonlinear residue and a certificate-strength/surgicality
trade-off (§5).** All numbers from
`methods/taste_decomposition/debias/leace/results/battery_leace.json`
(categorical primary); run-1 continuous-only battery preserved as
`battery_leace_r1_continuousZ.json`.

Predecessor: `notes/2026-08-07__debias_audit_fable.md` — GRL (gradient-reversal
adversarial debiasing) retired with a definitive negative verdict on both
architectures. This pilot tests the literature-standard successor that note
named: post-hoc closed-form linear projection on FROZEN representations.

Terms unpacked on first use: **LEACE** = LEAst-squares Concept Erasure (Belrose
et al., NeurIPS 2023) — a closed-form affine projection that provably removes
all LINEAR decodability of a named concept from a fixed representation, with
minimal mean-squared distortion. **V1–V4** = the planted-check gates (EXPLOIT /
REMOVAL / SPECIFICITY / CONSISTENCY) carried over from the GRL battery design
(`notes/2026-08-05__taste-decomposition-design.md` §9). **h** = pooled last-token
hidden state of the frozen trained dense model (4096-d). **plant** = `⟦QX7⟧`,
the synthetic y-correlated token (P(plant|y=1)=.654, P(plant|y=0)=.351,
plant-alone AUC .652). **realtok** = `⟦RS4⟧`, the content-derived real-signal
token (alone AUC .617, R² on the nuisance matrix .043). **head** = the refit
score head: StandardScaler + logistic regression (C=1), TRAIN rows only.

## 1. The instrument

Per arm: (1) pooled h for all rows from the FROZEN trained model — no
retraining anywhere; (2) closed-form LEACE eraser for the named concept(s), fit
on TRAIN rows only; (3) refit logistic head on projected reps (train only);
(4) readouts with paired docket-level bootstrap (2,000 resamples).

The exact formula implemented (`debias/leace/leace.py`), Belrose et al. 2023
Thm 4.3:

    r(x) = x − W⁺ P_{W Σ_XZ} W (x − μ_X)

with Σ_XX the representation covariance, W = Σ_XX^{−1/2} (whitening;
pseudo-inverse square root via eigendecomposition, relative eigenvalue cutoff
1e−9), W⁺ = Σ_XX^{+1/2} (unwhitening), Σ_XZ the representation–concept
cross-covariance, and P_{W Σ_XZ} the orthogonal projection onto
colspace(W Σ_XZ) (SVD basis, relative singular-value cutoff 1e−6). Whiten →
project out the concept's cross-covariance directions → unwhiten.

**Certificate form (corrected per the lit review,
`notes/2026-08-10__litreview_spurious_debiasing.md` R1a).** The strong
guarantee — zero cross-covariance ⟺ no linear classifier beats the constant
predictor under ANY convex loss — holds for **categorical one-hot Z** (Belrose
Def 2.3 / Thm 3.1). For **continuous Z** the guarantee is **OLS-loss-only**
(their §4.3); the any-convex-loss form does not provably extend. The first run
of this battery fit the real-nuisance erasers on standardized continuous
matrices while probing binarized targets — certificate and readout were
different objects (the plant arm, being binary, was always in the strong form).
**Fix applied: every continuous nuisance is binned to TRAIN-DECILE one-hot
columns before fitting (`onehot_bins`; decile edges from train quantiles
.1–.9, deduplicated), and the eraser is fit on the one-hot matrix** —
`certificate_form: categorical` in every gate. Deciles, not median splits: the
.5 quantile is always among the edges, so each median-split probe target used
below is a UNION of erased categories — a linear function of the one-hot Z —
and inherits the certificate (verified numerically: probe exactly .5000 on
train AND holdout in the unit check). Continuous-Z erasers are retained as
labelled SECONDARY pair readouts (the review's "erase both" option); the
run-1 continuous-only battery is preserved unmodified as
`results/battery_leace_r1_continuousZ.json`. `leace.py`'s docstring is
corrected to state both forms. The guarantee is LINEAR-only and
in-sample-on-fit-rows in either form; both scope limits are probed below
(nonlinear MLP residue; eval-row transfer diagnostics). Ceiling on ANY form
(Ravfogel, Goldberg & Cotterell ACL 2023, Thm 3.4): even under guardedness
against binary readers, a multiclass softmax can recover Z almost perfectly —
so no certificate here composes past binary-linear consumers.

Verification, both required checks:
- **synthetic hand-check** (`python leace.py`): planted 2-concept synthetic —
  post-erasure max |cross-cov| on fit rows 2.2e−15 (machine zero), linear probe
  AUC .498 train / .481 holdout from a raw .988, projection rank exactly k=2:
  PASS.
- **reference-implementation check**: machine-equal to the `concept-erasure`
  package's `LeaceFitter` with `shrinkage=False` — max |Δ| = 1.4e−13 on a
  3,000×48 synthetic. (The package defaults to covariance shrinkage; our
  closed form is the exact no-shrinkage estimator. n_train=7,618 > d=4,096 and
  the eigenvalue cutoff handles conditioning.)

## 2. Arms and provenance (everything reused, one GPU pass total)

| arm | corpus | reps used | source |
|---|---|---|---|
| B00 vanilla real | population.csv | `rep_h` (pooled h, 4096-d) | existing bottleneck-battery run (sk3 `debias_pilot/runs/B00_vanilla_real`) |
| B01 vanilla planted | corpus_planted.csv | `rep_h` | existing run B01 |
| R00 vanilla real (pooled arch) | population.csv | `rep` (= pooled h) | existing pooled-battery run R00 |
| R06 vanilla realtok | corpus_realtok.csv | `rep` | existing pooled-battery run R06 |
| R06-stripped | population text through R06's frozen model | `rep` | **the one GPU pass** (`extract_reps_leace.py`, GPU 7, 335 s) |
| R08 vanilla v3b | year-balanced subsample | `rep` | existing pooled-battery run R08 |

The GPU-pass correctness certificate: R06's own scalar head applied to the
stripped-text reps reproduces `result.json`'s recorded token-ablated eval AUC
**exactly** (.7788764374130228 = .7788764374130228) — checkpoint reload and
extraction path byte-reproduce the original scoring pass.

Scope note (carried from the bottleneck battery): the LEACE certificate is
scoped to the refit-head decision path — the score is a linear function of the
projected h, so "concept linearly unreadable from projected h" ⇒ "the score
cannot linearly use the concept". The models' own heads (which read z=proj(h)
under the bottleneck arch) are reported as context only; every gate compares
refit heads under one protocol.

Refit-head baselines (eval AUC; model's own head in parens): B00 .7987 (.8283),
B01 .8238 (.8466), R00 .7741 (.7926), R06 .7849 (.8061), R08 .7201 (.7050).
The refit logistic on raw h sits ~.02 under the jointly-trained heads — all
comparisons below are refit-vs-refit, protocol-consistent.

## 3. Battery table (certificate_form: CATEGORICAL throughout — every continuous
nuisance train-decile one-hot before fitting; all probes on eval rows,
train-fit, 3-seed MLP mean. Continuous-Z (OLS-form) secondaries in §3.1.)

| gate | eraser target (arm) | concept cols → rank | linear probe raw → after | **MLP probe after (nonlinear residue)** | task readout | literal gate | verdict |
|---|---|---|---|---|---|---|---|
| V1 EXPLOIT | — (B01 raw) | — | plant **.9995** (MLP .9998) | — | head jump +.0251 [+.002, +.047] | probe ≥ .95 | **PASS** |
| V2 REMOVAL (a) | plant (B01), binary = already one-hot | 1 → 1 | plant .9995 → **.4749** | | | ≤ .55 | **PASS** |
| V2 REMOVAL (b) | plant (B01) | 1 → 1 | | plant **.9629** (seeds .9631/.9632/.9624) — residue **+.41 above the bar** | | ≤ .55 full pass | **FAIL → linear-only erasure** |
| V2 REMOVAL (c) | plant (B01) | 1 → 1 | | | utility cost −.0384 [−.0568, −.0208] eval / −.0427 test; placebo decomposition §4 | report | y-tax dominated |
| V2 REMOVAL (d) | plant (B01) | 1 → 1 | | | jump after: **−.0133** [−.0388, +.0110] (was +.0251 [+.002, +.047]) | \|jump\| ≤ .005 | literal FAIL / CI PASS — jump gone, slight overshoot |
| V3a SPECIFICITY | standard-26 (R06), 26 ch → 58 one-hot | 58 → 52 | realtok 1.0000 → **.9089**; char_len fully dead (.500) | realtok .99995 → **.9251** — still far above chance | ablation Δ raw +.0233 [+.012, +.036] → proj **+.0046** [−.005, +.016]; Δ-of-Δ **−.0187**; head cost −.1279 [−.168, −.092] | \|Δ-of-Δ\| < .005 | **representation-survival PASS / causal-use EROSION FAIL** — see §5, the pilot's second headline finding |
| V3b SPECIFICITY | date (R08, year-balanced), 2 ch → 10 one-hot | 10 → 8 | year .719 → **.438** (dead) | — | task cost **−.0010** [−.0093, +.0089] | \|cost\| < .005 | **PASS** (literal; CI covers 0 but is wider than the .005 window at eval n=503) |
| V4 CONSISTENCY | length (B00), 2 ch → 18 one-hot | 18 → 16 | char_len .963 → **.500** (dead) | char_len MLP .969 → **.910** — residue is generic, not token-specific (§6 add-on) | implied influence **+.0385** [+.0126, +.0632] vs stratified .0349 / stacked .0269 / matched .016–.004 | same sign, ~2× | **PASS vs stratified (1.10×) & stacked (1.43×)**; matched reads 2.4–8.7× smaller |

Eraser diagnostics (every eraser): train cross-covariance with the one-hot
target = **machine zero** (3.7e−15 – 2.3e−14). Under the categorical form the
erased channel's own probes transfer CLEANLY to eval rows (char_len .500, year
.438 — vs .55x under the continuous form): decile one-hot erasure kills the
channel's linear readability off the fit rows too. Residual drift diagnostic:
max |corr| of projected reps with individual one-hot columns on eval .21–.30.
Distortion (mean edit / mean centered norm): plant (rank 1) .12, length (rank
16) .32, standard-26 (rank 52) .53.

### 3.1 Continuous-Z secondaries (OLS-form only — quotable only as this pair)

| gate | rank | key readouts (continuous form) |
|---|---|---|
| V3a | 25 | realtok lin after .9997; char_len after .555; head cost −.0762 [−.106, −.045]; ablation Δ proj +.0294, Δ-of-Δ +.0060 (run-1 JSON) — **surgical, but weak certificate** |
| V3b | 1 | task cost +.0003 [−.0040, +.0047] |
| V4 | 2 | implied influence +.0325 [+.0134, +.0545]; char_len after .556 |

## 4. V2 in detail — what erasure costs, and why (placebo decomposition)

Refit-head eval AUCs: B01 raw .8238 → plant-erased .7854; B00 unplanted
baseline .7987.

| eraser on B01 | concept ⊥ y? | in text? | cost (eval) | 95% CI |
|---|---|---|---|---|
| placebo_indep — random flag, plant's base rate | yes (AUC .503) | no | **−.0006** | [−.0028, +.0013] |
| placebo_ymatch — random flag, plant's exact P(flag\|y) rates | no (AUC .652) | no | **−.0293** | [−.0583, −.0022] |
| the actual plant | no (AUC .652) | yes | **−.0384** | [−.0568, −.0208] |

Reading: erasing a concept the reps don't carry and y doesn't touch is free
(−.0006). Erasing ANY concept with the plant's y-correlation — even one with
no textual presence at all — costs .029, because the concept's cross-covariance
direction necessarily contains a y-aligned component; LEACE removes it. The
channel-specific cost of the real plant beyond that tax is ≈ .009 (−.0384 −
(−.0293)), and the V2(d) jump obeys the identity exactly: jump_after =
jump_before − utility cost = +.0251 − .0384 = −.0133. Consequence for any
future debiased quote: **a LEACE-debiased AUC understates the true
"channel-absent" counterfactual by roughly the y-tax of the erased concepts** —
inherent to covariance-zeroing erasure, not a bug; quantifiable per concept by
the y-matched placebo, which should ship with every future application.

This also explains V2(d)'s overshoot: .8238 − .0384 = .7854 lands .013 UNDER
the .7987 unplanted baseline. The planted advantage is fully gone (CI covers
0); the error is on the over-removal side — the exact opposite failure mode of
GRL, which kept the channel at every operating point.

## 5. V3/V4 in detail

**V3a — the certificate-strength / surgicality trade-off (second headline
finding).** The two certificate forms give OPPOSITE specificity outcomes on
the same arm:

| form | rank | realtok probe after (lin / MLP) | token ablation Δ (head) | Δ-of-Δ | head cost |
|---|---:|---|---|---:|---|
| continuous (OLS-form) | 25 | .9997 / .99945 | +.0233 → +.0294 | +.0060 | −.0762 [−.106, −.045] |
| **categorical (strong form)** | **52** | **.9089 / .9251** | **+.0233 → +.0046 [−.005, +.016]** | **−.0187** | **−.1279 [−.168, −.092]** |

The weak-certificate surgery is surgical: the unnamed real-signal token keeps
its full causal role (Δ-of-Δ +.006, reliance if anything up). The
strong-certificate surgery — 52 directions cut instead of 25 — leaves the
token READABLE (probes .91–.93, far above chance) but strips most of its USE
by the linear head: the ablation delta collapses to +.0046 with a CI touching
zero. Two non-exclusive mechanisms, both reportable: (i) decile one-hot
columns capture NONLINEAR dependence on the nuisances that the token's
linear-only residualisation (R² .043) never removed, so part of the erosion is
legitimate removal of nuisance-carried token signal; (ii) 52 cross-covariance
directions estimated from 7,618 rows in 4,096-d carry sampling noise, and each
noisy direction obliquely clips a little of everything (the lit review's
"oblique projection can move task directions" caveat, now measured). Either
way the operational lesson is the same: **the strong certificate is not
surgical at tens of channels** — erasing the full standard set costs the head
−.128 and collateral-damages an unnamed real channel's usability.

**V3b.** On the year-balanced subsample (date-alone OOF AUC .435, i.e. nothing
usable), the categorical date eraser (rank 8) costs −.0010 [−.0093, +.0089] —
the "verifiably unused channel must cost < .005" gate passes on the point
estimate; the CI covers 0 but is wider than the gate window at eval n=503.
The channel is genuinely dead (year probe .719 → .438). Continuous secondary:
+.0003 [−.0040, +.0047].

**V4.** The categorical length eraser (rank 16; char_len probe fully dead at
.500 on eval) implies an influence of +.0385 [+.0126, +.0632]. The instruments
recomputed on the SAME refit head's eval predictions (apples-to-apples):
decile-stratified drop .0349 (ratio 1.10 — granularity-matched to the decile
one-hot eraser), stacked increment .0269 (ratio 1.43) — both same-sign and
within 2×. Matched sampling reads smaller (.0162 / .0079 / .0044 at calipers
.01/.02/.05; ratios 2.4–8.7) — already the low-reading instrument on this cell
in the closure campaign; sign agrees everywhere. Continuous secondary: +.0325
[+.0134, +.0545]. Called **PASS** on the two primary instruments with the
matched-sampling divergence stated.

## 6. Utility-distortion frontier (B00 real corpus, refit-head eval AUC, base .7987)

Categorical = primary (strong certificate); continuous = OLS-form secondary.

| erased set | k raw | one-hot cols → rank | eval AUC (cat) | Δ (cat) | test (cat) | dist (cat) | Δ (continuous) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mined top-1 | 1 | 3 → 2 | .7931 | −.0056 | .6868 | .24 | −.0083 |
| length | 2 | 18 → 16 | .7602 | −.0385 | .5841 | .32 | −.0325 |
| mined top-11 | 11 | 32 → 22 | .7528 | −.0459 | .6512 | .36 | −.0377 |
| mined top-22 | 22 | 61 → 41 | .7460 | −.0527 | .6331 | .40 | −.0293 |
| mined 45 (full Track-B map) | 45 | 133 → 91 | .7321 | −.0666 | .6098 | .45 | −.0354 |
| standard 26 | 26 | 58 → 52 | .6750 | −.1237 | .5361 | .49 | −.0654 |
| standard 26 + mined 45 | 71 | 191 → 143 | .6433 | −.1554 | .5114 | .56 | −.0871 |

Mined sets = the round-5 cumulative Track-B map for this cell (45 channels,
8 parents retired; `notes/2026-08-06__closure_nc_responded.md` §8.2), scored
over all 9,521 rows, nested by |train-AUC − .5| (ordering declared, used only
to define the nesting). Reading:

- **Erasure scales without a cliff, but the strong certificate has a real
  price** — roughly 2× the OLS-form cost at every point. The categorical
  frontier is monotone in the nested mined sets (−.006 → −.046 → −.053 →
  −.067); the run-1 continuous non-monotonicity was within noise.
- The full 45-channel mined map is affordable even in strong form (−.067,
  eval .732). The expensive sets are the ones containing length + topic
  (standard-26: −.124; all 71: −.155, eval .643). Compare GRL at λ=1.0
  under the bottleneck: −.138 while removing NOTHING (probe 1.000); the
  strong-form 71-channel scrub costs about that much and actually kills the
  linear channels.
- **Test-split damage runs ahead of eval damage** under the categorical form
  (length: test −.117 vs eval −.039; 71 channels: test −.190 vs eval −.155) —
  erasers fit on train transfer their cuts imperfectly; quote eval AND test.

Add-on (nonlinear residue for a continuous channel,
`results/addon_charlen_mlp.json`): char_len after the length erasure on B00 —
categorical form: linear .963 → **.500** (dead) but MLP .969 → **.910**;
continuous form: linear → .556, MLP → .921. The nonlinear residue is GENERIC —
not an artifact of a discrete planted token, and not fixed by the strong
certificate form: LEACE of either form cuts only linear readability.

## 7. Verdict

**LEACE is adopted as a LINEAR-ONLY erasure instrument with stated scope — not
as a full removal-certification instrument.** Gate-by-gate (certificate_form:
categorical): V1 PASS; V2(a) PASS (the linear guarantee holds exactly on fit
rows, .475 on eval); **V2(b) FAIL — a fresh 2-layer MLP still reads the
"erased" plant at .963** (and the erased length channel at .910 under the
strong form, §6 add-on), an Elazar & Goldberg-shaped residue reproduced with
ground truth on the closed-form projector and generic across channel types and
certificate forms, so no nonlinear-scope removal claim is ever quotable from
it; V2(c/d) jump eliminated at a placebo-decomposable cost; **V3a split:
representation-survival PASS, causal-use EROSION at rank 52** (Δ-of-Δ −.019;
the OLS-form pair was surgical at +.006 — certificate strength trades against
surgicality, §5); V3b PASS; V4 PASS on 2 of 3 instrument families (1.10× the
granularity-matched decile-stratified instrument).

Operating envelope implied by the battery: **small named sets (1–2 channels,
rank ≤ ~18) are where the strong certificate is affordable and surgical**
(mined top-1 −.006; length −.039 with V4 consistency 1.1×); at tens of
channels the strong form doubles the utility cost and collateral-damages
unnamed real channels' head-usability (V3a). Certification-grade erasure
should be per-channel or small-set; bulk scrubs of the full named map are a
measurement of joint channel load, not a usable debiased scorer.

Why linear-only is still worth having HERE: this program's dense standard
scores with a LINEAR head on h (the scalar head; the bottleneck head∘proj is
also linear in h), so "concept linearly unreadable" ⇒ "this score path cannot
use it" — a real, checkable certificate the GRL line never delivered at any
operating point. The certificate binds ONLY the frozen-reps + refit-linear-head
readout protocol: any retraining, any nonlinear consumer, and the .963 residue
is in play.

Standing rules if used: (1) always report the MLP residue next to any erasure
claim; (2) always run the y-matched placebo and report the y-tax alongside the
utility cost — debiased AUCs understate the channel-absent counterfactual by
that tax; (3) the guarantee is in-sample — quote train-exact + eval-probe
numbers, never "provably zero" off the fit rows (categorical-form target
probes transferred cleanly here, .50/.44 on eval; continuous-form ones sat at
≈.55; test-split utility damage runs ahead of eval, §6); (4) the two ADOPTED
observational instruments (stacked increment + matched sampling) remain the
instruments of record for spurious-influence ESTIMATION; LEACE adds the
intervention-style readout they cannot provide, and V4 shows the two agree
within 1.2× (stacked) on the one channel where all instruments run; (5)
**certificate_form: categorical is mandatory** — continuous-Z erasers carry
only the OLS-form guarantee and are quotable only as labelled secondary
readouts; (6) the certificate never composes past binary-linear consumers
(Ravfogel ACL 2023 Thm 3.4: a multiclass softmax can recover Z despite
guardedness) — the honest claim after a LEACE pass is "the score, being a
linear functional of the erased representation, cannot use the channel," never
"the channel is gone."

Future work (identified by the lit review as without precedent; NOT run in this
pilot): **erase-then-refinetune** — apply the categorical LEACE pass to h,
then fine-tune a fresh LoRA adapter over the same frozen backbone on the task,
and test whether the erased channel RETURNS in the new model's representations.
This is the composition question the certificate cannot answer by construction
(any retraining is a new, nonlinear consumer of the substrate) and would be the
first direct measurement of erasure durability under continued training.

## 8. Artifacts

- Code: `methods/taste_decomposition/debias/leace/{leace.py, run_battery_leace.py,
  build_mined_channels.py, extract_reps_leace.py, addon_charlen_mlp.py}`
- Results: `methods/taste_decomposition/debias/leace/results/battery_leace.json`
  (+ `addon_charlen_mlp.json`); mined map
  `leace/build/mined_channels_nc45.npz` (45 cols, 9,521 rows, 259 NaN cells
  train-median imputed)
- Reps (sk3): `datasets/notice-and-comment/debias_pilot/runs/{B00,B01,R00,R06,R08}*/reps.npz`
  + NEW `runs/R06_vanilla_realtok/reps_stripped.npz` (the one GPU pass;
  extraction log `debias_pilot/logs/extract_leace.log`)
- GPU ledger: GPU 7 CLAIM → RELEASE rc=0 (2026-08-08, 335 s job; entry
  `gpu_ledger.txt` agent=claude-leace-pilot-fable). No GPU held at close.
- LEACE verification: synthetic self-check PASS (`python leace.py`);
  machine-equal (max |Δ| 1.4e−13) to `concept-erasure`'s `LeaceFitter` with
  `shrinkage=False`; `onehot_bins` unit check — median-split probe exactly
  .5000 train and holdout after decile-one-hot erasure.
- Certificate-form review: `notes/2026-08-10__litreview_spurious_debiasing.md`
  §R1a (Belrose Thm 3.1 vs §4.3; Ravfogel ACL 2023 Thm 3.4); run-1
  continuous-Z battery preserved as `results/battery_leace_r1_continuousZ.json`.
