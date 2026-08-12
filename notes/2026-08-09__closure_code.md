# Layer-3 articulation closure — CODE cell (GitHub pull-request MERGE, v3 enriched text)

Date: 2026-08-07 (opened; dated file name 2026-08-09 per the campaign brief).
Status: **CONFIRMATORY, run under the FROZEN protocol**, with one recorded cell-specific
adaptation (§2) declared *before* any mining slice was built.
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` — FREEZE DECLARATION 2026-08-06 plus
FREEZE ADDENDA 1 (B-side missing mass + stacked increment), 2 (Track-B upstream-factor
mode + MIXED rule), 3 (MIXED decomposition pass), 4 (position-in-container prior), and
design §11 (fused-must-beat-bank).
Cell instrument note: `notes/2026-08-06__code-abank-enriched-rescore.md` — **all of its
caveats are binding here**.
Worked precedents: `notes/2026-08-06__closure_nc_responded.md`,
`notes/2026-08-06__closure_cw_community.md`.
Code + artifacts: `methods/taste_decomposition/closure/code_v3/`.

---

## Terminology, spelled out on first mention (standing rule)

**PR** = pull request, a proposed change submitted to a GitHub repository.
**Merge outcome / y** = 1 if the PR was merged, 0 if it was closed without merging. This
is the cell's label and no proposer ever sees it.
**v3 enriched text** = the modelled document for one PR: `Title` + `Description` (the PR
body) + up to 20 inline `Review comments` + the unified `Diff`, capped at 24,000
characters (the diff is what gets truncated).
**v2** = the retired diff-only text.
**V** = programmatic non-LLM features, in two blocks here: **V_exec** (17 features derived
from the docker test-execution fleet — verdict category, return code, pass-to-fail and
fail-to-pass gates) and **V_text** (19 deterministic geometry features recomputed on the
v3 text — section lengths, comment-bullet count, file/hunk/line counts, …). **V_all** =
both blocks.
**A** = the articulated-criterion bank: 83 portable code-review criteria scored by
**Gemma-4-31B-it** on the v3 text, one token per (row, criterion), plus one `applied`
indicator per criterion recording whether the judge answered or abstained.
**VA** = V and A concatenated. **lin / nl** = the frozen Layer-1 linear (standardise +
logistic regression, C = 1) and nonlinear (HistGradientBoosting, `max_leaf_nodes` ∈
{15, 31}, lr .06, 400 iterations, early stopping, inner GroupKFold(3), seeds 0/1/2)
aggregations of the same score matrix.
**T** = the dense readout — Llama-3.1-8B with LoRA (rank 16, α 32), 2,048-token window,
trained on the v3 text.
**Δ_beyond** = T − VA_nl, the unarticulated residual. **Δ_r** = that residual at mining
round r. **ε** = .005, the per-round saturation threshold. **AUC** = area under the ROC
curve. **OOF** = out-of-fold. **GKF** = GroupKFold. **Track A** = the quality-criterion
proposer; **Track B** = the suspected-spurious-channel proposer. **MIXED** = a Track-B
channel whose conjectured upstream cause plausibly also causes real quality.
**Good-Turing missing mass** = estimated probability that the next independent proposal
names a species not yet seen. **P** = number of proposers in a sealed fleet round.
**k_A / k_B** = criteria scored per round on Track A / Track B.

---

## 1. What this cell is, and the four caveats that travel with every number

The cell gated into the confirmatory roster on the enriched-rescore result: **within
repository, dense beats the best articulated stack by +.0576 (eval) and +.0390 (test),
Wilcoxon p = .006 / .040** — the two splits agreeing at the honest level where they
flatly contradicted each other pooled (pooled eval said VA_nl .742 > dense .649; pooled
test said dense .737 > VA_nl .581).

Four caveats from the instrument note are binding on this campaign and are repeated at
every quoted number:

1. **Within-repo readouts ONLY. Never pooled.** Pooled AUC on this corpus is
   composition-dominated — the GBM exploits between-repository structure that inflates
   pooled AUC and does not survive repo-centering. No pooled Δ is ever quoted as the
   residual, in either direction.
2. **The .592 baseline is RETIRED.** The published `V .576 / V+A .592` row came from a
   deterministic *coded* bank (no language model) scored on bare diffs, and spliced a
   pooled V onto a grouped V+A; the ladder no longer reproduces from surviving artifacts.
   Nothing in this campaign is ever differenced against .592.
3. **Both input and instrument changed** versus the historical row (bare diff → enriched
   PR object; coded programs → Gemma-4-31B judge). Every A number here is a
   new-instrument number.
4. **Three asymmetries cut toward understating the residual** and one cuts the other
   way. Understating: the judge saw up to 11.6K tokens while dense saw 2,048; A got the
   13 enrichment-sensitive criteria that v2 could not even ask about, and those are
   exactly the ones carrying signal; V_all is the most generous V available. Cutting the
   other way: **dense v3 was single-seed (42)**, worth roughly ±.02 on this corpus, which
   made the +.039 test margin about two seed-sigmas — this is what round 0 item 1 exists
   to settle — and **V_exec is below chance out-of-repo on test (.4578)**, i.e. the
   execution layer anti-transfers across repositories and contributes noise, not floor.

---

## 2. WITHIN-REPO PROTOCOL ADAPTATION — recorded BEFORE any mining slice was built

The frozen protocol is written for cells whose readout is a pooled AUC on a MONITOR
split. This cell's readout cannot be pooled (caveat 1). The adaptation below changes
*where the number is read*, never *what decides a round*, and is fixed here in advance.

### 2.1 Population, and why this cell needs no dense-honesty intersection

Population = the A/V-scored evaluation population: **11,452 PRs across 255 repositories**
(eval 5,822 / 112 repos; test 5,630 / 143 repos), pos-rate .8038.

The dense v3 chain trained on the 47,659 TRAIN rows, whose repositories are disjoint from
eval and test by construction (`build_manifest.json` integrity block: train∩eval =
train∩test = eval∩test = 0 repositories), and the A bank was scored on eval+test only.
**Every row of the closure population is therefore already dense-held-out, and T is honest
on all of it.** The N&C DECISION-1 problem — MONITOR ⊂ dense-held-out shrinking the
monitor to n = 377 — does not arise: MONITOR here is automatically both T-honest and
VA-honest.

Recorded flag: **eval is the dense chain's checkpoint-SELECTION split**
(`--selection_split eval`); test was never read during training. Both halves are reported
as replicates and **test is the selection-free reading**.

### 2.2 Splits — by REPOSITORY, the cell's only legitimate group key

FIT+MINE = repositories with `sha256(repo)/2²⁵⁶ < .80`; MONITOR = the rest. Repository is
also the dense chain's split key and the fold key of every stack in the cell, so
FIT+MINE and MONITOR are repo-disjoint by construction (asserted in
`build_splits_code.py`). MONITOR is never read by any proposer.

| split | rows | repos | eval rows | test rows | pos-rate | repos scorable (n ≥ 20, both classes) | scorable rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIT+MINE | 8,697 | 201 | 4,285 | 4,412 | .8155 | 115 | 7,795 |
| **MONITOR** | **2,755** | **54** | 1,537 | 1,218 | .7670 | **29** | **2,489** |

Mining slice M = the FIT+MINE rows (all dense-honest). The disagreement slice each round
is the top |dense percentile − VA_nl percentile| inside M, rows already read in earlier
rounds excluded.

### 2.3 Δ_r is the n-weighted WITHIN-REPO readout — the definition, fixed

For a row set R and two score vectors p_a, p_b: for every repository with **n ≥ 20 rows
in R and both classes present**, compute AUC(y, p) inside that repository; then

  Δ = Σ_r n_r · AUC(p_a)_r / Σ_r n_r  −  Σ_r n_r · AUC(p_b)_r / Σ_r n_r.

The `n ≥ 20` + both-classes rule is inherited **verbatim** from the instrument that
produced the gate numbers (`abank_rescore/within_repo.py`), so the gate table and the
campaign are on one ruler. Uncertainty on Δ is a **leave-one-repository-out group
jackknife** on the n-weighted statistic, plus the paired **Wilcoxon signed-rank** over the
per-repository Δ vector (the same test that produced p = .006 / .040).

### 2.4 Readout tiers, declared in advance

Following the enlargement principle applied on N&C and CW (prefer the better-powered
honest readout over a thin one):

- **saturation statistic** = the within-repo VA_nl gain on **MONITOR** (29 scorable
  repositories, 2,489 rows) — VA-honest, and the split no proposer ever sees. This is
  what the ε = .005 stopping rule is applied to.
- **Δ_r headline level** = the within-repo Δ on the **full honest population** (VA fit
  grouped-OOF inside FIT+MINE, refit-and-predicted on MONITOR): better powered, mildly
  mining-contaminated and therefore conservative — exactly the pilot's arrangement.
- **replicates** = within-repo Δ on eval-only and on test-only, reported every round;
  test-only is the selection-free reading.

### 2.5 Levels are protocol-specific

Per the pilot amendment, the closure-protocol Δ (VA fit on FIT+MINE only) is **not**
comparable to the Layer-1 gate Δ (+.0576 / +.0390, VA fit grouped-OOF within each split).
Only round-over-round changes plus the stated honest levels are quotable.

---

## 3. ROUND 0, item 2 — concept census of the incoming bank

Freeze requirement: "concept census of the incoming bank at round 0", identity by
full-recall blind pairwise adjudication, **never** by an embedding threshold. Cheapest
decisive test first; the embedding is used only to decide what gets *read*, and only
within one register (every text is a code-review aspect rubric from one catalog, so the
cross-register prohibition does not bite).

| level | instrument | count |
|---|---|---:|
| L0 | rubrics delivered (the portable 83) | **83** |
| L1 | distinct normalised names (exact) | **83** |
| L2 | score columns surviving the frozen degeneracy screen, fit on FIT+MINE only | **79** |
| L2′ | `applied`-indicator columns surviving the same screen | 78 |
| L3 | value clusters after collapsing \|Pearson r\| ≥ .98 | **79** |
| L4 | in-register embedding shortlist (bge-large, τ = .79) | 13 pairs |
| L5 | **effective concepts** after blind pairwise adjudication (strict: both judges SAME) | **80** |
| L5′ | same, loose rule (either judge SAME) | 78 |

Instrument health: two sealed blind judges, raw agreement **.846**, both **4/4** on an
authored anchor battery (2 authored paraphrase pairs → SAME, 2 authored near-miss pairs →
DIFFERENT; both judges got all four right, so the instrument has a demonstrated positive
*and* negative control). The two disagreements were both inside topical families —
testing-pyramid phrasing and import-ordering phrasing — and both resolved DIFFERENT under
the strict rule.

The four L2 casualties are corpus facts, not decoding failures: **a118** Rust crate
organisation, **a188** ACSL specification constructs, **a236** JS/TS parameter
conventions, **a273** Python structural pattern matching — each ≥ 99.3% NA on FIT+MINE
because the language or construct essentially never appears.

**This bank is the least redundant in the program.** The peer bank collapsed 154 → 54
effective concepts (−65%); the N&C bank 198 → 157 (−21%); this one **83 → 80 (−3.6%)**,
with **zero** value-level near-duplicates (max column \|r\| = **.895**, and **0.0%** of
column pairs reach .90). It is 80 nearly-orthogonal measurements. That is what you would
expect from a bank whose criteria were *mined from a catalog of 394 distinct code-review
aspects* rather than authored free-hand, and it means the aggregation cannot be dismissed
as one concept counted eighty times.

**The one substantive merge is the one that matters most.** The three strict merges are

| merged pair | |
|---|---|
| a89 "Behaviour-focused and interaction-oriented testing" | = a127 "Behavior-focused/user-centric testing" |
| **a78 "Change/commit/PR communication quality"** | **= a105 "Change description clarity, completeness, and rationale"** |
| a15 "Comments: clear, intentional, and why-focused" | = a50 "Commenting strategy and quality" |

and the middle one is the cell's headline mechanism. The enriched rescore reported a78
and a105 as the **#1 and #2** most predictive articulated criteria on eval (.550, .549).
The census says they are **one concept measured twice**. So the finding "the articulable
part of merge preference is substantially about PR *communication*" is carried by a
single concept occupying two bank slots — which is exactly why that concept is the first
thing the Addendum-3 decomposition pass takes apart (§6).

Artifacts: `closure/code_v3/census_code.json` (all levels + the concept map),
`census_blind_packet.json` (what the judges saw), `census_verdicts_judge{A,B}.json`,
`census_code.py`.

---

## 4. ROUND 0, item 4 — the position line (FREEZE ADDENDUM 4)

**Channel: within-repository PR-number percentile.** The container is the repository; the
PR number is monotone in submission time, so a PR's within-repo PR-number percentile *is*
its position in that repository's own timeline. Status: **observed covariate only** — it
never enters V, A, the bank, or any closure fit. This is the channel Addendum 4 was
written for (no proposer in the program has ever named an ordinal channel unprompted, yet
two of the program's strongest spurious findings are exactly that family), and on this
cell it was already measured once at build time, so round 0 re-reads it on the exact
closure population under the within-repo protocol.

| readout | all (144 repos, 10,284 rows) | eval (71 / 5,384) | test (73 / 4,900) |
|---|---:|---:|---:|
| pooled AUC | .4928 | **.4961** | **.4893** |
| **within-repo, n-weighted** | **.4993** | .4845 | .5155 |
| within-repo median | .5138 | .4832 | .5364 |
| repo-level AUC SD | .1745 | .1711 | .1759 |
| **repos with \|AUC − .5\| > .15** | **34.7%** | 32.4% | 37.0% |
| repos with AUC > .65 / < .35 | 19.4% / 13.9% | 15.5% / 15.5% | 23.3% / 12.3% |

The pooled figures reproduce the build note exactly (eval .4961, test .4893), which is the
join check on this readout.

**Three things, and the third is the one that matters for the campaign.**

1. **Recency is worth nothing on aggregate and a great deal locally.** The n-weighted
   within-repo AUC is .4993 — dead chance — while a third of repositories sit more than
   .15 away from chance, in *both* directions (19% above .65, 14% below .35). The channel
   is real, strong and repo-local, and it cancels on aggregation. The paper sentence
   remains "queue position predicts merge outcome within particular repositories, at
   \|AUC−.5\| > .15 in a third of them, but carries no pooled signal" — never "recency
   predicts merge at .81", which was one repository's bump subset.

2. **Both instruments partially track it, and the dense model tracks it harder.** Mean
   within-repo Spearman with position: **dense +.148**, VA_nl +.093 (median dense +.162;
   48% of repos have \|ρ_dense\| > .2). And the tracking is *targeted*: across repos, the
   correlation between how predictive position actually is in a repo and how strongly the
   instrument aligns with position there is **+.259 for dense** and **+.222 for VA_nl**.
   Both instruments have partly learned the repo-local recency channel from text; dense
   has learned more of it. That is a real, named, non-quality channel inside T — exactly
   the kind of thing Track B exists to find, and it was found before any proposer ran.

3. **Discounting it does not touch the residual.** Stratifying *within each repository* by
   position quartile and re-reading both instruments on the strata (n-weighted, 7,718
   rows):

   | readout | T | VA_nl | Δ |
   |---|---:|---:|---:|
   | within-repo, unstratified (all 144 repos) | .6829 | .6342 | **+.0487** |
   | within-repo **× position-quartile stratified** | .6768 | .6259 | **+.0509** |

   The residual is unchanged — very slightly *wider* after the discount, the same pattern
   the N&C and CW discounts produced. So the strongest a-priori nuisance channel on this
   cell, the one the freeze specifically told us to look for, does not explain the code
   residual.

Artifacts: `closure/code_v3/position_line.json`, `position_line_ext.json`; also inside
`round0_results.json` → `position`.

---

## 5. ROUND 0, item 1 — the seed chain, and what the gate quantity looks like before it lands

### 5.1 The chain

Dense v3 seeds **1** and **2** are training **chained on one ledger GPU** (sk3 GPU 6,
claimed and released in `gpu_ledger.txt` as `agent=claude-closure-code-v3`,
`job=dense_seeds_1_2`). The recipe is byte-identical to the seed-42 run: the trainer
`methods/dense/train_reward_model.py` was verified unchanged (md5
`845495c9…`, mtime 2026-07-12, i.e. untouched since seed 42 ran), and the only arguments
that differ from `run_pr_dense_v3.sh` are `--seed` and `--output_dir` — Llama-3.1-8B, LoRA
r16/α32, lr 5e-5, batch 16, grad-accum 1, `max_length` 2048, 2 epochs,
`--gradient-checkpointing`, `--class_weight_auto`, `--selection_split eval`.

Launch verified on all three server-diligence conditions: PID 3225002 running the exact
command line; `nvidia-smi` shows it resident on GPU 6 (uuid `GPU-49ced701…`) at 41,258 MiB,
100% utilisation, sole tenant; and the log emits real losses from step 1 (0.6914 → 0.4183
by step 301) at 12.8 optimizer steps/min, matching seed 42's rate. Each seed is scored by
`abank_rescore/score_one_seed_v3.py`, which **merges** into `eval_pass_results.json` rather
than rewriting it, so seed 42's stored numbers and per-row predictions are never touched.
Chain ETA ≈ 9 h/seed ⇒ seed 1 ≈ 05:30, seed 2 ≈ 15:00 on 2026-08-08.

Runner: `dense_standard_v3/run_code_v3_seeds12.sh`, log `runner_v3_seeds12.log`.

**Two registry trainer/scorer landmines checked against this chain (both clear).**
(i) `score_eval_dense_v4.py` hardcodes `MAXLEN = 1024`, which silently truncates a
code_v3 model trained at 2,048. The chain does **not** call it — it calls
`abank_rescore/score_one_seed_v3.py`, verified on disk at `MAXLEN = 2048`, i.e. matched to
training. (ii) `train_reward_model.py:327` asserts 80/10/10 splits ±.02; code_v3 is
.8063 / .0985 / .0952, inside the band on all three, and seed 1 is training, so the assert
passed.

**Progress at 02:25 on 2026-08-08 — the chain is BEHIND the ETA I published.** Seed 1 is
at epoch 2, step 4,179 / 5,958 (70%), running at ~11.0 steps/min (I had assumed 12.8).
Its validation trajectory is .6531 (step 2,384) → .6660 (2,979) → **.6878 (3,575)** →
.6656 (4,171), i.e. checkpoint selection is currently sitting on a pooled-eval AUC of
**.688 against seed 42's final .6488**. Two things follow. First, revised ETAs: seed 1
preds ≈ **05:40**, seed 2 preds ≈ **15:00** — the 3-seed gate cannot close overnight, only
the 2-seed one can. Second, the direction looked favourable — a *higher* T makes Δ = T − VA_nl *larger*.
**[CORRECTED at 09:50 — see §9.2.]** I inferred from that pooled curve that the
single-seed caveat was worth more than the ±.02 the instrument note assumed (a ~.04
pooled gap). It is not: when seed 1 landed, its *within-repo* T on eval came in at .7095
against seed 42's .7093 — a spread of **.0002** against a pooled gap of +.031. Pooled AUC
is untrustworthy across seeds on this cell for the same compositional reason it is
untrustworthy across splits, and it should not have been used even as a directional
signal.

**Arithmetic of the gate, for planning.** VA_nl within-repo (combined, Layer-1 protocol)
is fixed at .6342 and seed 42's T_within is .6829. For the 3-seed residual to fall to
≤ .02 the 3-seed mean T_within must be ≤ .6542, which requires seeds 1 and 2 to average
**T_within ≤ .6399 — about .043 below seed 42**. Seed 1 is currently selecting a *better*
checkpoint than seed 42, so that is not the way to bet; but it is measured, not assumed,
and the gate stays open until both seeds land.

### 5.1a MANDATORY OOF ALIGNMENT GATE — PASSES EXACTLY

Registry landmine (2026-08-10): `*_va_nl_oof_*.npy` arrays are keyed in **bank
`item_ids` order, not population/join order**, and a misaligned join reads AUC ≈ .50.
Five cells were found misaligned. The gate is: `AUC(y, oof_seed0 in the assembled row
order)` must equal the published `nonlinear.VA.auc_seeds[0]` to < 1e-9 — exact identity,
seed 0 only, never mean-of-3.

| split | assembled-order pooled AUC, seed 0 | published `auc_seeds[0]` | abs diff | shuffled counterfactual |
|---|---:|---:|---:|---:|
| eval | .7420343966892524 | .7420343966892524 | **0.0** | .4930 |
| test | .5743519690037469 | .5743519690037469 | **0.0** | .5033 |

**GATE_PASS = true**, and it passes on the non-binding seeds 1 and 2 as well (all six
seed × split cells at abs diff 0.0). The deliberately shuffled counterfactual lands at
.493 / .503 — the exact landmine signature — so the gate has real power on this cell and
is not passing vacuously.

Why the alignment is right here: `readout_code_v3.py` builds its row frame as
`concat(eval.csv, test.csv)`, `build_v` is a left merge (order-preserving), the A matrix
is reordered *to* that frame, and each `oof` vector is saved in `d[d.split==sp]` order —
which is split-CSV order. `V_matrix_v3.parquet` is built by the identical concat, so
`cells_code.load()` reproduces that order exactly. The final run had
`PARTIAL_DRY_RUN = false`, so no partial-row re-indexing ever occurred.

**Everything downstream is therefore cleared**: the gate table (§5.2), the design-§11
fused readout (§5.3), the length stratification (§6) and the closure baseline (§8) all
read these vectors. Note the gate's own numbers are *pooled* AUCs; they are used only as
an alignment identity and are never quoted as results on this cell.
Artifact: `closure/code_v3/oof_alignment_gate.json`.

### 5.2 The gate quantity at seed 42, with its uncertainty properly characterised

This is the ruler the +.0576 / +.0390 gate numbers were measured on (Layer-1 protocol,
VA_nl = mean of the three stack-seed grouped-OOF vectors fit **within** each split). My
estimator reproduces the published table exactly, which is the join check:

| | eval | test | **both splits pooled at the repo level** |
|---|---:|---:|---:|
| repos scored (rows) | 71 (5,384) | 73 (4,900) | **144 (10,284)** |
| dense (seed 42), within-repo n-wtd | .7093 | .6540 | .6829 |
| VA_nl, within-repo n-wtd | .6517 | .6150 | .6342 |
| **Δ, n-weighted** | **+.0576** | **+.0390** | **+.0487** |
| Δ, unweighted (equal repos) | +.0610 | +.0499 | +.0553 |
| Δ, median repo | +.0533 | +.0597 | +.0564 |
| dense wins in | 44 / 71 | 43 / 73 | 87 / 144 |
| Wilcoxon p (paired, per repo) | .006 | .040 | **.00071** |
| repo-cluster bootstrap CI on the n-wtd Δ | [+.002, +.112] | [−.005, +.083] | **[+.014, +.085]** |
| bootstrap P(Δ > 0) | .980 | .958 | **.996** |
| leave-one-repo-out jackknife SE | .0298 | .0235 | .0188 |

Two honest points that the published row did not carry.

1. **The n-weighted magnitude is much less certain than the p-values suggest.** On the
   test half alone the repo-cluster bootstrap CI on the n-weighted Δ is [−.005, +.083] —
   it includes zero — while the paired Wilcoxon over the same 73 repos gives p = .040.
   The two are not in conflict: the Wilcoxon weights repositories equally and tests sign
   consistency, whereas the n-weighted mean is dominated by a handful of large
   repositories. The *equal-repo* Δ (+.050 on test, CI [+.001, +.100]) is the one that
   matches the Wilcoxon. Combining the two dense splits at the repository level — which
   is legitimate here precisely because the readout is repo-centred and the two splits
   contribute disjoint repositories — gives **Δ = +.0487, CI [+.014, +.085], P(>0) = .996,
   Wilcoxon p = .0007 over 144 repositories**, and that is the best-powered honest
   statement of the cell's residual at seed 42.
2. **Combining the splits is not the same as pooling.** The forbidden operation on this
   cell is pooling *rows* across repositories into one AUC; adding repositories to a
   within-repo average is the opposite of that.

### 5.3 Design §11 — the fused arm beats the bank on this cell (no audit trigger)

Standing rule (design §11): a final ledger where max(fused arms) ≤ VA_nl auto-triggers an
audit. Fused arm here = a grouped-OOF logistic stack of the two already-out-of-sample
score columns (VA_nl OOF, dense probability), fit by GroupKFold(repository) and read
within repo.

| within-repo, n-weighted | eval | test | both |
|---|---:|---:|---:|
| bank VA_nl | .6517 | .6150 | .6342 |
| dense T (seed 42) | .7093 | .6540 | .6829 |
| **fused (bank + dense)** | **.7187** | **.6620** | **.6920** |
| fused − bank | **+.0670** | **+.0470** | **+.0578** |
| Wilcoxon p / jackknife CI | 1.4e-7 / [+.043, +.091] | .0071 / [+.006, +.088] | 1.7e-8 / [+.033, +.082] |
| fused − dense | **+.0094** | **+.0080** | **+.0091** |
| audit trigger (fused ≤ bank)? | **no** | **no** | **no** |

**Code is a cell where fusion exceeds BOTH parents**, not merely max(parents). That is the
opposite of the cap_crowd / Style-Invitational pattern, where fusion reached the better
parent and stopped, and it is independent corroboration that the bank and the dense model
carry partly non-overlapping signal here — which is what a real (if small) residual means.
Artifact: `closure/code_v3/fused_check.json`.

---

## 6. ROUND 0, item 3 (continued) — what the stack actually leans on, and the length test

The Addendum-3 parent screen (SHAP, refit on FIT+MINE only) was run to choose which bank
criteria get decomposed, and its by-product is a blunt description of the articulated
stack. Top features by mean |SHAP|:

| rank | feature | mean \|SHAP\| |
|---|---|---:|
| 1 | **`vt_len_title` (title length)** | **.2977** |
| 2 | a37 Refactoring quality and practice | .1315 |
| 3 | a123 Contribution readiness and submission norms | .1213 |
| 4 | a1 Simplicity (KISS/YAGNI) and complexity management | .1189 |
| 5 | `vt_add_del_ratio` | .1000 |
| 6 | `ve_rc` (execution return code) | .0765 |

The single strongest feature in the V+A model is a pure geometry feature — **title length,
carrying more than twice the weight of the best criterion**. On the Style Invitational cell
that pattern turned out to mean "the bank is a length model" and dissolved the cell's
apparent bank-beats-dense result, so it has to be tested here rather than noted.

**The length test — and this bank passes it.** Within-repo × feature-quartile
stratification (same estimator as the position line):

| stratifying feature | alone AUC (within repo) | T_adj | VA_nl_adj | **Δ_adj** |
|---|---:|---:|---:|---:|
| *(unstratified within repo)* | — | .6829 | .6342 | **+.0487** |
| `vt_len_title` | .5056 | .6579 | .6223 | +.0356 |
| `vt_len_desc` | .4795 | .6592 | .6232 | +.0360 |
| `vt_len_diff` | .5075 | .6603 | .6177 | +.0426 |
| `vt_n_files` | .5200 | .6687 | .6178 | +.0509 |
| `vt_len_comments` | .5184 | .6701 | .6164 | +.0537 |
| `vt_len_total` | .4813 | .6700 | .6171 | +.0529 |

Three readings.

1. **No geometry feature predicts merge on its own within repo** (.480–.520). Title length
   is load-bearing *inside the model* (it is where the trees split) without being
   predictive *by itself* — it is acting as a conditioning variable, not a channel.
2. **The bank does not collapse under length stratification.** VA_nl falls from .6342 to
   .616–.623, a loss of .012–.017. Style Invitational's bank fell to .5409 and 0 of 32
   rubrics survived. **The code bank is not a length model**, and the SI verdict does not
   transfer to this cell.
3. **The residual survives every geometry stratification**, at +.036 to +.054 against an
   unstratified +.0487. The harshest cut (title or description length) costs about a
   quarter of it; the rest cost nothing or slightly widen it.

Artifacts: `closure/code_v3/length_stratification.json`, `parents_code.json`.

## 7. Decomposition-first pass — parents selected, prompt staged, not fired

Per the brief ("decomposition-first pass on MIXED channels first") and FREEZE ADDENDUM 3,
the parents are chosen before round 1 from two recorded sources: the SHAP
surface-interaction screen on FIT+MINE, and the parent the brief names outright.

| parent | source | surface-interaction mass | alone AUC (FIT+MINE) | interacts with |
|---|---|---:|---:|---|
| a123 Contribution readiness and submission norms | SHAP | .1136 | .554 | comment mean length, comment length, title length |
| a1 Simplicity (KISS/YAGNI) | SHAP | .1065 | .542 | comment length, `ve_rc`, comment mean length |
| a37 Refactoring quality and practice | SHAP | .0824 | .574 | add/del ratio, comment mean length, title length |
| a175 Documentation formatting and style conventions | SHAP | .0377 | .535 | hunks, comment mean length, add/del ratio |
| **a78 Change/commit/PR communication quality** | **brief** | — | — | description length, description bullets, issue reference |

**Census dedup, recorded:** a105 "Change description clarity, completeness and rationale"
was dropped as a *separate* parent because the round-0 census adjudicated it identical to
a78 (both blind judges SAME); decomposing both would spend two of the round's k slots on
one concept. **Both** bank criteria are retired from readouts once a78's components are
scored, per Addendum 3.

5 parents → 10 components (candidate-real + surface each) + 5 Addendum-4
position-fingerprint channels = **15 criteria** for the decomposition round. The sealed
decomposer prompt is built (6,908 chars, label-blind: parent name, parent rubric text,
its surface partners, corpus, construct — no AUCs, no labels) and staged at
`scratchpad/code_v3/code_v3_rd/prompt_decomposer.txt`. **It has not been fired: the gate
in §5 decides whether any round runs.**

---

## 8. Closure-protocol round-0 baseline — COMPLETE

V / A / VA refit under the **closure** protocol: the degeneracy screen, the imputation
medians, the scaler and the HistGB grid selection all see FIT+MINE only; predictions on
MONITOR come from a refit; the honest-full vector is grouped-OOF inside FIT+MINE plus the
MONITOR refit. VA_nl = mean over HistGB seeds 0/1/2. 193 features survive the screen.

Two recorded implementation facts. (i) The first attempt died on a shape bug after the V
block — the per-seed MONITOR vectors were passed to a full-length within-repo reader —
costing the run but no result; the fix (`_expand`) and per-block progress logging are in
the current file. (ii) The rerun is **cheaper by design**: only VA needs the honest-full
vector, because Δ is read off VA, so V and A are fit MONITOR-only (`want_oof=False`).
That removes about two thirds of the HistGB work and changes no quoted number.

### 8.1 The round-0 residual, all four tiers

| tier | repos (rows) | T within | VA_nl within | **Δ** | dense wins | Wilcoxon p | repo-jackknife CI |
|---|---:|---:|---:|---:|---:|---:|---|
| **MONITOR** (decision split) | 29 (2,489) | .6624 | .6152 | **+.0472** | 18/29 | .149 | [−.078, +.173] |
| **honest-full** | 144 (10,284) | .6829 | .6307 | **+.0522** | 87/144 | **.0011** | **[+.015, +.089]** |
| eval | 71 (5,384) | .7093 | .6506 | +.0586 | 44/71 | .018 | [+.002, +.115] |
| test | 73 (4,900) | .6540 | .6088 | +.0452 | 43/73 | .022 | [−.004, +.094] |

**The headline finding of this block is that the closure protocol does NOT move the
level on this cell.** The pilot amendment warns that closure-protocol Δ is
protocol-specific and not comparable to the Layer-1 Δ, and on N&C the difference was
brutal (+.092 matched → +.036 under closure). Here:

| | Layer-1 protocol (VA fit within split) | closure protocol (VA fit on FIT+MINE only) |
|---|---:|---:|
| eval Δ | +.0576 | **+.0586** |
| test Δ | +.0390 | **+.0452** |
| combined / honest-full Δ | +.0487 | **+.0522** |

The two agree to within .006, and the closure figure is if anything slightly *larger* —
refitting the bank on repo-disjoint FIT+MINE data costs VA_nl about .001 on eval and .006
on test while leaving T untouched. **So the r = 0 anchor of the closure curve sits on
essentially the same scale as the gate number, and round-over-round movement can be read
against a residual of ~ +.05 rather than against a differently-scaled quantity.** The
formal caveat still stands (levels are protocol-specific; only round-over-round changes
and the stated honest levels are quotable), but on this cell it is close to a no-op, and
that is worth knowing before the curve starts.

### 8.2 MONITOR is too thin to decide anything on its own — a protocol warning

| block, closure protocol, MONITOR | VA_lin within | VA_nl within | per-seed | **seed spread** |
|---|---:|---:|---|---:|
| V (36 programmatic) | .5323 | .5687 | .578 / .550 / .567 | **.0281** |
| A (83 criteria + 83 applied) | .5758 | .6116 | .594 / .609 / .619 | **.0251** |
| **VA** (193 features) | .5963 | **.6152** | .614 / .627 / .600 | **.0274** |

**The MONITOR seed spread is ~.027, five times ε = .005**, and the MONITOR Δ itself
(+.0472) carries a repo-jackknife CI of [−.078, +.173] with Wilcoxon p = .149 over 29
repositories. The frozen saturation statistic on this cell is therefore far noisier than
the threshold it is compared against: one HistGB seed can move it by more than five
ε-units. This is the same structural problem N&C hit (its thin T-honest MONITOR had a
.0365 seed spread versus .0074 on the enlarged set), and it is recorded **before any round
runs**, with the mitigation that campaign adopted:

* the saturation statistic stays on MONITOR per the freeze, but **every round-over-round
  gain is reported with its seed band and a repo-cluster bootstrap CI**, and no gain is
  called sub-ε on a point estimate alone;
* the **honest-full tier is the better-powered companion** (144 scorable repos vs 29) and
  its gain is reported beside the MONITOR one every round;
* a saturation call the two tiers disagree about is reported as a disagreement, never
  resolved silently.

Two further descriptive facts. **A dominates V** (.6116 vs .5687 on MONITOR) — the
articulated criteria, not the programmatic features, are what the scorecard is made of
here, the opposite of N&C where V dominated A and the bank added ~.015. And
**Δ_interact is small**: VA_lin .6092 → VA_nl .6307 on the honest population, i.e.
**+.0215**, against N&C's +.092. The nonlinear aggregation is doing comparatively little
work on this cell, which is another way of saying the bank's signal is in its criteria
rather than in the interactions the GBM finds among them.

### 8.3 Swap baseline — the bank has almost no independent signal where dense is wrong

Pair algebra restricted to (merged, closed-unmerged) pairs **inside a single repository**.
w₊ = fraction of within-repo pairs the dense model orders correctly.

| tier | within-repo pairs | w₊ | **C₊** = P(bank right \| dense right) | **C₋** = P(bank right \| dense wrong) |
|---|---:|---:|---:|---:|
| **honest-full** | 161,761 | .656 | **.6687** | **.5099** |
| eval | 102,954 | .671 | .6551 | .5192 |
| test | 58,807 | .630 | .6940 | .4955 |
| MONITOR | 38,943 | .644 | .5091 | .5830 |

**C₋ = .5099 is the load-bearing baseline number, and it is much worse than N&C's .587.**
On the ~34% of within-repo pairs the dense model orders backwards, the articulated bank is
right essentially at chance. So the code bank is *not* holding independent signal in the
region where dense fails — it is simply uninformative there. Two consequences for the
rounds:

1. The swap diagnostic has almost no headroom below it. Any round that raises C₊ while C₋
   moves *below* .5 is the swap signature (the bank taking on dense's errors), and on the
   test half C₋ is already at .4955, so that boundary is one round away.
2. It sharpens what the +.05 residual *is*: dense is not merely a better-calibrated version
   of the bank on the same pairs; the two instruments disagree on a third of within-repo
   pairs and the bank contributes nothing on that third. That is consistent with §5.3's
   result that fusion beats both parents.

The MONITOR row inverts (C₊ .509 < C₋ .583). With 29 repositories, a VA_nl of .615 and a
seed spread of .027 that is most plausibly noise, but it is reported rather than dropped
and it is one more reason not to read MONITOR alone.

Artifacts: `closure/code_v3/round0_results.json` (`closure_round0`, `swap_baseline`,
`position`), `round0_state.npz` (the r = 0 prediction vectors), `round0.log`.

## 9. Gate verdict — PENDING, with the decision rule fixed

| | |
|---|---|
| **STATUS 2026-08-09** | **CLOSED — GATE HOLDS.** Δ = +.0519 [+.015, +.089], p = .00022 (see §9.3). Rounds authorised, not started (GPU lanes assigned elsewhere; slotted lane C). |
| **Gate** | 3-seed within-repo residual **> .02** → rounds 1..5 run; **≤ .02** → STOP at round 0 and the seed verdict is terminal |
| **Where it stands at 1 seed** | Layer-1 protocol: eval +.0576, test +.0390, repo-combined **+.0487** [+.014, +.085], P(>0) = .996. Closure protocol: honest-full **+.0522** [+.015, +.089], Wilcoxon p = .0011 — the two protocols agree |
| **What is missing** | dense v3 seeds 1 and 2 (chained on sk3 GPU 6; ETA ≈ 05:30 and ≈ 15:00 on 2026-08-08) |
| **How to close it** | pull `rm_out_seed{1,2}/preds_{eval,test}.csv` into `closure/code_v3/dense_seed{1,2}/` and run `python3 gate_readout.py` (see RUNBOOK) |

### 9.1 The gate readout, and the two conditions for firing rounds

`gate_readout.py` is the single command that decides the campaign. It is seed-count
agnostic, **runs the mandatory OOF alignment gate first and refuses to report if it
fails**, and emits per-seed within-repo T with its across-seed spread, Δ under both
protocols, three tiers (eval / test / both-splits-combined-at-the-repository-level), the
n-weighted *and* equal-repo Δ with repo-cluster bootstrap CIs, jackknife SE and paired
Wilcoxon, and the verdict against the frozen .02. It carries **`BINDING: true` only at 3
seeds**; with fewer it self-labels INTERIM and states that rounds stay held. Verified
end-to-end at 1 seed, where it reproduces every round-0 number
(`gate_readout_1seed.json`).

**Rounds fire only when BOTH conditions hold: `BINDING: true` (3 seeds) AND a free ledger
GPU.** The compute condition is not the interesting one — at 02:30 on 2026-08-08 all eight
sk3 GPUs were claimed by other agents (`claude-vat-fullgrid` on 0 and 2,
`claude-decor-battery-fable` on 7, GPU 6 running this cell's own seed chain), so no round
could have been scored regardless. The **methodological** condition is the binding one:
the mining slice is defined as top |dense percentile − VA_nl percentile| and the dense
score is the **seed ensemble**, so a slice built before all three seeds land is a
*different* slice — firing early would spend the sealed fleet's one blind look on the
wrong one. Coordinator confirmed the hold on this ground; no deviation from the 3-seed
wording.

### 9.2 TWO-SEED INTERIM READING (2026-08-08 09:50) — **INTERIM, NOT THE GATE**

Seed 1 finished training at 05:13 and scored at 05:27 (`rm_out_seed1/`, merged into
`eval_pass_results.json` without touching seed 42). Seed 2 started 05:27 and is at step
2,901 / 5,958 (49%), sole tenant on GPU 6, ETA ≈ 14:50. **`BINDING: false` — rounds stay
held.** OOF alignment gate re-run and passed (shuffled counterfactual .493 / .503).

**Within-repo T per seed, and the spread** (the round-0 item-1 deliverable, at 2 of 3):

| tier | seed 42 | seed 1 | spread | 2-seed ensemble | (pooled, NOT a readout) |
|---|---:|---:|---:|---:|---|
| eval (71 repos) | .7093 | .7095 | **.0002** | .7185 | .6488 → .6795 |
| test (73 repos) | .6540 | .6294 | .0246 | .6487 | .7373 → .7229 |
| **both, combined at repo level (144)** | .6829 | .6713 | .0116 | **.6852** | .6933 → .7008 |

**Residual, both definitions of "the N-seed residual"** (the same ambiguity
`maps_hw_si/cells.py` recorded as T_ensemble vs mean-of-seed-AUC; both are reported and
both must clear .02):

| tier | ensemble Δ | 95% CI | Wilcoxon p | dense wins | **mean of per-seed Δ** | per-seed range |
|---|---:|---|---:|---:|---:|---|
| eval | **+.0668** | [+.007, +.124] | .00069 | 50/71 | **+.0577** | [+.0576, +.0578] |
| test | **+.0337** | [−.007, +.079] | .029 | 42/73 | **+.0267** | **[+.0144, +.0390]** |
| **both combined** | **+.0510** | **[+.014, +.087]** | **.00011** | 92/144 | **+.0429** | [+.0371, +.0487] |

Closure protocol tracks it: combined +.0545 [+.021, +.091], p = .00009.
**Most conservative of all four readings = +.0267 (test, mean of per-seed Δ) — still above
the .02 threshold.**

**Three things this reading establishes, and one of them corrects me.**

1. **CORRECTION to §5.1.** From seed 1's mid-training validation curve I wrote that the
   seed-to-seed gap on this cell "looks like ~.04 pooled, not the ±.02 the instrument note
   assumed". That was read off a *pooled* number and it does not survive the within-repo
   readout. Seed 1 beats seed 42 by **+.031 pooled on eval** but by **+.0002 within repo**;
   it is .014 worse pooled on test but .0246 worse within repo. **The pooled seed gap and
   the within-repo seed gap are essentially unrelated** — the composition effect that makes
   pooled AUC untrustworthy *across splits* on this cell makes it untrustworthy *across
   seeds* too. The honest statement is the within-repo one: seed spread .0002 on eval,
   .0246 on test, .0116 combined.

2. **The eval-side residual is seed-invariant; the test-side residual is not.** Per-seed Δ
   on eval is +.0576 / +.0578 — agreement to two ten-thousandths across independently
   trained models. On test it is +.0390 / **+.0144**, and **seed 1's test-side residual
   falls below the .02 threshold on its own**. This is precisely the fragility the seed
   chain was run to expose, it vindicates the instrument note's "test-side margin is thin"
   caveat, and it means any final claim must be split-resolved: the code residual is solid
   on eval and seed-fragile on test.

3. **Ensembling gains are real and must not be double-counted.** The 2-seed ensemble T
   (.6852 combined) exceeds *both* single seeds (.6829, .6713), so the ensemble Δ (+.0510)
   is mechanically larger than the mean of per-seed Δs (+.0429). Both are reported
   everywhere; the ensemble is the row-level score every fit and stratification in this
   campaign actually consumes, the mean-of-per-seed is the conservative companion.

Artifacts: `closure/code_v3/gate_readout_2seed.json`, `gate_readout.py`.

**Superseded:** the 1-seed interim (combined Δ +.0487, [+.013, +.085], p = .0007).

---

### 9.3 THE BINDING 3-SEED GATE (2026-08-09) — **PASS**

Seed 2 finished training 15:39 and scored 15:54 on 2026-08-08; GPU 6 released at 22:54Z
(`gpu_ledger.txt`, `agent=claude-closure-code-v3`, `job=dense_seeds_1_2`). All three dense
seeds present; `gate_readout.py` self-labels **FINAL / `BINDING: true`**. OOF alignment
gate re-run first and passed (shuffled counterfactual .493 / .503).

Pooled selected AUCs, for the record only: seed 42 .6488/.7373, seed 1 .6795/.7229,
seed 2 .6721/.7136 (eval/test). **These are never a readout on this cell.**

#### 3-seed within-repo T ± spread — the round-0 item-1 deliverable

| tier | seed 42 | seed 1 | seed 2 | **mean** | **SD** | spread | 3-seed ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|
| eval (71 repos) | .7093 | .7095 | .7008 | **.7065** | **.0049** | .0087 | .7196 |
| test (73 repos) | .6540 | .6294 | .6377 | **.6404** | **.0125** | .0246 | .6494 |
| **both, combined at repo level (144)** | .6829 | .6713 | .6707 | **.6750** | **.0069** | .0122 | **.6861** |

#### The residual, per seed and both aggregate definitions

| tier | per-seed Δ (42 / 1 / 2) | **mean of per-seed** | **ensemble Δ** | 95% CI | Wilcoxon p | dense wins |
|---|---|---:|---:|---|---:|---:|
| eval | +.0576 / +.0578 / +.0491 | **+.0548** | **+.0679** | [+.010, +.123] | .0011 | 50/71 |
| test | +.0390 / **+.0144** / +.0227 | **+.0254** | **+.0344** | [−.008, +.082] | .038 | 41/73 |
| **both combined** | +.0487 / +.0371 / +.0365 | **+.0408** | **+.0519** | **[+.015, +.089]** | **.00022** | 91/144 |

Closure protocol agrees throughout: combined **+.0554** [+.023, +.091], p = .00016.

#### VERDICT

> **GATE HOLDS.** Frozen statistic = within-repo n-weighted Δ, both splits combined at the
> repository level, Layer-1 protocol, dense = 3-seed ensemble: **Δ = +.0519,
> 95% CI [+.015, +.089], Wilcoxon p = .00022 over 144 repositories / 10,284 rows**, against
> a threshold of .02. The conservative companion (mean of per-seed Δ) is **+.0408**, and
> the single most conservative reading anywhere in the table — the test half read as the
> mean of per-seed Δs — is **+.0254**, still above .02. **Rounds 1..5 are authorised.**

#### What the seed chain actually bought, stated plainly

1. **The gate passes on every aggregate reading, and it was not a foregone conclusion.**
   Both new seeds came in *below* seed 42 within repo (combined .6713 and .6707 vs .6829),
   which is the direction that shrinks Δ. The 3-seed combined residual is +.0408 (mean of
   per-seed) against seed 42's +.0487 — the single-seed figure was **optimistic by ~16%**.

2. **The published test-side +.0390 does not reproduce and must be retired as a headline.**
   At three seeds the test residual is **+.0254 (mean of per-seed) / +.0344 (ensemble)**,
   roughly a third smaller, its CI includes zero on the ensemble reading, and **one of the
   three seeds (seed 1, +.0144) falls below the .02 threshold on its own**. The correct
   sentence going forward is *"test-side residual +.025 to +.034, seed-sensitive, one seed
   in three below threshold"* — never "+.039".

3. **The eval side is solid and nearly seed-invariant.** +.0576 / +.0578 / +.0491, mean
   +.0548, all three above threshold, T_within SD .0049.

4. **The ±.02 single-seed caveat is now measured, and it was mis-stated in both
   directions.** Within repo the T spread is **.0049 (eval) / .0125 (test) / .0069
   (combined)** — smaller than the assumed ±.02 on eval, and the *residual's* per-seed
   range on test is [+.0144, +.0390], i.e. the fragility is real but it lives on the test
   half specifically, not uniformly. (And per §9.2, none of this is visible in pooled AUC:
   seed 1 is the *best* pooled-eval model and the *worst* within-repo one.)

Artifact: `closure/code_v3/gate_readout_3seed.json`.

**[SUPERSEDED 2026-08-10: lane C cleared, rounds STARTED — see §11.]** GPU lanes were
assigned elsewhere at the time of the gate; Everything downstream is staged and unfired: the
decomposition-first pass on the 5 census-deduped parents, the sealed P=6/3-family fleet,
the 4 corpus-matched PR probes, `score_round_code.py`, and the within-repo `readout_code.py`
with dual-tier seed bands and repo-cluster CIs. The mining slice will be built against the
**3-seed ensemble**, which is now final.

For the gate to fail, the two new seeds would have to be *substantially better* within
repo than seed 42 — the repo-combined Δ would have to fall from +.049 to ≤ .02, i.e. the
3-seed mean T_within would have to drop by roughly .03 relative to VA_nl. That is
possible (single-seed dense on this corpus is worth about ±.02) but it is not the way to
bet; the gate exists precisely because the test-side margin was ~2 seed-sigmas and the
honest thing is to measure it rather than argue about it.

## 10. Terminal status line for the strict list (current)

> **code / PR-merge (v3 enriched):** Layer-1 T ✓, Layer-3 **ROUND 0 COMPLETE and the
> 3-SEED GATE HOLDS**. Frozen statistic **Δ = +.0519, [+.015, +.089], Wilcoxon p = .00022**
> over 144 repos / 10,284 rows (threshold .02); conservative mean-of-per-seed **+.0408**;
> most conservative reading anywhere **+.0254**. Closure protocol +.0554 [+.023, +.091] —
> **the closure refit does NOT shrink the level on this cell**, unlike N&C.
> **Rounds 1..5 AUTHORISED, NOT STARTED** (GPU lanes assigned elsewhere; slotted lane C
> behind math.SE on GPU 7); the mining slice will be built on the now-final 3-seed ensemble.
> **The published test-side +.039 is RETIRED**: at 3 seeds the test residual is **+.0254
> (mean-per-seed) / +.0344 (ensemble)**, CI includes 0 on the ensemble reading, and **one
> seed of three (+.0144) is below threshold** — eval is near seed-invariant
> (+.0576/+.0578/+.0491, mean +.0548). Within-repo T 3-seed SD **.0049 eval / .0125 test /
> .0069 combined**; single-seed seed-42 figures were optimistic by ~16% combined. Bank census
> 83 → **80 effective concepts** (least redundant bank in the program; a78 ≡ a105 — the #1
> and #2 criteria are one concept). **Not a length model** (VA_nl loses only .012–.017 under
> within-repo length stratification, vs Style Invitational's collapse to .541). A dominates
> V (.612 vs .569) and Δ_interact is only +.022 — the signal is in the criteria, not the
> GBM. **Swap baseline C₋ = .510** — the bank is at chance on the third of within-repo
> pairs dense gets wrong (N&C had .587), so it holds no independent signal there and the
> swap boundary is one round away on test (C₋ = .4955). Position channel: within-repo .4993
> on average but 35% of repos beyond \|AUC−.5\| > .15, both instruments partially track it
> (dense ρ = +.148), and discounting it leaves Δ = +.0509. Design §11 **passes and this is
> the program's first fused-beats-BOTH-parents cell** (fused .6920 > dense .6829 > bank
> .6342). **PROTOCOL WARNING: the frozen saturation statistic is 5× noisier than ε here**
> (MONITOR = 29 scorable repos, VA_nl seed spread .027 vs ε = .005; MONITOR Δ +.047 with
> CI [−.078, +.173]) — every round gain must be quoted with its seed band and the
> honest-full companion. **Pooled numbers never quoted.** Mining rounds **GATED** on the
> 3-seed residual; seeds 1–2 training on sk3 GPU 6.

---

## Appendix — artifact index

| artifact | path |
|---|---|
| campaign code + all round-0 outputs | `methods/taste_decomposition/closure/code_v3/` |
| runbook (how to resume after the seeds land) | `closure/code_v3/RUNBOOK.md` |
| splits + geometry report | `closure/code_v3/splits.npz`, `splits_report.json` |
| gate table + closure baseline + position + swap | `closure/code_v3/round0_results.json` |
| gate uncertainty (bootstrap, weighting sensitivity) | `closure/code_v3/gate_uncertainty_seed42.json` |
| design §11 fused check | `closure/code_v3/fused_check.json` |
| bank concept census (L0→L5) + blind packet + verdicts | `closure/code_v3/census_code.json`, `census_blind_packet.json`, `census_verdicts_judge{A,B}.json` |
| position line | `closure/code_v3/position_line.json`, `position_line_ext.json` |
| length/geometry stratification | `closure/code_v3/length_stratification.json` |
| decomposition parents + staged prompt | `closure/code_v3/parents_code.json`, `code_v3_rd_parents_used.json`, `scratchpad/code_v3/code_v3_rd/prompt_decomposer.txt` |
| incoming instrument (83-criterion scores, V matrix, VA_nl OOF, dense seed-42 preds) | `closure/code_v3/abank_rescore/`, `closure/code_v3/dense_seed42/` |
| dense seed chain (sk3) | `datasets/code-review/dense_standard_v3/run_code_v3_seeds12.sh`, `runner_v3_seeds12.log`, `rm_out_seed{1,2}/`, `abank_rescore/score_one_seed_v3.py` |


---

## 11. DECOMPOSITION-FIRST PASS (FREEZE ADDENDUM 3) — authored, audited clean, scoring in flight

Lane C cleared 2026-08-10 and the rounds were greenlit. The decomposition pass runs first,
as the brief directs.

### 11.1 RECORDED FLEET DEGRADATION — the Claude family is gone this session

**The hard subagent cap was reached (500/500), so no Claude leg is available for any role
in this campaign.** This is a larger degradation than any previous cell recorded, and it
changes three instruments, all noted at the point of use:

| role | freeze / worked-campaign instrument | what actually ran | class |
|---|---|---|---|
| decomposer | sealed Claude Opus subagent | **gpt-5.6-luna** via `codex exec`, effort high, read-only scratch wd outside the repo | frontier |
| blind routing auditor | fresh Sonnet-class subagent per round | **gpt-5.6-luna** via `codex exec`, effort high, fresh sealed context | > Sonnet |
| proposer fleet (rounds 1..5) | Claude ×2 + gpt-5.6-luna ×2 + GLM ×2 (P=6/3 families) | Claude unavailable → **codex ×4 + GLM ×4, P=8 / 2 families** | above the P floor, **below the family floor** |

GLM was checked first per the directive and **both keys are live** (smoke-tested, 1.0 s and
1.2 s, `model=glm-5.2`). So the fleet clears the freeze's P ≥ 4 floor with room (P = 8) but
sits at **2 families, not 3** — recorded as a degradation, and it means the cross-family
species statistics for this cell are not comparable to the 3-family cells.

### 11.2 The pass

Parents (§7): 5 census-deduped MIXED parents. The sealed decomposer returned a
well-formed object on the first call — **10 components (5 candidate-real + 5 surface,
exactly 2 per parent) and 5 Addendum-4 position channels**, parse-checked before use.

| | candidate-real (→ A) | surface (→ B) |
|---|---|---|
| Contribution readiness | Evidence of Correct, Complete Contribution | Submission-Norm Marker Extent |
| Simplicity (KISS/YAGNI) | Requirement-Bounded Design Simplicity | Complexity-Discussion Surface Extent |
| Refactoring quality | Substantive Behavior-Preserving Refactoring | Refactor-Presentation Marker Extent |
| Documentation formatting | Semantically Useful Documentation | Documentation-Format Marker Extent |
| **PR communication quality** | **Actionable Change Rationale** | **Communication-Scaffolding Extent** |

Position channels (all → B): Conventionlessness Cue, Shared-History Presupposition,
Concurrent-Activity Cue, Version-Era Anchor, Automation-Generated Submission.

### 11.3 Blind routing audit — clean

| | round d |
|---|---|
| criteria audited (+ planted) | 15 (+4) |
| **misrouting rate** | **0.0%** |
| disputes → arbiter | **0** |
| **corpus-matched probes** | **4/4 separated** |
| final routing | **5 A / 10 B** (2 mixed) |
| auditor confidence | high on 13/15, medium on 2 |

Both authored probe pairs separated as designed: *"states why the change is needed"* and
*"names the concrete failure it addresses"* → quality-relevant; *"extent to which a PR
template is filled in"* and *"conventional-commit and issue-reference formatting"* →
incidental. A 0% misrouting rate with 4/4 probes is the cleanest audit any cell in the
program has opened with, and it is worth noting that the decomposer and the auditor were
the *same base model in different sealed contexts* — so the audit was not adversarially
independent by family, only by context. Recorded as a limitation of this session's fleet.

### 11.4 Scoring — in flight, anchors already passed

Gemma-4-31B offline batch on sk3 GPU 7 (lane C), ledger-claimed after a 4 MiB co-tenant
check, launched with `setsid --fork` so the job runs at **ppid = 1** and survives session
loss; verified resident on GPU 7 alone (EngineCore 64.7 GB, single card, co-tenant on
GPU 1 belongs to another agent and was never touched).

**Anchor battery K = 50 per tier — all three gates PASS, and by a wide margin:**

| tier | mean over answered | NA rate | mean with NA→0 |
|---|---:|---:|---:|
| merged (pos) | .4825 | .009 | **.4780** |
| closed-unmerged (neg) | .3689 | .013 | **.3640** |
| word-scrambled | .0291 | .291 | **.0207** |

`scram < neg < pos` on all three. Note the contrast with the incoming 83-criterion bank,
whose anchors read .3066 / .2307 / .0013: these 15 decomposed criteria have a **pos−neg
separation of .114 against the bank's .076**, and a far lower NA rate (.01 vs .58) — the
decomposition produced criteria the judge can almost always answer, which is exactly what
splitting a parent into a substantive component and a surface component should do.

Throughput note: 19 prompt/s versus the base bank's 63. The cause is prefix-cache
amortisation — the ~7K-token PR context is prefilled once per row and shared across the
criteria scored for that row, so 15 criteria amortise it 5.5× worse than 83 did. ETA ≈ 2.6 h
for 11,452 × 15 = 171,780 prompts. This sets the cost of every later round
(25 criteria ≈ 1.6 h), which is worth knowing for lane scheduling.

### 11.5 Accumulated rulings applied

* **enforced collapse gate** — collapsed criteria (>98% modal or all-NA) are now *excluded*
  from every block in `readout_code.py`, with ids and statistics recorded under
  `enforced_collapse_gate`, never deleted;
* **item-view assertion** — `run_batch` asserts the (row, criterion) ownership map of the
  flattened prompt list, the corner entries, the reshape, and non-empty judge text, so a
  transposed or short batch cannot reach a score matrix;
* **parseability / completeness waits** — the scorer refuses to write if the generation
  count differs from the prompt count (interrupted-generation gate); the decomposer and
  auditor runners both parse-and-count-check before their output is accepted;
* **`setsid --fork` + ppid = 1** — verified on both the scorer and the chained follow-on;
* **swap algebra, dual-tier seed bands, repo-cluster CIs** — already in `readout_code.py`.


## 12. Infrastructure log — outage, and two bugs the dry run caught before they cost a run

### 12.1 sk3 unreachable (jump-host failure), detached jobs unaffected

From ~18:15 on 2026-08-10 the laptop lost the path to sk3: `kex_exchange_identification:
Connection closed by remote host`, 100% ICMP loss to `skampere3`. Diagnosed to the **jump
host, not sk3** — the ssh config routes through `ProxyCommand ssh -4 -W %h:%p whale`, and a
direct probe of `whale` returns `Connection reset by 171.64.75.72 port 22`. So sk3 itself
was never shown to be down.

Consequences, and why they are small: both GPU jobs were launched with `setsid --fork` and
run at **ppid = 1**, so neither depends on my session or my ssh path — the decomposition
scorer (3 of 8 shards done at the last contact) and the chained code_competitions dense T
continue regardless. A patient reconnect waiter (120 s backoff, one probe at a time, per
the NAT64 gentle-retry rule; IPv6 not touched) is armed and will re-report scorer state on
recovery. The one thing that *is* blocked is fetching new slice-card text, and that path
fails loudly: `cells.fetch_texts` asserts rather than returning short text, which was
confirmed during the outage.

### 12.2 A `sys.path` shadowing bug — silent, and caught only by running the code

`readout_code.py` and `stage1_slice_code.py` both did

```python
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "maps_hw_si"))
```

The second insert puts **maps_hw_si at position 0**, so a bare `import cells` resolved to
`maps_hw_si/cells.py` — the HashtagWars/Style-Invitational adapter — instead of this cell's
shim. It surfaced as `TypeError: load() missing 1 required positional argument: 'cell'`,
but the dangerous form of this bug is the silent one: had the two `load()` signatures been
compatible, the readout would have run to completion on **another cell's population**.
Fixed by inserting `maps_hw_si` first and `HERE` last, with the reason in a comment.

**Blast-radius check, run immediately:** a scan of the whole `closure/` tree found the
pattern in five further files, **all of them mine** (`round0_code.py`, `census_code.py`,
`gate_readout.py`, `build_splits_code.py`, `select_parents_code.py`) and **none of them
affected** — they import `cells_code` by its unambiguous module name, and `closure_core`
exists only in `maps_hw_si` so it resolves correctly either way. No other cell in the
program uses the pattern. **No already-computed result on this cell is affected**: the
census, the splits, the gate table, the §11 fused check and the closure r = 0 anchor all
go through `cells_code`.

### 12.3 A round-label crash, and the dry run itself

`stage1_slice_code.current_blocks` did `int(rnd)` on anything that was not `"d"`, so a
non-integer round label crashed it. Now tolerant: prior rounds are `["d"] + 1..r-1` for an
integer round, and `[]` for anything else.

### 12.4 Dry-run outcome — readout path PROVEN

A **third** bug surfaced on the rerun: `discount()` passed the MONITOR-length
`rbx["nl_mon"]` into a full-population within-repo reader — the identical shape error that
had already bitten `round0_code.py`. Fixed with `_expand`, and because this class has now
recurred, `cells_code.within_repo_auc` gained a **named shape guard** that fails with an
explanatory message instead of an opaque `IndexError` (or, worse, passing silently on a
cell where the two lengths happen to coincide).

With that, the readout ran clean and every stage produced correct structure:

| stage | dry-run outcome |
|---|---|
| routing + **enforced collapse gate** | n_A 5, n_B **10 → 9** — the planted constant column B03 was excluded from every block and recorded |
| Track-A curve | all four tiers (MONITOR / honest-full / eval / test) with gains |
| MONITOR seed band | .0274, matching the round-0 value |
| spurious map | 18 channels (9 ids × score + applied), 0 dropped by the degeneracy screen |
| discount | both `ALL_B` and `STRICT_no_mixed`; matched-sampling trigger correctly **not** fired at spurious-alone .5055 < .65 |
| stacked increment | joint-B, dense, and the dense-over-B increment, all tiers |
| swap algebra | ΔC₊ / ΔC₋ and the signature flag, all tiers |

The *levels* in that run are meaningless — a synthetic column was built to separate the
label almost perfectly, which is why VA_new reads ~.90 and Δ_adj goes negative. That is the
point of a dry run: it proves the plumbing without pretending to be a measurement. All
`code_v3_rDRYRUN*` artifacts were deleted afterwards.

Both bugs were found by **dry-running `readout_code.py` end-to-end on synthetic scores**
during the outage — 300 lines of readout code that had never been executed, and that the
real 2.6-hour scoring job feeds directly into. The synthetic set deliberately includes one
constant column so the **enforced collapse gate** is exercised rather than assumed. All
dry-run artifacts carry the tag `code_v3_rDRYRUN` and are deleted afterwards, so nothing
that could be mistaken for a measurement is left on disk.


## 13. ROUND d RESULT — the decomposition pass closes ~13% of the residual, and the swap fires

Scoring completed during the outage (8/8 shards, 171,780 prompts, pooled NA **.0086**,
**0/15 collapsed**, modal fraction max .825). Readout run on the repaired code, with the
dense score now the **final 3-seed ensemble** — so the r = 0 anchor is restated here on that
ensemble (HONEST +.0554) rather than the seed-42-only +.0522 of §8.

### 13.1 The first curve point

| tier | T | VA_nl r0 | VA_nl **r_d** | **gain** | Δ r0 | **Δ r_d** |
|---|---:|---:|---:|---:|---:|---:|
| **MONITOR** (29 repos) | .6763 | .6152 | .6334 | **+.0182** | +.0611 | **+.0429** |
| **honest-full** (144 repos) | .6861 | .6307 | .6377 | **+.0070** | +.0554 | **+.0484** |
| eval (71) | .7196 | .6506 | .6586 | +.0080 | +.0690 | +.0610 |
| test (73) | .6494 | .6088 | .6148 | +.0060 | +.0406 | +.0346 |

Routing 5 A / 10 B, misrouting 0.0%, probes 4/4, **0 criteria lost to the enforced collapse
gate**.

**Saturation flags: NOT sub-ε on either tier** (+.0182 MONITOR, +.0070 honest, both above
ε = .005). Trailing run 0 of the required 2 → **mining continues to round 1.**

**But the ε comparison must be read against the seed band, and this is the §8.2 warning
biting on the very first round.** The MONITOR VA_nl seed spread is **.0274**, i.e. the
MONITOR gain of +.0182 is *smaller than the noise on the statistic it is being compared
against*. The honest-full tier is the better-powered one and it agrees in sign at +.0070.
So the defensible statement is: **the decomposition round bought a small but real gain —
+.0070 on 144 repositories, a 12.6% reduction of the residual (+.0554 → +.0484) — and the
30% reduction visible on MONITOR (+.0611 → +.0429) is not separable from seed noise and
must not be quoted.**

### 13.2 The swap signature fires — on round one

| | ΔC₊ | ΔC₋ | Δρ | signature? |
|---|---:|---:|---:|---|
| r0 → r_d (honest) | **+.0188** | **−.0061** | +.019 | **YES** |

C₊ up and C₋ down is exactly the pattern the algebra was written to detect: the bank gained
rank agreement with the dense model partly by **inheriting its errors**. It matters more
here than it did on N&C because this cell's round-0 C₋ was already **.5099 — chance** (§8.3),
so pushing it down puts the bank *below* chance on the pairs dense gets wrong. Part of the
+.0070 honest gain is therefore dense-imitation rather than independent articulation, and
the honest reading of round d is a **real but partly-imitative** gain. Round 1 should be
watched for the same signature; two consecutive swap rounds would mean the miner has
started teaching the student the teacher's mistakes.

### 13.3 Spurious map — ten named channels, all weak

Within-repo alone-AUC on the honest population:

| channel | alone AUC | MIXED | conjectured upstream parent |
|---|---:|---|---|
| **Version-Era Anchor Extent** | **.4629** | | temporal position / project lifecycle stage |
| Refactor-Presentation Marker | .5278 | | surface carrier of "refactoring quality" |
| Complexity-Discussion Surface | .5251 | **yes** | surface carrier of "simplicity" |
| Shared-History Presupposition | .5233 | **yes** | late arrival / accumulated repo history |
| Communication-Scaffolding | .5097 | | surface carrier of "PR communication quality" |
| Concurrent-Activity Cue | .5046 | | repository busyness around the PR |
| Documentation-Format Marker | .5038 | | surface carrier of "documentation formatting" |
| Submission-Norm Marker | .4932 | | surface carrier of "contribution readiness" |
| Conventionlessness Cue | .4939 | | early vs late arrival / codebase maturity |
| Automation-Generated Submission | .4928 | | repository automation maturity |

**Every channel is weak** — the largest deviation from chance is .037, and the joint
spurious model reaches only **.5364**, far below the .65 matched-sampling trigger (not
fired). For calibration, N&C's joint nuisance model hit **.672** after one round and .712
after three. **The code cell's named nuisance space is much thinner than N&C's**, which is
consistent with everything else this cell has shown: the position channel is null on
average (§4), no geometry feature predicts alone (§6), and the bank is not a length model.

The strongest channel being **anti-predictive version-era vocabulary** (.463) is the only
substantive spurious finding so far: PRs whose language pins them to an older toolchain era
fare slightly worse, within repository.

### 13.4 The discount is null again — third independent confirmation

| readout (honest) | ALL 10 channels | STRICT (2 mixed excluded) |
|---|---:|---:|
| spurious-alone (HistGB, within-repo) | .5364 | .5319 |
| **Δ_adj** | **+.0731** | **+.0674** |
| Δ undiscounted | +.0484 | +.0484 |

**Δ widens under discount, at both ends of the MIXED band** — the same behaviour N&C and CW
showed, and the same warning applies: **Δ_adj is not an effect size**, because stratifying
on a nuisance score costs VA more than it costs T. The defensible claim is the negative one:
*ten named nuisance channels, five of them derived by reasoning from unseen upstream causes
and five of them position fingerprints, do not explain the code residual.*

The stratification-free stacked increment says it without any stratification: after a
logistic stack absorbs **all ten** named channels, the dense score still adds **+.1442**
within-repo, against the bank's +.0904.

Together with §4 (position discount leaves Δ at +.0509) and §6 (every geometry
stratification leaves Δ at +.036 to +.054), this is now the **third independent family of
nuisance controls that fails to explain the residual**.
