# Math vertical V/A/T closure (2026-06-12)

Affirmative close of the three math legs (Math Stack Exchange, AoPS, mathlib4)
under the Verifiability / Articulability / Taste decomposition. Each leg has a
community-preference label; we measure how much of it is recoverable by
deterministic code metrics (V), by LLM-judged-but-not-mechanizable criteria
(A), with the remainder above the best measurable layer being taste+noise (T).
Dense ceilings (C) are now UNPARKED (Llama-3.1-8B LoRA sweeps running on sk3 GPU 3
as of 2026-06-14) — see the Verification & dense status section below.

**One-line finding:** across all three legs, deterministic V predicts
*correctness / completeness* but sits at or near the question-only floor for
*preference*; the discriminative weight lives in the A-band (judgeable style,
elegance, directness) and an irreducible T residual. Even in the most
machine-checked domain on earth (mathlib, where correctness is a compiler
guarantee) the articulated norm layer is ~77% judgment-calls, ~14%
mechanizable, ~6% openly contested.

---

## Verification & dense status (2026-06-14)

**Independent adversarial verification — PASSED.** A 6-agent verification workflow
(~295K tokens, 115 tool calls) re-loaded the raw data and reproduced *every*
headline AUC to 3–4 digits, then ran adversarial attacks; the science survived:

- Math.SE A-layer reproduced exactly: combined GroupKFold AUC **0.661 eval / 0.639
  test** (claim 0.661/0.638); train-fit→held-out 0.6555/0.6315; every a01–a14 beats
  the 0.461 floor; floor reproduced at 0.4615 with the documented recipe.
- **The +0.12 gap is NOT a weak-V artifact.** A skeptic built the strongest
  deterministic V it could (rich text features + a full per-fold answer-text TF-IDF,
  30K features) and it plateaus ~**0.59**, still ~0.06–0.07 below A-only (0.66) — and
  answer length alone is at chance (v3.3 balancing removed the length confound). V+A
  everything 0.672/0.652 vs A-only 0.661/0.639 (A subsumes V, +0.01) reproduced.
- AoPS reproduced exactly: base rate 0.696 raw / 0.687 cleaned; lexicon ladder
  0.706/0.714/0.716; post-only TF-IDF **0.727**; admissibility audit clean (no
  reference-derived feature in the ladder; sim-to-ref 0.741–0.771 correctly excluded);
  two-regime collapse to V-new 0.560 / TF-IDF 0.594 on verified-correct (n=6140);
  0.73 robust to tighter transcription filters (0.726→0.701). thanks_resid OOF
  R²=0.5216; elegance→thanks within-problem ρ+0.066 (p~5e-11) survives length AND
  correctness controls; same_approach→thanks NULL (AUC 0.499).
- mathlib y reproduced as literally `n_review_threads==0` (agreement 1.0).

**Two recorded concerns (not blockers, affect framing):**
1. The 14 A-metrics are highly collinear (PC1=79% of variance, mean inter-metric
   r=0.77); the combined LR beats the best single metric (a07=0.645) and the mean
   A-score (0.648) by only ~0.013–0.016. So the A-layer is closer to **one latent
   quality factor** than 14 independent dimensions — the A>V gap is real and robust,
   but "14 metrics" overstates the dimensionality; report it as a single judged
   quality axis with a +0.12 lift over deterministic V.
2. The A-layer was scored on a near-one-answer-per-question subset (1.008 mean) while
   the dense pool is 1.80 answers/question (48.8% multi-answer). The A-AUC is thus the
   easier single-answer regime. **For the C-vs-A comparison, dense test AUC should be
   read BOTH on the full test set AND restricted to the A-scored answer_ids** (same
   rows the A-layer saw) — that restricted number is the apples-to-apples C≥A check.

**Dense (C) ceilings — DONE + verified (2026-06-14/15).** All 3 legs Llama-3.1-8B
LoRA (r16/α32, lr5e-5, batch16, max_len1024, 2 epochs, gradient-checkpointing ON),
sequential on one GPU. Data-scaling ladder [0.25, 0.5, 1.0]; **C = the frac=1.0
endpoint**, reported as the CLEAN held-out **eval** AUC (the trainer selects the best
checkpoint on *test*, so test is mildly optimistic; eval was never used in selection
and eval≈test confirms negligible selection inflation):

| leg | floor | V (code) | A (judge / TF-IDF) | **C (dense, clean eval)** | C scaling 0.25→0.5→1.0 |
|---|---|---|---|---|---|
| **Math.SE** | 0.461 | 0.55 | **0.638** (A, test) | **0.794** | 0.767→0.782→0.794 |
| **AoPS** (full pool) | — | 0.64–0.71 | **0.726** (TF-IDF) | **0.777** | 0.740→0.764→0.769 |
| **mathlib** (title-only) | 0.50 | null 0.49–0.52 | — | **0.617** | 0.577→0.606→0.623 |

**Math.SE C−A is real and apples-to-apples.** Restricting the dense test/eval preds
to the *identical* A-scored answer_ids: C-on-A-subset = 0.794 (test) / 0.797 (eval)
vs A = 0.638 / 0.661 → **C−A = +0.14–0.16 on the same rows.** Verified clean:
(a) no metadata leakage (text is plain "Question:…Answer:…" math; score/vote/badge
tokens <1% false-positives); (b) confounds null — rho(dense_prob, answer_position)
+0.01, year +0.05, n_answers −0.01; (c) the one non-null, rho(dense_prob, propensity/
decile)=+0.16, is **benign** — the balancing variable predicts the label at AUC 0.498
(chance), within-decile P(y=1)∈[0.47,0.53], so it cannot inflate C; (d) selection-on-
test inflation negligible (eval 0.794 ≈ test 0.794). The dense model reads the math and
assesses answer quality ~0.79, far above the code-checkable V (0.55) and the
articulable-judge A (0.65) — **a large fully-tacit/semantic layer the judge doesn't
reach** (cf. tacitness two-layers: this is the C−A "replicable-dense-beats-articulable"
band). NOTE for interpretation: C reads only `text`, so part of C−A may be writing-
style/register the model uses as a quality proxy (no answerer-id in the v3.3 pool to
separate style from correctness — open decomposition).

**AoPS C ≈ 0.77 > TF-IDF 0.726 by only +0.05** — a thin dense gain, expected because
the same_approach label is heavily lexical (corr 0.355 with sim_word); dense adds a
small semantic increment over bag-of-ngrams. **mathlib C ≈ 0.62 ≈ title-TF-IDF 0.63**
— the dense model extracts ~nothing beyond title bag-of-words, and even the dense
artifact ceiling stays near the floor, reconfirming friction is process- not artifact-
determined (the title is the only admissible text; no patch exists).

---

## Leg 1 — Math Stack Exchange (sharpest V vs A head-to-head)

- **Label:** community answer quality = accepted ∧ score≥3 (y=1) vs score≤0
  (y=0), on the **v3.3 propensity-balanced** pool (99,722 rows, 50/50; splits
  train 79,873 / eval 9,995 / test 9,854).
- **Floor** (question-only TF-IDF+LR): **0.461**. Only margin above this counts.
- **V (deterministic, code-checkable):**
  - claim verification (sympy on extracted equalities): **0.541** — and
    *unchanged* by a 9.9h recovery run that raised checkable coverage 45%
    (more verification coverage does not move V; correctness saturates).
  - lint battery (v18–v26: step-chains, LaTeX, symbol hygiene, typos,
    near-dup, …): **0.567 eval / 0.569 test**.
  - lint + claim combined (9.8K join): **0.578 test**.
  - → **deterministic-V ceiling ≈ 0.54–0.58.** Dominant signal is
    presentation/process proxies (near-dup, hedging, directness), not
    correctness tools.
- **A (LLM-judge a01–a14, essay-grounded rubrics, Qwen3.5-122B):**
  scored over v3.3; **EVAL + TEST done (8,456 / 8,282 clean, ~15.5% judge-fail
  each); train scoring (bonus coverage).**
  - **Combined A-layer AUC = 0.661 eval / 0.638 test** (q-grouped CV;
    +0.20 / +0.18 over the 0.461 floor). Every metric beats floor on BOTH
    splits (0.60–0.65 eval, 0.585–0.629 test); subsets stable (depth 0.654/
    0.635, priority 0.649/0.635, exposition 0.636/0.621).
  - **V-vs-A head-to-head (identical rows, both splits):**
    deterministic-V (lint) 0.551 / 0.554 → A 0.661 / 0.638 → V+A 0.671 / 0.649.
    **Articulability gap A−V ≈ +0.12 eval, +0.10 test — stable across splits.**
    (Canonical full-eval lint-V is 0.567; 0.551/0.554 are the judged-ok subset.)
  - **ALL-V + ALL-A combined (55 features: lint v18–26 + lexicon v11–14 +
    claim v15–17 + a01–a14):** V-everything (41 feats) = 0.557 / 0.559;
    **V+A-everything = 0.671 / 0.648.** Key: V+A is only +0.01 above A alone —
    **A subsumes V; once the judge is in, deterministic V adds almost nothing.**
    The articulable layer dominates the verifiable layer for this label.
  - **V/A convergence pairs are NULL on both splits** — judged a04 vs
    mechanical v13 clearly-count ρ +0.03; a10 honesty vs v11 hedging ρ +0.00;
    a11 vs v12 direct-open ρ +0.01–0.03; a08 vs v14 connectives ρ +0.05–0.07.
    Lexicon-V proxies combine to only 0.525–0.538: **the mechanical twins do
    NOT capture what the judge means** — the A-gain is genuine judgment, not
    lexicon-in-disguise. (This is the project's "code↔judge convergence"
    scorecard item, answered: they do NOT converge.)
  - Caveat: ~15.5% terminal judge-fail (degenerate Qwen-FP8 on messy answers);
    excluded rows may be lower-quality, a mild upward selection on A.
  - **On the split structure:** the 14 per-metric AUCs need NO training
    (roc_auc of raw judge score vs label). The only trained object is the
    A-combiner LR (15 params: 14 metrics + intercept). It was originally
    CV'd within-split; fitting it the textbook way — **train on the TRAIN
    split, evaluate held-out — gives eval 0.655 / test 0.631**, matching the
    within-split CV (0.661 / 0.638) and the cross-split fits (eval→test 0.639,
    test→eval 0.660). **A 15-param model needs ~hundreds of rows, not the 80K
    train split** — 2,672 scored train rows fit it identically. So the full
    train-split scoring is unnecessary for the measurement (the run crashed at
    2.7K rows on a co-tenant GPU OOM; moot). eval and test are two independent
    held-out confirmations; a separate dev split would only matter if we were
    tuning the judge/features on labels, which we are not.
- **T:** residual above A (and above C when un-parked).

## Leg 2 — AoPS (PRIMARY label = same-approach-as-editorial, the code-parallel)

- **PRIMARY label — same-approach-as-editorial (the code-parallel):**
  LLM-judged, base rate 0.699. **Headline predictability ~0.73 (full-pool,
  ex-ante)** — this is the AoPS number comparable to the LC/CC/CF code legs.
  - Ex-ante ladder (similarity-to-reference ruled INADMISSIBLE): V-struct
    0.544 / V-answer 0.602 / V combined 0.636 / register 0.653 / **post-only
    vocabulary 0.727** (the approach fingerprint, never sees the editorial) /
    everything 0.737. 11 interpretable deterministic lexicons recover ~90%
    (0.706 / 0.714 with V-old). So **deterministic V ~0.64-0.71, ceiling ~0.73.**
  - **Judge audited:** 50-case blind re-judgment, agreement 0.72 (0.85
    high-confidence), judge conservative not yes-biased — the 0.73 label is sound.
  - **Two-regime caveat (NOT the headline):** the 0.73 is largely the
    *solution-completeness gradient* (serious complete solution vs
    fragment/meta/wrong), which is highly articulable. Restricted to
    verified-correct solutions only, base rate jumps to 0.88 and approach-
    identity predictability falls to **0.59** — *which* route a correct solver
    took is weakly predictable. Report 0.73 full-pool with this nuance, never
    a flat 0.59.
  - **Code convergence:** the within-correct stratum runs 0.5-0.6 in BOTH math
    (AoPS 0.59) and code (cc+luogu AC 0.608, luogu 0.507, LC 0.53-0.56); the
    code legs were judged ~94% accepted-only, so they only ever saw this
    stratum — which is why their reported numbers sit ~0.6-0.73 as well.
- **Secondary label — thanks_resid (community PREFERENCE axis, distinct):**
  "thanks" residualized for position/age/views/rank/n-solutions (OOF R2=0.522).
  What the community *rewards*, not what matches the editorial.
  - **V:** correctness saturates; editorial-register 92% learnable but doesn't
    earn thanks; one live deterministic signal is **length** (AUC 0.594).
  - **A:** judge-rated **elegance is the first live A-signal** (within-problem
    rho +0.066, p=5e-11, AUC 0.544; length-independent). Approach-match/novelty
    null (0.500): community rewards execution quality, not the route taken.

## Leg 3 — mathlib4 (correctness fully outsourced to the compiler)

- **Label:** review-friction-among-merged PRs (= "had review threads"),
  19,356 PRs balanced 9,678/9,678 (splits 15,434 / 1,966 / 1,956).
- **V (deterministic lint/build): null — 0.49–0.52** (warnings 0.49, lint
  errors 0.50, rebuild time 0.52). Saturation: only 2.2% of built first-drafts
  carry any warning, genuine lean breakage ~6.9% and era-declining; high- and
  low-friction PRs start equally clean. **Scope note (verifier-flagged):** this
  null is specifically the *artifact* (code-size + lint/build) V — additions/
  deletions/changed_files/size also sit at ~0.51. *Process/timeline* metadata is
  a different story: `days_open` 0.67, `n_commits` 0.70, `n_force_pushes` 0.77,
  full numeric set 0.78. But these are **leakage, not V** — they are downstream
  of the very review activity that *defines* the label (y = `n_review_threads==0`),
  so their predictiveness *confirms* (not contradicts) that friction is a
  process/attention variable, not artifact-determined. All count + timeline
  columns are excluded from any dense feature set; the mathlib dense leg uses the
  PR title only.
- **A (descriptive) — this is the only A we ran for mathlib:** 86K-thread norm
  taxonomy (76,046 classified by an LLM into 18 categories + a V/A/T tag) →
  **vat A 77.3% / V 14.1% / T 5.9%.** This is an LLM judgment of each norm's
  *articulability type*, NOT a prediction of the friction label. Even with
  correctness machine-checked, the articulated review norm is dominated by
  statable-but-not-mechanizable judgment calls (proof style 23%, naming 12%,
  docs 12%, API design…).
- **A (predictive) — NOT RUN; no A-AUC for mathlib.** A norm→friction model is
  circular (the friction label *is* thread-existence; y=1 PRs have 0 threads).
  A proper predictive A would be a diff-judge reading the PR code ex-ante to
  predict friction — but (a) we only have diff *counts*, not patch text, in
  friction_dataset, and (b) the artifact-V on those counts is already null
  (0.49–0.52), suggesting friction is a process/attention variable, not
  artifact-determined. **mathlib is therefore the DESCRIPTIVE-articulability
  leg, not a V-vs-A head-to-head leg** — a different (and weaker) kind of
  evidence for the same thesis than Math.SE/AoPS provide.
- **PIVOT (2026-06-16): friction label retired → accept/reject.** The friction
  label is circular (label = thread-existence) and self-collapsing. Replacement:
  **accept (merged) vs reject (closed-unmerged)** — the reviewers' actual decision,
  non-circular. Built from `pr_reviews_mathlib4.jsonl` (37,249 closed PRs; the
  friction build had discarded all 4,462 unmerged). Cleaned pool (drop WIP/test,
  require engagement, size≥10, drop NONE/outsiders to control the author confound):
  **`accept_reject_dataset.csv.gz`, n=35,796, accept 0.908** (use class weights).
  Validated non-degenerate: title-only TF-IDF AUC 0.62, and **0.623 within a fixed
  author tier (CONTRIBUTOR)** — a genuine *content* signal, not just author identity.
  CONFOUNDS handled: author_association (NONE outsiders 0.28 vs MEMBER 0.94 →
  outsiders dropped/column kept), WIP/abandoned PRs filtered, year drift noted.
  TODO for a real V<A<C: **fetch PR diffs** via `head_oid` (100% present) — title is
  a crippled artifact; the diff is what reviewers judge. Recommend fetching a
  balanced subset (~3.3K rejects + matched accepts) before scaling.
- **RESULTS (2026-06-16, all 35,796 diffs fetched from GitHub): accept/reject ladder
  is FLAT — dense ≈ TF-IDF, no tacit layer.** Fetched real PR diffs (`pr_diffs.jsonl`,
  leanprover-community/mathlib4 via the pulls API; real Lean code, median ~10KB; 99.5%
  usable). Full-data dense-C: Llama-8B LoRA on `title+diff`, 28,424 train,
  clean held-out **eval-split AUC 0.7315** (test-split 0.767, the ~0.035 gap is
  checkpoint-selection optimism).
  - **Apples-to-apples is essential here and it kills any apparent gap.** On the
    *identical* eval split (n=3,602) with the *identical* input (title+diff):
    V (deterministic diff features) **0.620** · TF-IDF title-only 0.698 · diff-only
    0.723 · **TF-IDF title+diff 0.731 ≈ dense 0.7315 (C−TF-IDF = +0.0005).** The
    neural model recovers **nothing** beyond lexical n-grams.
  - A first look (dense 0.767 vs an earlier diff-ONLY TF-IDF CV of 0.664) had *looked*
    like a +0.10 semantic band, but that was two stacked artifacts: (1) dense saw
    title+diff while that baseline saw diff-only (the title alone is worth +0.07);
    (2) this eval split is ~0.06 easier than CV average. Corrected, the **CV-equivalent
    ceiling is TF-IDF title+diff = 0.672**, so dense's true C ≈ 0.67 = TF-IDF, which
    reconciles exactly with the balanced 6.5k run (dense 0.656 ≈ TF-IDF 0.661). The
    balanced run was not under-powered; it was right.
  - V features are sensible (`n_files`/sprawl→reject, focused `n_lean`/`n_theorem`→
    accept). A label bakeoff confirmed accept/reject is the best available y
    (reviewed-only 0.625 worse; changes-requested unusable at 2% base — mathlib
    reviewers comment, don't formally request changes).
  - **Reading:** mathlib is the FLAT leg — V (det. code-metrics) ~0.62 < lexical
    ceiling ~0.67 = dense-C; the modest +0.05 above V is **fully lexical/articulable
    (bag-of-words captures it), with zero tacit residue the dense model exploits**
    (contrast Math.SE dense 0.79 ≫ TF-IDF 0.59). Where correctness is compiler-verified
    and style is thinned into linters, the artifact carries only modest,
    fully-mechanizable signal. A non-circular, sensible, modest-signal leg — a strict
    improvement over the circular friction label, and the clean contrast that completes
    the 3-point shape: **Math.SE big tacit (C−A +0.14) / AoPS thin (+0.05) /
    mathlib flat (C−TF-IDF ≈ 0).** (Scripts: `datasets/math/mathlib/{same_split_ladder,
    ladder_breakdown,cv_ceiling}.py`. The A-judge predictive run is **not** worth doing
    — the V→ceiling gap is lexical, not tacit, so the trigger condition is void.)
- **DE-CONFOUNDING (2026-06-16): the pooled ~0.66 understated a CONFOUND-MUDDIED
  signal; the title's whole contribution was confound.** Top TF-IDF features exposed
  three confounds: (1) the mathlib3→4 PORT ERA — `#align`/`porting` (55/79/32% of
  2022/23/24 diffs, 0% in 2025+), porting PRs accept 0.94 vs 0.90; (2) RECENCY — accept
  drifts 0.95 (2021–23) → 0.88 (2025); `2025`/`copyright 2025`→reject; (3) CHANGE-TYPE
  (title-visible `conv_prefix`) — feat 0.91 / chore 0.94 vs **perf 0.61 / test 0.06 /
  refactor 0.87 / OTHER 0.50**. **Method to drop them:** stratify to a homogeneous
  slice — `conv_prefix=='feat'` × `year>=2025` (post-port) — and read **diff-only** so
  the title can't leak category/recency. **Result (5-fold CV, scripts
  `{confound_diag,deconfound_clean}.py`):**
  | feature set | ALL (pooled) | feat+2025–26 (de-confounded) |
  |---|---|---|
  | V (det. code) | 0.616 | **0.699** |
  | title-only | 0.638 | **0.566** |
  | diff-only | 0.664 | **0.718** |
  | title+diff | 0.672 | 0.720 |
  Two readings: (a) **title-only collapses 0.638→0.566** (near chance at 0.88 base) →
  its pooled value was ENTIRELY confound (change-type + year), not quality. (b)
  De-confounding RAISES the diff signal (pooling across eras/categories added
  heterogeneity that muddied it); within the clean slice **V 0.70 ≈ diff 0.72 ≈
  title+diff 0.72** (all within 0.02) → still deterministic-dominated, no semantic
  layer (dense=diff-TFIDF on full ⇒ dense≈0.72 here too; clean-slice dense run
  optional to fully nail). Surviving features are genuine quality: `sorry`→reject
  (incomplete), idiomatic `simp`/`grind`/`simpa`→accept vs manual
  `apply`/`intro`/`unfold`/`change`/`le_trans`→reject. **Corrected mathlib headline:
  within a de-confounded slice acceptance is ~0.72-predictable and almost entirely from
  deterministic code structure (V) — a STRONGER max-verifiability/flat-leg statement
  than the pooled 0.66.**
- **A-JUDGE (2026-06-16, predictive): A < V — the LLM judge UNDERPERFORMS code metrics.**
  Qwen3.5-122B-FP8 (same judge as the other legs) scored each de-confounded diff on 10
  articulated mathlib review norms (1-5, structured-JSON-constrained, 99.5% parse) plus a
  holistic "will this merge?" P(merge) readout. **Full ladder (de-confounded slice, 5-fold
  CV):** holistic P(merge) **0.587** · articulated-A (10 metrics, LR) **0.586** · **V 0.699**
  · diff-TF-IDF / dense-C **0.718**. So **A (0.59) < V (0.70) < C (0.72)**, with A barely
  above chance and articulated ≈ holistic (the rubric adds nothing over the judge's free
  guess). Robust: identical conclusion across a free-form run (A 0.61, 60% parse, Qwen-FP8
  degeneration: loops + curly-quote JSON) and the structured run (A 0.586, 99.5% parse);
  parse failures were label-unbiased (recovered accept 0.886 ≈ 0.884) so A is not a parse
  artifact, and the clean run gives the *lowest* A (degeneration added noise, not
  suppression). **Reading:** in the max-verifiability anchor, the predictive signal is so
  thin and mechanical (size, `sorry`-completeness, decl counts) that deterministic features
  and bag-of-words capture it directly, while a sophisticated judge reasoning about abstract
  norms (generality, library-fit, idiom) produces scores only weakly aligned with realized
  acceptance — it over-thinks a mechanical decision and loses to feature counts. This INVERTS
  the usual V≤A≤C: mathlib is V>A, no articulability gap AND no tacit layer — the cleanest
  "rubrics buy nothing here" leg. Caveat: this is THIS judge + diff-only input; part of what
  it misses is social/timing context not in the diff. Scripts `a_judge_mathlib.py`
  (StructuredOutputsParams JSON) + `a_auc_mathlib.py`; verdicts `a_metric_verdicts_mathlib.jsonl`
  (raw saved; v1/v2 free-form backups kept).

### CANONICAL clean accept/reject slice (2026-06-25) — `accept_reject_clean.parquet`

The accept/reject numbers above were re-audited for a **size confound** and a **label-noise
confound**; both found and removed. Canonical slice = `accept_reject_deconfounded.parquet`
filtered to `additions>0` AND `additions≤1000` AND NOT (`judgement==0` & `n_review_threads==0`).
**n=7956, base accept 0.943.**

- **SIZE confound:** the parquet's `size` column is BUGGY (all large PRs capped at 17) —
  use `additions`. On real `additions`, size→reject is a WEAK, mostly-process signal.
  Cutting mega-PRs (`additions>1000`, n=92: accept 0.065, 71% zero-review — abnormal
  abandoned mega-imports) and dropping empty diffs (`additions==0`, n=12) is hygiene.
  **DO NOT exclude small PRs** — small real PRs (≤10 additions, 14.5% of data, 91% accept)
  are legitimate focused lemmas/instances, not degenerate (verified by hand: `Prod.isUnit_iff`,
  `isTopologicalBasis_empty`, etc.).
- **LABEL-NOISE confound (a.k.a. "abandonment"):** 504 of 985 rejects (51%) are
  `n_review_threads==0` — **closed-unmerged with ZERO review engagement** (100% have zero
  threads). Their labels show process, not quality, closures: merge-conflict 30%,
  blocked-by-other-PR 29%, WIP 20%, migrated-to-fork 15%, large-import 15%, awaiting-author 6%.
  These are stale/blocked/unfinished/relocated PRs swept up — no maintainer read the code.
  They are NOT quality rejections and are dropped. (No-review *accepts* are kept — easy
  frictionless merges are genuine positive outcomes; the noise is one-sided.)
- **Resulting clean ladder (5-fold stratified CV on the diff):**

  | slice | n | base | size→rej | V | A | C(TF-IDF) |
  |---|---|---|---|---|---|---|
  | FULL (reference) | 8494 | 0.884 | 0.670 | 0.707 | 0.585 | 0.776 |
  | **canonical clean** | **7956** | **0.943** | **0.618** | **0.653** | **0.555** | **0.735** |
  | reviewed-only (alt) | 5939 | 0.923 | 0.566 | 0.621 | 0.549 | 0.708 |

  Ordering **C > V > A** is stable across all cleaning. Size→reject collapses 0.670→0.618
  (most of "bigger→rejected" was non-engagement closures, not scrutiny). Use this slice as the
  canonical mathlib accept/reject population. Scripts: `finalize_slice.py`, `clean_ladder.py`,
  `what_is_abandon.py`. NOTE: the **style/revision leg** (`revision_pairs.jsonl` V/A/C) remains
  RETRACTED — its `first_push_excerpt`-based "SIZE" was a label leak (header fallback ⟺
  `is_outdated`); see `style_signal_confounds_note.md`. Only the accept/reject leg is clean.

### DE-CONFOUNDED remeasure (2026-06-25) — `accept_reject_clean_deconf.parquet`

Top-TF-IDF audit of the clean slice found the V→C gap is NOT articulable quality — it decomposes
into three buckets: (1) **tactic-idiom** (articulable AND mechanically-checkable: grind/aesop/
simp/fun_prop/refine/rwa → accept; intro/apply/have/unfold/eq/congrarg → reject; the *aggregate*
automation-ratio is 0.497/chance — only the *fine-grained per-tactic* profile carries signal,
0.613 standalone); (2) **author identity** (contributor names in copyright headers, e.g.
`joël riou` — confound; dense model will overfit); (3) **topic** (top-level Mathlib area:
GroupTheory/RingTheory harder ~0.90 accept, Probability/AlgebraicGeometry easier ~0.98 — weak
0.547 social/process confound, NOT quality). After stripping C of tactic+author+topic the
residual ≈ V (0.635) → no articulable-but-not-mechanical band exists for A to own.

**Three controls applied** (artifact `accept_reject_clean_deconf.parquet`: +`diff_noauth`,
+`area`, +`tac_*` tactic counts):
1. **Strip author/copyright text** (`diff_noauth`) — kills the dense-overfit surface at zero cost
   (C 0.742→0.750). Do NOT balance by author (shreds data); strip the text.
2. **Control topic** via residualization (area one-hot partialled out). Safe: V is
   topic-independent (V resid-of-topic 0.649→0.635; V+topic 0.644 ≤ V). Topic is confound not
   quality → control, don't make an A metric.
3. **Promote tactic-idiom into V → V′** (V + 33 per-tactic counts).

**Final de-confounded ladder (train→eval, topic-residualized):**

| model | raw | topic-resid |
|---|---|---|
| V (orig) | 0.649 | 0.635 |
| **V′ (V + tactic-idiom)** | **0.683** | **0.680** |
| C (author-stripped TF-IDF) | 0.750 | **0.736** |
| A (m01–10) | ~0.46–0.56 (chance, split-noisy) | ~0.48 |
| A + V′ | 0.668 | 0.665 (A redundant) |

**Net:** mathlib accept/reject = **V′ 0.68 < C 0.736**, A redundant — the flat / max-verifiability
leg, now with author + topic confounds controlled and the one real signal (tactic-idiom) in V.
Residual V′→C gap (0.056) is fine-grained lexical (API/lemma tokens), not a recoverable A band.
Proposed tactic sub-rubric for A (#1 automation-appropriateness, #2 tactic-modernity, #3
manual-proof-smell, #4 decomposition-quality, #5 golf-vs-readability) is largely collapsible to
the deterministic tactic counts now in V′, so it would lift A toward V′, not beyond. Scripts:
`mathlib_top_tfidf.py`, `mathlib_tactic_decomp2.py`, `mathlib_authorstrip_topic.py`,
`mathlib_remeasure2.py`, `save_deconf.py`.

---

## Data-quality audit of the A & V pipelines (2026-06-16)

Manual spot-check of the judge inputs/outputs and feature computations, looking
for truncated context, parse errors, degeneracy, and miscomputation. Verdict:
**no headline-changing problems; a few documented quirks; one inert bug.**

- **Truncation is negligible everywhere.** Math.SE A-judge char-cap 9000 sits
  above answer p99 (5,810) → only **0.19%** of judged answers truncated (37/19,410),
  with negligible score effect. AoPS A-judge forum-cap 4,000 → only **1.97%** of
  forum solutions truncated; the wiki-solution cap (6) fires on 13% but does NOT
  bias the label (capped same_approach 0.76 > uncapped 0.68 — many-editorial
  problems simply have more match targets). The truncation worry did not materialise.
- **Parse failures are genuine model degeneration, correctly excluded.** Math.SE A
  13.8% judge_failed / AoPS A 13.2% / mathlib-A 11.6% unclassified — all are real
  Qwen-FP8 degeneration (output loops `.\n\n.\n\n.`, garbage tokens, or schema-
  violation like `{"rating":5,"comment":…}` instead of a01–a14). Selection-unbiased
  (Math.SE fail base 0.506 ≈ OK 0.509), so dropping them does not bias the measure.
- **Salvaged verdicts dilute A — FIXED.** 17.4% (Math.SE) / 13.5% (AoPS) of OK
  rows were recovered by field-wise regex when the JSON didn't close; these are
  noisier (single-ascore→y AUC 0.610 vs 0.649 clean). **Fix applied:** salvaged rows
  excluded → canonical clean verdicts written to `a_metric_verdicts_clean.jsonl`
  (16,032 rows). **Canonical de-noised A = 0.669 eval / 0.654 test** (was 0.661/0.638).
  Re-verified apples-to-apples on these clean rows: **C−A = +0.128 eval / +0.140 test**
  (C-on-clean-A-subset 0.796/0.793 vs clean-A 0.669/0.654) — gap narrows slightly but
  V<A<C holds firmly.
- **Scores are not degenerate.** Math.SE a01–a14 use the full 1–5 range (std
  1.2–1.5). AoPS `elegance` is compressed (44% mass on "4"), which caps its
  discriminative power — consistent with the thin +0.05 elegance signal.
- **One inert bug — FIXED.** Math.SE V feature `lint_error` was 100% NaN (empty
  column). Not load-bearing (V-lint AUC 0.5702 with or without it — the imputer
  skipped it). **Fix applied:** dropped from `mathse_lint_features.csv` (30→29 cols,
  `.bak` kept); V unchanged. The sparse features `n_arith_wrong` (0.4% nonzero) and
  `n_misspelled_theorem_mentions` (0.1%) are rare-event detectors, kept as-is.

Net (corrections applied): the **V 0.55 < A 0.669/0.654 < C 0.79** ladder holds, with
**C−A = +0.13–0.14** on identical clean rows.

---

## Cross-leg verdict

All AUCs are read against the chance floor 0.5 (AUC is base-rate-invariant).
The Math.SE 0.461 is a *learned* question-only baseline AUC (predict-before-
seeing-the-answer), NOT a prevalence; the AoPS 0.70 is a *base rate* (class
prevalence), NOT a floor — keep these distinct.

| leg | label | base rate | V (deterministic) | A (judge) | C (dense) | T |
|---|---|---|---|---|---|---|
| Math.SE | answer quality | 0.50 (floor 0.461) | 0.55–0.58 | **0.661 eval / 0.638 test** (+0.12/+0.10 over V) | **0.794** (+0.14–0.16 over A, same rows) | residual above C |
| AoPS | same-approach vs editorial (PRIMARY) | 0.70 (floor 0.5) | 0.64–0.71 lexicons | **~0.73** vocab ceiling (0.59 within-correct, caveat) | **0.777** (+0.05 over TF-IDF) | — |
| AoPS | thanks (preference, secondary) | within-problem | length 0.594 | elegance +0.05 AUC | (not run for thanks) | large |
| mathlib | review friction | 0.50 | null 0.49–0.52 | A77/V14/T6 (descriptive) | **0.617** title-only ≈ title-TFIDF | — |

**C completes the ladder V≤A≤C on the two head-to-head legs.** The shape differs by
leg: Math.SE has a LARGE dense-only band (C−A≈+0.15) — most of the recoverable signal
is tacit/semantic, above both code and the articulable judge; AoPS has a thin one
(C−A≈+0.05) because its label is lexical; mathlib's artifact band is ~flat near the
floor (friction isn't in the artifact). The Math.SE ordering 0.46 (floor) < 0.55 (V) <
0.65 (A) < 0.79 (C) is the cleanest single illustration of the decomposition.

*AoPS same-approach base rate (0.70) is driven by: (a) mechanical — the label
matches ANY of up to 6 editorials, so rate climbs 0.59 (1 shown) → 0.80
(5 shown); (b) real — canonical-technique prevalence, 0.88 for verified-correct
and 0.81 AMC8 vs 0.57 IMO (hard problems admit more approaches); (c) non-
solutions (6%) match at 0.003. Real-solution-only rate is 0.739.*

**The math vertical confirms the project thesis on its hardest case:** where a
community has strong correctness norms (or a compiler), V saturates and stops
discriminating preference; what experts reward and articulate lives in the
A-band (elegance, directness, exposition, the *kind* of norm invoked), with a
real T residual that no measurable layer closes. C (dense) ceilings remain
parked pending the locked datasets.

**The single sharpest number** is Math.SE: a deterministic-V ceiling of ~0.55
and an LLM-judged A-layer of ~0.65 on the SAME held-out answers, both splits —
a **+0.10–0.12 articulability gap that the mechanical lexicon twins do not
explain** (judged "honesty"≠hedging-count, judged "gaps"≠clearly-count). That
gap *is* the articulable layer the decomposition predicts: real, statable,
expert-judgeable quality sitting above what code can mechanize and below the
taste residual.

## Status: CLOSED (all splits)

- Math.SE: V 0.55–0.58, **A ~0.64–0.66, gap +0.10–0.12.** Verified across all
  three splits: combiner fit on TRAIN → eval 0.655 / test 0.631; within-split
  CV eval 0.661 / test 0.638; cross-split fits agree. ✓
- AoPS: editorial-similarity ~0.73 (within-correct 0.59 caveat); thanks =
  preference axis, elegance the live A-signal ✓
- mathlib: V null 0.49–0.52; A descriptive vat A77/V14/T6 ✓
- **Independent adversarial re-verification PASSED** (2026-06-14): every headline
  AUC reproduced to 3–4 digits and the +0.12 gap survived a best-effort strong-V
  attack (deterministic V plateaus ~0.59) — see Verification & dense status above.
- **C (dense) ceilings UNPARKED + running** (Llama-3.1-8B LoRA, sk3 GPU 3,
  sequential): Math.SE on the identical held-out rows as A/V (frac=1.0 first, then
  scaling curve); mathlib title-only; AoPS full pool + verified-correct regime.
  C-vs-A is read on the A-scored answer_ids for an apples-to-apples C≥A check.
  **Full train-split A-scoring is NOT needed** — the A-combiner is a 15-param LR
  that ~2.7K rows fit identically; per-metric AUCs are training-free.
