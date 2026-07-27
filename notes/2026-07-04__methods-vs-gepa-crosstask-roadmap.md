# ROADMAP: Cross-task evaluation of certificate methods vs GEPA — recovery-optimized, silver-validated

**User directive (2026-07-04, verbatim intent):** "nail down and fully expand a cross task full evaluation
of our methods vs. GEPA: recovery-bound optimized, but silver-label validated at the end. Across all tasks
that we have silver-data for. All comparisons fully valid, fully trustworthy/apples-to-apples and full."
Reconstruction-only remains the dominant paradigm; silver-label correlation is used ONLY as external
validity for this section (user re-confirmed 2026-07-04).

## 0. The design in one sentence

Both arms are optimized/derived **label-free on the reconstruction axis** (GEPA: prompt-form search;
ours: Ω-checklist certificate OPT_Ω / g1 / T), then both are **evaluated on the identical silver axis**:
per-metric correlation with human norm salience (CE-mapped silver + human gold where available), under
identical controls. The CW pilot (2026-07-04) shows why this matters: GEPA-attained recovery carries ~no
silver signal (desc_R gold ρ +0.01), the certificate side carries all of it (OPT gold ρ +0.27, partial
+0.17; paired-12 gold: ρ(OPT)=+0.59 vs ρ(GEPA*)=−0.00).

## 1. Inventory (sk3 scan 2026-07-04) — what "all tasks with silver data" actually is

26 dirs in bge_pertask; 24 are silver corpora (drop `checkpoints`, `r1_pooled`). **KEY REFRAME: silver
corpora map MANY-TO-ONE onto hierarchy tasks.** The metric universe + certificate live at the TASK level;
each corpus is a silver SOURCE re-matched onto that task's catalog. So the expensive side (certs) is ~5/8
done, and most of the 24 unlock via CE re-matching alone.

| hierarchy task (cert) | cert status | silver corpora (joined lines) | gold |
|---|---|---|---|
| creative-writing | ✅ done (46-metric R3; full-368 R2 optional) | creative_writing (2.9K), litbench_rationales (20K), wp_comments (2.9K, noise caveat) | ✅ |
| humor | ✅ done (265 R2) | humor (20K), humor_multi (20K) | ✅ |
| press-releases | ✅ done (208 R2) | press_releases (11.8K) | ✅ |
| code-review | 🔄 running (133 R2) | code_review (20K) | ✅ |
| math(-stackexchange) | 🔄 running (141 R2) | math (3.1K), math_se (3.1K), aops_forum (6.2K), competition_editorials (4.2K), mathlib (1.4K) | ✅ (math) |
| peer-review | ❌ NEEDED (~221 R2 groups) | peer_review (20K) | – |
| notice-and-comment | ❌ NEEDED | nc_public_comments (13.1K), notice_and_comment (1.0K) | – |
| legal-outcome-prediction | ❌ NEEDED (iff law in scope) | bva (13.6K), cavc (16.8K), courtlistener (20K), dol_arb (1.0K), law_se (2.6K), legaladvice_uk (20K), nlrb (2.9K), ptab_fwd (3.3K), reddit_supremecourt (8.5K), ttab (2.1K) | – |
| ??? | unmapped | crse (20K) — identify owner task first | – |

Hierarchies with NO silver corpus (excluded): grant-funding, news-homepages, patents.

## 2. What is actually holding us up (blocker classes)

- **B1 — CE re-match (THE critical path).** 19 corpora have cat=200 GENERIC PLACEHOLDER catalogs, so
  their existing top10s live in a meaningless label space. Fix: build per-task catalog from
  `<task>_general_r2_expanded.json` merged names(+descriptions); re-run the retrieval cascade
  (`bge_pertask/match_cascade.py` + trained CE in `bge_pertask/checkpoints/`) norm→catalog; emit
  matches_joined format directly (kills the positional-join trap at the source). GPU-light: ~hours for
  all 19 on one GPU. **Calibration gate: re-match `humor` with this pipeline and require it to reproduce
  the current humor result within noise before trusting any new corpus** (apples-to-apples anchor).
- **B2 — 2-3 new certificates.** peer-review, notice-and-comment (+legal if in scope): standard
  alpha_probe R2 sweep + chain (playbook proven 4×; ~4-10 GPU-h each, 1 GPU, stackable).
- **B3 — GEPA arm beyond CW.** Bank-wide name/desc rung scoring per task (generalize
  `score_name_rungs.py` beyond CW; 1 vLLM pass ≈ 1-2h/task) + GEPA* on a stratified 12-metric subsample
  per task to validate the desc_R≈GEPA* proxy (V7). Reviser budget: ~120 GLM calls/task — check quota;
  fallback local Qwen reviser (generation-side only, not eval — policy-compatible).
- **B4 — gold coverage.** Only 5 tasks have gold. CE-only tasks inherit the CW lesson (CE noise can mask
  real signal) → lead with gold tasks; weight CE-only correlations by CE recall in the meta-analysis.
- **B5 — known silver-quality hazards.** wp_comments label noise (memory); PR-style
  descriptive-not-evaluative norms and mass concentration — run the §V8 data-character audit per corpus
  BEFORE interpreting its correlation.

## 3. Validity requirements (the "fully trustworthy" contract — all MUST hold)

- **V1 same instrument:** Llama-3.1-8B executor everywhere; n_probes=300; probe split texts[60:360];
  orbit-target 4. No cross-executor pooling (same-family policy).
- **V2 same information:** GEPA seeded by metric description; Ω built from the same description +
  children. Neither arm sees anything the other doesn't.
- **V3 frozen universe:** per task, metric set = cert-scored ∩ catalog-joined, frozen BEFORE any silver
  number is computed. No post-hoc metric selection.
- **V4 same silver pipeline both arms:** identical matches_joined; identical salience (top-1/3/10 +
  gold); identical controls — perm null (1000), partial|log-size+H_M (headline), split-half reliability +
  attenuation ceiling, mass-coverage + capture-recapture, channel-agreement ρ(CE,gold).
- **V5 compute honesty:** report LLM-call budget per arm; correlations don't require call-parity but the
  writeup states both raw and per-call readings.
- **V6 stable splits:** hash-based item splits; same probe items for all rungs/arms within task.
- **V7 pre-registered proxy rule:** desc_R stands in for GEPA* bank-wide ONLY if, on that task's
  12-metric stratified GEPA subsample, median |GEPA*−desc_R| < 0.03 bits (CW: passed, 7/12 exactly 0).
  Otherwise run GEPA per-metric on that task.
- **V8 dataset-first for every corpus:** before scoring, spot-check ≥8 norm→top3 mappings by hand;
  report evaluative-vs-descriptive character; top-5 mass concentration; nonzero-salience fraction.
  (The PR audit is the template.)
- **V9 label hygiene:** labels appear ONLY in the optional Sorensen-strict replication (§5b), evaluate-only,
  never in optimization or selection. Metrics never label-aware (standing directive).
- **V10 report discipline:** size-partial + gold is the headline everywhere; raw ρ reported alongside;
  descriptive language (no verdicts while tasks are still landing).

## 4. Phases and compute

- **P0 — re-match wave** (1 GPU-day incl. validation): humor calibration gate → 19 corpora re-matched →
  V8 audits. Deliverable: matches_joined_v2 per corpus.
- **P1 — new certs** (2-3 GPU-days, 1-2 GPUs): peer-review, notice-and-comment (+legal decision point:
  user call on scope). Chains as-is.
- **P2 — GEPA arm** (1-2 GPU-days + GLM quota check): rung scoring all cert tasks; GEPA-12 per task (V7).
- **P3 — unified evaluation** (CPU): per task × corpus: ρ/partial for {name_R, desc_R, GEPA*, g1, OPT_Ω,
  T} × {CE top-1/3/10, gold}; per-corpus V8 audit block; meta-analysis across tasks (random-effects,
  weight by CE recall + n_metrics); the head-to-head "methods vs GEPA on silver axis" table is the paper
  figure.
- **P4 (optional) — Sorensen-strict replication** (≈5 GPU-h): see §5b.
- **P5 — notebook + writeup** (extends 2026-07-04 multi-task notebook).

Total new GPU: ≈4-6 GPU-days sequential on 1-2 GPUs (excl. legal: +1-2).

## 5. The two Sorensen follow-ups (user Q2c)

**(a) Range restriction — quantify, don't hand-wave.** Sorensen's own Fig 4 shows the mechanism: tasks
where all templates are low-signal (WiC, COPA — no template beats chance) have MI↔accuracy r ≈ 0.33/0.62
on 175B and ≈0/negative below; his high-r datasets are exactly those with large template dynamic range.
Our within-task OPT SD is ~0.14-0.18 bits vs full articulation span (name_R→T) ~0.4-0.9 bits. Analysis:
report Thorndike Case-2 corrected ρ̂ using the pooled-operationalization SD as the unrestricted reference,
alongside raw — as an illustration of what x-range costs us, clearly labeled as a what-if.

**(b) Strict Sorensen analog — per-FORM within metric (his design transposed).** His unit: template
(K=20 wordings of one task); x = label-free MI; y = accuracy of that template. Ours: FORM RUNG of one
metric (generic → name → description → GEPA rounds → g1-checklist → OPT-checklist; K≈6-10 forms);
x = label-free recovery R_form (have it); y = that form's verdicts scored against the v2 `judgement`
labels (AUC/MI), **evaluate-only**, on the 5 labeled tasks. Per-metric correlation across forms, report
the distribution over metrics (his Fig 4 analog). Cost: score K forms × ~1-2K labeled items × 5 tasks =
one vLLM pass per task ≈ 5 GPU-h total. Silver CANNOT supply per-form y (salience is metric-level, form-
invariant by construction) — this is why the strict replication needs labels and stays evaluate-only.

## 6. Sorensen positioning corrections (verified from PDF 2026-07-04)

Template = prompt templatizing function (wording/scaffold of the SAME task); K=20/dataset, N=500 items,
8 datasets, 8 models (GPT-2 124M → GPT-3 175B). Correlation = Pearson(MI_θ, acc_θ) across the 20
templates. **175B column: 0.92 SQuAD, 0.86 LAMBADA, 0.71 IMDB, 0.70 BoolQ, 0.68 ROCStories, 0.62 COPA,
0.56 CoQA, 0.33 WiC** (earlier "0.68-0.96" note was wrong). Small models: frequently ~0/negative. No
confound controls or reliability analyses. Effect-size band citations for our cross-construct setting:
Gignac & Szodorai 2016 (median meta-analytic r=0.19 across 708 correlations; 0.10/0.20/0.30 =
small/typical/large); Hemphill 2003; Funder & Ozer 2019.

## 7. Decision points — RESOLVED 2026-07-04 (user)

1. **Law: EXCLUDED** (original scope stands). Legal cert not run; 10 legal corpora out.
2. **GEPA reviser: GLM-4.7 (user 2026-07-04).** Two z.ai keys on sk3 = two quotas (toggle with `glmkey`
   k1/k2 when one exhausts). GLM-4.7 preferred over 5.2 (cheaper on quota; proven in GEPA-clustering
   roles; reviser is generation-side so weaker reviser costs only search quality, never validity).
   Apples-to-apples note: the CW GEPA-12 sweep used the 5.x reviser — REDO CW's 12 with GLM-4.7 during
   P2 (~120 calls) so every task shares one reviser.
3. **crse corpus:** still to identify owner task or drop (investigate during P0).
4. **CW at R2: RE-RUN APPROVED.** Queued on sk3 (`queue_cw_r2.sh`, PID 2624033): waits for GPU 3/4 to
   free from code_review/math, then alpha_probe 368 R2 metrics + chain_silver → replaces the R3* row.

## P/R cross-firing analysis (task #123) — attempt 1 post-mortem (2026-07-05)

Design: V[doc,metric] (M_ω verdicts over silver docs) vs A[doc,metric] (silver assignments); own-AUC,
specificity rank, micro P/R@k, corr with OPT_Ω. STATUS: **BLOCKED on doc linkage**, not on compute.

Validated facts (spot-checks, not assumptions):
1. `matches_joined_<task>.jsonl` has NO document linkage — `doc` and `row` are line indices (1 per norm).
2. TRUE join recovered: `matches_joined[k] == signals_<task>.jsonl[k+1]` (offset +1; 500/500 agree;
   all mj norms ∈ signals set). `signals` carries `i` = source-ITEM index; humor: 20,000 norms over
   20,000 items... i.e. ~1 norm/item in signals — and gold `id` shares the same 0..19999 space.
3. The item list behind `i` is NOT the modeling corpus: lexical alignment vs
   reddit_humor_modeling(_dedup) = 0.3-0.5% aligned, equal to shifted BASELINE (test: rare-token
   overlap, 600 items). data/humor sources = comedy-DISCUSSION scrapes (aspecialthing, standup forums);
   data/humor/extracted = 150-row QC sample only. code_review signals: same schema, same unknown list.
4. DESIGN caveat surfaced: for humor the silver "docs" are meta-discussion threads, not jokes — the
   P/R question must decide whether metrics should fire on the THREAD or the discussed OBJECT.
   code_review (comments about code artifacts) is the better-posed first testbed.

Recovery paths (in order): (a) find the writer of signals_*.jsonl (the norm-extraction script names its
input file = the item list) — first grep next session; (b) ask the silver-pipeline owner thread for the
i→item mapping; (c) regenerate norm→item from raw scrapes by substring-matching norms into comments
(bounded-effort fallback; norms are ellipsized quotes, partial-fragment matching needed).
GPU cost so far ≈ 0 (the aborted run scored an empty doc list). pr_crossfire.py analysis-section bugs
(empty-safe percentiles) also need the one-line fix before rerun.

## P/R doc-linkage RESOLVED + cert expansion (2026-07-05 pm)

**Doc-linkage blocker is dead — routed around, not recovered.** The orphaned `bge_pertask/signals_*.jsonl`
index (source corpus replaced, untraceable) is abandoned. The **GEPA+Gemma pipeline** makes it moot:
`data/<task>/input.jsonl` (items {unit_id,text}) and `data/<task>/gepa/deploy_round1_full.jsonl`
(Gemma norms {unit_id, signals:[{signal_text}]}) are BOTH keyed by `unit_id` — item↔norm is native.
Gemma signal_texts are verbatim substrings of input.jsonl. New driver:
`methods/metric_implementer/experiments/gepa_pr_crossfire.py` (two GPU-safe phases: `assign` = BGE
bi-encoder matches Gemma norms→R2 metrics → A[item,metric]; `score` = vLLM M_ω over the same items → V;
joins OPT_Ω cert + CUF unit census; own-AUC / specificity / micro-P/R@k / corr with OPT_Ω AND n_units).
15 CPU unit tests pass (loaders, AUC, empty-safe percentiles/rho, CUF schema).

**code_review assign face-valid:** 500 items (pool=88,734 with ≥2 norms), mean 17.1 Gemma norms/item →
mean **9.6 silver metrics/item**. Spot-checks sensible (spacing nit / `@deprecated` tag → style metric).
CAVEAT: BGE bi-encoder shows a generic-attractor bias (3/3 argmax = "Match existing local style"); the
CE reranker (`cross_encoder_llama8b`) is the sharpening upgrade — the specificity metric will quantify it.

**Scope reality — only `code_review` aligns cert + GEPA-items + matcher + hierarchy under one metric
space.** The other 4 OPT_Ω certs (CW/humor/math/PR) use the E7 *manifest* corpus (seed=7, `_load_texts`),
NOT input.jsonl, and their GEPA-item siblings live under different names (litbench, math_se, wp_comments…)
with different catalogs. So the joined per-text P/R runs cleanly on code_review now; extending to the
other 4 needs a norm-extraction pass over their cert corpus (P1-adjacent).

**Cert expansion LAUNCHED (user: "expand this [5 certs]... using the unit-level machinery").** OPT_Ω
certs building on GPU5 (`outputs/silver_r2/cert_expand.sh`, sequential, run_alpha_probe R2 → value_cert):
**peer-review, patents, news-homepages, notice-and-comment** — exactly the 4 tasks the CUF unit bank
ALREADY covers (@8B) but that lacked OPT_Ω. Doubles cert coverage 5→9; each of the 9 then has BOTH
OPT_Ω + CUF units. (notice-and-comment texts ~66 chars → degenerate-M risk, flagged.)

**CUF coverage summary (`outputs/unit_cert/cuf_coverage_summary.json`) — the expanded certified landscape,
9 tasks:** units/metric 1.2–1.75, dead-weight 2–9%, ATOM 81–91%, free-arm detect 63–89%, M-arm 65–86%.
**corr(n_units, OPT_Ω) ≈ 0 everywhere** (−0.10 math … +0.17 press-releases) — replicates the clean
deconfound (unit COUNT ≠ certificate VALUE) on a 9-task table.

**code_review P/R first-pass (BGE matcher, `outputs/pr_crossfire/pr_code_review.json`):** own-AUC mean
0.49 (≈chance; q10/50/90 = .35/.53/.61), specificity 0.55, only 17% of metrics own-best-decile; micro
P@5 0.12 / R@5 0.06. corr(OPT_Ω, ownAUC) = −0.45 and corr(OPT_Ω, spec) = −0.51 BUT on n=19 cert metrics
(underpowered — do NOT trust sign); corr(n_units, ownAUC) = +0.17. **DOMINANT CONFOUND: only 41/133
metrics scorable** — the BGE bi-encoder concentrates silver assignments onto a few generic attractors
(e.g. "Match existing local style"), starving the other 92 (<3 items → NaN AUC). So chance-level own-AUC
is partly matcher artifact. FIX RUNNING: CE-rerank via `cross_encoder_llama8b` (GPU6, same target — safe
without sign-off) → `pr_code_review_ce.json`; should spread assignments and raise n_scored/power.
OPEN (needs user sign-off — target change): V=M_ω scores "text SATISFIES norm m" while silver = "reviewer
INVOKED m"; for the review-comment corpus these differ (the thread-vs-object issue), so item-level
own-AUC≈chance may be partly a real target mismatch, not only matcher noise.

## Polarity: invoked-when-violated (2026-07-05 night, user engaged)

**Counts (FULL, Gemma deploy — trusted):** code_review 1,499,724 signals: **90% negative / 8% positive**
(signal_type: suggestion 66%, complaint 21%, observation 11%, praise 2.6%); crse 107,701: 66% neg / 21%
pos / 13% neu. (metric,polarity) sample table → `outputs/pr_crossfire/polarity_metric_counts_code_review
.json`. KEY: neg-share across code_review metrics = **0.93 ± 0.04** (range .83–1.00) — polarity is a
property of the GENRE, not metric-differentiating there; **crse is the polarity-differentiated testbed**.
Coverage map: Gemma polarity = 2 tasks (code_review, crse); Qwen extractions carry polarity (+ an inline
`rubric_matches` field — near-free (task,metric,polarity) counts at Qwen trust tier) for ~9 more wave-2
tasks; old bge_pertask silver norms have NO polarity; anchors_best_full has NO polarity.

**First signed test (`_signed_auc.py` → signed_auc_code_review.json):** split A by polarity, AUC existing
V_sat against each arm. AUC(V,A_neg) mean **0.479** (q10 **0.314** — fat inverse tail) vs AUC(V,A_pos)
mean **0.555** (q50 .552). Paired (n=14 metrics with both): **+0.029, Wilcoxon p=0.50** — direction
right, underpowered. Showcase: "Match existing local style" AUC_neg **0.314** / AUC_pos **0.595** (Δ=.28
on the largest-mass metric) = textbook invoked-when-violated. Power fixes queued: CE-quality A, full
corpus (703 pos signals in the 500-sample is the binding constraint), crse replication. CAVEAT: V is
scored on the THREAD (which contains the complaints), so part of the inverse relation may be the
executor lexically reading complaints rather than evaluating the object — the thread-vs-object V remains
the deeper design question (user sign-off).

## Overnight harvest 2026-07-06 am

**U2 (rewording-invariance) ANSWERED on code-review** (77/133 metrics; vLLM engine died 23:53, rest
resumed; run_bank patched to abort-on-EngineDead): r_self median **0.828** (only 0.7% of L1 units <0.5),
ε_id median **0.015** bits (q90 .043, max .135), pooled r\*=0.61, **zero verdict flips** (135/135
certified survive), certified_lo −0.023 mean correction. ⇒ census's implicit ε_id=0 was benign; units ARE
stable under strict paraphrase at 8B. → u2_harvest_code_review.json.

**CE-reranked P/R (code_review, `pr_code_review_ce.json`):** the trusted matcher changes the picture —
n_scored 41→**67**, silver/item 9.6→**17.5**, own-AUC 0.49→**0.54** (q90 .67), specificity 0.55→**0.64**,
own-best-decile 17%→**34%**, and **corr(OPT_Ω, own-AUC) flips −0.45→+0.23 (n=40)**, corr(OPT_Ω, spec)
+0.26. The BGE-attractor artifact drove the earlier negative sign. Micro-P@k still ~0.10-0.16 (satisfies-
vs-invoked target mismatch + polarity mixing still uncorrected in this readout).

**Cert-expansion:** peer-review CERT DONE (88 metrics), patents CERT DONE (7 — thin hierarchy),
news-homepages running, notice-and-comment queued. Cert coverage 7/9 → 9/9 later today.

**Polarity instrument (task #124):** TF-IDF+LR FAILED gate (0.70 acc/0.46 mF1); encoder-v2 WORSE
(mF1 0.24 — diagnostic: the norm→metric matcher encoder is trained to be POLARITY-INVARIANT, collapsing
complaint/praise of the same norm). v3 = LLM forced-choice (two probes: problem?/praise? → 3-way rule),
validating now on 20k held-out Gemma labels before any sweep.

## Polarity profile LANDED (2026-07-06 pm, 21 corpora × uniform v4 instrument)

`outputs/polarity_uniform/profile_draft.{md,json}`; 40k-sampled signals/task, LLM+context probes
(binary AUC .89/.91), calibrated hard labels; quantification-corrected binary shares as secondary
(sign-rule tpr .93 / fpr .36 — corrections saturate at extremes, prefer hard-label shares).

**User's hypothesis (community more complimentary than expert-verdict): NOT supported as posed** —
group means: community 48% neg / 19% pos; expert-feedback 43/39; institutional-verdict 43/10;
expert-revealed(PR) 32/3. **The organizing axis is DISCOURSE FUNCTION, not community-vs-expert:**
praise-rich = comparative-evaluation contexts (litbench_rationales 69% pos — rationales praise the
winner by construction; competition_editorials 52%; math_se 41%, aops 38%) vs problem-driven contexts
≈0 praise regardless of community/expert (legaladvice_uk 2%, law_se 4%, ptab 2%, ttab 2%) with
advice-seeking communities the MOST negative. Also neutral% is itself meaningful (ptab 83% neutral =
descriptive/factual register). CAVEATS: levels are instrument-relative (uniform instrument reads
code_review 60/16 vs Gemma-native 90/8 — Gemma's suggestion→negative convention is stricter);
cross-task COMPARISONS are the valid readout; anchors = judge-passed signals; taxonomy mapping is
DRAFT (litbench arguably its own "comparative-evaluation" group).

## Semantic-specificity + dual-polarity results (2026-07-06 pm)

DOC-OF-RECORD: `notebooks/2026-07-06__unit-semantics-and-polarity-findings.ipynb` (all figures/tables).
(1) **Semantic-specificity:** δ^M real +0.140 vs scramble +0.013 vs off-metric −0.006 (p<1e-5), δ^free
identical across arms (~0.16) — meaning is the causal channel of unithood; recalibrate δ_min^M to
scramble-q95. (2) **Dual-polarity:** corr(M⁺,M⁻)=+0.57 (acquiescence — independent YES-probes measure
applicability, not valence); M⁺ wins on praise-silver (p=.005), M⁻ fails on complaint-silver (p=.52);
next = contrastive readout M⁺−M⁻ (Vp/Vm saved in dual_polarity_code_review.npz).

**Instrument gap now precise:** M_ω = P(YES|"satisfies m") — satisfaction-valence only; conflates
violated with irrelevant; NO valence/applicability separation anywhere in the MI pipeline. Silver
salience = invocation ≈ breach record (Garfinkel: norms are background expectancies, articulated when
breached). PROPOSAL (analysis layer only, core certificate untouched per reconstruction-only): add V_app
("text engages/makes-relevant m") beside V_sat in gepa_pr_crossfire; test composition A_neg ~
V_app·(1−V_sat), A_pos ~ V_app·V_sat; report polarity-resolved silver-salience correlations in the
Sorensen-style validation (OPT_Ω may track breach-salience, not raw salience).
