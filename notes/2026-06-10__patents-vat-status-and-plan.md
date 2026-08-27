# Patents VAT: Status Map + Path Forward (2026-06-10)

## The vision (restated)

1. **End goal**: predict patent accept/reject under the VAT decomposition — across
   ALL rejection classes, not just prior-art ones. OARD gives per-app flags for
   §101, §102, §103, §112(a/b/d/f), and double patenting (2.19M apps), so every
   eval slices by rejection class.
2. **V (verifiable)**: §102/§103 rejections are prior-art based → claim-matching against
   cited art is a concrete verification procedure. These are the only classes where we
   can build literal verification; that's why they get the retriever machinery.
3. **A (articulable)**: all remaining rejection classes (§101 eligibility, §112
   clarity/enablement, double patenting, formalities, style) → rubric metrics.
   The hypothesis: articulated metrics capture the bulk of these.
4. **Known confound**: raw citation count ∝ legal sophistication ∝ acceptance. Fix:
   attach claim-matched citations to every application and **pad with hard-negative
   citations so every app has roughly the same number attached** — forcing the model
   to use match *quality*, not count.
5. **Instrument needed**: a retriever that, given a claim (element), finds the prior-art
   spec segments that anticipate it. Trained on gold (claim element → exact cited
   paragraph) pairs parsed from real Office Actions.

## Asset map (what exists, on sk3 under datasets/patents/processed/)

| Vision step | Asset | State |
|---|---|---|
| Outcome labels | `patents_final_outcome_balanced.csv.gz`; OARD rejections (2.19M apps) | ✅ done |
| Clean §102 cite pairs | `clean_102_pairs*.jsonl.gz` (action_type=102 filter), `sample_103_pairs*` | ✅ done |
| Prior-art text corpus | granted claim1 (6.2M), pgpub claim1 (8.3M), legacy (6.9M), GP supplement (~1.4M), full text (4.7M) | ✅ done |
| Retriever (claim-level) | bge-m3 v1–v5; **v3 best: MRR 0.242, R@100 0.69** on claim1 test (40K queries) | ⚠️ weak |
| Retriever (claim-level, more data) | v5 (claim-1 triples + mined hard negs): **MRR 0.104 — regressed −57%** | ❌ dead end |
| OA gold supervision | v3 downloader running: 10.8K/1.11M apps, 1.1 app/s, **ETA ~270h (~11 days)** | 🔄 running |
| OA §102 extraction | 2,193 OAs extracted (1,432 apps); cron loop every 30 min on GPU 5 | 🔄 running |
| Paragraph-level pairs | v1 deterministic 18,227 + opt3 LLM-picked 6,265 + opt4 coarse 14,974 | ✅ built |
| **v6 training set** | `training_pairs_v6_{train,val,test}`: 18,386 / 1,123 / 954 (group-split by app) | ✅ built, **NOT trained** |
| Paragraph-keyed specs | `paragraph_keyed_specs.jsonl.gz`: 1,744 specs (GP scrape of cited refs) | ✅ small, grows w/ pipeline |
| Hard-negative generation for final dataset | — | ❌ not started |
| Final classifier dataset (app + matched cites, count-balanced) | — | ❌ not started |
| VAT metric run on that dataset | — | ❌ not started |

## Key empirical lessons so far

- **Claim-1↔claim-1 is the wrong unit of supervision.** v5 trained on more claim-level
  data made the retriever *worse*. Real §102 anticipations live in the cited ref's
  *spec paragraphs*, not its claim 1. Hence the paragraph-level pivot (v6).
- **OARD `action_type` row-level filter is essential** — 79% of naive "examiner cites"
  are IDS-form noise, not rejection-grounding cites.
- **OA text gives exact locations**: ~half of extracted citations carry `[00xx]`
  paragraph anchors (deterministic gold); the rest are col/line, figure, or vague
  (handled by LLM-picker / coarse fallbacks).
- gzip `open("at")` corruption cost us ~187K v2 OA records (28,256 recovered,
  preserved as part_000); v3 uses rotating plain JSONL. Never again.

## Scale-up math (why the OA pipeline matters)

Current snapshot: 60K OA records → 4,499 §102 OAs (7.5%) → ~8 pairs/OA.
Full 1.11M-app todo list ⇒ ~2.7M OA records ⇒ ~200K §102 OAs ⇒ **~1.5M paragraph
pairs eventually** (vs 20K now). The pipeline accumulates passively; retrain at
checkpoints (v7 @ ~100K pairs, v8 @ full).

## Path forward

### Phase 1 — validate the paragraph-level retriever (this week)
1. **Train v6** on the 20,463 paragraph pairs (MNR loss, same recipe as v3).
2. **Honest eval**, two tiers:
   a. paragraph-level test set (954 held-out pairs, pool = paragraph corpus);
   b. *cited-ref recovery*: given app claim, retrieve over full corpus → is the
      examiner's actually-cited ref in top-K? This is the metric that matters
      for hard-negative mining quality.
3. Decision gate: if v6 ≫ v3 on (b), the paragraph pivot is validated; keep
   accumulating OA data and retrain. If not, debug before scaling.

### Phase 2 — build the end dataset (the actual deliverable)
4. **Design the classifier dataset**: per application (first-draft claims),
   attach K prior-art segments:
   - real: examiner §102/§103 cites, claim-matched to specific paragraphs via
     the v6 retriever (constrained to retrieve within the known cited doc);
   - fillers: hard negatives mined by the same retriever from the full corpus
     (high-similarity but never-cited refs), padding every app to the same K.
5. **Confound audit**: in the constructed dataset, verify (a) citation count no
   longer predicts acceptance, (b) match-score distributions of real vs filler
   overlap enough that the task isn't trivially solvable by a similarity
   threshold, (c) acceptance base rates balanced across CPC sections.
6. **Baselines**: logistic on similarity stats (the "cheap V" floor), dense
   Llama-8B on full assembled input (the ceiling).

### Phase 3 — VAT decomposition run
7. V metrics: programmatic claim-element-overlap checks against attached art
   (§102 single-ref; §103 multi-ref combination coverage — separate pipelines,
   error analysis by slice).
8. A metrics: rubric pipeline (online-rubrics + extracted norms) on the
   application text for §101/§112/dp/formality/style rejection reasons.
9. Gap analysis: V+A vs dense ceiling = taste/residual, per standard protocol.
   **Per-rejection-class decomposition**: using OARD flags, report how much of
   each class's predictive signal is captured by V vs A vs residual. Prediction
   for prior-art classes should be V-dominated; §112-type classes A-dominated —
   if not, that's a finding worth understanding.

## Gate verdict (2026-06-10 evening): paragraph pivot VALIDATED

Cited-ref recovery eval — 460 held-out queries (group-split), pool = 175,074 paragraphs:

| model | para MRR | para R@1 | para R@10 | para R@100 | doc MRR | doc R@10 |
|---|---|---|---|---|---|---|
| v3 (claim-trained) | 0.107 | 0.074 | 0.165 | 0.409 | **0.406** | 0.535 |
| v6a (quality≥3) | **0.131** | **0.094** | 0.183 | 0.454 | 0.358 | 0.496 |
| v6b (all pairs) | 0.122 | 0.070 | **0.228** | **0.478** | 0.368 | **0.546** |

- Both v6 variants beat v3 on the paragraph task (supervision-unit hypothesis confirmed).
- v6a = precision (MRR/R@1), v6b = recall (R@10/100); coarse pairs trade precision for coverage.
- v3 keeps doc-level MRR crown → two-stage architecture locked: claim-level index for
  doc retrieval, v6a for within-doc paragraph selection (the slot used for attachments).
- Raw results: outputs/eval_v6_cited_ref_recovery.json. Retrain v6.1 when extraction
  round 2 (23,995 OAs) lands → ~10× pairs.

## Locked design rules for Phase 2 (decided 2026-06-10 with Alex)

- **Labels**: BOTH final_acceptance (primary) and this-round rejection (diagnostic);
  per-class OARD flags as columns.
- **K (attachments per app)**: FIXED K (~95th pct of real cite counts, likely 8-10).
  No dynamic K, no λ-threshold — either reintroduces a count/crowdedness confound.
- **Filler sampling**: similarity-distribution-MATCHED to real cites (similarity is a
  matching variable, not an attachment criterion) → no score threshold can separate
  real from filler by construction.
- **Symmetric attachment**: identical within-doc paragraph-selection procedure
  (v6 retriever) for real cited docs and filler docs. Accepts small risk of picking
  the wrong paragraph in a real doc (mitigate: top-2/3 paras per doc).
- **Temporal validity**: fillers filtered to pub_date < filing_date. (Retriever
  training is already safe — examiner cites inherently predate filing; in-batch MNR
  negatives are never attached to apps.)
- **App representation = abstract + claims ONLY** — empirically citation-free
  (0.25%/0.18% inline-cite prevalence by label across 548K apps; nothing for a dense
  model to overfit on). If spec/Background text is ever added for §112 A-metrics,
  STRIP inline patent-citation patterns first and re-run the prevalence audit.
- **v0 fast path**: OARD already has per-app §102/§103 cited docs for ALL 2.19M apps —
  the 11-day OA download is NOT on the dataset critical path. v0 = claim-level FAISS
  (patent_claims_v3, exists) for filler docs + within-doc v6 paragraph selection.
  Full-corpus paragraph index (spec_chunks chunked, 165GB; embeddings NOT built) is
  a later upgrade.

## Phase 2 build log (2026-06-10 night)

- Step 1 ✅ 100,429 apps with clean §102/§103 cites (86,857 + 307,454; p95=9 → K=10).
- Step 2 ✅ 213,982/317,767 cited docs resolved (67.3%); 13,478 zero-resolved apps = control set.
- Step 3 ✅ top-200 filler mining, all 491,126 apps with claim embeddings. Two fixes:
  OOM under GPU-6 contention → chunk-local topk-then-merge (id tensor 30.5GB→6MB,
  POOL_CH 250K) — after patch the whole search ran in **6 minutes**.
- Step 4 ✅ first assembly (`phase2_dataset_v0`): 486,642 records, exactly K=10 each,
  labels 240K/246K, 57K dropped (no claim emb / short candidates). Bug fixed en route:
  app_ids.txt is tab-separated (app_id\tpgpub_id) — whole-line keys missed every lookup.
- **Spot-check finding (the night's key result)**: per-app sim aggregates carry ~no
  label signal (AUC mean 0.52, max 0.52) BUT real cites sit BELOW the filler sim
  range (real p50 0.545 vs filler p50 0.659; filler floor ≈ top-200 limit). Leak:
  AUC(label ~ sim_std) ≈ 0.61 — a rejected app betrays itself by one less-similar
  attachment among uniformly-high-sim fillers. Violates the distribution-matching
  rule in the low tail.
- **v0.1 fix (step 3b)**: shared 100K random year-known pool subset; per-app GPU sims;
  keep 256 quantile-spread (row, sim) per app → stage C matches low targets from
  these instead of clamping at the top-200 floor. Ran in 20 seconds. Old build
  archived at `phase2_dataset_v0/archive_top200only/`.
- Conceptual note for the paper: naive embedding similarity scores fillers HIGHER
  than real examiner cites — raw cosine is an anti-signal for anticipation. The V
  pipeline must do claim-element matching (v6 / LLM verification), not similarity.
- **v0.2 (FINAL build of the night)**: second leak found in v0.1 — per-app filler
  targets concentrate around the app's few real sims → rejected apps had LOW sim
  spread (AUC(label~sim_std)=0.60 again, opposite mechanism). Fix: draw ALL filler
  targets from the GLOBAL real-sim distribution (reals are themselves draws from it,
  so per-app marginals become label-invariant by construction).
- **v0.2 audit (passed)**: 488,390 records, K=10 exact, labels 247K/241K.
  AUC(label~sim_mean/max/min/std) = 0.501/0.473/0.483/0.521; joint logistic on all
  four = **0.547 (residual sim-texture bound)**; within-rejected real-vs-filler
  sim AUC 0.524. Year-gap: label-level clean (0.49/0.50); within-rejected
  real-vs-filler year tell 0.60 (real cites temporally closer) — fidelity-first
  artifact, acceptable per Alex; v1 candidate = joint (sim × year-gap) matching.
- Archives: `archive_top200only/` (v0), `archive_v0p1_perapp_targets/` (v0.1).
- NEXT: remaining #49 audits (CPC balance, length probe, resolution-rate bias,
  manual high-sim filler inspection) + baselines: cheap-V floor ≈ the 0.547
  logistic; dense ceiling needs ≥4096-token input format decision (app text +
  10 attachments won't fit 1024).

## True-cites testbed + V0 (2026-06-11, ~1:30am)

Pivot per Alex: archive filler dataset; claim-level testbed from TRUE examiner
cites = calibration ground truth for V implementations (not an end task — avoid
"it's just entailment"; only aggregates like # claims fell feed the app task).

- **Label clarification**: csv `judgement` = FIRST-DRAFT approval, not final
  disposition (label-1 apps: 99.6% have zero OARD rejections). "Rescue by
  amendment" is outside this label by design. patents_dataset.jsonl.gz carries
  BOTH first_draft_approved and final_outcome.
- **Build chain**: extraction apps ∉ balanced csv (93/6.5K only!) → app_id →
  PatEx application_data (100% coverage, 96% w/ pgpub; beware kind-code digit:
  US20010023252A1 → regex (20\d{9}), NOT digit-stripping) → patents_dataset
  pg_claims (full claims; 18% coverage — it's a filtered subset) → testbed.
- **truecite_testbed_v1**: 1,621 records / 1,106 apps / 10,394 fell vs 21,470
  standing claims / 28,551 elements (83% of para-anchored resolved to text;
  71% of art docs have full GP paragraphs). GP scrape run 3 added 6,367 docs.
- **V0 (embedding max-sim verifier)**: claim score = max cos over attached-art
  paragraphs. v3 pooled/within/indep = 0.548/0.578/0.631; **v6a = 0.581/0.621/
  0.685**; depth-only giveaway 0.522. → (a) topic-confound hypothesis CONFIRMED
  (within-app, v6a ≫ v3); (b) labels not depth-trivial; (c) raw similarity is a
  weak verifier → V1 = v6a top-paragraphs + LLM element-coverage judge; retrain
  v6.1 on round-2 data (~10× pairs).

## V1 post-mortem + V2 pivot (2026-06-11 early am)

- **Stub-claim bug**: pgpub claims include "(canceled)" amendment stubs (999 of
  31,864) → they were easy "standing" negatives inflating V0. CLEAN V0:
  v6a 0.562/0.607/0.569 (pooled/within/indep), v3 0.529/0.564/0.502 (chance),
  depth-only 0.541. v6a>v3 direction stands; magnitudes much weaker than the
  contaminated first read (0.685).
- **V1 whole-claim coverage judge: failed, and informatively.** Anticipated-rate
  9.1% fell vs 8.0% standing (AUC 0.53). V1-GOLD sensitivity test (same fell
  claims, exact OA-cited paragraphs): anticipated-rate fell to 3.4%, coverage
  flat (0.33 vs 0.35) → NOT pure retrieval starvation. Gold anchors are
  per-element: whole-claim strict coverage is structurally miscalibrated
  (examiner evidence covers a subset of elements per excerpt; BRI ≠ plain-text
  strictness). Also ~30% persistent parse failures (Qwen FP8 loops + odd
  claims; multi-seed retry recovers ~40%/pass).
- **V2 = per-element judging**: "does paragraph P disclose element E?" —
  matches the extraction's gold unit (182K pairs). Calibration run launched:
  1,500 gold pairs vs 1,500 mismatched-paragraph negatives, BRI-framed prompt.
  If the judge separates these, the V pipeline = per-element retrieval
  (v6a/v6.1) + per-element judge + aggregate coverage per claim.

## V2 results (2026-06-11 midday)

- **V2 per-element calibration**: conf-AUC **0.889**, 0/3,000 parse failures,
  5.3% false-disclose on mismatched pairs. The atomic V unit works.
- **V2-full** (3,273 indep claims, 23,895 elements, 71,685 pair judgments):
  aggregator is a LEGAL choice — mean-of-max 0.534; **min-of-max (§102: every
  element must be disclosed) 0.574/0.575** — best V so far (V0 0.569).
- **Mechanistic bottleneck identified**: min-aggregation compounds element-level
  retrieval misses (v6a within-doc top-3 ≈38%; P(all 5-7 elements found) is
  small) → fell claims' min scores collapse. The retriever is the quantified
  bottleneck — exactly Alex's "count on the retriever" position.
- Lift levers queued: **v6.1** (extraction round 2 → ~10× pairs, retrain on
  GPU 5 when it frees) + **top-10 per element** retrieval depth; rerun V2
  retrieve+judge as a clean retriever ablation.

## Per-rejection-class A results (2026-06-11 afternoon) — VAT pattern CONFIRMED

Join chain: datapoint→pgpub (text reconstruction vs patents_dataset, 4,996/5,000)
→ PatEx pgpub→app (4,987) → oard_rejections_by_app.csv (full universe).
A = logistic on 119 rubric-aspect scores (qwen 20x1_r2post judge, 803 dp).

| slice (accepted vs class-rejected) | n_rej | A-AUC |
|---|---|---|
| §101 eligibility | 33 | **0.759** (bootstrap CI 0.66–0.87) |
| §112(b) definiteness | 67 | 0.572 |
| §102 anticipation | 87 | 0.524 |
| §103 obviousness | 148 | 0.499 |
| §112(a) enablement | 28 | 0.495 |

- EXACTLY the VAT prediction: prior-art classes (need external evidence) are
  NOT articulable from the document alone → A null; §101 eligibility (a property
  of the text itself) IS articulable → A works.
- Face validity: top-weighted aspects for the §101 slice are the eligibility
  rubrics themselves (a31 subject-matter eligibility +1.47, a53 generic-computer
  insufficiency +0.85).
- Overall A-AUC ≈ 0.50 was a composition effect: prior-art rejections dominate
  the rejected pool (148+87 vs 33), drowning the §101 signal.
- within-rejected 112-vs-priorart discrimination: 0.51 (aspects separate
  accepted-vs-101-rejected, not class-vs-class among rejects).

## Open design questions (need a decision)

1. **Label semantics**: many apps get a non-final §102 rejection and are STILL
   eventually accepted (amended around it). Predicting *final acceptance* from
   first-draft + attached art means the V signal is "is there art that can't be
   amended around" — harder, but the honest task. Alternative: predict
   *this-round rejection* (cleaner V link, less interesting outcome).
2. **§103 scope**: §103 (obviousness) is multi-reference combination — claim
   matching is fundamentally harder than §102 single-ref anticipation. Start
   V with §102-only, treat §103 as articulable-tier?
3. **Downloader ETA**: 11 days at 1.1 app/s (24-way concurrency, likely API
   rate-limited). Accept passive accumulation (train v6 now, v7/v8 later) or
   invest in speedup?

## 2026-06-11 (eve): per-ASPECT rejection-class analysis + wrap-up tooling

Q (Alex): does the aspect bank fully capture 102/103/etc. rejection reasons?
A: **nominally yes, operationally no — and the per-aspect numbers prove the VAT point.**

- Bank (249 aspects) covers every statutory ground by name: §101 ×22 (a31/a53/a55/a56),
  §102 (a22/a34/a35), §103 (a26), §112a (a7/a9/a10), §112b (a11/a36), §112f (a13-a41), DP (a64/a93/a123).
- But per-aspect AUC vs actual rejection class (804 dp, qwen judge): novelty/obviousness rubrics
  a26/a34/a35 = 0.52-0.56 vs §102/§103 rejections (chance), while the SAME rubrics hit 0.63-0.64
  vs §101, and the dedicated §101 rubrics hit a31=0.727 / a53=0.878 (small-n caveat on a53).
- Interpretation: "is this novel?" asked of the document alone degenerates to "does it READ novel" —
  the relational fact lives in the prior art, outside the input. No articulation fixes that; only V can.
- Drafting-side proxies (a23 problem framing, a60 prior-art differentiation args) also chance: 0.43-0.55.
- Judged-set gaps: a55 (Alice/Mayo) + a56 (abstract-idea) never qwen-judged; a22 almost never applicable.
  Deferred (Alex: keep lean).

Persisted tooling (replaces yesterday's inline analysis):
- `scripts/per_class_a_analysis.py` → `runs/validity_full/v2/patents/per_class_a_results.json`
  (CV-logistic pooled + per-aspect AUC, both judges). Pooled qwen: §101 **0.780 [0.69-0.86]**,
  §102 0.561, §103 0.526, §112a 0.418, §112b 0.560. (Yesterday's inline 0.759 ≈ same story,
  slightly different estimator.)
- `scripts/assemble_vat_table_patents.py` (staged on sk3) — one command tomorrow: reads
  v61_ablation_results.txt + app_v_scores.json (+ OARD per-class V slices) + per_class_a_results.json
  + dense ceilings → `notes/2026-06-12__patents-vat-final-table.md`.

Chain health 16:20: v6.1a training 63% (slowed to 4.4s/it, contention), post-chain PID 3526220 +
app-level PID 3950725 both alive and gated correctly.

## 2026-06-11 (night): v6.1a verdict — contamination found, honest numbers much lower

Within-doc eval after v6.1a training showed apparent regression (0.207 vs v6a 0.595 MRR).
Investigation found the real story: **build_v6_training_set.py splits via random.seed(7)+shuffle
of the apps list — but the list GREW in round 2, so the permutation changed and v6a's old
train apps leaked into today's test split.**

Contamination-controlled eval (scripts/eval_v6_within_doc_honest.py; round-2-only pairs =
didn't exist when v6a trained; test apps unseen by v6.1a):

| model  | honest (n=2,274) MRR / top-3 | v6a-train-overlap (n=545) MRR / top-3 |
|--------|------------------------------|----------------------------------------|
| v6a    | 0.207 / 0.208                | 0.395 / 0.437  ← memorization signature |
| v6.1a  | 0.214 / 0.229                | 0.207 / 0.231  ← flat, as expected      |

Conclusions:
1. v6a's "38% within-doc top-3" was inflated ~2× by train/test leak; honest is ~21%.
2. 7× training data → +10% relative top-3 (0.208→0.229). Bi-encoder recipe has PLATEAUED.
3. V2's retrieval bottleneck is worse than believed → P(all elements retrieved) lower →
   explains min_of_max 0.574 ceiling. The v6.1a V2 ablation (judge running now) is the
   decisive pipeline number, but expectations should be modest.
4. Caveat: gold-paragraph MRR is pessimistic — other paragraphs may equally disclose an
   element (judge calib 0.889 suggests retrieved-but-not-gold paras are often fine).
   Pipeline-level AUC remains the metric that matters.
5. Distilled cross-encoder rerank (xenc-v1, queued) is now MORE important: if the bi-encoder
   is plateaued, rerank top-K is the cheapest remaining lift.
6. TODO for v6.2: replace seeded-shuffle split with stable hash of app_id (like
   element_para_xenc_v1 build) so splits survive data growth.
