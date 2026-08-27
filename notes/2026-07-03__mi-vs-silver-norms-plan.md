# Validating MI/recovery certificates against silver human-norm labels (Sorensen-style)

**Goal.** Borrow Sorensen et al. 2022 (ACL, "An Information-theoretic Approach to Prompt Engineering
Without Ground Truth Labels"): they show per-*template* label-free MI correlates (Pearson R 0.68–0.96)
with per-*template* accuracy. **Our analog:** per-*metric* label-free information score (T = I(M_ω;X),
OPT_Ω, recovery R — all already computed by our certificate) correlates with per-*metric* **silver
salience** = how often real human comments/reviews invoke that metric. If high-information metrics are
the ones humans actually articulate, the instrument is validated against a human-grounded external
signal it never trained on.

Tasks (10, everything but law): peer_review, litbench_rationales, humor_multi, humor, press_releases,
nc_public_comments, aops_forum, competition_editorials, creative_writing, mathlib.

## Verified data facts (2026-07-03)

**Silver labels** live at `sk3:/lfs/skampere3/0/alexspan/data/bge_pertask/<task>/`:
- `signals_<task>.jsonl` — `{"i": <docid int>, "s": "<norm phrase>"}`. `i` is a SOURCE-DOC id (repeats,
  ~8 norms/doc), NOT a unique norm id. Real reviewer/comment phrases (peer_review "would greatly enhance
  the readability"; mathlib "mark it as a `simp` lemma").
- `matches_ce_<task>.jsonl` — `{"signal_id": "<docid str>", "top10": ["a56","a31",...]}`. CE-reranked
  norm→R2-metric mapping, rank-ordered best-first.
- **JOIN IS POSITIONAL** for norm text: `matches_ce` line N ↔ `signals` line N (verified 0 mismatches /
  20k on peer_review; `signal_id`="2" appears 10× with 10 different norm texts). Do NOT key-join on
  signal_id — it's a doc id, many-to-one. (The other agent's "join by key" only holds where `i` is
  unique per doc; positional is the safe general rule. Their R2→R3 rollup finding IS correct + used below.)
- `catalog.txt` — `a{N}: <name>`. **`a{N}` index == `<task>_general_r2_expanded.json` merged_group
  index N** (verified: CW a0/a66 exact). So each `a{N}` carries a `merged_name` AND `merged_description`.
- `matches_<task>.json` (5 tasks only: math, code_review, humor, press_releases, creative_writing) —
  GOLD `{"id": <int>, "aspects": [3 a-ids]}`, unordered set of 3, key-join by str(id)==signal_id, used
  for recall (set-intersection). A cleaner (human-picked, not CE) silver signal where present.

**Rollup** at `sk3:.../norm-research/outputs/hierarchy/<task>_general_r{2,3}_expanded.json` (path in
memory was wrong; corrected here). r3 group has `source_r2_cluster_ids` (bare ints → `a`+str(id))
→ this rolls R2 `a{N}` up to an R3 `merged_name`. `_general_` bucket is the one matching the catalogs.

**Certificate coverage TODAY (the work):** only creative_writing is certified (1123 sigs.npz). humor 81,
peer_review/N&C 1 each, press_releases/math 0. So certs must be RUN for ~all tasks. Certificate driver =
`methods/metric_implementer/experiments/run_real_test.py` / `value_certificate.certificate()`; per-metric
sigs npz carries `M_i`, `sigs`, `prompts`; cert json carries `opt_omega_bits`, `gains`, `H_M`,
`eps_bits_adv`, `verdict`, `merged_name`.

## The granularity decision: score at **R2**, not R3

Silver labels are native **R2** (`a{N}`, 200–367/task). Certs so far are **R3** (CW 46). Two reasons to
move certification to R2 for this experiment:
1. **R3 rollup discards silver mass.** CW smoke: at top-10, 12,215 / ~28k silver hits map to R2 clusters
   with NO R3 parent in the cert set (cert covers 46/70 R3 groups; only R2 that rolled into those survive).
2. **R3 too coarse for small tasks.** peer_review has 13 R3 groups, N&C has 5 — you cannot correlate over
   5 points. R2 gives 200–371 metrics = a real correlation.
   R2 counts: CW 371, peer_review/humor_multi/etc ~200, press_releases/humor 324/366.
R2 scoring is fully wired (a{N}→r2_expanded[N]→merged_description = the cert seed). Keep an R3 rollup as a
secondary, coarser view.

## The measure (Sorensen faithful)

- **X-axis (label-free instrument score), per metric** — reuse the certificate outputs, NO new label:
  primary `OPT_Ω` (checklist recovery); flanks `T = H_M` (= I(M_ω;X)), `g1` (best single criterion),
  recovery `R`. These are our "MI(θ)".
- **Y-axis (silver salience), per metric a{N}** — how much human attention it draws:
  - `sal_top1[a]` = # norms whose CE-rank-1 == a  (strongest in smoke: Spearman +0.25 vs +0.21 for topk)
  - `sal_topk[a]` = # norms with a in top-k (k=3,10), optionally rank-weighted Σ 1/rank
  - `sal_gold[a]` = # gold records with a in its 3-aspect set (5 tasks; cleanest, human not CE)
  Report all; lead with top1 + gold.
- **Correlation** = Spearman (rank, robust) primary; Pearson on log1p(sal) secondary; per task, then a
  fixed-effects pooled fit across tasks with task dummies. This is the Fig-4/5 analog.
- **Controls (critical — Sorensen didn't need these, we do):**
  - PERMUTATION NULL: shuffle a{N}→salience, recompute Spearman, 1000×; report empirical p. Guards
    against "both correlate with metric frequency/size."
  - PARTIAL OUT SIZE: silver salience is confounded by cluster size (`total_leaf_rubrics`) and metric
    base rate. Report partial Spearman(sal, OPT | log total_leaf_rubrics, M_i_mean).
  - CE-QUALITY GATE: CW CE recall@10 is the WORST (0.23); math/PR/humor far better. Weight/annotate
    per-task correlations by CE recall so a weak correlation on CW isn't read as "no signal."

## Smoke test already run (CW, R3, local) — PROOF IT WORKS

Full pipeline executed end-to-end: 2877 CW norms → salience per a{N} → R2→R3 rollup → join to 46 cert
rows. **Spearman(sal_top1, OPT_Ω) = +0.25 (p=0.09); vs g1 +0.31.** Top-salient = Dialogue craft (2052
hits, OPT 0.76), POV (OPT 0.79); zero-salient = worldbuilding/setting. Directionally Sorensen's result on
the worst-case task at the lossy R3 granularity. R2 + better-CE tasks should be cleaner.

## UPDATE 2026-07-03 (other agent's join audit — USE THIS, simplifies Phase 0)

The join is now DONE for us: a pre-joined artifact `matches_joined_<task>.jsonl` exists for all 26 corpora
(all 10 targets verified present). Each line = `{"row": <stable per-norm line idx>, "doc": <doc id>,
"norm": <norm text>, "top10": [a-ids], "top10_names": [{id,name}]}`. So salience = just count a-ids in
`top10` — NO join to signals/catalog needed, NO positional-alignment risk. Their audit independently
confirmed the join is POSITIONAL (matches_ce row N in-order aligns to the Nth retained signal; 0 mismatch /
260,356 rows; `signal_id` repeats in 20/26 corpora), matching our empirical finding. Re-ran CW smoke against
the artifact: reproduces EXACTLY (Spearman +0.251 vs OPT_Ω, +0.310 vs g1). Phase-0 harness reads
matches_joined directly.

Two audit findings that affect result-reading:
- **6 corpora key-join cleanly** (unique doc-id per norm): the 5 gold tasks (math, code_review, humor,
  press_releases, creative_writing) + humor_multi. An i=0 filter bug (`if sid and stxt` treats int 0 as
  falsy) silently dropped ONE norm in each of the 5 gold corpora (mc = sig-1); fixed in match_cascade.py
  (`sid is not None and stxt`). CW unaffected (doc ids start at 1). Minor; re-materialized artifacts clean.
- **wp_comments = label NOISE, not a join defect** (<=25% sensible): same training anchor stamped positive
  for unrelated leaves in bge_train (spelling anchor -> both a6 spelling AND a190 setting) + low triplet
  volume. Not in our 10 targets, but it names a CE-QUALITY confound to watch: noisy bge_train -> noisy
  top10 -> attenuated correlation that is NOT "no articulability signal." Reinforces the per-task CE-recall
  weighting in the controls.

## PHASE 0 HARNESS BUILT + RUN (2026-07-03) — controls flip the CW read

Harness: `methods/metric_implementer/experiments/silver_validation.py` (CPU). Reads matches_joined,
rolls a{N}->R2/R3, joins cert, computes salience (top1/3/10, rank-weighted, gold), and ALL FOUR controls.
CW/R3 result:
  - raw rho(OPT, silver_top1) = +0.251, BUT perm-null p=0.087 (weak) and **partial(OPT | log size, base)
    = -0.02 ⇒ the raw correlation is ALMOST ENTIRELY a cluster-size confound.**
  - Diagnosis: rho(silver, cluster_size)=+0.66, rho(OPT, size)=+0.17 — silver salience mostly tracks how
    BROAD a metric is (how many R2 leaves rolled in), not how information-rich. Net of size, OPT↔silver ≈ 0
    on CW.
  - Attenuation ceiling ≈ 0.93 (silver split-half r=0.97 — HIGHLY reliable), so the low correlation is NOT
    measurement noise; it is real. (r_MI=0.9 placeholder — TODO wire real form-orbit var_phi.)
  - Coverage ceiling (the genuine silver-overlap upper bound): scored pool covers 0.67 of silver-invoked
    metrics, 0.42 of silver MASS is unmapped, Lincoln-Petersen N_hat≈68 vs 46 scored. This is the answer to
    "can we upper-bound silver overlap": YES via capture-recapture on {Ω-discovered} vs {silver-invoked} —
    NOT via §12.6 (OPT_Ω+ε bounds recovery of M_ω, a different RV than human attention Y_human).
  CAVEAT: CW is WORST-case (CE recall 0.23, R3 = lossy granularity). The real test is R2 on better-CE tasks
  with the size control. Do NOT generalize the CW null.

## Two computable ceilings on silver overlap (answering the user's upper-bound question)
  (A) Attenuation: rho_obs <= sqrt(r_MI * r_silver); r_silver = split-half salience reliability (measured),
      r_MI = form-orbit reliability (var_phi, to wire). Disattenuated rho_true = rho_obs/ceiling.
  (B) Coverage / capture-recapture: silver labels are an INDEPENDENT second capture list of "which metrics
      matter" (human comment + CE) vs our Ω discovery (LLM free-gen + GEPA). Unmapped silver mass = certified
      floor on unexplainable attention; Chao/LP -> total human-relevant metrics + Ω coverage. The silver
      analog of missing-impact / B_E. §12.6 does NOT bound this (different RV).
Note on the user's GEPA-vs-upper-bound worry: reconstruction-optimized R can EXCEED OPT_Ω (§12.6.4 F-class
escape) — for THIS experiment that's a testable feature (does silver track recoverability R > checklist
OPT_Ω? = humans track recoverability, not decomposability). Put both on the x-axis.

## ★ PHASE-1 BLOCKER FOUND (2026-07-03): catalog↔hierarchy join splits by catalog size

Before scoring, I audited whether the silver catalog `a{N}` names can be joined to a certificate-scorable
metric (with a description to seed GEPA). Result — join by NAME (not index; index a{N}==group[N] is a
COINCIDENCE that breaks: CW 368/368 by name but only 227/368 by index):
  - **cat=368 CW → 368/368** match creative-writing_general_r2_expanded.json ✓
  - **cat=366 humor → 284/366**, **cat=324 press_releases → 220/324** ✓ (partial but usable)
  - **cat=200 (peer_review, nc_public_comments, mathlib, aops_forum, humor_multi, litbench_rationales,
    competition_editorials) → 0/200.** Every "200" catalog is a GENERIC PLACEHOLDER (identical size,
    hand-authored-looking names like "correctness of mathematical derivations"), NOT derived from that
    task's hierarchy. No descriptions file on sk3 matches these names (widest-net search 0/200).
match_cascade/cascade_3stage read catalog.txt as-is; nobody re-derives it → the 200-catalogs' provenance
is orphaned + description-less.

**Consequence:** Phase 1 (score R2 certs seeded by merged_description, join to silver) is VIABLE NOW only
for the 3 rich-catalog tasks: **creative_writing, humor, press_releases** — which conveniently are also 3
of the 5 GOLD tasks (cleanest silver). The other 7 need their real per-task catalog REGENERATED from the
hierarchy (re-run match_cascade against `<task>_general_r2_expanded.json` names+descriptions) before they
can be certified/joined. That regeneration is a prerequisite task, not part of scoring.

**Revised Phase-1 scope:** start with creative_writing (already 46 R3 certs — reuse; run R2 fresh),
humor, press_releases. Defer the 7 placeholder-catalog tasks pending catalog regen.

## r_MI WIRED (2026-07-03): real form-orbit reliability
sigs npz carry `M_i_flip_rate` + `M_i_var_phi` per metric. r_MI = mean(1 - flip_rate) over scored metrics.
CW: flip 0.016-0.28 (median 0.13) → r_MI ≈ 0.87 → attenuation ceiling 0.916 (was 0.932 placeholder);
ρ_true(OPT,top1)=0.27. Harness `--flip <json>` arg added; /tmp/cw_smoke/cw_flip_rates.json is the CW file.

## PHASE-1 LAUNCHED (2026-07-03 PM): humor + press_releases R2 scoring

Driver: `run_alpha_probe --task <t> --level R2 --r2-bucket general --n-metrics 0 --no-glm
--skip-existing --n-probes 300 --orbit-target 4 --target-model Llama-3.1-8B`. Ω = children + free-gen
(qwen/llama/haiku via OpenRouter, NO GLM to save quota). Produces per-metric sigs.npz (M_i, sigs, prompts,
M_i_flip_rate, orbit) — the §12.6 certificate (OPT_Ω/gains/eps) runs AS A SECOND STEP on these sigs.
  - humor:          GPU 3, 285 R2 metrics, out=/lfs/…/outputs/silver_r2/humor/       (running, 4+ sigs @ 2min)
  - press_releases: GPU 4, 221 R2 metrics, out=/lfs/…/outputs/silver_r2/press_releases/ (running)
Verified before launch: (1) `r2_groups(task,general)` == silver catalog source, name-join humor 284/366,
PR 220/324 (CWD-relative outputs/hierarchy — must run from repo root). (2) probes load (humor 360, PR 360).
(3) press_releases needed WIRING: added config.py preset + manifest.py entry (.bak_pr backups; the old
"corrupted" note is stale — press_release_modeling_dataset_clean.csv.gz is the 73620-row rebuild, clean).
Env: HOME=/lfs, HF_HOME=/lfs/…/shared_hf_cache, envs/ai_usage/bin/python (base python3 lacks pandas).

**AFTER sigs complete:** (a) run the §12.6 certificate over each task's sigs → cert JSON (OPT_Ω/gains/eps/
H_M/verdict per metric); (b) extract flip_rates → r_MI; (c) run `silver_validation.py --level R2` joining
catalog→r2_expanded BY NAME (load_r2_index fixed to return r2_expanded merged_name = the exact cert key),
with gold (both have matches_<task>.json). Then the multi-task correlation table + scatter. R3 rollup secondary.

## ★ AUTONOMOUS CONTINUATION (2026-07-03, laptop closing): fully server-side, survives logout

All processes are detached (PPID 1, HOME=/lfs). Nothing depends on the laptop.
  - alpha_probe: humor PID 1485096 (GPU3), press_releases PID 1500595 (GPU4) — writing sigs.
  - chain_silver.sh (PID 1520475 humor, 1520476 press_releases): each waits for its sigs to reach
    285/221 (or the alpha_probe to exit), then CPU-only runs value_certificate --dir → cert_<task>.json,
    extracts flip_<task>.json, and silver_validation --level R2 (by-name join, gold, real r_MI) →
    <task>_R2.json. Script: /lfs/…/outputs/silver_r2/chain_silver.sh. value_certificate dry-run on
    humor's partial sigs CONFIRMED working (real OPT_Ω/verdict rows).

**HOW TO CHECK RESULTS WHEN BACK:**
  - `cat /lfs/skampere3/0/alexspan/outputs/silver_r2/{humor,press_releases}/STATUS` (DONE|FAILED)
  - results JSON: `/lfs/…/outputs/silver_r2/<task>/<task>_R2.json` (rho/partial/perm-p/ceilings/gold)
  - progress: `tail /lfs/…/outputs/silver_r2/<task>/chain.log`
  - then build the multi-task notebook (extend 2026-07-03__mi-vs-silver-norms-cw.ipynb): read the two
    _R2.json + CW, make the task×score correlation table + per-task scatter panels. The HONEST headline
    is the size-partial + gold (per the CW finding), not raw rho.
CODE CHANGES (with .bak): config.py + manifest.py (press-releases preset+entry — legit, keep),
silver_validation.py load_r2_index by-name fix. All synced to sk3.

## Work plan

**Phase 0 — harness (CPU, no GPU).** `experiments/silver_validation.py`:
  (a) load `matches_joined_<task>.jsonl` (pre-joined — no signals/catalog join) + r2_expanded + optional gold;
  (b) build `sal_*` per a{N}; (c) join to cert rows by merged_name (R2) or via rollup (R3);
  (d) Spearman/Pearson + permutation null + partial-out; (e) emit per-task json + a summary table.
  Validate on CW where a cert already exists (reproduce the smoke number at R2).

**Phase 1 — certificate scoring at R2 (GPU, the bulk).** For each of the 10 tasks, run the R2 certificate
over its `a{N}` metrics (seed = merged_description, target = M_i orbit, Llama-8B executor, 300 probes from
that task's source corpus). Order by cheapness/coverage: start peer_review (13→~200 R2), math, humor,
press_releases; then the rest. Reuse `run_real_test.py`; probes come from each task's `_load_texts`
(source-text availability per task = the one open item from the background agent — confirm before scoring
a given task). ~200 metrics × 300 probes × 4 forms; batch hard on 1 GPU.

**Phase 2 — correlate + write up.** Per-task Spearman(silver, {OPT_Ω,T,g1,R}) with nulls + partials;
pooled fixed-effects; a Sorensen-style scatter (silver vs OPT per metric, one panel/task) + a correlation
heatmap (task × instrument-score). Notebook `2026-07-0X__mi-vs-silver-norms.ipynb`.

## Source-text / probe corpus — RESOLVED (2026-07-03, self-checked; background agent died mid-run)

Certificate probes load from `methods/metric_implementer/manifest.full_manifest()` → `load_corpus(entry)`,
per task, NOT from the bge `signals` files. Key semantic point: the probe text is the **OBJECT** being
judged, while silver norms are extracted from **feedback ABOUT** those objects — same underlying objects,
different text type. This is CORRECT for the design (metric scored on X=object; salience = how often
humans commented on that property), and mirrors Sorensen (score on X, quality signal is external).
  - peer_review: `datasets/peer-review/splits/train.csv.gz`, col `text` = the PAPER (abstract/intro),
    NOT the review. ✓ correct probe corpus (norms are review comments about these papers).
  - Manifest HAS (5): peer-review, math (`math_se_modeling.csv.gz`), humor
    (`reddit_humor_modeling_dedup.csv.gz`), notice-and-comment, creative-writing
    (`writingprompts_modeling_clean.csv.gz`). These 5 are Phase-1-ready.
  - Manifest MISSING but corpus EXISTS in datasets/ (add manifest entry, then ready):
    press_releases (`press_release_modeling_dataset_clean.csv.gz`), litbench_rationales
    (`creative-writing/litbench-to-train*.csv.gz`), humor_multi (`reddit_humor_modeling_with_topics`),
    mathlib (`math/mathlib/a_metric_verdicts_mathlib.jsonl` — verdict file; needs object-text column check).
  - Corpus NOT FOUND (BLOCKED — locate or build before Phase 1): aops_forum, competition_editorials.
  - Task-name mapping gotcha: bge task `math`/`aops_forum`/`mathlib` are all "math" family but distinct
    corpora + distinct catalogs; keep bge-task↔manifest-task explicit (creative-writing vs creative_writing
    underscore/hyphen too).

## Other open items
- Score R2 fresh vs reuse existing R2 sigs (CW has R2-era artifacts; check provenance/executor match
  before reusing — see shared-Ω retarget confound memory: mixed native/retarget OPT_Ω is CONFOUNDED).
- 20k MAXSIG cap on the 6 giant corpora truncates the salience tail; fine for a head-dominated salience
  RANKING but note it.
- gold `matches_<task>.json` present for math, code_review, humor, press_releases, creative_writing —
  use as the cleaner (human, not CE) salience signal on those 5; it's the strongest validation subset.

## ★ HUMOR R2 RESULT (2026-07-03, autonomous chain DONE — first full-R2 task)

Pipeline: 285 R2 sigs (GPU3, ~3.3h) → value_certificate (155 CODIFIABLE / 130 FORM-DOMINATED — sk3 cert
is pre-form-gate-redesign, fine for OPT/H_M/g1) → flip 284/284 → silver_validation (265 scored; 19 dropped
for empty gains, 1 dup sig name benign). Join audit: cert∩catalog by name = 284/284 exact.

| salience | raw ρ(OPT) | perm p | **partial(OPT\|size,base)** | ρ(g1) | ρ(T=H_M) | ρ_true (disatten.) |
|---|---|---|---|---|---|---|
| CE top-1 | +0.185 | .004 | **+0.121** | +0.191 | +0.121 | +0.193 |
| CE top-3 | +0.197 | .005 | **+0.097** | +0.202 | +0.143 | +0.205 |
| CE top-10 | +0.256 | .002 | **+0.094** | +0.262 | +0.206 | +0.265 |
| GOLD (human) | +0.228 | .002 | **+0.117** | +0.230 | — | — |

**Key contrasts vs CW:** (1) the size-partial SURVIVES in CE too, not just gold (CW CE partial was −0.02) —
consistent with the prediction that CW was worst-case (CE recall 0.23); humor's silver is far less
size-confounded (gold-vs-size +0.22 here vs +0.66 silver-vs-size on CW). (2) r_MI real = 0.937,
r_silver = 0.975–0.997 → attenuation ceiling ~0.96 (not binding). (3) Coverage: 359 invoked vs 265 scored
BY COUNT (0.74, LP N̂≈360) but **only 5.0% of silver MASS on unscored metrics** (gold: 2.0% unmapped +
5.3% unscored) — the count gap is a long tail; mass-coverage ≈95%, ceiling not binding.
(4) g1 ≈ OPT everywhere — best-single-criterion tracks human salience as well as the full checklist OPT.

Harness fix applied post-hoc (reporting only, correlations unchanged): `silver_mass_on_unscored_frac`
added to coverage_ceiling (old `unmapped_mass` is trivially 0 at R2 since a2parent covers every catalog
a-id); gold block now reports `gold_total_mass` + fractions. Synced to sk3 — press_releases chain will
use the fixed harness automatically at its step 4.

**press_releases:** 85/221 sigs @ 23:31 PDT (~17.5 sigs/hr) → ETA ~07:15 PDT 2026-07-04; chain PID
1520476 healthy, polling every 120s.

## ★ PRESS_RELEASES R2 RESULT + FUNDAMENTALS AUDIT (2026-07-04)

Chain DONE 05:35 PDT. 208 scored (221 sigs; 209 with gains; join 208/209 exact by name). Headline: **null.**
raw ρ(OPT, CE top-1) = +0.060 (perm p=.39), top-10 +0.047 (p=.49), GOLD +0.085 (p=.21); partials +0.02–0.08.

**Fundamentals audited — pipeline is SOUND; the null is real, not a bug:**
1. Join exact (208/209 cert↔catalog by name); matches_joined 11,754 lines; gold present (35,328 mass).
2. Cert not degenerate: 0 zero-OPT rows, H_M mean 0.657 (only 15/209 < 0.2 bits). But: frac_H 0.36 vs
   humor 0.55; 63% FORM-DOMINATED (vs 46%); flip mean 0.143 vs humor 0.059 (36/220 metrics flip>0.25).
3. **Channel-agreement test (the discriminator): ρ(CE salience, gold salience) = +0.81/+0.88 on PR**
   (partial|size +0.80) — same as humor (+0.82/+0.80). The two independent human-attention channels agree
   strongly WITH EACH OTHER; both fail to track OPT. → measurement channel is fine; the CONSTRUCT
   relationship is absent on PR.
4. Reliability-filtered recompute (drop flip>0.25 or H_M<0.2; keeps 158/208): does NOT rescue —
   CE top-1 +0.060→+0.007, gold +0.085→+0.043. So not attenuation-by-noisy-instrument either.
   (Humor same filter: partial stays ≈ +0.09–0.11 — robust.)
5. Data character: PR "norms" are journalist coverage snippets, many descriptive-of-company not
   evaluative-of-release ("Volkswagen is playing it more pragmatic"); salience hyper-concentrated —
   top-5 metrics = 51% of top-1 mass, "Media access and follow-up" alone 17% (every "declined to
   comment" lands there). Gold is less concentrated and evaluative — and shows the SAME null, so
   concentration/noise is not the driver.

**Reading (descriptive, not verdict):** on press_releases, human coverage-attention is ~orthogonal to
metric info-richness (OPT_Ω) — consistent with the PR deconfound finding (outcome signal mostly
publisher-id + topic, not text-quality metrics). Cross-task pattern so far:
| task | CE partial | gold partial | note |
|---|---|---|---|
| creative_writing | −0.02 | +0.168 (p≈.06) | CE noisy (recall .23) |
| humor | +0.121 (raw p=.004) | +0.117 (raw p=.002) | cleanest positive |
| press_releases | +0.015 | +0.083 (p=.21) | null; channels agree ρ=.81 |

## ★ GEPA-RECOVERY vs CERTIFICATE on the SILVER axis (CW, 2026-07-04)

User question: for external validity, does the certificate (OPT_Ω) track human silver/gold salience
better than GEPA-reachable prompt recovery? Computed head-to-head on the 46-metric CW cert universe
(same R3 rollup as the CW notebook; desc_R = bank-wide GEPA* proxy since desc-seeded GEPA improved
7/12 metrics by ~0; GEPA* exact on the 12-sweep subset).

| score (x-axis) | CE top-1 ρ / partial | GOLD ρ / partial |
|---|---|---|
| name_R (name rung) | −0.14 / −0.25 | −0.14 / −0.19 |
| desc_R (≈GEPA*) | +0.13 / +0.05 | +0.01 / −0.10 |
| g1 (best criterion) | **+0.31** / +0.07 | +0.28 / +0.15 |
| OPT_Ω (certificate) | +0.25 / −0.03 | **+0.27 / +0.17** |
| T = H_M | +0.16 / +0.23* | +0.15 / +0.16* (*|size only) |

Paired GEPA-12 (n=12, indicative): GOLD rho(OPT)=+0.59 vs rho(GEPA*)=−0.00 vs rho(desc_R)=−0.21.

READING (descriptive): the human-salience signal lives on the CERTIFICATE side (g1/OPT/T), not on
prompt-attained recovery. Single-prompt R is executor/rung-bottlenecked — its across-metric variation is
mostly channel noise, so it carries ~no external validity; OPT_Ω integrates out prompt form and is the
quantity that correlates with human attention. name_R is mildly NEGATIVE — metrics whose bare name
executes well (taste-index metrics per what-gets-decompressed) are not the ones humans discuss most.
Caveats: CW only; CE channel noisy on CW (gold is the clean column); n=46 (SE≈0.15).
Extension: score name/desc rungs on humor (366 metrics, ~1 vLLM pass) to replicate on the cleanest task.
