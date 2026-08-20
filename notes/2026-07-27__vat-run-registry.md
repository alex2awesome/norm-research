# VAT run registry — the 8×3 grid and what remains to run
2026-07-28 REMAP (user slide): math verdict = MATHLIB PR merge (math.SE → community,
accepted∧upvoted); N&C verdict = RECEIVED-A-RESPONSE (change-made → curation/revealed row).
New chain T (test-split interim): N&C responded .825 (band +.19!), peer curation .588
(+.02), peer verdict TRAINING. Fig 3 dumbbells REPLACED by Table 2 (named top-V vs top-A
showcase; ypref_features.pdf kept on disk). AUDIT (a) DONE 2026-07-28: code .584 was MODERNBERT-large 512-tok
(NOT Llama), 6,000-char-capped diffs, lr 2e-4 epoch-1 overfit, select-on-test. LEVEL honest
(cluster .58-.62; signal-cell grouped-dense .609-.621; .786/.749 = repo-identity leak —
repo-rate alone .794/.706). Standard re-run in V4 with REBUILT csv (bigger cap, len
2048-4096, pos_weight, ≥3 seeds, select-on-eval); expect ~.61-.62. Chart bar hatched
provisional. AUDIT (b) DONE 2026-07-28 — lost-cells hunt results (all consolidated into charts):
- FICTION verdict = RoyalRoad Amazon-KU stub cell EXISTS: n=1,274 canonical, floor .588,
  V .560-.570, clean craft-bank NULL (.505, 150 proposals/0 kept, retest .90 = genuine).
  Chart: V .570 + null note. mcsweeneys_rejections (n=30) = editor meta-discourse, NOT labels.
- HUMOR verdict = contest corpus EXISTS (Wergle/Hull/Erma, n=929, prize tiers) but
  adjudicated NON-CELL: length-ctl floor ~.45, text +.001, V≈0 = the predicted
  taste-residual null. Chart shows the null as a finding.
- CODE curation = competitions DONE (y = same-approach-as-editorial, candidate-only):
  AC strict-L1 bank ens .731 / dense FT .690 (ModernBERT, grouped) / exec-V ~chance
  ("no V layer"); CC .729/.680; CF .627; LC .568. BANK > DENSE. The remembered "comp
  dense .7112" = sk3-only Llama run, laptop artifact NOT FOUND — treat .7112 unverified.
- MATH curation = AoPS same-approach (ONLY label; no official-vs-forum, no elegance-y):
  V .64-.71 (lexicons .706), TF-IDF .726, C .777/.769 Llama GROUPED. Secondary thanks_resid:
  length .594, elegance-A ρ+.066/AUC .544 first live A-signal; dense not run.
- ACADEMIA curation ALT = best_papers v2 BUILT n=11,613/960 awards: floor .603/V .681/
  bge .751 (+.15 headroom) — NO A-bank yet (V10, #52). oral/spotlight remains the complete
  alt (.550/.567/.588).
- JOURNALISM community = 52,112 tweet-engagement rows collected, unlabeled (V9, #51;
  use sum_likes, n_tweets right-censored).
- N&C co-signing: never attempted (V8 stands).
2026-07-28 late: humor contest tiers RECLASSIFIED curation (pool-picked = revealed, per
user); scaling-curves appendix added (figures/scaling_curves_llama8b.{json,pdf}, 14 sweeps
harvested from sk3); Fig 3 trimmed to the 2 significant panels, GitHub opposite-sign
finding in caption + rebuild = V11 (#53); Table 1 → TikZ grid; Table 2 restyled + math row.
IN FLIGHT: (b) lost-cells hunt: amazon book lists (fiction verdict), joke
competitions (humor verdict), CC/LC/CF competition editorial answers (code curation —
user: done for a while, comp dense .7112 known), AoPS editorial answers (math curation),
best-paper awards, Twitter likes. Fill cells + charts when they report.
2026-07-27. Single source of truth for which (field × preference-type) cells are DONE /
IN FLIGHT / QUEUED, matched to Table 1 + the three cross-field charts
(`latex/paper-3__articulation-gaps-vat/figures/make_vat_ycharts.py`) and the task list
(#V-tasks). Update this file whenever a cell lands. Legal tracked separately (App. A of
paper; excluded from charts by user rule).

Status codes: ✓ done (V/A or V+A + T where noted) · ▶ in flight · ◻ queued (task#) · — not applicable/uncollected.

| field | verdict | curation | community/crowd |
|---|---|---|---|
| Creative writing | — no labels (backlog idea: publisher/venue rejections) | ✓V .570 (Wigleaf) · ◻ A bank (V5) · ◻ T grouped | ✓ V .496/VA .520† /T .820p (WritingPrompts) · ◻ mature A bank (V5) · ◻ T grouped re-fit (V4) |
| Humor | — no labels | ✓V .606 (caption finalists) · ▶ A scoring NOW (sk3 GPU2, V1) · ◻ T (V4) | ✓ V .574/VA .564†/T .824p (reddit jokes) · ◻ mature A (V5) · ◻ T grouped (V4) · ▶ caption crowd-C A same scorer (V1) |
| Journalism | ✓ V .628/A-only .648/T .679p (press k≥3) · ◻ V+A stack fit + T grouped (V4) | collected (homepage placement; outlet-held-out census .593/llm .531; T .824 groupsplit prov) · ◻ integrate + stack | — most-read unlinked (backlog) |
| Peer review (Academia) | ✓ V .613/VA .690 · ▶ T training NOW (chain run 6) | ✓ V .550/VA .567 · ▶ T training NOW (chain run 5) | ✓ V .705/VA .761/T .896 interim · ◻ clean-eval scoring pass (post-chain) + topic-strat robustness |
| Regulatory comments | ✓ V .591/VA .594/T .623 (interim) · ◻ clean-eval pass | — | ◻ co-signing build (V8, #50) |
| Patents | ✓ V .601/VA .626 · ◻ T on claim-fell (V4 — NO honest dense exists) | — | ◻ forward-citations build (V7, #49) |
| Math Q&A | ✓ V .565/VA .673/T .794 (grouped, clean-eval) | — | ◻ un-binarized vote rebuild from sk3 raw (V2, #44) + Gemma re-score |
| Software code | ✓ V .576/VA .592/T .584 (repo-disjoint) | — | ◻ SO votes cell (V6, #48) |

†= May-2026 noah floor harness; verified (refit agent 2026-07-27) ~.10–.15 below mature
banks on bridge tasks — never present as comparable to mature-bank rows without the flag.
p = provisional (ungrouped/unverified split); mature standard = Llama-3.1-8B LoRA grouped
(methods/dense/run_dense_standard.sh recipe).

## Off-grid (tracked, not charted)
- Legal 12-domain ladder (App. A): done V/A per domain + Llama T; V+A stacks NEVER fit — queue if paper needs.
- Legal district citation-percentile (V3, #45): build queued.
- mathlib maintainer accept (math's 2nd verdict): V .680/VA .668/T .770 (split unverified — V4 verify).
- mathlib friction y: Llama dense .623 only.
- AoPS same-approach (match label, not preference): grouped Llama .769.
- Claim-matching (EXCLUDED from grid per user — examiner-citation task, closed).
- Peer curation/revealed + N&C 3-y contrasts: y-preference figures (Fig 3 + ypref_contrast).

## DENSE CHAIN — CLEAN-EVAL FINAL (2026-07-28; canonical per the standard)
| run | T eval (CANONICAL) | T test | vs V+A | band |
|---|---|---|---|---|
| peer verdict | .753 | .751 | .690 | +.06 |
| peer curation | .593 | .588 | .567 | +.03 |
| peer revealed | .871 | .896 (optimistic, n_eval 223) | .761 | +.11 |
| N&C responded | .808 | .825 | .635 | +.17 |
| N&C outcome | .622 | .623 | .594 | +.03 |
| N&C agree | .566 | .639 — DIVERGES, agree-y instability again (n_eval 505, docket-skewed) | .551 | +.02–.09 wide |
eval≈test everywhere except agree (report agree with BOTH numbers, never one).
sk3: methods/dense/eval_pass_results.json. Superseded interim table below.

## (superseded interim) DENSE CHAIN COMPLETE (2026-07-28 00:27; all test-split interim, eval pass running)
| run | T (test) | vs V+A | band |
|---|---|---|---|
| peer verdict (title-grouped) | .751 | .690 | +.06 |
| peer curation | .588 | .567 | +.02 |
| peer revealed | .896 | .761 | +.14 |
| N&C responded (docket-grouped) | .825 | .635 | +.19 |
| N&C outcome | .623 | .594 | +.03 |
| N&C agree | .639 | .551 | +.09 |
Old venue-confounded peer ~.77 superseded by honest grouped .751.

## HUMOR CAPTION MULTI-Y COMPLETE (2026-07-28, V1 CLOSED) — hole-(b) contrast #3
`datasets/humor/caption_multiy/cap_multiy_results.json`; anchors pos/mid/scram
.667/.378/.121 all shards; NA .21; gkf ≈ sgkf (estimator-robust, unlike N&C).
| y | n | V | A | V+A | A−V |
|---|---|---|---|---|---|
| finalist-B (hardneg headline) | 5,218 | .625 | .630 | .651 | +.005 |
| finalist-B (full pool, inflated secondary) | 18,838 | .632 | .716 | .728 | +.083 |
| crowd-C (median split, ≥100 votes) | 10,893 | .527 | .648 | .649 | **+.120** |
Reading (descriptive): crowd humor = V-at-chance but substantially ARTICULABLE (+.12);
editor-pick-vs-crowd-loved = near-null seam. Mature bank .648 on crowd captions validates
V5's floor-harness hypothesis (jokes .564† was instrument floor). Charts updated.

## ROUND-2 DENSE: captions DONE (2026-07-28) — BANK > DENSE on BOTH caption y's
finalist-B T .595 (test, contest-grouped) vs V+A .651 (Δ−.06); crowd-C T .547 vs V+A .649
(Δ−.10). Second bank>dense exemplar after competition code — dense NOT a universal upper
bound (short texts; Gemma-31B-judged 364-criteria bank out-extracts LoRA Llama-8B).
Single seed, test-split; eval-scoring pass optional. cw_full (96K WritingPrompts grouped)
training now.
## (launched) ROUND-2 DENSE CHAIN LAUNCHED (2026-07-28): cap_finalist → cap_crowd → cw_full
(WritingPrompts clean grouped, full data) — after eval pass drains. Watcher armed.

## BALANCING-APPENDIX AUDIT CORRECTIONS (2026-07-28, appendix_balancing.tex wired in)
- caption contests: "81-83" = rows PER contest (IDs 615-842, 227 contests) — not contest ids.
- mathlib: y=(n_review_threads==0) is the RETIRED friction label, NOT accept/reject —
  never describe the verdict cell with it.
- P2F clean = 6,557 artifact-corroborated (6,895 was handoff narrative; figures corrected);
  F2P = 996 corroborated (1,174 from uncommitted rescore harness).
- FICTION-CROWD PROTOCOL MISMATCH (user decision needed): 5,000-dp V/A cell sampled from
  litbench-to-train (upvotes>100, 31% pos, no prompt grouping) vs dense T trains on
  writingprompts_modeling_clean (score>=10, 50/50, prompt-grouped) — different thresholds
  AND populations; community-chart CW row is currently a mixed-protocol composite.
- CF competition: .627 pre-enrichment vs .686 post — registry cites .627, decide before quoting.

## In-flight jobs right now (sk3) — updated 2026-07-29
- code-review dense v2 seed chain GPU2 (wrapper 1256192): seed 1 finishing, seed 2 (~11h)
  auto-launches; waiter armed. Registry gets a seeds-1/2 addendum when they land.
- (Earlier entries here — GPU1 chain, caption scorer — COMPLETED and reported above.)

## Task-list index
#43 V1 captions multi-y (▶) · #44 V2 math.SE votes · #45 V3 legal citation-pct ·
#46 V4 dense-standard re-runs+queue · #47 V5 CW/humor mature banks · #48 V6 SO votes ·
#49 V7 patents forward cites · #50 V8 N&C co-signing.

## 2026-07-28 — V12 humor-verdict downloads, pass 1 landed
- **#HashtagWars (SemEval-2017 T6)**: `datasets/humor/hashtagwars/` — 12,734 labeled tweets, 112 hashtags (train 11,325 / trial 660 / gold 749). Labels: 2=winner (112), 1=top-10 (997), 0=not (11,625). `load.py` verified. → becomes the humor VERDICT cell once V/A instruments run.
- **Style Invitational**: `datasets/humor/style_invitational/style_invitational.jsonl` — 9,637 entries / 316 weeks (322 winner / 1,227 runnerup / 8,088 HM). ~1,100 remaining weeks resumable (see its README).
- Landscape card updated (was "identified; import queued"); verdict-chart humor cell now "12,734 labels downloaded; instruments queued".
- Newsjack rejects + SNL cut-for-time agent still running.
- Patents: EC dataset built (27,950 rows); FA rebuild finished on sk3 — agent resumed for the dual-y comparison.

## 2026-07-28 — V12 pass 2 landed (Newsjack + SNL): downloads COMPLETE
- **Newsjack rejects**: `datasets/humor/newsjack_rejects/newsjack_rejects.jsonl` — 1,981 records: 1,857 rejected / 124 aired-claimed (producer verdict). Sources: all 855 pages of 26 comedy.co.uk threads (plain browser UA fixed the 403), NewsBiscuit + garryabbott via Wayback. CAVEAT: aired/rejected labels are heuristic regex over forum text — treat as silver; cue flags kept for re-thresholding. Cloudflare content-signal `ai-train=no` recorded in README.
- **SNL cut-for-time**: `datasets/humor/snl_cut_for_time/snl_catalog.jsonl` — 2,636 rows: 87 cut / 2,549 aired. Only 26/87 cut rows have transcript text (30/36 official uploads have captions disabled) → text-based instruments NOT viable yet at scale; catalog + resume paths documented.
- **Verdict-cell decision stands**: #HashtagWars (12,734, clean labels, full text) is the instrumented humor-verdict cell; Style Inv. tiers secondary; Newsjack silver-label; SNL text-starved.
- V12 downloads DONE → next step is V/A instruments on HashtagWars (queue with V4/V5 GPU work).

## 2026-07-28 — Codex fleet status check (deadline day)
- Only **mathlib rescore** (job task-ms52agti) produced artifacts: `datasets/math/mathlib_rescore/`
  — v2 A-bank of 18 criteria (`rubrics_v2.jsonl`: 10 fidelity-rewritten + 8 new incl.
  m11 file/namespace placement, m12 existing-decl reuse, m14 public-API surface,
  m15 proof concision, m17 attribute/instance hygiene, m18 dependency hygiene),
  `score_mathlib_gemma.py` (offline-batch Gemma-4-31B, spawn, 3 blinded anchors/shard,
  {1.0/.5/0/NA}, temp 0), `retrieval_context.py` (optional TF-IDF nearest-decl context).
  AUTHORING ONLY — no scoring launched. BLOCKED on (a) a custodian placing
  `anchors_blinded.jsonl` (merged_trivial_fix + rejected_sorry seeds) and (b) GPU room:
  sk3 has 7/8 GPUs at 115-167GB from other users; GPU2 is held by our r2 eval pass.
- The other 3 Codex jobs (CW A-bank, code-review rebuild, humor contest bank) are absent
  from the companion registry AND left no artifacts on disk → treat as LOST, not pending.
  Their paper cells stay as currently reported (CW mined-A null .505; code T .584 audited-
  provisional; humor contest cells unchanged).

## 2026-07-28 — ROUND-2 CLEAN-EVAL PASS DONE (canonical T values)
`methods/dense/eval_pass_r2_results.json` (sk3), scorer `score_eval_pass_r2.py`, GPU2.
| run | test AUC (interim) | **clean-eval AUC (canonical)** | n_eval |
|---|---|---|---|
| cap_finalist | .5947 | **.6252** | 528 |
| cap_crowd | .5472 | **.5631** | 1,098 |
| cw_full (WritingPrompts clean grouped) | .7981 | **.7801** | 9,573 |
**REVISION to the bank>dense claim.** On the canonical eval split the caption-finalist
gap nearly closes: V+A .651 vs T .625 (Δ +.026, n_eval only 528) — quote as
"bank ≳ dense", NOT a clear win. Crowd captions still a clear bank win: V+A .649 vs
T .563 (Δ +.086). Competition code (bank ens .731 vs ModernBERT .690) remains the
strongest bank>dense exemplar. Charts updated (curation humor T .595→.625; community
humor T .547→.563; community CW T .820p→.780).
CW community T is now a real Llama-8B clean-eval number but stays HATCHED/provisional:
the fiction-crowd PROTOCOL MISMATCH is unresolved (T trains on writingprompts_modeling_clean,
score>=10, 50/50, prompt-grouped; V/A cell drawn from litbench-to-train, upvotes>100, 31% pos,
ungrouped). Note added to the chart cell ("T/A pools differ"). USER DECISION still needed.

## 2026-07-28 — NEW V/A WAVE LAUNCHED on gpt-5.6-sol (user directive: GEPA proposer AND executor)
- `task-ms5c92zt` **HashtagWars V/A** → datasets/humor/hashtagwars/va/ — y = staff-selected
  (top-10 ∪ winner) vs not, hashtag-grouped; ~12-20 deterministic V checks + 25-40 GEPA
  comedy criteria; anchors winner/random/scrambled per batch.
- `task-ms5c9ziu` **Style Invitational V/A** → datasets/humor/style_invitational/va/ —
  TWO y's (top-tier = winner∪runnerup; winner-vs-rest), week-grouped. Framed as
  discrimination AMONG already-printed entries (a curation contrast, not good-vs-bad).
- `task-ms5c9kdd` **CW mature A bank** → datasets/creative-writing/va_bank_v2/ — closes the
  .520†-floor-harness hole AND is required to run on the SAME pool as the dense model
  (writingprompts_modeling_clean grouped) so the fiction-crowd protocol mismatch is resolved.
- Claude/Opus agent: **code-review dense standard v2** → datasets/code-review/dense_standard_v2/
  — Llama-8B LoRA to standard, REPO-GROUPED splits, 24k diff cap, len 2048-4096, pos_weight,
  select-on-eval. Replaces the defective ModernBERT .584. Audit predicts .61-.62.

## 2026-07-28 — PRE-SUBMISSION FACTUAL AUDIT (Opus) + fixes applied
CRITICAL fixed:
- **MH odds-ratio exposure was mislabeled.** Table 2 and make_topfeatures_fig.py credited
  OR 1.21 to "tests flip fail→pass". WRONG: 1.214 [1.094,1.347] p=7e-5 is the
  **pass→fail (no-regression)** exposure; the F2P direction is ~NULL (MH ≈1.06). Fixed in
  both. The appendix already said this — the table contradicted our own appendix.
- **figures/fig_topfeatures.tex + fig_datagrid.tex quarantined** → figures/deprecated/
  (they carried banned F2P "1,174" / P2F "6,895" and a LAW column in a main-body grid).
  Also moved the superseded make_vat_verdict_bars.py / make_vat_vvat_bars.py. README there.
- **patents .756 de-benchmarked** in appendix: was prescribed as "the honest comparator";
  now framed as a leakage DIAGNOSTIC with the number removed; comparator = examiner-LOO .681.
MAJOR fixed: Fig-2 caption no longer says the code gap is "closed"/"criteria exhaust the
recoverable signal" (it is a TIE, ±.03-.04, dense arm provisional); caption now names
ModernBERT for the two code cells; CW/humor verdict cells described correctly (not "dashes");
press-release ladder .705 → .679 (grouped/deconfounded) with the .705 kept as the ungrouped
sweep value; legal described as 600 V/A + 2,000 dense with folds stratified NOT court-grouped;
N&C 21→22 agencies + responded-y stated as a separate wider-pool cell; patents cohort
described as grant/abandon with a nested 59,937-claim corpus; curation caption now leads with
the plotted best-papers bar (oral/spotlight is an in-panel note, not a bar).
ABSTRACT: literal "XX"/"YY" placeholders and a truncated sentence removed; "eleven domains"
→ eight (+law in App. A); "10k-100k texts" → 1k-600k; org-size claim corrected from
"agency size ⇒ articulability" to headcount→shallow V, docket throughput→deep V, n=19.
Also: litbench dropped from a scaling-panel title (run was aborted); GitHub TF-IDF
replication p=.020 dropped (unsourced).
STILL OPEN (human decisions): landscape-figure counts may be PRE-CLEAN vs the appendix's
canonical builds — story upvotes 99,994 vs 96,080; Math.SE 100,000 vs 99,722 (v3.3);
homepage 183,708 vs v8 133,130; citation 2,387 scored of 4,663 labelled. Also Table 2's
software-code A chips (.57/.57/.56) name a DIFFERENT feature set than memory's PR top-A
(test presence .567 / test-source correspondence .564 / dependency hygiene .591), and the
CW Table-2 row rests on n≈90. CodeForces .627 vs .686 still undecided in the appendix text.
**SUBMISSION BLOCKER**: main.tex had NO BODY (intro = 1 epigraph paragraph + 6 floats, then
appendix). Opus agent drafting Methods/Results/Related work/Limitations/Conclusion now.

## 2026-07-28 — PAPER BODY DRAFTED (submission blocker cleared)
main.tex now has: completed Introduction (+ contributions), §Measuring the articulability
gap (methods), §Results, §Related work, §Limitations, §Conclusion. **47 pages, 1 overfull
(pre-existing), 0 undefined citations.** Every number sourced from the registry/figures;
nothing new invented.
BIBLIOGRAPHY WIRED for the first time: `\bibliography{../refs-shared,../../methods/metric_implementer/references}`
+ `\bibliographystyle{iclr2026_conference}`. The two-file list is required — the tacit-knowledge
canon (polanyi1966tacit, nisbett1977telling, ...) lives ONLY in metric_implementer/references.bib.
- Added an **ML/alignment block to latex/refs-shared.bib** (11 new entries: christiano2017deep,
  stiennon2020learning, ouyang2022training, bai2022training, bai2022constitutional, liu2023geval,
  kim2024prometheus, lambert2024rewardbench, rafailov2023direct, lightman2024lets, guo2025deepseekr1).
  ALL TAGGED **METADATA-UNVERIFIED** — entered from model knowledge, not fetched. Venue/year/page
  fields MUST be checked before camera-ready. zheng2023judging / gao2023scaling / agrawal2025gepa
  were NOT duplicated (they already exist in metric_implementer/references.bib).
- Fixed a pre-existing parse error in methods/metric_implementer/references.bib line 523
  (`titleced{` → `title = {` in bloxham2016lets) that was silently dropping that entry.
- Reconciled fig:vat-community caption Δ: +.28 → **+.26** (T .780 − V+A .520).

## 2026-07-29 — V4 CLOSED: code-review dense standard v2 = .5734 (NOT the predicted .61-.62)
`datasets/code-review/dense_standard_v2/RESULTS.md`; sk3 rm_out_seed42; scorer
`methods/dense/score_eval_pr_v2.py`.
| readout | value |
|---|---|
| **clean-eval AUC (canonical)** | **.5734** (n_eval 6,372) |
| test-split (interim, DO NOT headline) | .7035 |
| within-repo eval / test | .586 / .575 (n-wtd) — agree to .011 |
| trainer best eval | .5736 @ epoch 2 (agrees with scoring pass) |
**Eval-vs-test divergence (.573 vs .704) is REPO COMPOSITION, not leakage**: in-split
repo-rate-alone is .802 eval / .854 test. Trustworthy level = within-repo ~.58.
NEVER headline .7035 or the .6403 eval+test pooling.
CONSEQUENCE: vs V .576 / V+A .592 the software-code cell has **no positive taste residual**
— dense text does NOT beat the V+A stack. The old .584 was accidentally near-right
(select-on-test + 837 dup rows inflating; 512-tok ModernBERT context deflating).
CORPUS NOTE — **the 44,751 figure is dead**: /tmp/final_ladder_table.parquet et al. were
cleared on sk3 with no surviving copy, AND the CSV that actually produced .584 was
24,866 rows / 258 repos, so .584 was never a 44,751-row measurement. Rebuild from primary
source: n=63,707 (51,571 pos / 12,136 neg), 1,267 repos, 24k cap, max_length 2048,
pos_weight .235, select-on-eval. REPO-DISJOINT VERIFIED (train&eval/train&test/eval&test = 0).
Table 1 / make_datagrid_fig.py still says 44,751 for this cell → needs updating or flagging.
CAVEATS: single seed (seeds 1,2 chained, ~20h; intra-epoch oscillation .47-.59 so assume
±.02); max_length 2048 not 4096 (24k char cap only feeds ~8k chars after truncation).

## 2026-07-29 — CODEX-AS-EXECUTOR FAILED; A-scoring moved to local Gemma
The gpt-5.6-sol GEPA wave PROPOSED fine but could NOT SCORE: the Codex managed sandbox
blocks nested model inference and no API credential was present. 2 of 3 runs stopped and
said so. The HashtagWars run completed a pipeline but materialized "A" with a
**deterministic articulated-codebook executor** — NOT an LLM judge. Its V .555 / A .544 /
V+A .565 (4,228 tweets, 40 hashtags) are therefore **NOT A MEASUREMENTS — never quote**.
Its anchors did pass (.781 winner > .563 random > .528 scrambled) and its subset design
is sound, so REUSE the subset + anchor design.
SALVAGE: the banks themselves are good and label-blind —
  hashtagwars/va/rubrics.jsonl (30), style_invitational/va/rubrics.jsonl (32),
  creative-writing/va_bank_v2/rubrics_initial.jsonl + reconstructed corpus.
Opus agent now scoring all three with **Gemma-4-31B offline-batch vLLM stacked on GPU2**
(gpu_memory_utilization .55 alongside the seed-1 training job, PID 2522251 — NOT killed;
stacking per standing rule). Outputs → RESULTS_gemma.md / results_gemma.json in each dir.

## 2026-07-29 — THREE NEW A CELLS MEASURED (Gemma-4-31B, one engine, ~525K judge prompts)
Artifacts: `datasets/{humor/hashtagwars/va,humor/style_invitational/va,creative-writing/va_bank_v2}/RESULTS_gemma.md`
+ results_gemma.json; scripts `datasets/va_gemma_banks/`; raw `outputs/va_gemma_banks/`.
Grouped 5-fold CV, pooled OOF AUC. Stacked on GPU2 at util .55 alongside the seed-1
training job (PID 2522251, untouched).

### 1. #HashtagWars — THE HUMOR VERDICT CELL IS NOW REAL
n=4,228 (397 pos / 3,831 neg), 40 hashtags, hashtag-grouped.
| readout | V | A | V+A |
|---|---|---|---|
| full | .5592 | **.6350** | **.6478** |
| balanced within hashtag (n=788) | .5606 | .6131 | .6235 |
Anchors 8/8 PASS; battery pos .929 / neg .772 / scrambled .081 (pos-neg AUC .854,
coherent-vs-scrambled 1.000). NA .283.
**Replicates the caption-crowd pattern on a VERDICT y: humor is verifier-weak but
substantially articulable (+.076 A over V).** Also a clean instrument-validity result:
the discarded deterministic codebook put A at .5445/.5511 — **the LLM judge is ~.09 higher**,
i.e. the codebook understated the articulated channel badly. Evidence FOR the
LLM-judges-do-all-measurement rule.

### 2. Style Invitational — V ≥ A, the OPPOSITE pattern (a real contrast)
n=9,637, 316 weeks, week-grouped. Two y's, never merged.
| y | pos | V | A | V+A |
|---|---|---|---|---|
| top tier (winner∪runnerup vs HM) | 1,549 | **.6227** | .6090 | .6161 |
| winner vs rest | 322 | **.6334** | .6070 | .6121 |
V+A does NOT beat V here. Reading: discrimination AMONG already-printed entries by one
long-running judge behaves unlike selection-from-a-slush-pile.
CAVEAT: **anchor shard 5 never passed after 4 draws** (kept in headline; leave-shard-5-out
sensitivity: top_tier .6124/.6005/.6064, winner .6222/.6123/.6238 — conclusions unchanged).
NA .436; 4 near-constant criteria flagged (Explanation discipline, Rhyme quality,
Misdirection fairness, Phonetic/orthographic precision); "Meter and scansion control" is
separately NA .966 but not flagged.

### 3. Creative writing — FIRST REAL CW A NUMBER; floor-harness hypothesis CONFIRMED
n=2,000 (1,009/991), 1,500 prompt groups, prompt-grouped, on the **canonical
writingprompts_modeling_clean** (NOT the reconstruction) → **the fiction protocol mismatch is
resolved**: A and dense now share a pool. Truncation >6,000 chars → first 3,600 + marker + last 2,400.
| V | A | V+A |
|---|---|---|
| .6039 | .6053 | **.6266** |
vs the May-2026 floor harness V+A .520† → **+.107, squarely inside the predicted .10-.15
floor-harness gap**. With dense clean-eval T .780 on the same pool, the CW band is ≈ +.15.
CAVEAT (real): the CW anchor pos/neg half is **REVERSED** (pos .686 < neg .744, AUC .403).
Coherence sensitivity is perfect (scrambled .008, AUC 1.000). Read CW A as "judge applies
the criteria to real prose and is not collapsed", NOT as a certified pos/neg separation.
NA .018, no near-constant criteria.


## 2026-07-29 — INDEPENDENT AUDIT of the session's measurements: PASS (+ corrections)
Recomputation audit: all download counts, all nine Gemma AUCs (HashtagWars V+A reproduced
to 6 dp), code-review .5734/.7035/63,707/repo-disjointness all verified digit-for-digit.
Corrections/annotations from the audit:
- **Gemma AUCs are sklearn-version-locked**: exact reproduction needs sklearn 1.9.0 (gemma4
  env); 1.7-1.8 give −.006 on HW V+A (GroupKFold assignment differs). Over 20 random
  group-fold draws HW V+A mean .6356 ± .0062 (reported .6478 at the optimistic edge) →
  quote these cells to ±.006, pin sklearn in any repro note.
- **Style Invitational has ~11% parse artifacts** (1,088 rows: section labels/bylines/
  fragments, 93% in the HM negative class). Sensitivity excluding them: top-tier
  .5979/.5828/.5964, winner .6171/.5980/.6083 — ~.02 lower but **V > A ordering survives**.
  Clean the negative class before quoting this cell in a paper. Also: scored all 316 weeks
  (superset of the 110-week precommitted design; no 110-week readout run).
- CW grouping is nearly item-level (1,193/1,500 prompt groups are singletons, max 16).
- Newsjack: ~17% of rows trip chatter cues — needs a joke-vs-chatter filter before ANY use.
- Codebook run files (hashtagwars/va/RESULTS.md) now carry a superseded-by pointer.
- Seed 1 tracking ~.03 BELOW seed 42 (epoch-1 best .539, latest .550) → multi-seed mean
  likely < .5734, which STRENGTHENS the no-positive-taste-residual conclusion.

## 2026-08-05 — code dense v2: multi-seed clean-eval + manual score audit (V4 addendum)

**Seed chain finished Jul 30** (`PR_DENSE_V2_ALL_DONE`, all three EXIT 0). Clean-eval pass
(score_eval_pr_v2.py, all rm_out_seed*) rerun 2026-08-05 on sk3 GPU0:

| seed | clean-eval AUC (n 6,372) | test AUC (n 6,298) |
|---|---|---|
| 42 | .5734 | .7035 |
| 1  | .5535 | .6706 |
| 2  | .5713 | .6868 |

3-seed mean **.566 eval / .687 test** (COMPLETE 2026-08-05, eval_pass_results.json all
seeds); pairwise pred corr ≈ .74; 3-seed ensemble .5715 (no gain). All three eval AUCs
sit below V+A .592 → **no positive taste residual, now multi-seed robust**. Test-side
spread .671-.704 confirms the eval/test gap is repo composition, not a seed fluke.

**Manual score audit** (full report: `notes/2026-08-05__code-dense-score-audit.md`;
artifacts sk3 `.../dense_standard_v2/audit_manual/`):
- **H5 CONFIRMED PRIMARY — the `text` field is a bare unified diff.** 100% of eval rows
  start `diff --git`; no PR title/body/review thread/CI status/author/timestamp
  (Title: 0.5%, Description: 1.3%). The no-residual claim must be scoped:
  "dense adds nothing over V+A **on diff-only input**" — NOT a property of code review.
- **H1 CONFIRMED BINDING on the remainder** — merge decided by unobservables: on
  android-maps-utils' 57 workflow-bump PRs, pr_number (recency, absent from text) AUC
  **.806** vs dense **.339**; ~10% of jac≥.9 near-dup diff pairs carry opposite labels.
- **H3 REJECTED** — scores healthy (prob .0002–1.0, sd .205); but calibration flat below
  decile 8 (all lift in top two deciles).
- **H2 (truncation) real in size, NOT binding** — 72.4% of rows exceed the 2,048-token
  window (50.1% at the 24,000-char build cap), yet dense AUC truncated .5739 vs
  untruncated .5735; length-alone AUC .5185.
- **H4 secondary** — within-repo (55 repos n≥30 both-class): dense median .5986 /
  n-wtd .5881 beats char-length .4730 in 41/55 (Wilcoxon p<1e-4); per-repo spread mostly
  sampling noise (null sd .117 of observed .152). TF-IDF+LR within-repo median .5469 —
  8B LoRA only ~.05 above bag-of-words.
- **Contamination found:** five `*_deep_NNN` repos are 100%-merged (462 eval rows, 7.3%)
  and own 83% of the confident tail; excluding them: pooled eval .5734 → .5877. Also 35
  both-label exact-duplicate rows.

**Follow-ups (need user sign-off — instrument-input change):** (1) rebuild `text` =
PR title + body + review thread (+ CI conclusion), truncate the diff not the discussion,
identical recipe + same repo-disjoint splits; (2) hygiene: drop `*_deep_NNN` all-positive
repos + 35 dup rows; (3) report recency channel as a named non-textual predictor.

## 2026-08-05 — Layer-1 nonlinear stack: peer verdict PILOT (taste-decomposition D1)

Design: notes/2026-08-05__taste-decomposition-design.md; full pilot write-up:
notes/2026-08-05__layer1_peer_verdict_pilot.md; code methods/taste_decomposition/.
Gate PASSED at machine precision (V .6128 / A .6835 / VA .6896 reproduced exactly).

| quantity | value |
|---|---|
| VA_lin | .6896 |
| VA_nl (seed 0; seeds 0-2: .6876/.6777/.6828) | .6876 |
| **Δ_interact** | **−.0020, CI [−.0091,+.0050] → NULL** |
| **Δ_beyond (T .753)** | **+.0654 → clears Layer-3 gate (>.02)** |

Read: the peer-verdict residual is NOT nonlinear combination of articulated criteria —
SHAP interaction list is instrument redundancy, not synergy (top pair v_kw_code ×
data/code-availability rubric = same construct twice). Two prereg carries: VA_nl =
3-seed mean (spread .0099 > any plausible interaction gain); Δ_beyond needs SAME-ROWS
dense rescore (T eval rows ≠ 6,030 A/V rows; small GPU job, queue before confirmatory).

## 2026-08-05 — Layer-1 nonlinear stack: five Gemma-bank cells (D1 rollout, gates 15/15 PASS)

Full note: notes/2026-08-05__layer1_gemma_cells.md; JSONs methods/taste_decomposition/results/.
sklearn 1.7.2; family1 gates reproduce ±.006, caption gates machine-exact (imports published code).

| cell | VA_lin | VA_nl (spread) | Δ_interact [95% CI] | T | Δ_beyond |
|---|---:|---:|---|---:|---:|
| cw_community | .6301 | .6207 (.0131) | −.0095 [−.027,+.016] null | .7801 | +.159 |
| hashtagwars_verdict | .6419 | .6301 (.0075) | −.0118 [−.039,+.008] null | — | — |
| style_inv_toptier | .6174 | .6651 (.0005) | **+.0477 [+.035,+.061]** | — | — |
| cap_crowd | .6485 | .6656 (.0036) | **+.0171 [+.008,+.022]** | .5631 | **−.103** |
| cap_finalist | .6508 | .6800 (.0022) | **+.0292 [+.008,+.048]** | .6252 | **−.055** |

Readings (descriptive):
- FIRST NON-NULL Δ_interact cells (3/5, all positive): nonlinear aggregation of the SAME
  articulated matrix beats linear. Style Inv's +.048 is LENGTH/FORMAT-mediated (v_char_count
  in 7/10 top SHAP pairs; V-only interaction gain +.040) → surface-feature nonlinearity,
  Track-B/spurious territory, NOT tacit criterion combination. cap_finalist's +.029 has
  near-zero V-only interaction → plausibly genuine A-criterion synergy (translatability ×
  quality-over-pandering).
- Both caption cells now have **VA_nl > T** (Δ_beyond −.055/−.103) → "bank > dense" on
  short texts SHARPENED: the articulated channel exceeds the dense bound once aggregated
  nonlinearly.
- CW community keeps a large residual (+.159 after nonlinear stack) → prime Layer-3 target
  alongside peer verdict and N&C responded.
- CAVEAT → prereg carry #3: bootstrap CIs are row-level; coarse-grouped cells (40 hashtags /
  316 weeks / ~225 contests) need a group-level bootstrap re-check before confirmatory
  quotes (narrowest margin +.008 may tighten).

## 2026-08-05 — code dense v3 (enriched text): backfill DONE, seed-42 TRAINING (D0)

Build report: notes/2026-08-05__code-dense-v3-enriched-build.md (round 1 + backfill round).
sk3: datasets/code-review/dense_standard_v3/. v2 untouched.

- Body backfill via GraphQL variant of fetch_pr_descriptions.py (separate output file,
  original pr_descriptions.csv.gz READ-ONLY): 23,601 attempted / 98.6% succeeded / 340
  "pr not found" (deleted PRs); 10.3% genuinely-empty bodies (count as covered — real signal).
- Coverage (gate = fetch attempted+succeeded): overall .909, **eval .976**, test .943.
  Diff-only rows now 9.5% (was 100% in v2 — the whole point).
- Hygiene identical to round 1: −37 `_deep` repos (3,666 rows), −122 both-label dup-diff
  groups; 59,111 rows / 1,228 repos; recency eval AUC ~.50 pooled (repo-local channel only).
- Training LAUNCHED: sk3 GPU1 PID 1809024, recipe byte-identical to v2 (seed 42,
  select-on-eval), ~6-8h ETA.
- **Baseline discipline for the Δ readout**: v3 eval = v2 eval minus hygiene drops
  (5,822 vs 6,372) → the honest reference is the v2 models RESCORED ON THE SAME ROWS
  (expected ≈ .5877 per audit), NOT .5734. Matched rescore of v2 seeds 42/1/2 on v3
  eval+test rows queued on GPU0 → dense_standard_v3/matched_baseline_v2seed42.json.

## 2026-08-05 — code v3 MATCHED BASELINE (v2 models on the 5,822 v3 eval rows)

dense_standard_v3/matched_baseline_v2seed42.json (+GPU confirm file); method = exact
subsetting of stored best_model preds (validated: reproduces all six published seed×split
values exactly; seed-42 additionally GPU-rescored from scratch → identical to 4 decimals).

| seed | matched eval (n 5,822) | matched test (n 5,630) |
|---|---:|---:|
| 42 | **.5851** [.568,.603] | .6618 |
| 1 | .5632 | .6270 |
| 2 | .5841 | .6478 |

**Quote discipline: v3 must beat .5851 eval / .6618 test (seed band mean .5775). Quoting
v3 against .5734 would credit enrichment with ~+.012 of pure hygiene.**

Compositional finding (all 3 seeds consistent): hygiene moves eval UP (+.010..+.013) and
test DOWN (−.039..−.044) — the all-merged `_deep` repos were MISRANKED in eval (mean prob
.497 vs .610 ordinary) but correctly-high in test. Lesson: cross-corpus AUC comparisons on
this dataset must be ROW-SET-MATCHED or they mostly measure which degenerate repos survived.

v3 training healthy at 49 min (GPU1 PID 1809024, past first validation checkpoint).

## 2026-08-05 — Layer-1 wave 2: peer curation + peer revealed (gates machine-exact)

Note: notes/2026-08-05__layer1_peer_curation_revealed.md; JSONs methods/taste_decomposition/
results/. Group-level (ntitle) bootstrap = PRIMARY per freeze #3.

| cell | VA_lin | VA_nl (±spread) | Δ_interact [group CI] | T | Δ_beyond |
|---|---:|---:|---|---:|---:|
| peer curation | .5669 | .5588 (.0035) | −.0081 [−.015,+.009] null | .593 | +.034 |
| peer revealed | .7606 | .7667 (.0028) | **+.0062 [+.0003,+.020] P(>0)=.98** | .871 | **+.104** |

- Curation: verdict-pilot pattern repeats (interactions = length collinearity + A×V
  redundancy). Δ_beyond +.034 ≈ all of Δ_total.
- Revealed: small but group-CI-positive Δ_interact, A-side gain with NEGATIVE V-only gain →
  candidate GENUINE interaction per the routing rule; tiny (~.006) and rides the standing
  topic-floor caveat (IMPACT cell). Δ_beyond +.104 → joins the Layer-3 candidate list
  (behind CW community +.159, peer verdict +.065), subject to topic-strat robustness.

## 2026-08-05 — Layer-1 COMPLETE CENSUS (waves 1-3; 13 cells run, 1 principled skip)

JSONs: methods/taste_decomposition/results/*_layer1.json (agents' consolidated notes for
N&C + wave 3 lost to a host restart — JSONs are the source of truth; gates all machine-
exact or ±.006 unless flagged). Group-level bootstrap PRIMARY throughout.

| cell | VA_lin | VA_nl (±) | Δ_interact [group CI] | V-driven? | T | Δ_beyond |
|---|---:|---:|---|---|---:|---:|
| **N&C responded** | .6350 | **.7244** (.008) | **+.089 [+.066,+.118]** | YES (V_nl .596→.709) | .808 | **+.084** (was +.173) |
| N&C outcome | .5937 | .6102 (.004) | +.016 [+.0006,+.039] | no (A-side) | .622 | +.012 |
| N&C agree | .5510 | .5844 (.009) | +.033 [−.011,+.097] null | mixed; A_lin .524→.568 | .566/.639 | −.018 eval/+.055 test |
| mathlib verdict | .6683 | .6721 (.038) | +.004 null (tiny n) | — | none (unverified) | — |
| patents verdict | .6233 | .6256 (.001) | +.0023 [+.0009,+.0056] stat-pos, NEGLIGIBLE (n=59,937) | — | none | — |
| press verdict | .6712 first-fit | .7011 (.011) | +.030 (V_interact +.036 → surface route) | YES | .679p | **−.022p** |

- **N&C responded headline: HALF the +.17 flagship gap is nonlinear recombination of the
  EXISTING 203 features — and it is V-DRIVEN** (V-only stack .596→.709). Per the routing
  rule this is verifier-side nonlinear structure (surface/route-B territory), not tacit
  criterion combination. Δ_beyond +.084 remains the #2 Layer-3 target after CW +.159.
- press verdict: FIRST-EVER combined V+A stack (.6712, closes the old V4 press-stack gap)
  BUT gate ambiguous (V reproduced .617 vs published .628; A .669 vs .648; k≥3 cache) —
  treat all press Layer-1 numbers as PROVISIONAL pending a gate check. Nonlinear stack
  .7011 EXCEEDS provisional T .679 → third "bank > dense" candidate.
- patents: textbook null — statistically positive only because n=60K; magnitude .002.
- **math.SE SKIPPED (correctly)**: published V .565/VA .673 is QWEN-judged (a01-a14 bank,
  judge_a_metrics_sk3.py); the Gemma re-score (V2 #44) never ran. DECISION NEEDED:
  commission Gemma re-score (recommended; already-queued work) vs waive judge rule with
  explicit flag. NEVER pool a Qwen-judged cell with the Gemma cells.

**Layer-1 program picture (13 cells): Δ_interact null in 7, negligible-positive in 1
(patents), surface/V-driven in 3 (Style Inv, N&C responded, press), candidate-genuine in
3 (cap finalist, cap crowd, peer revealed; N&C outcome borderline). Big residuals after
Layer 1: CW community +.159, N&C responded +.084, peer verdict +.065, peer revealed +.104
(topic-floor caveat). Bank>dense (VA_nl>T): cap crowd, cap finalist, press(prov).**

## 2026-08-05 — Layer-3 closure ROUND 1 (peer verdict, dual-track pilot) — INFORMATIVE NULL

Report: notes/2026-08-05__layer3_round1_peer_verdict.md; artifacts methods/taste_decomposition/closure/.
Machinery validated: misrouting 1/25 (4%, arbiter upheld auditor on the one dispute);
anchors PASS (coherent-vs-scrambled .972); 0/25 collapsed; 151,650 judge calls, 16 min GPU0.

- **Track A (14 new real criteria): closure NOT moving — MONITOR VA_nl +.0003 (ε=.005).**
  The null is informative: the new criteria DID hit their label-blind target (alone .620;
  VA_nl→dense Spearman .488→.522) yet bought ~zero label AUC → the articulable part of
  what dense perceives was ALREADY PRICED INTO the 154-bank. Round 1 = first of two
  consecutive sub-ε rounds for saturation.
- **Track B (11 nuisances incl. the arbiter-reroute): discount ≈ ZERO** — spurious-alone
  .604, but joint-stratified Δ_adj +.0895 ≈ pooled Δ +.0878. Shortcuts don't explain the
  peer residual either.
- **Both deflations have now failed on peer verdict** (Layer-1 Δ_interact −.002 null;
  round-1 mining null; shortcut discount null) → the residual is behaving like genuine
  tacit signal. Round 2 (interaction-shaped criteria; salvage rewrite included) decides
  saturation.
- Same-rows T (freeze #2) done for peer verdict: **.777** on 1,244 dense-held-out
  population rows (registry .753 = model's own eval split). Honest best-powered level:
  **Δ = +.088 (n=1,244)**. Closure-split Δ levels (+.136) are NOT comparable to Layer-1
  Δ_beyond — only round-over-round change is interpretable.
- **LANDMINE → prereg amendment**: 943/1,192 MONITOR rows were in the dense TRAIN split
  (T-on-MONITOR .857 contaminated). Confirmatory design MUST define MONITOR inside the
  dense-held-out rows.
- Round-2 lever: composite criterion P05 (ambition × evidence-specificity) strongest new
  feature by ~2× → push interaction-shaped proposals.

## 2026-08-05 — Layer-3 closure ROUND 2 (peer verdict): CURVE MOVED — round-1 null was premature

Report: notes/2026-08-05__layer3_round2_peer_verdict.md. Saturation NOT declared (1 of 2
consecutive sub-ε rounds); pilot continues to round 3 (cap 5).

| | r0 | r1 | r2 |
|---|---:|---:|---:|
| VA_nl MONITOR | .6633 | .6635 | **.6730** |
| honest Δ (n=1,244, T .7769) | +.0925 | +.0878 | **+.0840** |
r0→r2 gain +.0095 [+.0023,+.0170] P(>0)=.996 — REAL.

- **Proposal SHAPE is a hidden design parameter (~30× gain difference)**: round 2's only
  material change was interaction-shaped/composite criteria, and it is the round where
  VA_lin FELL while VA_nl ROSE (+.034 nl premium in the new A block) — the mined signal
  is combination-shaped. Confirmatory cells must FIX proposal shape in advance or curves
  aren't comparable. ρ(VA_nl, dense): .488→.522→.531 (mining keeps hitting its target).
- Arbiter salvage validated: resource-magnitude (exiled, .527 as nuisance) → artifact-
  REUSABILITY rewrite (scale-free) = 2nd-strongest round-2 criterion (.5614, ρ .229).
- Track B at 22 nuisances: alone .630, discount STILL ≈0 (Δ_adj +.0821 vs +.0840).
  Sentence complexity .5631 ≈ raw length .5672 → "length" may really be SYNTACTIC
  DENSITY. Non-idiomatic English strongly anti-predictive (.4384).
- Freeze flags: (1) no protocol trigger for sign-contradicting criteria ("restraint"
  alone .458, ρ −.169 vs its quality rationale); (2) nuisance-vs-merit boundary (writing
  fluency 2nd-strongest nuisance) is a substantive call, not a formality.
- Instruments: anchors 1.000, 0/25 collapsed; cumulative judge calls 303,300.

## 2026-08-05 — Layer-3 closure ROUND 3 (peer verdict): flat again; ROUND 4 DECIDES

Report: notes/2026-08-05__layer3_round3_peer_verdict.md. Curve: VA_nl .6633/.6635/.6730/
.6731; honest Δ +.0925/+.0878/+.0840/+.0811. Gains sub-ε [yes, no, yes] → trailing run 1;
round 4 decides (cap 5).

- **Round-2 "shape" conclusion CORRECTED**: round 3 was 14/15 composite (protocol
  deviation, honestly reported — Amendment 2 required 10/15; scores kept, freeze-before-
  eval) and returned +.0001 with NO nonlinear premium (.6187 lin → .6147 nl). Composite
  SHAPE does not drive closure gain — round 2's +.0095 was the CONTENT of those specific
  relations. 
- **Audit: 0.0% misrouting** (3-auditor record 4.0/4.0/0.0%); both planted shallow-probe
  pairs caught; belief-change salvage rewrite ROUTED INTO BANK (arbiter rewrites 2-for-2).
  One nuisance criterion collapsed (content-warning, 99.6% zeros — corpus fact) → collapse
  gate now programmatic.
- **Track B at 31 nuisances: .712 alone — approaching T .777.** Δ_adj +.1146 NOT quotable
  (stratifying on a .712 nuisance model ≈ conditioning on the label). Defensible claim =
  the negative one, 3 rounds running: discounting does not shrink the residual (+.0811).
- **AVAILABILITY CHANNEL IS BIPOLAR**: public repo URL .5934 (strongest single channel in
  pilot) vs anonymised/on-acceptance boilerplate .4778 — the bank's availability criteria
  can't see the live-link-vs-placeholder detail whose sign flips the channel. Actionable
  for the bank and for the paper's V/A story.
- **PROGRAM-LEVEL PATTERN**: ρ(VA_nl, dense) climbs every round (.488→.522→.531→.541)
  while Δ moved only −.0114 total, and new-criterion strength decays (best ρ .362→.337→
  .268). Articulation keeps capturing what the dense model DOES without capturing what
  predicts y — the outcome-relevant edge is not the articulable part. Judge calls 454,950.

## 2026-08-06 — Layer-3 closure PILOT CLOSED: SATURATION at round 4 — PLATEAU Δ = +.081

Final report + pilot summary: notes/2026-08-05__layer3_round4_peer_verdict.md; canonical
JSON methods/taste_decomposition/closure/round4_results.json.

Curve: VA_nl .6633/.6635/.6730/.6731/.6723; honest Δ +.0925→+.0807 (T .777, n=1,244).
Gains [+.0003, +.0095, +.0001, −.0008]; trailing sub-ε run 2 → saturation. 56 mined
criteria, 606,600 judge calls, total closure −.0118. Four independent corroborating
diagnostics all decayed to floor (best-new alone .621→.550; best ρ-vs-dense .362→.123).

**PILOT BOTTOM LINE (exploratory, pre-GEPA): the peer-verdict taste residual survived
Layer 1 (interact null), 4 rounds of dual-track mining (closure .010 total), and
shortcut discounting at 11/22/31/40 nuisances (null throughout). Plateau Δ +.081 is the
tightened UPPER BOUND — not a point estimate (dense is itself a .777 instrument; some Δ
is dense idiosyncrasy no articulation should capture).**

- **Articulation-prediction divergence** (program-level): ρ(VA_nl, dense) climbed .488→
  .541 through r3 (stopped .537 at r4) while Δ closed only .012 — mining captures what
  dense DOES, not what predicts y. Open alternative: the remainder needs a different KIND
  of articulation than criterion-proposal (next-study question), or is genuinely tacit
  for this instrument class.
- Track B final: 40 nuisances .713 alone (92% of chance→T distance); discount NULL at
  every set size. NEVER quote Δ_adj +.110 (stratifying on a .713 model ≈ conditioning on
  label). Instrument findings: availability channel BIPOLAR (live URL .593 vs anonymised
  boilerplate .478 — bank can't distinguish); surface fluency strongest NEGATIVE channels
  (non-idiomatic .438, passive voice .439).
- Open flag: r4 pos/neg anchor separation inverted (.361, ~1.3 SE at K=12; scrambled
  .997 fine) on a self-criticism-heavy Track A — either noise or "accepted papers
  volunteer LESS self-criticism". Recorded, does not touch the verdict.
- First A-routed collapse: seeds/splits-variability criterion 99% zeros (corpus fact);
  collapse gate now programmatic.
- Audit across 4 independent auditors: 4/4/0/4% misrouting; all planted probes caught;
  arbiter rewrites 2-for-2 bank-worthy; asymmetry principle (shape features manufacture
  false closure) → freeze.
- **Recommended freeze parameters (report §5e)**: MONITOR ⊂ dense-held-out; drop
  composite quota; anchors K≥50; matched sampling replaces decile stratification once
  spurious-alone >.65; sign-check promoted to re-audit trigger.

## 2026-08-06 — code dense v3 (enriched text) CANONICAL RESULT: eval .6488 / test .7373

sk3 dense_standard_v3/eval_pass_results.json (seed 42, clean-eval pass, per-row preds in
rm_out_seed42/). vs MATCHED v2 baseline (same rows, diff-only text): eval .5851 → **.6488
(+.0637)**; test .6618 → **.7373 (+.0755)**. Both splits agree → enrichment effect robust.

**Reading: the v2 "no taste residual in code" was INPUT-STARVATION** (audit H5 confirmed
causally). Face-value ordering now inverts (T .649 > V+A .592) BUT NO residual claim yet —
same-input rule: the code A-bank must be rescored on the enriched v3 text first, AND its
provenance (judge/protocol behind V .576/A .592) must be verified before rescoring (math.SE
lesson). Until then quote only: "enriching the input moved dense +.064/+.076 on matched
rows; the diff-only no-residual result does not generalize to the full PR object."
Remaining seeds 1-2 for v3: optional, queue if the cell goes in a headline figure.

## 2026-08-06 — code A-bank PROVENANCE VERDICT (stage 1 of enriched rescore): .592 is BROKEN

notes/2026-08-06__code-abank-enriched-rescore.md. Published code V .576 / V+A .592:
- **NO LLM JUDGE**: A = deterministic coded backend (tree-sitter/lizard/ruff/semgrep
  modules, methods/existing_metrics_runner/coded/) on bare diffs — violates the
  LLM-judges-do-all-measurement rule (the HashtagWars codebook lesson, but "no judge").
- **SPLICED PROTOCOLS**: .576 = pooled V, .592 = grouped V+A (within-protocol pairs:
  pooled .576→.632, grouped .549→.592). Understates A ~25% and mismatches the dense
  grouping.
- **~16/72 aspects are NOT code-review norms** (LeetCode outcomes, authorship/AST
  fingerprints incl. Chinese-comment ratio) — repo-fingerprinting contamination.
- **NOT REPRODUCIBLE**: V source CSV gone; good-faith reconstruction = 28,556/44,751 rows
  → grouped V+A .526 vs published .592. **NEVER quote .592 as a live baseline again** —
  historical, broken provenance chain. All V4-era code V/A rows inherit this caveat.
Stage 2 RUNNING: Gemma-4-31B on 83 judge-portable criteria over v3 enriched text (GPU6,
ETA ~2h45m); anchors K=50 PASS; note BOTH input AND instrument change vs .592 — never
difference the new numbers against it. Dense saw 2,048 tok vs A's 11.6K → dense-wins
results are conservative.

## 2026-08-06 — Swap decomposition + missing-mass retrospective (pilot post-analyses)

notes/2026-08-06__closure-swap-and-missing-mass.md; closure/{swap_analysis,missing_mass}.json.
Recompute reproduces round4_results.json to 1e-9.

**SWAP VERDICT (user question): YES in round 1, partial cancellation overall.** Pairs split
by dense correctness (D+ 77.6% / D− 22.2%): r0→r4 bank concordance +.0233 on D+ but −.0281
on D− → the D− loss cancels 34% of the D+ gain; losses land 1.64× over-represented on
dense-WRONG pairs. Round 1 = textbook swap (error-inheritance > insight-inheritance,
P(>0)=.94 → its flat AUC). Rounds 2-4 NOT swaps (r2 significantly anti-swap). Also: the
published ρ .488→.541 climb is partly monitor-contamination artifact — on honest rows the
climb is all round 1. **(ΔC₊, ΔC₋) joins every round's readout (zero extra judge calls).**

**REMAINING ARTICULABLE MASS ≈ +.003 AUC** (~4% of the +.081 residual); predicted round-5
gain +.0010 [.0000,.0048], 2% chance of clearing ε → the stop was sound. CI is readout-
noise-only (proposer variance unmeasured — pilot lacked independence). Mechanism:
**redundancy saturation, not value exhaustion** (new blocks carry alone-AUC .55-.65 but
convert 0.1-6% into stack increments). NEVER quote the power-law fit (at-bound).

**BANK-HYGIENE FINDING: the "154-criterion" peer A-bank = 95 distinct concepts (59
bit-identical duplicate columns), 54 after degeneracy screen.** Chao1 on the bank's own
construction: captured ~56% of its reachable concept space (missing mass .383). The pilot's
4 rounds ≈ DOUBLED the effective concept count (54→110) for +.012 AUC. Echoes the a-bank
degeneracy audit (mined banks 54-68% degenerate). Concept census at round 0 = new standard.

**PROSPECTIVE ESTIMATOR (for the freeze)**: P≥5 sealed independent proposers/round
(bank-conditioning moves to admission step); dedup to concept species at probe-calibrated
τ≈.78-.80; report Good-Turing missing mass + bias-corrected Chao1 (never classic f₁²/2f₂)
+ remaining-bound R̂ = [M̂/(1−M̂)]·Δ_r·λ̂; proposer-level bootstrap = the missing width;
STOP-M (R̂<ε AND M̂<.25) as an ADDITIONAL gate never replacing 2-consecutive-sub-ε; costs
~+60% scoring (species-representatives only), not P×.

## 2026-08-06 — SAME-ROWS T RESCORES COMPLETE (freeze #2 satisfied, all 6 cells + peer pilot)

notes/2026-08-06__samerows_T_rescores.md; results/samerows_T_*.json; per-row preds local
(slim) + sk3 (full). Held-out = rows NOT in the dense model's train split (~20% of each
A/V population; overlap ~.80 by construction of the grouped splits).

| cell | same-rows T (held-out) | n | vs VA_nl | Δ_beyond (matched) |
|---|---:|---:|---:|---:|
| CW community | .7967 | 408 | .6207 | **+.176** |
| N&C responded | .8167 | 1,904 | .7244 | **+.092** |
| N&C outcome | .6238 | 1,417 | .6102 | +.014 |
| N&C agree | .6034 | 1,009 | .5844 | +.019 (sits between the divergent .566/.639) |
| cap finalist | .6124 | 1,055 | .6800 | **−.068 (bank>dense, matched-rows confirmed)** |
| cap crowd | .5554 | 2,190 | .6656 | **−.110 (bank>dense, matched-rows confirmed)** |
| peer verdict (pilot) | .7769 | 1,244 | .6963 r4 | +.081 plateau |

- Confirmatory entry gate (>.02): CW community and N&C responded clear decisively;
  N&C outcome/agree do not.
- CAVEAT: CW held-out n=408 (small) — quote with CI when it headlines.
- **PROTOCOL FLAG: the N&C dense chain used selection_split=TEST (not eval)** — deviation
  from the dense standard; their eval leg is the only fully-clean holdout. Carry this
  caveat on all N&C T quotes; per-split detail in the JSONs.
- Data-location landmine: caption_multiy + most N&C v4 raw sources exist ONLY on the
  local Mac (sk3 has just dense_llama/) — populations reconstructed locally, shipped as
  slim CSVs.

## 2026-08-06 — MISSING-MASS ROBUSTIFICATION BATTERY (M1 fleet / M2 backtest / M3 recovery)

notes/2026-08-06__missing-mass-robustification.md; artifacts closure/robust_mm/.

**HEADLINE — the quotation contract fired: "no more articulable signal" is NOT quotable.**
M3 leave-out recovery (3×8 concepts, stratified, footprints removed): rediscovery .333
overall / .556 high-value (floor was ≥.70, P(≥.70)=.16), and — decisive — **lift over
never-removed retained controls = ZERO** (−.042 [−.29,+.21]; blind pairwise instrument
κ=.756 agrees exactly: .292 vs .292). Proposers generate from their own concept prior,
not from the gap: the disagreement slice does NOT steer them toward missing concepts.
**Correct claim: "Δ_plateau = +.081 is not discoverable by this class of miner"** (33%/56%
sensitivity, zero targeted-recovery lift).

**What SURVIVED and strengthened**: the remaining-VALUE bound. Fleet-based Good-Turing
(real cross-proposer recapture .20-.32, jackknife widths — first proposer-level CI)
gives remaining AUC **+.0014-.0024**, agreeing with the sequential +.0030. Species pool
is RICH (mass .42-.55; 6th proposer still adds 6.5 new species; 86% novel vs bank) but
value-poor → redundancy-saturation confirmed PROSPECTIVELY. Never quote the species-form
Chao1 (+.005-.015, unidentified at f2=6-9).

Also established:
- Depletion sanity: removing 8/54 concepts costs +.0064 honest AUC (= 54% of what 4
  mining rounds bought); re-adding the rediscovered third recovers .67 of the drop.
- **Embedding-τ CANNOT do cross-register concept identity** (bank CONSORT-register vs
  fleet ML-register: max cos .72 < τ .78; same fleet recaptures mined criteria 20-36%,
  bank 0%) → full-recall + blind pairwise = the detector.
- Estimator: upper-bound rights only (errs toward "keep mining" 3/3 backtests; point
  predictions inside noise band). STOP-M would NOT have fired on the pilot (M̂≈.45 ≫ .25)
  → saturation declarations must carry the remaining-mass estimate.
- Fleet ops: luna sufficed (8/8 calls clean, no sol); GLM Lite reachable but hard
  rate-limited (1302 storms; budget_tokens=2048/max_tokens=32000 the working config;
  wrong endpoint /api/coding/paas/v4 → use /api/anthropic/v1/messages); M3 replicates
  ran P=4/2 families, round-5-tagged P=6/3 families banked.
- Six freeze changes itemized in report §4.3 (chiefly: M3-with-control as a gate;
  no embedding thresholds cross-register; STOP-M reporting).

## 2026-08-06 — LAYER 2 COMPLETE (9 cells): grouped transfer + nuisance strata

notes/2026-08-06__layer2_robustness.md; results/layer2_*.json. VA_nl recomputes gated ≤.004
vs Layer-1 ledgers. Findings:
- **N&C docket-identity leak severe**: identity-alone .86-.92; N&C AGREE within-docket
  = .493 (CHANCE) — its entire edge is cross-docket. Read all N&C VA numbers with this.
- **Length = dominant nuisance failure** (7/63 dimension×cell failures): N&C responded
  −.073, N&C outcome −.047, cap finalist −.036 stratified drops on VA_nl.
- CLEAN cells (zero failures): peer verdict, cap crowd, peer revealed. Captions'
  contest-grouping shows within-group ≥ pooled (no identity leak).
- T stratification done only for peer verdict (nuisance-invariant, max drop .006);
  other cells' T-strata now computable from closure/samerows_preds/*_slim.csv (landed
  after the L2 run started) — folded into the confirmatory campaign readouts.
- One crash found+fixed: date-NaN in nuisance LR (median-impute for LR only).

## 2026-08-06 — GAP-CLOSER VERDICTS (D6): press RESOLVED, mathlib T RETIRED, competitions matrix LOST

notes/2026-08-06__gap_closer_batch.md.
1. **PRESS: no gap — provisional flag LIFTS on the Layer-1 numbers.** The published V .628/
   A .648 reproduce under each historical script's own splitter + sklearn-pinned env (a
   documented GroupKFold/StratGKF cross-version landmine); the .617/.669 pair was the
   campaign-standard-protocol ledger, not a failed gate. First-fit VA .6712 trustworthy;
   Δ_interact +.030 CI-positive; **no closure needed (VA_nl .701 > T .679p)** — press joins
   captions as a bank≥dense cell pending only a grouped dense rerun for T.
2. **MATHLIB: T .770 UNUSABLE — never quote.** Split is area-STRATIFIED not grouped
   (27/29 eval areas appear in train) AND the number traces only to a flagged-optimistic
   test reading on the pre-audit population, not the canonical n=7,956 slice. Area-grouped
   select-on-eval rerun SPECCED (→ scale-up wave); interim comparator = topic-residualized
   TF-IDF .736.
3. **CODE COMPETITIONS: the .731 bank's per-example matrix is NOT FOUND** (same category
   as the .7112 dense artifact). Published .731/.690 is same-population (n=2,495) so the
   comparison stands as a historical number, but Layer 1/closure need a bank RE-SCORE
   (→ scale-up wave, decision-sized). A first-pass wrong-matrix result was caught and
   corrected by independent verification (VERIFICATION_ADDENDUM in the JSON).
4. N&C agree: map-ready (brief written; maps batch needs a one-line loader entry).

## 2026-08-07 — SPURIOUS-MAP BATCH COMPLETE (6 cells, 12 rounds, 1.68M judge calls)

notes/2026-08-06__spurious_maps_batch1.md; closure/maps_batch1/. P=4/2fam throughout (GLM
5h-cap), Addendum-2 upstream mode from r1. All scrambled-anchor gates pass.

**Stacked increment splits the grid into three regimes**: dense-driven (peer curation
+.03-.05, peer revealed +.12-.13 over the nuisance model), bank-driven (cap crowd +.06-.09,
cap finalist +.03-.07 — dense increment ≈0), neither-much (N&C outcome/agree ≈+.02).

Conclusion-changing maps:
1. **peer revealed topic-floor NAMED AND SIZED — cuts against the BANK**: 11-13 nuisance
   channels alone (.752/.768: trend-aligned vocabulary .711, ecosystem naming .641,
   eval-breadth .640) match the entire 85-feature bank while dense sits .884. Discounting
   RAISES the residual: Δ_adj [+.134,+.197] (band; largest MIXED width in batch).
2. **cap_finalist negative Δ = dense WEAKNESS not articulation triumph**: within nuisance
   strata T falls to .501 (chance) while bank holds .615.
3. **N&C docket leak has a textual mechanism**: professional drafting/citation apparatus/
   document volume/OCR-scan signatures (.575-.609) alone match both instruments.
4. peer curation's date-strat failure named: trend-currentness channels (submission timing).
5. cap_crowd = informative negative: flattest map (max .546, surface-only) → its .63 bank
   advantage survives every discount.
Process: Track A near-exhausted on these cells (N&C outcome fired the stopping rule);
misrouting rises as banks fill (21/23 disputes → B); **B-side missing mass > A-side (mean
.62 vs .52) → maps are LOWER BOUNDS on the channel set**. Carry-forwards: corpus-matched
planted probes; caption pos/neg anchors uninformative at 13% pos-rate (scrambled gate does
the work). Four unpinned operational decisions documented in note §1.

## 2026-08-07 — CW COMMUNITY CLOSURE COMPLETE: NOT SATURATED AT CAP-5; Δ_beyond CORRECTED

notes/2026-08-06__closure_cw_community.md; closure/cw_community/ (129 files). ~875K judge
calls; GLM out ENTIRE campaign (weekly quota exhausted, resets 2026-08-13) → P=4/2fam.

- **Stage-0 population enlargement (408→7,008 honest rows, bank unchanged) CORRECTS the
  matched residual: Δ_beyond = +.141 [.128,.155], not +.176** (the 408-row read was
  data-starved: VA_nl .621→.651 with more rows; T unchanged).
- Curve r0-r5: VA_nl .6564→.6716; **5 rounds bought .015 of .139 (~11%) and the curve had
  NOT flattened** (one sub-ε at each end, three supra-ε between; never 2 consecutive).
  **Plateau language NOT licensed — stopped at cap.** TEST (touched once): Δ = +.1030.
- Fleet mass r5: A .283 / B .250; odds-form R̂ +.001-.002/round — UNDERPREDICTS the
  observed r3-r4 gains (+.009/.007): estimator tension, record not resolve.
- **Nuisance side OUT-PREDICTS the craft side**: best channel .582 (fragmented staccato
  lineation) > best of all 67 mined craft criteria (.579); planted shallow probes rank
  #2/#4 and were routed to nuisance 5/5 rounds (audit works).
- **Upstream mode discovery: "the writer had editing help"** — four rounds independently
  nominated fingerprints of this unseen parent (.550-.558), all MIXED-flagged.
- Every discount leaves Δ LARGER (matched full +.166, strict no-MIXED +.144, stacked
  control +.130 vs undiscounted +.128). Layer-2's markdown/lineation failure replicates blind.
- **VA_lin > VA_nl on this cell throughout** (.6812 vs .6716) → vs best articulated
  aggregation the residual is +.114; both reported.
- Freeze carry-forwards: sign-trigger caught 5 auditor misses but is over-sensitive
  (needs two-sided noise-scaled band); pre-GEPA (gepa_phrasing.py ready, unrun).
- OPEN DECISION: extend CW rounds 6-8 (curve still rising) vs accept cap-5 bound.

## 2026-08-07 — DENSE ARMS COMPLETE (HashtagWars / Style Invitational / patents claim-fell)

notes/2026-08-06__dense-arms-hw-si-patents.md. Recipe = dense standard, populations match
Layer-1 rows, pos-rate-matched grouped splits (a bucketer pos-rate bug was caught and
fixed BEFORE training — patents group size correlates with y, corr +.30).

| cell | T (clean-eval, seeds) | vs VA_nl | Δ_beyond |
|---|---:|---:|---:|
| HashtagWars verdict | **.6642** (3 seeds, range .020) | .6301 | **+.034** (sign robust) |
| Style Invitational | **.6343** (3 seeds, range .038) | .6651 | **−.031** → THIRD bank>dense cell (range > gap: read "no detectable dense lift") |
| patents claim-fell | **.7965** (SINGLE seed; test .8389) | .6256 | **+.171 — LARGEST in program; PROVISIONAL** |

**PATENTS FLAG (user sign-off pending)**: registry previously said "NO honest dense
exists" for this cell — that reflected a never-attempted run, not unlocatable text. Agent
located raw claim text (option3_claims_gemma_scale.jsonl), verified 0/59,937 alignment
mismatches, EXCLUDED Gemma disclosure-judgment columns from the corpus (anti-leak), and
V-with-length only reaches .60 (not a trivial length leak). Magnitude is consistent with
the verifiability-gap cells (N&C responded pattern), BUT: single seed, patents has known
leakage landmines (metadata .756; examiner effects), and the "no honest dense" row is
being overridden. STATUS: PROVISIONAL pending (a) user sign-off, (b) a code-cell-style
score audit (distribution, per-app_id decomposition, nuisance correlations) — the
patents map/closure round's Track B doubles as that audit. Never headline until then.

## 2026-08-08 — DEBIAS BATTERY HANDOFF (pre-Fable-audit state): V2 FAIL definitive at all λ

notes/2026-08-06__debias_pilot_nc.md (453-line handoff). 8 trainings, 6.2 GPU-h.
- V1 PASS on substance (within-model token ablation +.0275/.0689; plant probe 1.000 vs
  .540 control); V2 FAIL at λ=.1/.5/1.0/5.0 (fresh probe .997-1.000 throughout).
- **THE BATTERY CAUGHT A FALSE PASS**: at λ=1.0 the AUC gate alone (+.0039, inside
  tolerance) would have certified clean removal while the channel stayed perfectly
  readable. Also the diagnostic trap: co-trained adversary R² collapses to .110 (looks
  like success) while an independent probe reads .997 — NEVER read the adversary's own
  loss as evidence of removal.
- Failure modes: (a) adversary-defeated-info-remains CONFIRMED λ≤1; (b) λ=5 remedial =
  destruction WITHOUT removal (task −.114 while plant stays linearly decodable at 1.000
  and reliance QUADRUPLES); (c) correlated-leakage RULED OUT (within-stratum probe 1.000).
  Handoff verdict: GRL not trustworthy here. Fable auditor continuing (LoRA frozen-base
  hypothesis — consistent with the persistent linear plant direction).
- **N&C LANDMINE (independent of battery): the `year` field in N&C jsonls is a near-
  perfect LABEL LEAK** (pos rate 1.000 among rows carrying it) — never use it as a
  feature/nuisance; derive dates from docket IDs. (The debias pilot did; check other
  N&C instruments for exposure.)
- **N&C dense advantage is NOT docket-identity**: docket-disjoint vanilla .793 vs
  random-split .8167 (docket-alone .916 carries VA_nl, not T) — sharpens the N&C story:
  the bank rides docket identity; the dense model doesn't need it.
- Readout-power caveat: three length instruments DISAGREE IN SIGN on a 953-row split,
  agree on 1,904 — instrument-consistency checks need eval+test pooled minimum.

ADDENDUM to the year-leak landmine: exposure check done — Layer-2's N&C cells use `year`
from nc_vat_sample.jsonl/nc_unmatched_sample.jsonl where it is nearly always present
(6-7 imputed rows) and date-alone AUC is benign (.46-.55) → **Layer-2 N&C numbers are
CLEAN**. The pos-rate-1.000 leak is specific to the file(s) the debias pilot's corpus
build read. Rule: the leak is FILE-SPECIFIC — before using any year/date field from an
N&C jsonl, check its presence-vs-label pattern in THAT file.

## 2026-08-08 — PATENTS ROUND-0 AUDIT: CAMPAIGN STOPPED — +.171 IS A LEAK POST-MORTEM

notes/2026-08-07__closure_patents.md (725 lines); closure/patents/ (+RUNBOOK with revival
prerequisites). No fleet seated, no Gemma round scored — the stop rule saved ~750K calls.

**NEVER quote +.171 as a taste/articulation residual.** Gap accounting:
| explanation | eval (+.1751) | test (+.1955) |
|---|---:|---:|
| claim ORDINAL NUMBER (printed in text: "The device of claim 42 wherein…", 82% of elements) | 85.2% | 68.9% |
| other structure (dependency flag, length) | 3.5% | 6.9% |
| unexplained beyond V+A+structure | +.0197 [.001,.038] | +.0473 [.029,.067] |
claim_num alone .754; parsing the verbatim integer .725-.742. The named leak gates (class/
era) were CLEAN (.53-.55, dense survives their strata) — the killer channel was one nobody
named in advance, found by the audit's structure sweep.

- **Label-composition finding**: 17-31% of positives fell on §112/§101/double-patenting —
  grounds the packet's references can't address — and the model separates them AS WELL AS
  §102/§103 (§101 alone .9496); 43-45% of positives have ZERO disclosing references yet
  get prob .72 vs .33 → the model is not reading prior art. §102/§103-restricted rescue
  FAILS (ordinal channel gets stronger).
- **A-bank census: ONE concept** (4 columns, pairwise ρ .853-1.000, two exactly 1.0) —
  patents' Δ was "articulation never attempted", not "articulation failed".
- Addendum-3 decomposition of the drafting-order MIXED channel: surface wins decisively
  (surface marginal +.069 vs breadth +.004; breadth collapses to .510 under claim-number
  strata).
- Quarantine: `rejection_type` ≡ label (.988, sidecar — verified NOT in model input);
  `gold_disclose` arithmetic; gold-reference presence .998 construction artifact.
- Dense number itself SOUND (no input contamination, 3-seed pass finishing; class/era
  clean) — an honest dense model now EXISTS; registry's old "no honest dense" rule and
  the +.171 both point to this audit now. Cell revival prerequisites (real multi-criterion
  A-bank, claim_num-controlled design) in the RUNBOOK → D7 scale-up list.

ADDENDUM (patents audit, ablation update): refs_only .6433 > element_only .5988 (ordering
unconfounded; levels are off-distribution lower bounds). The model extracts a REFERENCE-
PROVENANCE signature from the label-conditional reference construction (gold appended to
positives only, 99.66%/0.00%, last slot 86.6%) that lexical surrogates (best .612) could
not see — §6a rewritten: this construction artifact is real and larger than surrogates
implied (second channel behind claim ordinal). Consequence: this corpus CANNOT support any
"identify the disclosing reference among K" evaluation without re-randomized slot order
AND symmetrically built negatives. Swap ablation (length/format-preserving derangement)
still landing = the clean entailment test.

ADDENDUM 2 (patents audit, swap ablation — the clean mechanism test): refs_swapped
(same claim + an unrelated application's references, identical shape/length) = .6527 vs
original .7965 — destroying the ENTIRE claim↔reference relationship costs .144; the model
keeps 51.5% of its above-chance discrimination with prior art that cannot disclose the
claim. In-distribution claim-side channel = .653 — BELOW the .754 that claim ordinal
number alone provides. Ranking vs calibration dissociate (mean prob .585→.224 while
ranking holds): under the threshold-free rule only ranking counts, and it is claim-side.
An entailment reader would fall to chance. (Deletion-pass levels confirmed understated,
as predicted; "retains 82% of AUC" framing rejected in favor of above-chance %.)

## 2026-08-08 — CW EXTENSION (r6-8) COMPLETE: SATURATION FIRED; DECOMPOSITION BEATS MINING

notes/2026-08-06__closure_cw_community.md (now 1,018 lines).
- **Saturation by the prereg rule at r6** (r5 −.0021, r6 +.0016 consecutive sub-ε).
  **CW plateau: Δ_TEST = +.1030** (T .8048 / VA_nl .7018, n=1,100; TEST touched TWICE,
  flagged — a third read needs a fresh split). MONITOR Δ_beyond ended +.1164 after r7.
- **HEADLINE: the Addendum-3 decomposition pass (r7) beat a full mining round** — 2
  decomposed columns, 28K judge calls, +.0055 vs r6's 15 criteria/175K calls/+.0016.
  **R7D03 "Revision depth: continuity, economy, closed setups" = .5856, the strongest
  quality criterion of the entire campaign** — recovered from the editing-help family
  that five rounds had labeled nuisance. Mechanism: a MIXED parent is a MIXTURE; scoring
  the mixture destroys both halves (editing-help .550-.558 pooled → split: real .586 to
  bank, surface .543 to nuisance). 7 parent channels retired.
- **Staccato adjudication (user challenge)**: as-written rubrics = surface (confirm_B —
  they count typographic density, no latent cause fingerprinted) BUT decomposition
  vindicates the intuition: craft half real (.567, banked) vs surface half (.577,
  nuisance). A planted "sentence-length variance" probe went to nuisance while its craft
  twin "Deliberate sentence rhythm" was banked — the audit separates them.
- Two-sided sign band adopted from r6: old one-sided rule over-fired 6:1 (1 re-audit vs
  5 recorded nulls).
- **Addendum-4 position round (r8): instruction-coverage gap CONFIRMED** (all proposers
  produce on-family channels when directed; 32/40 MIXED) BUT the measurement FAILED its
  anchor gate (coherent-vs-scrambled .32 — scrambled outscored real; paratext channels
  need paratext-appropriate negative controls, not word salad). All 10 columns excluded
  from readouts. CW table has NO timestamp/reply-order fields → **position is UNMEASURED
  on this cell, not null**; follow-up = re-fetch Reddit created_utc + within-thread rank
  as an observed covariate (→ D7 list).

## 2026-08-08 — N&C RESPONDED CLOSURE COMPLETE: THE +.17 FLAGSHIP GAP IS FULLY DECOMPOSED

notes/2026-08-06__closure_nc_responded.md; closure/nc_responded/round5_results.json. 1.21M
judge calls; P=4/2-family floor throughout; GEPA substituted by label-blind phrasing pass
+ fidelity gate (125/125).

**THE ACCOUNTING OF +.173 (original Δ_total):** ~half = V-driven nonlinear interactions
(Layer 1, VA .635→.724); round-0 honest Δ +.0359 → mined+decomposed down to **+.0210
pooled at r5 — and on the SELECTION-FREE EVAL HALF, Δ = −.0033: the bank has CAUGHT the
dense model.** The pooled +.021 is carried entirely by the test half, where this cell's
dense chain SELECTED (the select-on-test deviation is now decisive, not decorative).
Quote: "on selection-clean rows the enriched articulated bank matches T; the residual
pooled upper bound is +.021 (selection-contaminated)."

- **NOT saturated at cap** (like CW): the frozen rule had fired at r2, but r5 — spent on
  the two addenda instead of free mining — produced the campaign's ONLY super-ε gain
  (+.0085 [+.0001,+.0178]). **"The plateau was a property of the miner, not the
  residual."** Decomposition again beat mining: provision-by-provision engagement .5392
  (strongest A criterion) extracted from a channel four rounds called spurious; 7/8
  MIXED parents split successfully.
- **Position-in-container, third corpus, same signature**: docket-sequence (from doc_id)
  joint position model **.722 > the entire 38-channel mined map (.707)**, stacked
  increment +.079 over it, found by ZERO proposers in 160 sealed Track-B proposals.
  Does not threaten Δ (both instruments blind to it; discount null) but REFRAMES T .817.
- Bank census: 157 effective concepts of 198 (much cleaner than peer's 54/154); max
  single-criterion alone-AUC .553 — clean but weak individually.
- **Track-B missing mass NEVER CONVERGES** (.850→.725, recapture ≤.28) vs Track-A
  converging (.600→.450): "naming what makes a comment GOOD is a shared skill; naming
  what makes it LOOK good is not." Spurious-alone .723; discount null at every round and
  estimator; campaign/form-letter family ANTI-predictive (3× replicated).
- Corrections: honest-set per-channel Track-B strengths were mis-read (≤.083 off) —
  "first-hand exposure .593" family WITHDRAWN (full-population max |.064| from chance).
  Within-docket Δ not estimable (258 pairs); bank keeps 81% of its edge within docket at
  pair level.

## 2026-08-08 — THE STRICT LIST (user directive): 3×N decomposition grid tracking file

**notes/2026-08-08__vat-3xN-decomposition-grid.md is now the single tracking source for
method-stage coverage across (field × preference-type).** Update it whenever a cell gains
a stage. Peer review = first 3-way-ready field; math = second (all columns in build);
long poles = V8 co-signing (promoted — completes N&C's triple), V6, V9, V7. Peer
revealed closure added as the directive's one new upgrade.

## 2026-08-08 — CW POSITION COVARIATE LANDED (scale-up A job 3): FOURTH CORPUS, STRONGEST YET

Source: writingprompts_comments.jsonl.gz (Arctic Shift archive, on-disk; exact-text join
98.33% of the 7,008-row evaluation-valid population).
- **Relative thread position ALONE = .781 — within .011 of the dense content model (.792)
  on identical rows; the two are UNCORRELATED (r=−.009); combined .889.**
- Reading: on Reddit fiction, WHEN you reply is worth as much as WHAT you write, and the
  two are independent channels. Position is NOT in the text → T and the bank are both
  blind to it → **CW's tacit residual +.103 STANDS** (this does not discount it), but the
  outcome itself is ~half timing — a noise-ceiling/anthropological reframe of the y, same
  pattern as N&C's docket-sequence reframe of T .817.
- Position-in-container cross-cutting claim now has 4/4 corpora where position is
  observable (patents .754, N&C .722, code recency, CW .781) — zero of them proposer-
  discovered, all audit/covariate-discovered.
Mathlib + press 3-seed chains training (GPU5); tables when they land.

## 2026-08-09 — V+A+T FUSION FINAL (3 directions × 3 dense-below-bank cells)

notes/2026-08-07__vat_fusion_directions.md; methods/taste_decomposition/fusion/. All on
evaluation-valid rows E; E not selection-clean (inherited, matched across arms).

| arm | cap_crowd | cap_finalist | SI |
|---|---:|---:|---:|
| bank (VA_nl full@E) | .6217 | .6666 | .6508 |
| T original | .5554 | .6124 | .6490 ens |
| T + more data (D2) | .6087 | .6303 n.s. | no extra data (skip) |
| T + criteria-in-prompt (D3a) | **.6190** | **.6707** | deferred |
| bank + dense scalar (D1/1b) | .6204 (wash) | .6685 (wash) | .6624 (+.012, CI crosses 0) |

- **D3 closes the dense-below-bank gap** (+.064/+.058, P≥.99) — TO the bank's level,
  not above it. Weights sub-arm inert (user's over-reliance worry: no harm, no help).
- **D2 half-closes crowd** (+.053; the +46% extras are all-negative sub-100-vote
  captions — adaptive-vote-allocation artifact); finalist +.018 n.s.
- **D1 never hurts, adds ≈ nothing** once the bank is at full strength.
- **SYNTHESIS CORRECTION (supersedes the earlier "fusion dominates both parents"
  claim): fusion reaches max(parents), nowhere reliably exceeds it.** On dense-below-
  bank cells the two instruments measure the SAME learnable structure; the bank is the
  more efficient estimator at these n's. Combined with the closure results, the single
  cross-cutting statement: **dense models hold signal the bank lacks ONLY on the
  aesthetic-judgment cells (CW +.103, peer +.081); everywhere else the instruments
  converge — dense deficits are estimation artifacts (data/feature starvation), dense
  surpluses are either tacit residual or leaks, and the audits separate the two.**

## 2026-08-09 — HW + SI MAPS COMPLETE: SI's "bank>dense" DISSOLVES; HW residual halves

notes/2026-08-08__maps_hw_si.md (1,220 lines); closure/maps_hw_si/. Decomposition-first +
2 map rounds each; corpus-matched probes 4/4 on all six audits; pre-GEPA; P=4/2fam.

- **STYLE INVITATIONAL VERDICT REVERSED: the cell is a TIE and its bank is a LENGTH
  MODEL.** Honest rows: T .638-.649 vs VA_nl .6401 (Δ −.002..+.009). Within length
  strata the bank COLLAPSES to .5409 while T holds .5889; **0/32 frozen rubrics survive
  length stratification** (2 pooled). Bank increment over the V block +.006 vs dense
  +.023. Layer-1's "V≥A counter-pattern / third bank>dense cell" is RETIRED — never
  quote SI as bank>dense. Final Δ after rounds: −.0015 → cell TERMINAL (mapped, no
  residual, bank=length verdict).
- **HashtagWars: Δ_beyond +.0572 honest → +.0331 after decomposition-first → +.0252
  after r2 (56% reduction).** Strongest channel replicated 2×: boilerplate-tag placement
  (.35 anti-predictive) = the TEXTUAL SHADOW of the SemEval retrieval-batch construction
  (winner tweets fetched as a separate batch) — carry as a corpus-construction caveat on
  ALL HW numbers. Wide CI ([−.017,+.156], 8 held-out contests; MONITOR = 4 contests).
  Rounds continue to the stopping rule (extension dispatched).
- **Decomposition-first beat mining a THIRD time**: rewriting 9 rubrics to forbid their
  surface carrier +.0241 bank AUC vs 30 fresh criteria +.0079 ("character voice is
  implied": .499 → .574).
- **Humor INVERTS the A/B missing-mass asymmetry**: Track A the less-converged space on
  3/4 rounds (HW r1 M̂ .883, 5% recapture — program record). Humor's articulable space
  is deep and unconverged; its criteria are individually weak but plentiful.
- Position: both humor corpora carry row-order/retrieval-batch artifacts NOT in the text
  (within-contest rank .000 in-contest) — Δ-safe, but landmines for id/order-based
  modeling (position_leak_diagnostic.json).
- Anchor-design insight: surface-channel batches legitimately FAIL word-salad scrambles
  (extent-of-surface criteria score salad high); pos-vs-neg carried the gate (.728) —
  paratext-appropriate controls needed, replicating the CW r8 lesson.
- Ops: agent mistakenly killed a healthy EngineCore of ITS OWN run → half-empty score
  matrix QUARANTINED + rescored + a new interrupted-generation gate added to
  score_gemma_maps.py (self-caught; no co-tenant touched).

## 2026-08-09 — CODE ENRICHED-BANK READOUT COMPLETE: SMALL REAL RESIDUAL; v2 "NO RESIDUAL" RETIRED

notes/2026-08-06__code-abank-enriched-rescore.md; results/code_v3_enriched_layer1.json +
within_repo_dense_vs_vanl.json. 950K judge calls; anchors K=50 pass; 1/83 collapse;
Gemma-4-31B on v3 enriched text (83 portable criteria).

- **WITHIN-REPO (the trustworthy level): dense .7093/.6540 vs VA_nl .6517/.6150 →
  Δ = +.0576 eval / +.0390 test, Wilcoxon p=.006/.040.** Code has a small but real
  positive residual — the thin-residual group (~+.03-.06), not the +.10-.17 cells.
  The v2 "no taste residual in code" was an artifact of diff-only inputs + no-judge
  instruments.
- **NEVER QUOTE THIS CELL'S POOLED NUMBERS** (they contradict across splits: VA_nl .7422
  eval vs .5806 test = repo composition; the pooled-eval "bank wins" is NOT a bank win).
  Within-repo only, both splits, always.
- Bounds on the residual: dense v3 is SINGLE-SEED (±.02 → test +.039 ≈ 2σ); judge saw
  11.6K tokens vs dense's 2,048 (A advantaged); V_exec is BELOW CHANCE out-of-repo on
  test (.458) — execution verdicts anti-transfer across repos.
- **Mechanism: the articulable part of merge preference is HOW THE CHANGE IS
  COMMUNICATED** — top criterion both splits = "change/commit/PR communication quality"
  (.550/.552), with change-description clarity + contribution readiness behind it; all
  unanswerable in principle on a bare diff.
- Stage-1 provenance (recap): .592 UNVERIFIABLE (no judge, spliced protocols,
  fingerprint aspects, rebuild gives .526) — retired; never difference new numbers
  against it. Gate per the goal: within-repo Δ > .02 → code goes to CLOSURE (round 0 =
  dense seeds 1-2 + within-repo protocol).

## 2026-08-09 — GEPA REQUOTE LANE (mid-flight verdicts, notes/2026-08-09__gepa_requotes.md)
- **N&C substitute phrasing pass FAILS the freeze** (no fidelity objective, K=1 not K=3,
  no collapsed-criterion repair) → N&C's 67 surviving criteria join the GEPA queue; the
  N&C ≈0 headline KEEPS its pre-GEPA flag until rescored.
- Peer verdict: retroactive two-sided sign band → 1/56 re-routed ("Restraint" =
  polarity-inverse of the hype-vocabulary nuisance) → 55 survivors to rephrasing.
- CW: 10/84 criteria flagged for rephrasing, 8 winners accepted by fidelity margin,
  full-population rescore running (GPU7).

## 2026-08-09 — CODE CLOSURE ROUND 0 (the audit-grade baseline): gate PENDING SEEDS, everything else in

notes/2026-08-09__closure_code.md; methods/taste_decomposition/closure/code_v3/.
Confirmatory campaign under the FROZEN Layer-3 protocol with one recorded cell-specific
adaptation (§2 of the note, written BEFORE any mining): **every readout is WITHIN-REPO
n-weighted; pooled AUC is never a residual on this cell; the .592 baseline stays retired.**

- **Cell needs no dense-honesty intersection.** The dense v3 chain trained on the 47,659
  TRAIN rows whose repositories are disjoint from eval/test, and the bank scored eval+test
  only — so all 11,452 rows are dense-held-out and MONITOR is T-honest automatically. This
  is the first confirmatory cell where the N&C DECISION-1 thin-monitor problem does not
  arise. Splits: FIT+MINE 8,697 rows/201 repos (115 scorable), MONITOR 2,755/54 (29).
- **Gate quantity re-characterised at seed 42 (reproduces .0576/.0390 exactly).** New:
  combining the two dense splits at the REPOSITORY level (legitimate — the readout is
  repo-centred and the splits contribute disjoint repos) gives **Δ = +.0487 over 144
  repos / 10,284 rows, repo-cluster bootstrap [+.014, +.085], P(>0)=.996, Wilcoxon
  p = .0007**. Caveat now measured: the n-weighted magnitude is dominated by a few large
  repos, so test-alone's bootstrap CI is [−.005, +.083] even at Wilcoxon p=.040; the
  equal-repo Δ (+.050, [+.001, +.100]) is the estimand the Wilcoxon actually tests.
- **DESIGN §11 PASSES, and code is the first cell where fusion beats BOTH parents.**
  Grouped-OOF stack of [VA_nl OOF, dense], read within repo: **fused .6920 vs bank .6342
  (+.0578, p=1.7e-8) vs dense .6829 (+.0091)** on 144 repos. No audit trigger. Contrast
  with cap_crowd/SI where fusion only reached max(parents) — independent corroboration
  that bank and dense carry partly non-overlapping signal here.
- **Bank concept census: the LEAST redundant bank in the program.** 83 delivered → 79
  after the degeneracy screen → **80 effective concepts** (blind pairwise, strict; loose
  78), max column |r| = **.895**, 0.0% of pairs ≥ .90. Peer collapsed −65%, N&C −21%,
  code **−3.6%**. Judges agreed .846, both 4/4 on authored anchors. **The one merge that
  matters: a78 "PR communication quality" ≡ a105 "change-description clarity" — the cell's
  #1 and #2 univariate criteria on eval are ONE concept measured twice**, which is why the
  Addendum-3 decomposition pass targets it first.
- **Position line (Addendum 4), on the closure population.** Within-repo PR-number
  percentile: pooled .4928, within-repo n-wtd **.4993** — but repo-level SD .175 and
  **34.7% of repos at |AUC−.5| > .15** (19% > .65, 14% < .35). Both instruments partially
  learned it (mean within-repo ρ with position: **dense +.148**, bank +.093) and learned it
  *targeted* (corr between a repo's position-predictiveness and the instrument's alignment
  with position: dense +.259, bank +.222). **Discounting it does not move the residual:**
  repo × position-quartile stratified Δ = **+.0509** vs unstratified +.0487.
- **CLOSURE-PROTOCOL r=0 ANCHOR LANDED, and it does NOT shift the level** (the pilot's
  "closure levels are protocol-specific" caveat is close to a no-op here): honest-full
  **Δ = +.0522** (T .6829 / VA_nl .6307, 144 repos, Wilcoxon p = .0011, [+.015, +.089]);
  eval +.0586, test +.0452, MONITOR +.0472. Layer-1 vs closure agree to within .006 — so
  the curve's r=0 sits on the same scale as the gate. Contrast N&C (+.092 → +.036).
- **PROTOCOL WARNING for the curve: the frozen saturation statistic is 5x noisier than ε.**
  MONITOR = 29 scorable repos / 2,489 rows; VA_nl seed spread **.027** vs ε = .005, and the
  MONITOR Δ carries CI [−.078, +.173] (p = .149). Mitigation recorded before round 1: every
  round gain quoted with its seed band + repo-cluster CI, honest-full (144 repos) reported
  beside MONITOR every round, tier disagreements reported not resolved.
- **A dominates V on this cell** (MONITOR .6116 vs .5687) and **Δ_interact is only +.0215**
  (VA_lin .6092 → VA_nl .6307) — the opposite of N&C on both counts; the signal is in the
  criteria, not the nonlinear aggregation.
- **Swap baseline C₋ = .5099** (honest-full, 161,761 within-repo pairs, C₊ = .6687): the
  bank is at CHANCE on the third of within-repo pairs the dense model orders backwards
  (N&C had .587). It holds no independent signal there — consistent with the fused result —
  and the swap boundary is one round away on test (C₋ already .4955).
- Dense seeds 1 and 2 CHAINED on sk3 GPU 6 (ledger-claimed, sole tenant, PID verified
  resident, trainer md5-verified unchanged); ETA seed 1 ≈ 05:30, seed 2 ≈ 15:00 2026-08-08.
  Scorer merges into eval_pass_results.json, never rewrites seed 42.
  **GATE: 3-seed within-repo residual > .02 → rounds 1..5; ≤ .02 → STOP, seed verdict is
  terminal.** Round-1 machinery (decomposition-first pass with 5 census-deduped SHAP-picked
  MIXED parents, code-specific proposer harness, 4 corpus-matched PR probes, Gemma round
  scorer, within-repo readout) is built and staged; nothing fired.

## 2026-08-09 — SYNTHESIS REFINEMENT (cross-cell, from code round 0)

Code is the FIRST fused-beats-BOTH-parents cell (fused .6920 > dense .6829 > bank .6342,
p=1.7e-8, within-repo grouped-OOF) — refining the fusion claim: **fusion exceeds both
parents only where a genuine residual coexists with an informative bank (code); on cells
where one instrument subsumes the other (captions, SI), fusion reaches max(parents).**
Also cross-cell: code's bank is NOT a length model (VA_nl −.012-.017 under within-repo
length strata) — the SI collapse is a property of SI's bank, not of banks; and code's
position channel is tracked BY BOTH instruments (dense ρ +.148, bank +.093) yet
discount-null — third distinct relationship between position and instruments (leak/
ceiling/shared-tracked).

## 2026-08-10 — GEPA REQUOTES COMPLETE (goal item 6): ALL THREE PLATEAUS SURVIVE, PRE-GEPA FLAGS LIFT

notes/2026-08-09__gepa_requotes.md; gepa_* JSONs in the campaign dirs.
- N&C substitute pass FAIL confirmed empirically (22/67 criteria low-fidelity despite the
  house pass; 41% of those had recoverable headroom) → all three cells got the genuine
  K=3 fidelity-search treatment.
- Requoted plateaus (old → new): CW MONITOR +.1164 → +.1220, honest +.1269 → +.1260;
  peer verdict honest +.0807 → +.0836; N&C honest +.0210 → +.0222 with the decisive
  selection-free eval reading −.0033 → −.0038 (MORE closure). "Restraint" re-routed to
  nuisance under the two-sided band (55 peer survivors).
- **Every movement <4% of the number's own value and inside each cell's noise band →
  the plateaus are NOT artifacts of pre-GEPA phrasing. All three headline results now
  carry "phrasing-pass satisfied" lines; quote without the pre-GEPA flag.**

## 2026-08-10 — HASHTAGWARS SATURATION AT ROUND 4: Δ_plateau +.0286, WIDTH-DOMINATED

notes/2026-08-08__maps_hw_si.md §14; closure/maps_hw_si/ (183 files). Signed reading
governs (r3 +.0023, r4 −.0056 → 2 consecutive sub-ε); magnitude variant would continue —
both recorded.
- **NEVER quote an HW Δ as a point estimate**: contest jackknife SE .050-.061 = 2-3× the
  residual (8 held-out contests; two contests drive it in opposite directions). Direction
  stable (Δ>0 in 7/8 LOO at every mined state). Band at saturation [+.011,+.047];
  discounting the retrieval-batch provenance family alone → +.0107.
- Retrieval-batch textual shadow = strongest channel on 3/4 rounds (three independently
  sealed fleets, alone-AUCs .350-.369) — the construction caveat is load-bearing.
- **"Mining exhausted; rewriting was not"**: 4 rounds/54 criteria net +.0045 bank AUC;
  9 Addendum-3 rewrites net +.0241 (84% of total gain). The MIXED problem REGENERATES —
  each round's decomposition parents were the previous round's MINED criteria. Swap
  signature fired on 3/4 mined rounds, never on decomposition passes.
- Track-A mass never below .650, recapture ≤.15 → plateau = "not discoverable by this
  miner" (humor's deep-articulable-space pattern again).
- Routing perfect both new rounds (0/25, probes 4/4, zero disputes).
HW = TERMINAL (saturated). SI terminal line confirmed in note §12.0.

## 2026-08-10 — PEER REVIEW COMPLETE ACROSS ALL THREE PREFERENCE TYPES (first field)

notes/2026-08-09__peer_completion.md (613 lines); closure/peer_revealed/ + peer_curation_ext/.
Fleet at FULL P=6/3 families — **GLM-5.2 LIVE AGAIN on both Lite keys** (recharge landed).

**THE THREE-Y CONTRAST (same abstracts, same bank, same judge, same architecture):**
| y | Δ_beyond terminal | verdict |
|---|---:|---|
| curation (oral/spot) | **+.0020** (cap-5; best bank r4; noise-floor signature: missing mass RISES .544→.778 while gain→0) | ARTICULABLE — residual = outcome noise |
| verdict (accept/reject) | +.081-.084 (pilot, exploratory flag) | moderate resistant residual |
| revealed (citation-pct) | **+.1125** saturated r5; discount-robust **[+.076,+.254]**; topic-robust 12/12 strata | NOT articulable by this instrument class |
**A 56× residual difference across preference types → the tacit residual is a property
of WHOSE PREFERENCE IS MODELED, not of the text or the instrument.** (= the hole-(b)
headline, now with closure-grade evidence.)

- **Topic caveat REVERSED on revealed**: stratifying by topic RAISES Δ (bank loses .062
  vs dense .036 under topic conditioning; dense increment over topic+trend+bank +.0732
  [+.046,+.099]) — the topic floor was the BANK's crutch. Retire "revealed rides a topic
  floor" as a Δ-discount; keep it as a level caveat only.
- Revealed's named-nuisance set now BEATS its own 151-feature bank (.8033 vs .7717) —
  quote Δ_adj as the band [+.076,+.254], never a point.
- Decomposition PURIFIED a channel upward (trend-surface component .716 > .677 parent);
  formalism family robustly NEGATIVE (.363-.375, 4 rounds).
- **PROTOCOL OBSERVATION (freeze-relevant): the stopping rule is NOT monotone** —
  curation's rule fired retrospectively at r2, then r3-r4 closed a further .011, and r5
  criteria significantly HURT (−.0121, P=.03). Amendment-2's warning now observed. Best-
  bank-so-far, not last-bank, is the quotable state.
- Ops: 72 sealed fleet slots, 0 parse failures, probes 4/4 ×6 rounds, 5 disputes all→B,
  ~797K judge calls, poll-claim-verify-release on GPUs 3/7.

## 2026-08-10 — DEBIAS BATTERY: DEFINITIVE NEGATIVE. GRL RETIRED AS A REMOVAL INSTRUMENT

notes/2026-08-07__debias_audit_fable.md; methods/taste_decomposition/debias/.
- Root cause TWO layers, both plant-proven: (1) LoRA blind spot REAL (frozen base reads
  the plant at .955 with zero training; rank-16 deltas never cancel frozen-substrate
  information); (2) **fixing the architecture does not fix the procedure** — the 256-d
  bottleneck arch (both heads through a trainable projection, no capacity loss) STILL
  fails V2 at every λ (z-probe .97-1.00; at λ=1 plant reliance AMPLIFIED 6.7× while task
  AUC fell .138). GRL defeats its co-trained adversary and keeps the channel — Elazar &
  Goldberg 2018 reproduced inside our own instrument under maximally favorable conditions.
- The battery caught a would-be FALSE PASS a second time (λ=1 pooled: AUC gate +.0039
  "clean" with probe .999).
- **Verdict: design-§9 conditional approval resolves NEGATIVE; GRL retired. Instruments
  of record = stacked increment + matched sampling.** If a removal certificate is ever
  required: closed-form post-hoc projection (INLP/LEACE-class) WITH its own planted
  battery — noted, not proposed.

## 2026-08-10 — SCALE-UP C COMPLETE: FIVE NEW LEDGERS; JOKES +.26 BAND COLLAPSES TO +.015

notes/2026-08-08__scaleupC_builds.md (1,074 lines); ledgers in results/*_ledger.json;
matrices outputs/va_gemma_banks_scaleupC/. All T's seed-42 (seeds 1-2 chain running).

| cell | VA_lin | VA_nl | T | Δ_beyond |
|---|---:|---:|---:|---:|
| reddit-jokes community | .7169 | .7321 | .7470 | **+.0149** |
| math.SE ACCEPTED (asker verdict) | .6320 | .6320 | .6319 | **−.0001** |
| math.SE VOTE (crowd) | .6225 | .6242 | .6608 | **+.0366 → only new cell clearing the L3 gate** |
| AoPS curation | .7712 | .7705 | .7806 | +.0101 |
| homepage | .5490 | .5562 | .4322✗ | **n/a — NOT QUOTABLE** |

- **JOKES: the apparent +.26 band was ENTIRELY instrument error** (articulated channel
  under-measured +.153 by the noah floor harness; dense over-measured .077 by the
  ungrouped sweep). Mature instruments → +.015. Humor is now thin-residual EVERYWHERE
  (HW band [+.011,+.047], captions bank≥dense, jokes +.015).
- **MATH.SE UN-BINARIZATION SPLITS THE CELL — second field showing the preference-type
  gradient**: asker's accept verdict FULLY ARTICULABLE (−.000) vs crowd vote +.037, same
  matrix/bank/judge. The old fused y hid this. (Overlap check: 0.79% → retraining was
  justified, recorded.)
- AoPS first A-bank: .769 vs dense .781 — articulable (+.010); A beats V by .066.
- **HOMEPAGE: instrument failure, terminal as documented-unbuildable-cross-outlet** —
  census bank FAILS coherence (scrambled .387: entity detectors, not reading
  instruments); outlet-held-out T UNUSABLE (eval .432 below chance vs test .736, two
  held-out outlets disagree in sign). "Within-outlet instrument exists; no cross-outlet
  instrument" is the citable result; Δ flagged not-quotable in JSON + note.
- Δ_interact null-or-surface on ALL five (jokes' +.015 is V-driven 3.2× → L2 referral).
- Reuse log exemplary: jokes seeded from the May 366-aspect pool (no duplicate mining);
  AoPS dense REUSED at zero GPU (population re-scoped to held-out rows); math.SE V
  plumbing + a01-a14 seed axes reused.
- CAVEAT until 3-seed T: jokes +.015 and accepted −.000 are within eval/test spread —
  quote as "no more than a few points" / "no detectable residual".

**AESTHETIC-AXIS UPDATE (one gradient, 9 points):** CW .103-.126 > peer verdict .081 >
math.SE vote .037 > HW ~.029 > jokes .015 > AoPS .010 > math.SE accepted .000 ≈ N&C 0;
code +.049 (communication-mechanism) sits mid-thin. **And the cross-y claim now has two
fields: crowd/revealed y's hold the residual; institutional verdicts and expert
selections decompose** (peer: curation .002 vs revealed .113; math.SE: accepted .000 vs
vote .037).

## 2026-08-10 — SCALE-UP A HARVEST: PRESS "BANK≥DENSE" FALSIFIED; MATHLIB RECIPE COLLAPSE CAUGHT

notes/2026-08-08__scaleupA_dense_reruns.md; results/samerows_T_{press,mathlib}.json.
- **PRESS honest T = .7497 (3 seeds, company-grouped) vs VA_nl .7011 → Δ_beyond +.0486.**
  The provisional bank≥dense standing was a population-mismatch artifact (old T .679
  ungrouped). No standing-rule audit fires (dense wins). **The bank≥dense club is now
  ONLY the two caption cells.** Press clears the L3 gate → closure campaign dispatched.
- **MATHLIB: frozen-recipe COLLAPSE at 94.3% positive rate** (no class weighting →
  recall ~1.0/precision=base-rate across all seeds/checkpoints). **NEVER quote
  .580/.473.** Proof of recipe-not-data: class-weighted TF-IDF on the identical split =
  .677/.786. Corrected --class_weight_auto rerun on GPU3 in flight. RECIPE NOTE for the
  standard: pos_weight/class-weighting is REQUIRED on heavily imbalanced cells.
- Ops: the original chain had finished entirely (detached) — only the poller died;
  no rescoring wasted.

## 2026-08-10 — LEACE PILOT: ADOPTED AS LINEAR-ONLY ERASURE INSTRUMENT (stated scope)

notes/2026-08-10__leace_pilot.md; debias/leace/ (battery_leace.json).
- Linear erasure EXACT (post-erasure plant probe .4749; cross-covariance machine-zero;
  verified vs reference implementation + synthetic hand-check). **Because our dense
  standard scores through a head LINEAR in h, "linearly unreadable" genuinely certifies
  the score path** — what GRL never delivered.
- **Nonlinear residue .96 (plant) / .92 (length): NEVER quote a nonlinear-scope removal
  claim.** Linear-only, and the residue is generic across channel types.
- Utility frontier SCALES (all-71-channel erasure costs −.087 with no collapse; GRL lost
  −.138 removing nothing). Specificity passes (real signal survives; unused channel
  free). Consistency: agrees with stratified (0.93×) and stacked (1.21×); matched
  sampling reads 2-7× smaller — instrument-discrepancy note.
- **NEW METHODS SUBTLETY — the y-tax placebo**: erasing ANY y-correlated flag costs the
  y-correlation tax regardless of channel content (here −.029 of the −.038 total; the
  channel-specific cost was ~.009). Erased-model AUCs therefore UNDERSTATE the channel-
  absent counterfactual. STANDING RULES when quoting any erasure number: report the MLP
  residue + the y-matched placebo beside it. Stacked increment + matched sampling remain
  the influence instruments of record; LEACE adds the intervention-style readout.
Debias series status: GRL retired (negative) · LEACE adopted (linear scope) ·
decorrelated-training battery in flight (reliance scope).

## 2026-08-10 — LANDMINE (cross-worker catch): OOF ARRAYS ARE IN BANK item_ids ORDER

A grid worker (v3chain-gpu2) caught + solved an alignment landmine BEFORE any fabricated
CI shipped: *_va_nl_oof_*.npy arrays are keyed in bank item_ids order, NOT population/
join order — misaligned joins read AUC≈.50. Five cells were misaligned under the
dense_joins key (press, jokes, mathse×2; aops fine). **MANDATORY GATE now standard:
AUC(y, oof_SEED0 in assembled row order) == published nonlinear.VA.seed_aucs["0"] to
<1e-9 (exact identity; seed0 never mean3 — mean3's AUC ≠ mean of per-seed AUCs).**
Re-key recipe: bank meta item_ids (scaleupC cells; mathse npy = kept-subset order) /
CACHE.npz["ids"] (press) / maps_batch1 cells.py `ids` (N&C-peer samerows route).
Per-cell gate status recorded in every results JSON.
TRAINER LANDMINES (same report): train_reward_model.py:327 asserts 80/10/10 ±.02 splits
(aops dense join is eval/test-only → stock path unusable); score_eval_dense_v4.py
hardcodes MAXLEN=1024 (silently truncates code_v3 trained at 2048 — override required).

## 2026-08-10 — V3 GRID SHIPPING (overnight): 11 cells, 2 principled blockers, 1 new landmine

Builder shipped v3_grid/{build_v3_cell.py,train_v3_cell.sh,build_blockers.json} + first
manifests (HW k20+k10 twin, press). Grid = 11 cells; BLOCKED: aops_curation + code_v3
(banks scored ONLY on dense-held-out rows → no honest train-fold importance ranking; fix
= train-split Gemma pass, code ≈4M calls, decision-sized, parked). peer_curation (72%)
+ peer_verdict (22%) built on bank-covered SUBSETS with SUBSET_OF_ORIGINAL flags —
compare ONLY against original dense preds restricted to the same rows.
**LANDMINE: datasets/math/aops/va/dense_standard/split/ is a STALE ORPHAN** of a
superseded 13,071-row draw sharing the sha1 id scheme — a naive row_id merge silently
matches 2,307/5,202 and mislabels 1,933 as train. Never use it.
Overnight board: full-grid V+A+T (11 V3 arms + VAT stacks + master table), code closure
gate→rounds, math.SE-vote closure (gated on 3-seed), press closure, decorrelated-training
battery + cap_crowd recovery arm, LEACE adopted, Fable decorrelation-stack decision
(synthesizing all debias evidence + lit review), spurious-debiasing lit review, V8
co-signing build, mathlib class-weighted rerun. GLM live; all free GPUs authorized.

## 2026-08-10 — VAT-STACK MIRROR COMPLETE (8 dense-strong cells)

notes/2026-08-09__vat_stack_dense_strong_cells.md; results/vat_stack_*.json (8, each with
an alignment_check field — T_E matches samerows_T_*.json to 4 decimals; no cached OOFs
used, all recomputed in-process).

| cell | T | VA_nl (best arm) | VAT_nl | vs bank | vs T |
|---|---:|---:|---:|---|---|
| peer verdict | .7769 | .668 | .742 | >bank P=1.0 | <T |
| CW community | .7921 | .665 | .787 | >bank P=1.0 | ≈T (P .02) |
| N&C responded | .8167 | .791 | **.832** | >bank P=1.0 | **>T P=.99 (round0 arm)** |
| N&C outcome | .6238 | .612 | .623 | >bank P=.98 | tie |
| N&C agree | .6034 | .563 | .571 | >bank P=1.0 | <T |
| peer curation | .5936 | .529 | .554 | >bank P=.98 | <T |
| peer revealed | .8842 | .655 | .848 | >bank P=1.0 (+.20!) | <T |
| HashtagWars | .7315 | .529 | .645 | >bank P=1.0 | <T |

**Cross-cell answer (one claim): adding the dense score to the bank ALWAYS improves the
bank (9/9 arms, P≥.98, up to +.20) — the articulated bank is never inert — but the fused
stack exceeds T itself only where the bank holds complementary signal (N&C responded
+ code + nc_outcome tie = the procedural cells); on the aesthetic cells (CW, peer
verdict/revealed, HW) dense SUBSUMES the bank and fused plateaus at/below T.** This is
the fusion-side mirror of the taste gradient: complementarity lives where preferences
decompose; subsumption where the residual survives.
- N&C responded = SECOND fused-beats-both-parents cell (.832 > T .8167, P=.99).
- CW round8 note: its r8 anchor gate failed with 0 admitted criteria → round8 bank ≡
  round7 (already known; consistent).
- Process note: the agent correctly REFUSED to misapply the cached-OOF alignment gate
  to its fresh-refit design and verified equivalence its own way (grep-proof of no
  cached reads + 4-decimal T_E matches) — the right behavior; gate applies to cached-
  OOF joins only.

## 2026-08-10 — FABLE DECISION: THE PRODUCTION DECORRELATION STACK (design §13, freeze-ready)

notes/2026-08-10__decorrelation_stack_decision.md (rationale); §13 in the design note.
Three scoped instruments, none moonlighting:
- (a) INFLUENCE: stacked increment PRIMARY everywhere (+ new reliability-sensitivity band
  per Westfall & Yarkoni — the unreliable-control bias runs TOWARD our preferred
  conclusion and grows with n); matched sampling DEMOTED to sign/consistency duty
  (read 2-7× low + sign-flipped under near-identical protocols); LEACE erase-and-refit
  JOINS as the intervention leg — governs on leg disagreement, headline per-channel
  quotes, or spurious-alone > .65. Decile stratification RETIRED as a discount
  (conditioning-on-label ×2 documented); ≤.65 descriptive appendix only.
- (b) TRAINING PROTECTION: decorrelated reweighting freezes IFF its battery passes —
  frozen recipe: stabilized IPW, grouped-OOF propensities, 99th-pct clip, mean-1 renorm,
  loss-weights not sampler, n_eff/n ≥ .70, 2-epoch early-stop (Byrd-Lipton discipline).
  T_decor MANDATORY on confirmatory cells with s > .65, optional .55-.65, skipped below;
  ledger role = robustness arm beside T, NEVER replacement (T_decor≈T is ambiguous).
  Fail branch: AFR-style head-refit first, ODIN additive penalty second — each with its
  own planted battery + user sign-off.
- (c) REMOVAL CERT: LEACE linear-scope only, rare by design; frozen certificate language
  ("the score, being a linear functional of the erased representation, cannot use the
  channel"); MLP residue + y-tax placebo beside every quote; VOID under retraining or
  any nonlinear consumer — the dense standard's LINEAR HEAD is now part of the freeze.
- Per-cell procedure by s = spurious-alone: <.55 stacked alone · .55-.65 +matched(sign),
  decor optional, LEACE on disagreement · >.65 stratification banned, LEACE mandatory,
  T_decor + erasure-cost-vs-y-tax cross-check on confirmatory cells.
- Adopted from lit review: reliability band; ONE-HOT binning for LEACE (continuous-Z
  guarantee gap — fix already ordered to the LEACE battery); Bastings difficulty ladder
  for future batteries. Rejected: GRL (permanent), text editing (user + 3 citations),
  INLP, more adversaries, kernel erasure (watch-list), DRO/JTT as specified.

## 2026-08-10 — LANDMINE (permanent): N&C LAYER-1 OOF ROW ORDER UNRECOVERABLE

nc_layer1_stack.py rows_for() iterated PYTHON SETS directly (valid_out/valid_agr); saved
*_va_nl_oof_*.npy row order = per-process string-hash randomization → NEVER reproducible
(five candidate orderings all gate at ~.50 vs published .581/.609; today's live order
differs from saved = the randomization fingerprint). nc_responded same construction
(dict branch, possibly deterministic, but no gateable reference → untrusted).
- **RULE (program-wide, immediate): never iterate sets for row order; every saved array
  ships with an ids vector alongside; the seed0 exact-identity gate runs before any
  row-wise join.** (cells.py::_load_nc already knew: "never iterate them directly".)
- Ledger consequence: nc_agree/nc_outcome/nc_responded get (V3−T) CIs but
  boot_v3_minus_VA_nl: null with reason, until regeneration; hashtagwars has no npy;
  peer_verdict recoverable via its seed0 npy (never mean3).
- Schema trap: published seed-0 AUC lives at nonlinear.VA.seed_aucs["0"] (layer1 files)
  BUT nonlinear.VA["0"]["auc"] (scaleupC ledgers) — read both or misdeclare cells
  ungateable.
- Exposure: analyses that recomputed OOF in-process (VAT mirror, Layer 2, closure
  campaigns, maps batch via sorted cells.py) are SAFE by construction; exposure audit +
  regeneration commissioned (sorted keyed order + ids saved + identity-gated).
- Harvest tooling: fusion/v3_grid/harvest_v3_grid_cell.py refuses bank CIs unless the
  gate passes; records T_same_rows vs T_published separately for SUBSET cells; emits
  truncation_confound blocks.

ADDENDUM (LEACE corrected battery, R1a fix applied): certificate_form=CATEGORICAL now
mandatory (train-decile one-hot before fitting; median-split probes provably inherit the
categorical certificate; docstring corrected; run-1 continuous-Z battery preserved as
labelled secondary). Re-issued verdicts: V1/V2a/V3b/V4 PASS, V2b still linear-only
(MLP residue .96/.91). **NEW HEADLINE — certificate strength trades against surgicality**:
the strong-form 26-channel eraser (rank 52) erodes an UNNAMED real channel's causal use
(Δ-of-Δ −.019, head cost −.128) where the OLS form was surgical; frontier costs ~2× the
OLS form everywhere. **Operating envelope: small named sets (1-2 channels, rank ≤~18) =
affordable + surgical + V4-consistent (1.1× vs stratified); bulk scrubs of the full map
measure joint channel load and are NEVER a usable debiased scorer.** Every quote ships
MLP residue + y-tax placebo + eval AND test readouts.

## 2026-08-08 09:50 — CODE 2-SEED INTERIM (NOT THE GATE): residual holds; eval seed-invariant, TEST seed-fragile

notes/2026-08-09__closure_code.md §9.2; closure/code_v3/gate_readout_2seed.json.
Seed 1 landed 05:27 (eval .6795 / test .7229 pooled); seed 2 at 49%, ETA ~14:50.
**`BINDING: false` — rounds stay HELD until 3 seeds + a free ledger GPU.**

- **OOF ALIGNMENT GATE (registry 2026-08-10) APPLIED AND PASSES EXACTLY** on code_v3: all
  six seed x split cells at abs diff **0.0**, and the shuffled counterfactual reads
  .493/.503 — so the gate has power here, it is not passing vacuously. Every code_v3
  readout (gate table, §11 fused, length strat, closure r=0, swap) is cleared.
  Companion landmines also checked: the chain calls score_one_seed_v3.py at MAXLEN=2048,
  NOT score_eval_dense_v4.py (hardcoded 1024); trainer 80/10/10 assert passes (.806/.099/.095).
- **2-seed within-repo T:** eval .7093/.7095 (**spread .0002**), test .6540/.6294 (.0246),
  combined .6829/.6713 (.0116). Ensemble T combined **.6852**.
- **2-seed residual, both definitions:** combined ensemble **Δ +.0510 [+.014, +.087],
  p=.00011, 92/144 repos**; mean-of-per-seed +.0429. Closure protocol +.0545 [+.021,+.091].
  Most conservative of all readings = **+.0267** (test, mean-of-per-seed) — still > .02.
- **EVAL IS SEED-INVARIANT, TEST IS SEED-FRAGILE.** Per-seed Δ eval +.0576/+.0578 (two
  independently trained models agreeing to 2e-4); test +.0390/**+.0144** — seed 1's test
  residual falls BELOW the .02 threshold on its own. Any final code claim must be
  split-resolved: solid on eval, seed-fragile on test. This is exactly what the seed chain
  was run to expose.
- **NEW CROSS-CUTTING LANDMINE: pooled AUC is untrustworthy ACROSS SEEDS on this cell, not
  just across splits.** Seed 1 beats seed 42 by +.031 POOLED on eval but by +.0002
  WITHIN-REPO; it is .014 worse pooled on test but .0246 worse within-repo. The two gaps are
  essentially unrelated. A mid-run inference I drew from seed 1's pooled validation curve
  ("seed gap ~.04, bigger than the assumed ±.02") is RETRACTED — within repo the eval spread
  is .0002. Do not read seed quality off pooled AUC on composition-dominated cells.
- Ensembling gain is real: 2-seed ensemble T (.6852) exceeds BOTH single seeds, so the
  ensemble Δ is mechanically larger than the mean of per-seed Δs. Both reported everywhere.

## 2026-08-08 — V8 N&C CO-SIGNING BUILT: SECOND FIELD WITH ALL THREE PREFERENCE COLUMNS

notes/2026-08-08__v8_cosigning_build.md; results/nc_cosigning_ledger.json. 100% A-bank
reuse (zero new judging; 99.97% join); survived the session reset via self-finisher.
- **y = byte-identical text adopted ≥2× within the docket** (n=9,520, 3.59% pos, 1,814
  dockets). Design ruling: on regulations.gov, mass duplication IS the endorsement act —
  the form-letter "confound" is the outcome itself. `Duplicate Comments` metadata DEAD
  (0.06% coverage); shipped dedup mappers rejected by ground-truthing (MinHash over-
  merge; partial coverage) — counts recomputed exactly over 6.8M docs; `y_nearby` kept
  as sensitivity arm.
- Ledger: V .696/A .666/VA_lin .687/VA_nl .7472 (Δ_interact +.060); T .8374/.7572
  (3-seed); same-rows Δ_beyond +.056-.064 pooled. **BUT docket-identity-alone .6951 =
  92% of pooled VA_nl; within-docket VA_nl at CHANCE on powered subsets** — the vote
  column's signal is WHICH RULEMAKING, not which text. Within-docket figure weighting-
  dependent (.53-.73): never point-quote; only "pooled ≫ within-docket" replicates.
- **Gate: CONDITIONAL PASS, UNDERPOWERED — NOT promoted to closure** (P(Δ>.02)=.79-.83;
  3-seed T spread .088 > Δ .064; 34 positives/split). Routed to Track-B/L2(b) first
  (V>A + INVERTED length .418 = the surface trigger). Correct restraint.
- **STRUCTURAL CROSS-Y FINDING (same 9,520 texts, same instruments): co-signed comments
  get agency responses 43.9% vs 79.4% (φ=−.160), and the instruments rank within-docket
  on the VERDICT column (.6749) but sit at chance on the VOTE column. What earns an
  agency's reply is articulable at the comment level; what earns a signature is a
  property of the rulemaking.** N&C joins peer as a complete 3-column field.

## 2026-08-08 — MASTER FUSED LEDGER LANDED (notes/2026-08-10__vat_fullgrid.md): 16/16 CELLS, ZERO STANDING-RULE FLAGS

Full table in the note (T / VA_lin / VA_nl / VAT_nl / V3 / D1b per cell, alignment-gated).
Headlines:
- **peer_curation = the grid's one genuine V3 WIN: .6386 vs its dense T .5936 (+.045,
  P=1.00)** — where raw text is least learnable, articulated criteria are worth more
  than the text. Its §11 flag cleared twice over (D1b .6054 + V3 .6386).
- Everywhere else V3 sits just below its dense parent and reliably above its bank
  (7/8 non-caption cells: V3−T −.003..−.025, V3−bank positive) — "V3 imports the bank,
  doesn't unlock new text signal" REPLICATED on six new cells across four fields.
- **Fused ≥ bank on every cell (AUTO-FABLE-AUDIT flags: none)** — the standing rule is
  satisfied grid-wide, incl. cap_crowd (V3 .6303 > bank .6217, closing its old trigger).
- **T_decor RESOLVED: DOES NOT SHIP.** V2' failed on the UTILITY sub-gates while
  reliance-removal PASSED (92% of the planted channel's causal contribution removed;
  cost −.0276 ≈ 2 SD below the vanilla seed band; NOT the Byrd-Lipton escape hatch,
  which applies to reliance failures). Quotable summary: "decorrelated training removes
  92% of measured shortcut reliance at a ~.02-.03 AUC cost its own pre-registered gate
  judges too expensive." §13's fail-branch (AFR first, ODIN second, each with battery +
  user sign-off) is now the active path if training-time protection is ever needed.
- N&C _v2 OOF regeneration LANDED with ids → the three N&C bank CIs unblock.
- Open stragglers: nc_outcome V3 + HW k10 mid-harvest; peer_revealed V3 training;
  nc_responded + jokes_k10 V3 never started (worker died; datasets verified); mathlib
  scoring pass. Process disclosure: a scripted note edit spliced ~330 lines; rebuilt
  entirely from artifacts (nothing lost, structure reordered).

## 2026-08-08 — GRID TAIL: MATHLIB CELL KILLED; ONE RETRACTION; TRUNCATION MECHANISM CONFIRMED

- **MATHLIB TERMINAL AS DENSE-INSTRUMENT FAILURE**: class-weighted 3-seed T = eval .5643
  (spread .126!) / test .4670 (at/below chance all 3 seeds). Class weighting fixed the
  collapse but not the cell; no fused quantities built (they'd difference against a
  chance-level T). Citable result: "the dense instrument fails on mathlib merge" — joins
  homepage in the documented-instrument-failure category. NEVER quote any mathlib T.
- **RETRACTION: peer_verdict V3−VA_nl +.0738 withdrawn** (worker run predating the
  bank-index rule; no gateable index exists → null). Its V3−T −.0223 (P=.01) stands.
- N&C V3−bank deltas are NEGATIVE (outcome −.0234, agree −.0205, n.s.) — corrected from
  the morning generalization. **MECHANISM: V3−bank goes negative on exactly the cells
  where the criteria block pushes truncation to ~30% — importing the bank costs more
  text than it's worth there.** Pre-registered prediction for the matched-raw control.
- Revised V3 picture: V3−T negative 8/9 gated cells (peer_curation the sole win);
  V3−bank positive 5/7. **k∈{10,20} is a WASH** (HW +.0079 [−.015,+.028]) — carry ±.03
  on every V3 number.
- Grid final-emit correctly deferred: 3 arms mid-training (peer_revealed, nc_responded,
  jokes_k10); note regenerates by command when they land.

## 2026-08-08 — CORRECTION (user catch): MATHLIB AND HOMEPAGE "TERMINAL" VERDICTS RETRACTED AS OVERSTATED

**MATHLIB — "dense instrument fails" RETRACTED.** Internal inconsistency: class-weighted
TF-IDF on the IDENTICAL canonical split scores .677/.786 while the Llama arm sat ≤chance
— a bag-of-words beating an 8B LoRA on the same rows means the TRAINING RUN failed, not
the cell. Diagnosis: the canonical de-confounded slice is n=7,956 at 94.3% positive =
~450 negatives (~360 in train) — data starvation; the user's remembered working setup
(the historic .77) trained on the ~35,796-row pre-audit population. The .770 NUMBER
stays retired (leaky split/provenance), but the CELL is live. FIX DISPATCHED:
train-big/eval-canonical — train on the full pre-audit population minus the canonical
eval/test area-groups, evaluate on the canonical rows (the CW stage-0 / caption
more-data pattern). Interim honest comparator: TF-IDF .677/.786.
**HOMEPAGE — "instrument failure, terminal" NARROWED.** The historic story-grouped dense
T .824 (registry line 51, "T .824 groupsplit prov") is real; what failed in scale-up C
was the far stricter OUTLET-HELD-OUT design with only 8 outlets (2 held out, disagreeing
in sign = grouping variance at k=2, not proof of no transfer). Correct statement: T is
recoverable under story/date grouping (dispatched on the current build population, with
outlet-transfer reported as an unpowered secondary); the census BANK's coherence failure
(entity-detector criteria) stands as a separate A-instrument issue needing rebuilt
criteria. Cell = live pending both.
Lesson recorded: "instrument failure" verdicts must state WHICH design failed and check
historic runs before killing a cell (reuse-before-rebuild's evaluative twin).

ADDENDUM (grid V3 refinement): peer_revealed V3 = .8882 — ties its dense parent (+.0040,
P=.76), +.1642 over its bank, and EDGES the D1b combiner (.8805): criteria-in-prompt
beats appending-dense-to-bank where the dense reader dominates. With peer_curation
(+.0450 over a weak T .594) at the other pole, the sharpened claim: **V3 tracks
whichever parent is stronger and adds real value only where the dense reader is weak** —
consistent across all 12/14 landed arms. 2 arms training; final table on their landing.

ADDENDUM (correction-of-correction): **peer_verdict V3−VA_nl = +.0738 REINSTATED** — the
earlier retraction traced to harvester defects (single-fit schema unread; seed0-only
cells refused although the gate operates on seed0), not to the data; gate now verified
at exact identity (.6876125399665395 == published). Corrected tally: **V3−bank positive
7/9, negative ONLY on the two truncation-hit N&C cells** — the displacement mechanism
story is cleaner, not weaker. V3−T negative 8/10 gated.
- Harvester fixes ×4 incl. a silent .loc row-duplication on a non-unique index (1,236→
  1,238 rows; ambiguous keys now dropped with counts; >2% drop refuses the CI);
  press_verdict bit-identical pre/post = corrections not perturbations; independent
  second harvester agrees to every digit.
- **PROCESS RULE (2nd occurrence this session): a NULL from a gate is only trustworthy
  after checking the gate COULD have fired** — absent capability ≠ negative finding.
- NEW ALIAS TRAP: code_competitions_va_nl_oof_mean3.npy is the 999-row AtCoder curation
  cell, NOT code_v3's bank (11,452 rows) — same decoy class as the AoPS stale split.
- hashtagwars bank-CI null is genuine (no bank OOF array exists on disk for that cell).

## 2026-08-08 — B-SIDE RECOVERY CONTROL COMPLETE: SPURIOUS MAPS = LOWER BOUNDS, NOW WITH MEASURED RATES

notes/2026-08-08__bside_recovery_control.md; closure/robust_mm/bside/. P=5/3-family
sealed fleet (GLM live), full-recall dual-judge detection (3/36 disagreements arbitrated).
| stratum | sensitivity | retained control | lift |
|---|---:|---:|---|
| overall | .500 | .444 | +.056 (P=.57, null) |
| high | .333 | .000 | +.333 (P=.91, marginal) |
| mid | .833 | .833 | .000 |
| low | .333 | .500 | −.167 |
- **The census's STRONGEST channel (repo-URL, .593) was missed by the entire fleet.**
- **QUOTATION CONTRACT (final, both sides now symmetric): every Track-B census is a
  LOWER BOUND with these detection rates attached; "the map covers the channel space"
  may never be claimed** — mirroring the A-side (.333/.556, zero lift). Both miners
  sample from priors more than they track gaps; directed priors (Addendum-4 pattern)
  remain the only demonstrated fix for known blind-spot families.
- Missing-mass certification program COMPLETE: A-side value-form bound (two agreeing
  routes) + measured sensitivity; B-side per-round mass + measured sensitivity; STOP-M
  diagnostic; estimator upper-bound rights. All claim language calibrated.

## 2026-08-08 — MATH.SE VOTE: 3-SEED GATE PASS (+.0467); τ-FRAGMENTATION FIX; PEDAGOGY RULING

notes/2026-08-10__closure_mathse_vote.md (898 lines); closure/mathse_vote/.
- **Gate PASS on all three readings** (Δ_gate +.0467 eval; +.0355 eval+test; +.0241
  selection-free test). Seeds 1-2 both ABOVE seed 42.
- **τ-embedding fragmentation was real and inflationary**: the arrival-order channel
  (named by 4 proposers across 2 families) shattered into 4 singletons at τ=.79 →
  Good-Turing mass overstated 59% relative (.767 → **.483 STRICT two-judge = figure of
  record**). Judges agree 80% and their disagreements CONCENTRATE on channel granularity
  (one 4-member vs two 2-member arrival species) — the channel survives either way.
  Scored set = loose selection (audit-chain integrity: re-selecting post hoc after
  seeing outcomes is forbidden); strict governs mass accounting only. Both states kept.
- **Agent self-retraction**: the single-seed "arrival order absorbs the whole residual"
  claim fell at 3 seeds (Δ₀ MONITOR +.0136 [−.003,+.062]; within-question the bank
  LEADS by .010; conditional on the position family dense still adds +.094 vs bank
  +.072 — large part, not all). Convention trap recorded: stratified/matched readouts
  run on the seed-MEAN vector whose pooled AUC is the ENSEMBLE figure, not T.
- **ARBITER DOMAIN RULING (settles freeze open question (b) for math.SE): pedagogical
  REGISTER is nuisance (warmth, Socratic mode → B/mixed); pedagogical SUBSTANCE is
  merit (hint-only vs full derivation → A).** 3/3 upheld; probes 4/4; misrouting .12.
- LANDMINE: LaTeX tokenizes ~2 chars/token (vs ~4 prose) — context overflow at 4096;
  fix = raise max_model_len to 8192, never shorten the bank-matched text view.
- Round-1 scoring ~30 min out, readout chained.

## 2026-08-08 — PRESS CLOSURE (rounds 0-2): RESIDUAL CLOSES WITHIN NOISE; THE BANK IS A GENRE DETECTOR

notes/2026-08-10__closure_press.md; closure/press_verdict/ (101 files).
- **RETRACTION of the dispatched anchor +.0486** (not same-rows: T from 288 dense-eval
  rows vs VA_nl from all 2,956). Same-rows on the 605 dense-held-out rows: Δ = +.0066
  (Layer-1 protocol) / +.0212 (closure protocol), jackknife CI [−.028,+.070] — **inside
  noise BEFORE mining**. Best bank (r2): **Δ_beyond +.0093, +.0007 on the selection-free
  test half → the press residual closes at this cell's resolution** (jackknife SE .031 =
  3× the residual: further rounds cannot resolve anything; rounds 3-5 available if a
  reviewer demands; frozen rule technically not fired, flags [YES,no]).
- **STRUCTURAL FINDING: the press articulated scorecard is a GENRE DETECTOR** — the 38
  applicability bits ALONE reach .7322, above the full 126-feature bank (.7296);
  swapping all judged quality levels for the bare applicability mask costs .0014. What
  the bank measures is WHICH KIND of release, not how good.
- Census: 40→36 concepts (cleanest bank in program) but ZERO columns above .527 alone.
- Spurious map: distribution tier ANTI-predicts (.403-.414 both rounds: paid-newswire
  livery → less pickup); the fleet's two strongest candidate-real criteria ANTI-predict
  (independent evidentiary grounding .389, primary-source artifact .401); recurring
  miner error = proposing authorial voice as newsworthiness (3/3 disputes → B).
  Decomposition purified in BOTH directions (all 6 components beat parents).
- Non-monotone stopping observed AGAIN (3rd time: r1 hurt −.0071, r2 helped +.0190) —
  best-bank-so-far quoting vindicated.
- Reviewer push-points recorded honestly: 2 rounds only; 9 of 45 companies carry 94% of
  rows; r0 sign flips with the A-imputation convention (resolved conservatively).

## 2026-08-08 — DECOR BATTERY V2' GATE: FAIL (rc=3) — RELIANCE REMOVAL WORKS, UTILITY GATE FAILS; D03 FINDS THE MECHANISM

- **Gate verdict (binding spec, chain stopped by design):** V2' removal-of-reliance
  FAILS on utility. Sub-gate (b) causal reliance PASSES — plant-ablation Δ_eval
  collapses +.0275 → **+.0023** [−.0049,+.0096], i.e. **92% of planted reliance
  removed**, and unlike GRL the method never amplifies reliance. Sub-gates (a)/(c)
  FAIL in the utility direction: D02 eval .7650 is −.0276 vs R00 and .0101 below the
  vanilla 3-seed band [.7752,.7926] (~2 SD below the mean; band width .017 is real).
- **D03 attribution arm (plant weights on UNPLANTED corpus, same rows): the cost is
  the REWEIGHTING, not the planted text.** D03 eval .7615 ≈ D02 .7650 (D02−D03
  = +.0035) — both ~.03 below vanilla with no plant present in D03.
- **Mechanism (verified from weight files): pooled n_eff hides a CLASS-CONDITIONAL
  collapse.** Weights concentrate on the rare counter-correlated cell — (y=0,plant=1)
  w=1.66 vs (y=0,plant=0) w=.65 — shrinking the effective NEGATIVE sample ~20% on a
  78%-positive cell (n_eff frac y=0: plant .809 / standard .796 / length .941;
  pooled .939/.900/.975). Exactly where AUC resolution lives.
- **Registered in-battery prediction (logged BEFORE D10 landed):** length weights are
  gentle on negatives (.941) → D10's utility cost should be much smaller than
  D02/D03's. D07/D09/D10 + cap D20/D21 are running as DIAGNOSTICS on GPU4 (~4h
  chain); none of them is gate-certified.
- **Production implication (stands regardless of remaining arms):** gate on
  class-conditional n_eff, never pooled; on imbalanced cells decorrelate within the
  majority class only, or cap the minority-class weight ratio.
- Artifacts: notes/2026-08-10__decorrelated_training_battery.md §4;
  sk3 datasets/notice-and-comment/debias_pilot/ (runs/D*, results/battery_decor.json
  with alignment_gates block); code methods/taste_decomposition/debias/decor/.

## 2026-08-08 — MATH.SE VOTE ROUND 1: FIRST SUB-ε ROUND; ARRIVAL-ORDER FINGERPRINT IS NULL (.492)

- **3-seed gate PASS, final: Δ_gate +.0467** (EVAL mean .6709 vs VA_nl mean3 .6242;
  per-seed .6608/.6705/.6815; clears eval+test +.0355 and selection-free test half
  +.0241). Seeds 1-2 run early on stacked GPU4 into canonical dirs; RUN_DONE sentinel
  respected by the chain.
- **Round-0 audit (3 seeds): Δ₀ MONITOR +.0136** [−.003,+.062] — the closure protocol's
  refit bank reaches .6460, above the dispatched VA_nl .6242, so the governing residual
  is smaller than dispatched. Within-question (the tier matching the y-definition) the
  bank is AHEAD by .010. Census: L5=L0=32, max|r| .557, zero pairs ≥.90 — least
  degenerate bank censused in the program.
- **Answer-position covariate localized:** label rate .629 pos-0 vs .450 pos-1; no-text
  arrival-order model .654 pooled; ρ(is_first, dense)=+.089 vs ρ(is_first, bank)=
  −.00007 — the DENSE arm carries it, the bank doesn't. Matched sampling removes 55% of
  the MONITOR residual, exact strata 37%; length/LaTeX move it the other way
  (localization, not conditioning artifact). Conditional on the whole family the dense
  arm still adds +.094 vs bank +.072.
- **The curve: MONITOR Δ₀ +.0136 → Δ₁ +.0209, gain −.0073 [−.022,+.007] = FIRST of two
  required consecutive sub-ε rounds.** HONEST tier disagrees in sign (+.0131
  [+.002,+.024]) — MONITOR's ±.032 width talking; one more round before any plateau
  language. Redundancy not mining failure: the 5 best mined criteria out-score the best
  incoming rubric (.598 vs .573) yet add nothing once fitted. Swap check clean.
- **Round 1's central negative: the conjectured arrival-order fingerprint doesn't
  exist.** Four proposers/two families independently proposed "presupposes sibling
  answers exist"; Gemma scored it corpus-wide: **.492, chance**. Whatever carries
  arrival order into the dense model, it is NOT explicit sibling reference. Every
  Track-B channel with real alone-AUC is bank-owned (markup fluency ρ=.74 with V,
  response volume ρ=.90 ≈ v_word_count); dropping MIXED channels collapses the nuisance
  model .6245 → .501. Conditional on all named channels + full bank, dense adds
  +.031-.035.
- **B-side missing mass, post fragmentation fix: .767 → .483 strict (both judges), .417
  loose.** Judges agree 80%, disagree exactly on that family's granularity; scored set
  = loose selection (strict differs by one channel; re-selecting post-audit would break
  the audit chain) — recorded, not resolved by fiat.
- Round 2 = ADDENDUM-3 directed decomposition of the MIXED arrival-order parent (the
  .492 null says SPLIT the parent, don't re-propose it); strict-only boilerplate
  channel inherits into the pool.
- New landmines: LaTeX ≈2 chars/token → bank-matched items overflow 4,096 ctx (raise to
  8192, never shorten); stratified/matched readouts run on the seed-mean vector (pooled
  AUC = ensemble figure, NOT T).
- Artifacts: notes/2026-08-10__closure_mathse_vote.md (1,023 lines);
  methods/taste_decomposition/closure/mathse_vote/ (34 result JSONs, RUNBOOK, DISPATCH).

## 2026-08-08 — A-SIDE RECOVERY AUDIT COMPLETE (Fable): SENSITIVITY = PRIOR COVERAGE; DOSE-RESPONSE RISES ALL THE WAY TO THE BANK'S CEILING

- **User's question answered: YES, rediscovery keeps rising with concept strength — no
  plateau inside the bank's range.** Logistic on all 48 targets: +.57 log-odds per .01
  alone-AUC (p=.0034); strength rank-AUC .803 (perm p=.0003); match rate by strength
  quartile .08/.36/.23/.75, top-5 strongest .80; fitted P(rediscover) .30 at alone-AUC
  .52 → .98 at the bank max .607. The .333 headline sensitivity is mostly COMPOSITION:
  15/24 held-outs sit at alone-AUC ≤ .52 where spontaneous naming runs .1-.3.
- **Failure anatomy (16 misses):** 7 out-of-register clinical/editorial guideline
  concepts with no ML-abstract realization (unproposable in principle — correctly never
  proposed); 8 adjacent-but-strictly-judged (lenient rule recovers ~.04-.08; all 9
  judge disagreements are genuine partial overlaps); 1 nothing adjacent. Slice churn
  predicts rediscovery (rank-AUC .78) but only as a strength marker — ORACLE
  label-aware slices surface held-out activity no better than shown slices (.572 vs
  .565); mechanical cosine detection unusable both ways.
- **Mechanism verdict: STATIC-PRIOR SAMPLER, three independent confirmations** — (T1)
  9/11 matched proposals recur in fleets shown UNdepleted slices; (T2) same model on a
  different slice re-proposes its own species .44 vs different models same slice .21 —
  proposals follow the proposer, not the slice; (T3) rarefaction-shaped accumulation,
  6/49 species multi-family. Predicts: sensitivity rises with P and prior widening,
  NEVER with better slices; zero leave-out lift is the mechanism's signature → the M3
  gate must keep publishing the retained-control alongside sensitivity.
- **Interventions, ranked:** (1) taxonomy-directed prompting (Addendum-4 route): rep3
  held-out mean max-cosine .690 vs .591 sealed redraw; FIRST τ≥.79 hit in the battery
  (.803; all prior fleets maxed .722); target-specific (+.06 on non-targets). **Directed
  rounds are non-independent and must NEVER feed Good-Turing/Chao1 → TWO TIERS: sealed
  fleet for the mass estimator, directed sweep for coverage, never pooled.** (2)
  Estimator-safe: raise P — subset curves still rising at P=4; beta-binomial and
  zero-inflated fits agree ~70% at P≈6-7, ~80% at P≈8-10 (asymptote .88). Error-
  conditioned slices measured dead; judge leniency bounded +.04-.08.
- Artifacts: notes/2026-08-08__aside_recovery_audit.md;
  methods/taste_decomposition/closure/robust_mm/recovery_audit/ (q1-q5 scripts+JSONs,
  recovery_audit_summary.json). Scope: CPU + 24 frozen refits + 2 luna calls, no
  judging.

## 2026-08-08 — LANDMINE (cross-campaign): SMALL-POOL STABLE-HASH PROBE DRAWS REPEAT ACROSS ROUNDS

- Found in math.SE-vote round 2, fixed before its audit ran: with only four planted
  probe pairs, the stable-hash draw handed round 2 the SAME pair set as round 1,
  silently defeating the freeze guarantee that "a fresh auditor never audits the same
  planted pair as the previous auditor." A first fix (ban the previous round's
  plain-hash draw) STILL repeated at r2→r3; the working fix chains each round's draw
  off the previous round's ACTUAL REALISED draw.
- Action item for every closure campaign with planted probe pairs and ≥2 rounds (code
  rounds upcoming; any future campaign): check the probe-draw code for this before
  round 2 launches. Realized-draw chaining is the pattern of record
  (methods/taste_decomposition/closure/mathse_vote/decompose_r2.py lineage).
- Same landing: round-2 blind audit routing was PERFECT (0/12 misrouting vs round 1's
  3/25; probes 4/4; zero arbiter disputes) — decomposing parents ON the
  register/substance boundary removes the routing ambiguity that produced round 1's
  disputes.
- Branch decision REGISTERED PRE-Δ₂ (coordinator, before the round-2 readout landed):
  a decomposition round is not a proposing round — if Δ₂ is sub-ε it counts as sub-ε
  #2 per the frozen rule's letter, but plateau language additionally requires a full
  sealed proposer-fleet round (round 3, P raised to 6-8 per the A-side dose-response)
  to come back sub-ε. Directed sweeps, if any, stay in a separate tier and never feed
  the estimator.

## 2026-08-08 — SPECIES.PY NON-IDEMPOTENCE: FLEET-WIDE AUDIT CLEAN; GUARD PORTED TO ALL 7 COPIES

- Second landmine from the same math.SE session: re-running species.py on a completed
  round SILENTLY rebuilds the τ-only clustering and overwrites the merged species.json
  that the blind audit, arbiter, and Gemma scoring are keyed to. It fired ONCE (math.SE
  r1, during a routine no-op regression check), was caught by diffing the SINGLEJUDGE
  backup, restored, and integrity re-verified three ways (species ids == scored
  crit_ids == routing ids, names match by id).
- **Fleet-wide audit (coordinator): NO other campaign was damaged.** Per-round check of
  every species.json vs its own round's routing/scoring artifacts across cw_community,
  maps_batch1 (6 cells), maps_hw_si, nc_responded, peer_curation_ext, peer_revealed,
  press_verdict, mathse_vote: the only after-round rewrites are math.SE r1 (the known,
  restored case) and nc_responded's *_P6.json files, which are separate later
  sensitivity-arm artifacts, not overwrites. Apparent id "mismatches" in nc_responded
  are schema differences (species files are summaries; routing uses P## proposal ids).
- Guard ported to all 6 unguarded copies (maps_hw_si, code_v3, maps_batch1,
  peer_curation_ext, press_verdict, peer_revealed): refuses to overwrite a species file
  carrying b_merge or belonging to a scored/routed round unless --force; compiles
  verified; all 7 copies synced to sk3 (code-sync rule).
- Math.SE branch decision implemented and STAMPED PRE-Δ₂ (11:24:11 PDT, results file
  verified absent, scoring at 2%): round 3 fleet raised to P=8 across 3 families with
  distinct salts; two-tier rule ENFORCED in code (species.py drops tier-D from the
  Good-Turing pool; no-op verified on round 1: S_obs 57/51, f₁ 48/46 reproduce).

## 2026-08-08 — CW NULL-BANK RE-AUDIT (user-prompted grid fill): BOTH VERDICTS INSTRUMENT-LIMITED; WIGLEAF "NULL" WAS A MISLABEL

- Origin traced: both null verdicts come from ONE campaign
  (notes/2026-07-05__why-metric-discovery-plateaus.md:166-356, completed 07-06), weeks
  BEFORE the mature A-bank standard (GEPA + Gemma-4-31B + K≥50 anchors) existed; the
  mature pipeline was later applied to cw_community ONLY. The registry itself still
  listed "◻ A bank (V5)" outstanding for Wigleaf three weeks after the null.
- RoyalRoad (verdict, n=1,274 balanced): clean craft-bank AUC .505 = genuine
  chance-level bank UNDER THAT INSTRUMENT (150 proposals/0 kept, retest .90) — but
  never tested with the mature pipeline, and NO dense T was ever computed.
- Wigleaf (curation, n=1,568, 404 pos): bank AUC **.578, the highest craft-rankability
  in the CW leg**, above its .570 V floor, with mining saturated on top of it. The
  strict list's "NULL BANK" tag OVERSTATED this — it is a saturation finding.
  Correcting the tag. Checklist-(a) flag: 404 absolute positives (mathlib order of
  magnitude) — class-weighting + power caveat mandatory.
- Pre-kill checklist retro-applied: (b)(c) pass both; (e) seed spread was NEVER run on
  either bank point estimate; (d) design named (37-criterion k-medoid bank, 5-arm
  proposal system, judge identity not confirmed in notes).
- VERDICT both cells: NULL IS INSTRUMENT-LIMITED → rebuild justified in writing
  (reuse-before-rebuild satisfied): rescore EXISTING clean populations with the mature
  Gemma bank machinery (score_va_gemma_banks.py / score_scaleupC_banks.py) + first-ever
  dense LoRA T. Few GPU-hours; no new data collection. Rebuild dispatched.
- Audit note: notes/2026-08-08__cw_nullbank_reaudit.md (file:line citations throughout).

## 2026-08-08 — V7 PATENTS FORWARD-CITATIONS BUILT THROUGH V LEGS: GROUND TRUTH 21/21; THREE RAW-CHANNEL LANDMINES; STRUCT BEATS THE ENTIRE V SURFACE

- **Ground-truth pass (V8 lesson, now standard): 21/21 exact matches** against a full
  scan of 151,140,729 raw PatentsView citation edges. Three channel facts, each of
  which would have silently corrupted a naive build:
  1. `citation_date` = the CITED patent's grant date (2,721/3,000 month-match cited,
     0/3,000 citing) — the existing script's comment promised the opposite; citing
     dates require the g_patent join.
  2. Zero-citation patents are ABSENT from forward_citations.parquet (min=1) — naive
     joins delete the bottom of the distribution.
  3. `citation_category` re-labels eras: "cited by other" IS the applicant bucket
     2002-2012, so literal "cited by applicant" is identically 0 for pre-2008 cohorts.
     Era-robust examiner-independent measure = tot5 − exm5.
- **Design:** y_fwd5 = within-(grant-year × CPC-class) MEDIAN split of DISTINCT citing
  patents within 5 years (74,846 dup pairs collapsed). Age confound killed by the
  window (all-time counts fall 22.7→8.0 across 2005-15 cohorts; windowed flat
  3.58-4.03). Families grouped by MinHash-LSH on claim-1 shingles + claim-Jaccard ≥.3
  guard (title-only pass had merged 26 unrelated "semiconductor device" patents).
- **Leak battery (the claim-fell killer) CLEAN:** grant year .491 / CPC section .506 /
  cohort OOF .532 / split identity .4993 / corr(group size,y) +.013. Metadata
  exclusion asserted in code ×4.
- **V legs (n=16,000, 15,973 groups, 884 cohorts, pos .509):** V_lin .5899, V_nl
  **.5950** (spread .002). **The 4-column STRUCT block (declared nuisance) .6018 beats
  all 58 V columns**, almost entirely num_claims (.596 alone) — invisible to every
  instrument (only claim 1 shown); banked as STRUCT per RUNBOOK; Δ will be quoted over
  V+A+STRUCT as well as V+A. No claim-fell-style structural killer (max single V col
  .552 vs claim-fell's ordinal .754).
- **Two NEW cross-cell landmines:** (a) `stable_hash_bucket_map`'s pos-rate term
  collapses on near-singleton groups (would have yielded eval/test n=14) — replaced,
  will bite any cell with ~1-member groups; (b) constructed-anchor calibration: first
  degraded tier scored .190 ≈ scrambled .192 (would have invalidated every shard),
  recalibrated to pos .688 > neg .383 > scram .185.
- Six deviations documented in the note (incl. claim-fell rows/bank NOT reused — its
  own RUNBOOK demands rebuild at the new unit; fresh 35-criterion bank, 28A/7B,
  4 proposers, hand-logged semantic merges; self-citations unremovable → examiner-only
  y reported alongside; calendar-year window resolution).
- IN FLIGHT: A-bank Gemma scoring (560K calls, resumable) + dense T (checkpoint
  .5953 at step 160 — early, not a T). Ledger will land at
  methods/taste_decomposition/results/patents_fwdcites_ledger.json.
- Artifacts: datasets/patents/v7_community/; notes/2026-08-08__v7_patents_forwardcites_build.md;
  outputs/va_gemma_banks_patents_fwdcites/; 3 commits, local/sk3 md5-verified.

## 2026-08-08 — DECOR D10: REGISTERED PREDICTION CONFIRMED — UTILITY COST IS MONOTONE IN NEGATIVE-CLASS n_eff SHRINK

- The prediction logged BEFORE D10 ran (length weights are gentle on negatives →
  much smaller cost) is confirmed: D10 eval **−.0050** vs R00, against −.028/−.036
  for the harsh weight sets. Dose-response across all four N&C arms is MONOTONE in
  negative-class effective-sample fraction: D09 (n_eff 1.000) +.003 · D10 (.941)
  −.005 · D02 (.809) −.028 · D07/D03 (.796/.809) −.036/−.031.
- Cross-cutting mechanism, now with a registered-prediction confirmation: the cost of
  decorrelated training is NOT "reweighting per se" — it is the shrink of the
  effective minority/negative sample the weights induce. Gate on class-conditional
  n_eff; cap the minority weight ratio; majority-only decorrelation where possible.
- Remaining: cap arms (balanced classes, weights spread both classes, n_eff .974 —
  prediction: near-zero cost) + full analyzer pass → final table in
  notes/2026-08-10__decorrelated_training_battery.md.

## 2026-08-08 — DECOR BATTERY COMPLETE: NOT CERTIFIED FOR PRODUCTION; RELIANCE-REMOVAL MECHANISM CERTIFIED; CAP_CROWD "BANK>DENSE" NEEDS RE-BASING

- **Final verdict:** V2' FAILED on utility → T_decor arms do NOT ship. But the
  reliance-removal mechanism PASSES decisively — the ONLY debias-series instrument
  that actually removes shortcut reliance (GRL amplified +.028→+.123; reweighting
  collapses +.0275→+.0023, 92% removed), with the failure mechanism fully identified
  and dose-response-confirmed across 5 arms (utility cost monotone in minority-class
  n_eff: 1.000/.941/.809/.809/.796 → +.003/−.005/−.028/−.031/−.036).
- V3'a: real signal NOT eroded (realtok reliance GREW +.0097) — fails the literal
  .005 band in the harmless direction. V3'b PASS.
- **V4' INDETERMINATE, and that is itself a §13 finding:** the two adopted readout
  instruments DISAGREE IN SIGN on this cell at n≈2K (matched-sampling −.0117 vs
  stacked-increment +.0255). The §13 "read jointly" rule now has a measured
  incoherence scale; treat sub-.03 debias readouts at n≈2K as unreadable.
- **cap_crowd shortcut-suppression hypothesis NOT supported:** T_decor .5988 vs
  T_van .6047 (paired −.0059); joint-B reliance was already ≈0 in vanilla.
- **SIDE FINDING requiring re-base:** a fresh same-recipe vanilla hits .6047 on the
  SAME rows where the archived cap_crowd T = .5554 → the caption-crowd
  "bank>dense" gap is trainer-dependent: −.110 archived → −.066 matched-trainer.
  RE-BASE before quoting that gap anywhere (Style-Invitational echo: instrument
  effects masquerading as cell structure).
- **Production spec if ever fielded** (recorded, not shipped): stabilized IPW
  w=P̂(y)/P̂(y|s), grouped-CV OOF train-only, p-floor 1e-3, 99th-pct clip, mean-1
  renorm, per-example loss weights, 2-batch gradcheck, **NEW GATE: class-conditional
  n_eff ≥ .95 in the minority class** (pooled n_eff is not a safe pre-check). Valid
  roles: reliance CERTIFICATE + targeted decorrelation of mild channels.
- Artifacts: notes/2026-08-10__decorrelated_training_battery.md (final);
  methods/taste_decomposition/debias/results/battery_decor.json (CIs +
  alignment_gates); GPU claims released. Coordinator v3-grid relay preserved at
  fusion/v3_grid/COORDINATOR_RELAY_2026-08-08.md.

## 2026-08-08 — MASTER FUSED LEDGER FINAL EMIT: GRID CLOSED 16/16, ZERO §11 FLAGS (notes/2026-08-10__vat_fullgrid.md)

- Coverage: VAT 16/16 · Direction-1b 4/4 · V3 13/14 (aops/code structurally blocked;
  nc_responded landed .7950, cw .6912, jokes k10 confirmed .7411) · matched-raw
  displacement control 4/4. Best fused arm beats the bank at full strength on EVERY
  cell — the standing rule closes with zero flags.
- **DISPLACEMENT PREDICTION REFUTED ON N&C, CONFIRMED ON PRESS (pre-registered, logged
  at full prominence):** the claim that V3's negative N&C deltas were a truncation
  artifact is WRONG — displaced text carried nothing (raw_matched − T = +.004/+.003/
  −.002) and V3 stays below raw at matched budget (nc_responded −.0202, P=.01). It
  was RIGHT on press_verdict, the one cell already truncating (−.0218 P=.03; block
  worth +.0188 matched). Transferable: **raw_matched − T is the diagnostic; the
  truncation RATE (~.30 both groups) predicted nothing.**
- **Cross-cutting (paper-grade):** (1) Δ_beyond spans −.028 → +.229 across cells —
  wider than every methodological delta in the battery: taste size is a property of
  the CELL, not the instrument. (2) Δ_interact NEGATIVE on 10/16 cells — nonlinear
  interaction among articulated criteria is not a general signal source. (3) §11
  fused-vs-bank comparisons are only meaningful at matched training strength (all 4
  D1b cells agree in sign; E-refit vs full-strength gives pessimistic, once
  sign-flipping reads).
- **NEW LANDMINE: GroupKFold fold assignment is ARCHITECTURE-DEPENDENT** (unstable
  argsort, linux vs arm64) — generalizes the sklearn-version landmine: any frozen
  grouped-OOF number must be reproduced on the machine that produced it. Also
  re-confirmed: pre-_v2 saved OOF arrays don't reproduce (≈.50 on three cells) —
  _v2-with-ids files are the only citable ones.
- Exclusions with measured reasons: T_decor (utility gate), mathlib (dense at/below
  chance — big-train rerun still pending separately), aops/code V3 (no bank scores on
  dense train split).
- Orchestrator self-corrections logged at full prominence: scripted-splice deletion
  (rebuilt from artifacts), the wrong peer_verdict retraction (null from a gate that
  couldn't fire), and the displacement refutation above. All GPU claims released.

## 2026-08-08 — FROZEN DESIGN (before any scoring): UNTRAINED-T FUSION ARM (user directive)

- User (2026-08-08): "in addition to the V+A+T trials we've been doing with trained T,
  I think we need V+A+T with an untrained T as well as another baseline."
- **T₀ (untrained dense)** = the SAME base checkpoint the LoRA T trains from
  (Llama-3.1-8B), ZERO-SHOT, offline batch vLLM. Elicitation frozen ex ante: one fixed
  template per cell — a one-sentence cell-specific question naming the preference
  variable + the document (same 1024-token truncation as T) + "Answer Yes or No" →
  score = P(Yes) from first-token logprobs over {Yes, No}. Templates written for all
  cells BEFORE any scoring, stored in one json; NO prompt iteration against labels
  (that would make it trained-by-selection); NO bank-criterion text in the prompt.
- **Readouts per cell, same E rows and same frozen fusion protocol as the master
  ledger:** (1) T₀ alone; (2) VAT₀_nl = Layer-1 HistGB stack on [bank + T₀ column],
  grouped-OOF refit on E — identical machinery to VAT_nl (direction1_mirror.py), only
  the dense column swapped; (3) paired group bootstraps: VAT₀−VA_nl (does fusion help
  at all without training), VAT_nl−VAT₀ (the value of training), T₀−T.
- Question: how much of each cell's fusion gain requires training on the community's
  labels vs generic LLM prior. Prediction NOT registered (exploratory arm; two-sided).
- Eligibility = the 16 master-ledger cells (bank-on-E exists). Inventory existing
  zero-shot scores first (reuse-before-rebuild) before generating new ones.
- SCOPE AMENDMENT (user, same day): "for all cells" — the T₀ arm runs on ALL 16
  current ledger cells now, and becomes a STANDING COLUMN of the battery: every
  future cell (V6 SO, V7 patents, V8 co-signing, V9 tweets, CW rebuilds, homepage,
  mathlib when resolved) gets T₀ + VAT₀ in its ledger at build time.

## 2026-08-08 — UNTRAINED-T ARM COMPLETE (16/16): FUSION DOES NOT WORK WITHOUT TRAINING; Δ_beyond IS LEARNED PREFERENCE, NOT LLM PRIOR

- Frozen design HELD: 16 templates written from label definitions, committed (a47d8fc,
  sha256 50c1a5a9…) BEFORE any score existed; no template touched after; one collapse
  (hashtagwars, 7 distinct P(Yes)/924 rows → T₀ reads "uninformative" not "chance")
  recorded, not patched; 4 cells saturate median P(Yes)=1.0.
- **T₀ (base Llama-3.1-8B zero-shot) alone: .431–.573, mean .511; T₀−T NEGATIVE on
  16/16 cells** (P=.00 on 15), including where trained T is strongest (peer_revealed
  .8842 vs .4988; nc_responded .8167 vs .4310 — the generic prior is ANTI-correlated
  with agency response).
- **VAT₀−VA_nl: median +.0003; positive at P≥.95 on exactly ONE cell; significantly
  NEGATIVE on code_v3 (−.0095, P=.01** — a chance-level column actively costs the
  stack). VAT−VAT₀ positive at P≥.95 on 13/16. Median untrained share of the fusion
  gain ≈ 0%; <10% on 12 cells, negative on 8.
- **The one exception is diagnostic: cap_finalist +.0374 [+.013,+.062] P=1.00 (+34%
  share)** — one-line captions under an EDITOR label, the one cell where a generic
  prior can guess editorial shortlisting; the SAME corpus under the crowd-median label
  (cap_crowd) gets −18%. The prior tracks editorial taste a little and crowd taste
  not at all — a cross-y contrast on identical text.
- **Program consequence: falsifies the "the dense arm is just an LLM reading the
  text" reading of Δ_beyond.** Identical rows/folds/bank/1024-token budget: Δ_beyond
  survives with the trained column and evaporates with the untrained one. Stated
  caveat: bounds THIS instrument (one 8B base checkpoint, one frozen question per
  cell), not LLM priors generally — but that is the confound the arm was registered
  to kill. T₀+VAT₀ is a STANDING COLUMN for every future cell (user scope: all cells).
- Discipline: 16/16 E-rows assert-matched to the master ledger (n_E, groups, T);
  micro-landmine: peer_verdict has 5 duplicate ntitle keys in E → joins use
  position-prefixed uids. Platform check: 14/16 reproduce published VA/VAT to 1e-4
  on the kept box; press_verdict + code_v3 flagged (ledger's sklearn/arch combo
  unavailable on either box) — does not touch T₀ comparisons (folds shared
  bit-identically within-run). sk3 code_v3 variant still computing; sign stable.
- Artifacts: results/t0_untrained_arm.json; notes/2026-08-08__t0_untrained_arm.md;
  fusion/t0_templates.json + t0_*.py; GPU 6 claimed/released.

## 2026-08-09 — FULL SWEEP QUEUED (user /goal): 3 GPU lanes + F2 DECONFOUNDED-FUSION SPEC FROZEN

- User goal: full metric-discovery + full spurious discovery + full VA + deconfounded
  fused arms (T trained AND untrained) for ALL cells. Queue + stage matrix + lane
  assignments: notes/2026-08-09__full_sweep_queue.md. VA is already DONE on all 16
  grid cells (never-repeat); discovery gaps queued: jokes/cap_finalist/cap_crowd
  (lane A, GPU5), mathse_accepted/aops/nc_agree/nc_outcome (lane B, GPU6),
  mathse_vote r2-completion + r3 P=8 fleet + code gate readout + rounds (lane C, GPU7).
- **F2 FROZEN before any run** (full text in the queue note): per cell on ledger E
  rows — VA_enriched (terminal bank + promoted criteria), NUIS (Track-B channels +
  declared STRUCT), VAT_dec = stack on [enriched + nuisance + T column], twin arm
  with T₀. PRIMARY = stacked increment VAT_dec − (VA_enr+NUIS): the taste residual
  conditioned on everything nameable INCLUDING named nuisance. §13 stack applies
  (matched-sampling sign duty, reliability band). T₀ twin expected ≈0 (registered
  expectation from the t0 arm; positive = nuisance-prior interaction flag).
- Overnight loose ends found at queue-build: code 3-seed gate readout MISSING though
  seeds trained (agent died at session limit pre-readout); mathse_vote r2 results
  file absent (scoring chain state unknown). Both = lane C first items.

## 2026-08-09 — CODE 3-SEED BINDING GATE: **HOLDS** (Δ +.0519). Rounds authorised, NOT started.

notes/2026-08-09__closure_code.md §9.3; closure/code_v3/gate_readout_3seed.json.
Seed 2 trained+scored 2026-08-08 15:39/15:54; GPU 6 RELEASED 22:54Z. OOF alignment gate
re-run and passed (shuffled counterfactual .493/.503). `BINDING: true`.

- **GATE HOLDS.** Frozen statistic (within-repo n-wtd Δ, both splits combined at the repo
  level, Layer-1 protocol, dense = 3-seed ensemble): **Δ = +.0519, [+.015, +.089],
  Wilcoxon p = .00022, 91/144 repos, 10,284 rows** vs threshold .02. Conservative
  mean-of-per-seed **+.0408**. Closure protocol **+.0554 [+.023, +.091] p = .00016**.
  Most conservative reading anywhere in the table = **+.0254** — still clears.
- **3-seed within-repo T ± spread** (round-0 item 1): eval .7093/.7095/.7008 (mean .7065,
  **SD .0049**); test .6540/.6294/.6377 (mean .6404, **SD .0125**); combined
  .6829/.6713/.6707 (mean .6750, **SD .0069**). Ensemble T combined **.6861**.
- **RETIRE the test-side +.0390.** At 3 seeds the test residual is **+.0254 (mean-per-seed)
  / +.0344 (ensemble)** — a third smaller — its ensemble CI includes 0 ([−.008, +.082]),
  and **seed 1 alone gives +.0144, BELOW the .02 threshold**. Correct phrasing henceforth:
  "test-side residual +.025 to +.034, seed-sensitive, one seed in three below threshold".
  EVAL is near seed-invariant: +.0576/+.0578/+.0491, mean +.0548, all above threshold.
- **Both new seeds came in BELOW seed 42 within-repo** (.6713, .6707 vs .6829) — the
  direction that shrinks Δ — so the single-seed gating figure was **optimistic by ~16%**
  combined (+.0487 → +.0408 mean-per-seed). The gate was not a foregone conclusion.
- Cross-cutting (see 08-08 entry): pooled AUC is untrustworthy ACROSS SEEDS here, not just
  across splits — seed 1 is the BEST pooled-eval model (.6795) and the WORST within-repo
  one. Never read seed quality off pooled AUC on composition-dominated cells.
- **Rounds 1..5 AUTHORISED but NOT STARTED** — GPU lanes assigned elsewhere (full-sweep
  queue, notes/2026-08-09__full_sweep_queue.md); slotted lane C behind math.SE on GPU 7.
  All machinery staged and unfired; the mining slice will be built on the now-final 3-seed
  ensemble (this is why firing early was refused).

## 2026-08-09 — MATH.SE VOTE: Δ₂ SUB-ε (#2 BY LETTER); ROUND-3 P=8 FLEET MID-FLIGHT — DOSE-RESPONSE CONFIRMED AS MEASUREMENT

- **Δ₂ (decomposition round): gain +.0013 MONITOR / −.0003 HONEST — sub-ε #2 by the
  rule's letter**; residual after r2: Δ_beyond .0363 MONITOR / .0284 HONEST. Per the
  registered pre-Δ₂ branch, NO plateau language until the r3 SEALED FLEET also returns
  sub-ε. Discount table: all-B joint conditioning leaves Δ_adj .029-.041 (≥ pooled) —
  the spurious set predicts y (.59-.62 alone) but does not absorb the dense edge.
  **swap_signature TRUE on HONEST** — r3 readout must be checked for the same pattern
  before any saturation language (bank buying rank agreement ≠ bank out of content).
- **Round 3 (sealed P=8, 3 families, 8 salts, TIER S only, 200 proposals): the A-side
  dose-response prediction confirmed as a DIRECT MEASUREMENT** — Track A Good-Turing
  missing mass .533 → .283, cross-proposer recapture .158 → .360; Track B .483 → .350,
  recapture .098 → .378, zero singletons in the scored B set. Strongest available
  evidence that rounds 1-2's sub-ε readings were not an under-powered miner.
- **B01 (arrival-order/from-scratch-setup) named by ALL EIGHT proposers** — highest-
  consensus channel of the campaign (in r1 the τ clusterer had shattered it into
  singletons). Blind merge now CONFIRMS the clusterer (47→45 species vs r1's 51→41).
- Audit: misrouting .12, probes 4/4 (chained draw used the brief's other named pair).
  **First arbiter reversal:** A05 notational-discipline → MERIT; A08/A14 register
  channels upheld as nuisance. Boundary now triangulated by two fleets, two auditors,
  two arbiters: *notation as argument = merit; notation as typography = nuisance.*
- Handoff: campaign agent at end of context (record = notes/2026-08-10__closure_
  mathse_vote.md, 1,436 lines, governing addendum included). Δ₃ chained on GPU 7
  (PIN_GPU fix), writes mathse_vote_r3_results.json; recovery-by-harvest documented.

## 2026-08-09 — V9 JOURNALISM COMMUNITY CELL COMPLETE (tweets): Δ_beyond +.0348/+.0212 WELL-POWERED; SECOND EXPERT-VS-CROWD ANTI-CORRELATION

- **Channel decision: TWEETS over the reddit arm**, for grid reasons: the tweet
  population IS the homepage-captured URL corpus (same 9 outlets) → journalism gets a
  cross-y contrast on ONE population sharing headlines + V bank + A bank (reddit
  overlaps 2.3%); within outlet×day grouping structurally kills the domain confound
  reddit needed machinery for; A-bank reuse population-exact (zero new judging).
  SCOUT-RECORD CORRECTION: the reddit "36h-refetched final scores" claim is FALSE
  (rows_with_retrieved_2nd_on = 0 of 621,352); r/politics was never collected.
- **Ground truth held:** 52,112 rows, zero duplicate URLs; 1,393 true zeros recorded,
  4,913 API errors dropped as missing-at-random; the 62.2% cap is a UNIFORM retrieval
  limit absorbed by within-group rank (ρ=.926 with max_likes among capped); label =
  trailing attention (type:Latest); the earlier "8% coverage" worry was the wrong
  statistic — the scraper completed groups (median within-group coverage 1.000; 602
  groups ≥95%).
- **Ledger (n=31,129, 508 outlet×day groups, pos .4999):** V_nl .5399 · VA_lin .5704 ·
  VA_nl .5947 (seed spread .0014) · T .6300 eval / .6478 test · Δ_interact +.0247
  [+.019,+.031] · **Δ_beyond same-rows +.0348 eval / +.0212 test** (pooled +.0352 is
  cross-population, never quote). WELL-POWERED unlike V8: T seed spread .0050 ≈ Δ/7.
  Group identity alone .5000 exactly; length alone .4972; holds in all 6 outlets
  (.551-.623) and under censoring-robust y_maxlikes. OOF alignment abs_diff 0.0.
- **CROSS-Y FINDING: journalism is the SECOND field where expert and crowd
  anti-correlate on identical text** — 861 dual-label rows: φ = −.141 (P(top placement
  | high engagement) .513 vs .652), joining N&C's φ = −.160.
- **Two shared-helper defects (any short/high-NA cell):** (a) scramble() reverses
  alternate words from a 2-headline pool → proper nouns survive and judges correctly
  score them coherent; (b) run_battery's all-NA drop SELECTS FOR failed scrambles, and
  score_bank counts a nan scram mean as failure when it is the ideal outcome.
  Certification still passes on the independent placement channel (.647). Deliberately
  NOT patched mid-wave (would move every cell's numbers); queue a helper-fix + re-cert
  pass BETWEEN waves. Recommend coherence scored on non-NA count.
- Artifacts: results/journalism_tweets_ledger.json (+ oof ids/npys);
  datasets/journalism-tweets/va/ (population, manifest, prekill_baselines, dense 3
  seeds); outputs/va_gemma_banks_journalism_tweets/; notes/2026-08-08__v9_journalism_
  community_build.md; commit 10e6cf2, local/sk3 sha1-verified.

## 2026-08-09 — JOKES ROUND 1: ONE PROPOSING ROUND CLOSES THE RESIDUAL; THE BANK WAS MISSING PROSODY

- **Headline: VA_nl MONITOR .7278 → .7448 (gain +.0170 [+.0104,+.0238], 3.4× ε) — the
  round's gain EXCEEDS the entire round-0 residual.** Δ_beyond MONITOR: +.0143 → −.0027
  (campaign T .7421) / +.0241 → +.0070 (ensemble T .7519). Stopping clock reset;
  round 2 required. Caveat carried: swap_signature true but asymmetric (C₊ +.0230 /
  C₋ −.0036) — mostly genuine content, some inherited dense ordering.
- **The star criterion: "Read-aloud cadence" alone-AUC .682 on HONEST — a single mined
  criterion out-predicts all 47 criteria of the GEPA'd expert humour bank (best
  incumbent univariate .592).** The bank was built from WRITTEN-craft rubrics and
  missed prosody. Also mined: punchline closure .630, clean semantic double-use .592.
  GEPA phrasing pass required before any final quoted number.
- **Cross-cutting replication #2 of "proposers name position channels but cannot see
  them":** 8/8 sealed proposers named the era channel (strongest consensus of the
  round), yet the judged score correlates with the RECOVERED REAL timestamp (86.1% of
  rows) at ρ = −.034; no B channel exceeds |ρ|=.058; the observed era ordinal alone
  reads .596 (label rate monotone 2015 .353 → 2019 .575, dense tracks it) and adds
  +.0226 over all 15 named channels stacked. Replicates math.SE r1 (.492 fingerprint
  vs .614 ordinal) under UNANIMOUS naming.
- **τ-clustering failed in BOTH directions within one round** (extended two-judge blind
  merge to both tracks): A over-merged 47→70, B under-merged 49→38, identity anchors
  passing throughout. Figure-of-record missing mass: A .425 [LOPO .410-.524] /
  B .363 [.343-.414] — wide-open concept space (recapture .27/.24), consistent with a
  round that found large gains.
- Track B: spurious-alone joint .699 → matched sampling triggered; 12/15 channels
  MIXED (the auditor/arbiter pushed 5 craft/circulation channels to incidental — on
  humour the nuisance/merit boundary runs through the middle of craft); decile Δ_adj
  stable +.004-.007; matched estimator reported as sensitivity only. Dense increment
  over B + enriched bank +.0197.
- Coordinator carries: (a) TWO T conventions on this cell — master ledger .7469 =
  seed ensemble, campaign T .7377 HONEST / .7421 MONITOR = mean-of-AUCs, never mixed;
  (b) split-half noise 3× ε (eval-only Δ +.0072 vs test-only +.0218 at r0, registered
  before rounds).
- Gates: OOF abs diff 0.0; fleet 16/16 P=8; probes 4/4; anchors .914.
  Artifacts: closure/jokes_community/; notes/2026-08-09__closure_jokes.md.

## 2026-08-09 — HOMEPAGE CURATION: DENSE T FORMALLY RECORDED; CENSUS-BANK COHERENCE FAILURE MECHANISM FOUND; BANK V2 (29 CRITERIA) ARMED

- **Corrected dense T recorded** (samerows_T_homepage_storygrouped.json, recomputed
  from raw per-row predictions, all six AUCs reproduce exactly): eval mean .7109
  (.7173/.7093/.7061, spread .0113) / test mean **.7397** (.7429/.7401/.7362, spread
  .0067). Side-by-side with historic .824 (different split) and retired .4322
  (outlet-held-out, unpowered), explicit never-conflate ruling + row-id restriction
  for any Δ_beyond (press precedent).
- **Coherence-failure mechanism (per-criterion, from saved anchor blocks, zero GPU):**
  the census bank's two best predictors (hard-vs-soft .605, top-tier-story .599) score
  WORD SALAD at 1.00 — entity detectors, not reading instruments; every criterion
  returns NA on ≥50% of scrambled anchors (two at 100%), so nanmean row-scoring +
  silent all-NA drops let token-presence detectors dominate → scrambling RAISED the
  bank score (.5776 scram vs .5414 pos). Caveat: 18 rows, biased draw; stage 0
  rescores all 14 census criteria through a fresh K=50 battery.
- **Bank v2 (rubrics_v2.jsonl, 29 criteria):** 8 salvaged (3 de-genred: hard/soft →
  public-consequence, etc.), 6 dropped (2 entity detectors, 4 topic memberships at
  49-72% NA), 21 new incl. a 5-criterion coherence backbone (actor-action RELATION
  required), 2 negatively-oriented craft criteria (anti-length-model), and 3
  PAGE-RELATIVE criteria (rank focal headline against same-capture headlines — the
  within-story-type requirement). LOAD-BEARING FIX: NA = empty input only; word salad
  scores 0.0; section membership must not decide the score.
- Diagnostics built in before scoring: press-form applicability-mask ablation;
  frozen-before-A-scores story-type stratification (9 buckets); same-rows Δ by row id
  with pooled quarantined; NEW dense alignment gate (preds carry no ids → positional
  join asserted via group+judgement sequence equality, all 3 seeds); per-criterion
  coherence battery. T₀ template added under freeze discipline (hashes both sides).
- GPU stages armed behind a ~24h polling launcher; coordinator approved ALLOW_STACK=1
  on GPUs 0-3 beside the patents jobs (journalism priority; headroom verified).
- Note: notes/2026-08-09__homepage_curation_completion.md (5 recorded deviations,
  incl. GEPA-leg-via-reflective-revision rationale).

## 2026-08-09 — MATH.SE ACCEPTED: TERMINAL AT ROUND 2 (two sub-ε P=8 PROPOSING rounds) — RESIDUAL IS ARRIVAL ORDER; "FULLY ARTICULABLE" SURVIVES

- **Master-ledger overstatement corrected 3×:** the E-frame +.0702 was the E-refit
  artifact; the closure refit reproduces the full-grid's own full-fit reference to
  .0012 → **honest same-rows residual +.0229 (3 seeds); MONITOR Δ₀ +.0067;
  within-question (the tier matching the y) the bank is AHEAD by .043.** Layer-1's
  "fully articulable (−.0001)" headline SURVIVES.
- **Stopping rule fired cleanly: r1 gain −.0009, r2 gain +.0028 — two consecutive
  sub-ε SEALED P=8 PROPOSING rounds** (decomposition clause never needed). Mechanism =
  redundancy at the ceiling: 400 proposals → 20 bank-joiners → no criterion beat the
  incumbent best (.562/.565 vs .567); missing mass static across independent fleets
  (A .333→.325, B .288→.263) — out of species THAT MATTER, not out of species.
  Caveat: r2 swap_signature true (r1 clean).
- **The residual is arrival order, stronger than the vote cell:** accept rate by
  position .503/.402/.207/.131/…/.000 at pos-7 (asker is one-shot-and-early; voters
  keep flowing). **No-text 6-variable position model .6754 pooled / .6600 HONEST —
  ABOVE dense T .6375**; ρ(position-model, dense) +.148 vs +.008 for the bank.
  **Matched sampling removes the WHOLE residual (Δ_adj −.0011 HONEST)**; length/LaTeX
  strata push the other way (+.0450) = localisation. Conditional on the position
  family, dense adds +.005.
- **Position-blindness replication #3, now with two labels × two campaigns × three
  corpus-wide passes at chance:** r1 "Reply-position framing" (7/8 proposers) .516;
  r2 "Prior-answer dependence" (6/8, fleet blind to r1) .510; vote cell .492. The
  channel is UNANIMOUSLY nameable and never text-visible.
- Cross-y contrast now sharp on one corpus: ACCEPTED = articulable + position artifact
  (matched sampling → ≈0); VOTE = genuine ~.03 residual that mining cannot close.
- Process: blind two-judge merge ran BEFORE the audit (ordering fix; fired hard both
  rounds, promoted arrival-order to largest species both times); arbiter upheld
  auditor 10/10; n_answers NOT structurally neutral on this y (.650 inverted pooled,
  .500 within-question — denominator arithmetic, explains tier sign disagreement);
  lane deviation: r2 Gemma pass on GPU 7 (GPU 6 co-tenant), ledger-noted, rc=0.
- Artifacts: closure/mathse_accepted/; notes/2026-08-09__closure_mathse_accepted.md.

## 2026-08-09 — JOKES ROUND 2: RESIDUAL CLOSED BY MINING (bank now BEATS dense on HONEST); SWAP-ALGEBRA RULING; A-SPACE OPENS AS BANK CLOSES

- **VA_nl MONITOR .7448 → .7527 (gain +.0079 [+.0026,+.0109], above ε — clock still
  zero). Δ_beyond MONITOR −.0106 (campaign T) / −.0009 (ensemble T); on HONEST the
  enriched bank .7493 EXCEEDS the dense standard .7469.** Best new criterion:
  surprise-coherence balance .667. Discounted Δ_adj negative on both mixed bands.
- **SWAP-ALGEBRA RULING (agent corrected the coordinator's inverted instruction, and
  is right):** C₋ = P(bank correct | dense wrong). C₋ RISING = independent signal
  (favourable). The ADVERSE pattern is C₊ up while C₋ FALLS (bank inheriting dense
  ordering) — which is what r1 did (flagged); r2 lifted both (+.0096/+.0093,
  signature false). Rule of record for all campaigns: flag dC₊>0 with dC₋≤0; dC₋>0
  is the good direction.
- **A-side missing mass OPENS as the bank closes: .425 → .600** (72/120 proposals
  singletons) while B stays put (.363→.350). The honest reading of this cell: the
  residual was never a taste bound — it was an UNMINED BANK. (Contrast: mathse
  accepted's mass was static at ceiling.)
- Addendum-4 fingerprint fails a 3rd time on this design (max|ρ| .052 across 11
  channels vs observed created_utc; ordinal still adds +.0391 over all named
  channels). τ-clustering both-directions failure replicates (A 49→89, B 43→39).
- **Screens: "Read-aloud cadence" .682 is now QUOTABLE** (fidelity .724, modal .368,
  NA .005, sign ok — passed without rephrasing). Cumulative: 24 mined, 16 quotable,
  1 collapsed-excluded, 7 GEPA-targeted, 2 sign-triggered both KEEP (defect-naming
  criteria are sub-chance by construction).
- **Two instrument defects fixed + recorded:** (a) the COLLAPSE GATE was flagged but
  not enforced (clean_fit's <5 off-modal threshold let a modal-.988 criterion through
  at n=16K; enforcing raised r1's gain to +.0175) — check other campaigns' clean_fit
  usage; (b) file-existence-vs-completeness race in audit finalize read a half-written
  verdicts file (wait loops now test parseability).
- r2 hygiene: fleet 16/16, probes 4/4 rotated, misrouting .04 (down from .24),
  anchors .780, 0 collapsed. Next: r3 = decomposition (doesn't count), r4 = proposing.

## 2026-08-09 — HOMEPAGE BANK V2: SMOKE GATE .99 (census .56 draw-noisy); ONE STAGE-0 CLAIM CORRECTED; FULL SCORING LIVE

- **CORRECTION to the 2026-08-09 homepage entry:** the "two best predictors score
  word-salad at 1.00" claim came from the 18-row saved-anchor read; at a fresh K=50
  battery, a13 hard-vs-soft reads .581 coherence (ABOVE chance) — that half is
  RETRACTED (its de-genring to b14 stands on section-label reasoning alone). The
  archived .3869 coherence figure itself is DRAW-NOISY (.5617 on a fresh K=50 draw,
  same prompts) — never lean on the scalar; the per-criterion decomposition is the
  durable evidence: 9/14 census criteria below chance on coherence, a02 rates word
  salad .80 (2× real headlines, coherence AUC .241), the all-NA drop is 13×
  concentrated on scrambled rows, and all six dropped criteria vindicated
  (.401-.496). Two salvaged criteria (a02 .241, a14 .440) are below chance in the
  LEGACY instrument — recorded as a falsifiable prediction that the v2 repairs fix them.
- **v2 gate: coherent-vs-scrambled .9900 (0/29 below chance; scrambled row mean
  .0000; zero all-NA drops; item NA .0000).** Backbone .840-.970. The press
  genre-detector route is closed by construction (applicability channel empty).
  Shard 0 certified attempt-0 (pos .897 > neg .810 > scram .000).
- **Caution of record (registered before Layer-1):** the v2 bank's UNWEIGHTED row
  mean reads pos-vs-neg .479 — below chance. If A_lin also lands near chance, the
  honest conclusion is that this cell's articulated surface carries little placement
  signal once the genre channel is removed (the census .5979 was plausibly the genre
  channel itself). More informative than the census number either way. Five craft
  criteria near-ceiling on professional headlines — kept (no post-hoc screening),
  flagged by distribution check.
- Co-tenant identity corrected in the ledger: GPUs 0-3 hold another user's 4-GPU DDP
  run (l1ly, lucky-pretraining-seeds) — the patents V7 jobs have FINISHED. Homepage
  scoring live stacked on GPU 0, ~4h to Layer-1 ledger.

## 2026-08-09 — AOPS ROUND 0: RESIDUAL SURVIVES INTACT (+.0112 MONITOR); THE POSITION PATTERN BREAKS; r1 MID-FLIGHT

- **Δ₀ +.0112 MONITOR [−.031,+.056, LOPO SE .022] / +.0096 HONEST — the master ledger
  does NOT overstate this cell** (three VA fits agree to .003: OOF .7705 / full-fit
  .7735 / closure refit .7710). Contrast mathse_accepted (cut 3×). Within-problem the
  bank is AHEAD (−.0128). Selection-free test half +.0215 vs eval −.0017 — split
  disagreement carried. Structural: this cell's A/V population IS the dense held-out
  set → HONEST = E exactly, ONE dense seed, single T convention for every readout.
- **The sibling position pattern BREAKS here (an informative negative):** observed
  ordinals recovered at 100% coverage — first-solution rate falls .807→.616 by rank
  8+, thread age U-shaped — yet the joint position model reads .6332 vs T .7806 (on
  mathse_accepted it BEAT T); ρ(position,T)−ρ(position,bank) = .019 (vs .140 there);
  NO discount shrinks Δ (position/length/LaTeX/age all ±.006 of pooled); matched
  sampling NOT triggered (recorded before numbers read). Whatever AoPS's residual is,
  it is NOT arrival order. Diagnostic-only curiosity: thanks_received INVERTED (.428
  — more-thanked solutions are LESS likely to take the editorial's approach).
- Census: L0=L5=44, zero merge edges, max|r| .848, collapse 0%, alone-AUC max .667 —
  least degenerate bank in the program (hive-mind caveat travels). Surprise:
  **v_numeral_density .680 out-scores every articulated criterion.** Collapse gate
  (now ENFORCED in closure_core.clean_fit, modal>.98) dropped v_list_marker_count
  (modal .9843) at round 0 — the exact jokes pattern.
- **RETROACTIVE CAVEAT FOR MATH.SE CAMPAIGNS:** the blind two-judge merge is now
  generalized to BOTH tracks (run before audit). The math.SE campaigns merged Track B
  ONLY → their A-SIDE missing masses (accepted .333→.325; vote .533→.283) are
  potentially INFLATED by the same f₁ fragmentation mechanism. Not re-run; recorded
  as a caveat wherever those A-masses are quoted.
- Round 1 mid-flight: 13/16 slots parsed (165 proposals; 3 GLM slots on patient
  retry, P=7 already above degradation floor); cell-specific constraint recorded (no
  reference-comparison criteria — the judge never sees the editorial); MODE-3/4
  Track-B for the thread container; probe chains verified disjoint r1-r5.
- Artifacts: closure/aops_curation/ + DISPATCH.md; notes/2026-08-09__closure_aops.md.

## 2026-08-09 — V7 INTERIM: DENSE COMPLETE (T .6730 eval / .6826 test, tight); A-BANK SHARD-7 CRASH = CHARS-VS-TOKENS LANDMINE (patents variant)

- Dense 3/3 seeds complete + scored: eval .6765/.6697/.6729 (mean .6730), test
  .6837/.6836/.6805 (mean **.6826**) — spread .007/.003.
- A-bank had crashed at 7/8 shards (no _meta, no battery — earlier "finished" premise
  wrong, agent corrected it before harvesting). Cause: truncation budget in
  CHARACTERS vs engine limit in TOKENS — patent claims are token-dense (reference
  numerals, chemical names, indexed variables), the LaTeX ≈2-chars/token landmine's
  patents variant. **Rule extended: truncate in TOKENS, never characters, on any
  token-dense corpus (LaTeX, patent claims, code).**
- Fix ruling recorded: --max-model-len 4096→8192 with prompts BYTE-IDENTICAL (never
  re-truncate mid-bank — shard consistency beats budget purity; temp-0 decode of an
  under-limit prompt is invariant to a larger cap; no RoPE scaling at 8192 on
  Gemma-4). Two engine instantiations noted as first check for any shard-7 anomaly.
  Shard-7 resume on GPU 7; ledger step pre-flighted (OOF gate diff exactly 0).

## 2026-08-09 — JOKES ROUND 3 (decomposition, exempt): POSITION-BLINDNESS MECHANISM FOUND — THE NAMEABLE THING NEAR AN ORDINAL IS ITS CO-OCCURRING CRAFT

- Round 3 ran with NO proposers at all (decompose_only_merge.py; components TIER D,
  empty tracks block → structurally exempt from stopping + Good-Turing). Blind audit
  PERFECT: 0/8 misrouted, probes 4/4, zero disputes; campaign misrouting .24→.04→.00.
  Gain −.0024 (expected — components restate absorbed parents). Clock still zero.
- **MECHANISM for the Addendum-4 fingerprint failures (now 3 cells): decomposing the
  era parent splits it into load-bearing references (craft, .562) vs pure dated
  period markers (.454, mildly ANTI-predictive).** When 8/8 proposers named "era
  anchoring," the judgeable instruction measured the TOPICAL CRAFT that co-occurs
  with age — not age. That is why observed ordinals keep adding over every named
  channel (+.023 r1, +.039 r2): **the nameable, judgeable thing in the neighbourhood
  of a container ordinal is the craft that co-occurs with it.** Positive argument for
  observed-ordinal covariate lines as PERMANENT design (Track B cannot recover
  ordinals in principle). Registered follow-up: test on patents/code where ordinal
  families are stronger.
- Freshness parent = two OPPOSITE-SIGN channels cancelling: genuine comic invention
  .654 (real) vs overlap-with-stock-joke-material .660 — the latter now the
  strongest Track-B channel of the campaign. Both halves out-predict the parent
  (.386): superposition inside one criterion, measured.
- Readout ruling: joint-B .723 on HONEST → decile stratification losing resolution;
  the STRATIFICATION-FREE STACKED INCREMENT is the readout of record from here
  (stable: dense over bank+nuisance +.0149 HONEST / +.0165 MONITOR ALL_B).
- PRE-REGISTERED terminal-language distinction (before r4): if A-side mass rises
  again while gains fall, the verdict is "bank saturated while the concept space
  kept opening," NOT "miner exhausted." Cumulative: bank 102 features, 29 nuisance
  channels, 4 parents retired.

## 2026-08-09 — LANDMINE (infra): PLAIN `setsid` DOES NOT DETACH WHEN ALREADY GROUP LEADER — USE `setsid --fork` + ASSERT ppid=1

- V7 shard-7 rescore died at ssh-session teardown: plain `setsid` EXECS WITHOUT
  FORKING when the caller is already a process-group leader, so the "detached" job
  never left the session. Earlier jobs survived the same pattern by luck. Symptom was
  worse than death: EngineCore took the SIGTERM but the python parent WEDGED at 3%
  CPU for ~49 min — an "is the process alive" check reads healthy while nothing runs.
- Rule of record: `setsid --fork` (or nohup + & + disown) and ASSERT ppid=1 + own
  session id after launch, for every detached sk3 job. Liveness checks must measure
  PROGRESS (output growth), not process existence.
- Recovery followed the kill discipline exactly (wrapper 1299485 → python 1299487 →
  EngineCore 1299831, targeted PIDs, in order); watchdog crons ruled out as
  killers before relaunch. Distinct from the earlier char-vs-token crash (unrelated
  failures, both recorded). Dense stands confirmed (T .6730 eval / .6826 test); no
  A-side number quoted until the battery certifies.

## 2026-08-09 — AOPS ROUND 1: NO CLOSURE (registered); POSITION-BLINDNESS BOUNDARY FOUND — LEXICALIZED ORDINALS ARE VISIBLE; PROBE BATTERY CATCHES A BIASED AUDITOR

- **Curve: Δ₀ +.0112 → Δ₁ −.0004 MONITOR, but the registered reading (pre-r2) is NO
  MEASURABLE CLOSURE:** MONITOR half-width ±.043 dwarfs the move; HONEST says gain
  +.0010; matched sampling puts MONITOR Δ_adj back at +.0187. Gain +.0116 above ε →
  clock not started; r2 proposing round launched. **Swap signature FIRED ADVERSE
  (dC₊ +.0027, dC₋ −.0047): the r1 increment is NOT quotable as clean articulation
  gain.**
- Bank 65→76 (+11 criteria); best newcomer "method sophistication proportional to the
  problem" .654; the multiple-choice-option-list channel (.595) is corpus-specific in
  a way no rubric-writer reached. Incoming best .667 not cleared.
- **HEADLINE — the 3×-replicated arrival-order negative does NOT replicate here, and
  the exception proves the mechanism:** fingerprint "inter-post referencing and
  sequence markers" (8/8 proposers) reads .582 ≈ observed sol_rank .579 (joint
  position model .633). On math.SE fingerprints read .49-.52 vs observed .61-.66.
  **AoPS sequence furniture is LEXICALIZED in the text ("Solution 2", "another way");
  math.SE's is only pragmatic. Refined law: ordinals are text-recoverable exactly
  when the community's conventions write them into the document.** (Fits the
  ordinal-craft mechanism from jokes r3: what's judgeable is what the text carries.)
- **Missing mass, two-judge merge BOTH tracks: τ HID the difference (.500/.500) —
  strict says A .5583 (τ over-merged A; A far LESS covered) vs B .3375 (τ
  under-merged).** Concretely validates the retroactive math.SE caveat: campaigns
  that merged only B and quoted A at τ have unreliable A-masses.
- Discount: mined joint-B .7231 arms matched sampling; **discounting RAISES Δ on
  every estimator** (decile +.0325, matched +.0284 HONEST) — the strong B channels
  are bank columns in judged clothing. Stacked increment: dense over B+bank +.0201
  HONEST / +.0120 MONITOR.
- **Probe-battery precedent (first auditor failure of the program):** auditor #1
  failed a planted probe 3/4; readmission rule REGISTERED BEFORE the replacement ran
  (audit_readmission_rule.json, incl. both-fail branch + no-third-draw clause);
  auditor #2 passed 4/4; the two disagreed 27/29→2 cases, BOTH in the failed probe's
  direction = systematic bias, caught exactly as the battery is designed to.
  Auditor #1 retained, never deleted.
- Hygiene: anchors .9789, 0/25 collapsed, GPU 6 claim/release rc=0, 133,800 prompts
  in 24 min. R2 fleet mid-flight.

## 2026-08-09 — JOKES ROUND 4: FIRST SUB-ε PROPOSING ROUND (clock 1/2), SWAP-ADVERSE — INCREMENT RULED NOT-QUOTABLE; R5 = CAP, DECIDES TERMINAL

- **Gain +.0011 [−.0039,+.0030] p=.69 — sub-ε — AND adverse by the swap rule of
  record** (dC₊ +.0058, dC₋ −.0033; Spearman(bank,dense) .6955→.7072, largest jump
  since r1). The two readings agree: the miner found nothing while the aggregator
  bought rank agreement. **Ruled at landing: r4's +.0011 must never be quoted as a
  real articulation increment.** Clock 1/2; round 5 (cap) decides the terminal
  verdict either way; if r5 is not sub-ε the campaign terminates BY CAP with the
  stopping rule never having converged — to be stated plainly if so.
- **Trend rulings:** (a) A-side mass .425→.600→.475 — r2's spike was one round's
  fan-out, NOT monotone opening → terminal language updated to "bank saturated,
  concept space CHURNING" (neither pre-registered extreme); (b) B-space genuinely
  consumed (.363→.350→.287, recapture .24→.28→.32); (c) τ both-directions failure
  is a STABLE property of the cell (3rd straight round); (d) the craft/circulation
  boundary RE-OPENED post-decomposition (misrouting .00→.16, all 4 disputes Track
  A→incidental in the same family, arbiter upheld all; "pre-owned chestnut
  provenance" tops Track B at .632 for the third round running — freshness/
  provenance is humor's persistently contested territory).
- Conduct: GLM window verified before commit (8s at full budget) → full P=8/3fam;
  probes 4/4; anchors .9374; stacked increments +.0169/+.0152. Cell state: Δ_beyond
  −.0093 campaign-T / +.0004 ensemble-T; bank .7511 > dense .7469 HONEST.

## 2026-08-09 — JOKES TERMINAL: CLOSED BY MINING, 5 ROUNDS, RULE-FIRED (not cap) — ENRICHED BANK BEATS DENSE UNDER BOTH T CONVENTIONS

- **Stopping: two consecutive sub-ε proposing rounds (r4 +.0011, r5 −.0003), clock
  2/2 — the rule fired on its own terms** (cap coincided; the decomposition exemption
  did real work: without it r3 would have started the clock early).
- **Terminal ledger:** bank 74→124 features; VA_nl terminal .7552 MONITOR / .7527
  HONEST vs dense .7421/.7519 (campaign/ensemble) → **Δ_beyond TERMINAL NEGATIVE
  under both conventions on both populations (−.0033 to −.0149); total closure
  +.0274 [+.0144,+.0361] p=1.00.** Strict merged masses: A .564 / B .300.
  Stacked-increment discount stable (+.0134-.0162).
- **GEPA stages 1-4 complete: 15/17 rephrasings accepted (sealed, probe-rows-only
  decisions), worth +.0041 MONITOR — a SIXTH of total closure came from fixing
  WORDING, not finding criteria.** Best fix modal .743→.398. Read-aloud cadence .682
  stands as authored (never needed targeting). Cumulative collapse gate: 1 exclusion
  total.
- **Seven cross-cutting lines** (full text §8 of the campaign note): (1) prosody —
  the residual was an unmined bank, not a taste bound (pre-campaign +.015 would have
  been quoted as taste); (2) ordinal-craft mechanism ×3 measurements; (3) MIXED
  channels UNDERSTATE both components (freshness .386 → .654/.660 opposite signs);
  (4) swap and sub-ε AGREE at both terminal rounds — that agreement is what makes
  termination trustworthy; (5) τ fails both directions EVERY fleet round —
  blind-pairwise identity is load-bearing; (6) the craft/circulation boundary is
  irreducible on humor (one family drove every re-route, survived a targeted
  decomposition); (7) bank saturated while A-space CHURNS (.425→.600→.475→.564) and
  B-space is consumed (.363→.300, recapture .24→.385) — two different exhaustions,
  distinguished in the claim.
- **Claim discipline:** plateau = "not discoverable by this miner at P=8 over 4
  fleet rounds with A-mass ≈.56," never "no such criteria exist"; r4's +.0011 is not
  an increment; never difference the two T conventions; round deltas are not robust
  (split-half > ε), the terminal LEVEL is.
- Artifacts: jokes_community_TERMINAL_LEDGER.json; notes/2026-08-09__closure_jokes.md
  §7-9. Lane A rolls to cap_finalist.

## 2026-08-09 — CAP_FINALIST TERMINAL (BY CAP, STILL GAINING): BANK'S LEAD WIDENS TO −.06/−.07; DENSE CARRIES NOTHING; VIEW-REPAIR LANDMINE

- **LANDMINE (cross-campaign): ITEM-VIEW MISMATCH.** Inherited maps_batch1 r1/r2
  scored mined criteria on the CAPTION ALONE while the bank scored CARTOON+CAPTION —
  a caption is near-ungradeable without its drawing. Round 3 ran as a VIEW-REPAIR
  pass (TIER R, no fleet, clock- and Good-Turing-exempt) rescoring all 50 criteria on
  the matched view (rank-corr old-vs-new only .678/.776). **The repair REVERSED the
  inherited curve: +.0344 MONITOR / −.0111 HONEST → +.0140 / +.0005 — the mismatched
  view was manufacturing MONITOR-only gain.** Rule: every mined criterion scores on
  the BANK'S item view, asserted in code.
- **Terminal answers:** mining widens the bank's lead — Δ_beyond MONITOR −.027 →
  **−.0603**, HONEST −.053 → **−.0680** (terminal VA_nl .7100/.6803 vs T .6497/
  .6124); the dense residual carries NOTHING (stacked dense increment over bank+38
  nuisance channels −.0028/−.0021, all CIs straddle zero, stable across 3 nuisance-
  set sizes). **Terminated ON THE CAP while still gaining** (r5 was the campaign's
  largest, cleanest gain: +.0204 MONITOR / +.0129 HONEST, only HONEST CI excluding
  zero, swap NOT adverse; clock reset to 0 at termination). Honest statement: mining
  NOT exhausted; terminal VA_nl is a LOWER bound (GEPA stages 2-4 unrun — queued as
  F2-adjacent follow-up; cannot flip any sign).
- **Addendum-4 structurally null here — first "cannot carry by construction"
  instance:** 3 finalists/contest ⇒ constant label rate (contest ordinal .5022); the
  corpus has NO arrival-order ordinal at all.
- **The editor-vs-crowd contrast, sharpened on identical text: within-contest crowd
  rank runs .335 AGAINST the finalist label** (hard-negative construction), and 16/62
  mined craft criteria are significantly ANTI-predictive — the standard comic-craft
  vocabulary runs BACKWARDS on the editor's pool (kept, aggregator sign-free).
- **MODE 5 (generic-prior-guessable quality) named by 7/8 proposers — the T₀
  exception construct is articulable.** τ both-directions failure replicates on a 2nd
  cell (stable program-wide). V9 anchor rule AMENDED with reasoning: non-NA-count
  coherence is structurally degenerate at NA≈0 (exactly .5000 by construction) —
  item-mean reading operative (.9994-1.000 here); scrambled anchors manually
  inspected.
- Deviation recorded: r5 ran P=5 / 2 families (session Claude-subagent cap hit;
  judge/auditor legs moved to luna+glm, all recorded). Artifacts:
  closure/cap_finalist/ + cap_finalist_TERMINAL_LEDGER.json;
  notes/2026-08-09__closure_cap_finalist.md.

## 2026-08-10 — CAP_CROWD TRANSPLANT (registered predictions P1/P2): LEVEL SHIFT, NOT SIGN INVERSION — PARTIAL RETRACTION OF CAP_FINALIST'S "RUNS BACKWARDS"

- Registered before fitting, tested by transplanting cap_finalist's criterion TEXT
  onto cap_crowd (exact-name join was empty): **P1 PASSES 12/12** — every editor-cell
  sign-contradicting craft criterion reads significantly ABOVE .5 on the crowd label
  (e.g. late-twist .449→.579). **P2 FAILS informatively** — rank correlation of all
  25 transplanted criteria across labels is POSITIVE (ρ +.411 p=.041): editors and
  crowds AGREE on which craft matters more.
- **Corrected mechanism: a uniform ~.04 LEVEL SHIFT (mean alone-AUC .514 editor →
  .553 crowd), not an inversion** — enough to push the weak half of a craft bank
  below chance on the editor's hard-negative pool while the same criteria stay
  positive for crowds. cap_finalist §10.7's "comic-craft vocabulary runs backwards"
  is PARTIALLY RETRACTED: the observation (16/62 anti-predictive there) stands; the
  ordering does not invert; P3's falsifier did not trigger (hard-negative
  construction remains the explanation for the shift).
- Read cap_finalist's cross-cutting line 7 against this paragraph wherever quoted.
  Source: notes/2026-08-09__closure_cap_crowd.md §7.1-7.2.

## 2026-08-10 — CAPTION DENSE ARMS TRAINED CAPTION-ONLY (cartoon never shown): VIEW ASYMMETRY CONFIRMED; CTX-CONTROL FROZEN

- Verified on disk (datasets/caption_contest/dense_llama/{finalist,crowd}/split/
  train.csv): the dense standard's text column is the BARE CAPTION — no cartoon
  description — while the A bank judged CARTOON+CAPTION. The dense arm scored
  punchlines without their setup: the strong form of the "dense structurally
  disadvantaged" explanation, and the DENSE-side sibling of the mined-criteria
  item-view landmine (cap_finalist r3 view-repair).
- **FROZEN CONTROL (before any run):** retrain both caption dense arms with input =
  cartoon description + caption (population desc column), IDENTICAL splits (byte-same
  group/judgement columns, same rows), identical frozen recipe/seeds. Quote archived
  caption-only T and new ctx-T side by side, never replacing silently; all downstream
  Δ_beyond re-derivations name both views. Seed 42 first, seeds 1-2 after
  validation. This is an instrument control; canonical-row replacement is a separate,
  later decision.
- Bears on: cap cells' master-ledger rows, cap_crowd re-base caveat (now THREE dense
  conventions on that cell: archived .5554 / matched vanilla .6047 / ctx pending —
  label every quote), the humor text-richness gradient, and the user's swap question.

## 2026-08-10 — USER DECISION: NEW YORKER CAPTIONS RETIRED FROM BOTH CANONICAL SLOTS

- User: "retire New Yorker Caption contest from both curation and community, and
  replace it with the next best curation (style invitational?)".
- New humor canonical: VERDICT = HashtagWars (unchanged) · CURATION = **Style
  Invitational** (promoted from appendix) · COMMUNITY = **reddit jokes** (promoted
  from replication; terminal, closed-by-mining, strongest humor cell).
- Caption cells (finalist + crowd) move to APPENDIX/CONTRAST status: their science
  stands (transplant level-shift, T₀ exception, view-repair landmine, editor-vs-crowd
  same-text contrast) and is quotable as appendix material; they no longer carry the
  canonical humor rows. The ctx-control (cartoon-inclusive dense) continues as an
  instrument note — its outcome annotates the appendix, not the headline.
- FLAG (coordinator): SI's terminal verdict was "TIE, bank = length model (0/32
  rubrics survive length strata)" — promoting it to PRIMARY curation makes an SI
  mature-bank rebuild + closure campaign the natural next queue item, else the
  canonical humor-curation row rests on a degenerate instrument. Queued as a
  recommendation pending user confirmation.

## 2026-08-10 — MATH.SE VOTES TERMINAL: MAPPED, NOT CLOSED — Δ₃ SUB-ε ON THE SEALED FLEET ROUND; RESIDUAL ~.025-.038 STANDS

- **Δ₃ (P=8 sealed fleet): gain −.0019 MONITOR / +.0032 HONEST — sub-ε.** The
  registered branch's requirement (plateau language only after a PROPOSING fleet
  round comes back sub-ε) is satisfied: r1 sub-ε (proposing), r2 sub-ε (decomposition,
  by letter), r3 sub-ε (fleet). **TERMINAL. Standing residual: Δ_beyond .0381 MONITOR
  / .0252 HONEST** — survives conditioning on all named channels (+.031-.035, r1) and
  the full discount table (r2). swap_signature true at r3 = the jokes-terminal
  pattern (adverse-because-nothing-left; swap and sub-ε agreeing is what makes the
  termination trustworthy). Plateau claim: "not discoverable by this miner at P=8
  over 2 proposing + 1 decomposition + 1 fleet rounds, A-mass .283
  (dose-response-validated)". First crowd cell with a genuine mining-resistant
  residual whose carrier is measured (arrival order, not text-recoverable) but not
  fully absorbed.
- **Math field now FULLY TERMINAL 3/3** (accepted articulable/position; votes
  mapped-not-closed ~.03; AoPS live but its verdict-column sibling terminal) — wait:
  AoPS is curation and still live; the MATH.SE corpus pair is complete and is the
  grid's cleanest same-text cross-y contrast.

## 2026-08-10 — V6 STACKOVERFLOW VOTES BUILT + ARTICULABLE: BANK ≥ DENSE; "VOTE COLUMNS ARE NOT A NATURAL KIND"

- Cell: y = answer Score above its own question's median (mirror of math.SE design;
  raw thresholds rejected — question-popularity confound; y_accepted carried
  separately, never merged). n=12,202 / 5,972 questions / pos .524. Corpus: existing
  2026-06-11 SO mirror; the shipped so_python_v2 pool deliberately NOT reused (fused
  accepted∧Score≥3 label = verdict/vote confound; question-disjoint).
- **Ledger: V_nl .638 · A_lin .697 · VA_lin .700/.710 within-question · T .7074/.7050
  (3 seeds) · Δ_interact −.0014 · same-rows Δ_beyond −.0297 eval / −.0143 test →
  ARTICULABLE (scope: at 8B dense capacity), Layer-3 not eligible.** Within-question
  EXCEEDS pooled everywhere (first vote cell; consequence of the y design). Question
  identity alone .6043. Position = strongest trivial channel (.6552 within-question,
  beats TF-IDF) — reported beside every number. Anchors 1.000; OOF diff 0; collapse
  gate dropped 1.
- §11 trigger (bank ≥ dense) FLAGGED not self-cleared: 1-seed judge-context dense
  probe moved T +.007/−.001 (inside noise) — full fused arms come with the battery.
- Ground truth: SO Body is MARKDOWN not HTML (ingest docstring wrong; legacy
  strip_html shreds code); blockquote regex ate REPL prompts.
- **Cross-cutting: vote columns now span −.03 (SO) to +.11 (peer citations) across
  fields — crowd preference is NOT a natural kind**; code's two columns land on
  opposite sides of the gate (PR merge +.05 vs SO ≤0).
- Artifacts: results/so_votes ledger + datasets/stackoverflow build per agent note.

## 2026-08-10 — CW REBUILD COMPLETE: THE INSTRUMENT SWAP MOVES THE TWO CELLS IN OPPOSITE DIRECTIONS

- **RoyalRoad (verdict):** mature bank lifts the cell off chance — A_lin .5628, VA_nl
  .5558 vs old-instrument .505 (+.058). **T is at chance (.4994; seeds .482-.531) and
  the same-split baselines don't beat chance either** → per the checklist this is a
  POWER statement (141-row eval), not a training failure and not headroom: Δ_beyond
  −.056 is NOT quotable. Cell = articulated-only row, power-capped. Anchors certified
  (.658 pos-vs-neg, scram .000).
- **Wigleaf (curation): the bank is NOT certified, and that is the finding.** K=50
  battery INVERTS (pos .8798 < neg .9016, pos-vs-neg AUC .498) while
  coherent-vs-scrambled is .993 — the judge executes craft criteria perfectly and
  cannot see the editor's cut: 83% of responses saturate at 1.0 because BOTH classes
  are already-published literary flash. Replicated under char- and token-truncation.
  Same family as cap_finalist's hard-negative level shift: **publishable-quality
  pools defeat generic craft banks.** VA_nl .6051 ≈ T .6054 (Δ_beyond +.0003) but A
  adds nothing over V and log-length alone reads .680 — the articulated ceiling here
  is surface, not craft. Old ".578 best craft-rankability" verdict: new instrument
  reads it WORSE (.541) — direction, not magnitude, is the quotable part (CIs ±.032).
- Discipline: enforced collapse gate active (verified +.0004 isolated effect); token
  truncation material for RoyalRoad (1,077/1,274 chapters over budget; char-trunc run
  preserved); assembled-order gates 0.00e+00 both; Wigleaf class-weighted with the
  404-positive caveat carried INSIDE the ledger JSONs.
- **Two new landmines:** (a) split-provenance — the n=1,274 topicstrat cell uses
  md5("split::"+fiction_id)%1000, NOT deconfound_v2's md5(fiction_id)%10 (wrong rule
  reproduces at chance .656); (b) train_reward_model.py gained opt-in
  DENSE_SPLIT_FRACTION_ATOL (default unchanged) for stable-hash splits landing off
  the nominal fractions — reshuffling would violate the stable-split rule.
- Option noted, not launched: a Wigleaf discovery campaign would need criteria that
  can rank WITHIN published-quality pools (the editor's actual axes); generic-craft
  saturation predicts standard mining saturates too.
- Artifacts: results/cw_{royalroad_verdict,wigleaf_curation}_ledger.json (+oof/ids);
  notes/2026-08-08__cw_royalroad_wigleaf_rebuild.md; outputs/va_gemma_banks_cw_expert/.

## 2026-08-10 — USER DECISIONS: SI REBUILD GO; COMPETITIONS UNPARKED (retrieve, don't redo); JOKES FUSED-ARM AUDIT; BBC MOST-READ CELL ADDED

- SI mature-bank rebuild APPROVED (canonical humor-curation); captions retirement
  confirmed final.
- Code competitions UNPARKED: user — "we have this matrix... retrieve the scores we
  have and go forward with new features. Don't redo work." Confirmed on disk:
  results/code_competitions_layer1.json (A_lin .6907 / A_nl .6696, n=999, 634 groups,
  pos .851; flags: no V layer ever built, T population-mismatched). Forward plan =
  keep A-scores, ADD V features + honest same-population dense T, same-rows readouts.
- Jokes VA-beats-VAT audit: timing artifact hypothesis → F2 refit ([enriched bank +
  nuisance + T] stack) runs FIRST on jokes; registered prediction fused > enriched
  bank by ≈ the measured stacked increment (+.013-.016). Escalates to full audit if
  refuted.
- NEW CELL: BBC most-read (journalism community #2; dataset already built per
  2026-06-12 taxonomy note) — tests whether tweets' platform dynamics vs a
  same-outlet readership list changes the community residual.

## 2026-08-10 — F2 JOKES AUDIT: PREDICTION CONFIRMED, FUSED-BEATS-BANK RESTORED (+.0382); TRACK-B ALONE OUT-PREDICTS THE FULL BANK ON JOKES

- **The VA-beats-VAT trip was the stale-arm timing artifact, as registered:** F2 refit
  on [enriched 124-col bank + 58 nuisance cols + T] gives (d) .7596 vs enriched bank
  (a) .7214 — fused beats bank by +.0382. PRIMARY stacked increment (d)−(c) =
  **+.0151 [+.0086,+.0206] P=1.000 — inside the registered +.013-.016 band.** No
  escalation. §13 discharged: Westfall-Yarkoni attenuation makes the increment
  LARGER at every simulated reliability (survives incremental-validity critique in
  the right direction); matched sign check on the top channel agrees at all three
  calipers.
- **COORDINATOR RULING on the flagged LEACE trigger:** spurious-alone .7282 > .65
  puts per-channel matched MAGNITUDES in the untrustworthy regime — LEACE becomes
  mandatory only for any HEADLINE per-channel number on this cell; the primary
  (d)−(c) is not per-channel and stands without it. Ruling recorded.
- **Escalated finding 1: the named spurious channels ALONE (.7282) out-predict the
  entire 124-criterion articulated bank (.7214) on jokes** — Track-B is the LARGER
  nameable block (+.0248 over the bank, more than T adds on top). On crowd humor,
  what the community responds to is better described by the "nuisance" vocabulary
  (stock-material overlap, era-craft, register) than by the craft bank.
- **Escalated finding 2: the untrained twin adds exactly nothing inside the
  deconfounded design** ((e)−(c) = −.0002, P=.43) — independent replication of the
  T₀ result + rules out nuisance-prior interaction (the registered secondary's
  target).
- **CORRECTION (t0 arm): code_v3 "VAT₀−VA significantly negative (−.0095 P=.01)"
  RETRACTED** — that was the mac fit which failed the ledger gate; the ledger-exact
  sk3 fit reads −.0021 [−.0099,+.0054] P=.31. VAT₀−VA is significantly negative on
  NO cell; headline unchanged (untrained share <10% on 15/16, median ≈0%).
- **SO votes: F2-BLOCKED, correctly** — banks on disk but no Track-B map (no closure
  campaign), no master-ledger E row, no T₀ column. Queue ruling: SO gets a discovery
  campaign before F2 (added to sweep queue after current lanes).
- Design caveat carried: (a) is the E-refit of the enriched bank; the TERMINAL_LEDGER
  .7527 is the campaign's FIT+MINE→held fit — name both designs whenever compared.
  Battery: 10 remaining terminal cells running on their ledger-reproducing boxes.

## 2026-08-10 — CODE ROUNDS OPEN + COMPETITIONS UNPARKED: V LAYER BEATS THE LLM BANK; TWO DEVIATIONS AND A FOLD LANDMINE

- **Code decomposition pass: cleanest opening audit in the program** (0.0% misrouting,
  0 disputes, probes 4/4; 5 parents → 5 real + 5 surface + 5 position channels).
  Instrument win: the decomposed criteria are ANSWERABLE — NA .01 vs the incoming
  bank's .58, pos-neg separation .114 vs .076. Scoring in flight (GPU 7, ppid=1;
  prefix-cache pricing recorded: ~1.6 h per 25-criterion round).
- **FLEET DEGRADATION RECORDED (largest of any cell): Claude family unavailable
  (session subagent cap) → fleet = codex ×4 + GLM ×4 = P=8 / 2 FAMILIES — above the
  P floor, BELOW the family floor**; decomposer and auditor were both luna in sealed
  contexts (independent by context, not by family). Carried as a limitation on every
  round-1+ number from this campaign until a Claude-family leg can be re-added.
- **Competitions unparked: the new 27-feature deterministic V layer BEATS the
  139-criterion LLM bank** (V_lin .7258 / V_nl .7289 vs A .6875/.7130; V+A .7383;
  A adds +.009 over V). Dominant channel is ANTI-predictive code size — v_n_chars
  alone .262 (.738 reversed): shorter, simpler code is more likely the editorial
  approach. Language inert alone but conditions size (+.018).
- **NEW LANDMINE (small-minority cells): plain GroupKFold(5) at 149 minority rows is
  a ~.02 fold-composition artifact** (26-37 negatives/fold, A_lin .6683);
  StratifiedGroupKFold(5, shuffle) equalises (29-30) and reproduces the recorded
  .6907. Rule: stratify grouped folds whenever the absolute minority count is small.
  A_lin reproduces; recorded A_nl .6696 superseded (live .7130; old file already
  carried GATE_FAILED flag).
- Honest same-population dense T chained (class-weighted pos_weight=n_neg/n_pos,
  same-rows OOF saved, folds abort <15 negatives). Notes: closure_code.md §11 +
  2026-08-10__code_competitions_unpark.md.

## 2026-08-10 — BBC MOST-READ (journalism community #2): GROUND-TRUTH PASS REJECTS SHIPPED POOL; INSTRUMENTS DONE, T + LEDGER LANDING

- Ground-truth pass (V8 standard) rejected the shipped pool on 4 defects: capture
  type largely determines y (popular_page pos 1.000 / react .955 / parser-none .000 =
  29.7% near-single-class); the manifest's length effect is an artifact of that
  (within-morph 44.83 vs 44.76); **shipped splits are not day-grouped (3,343/3,421
  days span >1 split — leaks the day)**; a 2.3% link-kind stratum where href fixes y.
  Repair = filter + day-grouped re-split, NOT a rebuild (all 51,790 morph rows
  re-derive label+rank from raw captures with 0 mismatches). Label = trailing
  readership (day-D capture reports D−1), named as such.
- Population 50,761 / 2,251 day-groups / pos .4405. **Zero row overlap with V9**
  (era-disjoint corpora) → the tweets-vs-BBC contrast is CELL-LEVEL, era+mix
  confounded; no paired test licensed. **Day identity alone .5807 vs V9's forced
  .5000** (natural membership y) → within-day readouts PRIMARY here, pooled
  secondary — the reverse of V9; enforced in the ledger.
- Substantive teaser (gated, real): negatives are 3× question-framed and 3×
  How/Why/What-opening — BBC readers under-click explainer framing relative to
  homepage prominence. V-only .6586, TF-IDF .7446 — a high-articulable-surface cell;
  not a length model (char length .491).
- **Shared-helper anchor defects REPLICATE on an independent cell** (scram=nan scored
  as failure; scram=1.000 from single surviving entities) — confirmed general to
  short/high-NA items; V9's scramble repair shipped from the start here. New
  micro-landmine: `pgrep -f` self-match deadlocked V9's waiter — use `kill -0` on
  the recorded PID.
- A-bank reuse population-exact (BBC is an authoring outlet of the news-values bank);
  anchors on the same independent channel as V9 (comparable certifications). Bank
  4/8 shards; dense chained GPU 1 (ppid=1 asserted). Ledger →
  results/bbc_mostread_ledger.json; note notes/2026-08-10__bbc_mostread_build.md.

## 2026-08-11 — SI REBUILD MILESTONE: THE "BANK = LENGTH MODEL" VERDICT RESTED ON A CONTAMINATED POPULATION; ANCHOR PROTOCOL INVALID FOR MIXED-ORIENTATION BANKS

- **Ground-truth pass (V8 rule): 1,574/9,637 rows (16.3%) contain no joke text** —
  parser artifacts (orphan bylines 1,111, list headers 226, section markers 133,
  cartoon selectors 87, truncated orphans 17), **11× concentrated in the negative
  class** (HM 19.1% vs winner 3.1%), mean 22 chars. Removing them: length-alone
  pooled .6227 → **.5520** (within-week .6181 → .5589). **~60% of the v1 "length
  model" was parse-artifact detection.** Byline format itself at chance (.4985);
  detector conservative (real short entries kept); fragments flagged, not deleted.
  → The SI v1 terminal verdict ("TIE, bank=length, 0/32 survive length strata") is
  now CONFOUNDED — carried as a caveat; the clean-population rebuild decides the row.
- **SHARED-HELPER BUG (upstream fix due in score_va_gemma_banks): the K=50 battery
  averages criteria UNSIGNED — negatively-oriented criteria cancel the contrast.**
  Raw battery read pos<neg (.509, "judge can't tell winners") purely from 8
  flaw-oriented criteria; sign-corrected shard-0 contrast +.0643 vs raw +.0015 (43×).
  Any mixed-orientation bank inherits this. Joins the scramble/all-NA defects on the
  between-waves helper-fix list.
- **Wigleaf saturation lesson refined: FLAW-detectors saturate on published pools
  (the flaws were edited out); survivors must name flaws published work still has**
  (stock template modal .50, stock referent .25). Week-spread smoke killed 5
  saturating flaw criteria pre-run; single-contest smokes give false collapses.
- Built: clean population 8,063/316 weeks (pos .188, minority 1,216); 36-criterion
  bank (8 negatively oriented, all with length-orthogonality justifications); NA .182
  uniform; NEW dense on clean data (old T fragment-trained → unusable); token
  truncation fires on zero rows.
- **BLOCKED on a NEW sk3 outage mode: SSH resets at the jump host during key
  exchange (since ~00:30) — pre-login, so neither AFS-token nor disk-full.**
  Idempotent resume armed (scratchpad/si_resume.sh); detached dense survived.
  **Bank scored but NOT certified until the sign-corrected battery lands — nothing
  from it quotable.** Note: notes/2026-08-10__si_mature_bank_rebuild.md.

## 2026-08-11 — CODE INTERIM: OUTAGE = WHALE JUMP HOST (sk3 itself fine); SYS.PATH SHADOWING LANDMINE CAUGHT BY DRY-RUN

- **Outage attribution corrected: `whale` (the ProxyCommand jump host) is resetting
  connections, NOT sk3** — sk3 was never shown down; all detached GPU jobs (ppid=1)
  unaffected: code decomposition scorer 3/8 shards, competitions dense T chained.
- **NEW LANDMINE: sys.path insertion order lets a bare `import cells` resolve to
  ANOTHER CELL'S adapter** (readout_code.py had maps_hw_si at position 0 → the
  HW/SI adapter). Surfaced here as a TypeError; the dangerous version completes a
  readout on the wrong cell's population silently. Blast-radius scan of closure/
  clean (5 files carry the pattern, none affected — by-name imports). Rule: import
  cell adapters BY NAME (cells_code), never bare; dry-run readouts on synthetic
  scores before real scoring feeds them (also caught a round-label crash and a
  MONITOR-length/full-population shape error; named shape guard added).
- Rulings verified-by-running: tokens-not-chars card renderer (339-737 tok cards;
  char budgets removed — they hand minified/CJK code more content at equal nominal
  size); both-track pre-audit merge tested with a PLANTED cross-track duplicate
  (caught at cosine .874; A copy kept since audit can re-route, dropped-A is
  unrecoverable).

## 2026-08-11 — SI ADDENDUM: THE v1 KILL WAS JUDGED AGAINST AN ARTIFACT-INFLATED BAR

- On the CLEAN population the entire 19-feature V block reads V_lin .5511 / V_nl
  .5836 (vs .6315 contaminated) — indistinguishable from raw char count alone
  (.5520), and .5206 within length deciles. **The parse artifacts inflated the
  length/format baseline by ~.08: v1's own bank (.613) would have BEATEN V
  comfortably on clean data.** "Bank loses to length" was substantially "bank loses
  to a fragment detector." Properly bounded: this shows the terminal comparison was
  unfair, NOT that the v2 bank succeeds — Layer-1 decides that.
- Two contamination flags beyond this cell: (a) the unsigned-averaging anchor bug is
  upstream in score_va_gemma_banks — any negatively-oriented bank inherits it;
  (b) parse_results.py feeds EVERY SI-derived artifact (v1 bank, old dense arm) —
  anything built on the 9,637-row population is suspect.
- v2 bank remains scored-not-certified until the sign-corrected battery lands
  (whale outage; 5h skip-checkpointed resume armed).

## 2026-08-11 — F2 BATCH (6 cells local): FUSED-BEATS-BANK PASSES 6/6; THE DECONFOUNDED RESIDUAL LEDGER TAKES SHAPE

| cell | enr bank | NUIS | enr+NUIS | +T (d) | T | PRIMARY (d)−(c) | untrained (e)−(c) |
|---|---|---|---|---|---|---|---|
| cw_community | .6652 | .6651 | .6931 | .7896 | .7921 | **+.0988 [+.088,+.110] P=1.00** | +.0005 n.s. |
| peer_revealed | .7202 | .7511 | .7837 | .8736 | .8842 | **+.0927 [+.056,+.129] P=1.00** | −.0006 n.s. |
| hashtagwars | .5357 | .6267 | .5863 | .6127 | .7315 | +.0297 [−.007,+.063] P=.94 | −.0039 n.s. |
| nc_responded | .7882 | .7195 | .8042 | .8325 | .8167 | **+.0200 [+.002,+.038] P=.99** | −.0058 n.s. |
| jokes | .7214 | .7282 | .7462 | .7596 | .7469 | +.0151 [+.009,+.021] P=1.00 | −.0002 n.s. |
| cap_finalist | .6014 | .6298 | .6131 | .6165 | .6124 | +.0002 n.s. | +.0140 P=.86 |

- **Fused-must-beat-bank: PASS on all six.** Untrained twin ≈0 everywhere (secondary
  behaves as registered; cap_finalist's +.014 P=.86 echoes its T₀ exception, n.s.).
- **The Track-B-is-the-larger-nameable-block pattern now holds on THREE cells:**
  NUIS alone beats the enriched bank on cap_finalist (.630 vs .601), jokes (.728 vs
  .721), peer_revealed (.751 vs .720) — all crowd/revealed-type or editor-pool cells.
- **The fully-deconfounded taste residuals (conditioned on enriched bank + all named
  nuisance): CW +.099 and peer citations +.093 remain the program's two genuine
  large residuals; nc_responded +.020 and jokes +.015 small-but-real; cap_finalist
  zero.** HW's E-frame arms are weak (8-group cell; d .613 ≪ T .732) — its F2 row is
  frame-limited, band quoted with the coarse-group caveat.
- Frames: all arms E-refit on shared folds (direction1_mirror), join assertions +
  ids-sha per cell; never quote (d)−(c) against campaign-frame Δs without naming
  both designs. Remaining: peer_verdict/curation, nc_outcome/agree (computing local);
  mathse ×2 + press (sk3, blocked on whale).

## 2026-08-11 — CODE COMPETITIONS UNPARKED AND CLOSED: **NO RESIDUAL** (dense .697 < V+A .738)

notes/2026-08-10__code_competitions_unpark.md; methods/taste_decomposition/code_competitions/.
User directive: "retrieve the scores we have … go forward with new features. Don't redo work."
A bank KEPT AS SCORED (never recomputed); what was built is the missing V layer + an honest
same-population dense T. Population reproduced and asserted: **n=999, 634 canonical_pid
groups, 850 pos / 149 NEG** (the minority count travels with every number).

- **PROTOCOL FIX FIRST, and it was worth ~.02.** Plain GroupKFold(5) gives 26-37 negatives
  per test fold and reads A_lin .6683; **StratifiedGroupKFold(5, shuffle=True)** equalises at
  29-30 and reads .6834-.6919 across fold seeds, reproducing the recorded A_lin .6907. The
  unstratified reading was a fold-composition artifact. At 149 minority rows stratification
  is not optional.
- **SAME-ROWS LADDER (per-seed mean / ensemble): V .7289/.7429 | A .7130/.7282 |
  V+A .7383/.7535 | T_dense .6967/.7241.**
  **Δ_total = −.0322, Δ_beyond = −.0416 (means) / −.0294 (ensembles).** All three dense
  seeds (.7060/.7080/.6762) sit below the V+A mean → **the sign is not a seed artifact.**
  Cell joins the DENSE-BELOW-BANK group (cap_crowd, SI, press).
- **THE NEW V LAYER BEATS THE 139-CRITERION LLM BANK** (+.016 nl) and beats dense outright.
  V is an **anti-predictive CODE-SIZE channel**: v_n_chars alone .262 (= .738 reversed),
  with identifier/line/operator/branch counts all in .26-.32. Shorter, simpler code = the
  editorial's approach. Language is inert alone (.497) but conditions the size features
  (removing it costs V_lin .018).
- **Design §11: fused .7537 vs bank ensemble .7535 = +.0002 — a WASH.** Numerically clears
  the rule, substantively fails it; reproduces "fusion reaches max(parents), nowhere
  reliably exceeds it" on dense-below-bank cells. **Contrast with the code_v3 PR-merge cell,
  where fusion beat BOTH parents (+.058 over bank) — the two code cells sit on opposite
  sides of this line, which is the useful finding.**
- **The retired population-mismatched T was .69; the honest same-population T is .6967/.7241
  — the LEVEL barely moved.** What changed is legitimacy: the number is now same-rows, and
  on that footing the verdict flips from "unknown, flagged" to "no residual".
- A_lin reproduces (.6875 vs .6907); **A_nl does NOT** (.7130 live vs .6696 recorded, and
  .6929 unstratified — both above it). The surviving file already carried GATE_FAILED, so
  the recorded nonlinear number is **superseded, not a target**.
- Bounds: 149 negatives; dense seed spread .0318 ≈ the deltas themselves, so the SIGN is
  solid and the MAGNITUDE is soft. V's edge rests on the size channel — a length-neutralised
  arm should be expected to lower V and V+A.
- REUSE not rewrite: `ac_dense.py` imports CodeDataset/mb_predict from the very script that
  produced the retired T (`scripts/dense_ceiling/run_dense_ceiling.py`), changing exactly two
  things — class weighting (pos_weight = n_neg/n_pos on the train fold; the shared runner
  used unweighted BCE) and saving pooled OOF vectors (it averaged per-fold AUCs).

## 2026-08-11 — SI v2 BANK: CERTIFICATION FAILED — INSTRUMENT FAILURE, NOT CELL FAILURE; PAIRWISE REDESIGN PROPOSED

- **Sign-corrected K=50 battery: pos-vs-neg .483 (SE ≈.058 → chance, not inverted);
  coherent-vs-scrambled .884** — the judge reads content fine and cannot see the
  winners-vs-honorable-mentions cut. Agent self-correction logged: shard-0's +.0643
  margin held on only 3/7 shards — the sign fix was real but reveals the failure.
- Acceptance: **0/33 criteria clear |AUC−.5|≥.05** (v1 managed 2); median .0065;
  direction validity **9/29 (31%)** — A_lin .5647 is a fitted composite over
  near-chance columns, not a measurement. The length-confound design DID work
  (movers don't shrink under stratification) — designing away length succeeded;
  designing in signal did not.
- Ledger (clean pop, n=8,063/316 weeks/pos .188): V_nl .5960 · A_lin .5647 · VA_nl
  .6011 · T .6241 · Δ_beyond +.0076 eval/+.0195 test. **NOT terminal per §14: dense
  seed spread .0351 > Δ_beyond (V8 signature), while T .624 over a .552 length
  baseline says the target is real. INSTRUMENT failure, not cell failure.**
- Contamination finding stands, now deflationary: on clean data "bank = length
  model" is literally false because BOTH sit near chance (A_lin .5647 > V_lin .5587;
  the ordering reversed because V collapsed, not because A rose).
- **Diagnosis: item-level ABSOLUTE scoring cannot carry a COMPARATIVE editorial
  construct** (one winner from ~25 entries/contest) — the third instance of the
  published-pool family (Wigleaf saturation, cap_finalist level shift), now at its
  purest. **Proposed (needs user sign-off — new measurement shape): (1) within-week
  PAIRWISE readout; (2) frontier-judge probe on a few hundred within-week pairs to
  separate "criteria wrong" from "Gemma-4-31B can't see this" (currently
  confounded).** No third same-shape bank.
- Upstream actions re-flagged: score_va_gemma_banks unsigned-averaging fix;
  parse_results.py contamination touches all old SI artifacts.

## 2026-08-11 — CODE_V3 DECOMPOSITION ROUND (Addendum 3) LANDS: Δ +.0554 → +.0484, SWAP FIRES ROUND ONE

notes/2026-08-09__closure_code.md §11-§13; closure/code_v3/code_v3_rd_results.json.
Scoring 8/8 shards, 171,780 prompts, NA **.0086**, **0/15 collapsed** (base bank NA was .58).
Readout on the FINAL 3-seed ensemble, so the r=0 anchor is restated at +.0554 honest.

- **CURVE POINT r_d:** honest-full (144 repos) VA_nl .6307 → .6377, **gain +.0070**,
  **Δ +.0554 → +.0484** (12.6% of the residual closed). MONITOR (29 repos) +.0611 → +.0429.
  Routing 5A/10B, misrouting **0.0%**, probes **4/4**. NOT sub-ε on either tier → **round 1
  proceeds** (trailing 0 of 2).
- **THE ε COMPARISON MUST CARRY THE SEED BAND.** MONITOR VA_nl seed spread **.0274** > the
  MONITOR gain **+.0182** — the gain is smaller than the noise on its own statistic. Quote
  the honest-full +.0070/144-repo figure; **the 30% MONITOR reduction is NOT quotable.**
  This is the §8.2 protocol warning biting on round one, exactly as pre-registered.
- **SWAP SIGNATURE FIRES AT ROUND ONE: ΔC₊ +.0188, ΔC₋ −.0061, Δρ +.019.** Matters more here
  than on N&C because this cell's round-0 C₋ was already **.5099 = chance**, so the bank is
  now BELOW chance on the pairs dense gets wrong. Part of the gain is dense-imitation, not
  articulation. Watch round 1; two consecutive swap rounds = the miner is teaching the
  student the teacher's mistakes.
- **SPURIOUS MAP: ten channels, ALL WEAK** (alone-AUC .463-.528, max deviation .037); joint
  spurious model **.5364**, far below the .65 matched-sampling trigger. **N&C hit .672 after
  one round and .712 after three — the code cell's nuisance space is much thinner.**
  Strongest and only substantive channel: **version-era vocabulary, ANTI-predictive .463**.
- **DISCOUNT NULL AGAIN — third independent family of controls to fail.** Δ_adj +.0731 (all)
  / +.0674 (mixed excluded) vs undiscounted +.0484 — Δ WIDENS, same as N&C/CW (Δ_adj is not
  an effect size). Stratification-free: after a stack absorbs all ten channels the dense
  score still adds **+.1442** within-repo (bank +.0904). With §4 position (leaves +.0509) and
  §6 geometry (leaves +.036-.054), three independent nuisance families do not explain it.
- **FLEET DEGRADATION (recorded): Claude family unavailable all session (subagent cap
  500/500).** Decomposer and blind auditor both = gpt-5.6-luna via codex exec (frontier,
  >Sonnet) — so the audit was independent by CONTEXT but not by FAMILY. Fleet for rounds
  1-5 = codex ×4 + GLM ×4 = **P=8 / 2 families**: above the P floor, BELOW the family floor.
  Both GLM keys verified live.
- Throughput datum for lane scheduling: **19 prompt/s at 15 criteria vs 63 at 83** — the
  ~7K-token PR context is prefilled per row and amortised across that row's criteria, so
  fewer criteria per row is strictly worse. A 25-criterion round ≈ 1.6 h.

## 2026-08-11 — BBC MOST-READ COMPLETE: LARGE WELL-POWERED RESIDUAL (+.086/+.069); TWEETS' SIGNAL IS NOT A PLATFORM ARTIFACT

- **Ledger (n=50,761 / 2,251 day-groups / pos .4405, within-day primary):** V_nl .6718
  · A_lin .6908 · VA_nl **.7370** · T **.8230 eval / .8097 test** · Δ_interact +.0282
  [+.026,+.031] · **same-rows Δ_beyond +.0864 eval / +.0690 test — the best-powered
  residual in the grid (T seed spread .0021 ≈ Δ/40)**. Pooled ≈ within-day (±.004) —
  day composition not exploited. Instruments ORDER the winners (Spearman vs −rank
  +.144 over 22,357 ranked positives, bank-carried). Era-stable 2017-2023. OOF 0.0.
- **The platform-dynamics answer (user's question): DEGREE, not kind.** Every
  instrument is stronger on same-outlet readership than cross-platform amplification
  (headlines predict BBC clicks far better than Twitter pickup), but the
  decomposition SHAPE matches: Δ_interact nearly identical, VA_nl reaches 89.1% of T
  on BBC vs 94.4% on tweets — **tweets is proportionally slightly MORE articulable;
  V9's signal is not a platform artifact.** Cell-level contrast only (zero row
  overlap, era+mix confounded).
- Community-column consequence: journalism community carries a SUBSTANTIAL residual
  on the readership platform (+.07-.09) — the community spread now runs −.03 (SO) →
  +.02-.03 (tweets) → +.07-.09 (BBC) → +.10-.11 (CW, citations) with the two clean
  zeros both platform-vote cells. BBC → discovery queue (journalism priority).
- **Third anchor-discipline finding: anchor rows must match the provenance the
  system prompt asserts** — the BBC prompt says "the BBC News home page" while the
  shared anchor pool is mostly non-BBC; battery .481 → .602 with provenance-matched
  anchors (V9-comparable). Ledger untouched (A scored anchor-free in-pass). Joins
  the scramble/all-NA/unsigned-averaging fixes on the between-waves helper list.
- Artifacts: results/bbc_mostread_ledger.json (+oof/ids); datasets/bbc-mostread/va/;
  outputs/va_gemma_banks_bbc_mostread/; notes/2026-08-10__bbc_mostread_build.md;
  commits 4de0770, cf861d2. Smoke-flagged criterion retraction noted (no collapse at
  scale).

## 2026-08-11 — CODE ROUND d: 12.6% OF RESIDUAL CLOSED (honest-full +.0070); THIRD NUISANCE FAMILY FAILS; SWAP ADVERSE AT ROUND ONE

- Curve: Δ r0 +.0554 (restated vs final 3-seed ensemble) → **Δ r_d +.0484 honest-full
  (144 repos); gain +.0070 = 12.6% of the residual.** Not sub-ε → round 1 proceeds
  (trailing 0/2). Audit: 5A/10B, 0.0% misrouting, probes 4/4, 0 collapsed.
  **Self-enforced §8.2: the MONITOR gain (+.0182) is SMALLER than the MONITOR seed
  spread (.0274) — the 30% MONITOR reduction is NOT quotable; honest-full is the
  quotable tier.**
- **Swap signature ADVERSE on the first round** (ΔC₊ +.0188, ΔC₋ −.0061), biting
  harder than elsewhere because round-0 C₋ was already at chance (.5099) — the bank
  is now BELOW chance on pairs dense gets wrong: part of the +.0070 is imitation.
  Two consecutive swap-adverse rounds ⇒ miner teaching the student the teacher's
  mistakes; round 1 is the test.
- **Code's nuisance space is markedly THIN: joint spurious model .5364** (vs N&C .672
  after one round); ten channels all weak (.463-.528); the one substantive channel is
  version-era vocabulary, ANTI-predictive (.463). Discount WIDENS Δ (+.0731/+.0674
  vs +.0484) — **third independent nuisance family to fail on this cell** (position
  leaves +.0509; geometry/length leaves +.036-.054; mined channels leave more).
  Stacked: after absorbing all ten named channels, dense still adds **+.1442
  within-repo** (bank +.0904). The code residual is the program's most
  control-resistant.
- Fleet limitation carried: P=8/2 families (Claude cap), below family floor.
  code_competitions closure logged by the agent (registry + strict list).

## 2026-08-11 — USER DECISION: JOURNALISM COMMUNITY CANONICAL = BBC MOST-READ (main body); TWEETS → APPENDIX. SO AUDIT ORDERED

- User: "keep twitter for the appendix, and BBC most read for the main body."
  Journalism community primary = BBC most-read (best-powered residual in grid);
  tweets = appendix/contrast (the platform-dynamics pair stays quotable as a
  degree-not-kind contrast). Strict list updated.
- User: "run an audit on StackOverflow — is there something we might be missing?"
  Independent skeptical audit dispatched. Ranked suspicion #1: VIEW ASYMMETRY (the
  caption lesson) — did the dense arm see the QUESTION context the bank judged
  against, or the answer alone? A within-question y makes question context
  load-bearing; an answer-only dense arm would be handicapped exactly like the
  cartoonless caption models.

## 2026-08-11 — SO DUAL-TRACK CLOSURE CAMPAIGN DISPATCHED (user order); T reference deferred to the audit's question-inclusive control ruling

## 2026-08-11 — SO ROUND 0: ε-RESOLVABILITY POWER CHECK BLOCKS THE FROZEN DESIGN — FIXED BY CROSS-FITTED DENSE; QTRUNC T ADOPTED; Δ₀ = WASH

- **NEW DISCIPLINE (make standard): pre-campaign ε-resolvability power check.**
  MONITOR ⊂ dense-held-out left n=501; paired round-over-round sd .00566 > ε=.005.
  Killer diagnostic: DROPPING the bank's strongest criterion (alone |AUC−.5|=.106)
  moves MONITOR VA_nl by −.0003 ± .0057 — if removing the best criterion is
  invisible, adding one cannot be visible; every round would read sub-ε on noise and
  the campaign would terminate at r2 on nothing. Fix (launched, not just proposed):
  5-fold cross-fitted dense via question-hash buckets (shared trainer untouched);
  honest set = union of test tenths (6,056 rows, selection-free) → MONITOR ~1,220,
  paired noise ≈.0036 < ε. Eval tenths kept as selection-touched sensitivity only.
- **T convention resolved with the audit (coordinated): qtrunc = question-inclusive,
  answer-preserving dense — .7277 eval / .7231 test seed-42, the strongest view;
  adopted as campaign-primary (3-seed completion running); title-view = labelled
  secondary.** Question context is worth ~+.02 to dense once displacement is avoided.
- **Δ₀ on MONITOR: wash under all three T conventions** (qtrunc −.0097 [−.049,+.030])
  — confirms the audit's re-read of the cell. Swap near-symmetric (C₊ 60 vs C₋ 72
  on 501 rows): two readers making DIFFERENT errors of similar size, not a dense
  advantage. The V6 "bank beats dense" phrasing is retired; the row reads
  "articulable — bank ≈ dense (wash)" pending campaign rounds.
- Fleet limitation for the whole campaign: 2 families (Claude cap), logged per round.
  Collapse gate dropped 1/40; 14/14 alignment gates; MONITOR trivial channels
  recorded (question id .711 / charlen .609 / position .592).

## 2026-08-11 — CERTIFICATE BACKFILL DISPATCHED (user order) + E-VALUE-ANALOG COLUMN ADDED TO F2

- Backfill: strict two-judge blind A-side merges on the ARCHIVED proposal pools of
  all 9 older campaigns (peer ×3, cw, HW, press, nc_responded, mathse ×2) —
  judge pair = luna + glm (Claude cap exhausted; identities recorded); originals
  never modified (new certA_strict.json per campaign); output = old-τ vs strict
  side-by-side table; quoted A-masses that move get corrected here on landing.
- F2 gains a per-cell E-VALUE-ANALOG sensitivity column (formula FROZEN in the F2
  note before any computation): minimum alone-AUC an unfound nuisance channel needs
  to absorb the PRIMARY increment (X), robustness ratio vs strongest found channel
  (X/Y), and the strict-B-mass-coupled odds-form bound (Z). Registered-simulation
  fallback allowed under a pre-frozen sweep design.

## 2026-08-11 — BACKFILL PRE-FLIGHT CORRECTIONS (agent catch, before any judge spend)

- **FRAMING CORRECTION: the strict merge does NOT systematically deflate Track-A
  masses** — on AoPS it RAISED A (.5000→.5583) while lowering B (.5000→.3375),
  because species_merge adds only CROSS-PROPOSER edges and so undoes τ's
  within-proposer merges. Per-cell direction unpredictable; the backfill table is a
  correction, not a deflation. (My earlier entry implied inflation-only; corrected.)
- Judge design ruling: sol+luna pair (Claude cap exhausted; GLM payload-broken;
  Codex-for-judging standing rule), single-family caveat documented symmetrically
  with the existing Sonnet-Sonnet merges; **AoPS Track-A re-run under the new pair =
  cross-instrument anchor, gating the table's readability** (reproduce .5583 within
  LOO band → deltas readable; else report as new-instrument re-survey only).
- Scope: nc_responded already strict (dropped); peer_verdict A-pool possibly
  unrecoverable (brief inventory only, no re-mining); cw_community schema adapter;
  maps_batch1 deduped against _ext dirs; species_merge apply extended to write NEW
  _strictA files (in-place rewrite was unguarded — code change + all-copy sync).
- Old-τ column banked for the six runnable cells: peer_curation r5 .7778 /
  peer_revealed r5 .5667 / hashtagwars r4 .7333 / press r2 .5889 / mathse_vote r3
  .2833 / mathse_accepted r2 .3250.

## 2026-08-11 — E-VALUE FORMULA FROZEN (committed pre-compute); TWO NEW LANDMINES; F2 7/11; STRUCT-ORDINAL ENFORCEMENT

- E-value analog frozen (89a154dd0, sha 3a2be4bf…) BEFORE any value: X = adversarial
  rank(T)-blend threshold (explicit LOWER bound), RR = (X−.5)/(Y−.5), Z = expected-max
  of unfound species from exponential tail × strict M̂; verdicts ROBUST /
  ABSORBABLE-IN-PRINCIPLE / n-a. First value: hashtagwars X > .7315 — not absorbable
  by any single channel weaker than T itself; RR ≥ 1.55.
- **LANDMINE: cross-cell sys.modules contamination** — closure dirs ship same-named
  modules; two cells' adapters in one process cross-link (mathse_vote's gate read
  mathse_accepted's OOF — caught loudly ONLY because a gate existed). **Rule: one
  cell per process.** Also: peer_verdict's 5 duplicate ntitles collapse id→position
  dicts — E taken positionally with elementwise id assertion.
- Strict-B census for Z: strict only jokes (.30) + cap_finalist (.38); 8 cells
  τ-era-tagged pending backfill; peer_verdict NO B-mass (pre-species pilot) →
  Z_UNAVAILABLE, X/RR stand.
- **Coordinator enforcement: mathse cells' F2 nuisance blocks must include OBSERVED
  position-ordinal STRUCT columns** (text channels cannot carry ordinals; the landed
  mathse_accepted +.0349 overstates its deconfounded residual if Gemma-only —
  recompute ordered; jokes gets a STRUCT-inclusive variant row).
- F2 landed 7/11: jokes +.0151 · cw +.0988 · peer_revealed +.0927 · mathse_accepted
  +.0349 (pre-STRUCT) · HW +.0297 · nc_responded +.0200 · cap_finalist +.0002.
  NUIS-beats-enriched-bank now 4/7 cells.

## 2026-08-11 — F2 UPDATES: MATHSE +.0349 RETRACTED (STRUCT enforcement); JOKES = FIRST "ROBUST" E-VALUE CERTIFICATE; PRESS F2 FRAME GUARD

- **mathse_accepted F2 +.0349 RETRACTED** — computed Gemma-only (adapter n_struct=0);
  STRUCT-inclusive recompute running with all six observed ordinal columns (raw
  observed only; the npz's label-fitted joint models explicitly excluded — recorded).
  Expected per the campaign's own matched result (Δ_adj −.0011): the cell reports as
  "carrier = structural ordinal," not taste. mathse_vote runs STRUCT-inclusive from
  the start. jokes needed no change (already STRUCT-inclusive, created_utc carried).
- **First E-value certificates: jokes ROBUST on STRICT mass** — X > AUC(T) .7469 (no
  single channel weaker than T can absorb +.0151), RR 1.55, Z .6676 from strict M̂
  .30 → X > Z. HW pending strict mass (τ-era .65).
- **press_verdict F2 +.0899 P=1.00 LANDED WITH A FRAME GUARD (do not quote as a
  residual):** its (c) is an E-REFIT bank and press is the worst E-refit-pessimism
  case (.681 vs .744 full) — NOT comparable to the closure verdict (+.0093 same-rows
  vs full-strength terminal bank). Matched-strength (D1b-style) companion ORDERED for
  press + any cell with fullfit−E-refit gap >.02 (peer_curation, peer_revealed to be
  checked); companion = the quotable number for those rows.

## 2026-08-11 — MATCHED-STRENGTH COMPANION BUILT (measured trigger, all cells); τ-vs-STRICT MASS IS NOW CERTIFICATE-DECIDING

- Companion implemented per spec: stage-1 full-population grouped-OOF enriched bank
  read on E; stage-2 frozen stack over [bank_full_oof + nuisance] ± T/T₀ with
  byte-identical folds — increments differ ONLY in bank training strength.
  Trigger computed on the MEASURED enriched-bank gap per cell (not pre-selected);
  7 cells fire (press +.063, peer_revealed +.069, peer_curation +.061, cap_finalist
  +.086, jokes +.035, mathse_accepted +.042, mathse_vote +.025). cw_community and
  peer_verdict: companion UNDEFINED (E = whole population) — recorded, not
  fabricated. quoting_rule embedded per cell JSON; both E-value variants
  (evalue_analog + evalue_analog_matched) side by side.
- **τ-era vs strict mass now decides certificates: hashtagwars = ABSORBABLE-IN-
  PRINCIPLE (Z .7799 from τ-era M̂ .65 → Ŝ_unf ≈61) while jokes = ROBUST with
  IDENTICAL X-side numbers (strict M̂ .30 → Z .6676).** Nine of eleven cells carry
  provisional τ-era Z — the certA backfill's stakes just rose: several verdicts
  expected to flip ROBUST when strict masses land.
- Battery: 8/11 arm-sets landed; mathse pair recomputing STRUCT-inclusive;
  peer_verdict/peer_curation computing.

## 2026-08-11 — F2 9/11: MATH.SE VOTES' RESIDUAL COLLAPSES UNDER OBSERVED-ORDINAL CONDITIONING; PRESS = STRONGEST ROBUST; THIRD "FLAG ≠ ENFORCE" INSTANCE

- **mathse_vote_score (genuinely STRUCT-inclusive, npz verified on box): the
  mining-resistant residual collapses to +.0085 [−.004,+.021] P=.92 once the
  OBSERVED ordinal family is in the conditioning block** (NUIS incl. ordinals .6691
  > enriched bank .6238). E-value agrees: **RR = 0.90 — the only cell < 1** (a
  single unfound channel weaker than one already found would absorb it) →
  ABSORBABLE-IN-PRINCIPLE. **Verdict refinement: the vote cell's carrier is largely
  STRUCTURAL too** (campaign frame's +.025-.038 was conditional on named TEXT
  channels only; the arrival-order family carries most of it, mirroring accepted).
  The math.SE pair now reads: accepted = fully structural; votes = mostly structural
  + small unresolved remainder.
- **press_verdict: ROBUST at RR 2.66 (strongest in battery)** — X > AUC(T) .7744,
  Z .6138; frame guard holding (E-refit +.0899 never quoted vs closure verdict;
  matched-strength companion queued as the quotable number). Press is one of few
  cells where the bank DOMINATES the named nuisance (.686 vs .581).
- **mathse_accepted retraction UPHELD — second Gemma-only instance:** the position
  npz was missing ON SK3 and the adapter FLAGGED the declared-but-missing STRUCT
  block and continued with 30 columns (tell: NUIS .5615 vs the campaign's joint
  position .6600 on identical rows). **Third program instance of flag ≠ enforce
  (collapse gate, battery sign, now missing STRUCT) → RULE: declared blocks that
  fail to load must RAISE, never warn** (committed a09ee708d). File synced; cell
  re-running.
- Battery 9/11 landed; verdicts so far: ROBUST press + jokes(strict); ABSORBABLE HW
  (τ-era Z) + mathse_vote (RR<1); null cap_finalist; pending others.

## 2026-08-11 — F2 SK3 SIDE COMPLETE: MATHSE_ACCEPTED FINAL (+.0086, 75% position); PRESS QUOTABLE = +.0622 COMPANION; LEVEL-vs-INCREMENT DISTINCTION OF RECORD; TWO RULINGS

- **mathse_accepted final (STRUCT-inclusive, npz verified): PRIMARY +.0086
  [−.0015,+.0192] P=.949 — ~75% of the retracted +.0349 was position.** NUIS jumps
  .5615→.6853 (sanity-matched to the campaign's joint position .6600). RR .64
  E-refit / 1.00 matched → ABSORBABLE. The grid's clearest structural-carrier cell.
- **Companions landed (sk3): press +.0622 [+.0326,+.1052] = THE QUOTABLE press
  number** (E-refit +.0899 labelled not-quotable); jokes +.0115; mathse_accepted
  +.0073 n.s.; **mathse_vote INVERTS: companion +.0178 [+.0060,+.0296] > E-refit
  +.0085, matched E-value flips ABSORBABLE → ROBUST (RR 1.41)** — E-refit was
  PESSIMISTIC there; measuring the enriched-bank gap per cell (vs inheriting
  original-bank gaps) was load-bearing (its gap sat just under the trigger).
  math.SE votes' verdict: mostly structural + SMALL ROBUST REMAINDER (+.018 matched).
- **DISTINCTION OF RECORD (press made it vivid): LEVEL residual (Δ_beyond = T − bank,
  the closure estimand; press +.0093) vs INCREMENTAL information (stacked (d)−(c);
  press +.0622) are DIFFERENT quantities — similar levels via partly uncorrelated
  signal means level ≈ 0 with increment ≫ 0, both true. Every quoted number names
  which it is; never conflate.**
- **RULING 1: §11 re-based for F2 rows to (d) > (c)** (fused must beat the strongest
  nameable stack; (d) vs (a) passes trivially under a strong nuisance block —
  mathse_accepted's +.1207 "PASS" was nearly all nuisance). **RULING 2: LEACE Leg-3
  scope = per-channel EFFECT claims only** (descriptive alone-AUCs exempt); 5 cells
  have spurious-alone >.65 → LEACE runs if the paper quotes per-channel effects there.
- peer_curation F2: NULL (−.0006 P=.48). peer_revealed sweep NON_MONOTONE (recorded,
  not smoothed). Remaining: peer_verdict arms, cw E-value, mac companions.

## 2026-08-11 — RULINGS IMPLEMENTED (commit 9bf23e297): UNDER THE RE-BASED RULE, THE DECONFOUNDED RESIDUAL IS SIGNIFICANT ON 5 OF 10 LANDED CELLS

- Re-based §11 ((d)>(c)) per-cell with bootstrap CI + SIGNIFICANT_AT_P95 flag; old
  (d)-vs-(a) retained as context labelled TRIVIAL-UNDER-NUISANCE (mathse_accepted's
  "+.1207 PASS" drops to +.0090 — the old rule discriminated nothing).
- **Significant deconfounded increments (P≥.95): cw +.0966 · peer_revealed +.0899 ·
  press +.0830 · nc_responded +.0282 · jokes +.0134. Point-only (CI includes 0,
  verdict_qualified guards against pass-as-positive): HW +.0264 · mathse_accepted
  +.0090 · peer_curation +.0073 · mathse_vote +.0067 · cap_finalist +.0034.**
- Press is the one cell where the nuisance block DEGRADES the bank ((c) .6686 <
  (a) .6860) — genuinely weak nuisance, consistent with its RR 2.66.
- LEACE Leg-3 markers embedded per artifact (required:false now; auto-flag on the 5
  spurious-alone>.65 cells if a per-channel effect is ever quoted). estimand
  (LEVEL vs INCREMENTAL) tags travel inline with every number. Seed-mean vs
  bootstrap-seed0 relationship stated in-JSON.
- Remaining to battery close: peer_verdict arms, cw E-value, mac companions ×7.

## 2026-08-11 — USER RULING: PATENTS HAS NO COMMUNITY CELL — FORWARD CITATIONS ARE PROCESS-ENFORCED, NOT PREFERENCE

- User: citations are enforced by design by the patent process (duty of disclosure /
  examiner citation practice) — they are relatedness/obligation signals, not a
  community's revealed preference. RULING: patents = VERDICT-ONLY field in the grid
  (itself informative — some fields structurally lack crowd preference channels).
  This matches the program's own earlier reasoning about engagement labels (the
  2026-06-16 tweetapi note called such labels IMPACT cells, not articulability cells).
- V7 forward-cites is RECLASSIFIED, not discarded: it stands as an appendix IMPACT
  cell (clean leak battery, ground-truthed label, V+T built) — quotable as
  "predicting technological citation impact," never as community preference. Its
  ledger completes as an appendix artifact; no discovery campaign will be run on it.
- Strict list + notebook updated (patents removed from the community column).

## 2026-08-11 — V_NEW WAS NEVER REAL (user catch): COMPILATION PILOT DISPATCHED

- User caught that the notebook's V_new ≈ V everywhere. Honest state: the discovery
  program routes all mined criteria into the JUDGED bank; no stage ever compiled
  discovered criteria into deterministic code — the V_new column was a display
  fallback except code-competitions (whose user-ordered 27-feature coded layer beat
  the LLM bank — the existence proof).
- PILOT dispatched (3 terminal cells: jokes, press, nc_responded) reusing the seam
  program's AGENTIC-COMPILE machinery: codability triage → compile → held-out
  certification (ρ vs judged parent + alone-AUC) → V_new = V + certified compiled
  columns → frozen-stack refits (V_nl / V_new_nl / VA_nl / (V_new+A)_nl). The
  compile-success profile per criterion family is itself a codability measurement
  (bridge to Paper #1 themes). Compiler family = luna (recorded). Scale-up only
  after the pilot table.

## 2026-08-11 — MATCHED-STRENGTH CORRECTION REVERSES TWO CELLS; PEER_VERDICT TRIPS THE INTERACTION FLAG; E-VALUES 10/10 (E-refit footing)

- **Companions computed per cell (direction NOT universal — peer_revealed and
  mathse_vote GROW at matched strength):** HW sign-flips (+.0297 E-refit → −.0230
  P=.21; its E-refit bank .5357 was essentially chance vs .6751 full — the largest
  starvation gap, +.1393); **cap_finalist goes SIGNIFICANTLY NEGATIVE (−.0224
  [−.0454,−.0005] P=.021): adding T to the full-strength bank+nuisance actively
  hurts → §11 FAIL on the matched footing (second verdict block recorded;
  FAIL-matched/PASS-E-refit = the E-refit increment was a bank-starvation artifact,
  not residual).** Note: consistent with the cartoonless-dense handicap on caption
  cells — the ctx-control (cartoon-inclusive dense) retests this arm when it lands.
- **Matched-footing significant residual set now: peer_verdict +.0588 · cw +.0966 ·
  peer_revealed +.0969 · press +.0622 · mathse_vote +.0178 · jokes +.0115
  (nc_responded companion in flight).** press/HW/cap corrections all flowed from the
  frame guard — computing the gap per cell was load-bearing.
- **peer_verdict is the ONLY cell tripping the registered SECONDARY: (e)−(c) =
  +.0175 P=.996 — nuisance-prior interaction** (T₀ alone .5573 ≈ chance, so it is
  interaction, not independent prior signal). Flagged for its own look; every other
  secondary within ±.009 of zero.
- E-values complete on E-refit footing: ROBUST cw (RR 3.58), nc_responded (3.41),
  press (2.66), peer_revealed (1.78), jokes (1.55, strict M̂); ABSORBABLE HW (τ-era
  Z), cap_finalist (.95), mathse_vote (.90 — flips ROBUST 1.41 on matched),
  mathse_accepted (.64). Commit f50c35ed3. Remaining: 3 mac companions + 5 matched
  E-values → battery close.

## 2026-08-11 — F2 BATTERY COMPLETE (11/11): THE DECONFOUNDED TASTE LEDGER — 6 SIGNIFICANT / 4 NULL / 1 NEGATIVE

- **Governing increments (companion where enriched-bank gap >.02, else E-refit):
  SIGNIFICANT — cw +.0988 · peer_revealed +.0969 · press +.0622 · peer_verdict
  +.0588 · nc_responded +.0200 · jokes +.0115. NULL — mathse_vote +.0085 ·
  mathse_accepted +.0073 · peer_curation +.0008 · hashtagwars −.0230.
  SIGNIFICANTLY NEGATIVE — cap_finalist −.0224 (P=.02).** The two coordinator-
  ordered corrections produced the hardening: STRUCT inclusion killed
  mathse_accepted (+.0349→+.0073); matched strength killed HW (+.0297→−.0230) and
  cap_finalist (+.0002→−.0224).
- Cross-cutting: (1) **the named nuisance block beats the enriched bank on 6/11
  cells** — Track-B is the larger nameable block on most cells; only press and
  nc_responded have banks that clearly dominate their nuisance; (2) untrained twin
  inert 10/11 (peer_verdict +.0175 P=.996 = the sole nuisance-prior interaction
  flag, queued for its own look); (3) E-refit pessimism is BIDIRECTIONAL (inflated
  5 cells, deflated 2) — per-cell gap measurement was load-bearing; (4) at equal RR
  1.55, strict-mass jokes certifies ROBUST while τ-era HW does not — the backfill
  decides several verdicts.
- E-values: ROBUST cw 3.58 / nc_responded 3.41 / press 2.66 / peer_revealed 1.78 /
  jokes 1.55(strict); ABSORBABLE HW / cap_finalist .95 / mathse_vote .90 /
  peer_curation .73 / mathse_accepted .64. peer_verdict E-value computing.
- Process: the duplicate-ntitle landmine bit a THIRD time (f2_evalue/f2_matched);
  y_equal_elementwise caught it each time — assertion stays mandatory in this
  script family (bffaf4642). NON_MONOTONE sweep on peer_revealed recorded.
- Note: notes/2026-08-11__f2_deconfounded_fusion.md; results/f2_deconf_<cell>.json.
  Goal items 4-5 (deconfounded fused, trained + untrained) are COMPLETE for all
  terminal cells; new cells inherit the battery at terminal.

## 2026-08-11 — CERT-A BACKFILL COMPLETE: ALL SIX MASSES MOVE (3↓/3↑, .03-.37); ANCHOR FIRES THE FALLBACK — RESULTS ARE A NEW-INSTRUMENT RE-SURVEY

- Strict Track-A masses (sol+luna judges, anchors 6/6): peer_curation .778→.589 ·
  peer_revealed .567→.600 · **hashtagwars .733→.367** · press .589→.622 ·
  **mathse_vote .283→.475** · **mathse_accepted .325→.558**. Direction 3 down / 3 up
  — the unpredictable-direction framing is confirmed empirically. Mechanism: strict
  S_obs = N − edges in 7/7 (union-find barely chains); deltas mostly measure how many
  τ clusters were WITHIN-proposer merges the strict rule refuses.
- **ANCHOR VERDICT: the pre-registered fallback branch fires.** Sonnet's AoPS .5583
  sits inside the sol+luna LOO band, but the band swallows the effect: the judge-
  family delta (−.0667) is NOT small vs the correction (+.0583) — the families
  disagree on the AoPS correction's SIGN (concordance 95.3% historic pair vs 86.7%
  new; luna liberal at 45.3% SAME). **Quoting rule: these strict masses are a
  SEPARATE INSTRUMENT (sol+luna re-survey), never deltas against Sonnet-era figures;
  each certA_strict.json carries cross_instrument_calibration; τ-era A-mass quotes
  now carry "sol+luna strict re-survey reads X" companions.** Partial stabiliser
  noted: the both-SAME AND rule holds edge counts 41-49 across all six judge
  pairings despite single-judge SAME rates 32-45%.
- Scope: cw_community ALREADY strict (blind full-recall partition — no adapter run,
  no duplicate estimate); nc_responded already strict (confirmed); peer_verdict
  A-POOL UNRECOVERABLE (no proposer field → f₁/LOO undefined; stays τ-era with
  caveat, not re-mined). maps_batch1 dupes byte-identical; _ext dirs = record.
- Code: cmd_apply writes NEW _species_strictA.json (in-place only via --inplace);
  LOO jackknife added; runner reconstructed from on-disk bmerge prompts
  (run_bmerge_judges.py); all copies synced, divergences recorded.
  notes/2026-08-11__certA_backfill.md.
- NEXT (queued): peer_verdict retroactive Track-B round (GLM payload probe first);
  then certB strict re-survey for the τ-era-Z cells (Z consumes B-masses — the
  F2 verdicts hang on those, not on certA).

## 2026-08-11 — Z PASS RE-RUN AFTER RESOLVER FIX: PROVENANCE CORRECTED, ONE VALUE MOVED (cw .15→.20), ZERO VERDICT FLIPS

- Actual failure mode narrower than diagnosed: resolver keyed on b_merge.strict
  (mathse pair carries b_merge without the inner flag). Rule of record: presence of
  a top-level b_merge block IS the certificate; nc_responded's species_b two-judge
  merge and cw's blind partition added as recognized shapes.
- The mislabelled cells' M̂ VALUES were already post-merge — the mislabel was
  provenance, not magnitude: **relabeling changed one number (cw Z .5744→.5808,
  still ROBUST; from reading the terminal blind partition instead of a stale
  missing_mass.json) and zero verdicts.** Expectation-setting for the certB
  backfill: where species files already carry post-merge good_turing, strict
  certificates CONFIRM numbers rather than move them.
- Z_judge_family recorded in every Z; judge_labelled=false on three campaigns whose
  merge artifacts genuinely contain no model name (nc_responded, cw,
  mathse_accepted) — a record gap, not a parsing miss; cross-cell Z comparisons
  carry the caveat. f2_rez.py = seconds-fast Z/verdict refresh (X/Y/RR never re-run)
  for the remaining backfills. Commit 90fd3f614.

## 2026-08-11 — HASHTAGWARS DEEP AUDIT: Δ_beyond RULED NULL; RESIDUAL WAS INSTRUMENT DEGRADATION; SEVERE-BUT-TEXT-INVISIBLE METADATA LEAK

- **The retrieval batch is a severe metadata leak AND irrelevant to the residual:**
  Snowflake IDs are time-ordered; 88/101 contests have fully DISJOINT positive/
  negative ID ranges; a LABEL-FREE largest-ID-gap split recovers y at **.974**
  (two API sweeps: winners pulled first, filler later). But it is INVISIBLE to
  text: grouped-OOF tf-idf text→sweep .4787 (chance); all 44 channels ρ ≤ .063 with
  posting time within-class. **r1:B04's "batch visible in text" reading RETRACTED.**
  Method ruling: at 97.4% collinearity with y the sweep CANNOT be a discount channel
  — stratified/matched Δ_adj is undefined; refusing to produce that number is the
  finding; within-class time tests are the valid substitute.
- **The residual's actual source: the decomposition pass is the campaign's ONLY
  judging batch to FAIL the scrambled-anchor gate (.5876, salad > real negatives)**
  — a degraded A block depresses VA_nl and inflates Δ_beyond, which is why
  REWORDING nine rubrics beat 54 mined criteria 5:1 (84% of closure verified).
  Instrument degradation, not taste.
- **STRICT-LIST ROW (adopted verbatim): Δ_beyond NULL, not quotable as taste** —
  campaign +.0286 has jackknife SE .0607 (t=.47); F2 matched −.0230 [−.063,+.029];
  the +.056 "level gap" was never significant (t=1.14). E-refit arms =
  STARVATION FRAME ONLY, incl. "nuisance .6267 > bank .5357" — a 44-vs-158-column
  refit diagnostic, NOT a fact about nuisance → **the F2 cross-cutting
  "NUIS>bank on 6/11" is corrected to 5/11 + one starvation artifact.**
- Cross-fitted rebuild REJECTED with arithmetic (projected SE .0271, t≈1.05 — five
  trainings for a still-null): the decisive cheap job = RESCORE the 9 decomposition
  components under a passing anchor battery (ordered). Curation-shaped confirmed:
  fixed quota 9.92±.28 positives/contest over pools 12-181 — level-shift regime,
  not commensurable with independent-judgment cells.
- Infra flag from the audit: sk3 login shows the AFS "Could not chdir" root-disk-full
  signature — checked this turn.

## 2026-08-12 — HOMEPAGE CURATION COMPLETE: ARTICULABLE (Δ_beyond +.0068/+.0109 same-rows); JOURNALISM COLUMN DONE; MATHLIB 3-SEED DATA FOUND COMPLETE ON DISK

- **Homepage v2 ledger (n=12,998 snapshot-grouped):** V_nl .6432 · A_lin .6623 ·
  VA_nl .7291 (spread .0026) · Δ_interact +.0589 · **same-rows Δ_beyond +.0068 eval /
  +.0109 eval+test vs T .7109/.7251 — residual essentially CLOSED; largely
  articulable.** Bank validity: coherence .9900, 0/29 below chance, NA exactly 0
  over 376,942 cells, anchors 6/6 attempt-0, all alignment gates 0.00e+00.
- **Same-rows discipline was load-bearing AGAIN:** pooled VA_nl .7291 > pooled T →
  Δ −.0182 "bank beats dense" WOULD HAVE TRIPPED the auto-audit falsely — the press
  population-mismatch failure pattern, prevented by design this time.
- **Two new substantive findings:** (1) top-half placement goes to the LESS
  distinctive headline — the page-relative criteria are the bank's two most
  NEGATIVE columns (b26 .4343, b27 .4322); the top story is what the rest of the
  page is also about (running-story repetition; corroborated by b09 .5538 positive).
  The census bank could never have found this (no page-relative family). (2) No
  single criterion exceeds .566 — A_lin .662 is genuinely multivariate (the census
  bank's signal sat in 2 salad-blind columns).
- Genre channel GONE by construction (mask has zero variance — A_mask undefined;
  judged levels carry the whole +.0816, vs press's +.0014 over its mask). Stratified:
  ~95% of A's edge over V is genuine WITHIN-story-type ranking (A_lin .6623 → .6312
  stratified; sport the one failing stratum .487). T₀ .4902/.4925 = chance
  (T−T₀ +.2279 — the ceiling is entirely learned).
- Artifacts: homepage_curation_storygrouped_ledger.json, samerows_T (filled), _t0,
  rubrics_v2, batteries; notes/2026-08-09__homepage_curation_completion.md
  (incl. corrected legacy triage superseding the preliminary a13 claim). Cell →
  discovery queue. **JOURNALISM: press terminal · homepage complete · BBC built ·
  tweets built (appendix) — the column the user prioritized is DONE at
  instrument level.**
- **MATHLIB: all 3 seeds BOTH arms already trained+scored on disk** (bigtrain eval
  .682/.600/.629, test .535/.571/.544; regime eval .621/.550/.551, test
  .523/.490/.552) — no GPU needed; the split-divergence replicates across 6
  trainings (eval≫test both regimes; test means ~.52-.55, seed spread ~.04).
  Resolution write-up queued to the corrections agent (CPU).

## 2026-08-12 — MATHLIB RESOLVED: NO HONEST DENSE T (select-on-eval inflation proven); HW FINAL: GATE "FAILURE" WAS A COMPOSITION ARTIFACT — NEW ANCHOR-GATE RULE

- **MATHLIB row adopted:** dense fails on honest rows across 3 regimes / 9 seeds
  (test .467 canonical / .522 regime / .550 train-big; CategoryTheory .517 chance).
  NOT starvation (7.5× negatives lifted eval .564→.637, test flat); NOT a harder
  test half — **TF-IDF (.6796→.7883) and size-alone (.6024→.6909) both score HIGHER
  on test**: the dense-only eval≫test gap is checkpoint-selection optimism amplified
  by instability (seed-to-seed prediction ρ .35-.67 — three seeds aren't estimating
  one function). **Quote no T; Δ_beyond not computable; the cell's readable ceiling
  is LEXICAL (TF-IDF .788).** Bank VA_nl .672 is a reference (different split
  design), never differenced. Third neural-reader-fails cell class (with RoyalRoad
  power-cap — but distinct shape: here the articulated+lexical sides read well).
- **HW FINAL — the audit corrected itself, on the record:** the "instrument
  degradation" reading is RETRACTED. The decomposition batch's .5876 gate was a
  COMPOSITION ARTIFACT: the A-routed 9 components pass at **.9897** while the
  extent-of-surface channels invert (.0000) — CORRECTLY, because scrambling does
  not destroy an extent (2× length, kept capitals/hashtags score high, as they
  should). Δ curve reproduces byte-exactly (diffs 0.0000); the NULL verdict stands
  on statistics alone (t=.47; matched −.0230). The 84%-from-rewording finding is
  RE-FOUNDED on craft-surface entanglement inside frozen rubrics (the Addendum-3
  MIXED mechanism with a positive closure outcome), not judge quality. No GPU used;
  reuse-before-rebuild prevented an unnecessary rescore.
- **NEW STANDING RULE (adopt everywhere): the scrambled-anchor gate is computed on
  the subset whose value scrambling DESTROYS (in practice the A-routed subset),
  never pooled over batches containing extent-of-surface channels** — a pooled gate
  on a mixed batch measures composition, not reading. Record both; the subset
  figure gates. Retroactively clears the campaign's only "gate failure"
  (→ zero-with-composition-note in any cross-cutting count).

## 2026-08-11 — CW expansion inventory (dataset-holes mandate) + BBC closure round 0
- **CW RoyalRoad — EXPANDABLE, but only via re-matched build.** Raw crawl `royalroad_stubs_v1.jsonl` = 2,367 usable fictions (100% wayback, balance 1,184/1,183); current n=1,274 leaves 1,093 unused. The 1,274 is a deliberate topic×era match, and the matching is load-bearing: full-pool unmatched lexical floor .606 (≥.6 = DIRTY under §17m), era-match-only .600 (boundary); current build .524 CLEAN. Topic content, not capture era, is what leaks at larger n. **DECISION (coordinator, under user's "fill the dataset holes" directive): option (b) approved** — bge-large cluster all 2,367, find the largest topic×era-matched subsample holding lexical <.58, rescore bank on new rows only, retrain dense at new n. RoyalRoad 5-fold cross-fit (honest n=651, 4.6× old eval) continues in parallel as the fallback ceiling.
- **CW Wigleaf — DATASET EXHAUSTED, accept n=1,568/404.** Year coverage complete (2008–2025, all 18 years). 98.5% of wayback-recoverable positives and 97.9% of recoverable negatives already used; the larger top50_texts.jsonl is mixed-provenance (500 live) and would reinstate the fetch_source .90 leak the current build killed (.500). Growth requires new crawling at uncertain yield. Power caveat stands; pairwise frontier probe (270 units × 45 criteria, anchors 45/45 on smoke) is the instrument route.
- **BBC most-read closure ROUND 0 — GATE PASS (+.0749), rounds run.** VA_nl .7482 / T .8231 on MONITOR (2,060 rows, 88 days, taken inside dense-held-out). Order-join proven element-wise with shuffled counterfactuals (.5038/.4956). ε-resolvable depth 0 (paired seed SD .00252 < ε .005) with caveat: zero-change 95% width ≈ 2ε, one null pair reads +.0034 — no single round's near-ε gain quotable; two-consecutive-rounds rule governs saturation. Swap baseline C₊ .8227 / C₋ .4079. **FREEZE ADDENDUM 4: page-position channel structurally unavailable** — scraper builds negatives excluding most-read hrefs (0/33,400 overlap), so position is perfectly confounded with y; the "lists reflect placement" worry needs a raw-capture re-parse, recorded not mined. Fleet degraded to P=8/2 families (Claude cap 500/500) — freeze floor, recorded per round; GPU: one card, yielding to CW priority. Artifacts: closure/bbc_mostread/, notes/2026-08-12__closure_bbc_mostread.md, commits 9d33589/8c82a8d.

## 2026-08-11 — homepage curation closure ROUND 0 = TERMINAL (noise-floor) + BBC r1 hold
- **homepage curation: TERMINAL AT ROUND 0.** Δ₀ +.0035 vs .02 gate (VA_nl .7288 / T .7323 on MONITOR n=622/72 snapshots). Decisive reason is NOT the gate: ε-resolvability FAILS (paired seed SD .00882 = 1.8× ε; dense seed spread on MONITOR .0303 = 9× Δ₀; T−VA_nl CI [−.0429,+.0642] spans zero). Verdict wording = press_verdict exactly: **residual closed at this cell's resolution; stopping rule NOT fired.** NEVER quote as "articulation saturates on homepage" — only "no residual measurable here"; MONITOR width binds, cross-fitting doesn't rescue. Pre-round-0 resolvability check fired before any fleet/judge/GPU spend (same code passed BBC at .00252 same day). sklearn drift (1.8.0 vs ledger 1.9.0) + item-view asymmetry (V headline-half vs dense full-text, max 720 tok, no truncation → SO displacement mode impossible) recorded. notes/2026-08-12__closure_homepage_curation.md, commits b7572c6/da33791.
- **Journalism column shape: residual concentrated in ONE cell** — BBC most-read (Δ₀ +.0749, resolvable) vs homepage noise-floor terminal vs tweets appendix-queued. BBC r1 fleet HELD at 1 family (GLM 1302 on both keys, verbatim responses + request IDs in round log); proposer framing carries the both-directions augmentation as a labelled addition, prereg instruction intact. Tweets r0 (fleet-free) next in lane.
- **V_new pilot certification (local box, NOT sk3 — CPU pipeline runs on laptop):** jokes 4/18 compiled certified (top ρ_monitor .647/.610/.554/.392 — r1:A03, r2:A07, r5:A02, r3:A03), press 13/33 (top .598); nc_responded still compiling. Refit chain live (PID 40878 local). Cert gate = ρ_monitor ≥ .30 AND modal ≤ .98, declared pre-compile; compiler gpt-5.6-luna recorded per row.

## 2026-08-11 — journalism tweets (V9) closure r0 = TERMINAL (real residual, untrackable) — column round-0 sweep COMPLETE
- Anchor (MONITOR 1,258 rows / 32 groups): VA_nl .6168 / T .6575 → Δ₀ +.0407 — gate PASSES (2× threshold). Residual resolvability HOLDS: T−VA_nl +.0445, CI [+.0133,+.0791], P(>0)=.9975 — a REAL residual. But ε-resolvability FAILS (paired SD .00804 = 1.6× ε): a round's sub-ε gain is indistinguishable from noise, so the stopping rule would fire at random. **TERMINAL AT ROUND 0, resolution-bound — verdict wording: "a real residual of ~+.04 that this closure design cannot resolve round-over-round." OPPOSITE failure from homepage** (homepage: no residual measurable, CI spans zero; tweets: residual exists, progress untrackable). Never collapse the two; neither is "saturated."
- Binding constraint = 32 MONITOR groups (group-resampled CIs), not row count; cross-fitting shrinks fit-seed noise only. MONITOR enlargement (~124 held-out groups exist) = split redesign requiring a prereg amendment — correctly NOT made post-hoc; PARKED. Coordinator recommendation: leave parked — tweets is an APPENDIX cell (user ruling), and the real-but-untrackable verdict is itself the appendix row.
- Methodological result of the lane: the pre-round-0 ε-resolvability check discriminated three Layer-1-similar cells into three different situations (BBC resolvable +.0749 / homepage no-residual / tweets real-residual-untrackable) before fleet/judge/GPU spend on two of them.
- Lane hand-off written: notes/2026-08-12__closure_journalism_tweets.md (RESUMPTION STATE for all 3 cells; BBC r1 unblock = fresh Claude legs OR GLM clearing 1302; slice deterministic on disk, no proposer sees it twice). Swap C₊ .7425/C₋ .3770; no sklearn drift; item view byte-identical. Commits 8f30345/3eb5608.

## 2026-08-11 — CW triple landing: RoyalRoad cross-fit + Wigleaf pairwise probe + expansion checkpoint
- **RoyalRoad cross-fit (honest n=651/308 pos, 4.6× old eval): T-at-chance CONFIRMED, not a power artifact.** T_pooled .4981 [.4532,.5399], fold-mean .4986, rank-pooled .4994 (estimator spread .0013); prior single-split .4994 reproduced. Same-rows bank legs: V_lin .5764 / A_lin .5875 / VA_lin .5946 / VA_nl .5718 → **Δ_beyond −.0732: bank beats dense by ~.09 on identical rows — FLAGGED not asserted** (standing upper-bound-fails rule). Two live mechanisms before anyone quotes it: (i) view asymmetry — dense reads first 1,024 tokens (~35% of median 2,918-token chapters, never the ending); A judge reads head 960 + tail 640 and SEES THE ENDING; (ii) 8B LoRA at ~1k rows = weak ceiling. **AUDIT ORDERED: head+tail-view dense arm on the same folds** (the SO-qtrunc pattern repeats — second cell where view asymmetry is the prime suspect for bank>dense).
- **Wigleaf pairwise probe: COMPARATIVE FRAME WORKS.** 270/270; anchors pick-real .9963; tie rate 2.8% vs 83% of absolute responses pinned at 1.0 — the forced-choice frame restores the headroom the 3-point scale lost. Composite (majority of 45) AUC .610 [.5425,.6775] = holistic .610; same-MAGAZINE stratum strongest (.6231) so venue isn't driving it; 23/45 criteria CI-exclude .50. Substantive: cut rewards Ending resonance .6325 / Distinctive voice .6175 / Ending earned .6150; BELOW chance on Prose economy .460 / Causal progression .4575 — longlist pieces are the tidier ones (cap_finalist-family echo: generic tidiness saturates, the cut wants voice+endings). Estimand caveat binding: matched-pair forced choice ≠ unpaired grouped-OOF — licenses "comparative judging separates the cut," never a .610−.5407 delta.
- **RoyalRoad expansion checkpoint: n=1,742 at lexical .5759** (register .5382, era-y corr .000) — clears the <.58 margin and 1,450 floor; proceeded per pre-authorization. 719 new rows / 1,023 carried; **NOT a superset — 251 old rows fall out at k=24 rematch** (logged; old-build numbers stay quotable as the n=1,274 design). Chain on GPU0: bank rescore new-rows-only (32,355 prompts, K=50) → dense 3 seeds at n=1,742 → Layer-1.
- **Wigleaf P1 (Bradley-Terry θ) REJECTED — correctly**: at ~1 comparison/item θ is unidentifiable and label-circular. **P2 pairwise-native Layer-1 APPROVED with 600-pair expansion** (unit=pair, A_pair {−1,0,+1}, V_pair=ΔV testing the length channel, T_pair=existing dense score diff no retrain, group=pair OOF, antisymmetry enforced both orientations).

## 2026-08-11 — Style Invitational pairwise probe VERDICT + caption closeout
- **SI: frontier judge separates, v2 criteria do not.** Holistic pairwise ("which did the editor pick?") **.810** [.750,.858] on 200 matched pairs (2,211 comparisons, 0 unanswered) vs best v2 criterion .575 (−.003 vs pick-the-longer); other 7 criteria .475–.565. Same winner-vs-HM contrast reads **.483 under absolute Gemma-4-31B scoring** → BOTH diagnoses partly right: mode mattered (comparative rescues the construct — the editor's cut IS text-recoverable; v2 cert failure was never evidence of noise) and content mattered more (pairwise does NOT rescue the criteria — 0/8 clear; the .575→.810 gap = unnamed criteria mass). Controls: judge BETTER when winner is shorter (.840 vs .786 — anti-length); side-A .531, swap consistency .800, pooled .810 is the conservative order. **Scrambled anchor RETIRED as invalid-by-construction on this corpus** (4–10-word entries + rearrangement contests → scrambles stay prompt-appropriate; read .067) — generalizable rule: scramble anchors need items long enough that shuffling destroys sense; gate rests on ANCHOR_FRAGMENT .800 + position + length split. Caveats: single family (gpt-5.6-sol); **.810 NEVER quotable vs VA_nl .6011 / T .6241 without §6.5 commensurability run**. Plan approved: Phase 1 = Bradley-Terry holistic θ on sparse within-week graph (degree 5, per-week intercepts, side term, no scramble anchors) + §6.5; Phase 2 = mine Track-A criteria against the PAIRWISE residual (absolute-residual mining is what built the failed v2 bank). notes/2026-08-11__si_pairwise_probe.md.
- **cap_crowd closed per retirement** (r4 readout +.0077/+.0053 recorded for completeness; NO terminal ledger, none quotable). Keeper finding — **cross-cell criterion transplant: curation/community contrast is a LEVEL SHIFT not a sign inversion** (all 12 "sign-contradicting" criteria cross above .5 on cap_crowd; rank corr of criterion importance +.411 p=.041) — editors and crowds agree on which craft properties matter, they differ in threshold; corrects cap_finalist §10.7's "comic-craft vocabulary runs backwards."

## 2026-08-11 — V_new COMPILATION PILOT VERDICT (jokes / press / nc_responded)
- **Certified 28/430 terminal criteria (6.5%)** (triage 80/430; compiler gpt-5.6-luna recorded per artifact; cert = held-out ρ≥.30 + modal≤.98, declared pre-compile). Codable families: phonetic/lexical/structural/register; uncodable: semantic (0/194 nc), relational, affective.
- **Priority target FAILS: "Read-aloud cadence"** (judged .687 alone) compiles at ρ=.098 / alone .548 — the judged criterion is NOT mechanical prosody; jokes' prosody island codable only in the weak sense.
- **Separations (frozen fit_block, MONITOR, group bootstrap):** V_new>V only on nc +.0389 [+.0040,+.0772] (jokes +.0067 [+.0017,+.0152] tiny; press noise). (V_new+A)>VA only on nc **+.0212 [+.0016,+.0421]**; ≈0 jokes; noise press. Expected mechanism holds for jokes/press: compiled columns are translations of criteria the judge already scores → inert on top of A.
- **LENGTH FLAG (binding gate):** top certified columns heavily length-correlated (nc ρ .53–.72; press .56–.66) — the nc gain may be re-packaged verbosity. **Layer-2 length-stratified readout on the V_new stack = named PRE-REGISTRY GATE; the +.0212 is NOT quotable until it passes.**
- **Scale-up ruling: SELECTIVE** — cheap CPU V-enrichment for structural/register-heavy banks with thin V (nc pattern), gated on the length screen; NOT a general stage (semantic banks don't compile; strong mined criteria are judge-bound). Infra fix: parse-retry for codex modules (2/15 lost to syntax).
- Artifacts: notes/2026-08-11__vnew_compilation_pilot.md; methods/taste_decomposition/vnew_pilot/ (increments_boot.json etc.). Also closed by same agent: SO-votes audit (MIXED: articulable stands; bank-beats-dense = view artifact) + decorrelated-training battery (V2' FAIL utility / reliance-removal PASS).

## 2026-08-11 — CW rulings: rr_v2 anchor FAIL + Wigleaf pairwise length flag + Stage A built
- **rr_v2_k24 expansion: A-bank anchor battery FAILS on the expanded pool** (pos-vs-neg ordering AUC .445 vs original .658; scrambled still .0000 / coherence .995 — judge fine, pool inseparable). No batch effect (new-rows mean .7270 ≈ original .7232). Read with the lexical floor rise .524→.5759: expansion buys +468 rows and LOSES the bank's certification. **RULING: rr_v1 (n=1,274) stays the cell of record; rr_v2_k24 = labelled sensitivity design, A-numbers flagged Wigleaf-style.** Dense rr_v2 ledger completes anyway (seed2 running).
- **Wigleaf P2 wave-1: THE LENGTH TEST FIRES.** length-only V_pair (log-size) .6730 > V_lin .6636 > VA_nl .6523 > A_lin .6430 > composite .610. A two-feature size difference beats the 45-criterion bank → honest headline: "the pairwise signal is substantially SIZE" (Top-50 vs longlist differ systematically in length; venue has a length remit — may be editorial scope not craft). Judge verdicts still carry real non-trivial signal (A_lin .643, gates passed) but every Wigleaf pairwise number now travels with the length caveat. VA_lin<V_lin at n=200 = overfit signal; n=600 wave 40/484 running.
- **T_pair not computable as specced** (existing dense OOF covers only 322 rows → ~25/600 pairs). **APPROVED: cross-fitted Wigleaf dense arm** (RoyalRoad template) — also fixes the item-level T flaw (170-row eval WAS the selection split; same select-on-eval defect the RoyalRoad cross-fit corrected).
- **cw_transfer_v1 Stage A built + guarded**: pool 146,390 dedup (WP 96,080 + LitBench 87,654 chosen/rejected), pilot 24,000/13,457 prompt groups, stable-hash group split; leakage guard 0 collisions vs all 3,560 RR+Wigleaf texts; --init_adapter opt-in (frozen recipe byte-identical when unset). Queue: rr_v2 dense → head+tail audit (1600-token view, second-difference declared; head614+tail410@1024 decomposition arm only if it moves) → Stage A.

## 2026-08-11 — SI Phase 1 checkpoint: cross-family control REWRITES the headline
- **The .810 claim is capability-graded, not construct-general.** GLM-5.2 on the same 30 matched pairs (byte-identical prompts/order): **.533 = chance** [.361,.697], while passing fragment anchors 6/6 (it reads the entries fine); sol reads .933 on the same slice; per-pair agreement .533. Honest claim: "the editor's cut is recoverable by gpt-5.6-sol at .810; glm-5.2 cannot do it at all." Reframes Gemma's .483 as a capability reading, not construct evidence. Any SI instrument inherits a hard single-model dependency; third family (Claude judge) = next control when subagent cap resets. Open question flagged for OSL line: does separation rise monotonically with judge capability?
- **BT instrument running**: eval∪test scope (1,607 items / 80 weeks, commensurable-by-construction with same-rows VA_nl .6165/.6042 and T .6241/.6237), circulant graph mean degree 5.44 all-weeks-connected, 4,406 comparisons + 40 anchors + 440 swaps; ~9% done, latency-bound (~560s/call at effort=high, kept for comparability with the .810 wave); ~3–5 machine-hours total, full 8,063-row instrument would be ~5× (price into Phase 2). θ label-free vs label-fitted VA_nl/T — asymmetry runs AGAINST θ, travels with any comparison. **.810 stays QUARANTINED until §6.5 exists.** Resume: datasets/humor/style_invitational/pairwise/phase1/RESUME.md; runner PID 86711 detached; watcher armed.
- **Discipline lapse recorded (agent self-report): one `pkill -f "codex exec"`** — all matches were its own runner's children; second round done PID-by-PID; standing kill-by-PID rule reaffirmed. Also §6.2 cost under-estimate corrected (~800 calls → ~3–5 machine-hours at this scope).

## 2026-08-11 — CW: rr_v2 ledger final + Wigleaf n=600 REVERSAL (size-matching kills the craft bank)
- **rr_v2_k24 final ledger** (sensitivity design, per ruling): n=1,742, battery FAIL .445, A_lin .5627, VA_nl .5505, T .5112, Δ −.0393. Two keepers: (1) **two-batch judging seam is CLEAN** (carried .7262 vs new .7270; batch membership predicts y at .5098) — the merge technique is reusable; (2) **n ruled out TWICE as RoyalRoad's T constraint** (1,274→1,742 moves T .4986→.5112, still chance; cross-fit n=651 agreed) — head+tail view is the last standing hypothesis; fold 3/5 running = decisive test.
- **Wigleaf n=600 (wave 2 484/484, gates hold: anchors .9982, ties 2.6%): the pilot reading DIES.** Composite regresses .610→.5517 [.5117,.5917]; same-year stratum →.5023. Size-matched stratification (300/300): **A_lin = .4986 — CHANCE — among size-matched pairs** (vs .6300 size-divergent); all of the craft bank's pairwise signal rides the size channel. Refinement that keeps it from pure scope: V_lin holds .6140 size-matched while length-only drops to .5361 → real non-size SURFACE-STYLE signal exists (TTR, sentence-length variability, dialogue fraction, readability). **Claim of record: "Wigleaf's editor's cut is predictable from scope and surface style, and not at all from the articulated craft bank once size is held constant."** The pilot's "endings and voice" reading (.6325/.6175) is RETIRED — did not survive 3× data; never quote it.
- Lane: head+tail audit fold 3/5 (GPU0) → Wigleaf cross-fit (honest n=747 built, 4.4× old eval; T_pair auto-fills) → Stage A. 

## 2026-08-12 — SI Bradley-Terry θ LANDED: §6.5 exists; label-free judge instrument beats bank AND dense by ~.2
- Fit: 4,846/4,886 comparisons used (2 pair-slots unanswered), converged, side term γ=.0057 absorbed, λ-insensitive (.3/1/3 within .003). Anchors .95 (n=40, fragment-only); swap consistency .670 (n=440) — moderate per-comparison noise, aggregated over mean degree 5.4.
- **§6.5 commensurable table (same rows as the cell's frozen ledger):** θ AUC pooled **.7933 eval / .8251 test** (within-week .7988/.8358) vs VA_nl .6165/.6042 and T .6241/.6237 → **θ − T ≈ +.17/+.20, θ − bank ≈ +.18/+.22.** θ is LABEL-FREE (judge comparisons only) while VA_nl is label-fitted and T label-trained — the asymmetry runs AGAINST θ, so this understates the result. The .810 probe number is now contextualized (de-quarantined WITH the §6.5 frame + capability-graded caveat: gpt-5.6-sol only; GLM chance; single-family dependency stands).
- **Reading: SI's articulation headroom is ~.22 at commensurable scale — the largest instrument-revealed residual in the humor column** — and the cell's dense T (.624) badly under-reads the construct ceiling (θ .83 on test). "The taste is in the judge, not the labels": a frontier model can SEE the editor's cut; neither the articulated bank nor an 8B trained dense can.
- **Phase 2 GO-condition MET** (registered trigger: θ-vs-bank gap replicates at commensurable scale) → Phase 2 = mining Track-A criteria against the PAIRWISE residual becomes the humor-curation closure campaign. QUEUED, blocked on the same 2-family fleet floor as BBC r1 (Claude cap 500/500 + GLM 1302); cost note: sol-judged waves ≈ 3–5 machine-hr each. SI agent out of context — fresh agent takes RESUME.md + this entry when fleet unblocks.

## 2026-08-12 — USER RULING: SI θ framing tightened
- θ is a **text-recoverability probe**, not a model: in-sample instrument reading (label-free judge choices + BT aggregation over ~5 comparisons/item), no generalization claim, NEVER a ladder bar or T substitute. θ−T bundles capability + elicitation frame + calls-per-item → enters no Δ_beyond, is not "taste." Licensed claim only: frontier judge recovers the editor's cut at ~.79–.83 on rows where bank reads ~.61 and trained 8B ~.62 (both θ and T = lower bounds on M* via different channels). Prior registry entry's "beats bank AND dense" phrasing deprecated; strict-list row rewritten accordingly.

## 2026-08-12 — CW RESOLUTION LANDING: view asymmetry CONFIRMED on RoyalRoad; Wigleaf honest T; T_pair complete
- **RoyalRoad bank>dense flag RESOLVED — it was the VIEW.** Head+tail arm (head 960 + tail 640, max_len 1600; same folds/seed/rows as cross-fit): T .4986 → **.5846 (+.0860, every fold up)**. Δ_beyond −.0732 → **+.0128: bank ≈ dense at 8B once the arm sees the ending.** Second cell running the SO-qtrunc pattern (view asymmetry manufactures bank-beats-dense). Caveat: arm bundles view AND budget (1024→1600); **decomposition arm APPROVED (head 614 + tail 410 @ 1024)** to separate them. Standard-1024 T .4986 retired as the cell's reference once decomposition confirms; until then judge-view T .5846 quoted with the two-change caveat.
- **Wigleaf honest T = .5589** (cross-fit n=747, selection-free; pooled .5637). The old **.6054 was select-on-eval optimistic by ~.046** (170-row eval WAS the selection split) — NEVER quote .6054 again. Same defect now corrected on both CW cells.
- **Wigleaf P2 complete: T_pair .5618** (134/600 both-OOS pairs, antisymmetric), **Δ_beyond_pair −.1004** — articulated stack exceeds the pairwise dense ceiling by .10, BUT the margin is carried by V (surface/scope), NOT the craft bank (chance .4986 size-matched). Consistent story across frames: Wigleaf = scope + surface style.
- **Stage A trained**; defect flagged: score_eval_dense_v4.py globs rm_out_seed* but Stage A wrote rm_out_pilot → eval_pass_results.json empty. FIX APPROVED = rename/symlink the adapter dir to match the glob (no shared-scorer edit while V3 is using it). Adapter intact at stageA/rm_out_pilot/best_model.
- V3 v3_aug arm a fold 0 running on GPU0, head+tail base view, VA block FIRST (chapters can't co-fit; documented deviation), fold-varying train-only importance verified. Target: exceed bank alone (.5946 lin/.5718 nl) vs judge-view T .5846.

## 2026-08-12 (overnight) — N&C agree + outcome closure: TERMINAL BY THE FROZEN STOPPING RULE (no r3)
Coordinator-executed (subagent cap spent; V_new agent handed off with clean state). Inspection of the existing maps_batch1 campaign artifacts shows the queued "r3→" would VIOLATE the frozen stopping rule — it was queued before the r1/r2 readouts were consolidated:
- **nc_outcome TERMINAL**: Δ₀ NEGATIVE at r0 (Δ_beyond −.0085 HONEST n=1,417 / −.0154 MONITOR n=694 — bank ≥ dense, gate FAIL); Track-A trajectory VA_nl_MONITOR .6311 (r0) → .6359 (r1, +.0048 sub-ε) → .6246 (r2, negative) = **2 consecutive sub-ε PROPOSING rounds → rule fired at r2**. Retroactive ε-resolvability: seed spread .0199 ≈ 4×ε — the closure curve was never readable at this MONITOR size (homepage/tweets-class finding). Best mined criterion: "Firsthand Operational Knowledge" alone .616 HONEST.
- **nc_agree TERMINAL**: Δ₀ −.0224 HONEST n=1,009 / −.0163 MONITOR n=487 (gate FAIL); VA_nl_MONITOR .6172 (r0) → .6062 (r1) → .6059 (r2) = 2 consecutive sub-ε → rule fired. Seed spread .0179 ≈ 3.6×ε. Standing caveats travel: (1) divergent y — T quoted as pooled .6034 + eval .566 + test .639, never one number; (2) docket-identity confound — within-docket VA_nl = .4934 CHANCE, docket-id alone .8616 (the cell's edge is cross-docket composition).
- Both cells: verdict wording = "bank ≥ dense at round 0; no dense-over-bank residual to close; Track-A mining plateaued sub-ε in 2 rounds." NOT "saturated." "After closure" precondition for F2 now satisfied → F2 deconfounded rows (adapters being built tonight; T₀ columns already exist 16/16).

## 2026-08-12 (overnight) — INFRA: CODEX USAGE LIMIT EXHAUSTED (resets Aug 18) + GLM substitution on BBC b_merge
- codex CLI (gpt-5.6-sol AND -luna) hit the account usage limit mid-b_merge — "try again at Aug 18th 4:13 PM". Burned by tonight's heavy waves (SI BT 4,886 comparisons at effort=high, Wigleaf 684 pairwise, compile fleet, BBC fleet). **Every codex-routed job is dead until 2026-08-18** unless credits are added: SI Phase 2, code rounds judging, rescue reviews. Track A b_merge leg 1 (sol) completed BEFORE the wall: 122/122 verdicts, 69 SAME. luna A + sol B + luna B legs all rc=1 instantly (quota).
- **Substitution (upgrade, not degradation): GLM-5.2 as second judge** via run_bmerge_judges_glm.py — imports PREAMBLE/render/extract_json from the codex runner so the instrument is byte-identical, transport only differs. sol+glm = genuine CROSS-FAMILY two-judge merge vs the planned sol+luna hive-mind pairing (the runner's own recorded caveat) — independence claim strictly stronger on Track A. Track B has NO codex leg → GLM single-judge, and per the freeze every downstream Track-B number carries a SINGLE-JUDGE flag until a second family (fresh-session Claude leg, or codex after Aug 18) judges the same packet. Legs running.

## 2026-08-12 (overnight) — BBC r1 pipeline COMPLETE TO PRE-SCORING BOUNDARY (coordinator-executed)
- **Strict blind merge applied in place** (species.json carries both blind_merge certificates, all anchors pass): Track A 81→75 species via sol+GLM CROSS-FAMILY strict merge (45 edges), merged M̂ .550→.450, recapture .19→.28; Track B 55→45 via GLM single-judge (35 edges), M̂ .5625→.4125 — **B carries the SINGLE-JUDGE flag** until a second family re-judges the same packet (fresh-session Claude or codex post-Aug-18). PREMERGE τ-only tables preserved beside (good_turing_PREMERGE_tau_only).
- **Blind routing audit (corpus-matched BBC probe pairs authored: development-vs-numeral, curiosity-gap-vs-length, reader-consequence-vs-'you'-token, human-stake-vs-quotemarks): probes 4/4 PASS.** Auditor glm-5.2 (single-family, degradation recorded in verdicts file). Misrouting 5/25 (.20); arbiter (glm-5.2, provenance-sighted per protocol) ruled all 5 — notable rulings: B08 "everyday mystery/quirk" → A (the quirk IS the content); A13 UK-centric hook → A unmixed (domestic relevance is substantive for a BBC audience); B03/B06/B10 stay B mixed (production-format channels). **Final routing: A=16, B=9 (3 mixed).**
- Selected banks read construct-sensible: A = news-values criteria (explicit question, scene vividness, everyday relevance, emotional arousal, conversation potential, practical utility...); B = format channels (article-type label, scare-quoted hook, headline length, ordinal cues, listicle promise).
- **STOPPED at the frozen boundary: Gemma scoring of the 25 routed criteria needs a GPU** (CW owns GPU0 for V3 overnight) → morning item, then the round readout (gain vs ε with the two-consecutive-rounds rule).

## 2026-08-12 (overnight) — F2 deconfounded rows LAND for nc_outcome + nc_agree (N&C column COMPLETE: closure + F2 + T₀ on all three main cells)
- **nc_outcome: PRIMARY stacked increment (d)−(c) = +.0194 [+.0007,+.0375] P=.980 — small but significant deconfounded taste residual** (arms: bank_enr .6162 / NUIS .6096 / enr+nuis .6156 / +T .6358 / +T₀ .6187). SECONDARY (e)−(c) −.0024 ≈ 0 as predicted. Fused-vs-bank PASS (+.0197). Note the estimand lesson repeats: the closure LEVEL readout (bank ≥ dense) and the F2 INCREMENTAL readout (+.019 conditioned on everything nameable) are different questions — dense carries a small signal the enriched bank + 22 nuisance channels don't span, even though its level doesn't beat the bank.
- **nc_agree: PRIMARY −.0099 [−.0330,+.0150] P=.211 NULL** — no deconfounded residual. Two flags: (1) **NUIS ALONE (.6115) BEATS THE ENRICHED BANK (.5730)** — nuisance>bank now 6/13 cells; on this cell that's the docket-composition confound wearing a new coat. (2) **SECONDARY (e)−(c) = +.0208 P=.983 POSITIVE** — the F2 spec's registered anomaly flag ("a positive here flags nuisance-prior interaction"); with this cell's unstable y (eval .566/test .639) and docket confound, recorded as a flag, not interpreted.
- Env matched battery of record (mac, py 3.12.3, sklearn 1.7.2); adapters = generic contract on maps_batch1 artifacts; missing-STRUCT rule not triggered (no declared STRUCT on these cells). Results: results/f2_deconf_nc_{outcome,agree}.json.

## 2026-08-12 — MORNING RULINGS (user) + relaunch wave
- **USER RULINGS**: (1) N&C keeps verdict=responded + curation=outcome; **REMOVED from
  community** (co-signing not meaningful as community preference; V8 artifacts → appendix).
  (2) **RoyalRoad RETIRED** (CW verdict) and **Style Invitational RETIRED** (humor
  curation) "for now — exploring other options"; ledgers stay quotable as appendix.
  (3) HashtagWars retirement under consideration (still canonical humor verdict today).
- **RR V3 fulltext chain KILLED** at arm_a fold3/5 (wrapper 938783 then trainer 1722221,
  targeted PIDs; ledger RELEASE written; arm_a folds 0-2 artifacts kept under
  dense_crossfit_v3aug_fulltext/). Rationale: ~10h GPU remaining on a retired cell.
  Phase-2 watcher moot (RR_V3AUG_CHAIN_DONE will never fire).
- **BBC most-read r1 Gemma scoring LAUNCHED** sk3 GPU5 (pid 1745819 runner; engine
  loaded 172GB): 50,761 headlines x 25 routed criteria + K=50 anchor battery;
  bbc_mostread_population.csv generated from va/population.csv.gz (raw_headline view =
  bank view, persona matched to score_mostread_bank.py). Watcher armed (fire-on-done).
- **USER REQUESTS queued this wave**: (a) code PR-merge within-repo V-only leg (fills
  VERDICT V column); (b) patents honest-dense revival per RUNBOOK prereg conditions;
  (c) V3 fused arm on code competitions; (d) peer_revealed leak audit — author/
  institution identity channels (NOT in the 57 judged nuisance channels; F2
  nuisance_struct=0 for this cell) + pretraining-memorization probe.

## 2026-08-12 — morning wave LANDINGS
- **code_v3 (PR merge) V-ONLY WITHIN-REPO leg LANDED** (closure/code_v3/abank_rescore/
  v_only_within_repo.{py,json,log}; frozen Layer-1 HistGB seeds 0-2, GroupKFold(5) by
  repo, within-repo n-weighted >=20-row both-class repos): **V_nl .5525 eval / .5484
  test** (VA_nl same-subset .6517/.6150 — reproduces the recorded ~.63 frame). Pooled
  V_nl .675 eval vs .561 test = the known composition artifact, never quote. Reading:
  V features carry almost nothing within-repo; the bank does the work. V_new = V (no
  compiled columns for this cell). Notebook VERDICT row updated to V=.550.
- **peer_revealed IDENTITY LEAK AUDIT LANDED** (user-ordered; methods/taste_decomposition/
  peer_identity_audit/ + fusion/f2_identity_arm.py + results/f2_identity_peer_revealed.json).
  OpenAlex authorships fetched 4,663/4,663. Findings: (1) fold-overlap is real — 78.6% of
  held-out rows share >=1 author with train, 95.2% share an institution (GroupKFold by
  ntitle does not block identity); (2) author-identity-alone (train-encoder, leak-free
  construction) AUC **.671 held-out** / fame-logcites .674 — a REAL channel absent from
  the 57 judged nuisance columns; identity increment over bank+57 = **+.0236
  [+.0008,+.0473] P=.979 SIGNIFICANT**; (3) BUT the dense residual SURVIVES identity
  conditioning: **(d')-(c') = +.0744 [+.0395,+.1077] P=1.000** vs reference +.0927 —
  identity absorbs ~20%, not the residual. Institution channel weak (.550). Identity
  covariates rank BELOW the strongest found nuisance channel (Y=.716), so the E-value
  ROBUST verdict is unchanged. Remaining unaudited channel: pretraining memorization
  (T0-null +/-.0006 is indirect evidence against; direct probe = open item).
- **code_competitions V3 fused arm chain LAUNCHED** (user request): builder
  methods/taste_decomposition/code_competitions/build_code_competitions_v3aug.py
  (same-rows n=999, StratifiedGroupKFold(5,shuffle,rs=0) by canonical_pid, per-fold
  train-only permutation importance, VA-block-first prompt, arm a, max_len 4096);
  chain scripts/tools/code_v3aug_chain.sh on sk3 **GPU1** (GPU0 was grabbed by a
  co-tenant between kill and launch). LANDMINE hit + fixed: train_reward_model's
  80/10/10 split-ratio guard rejects 5-fold layouts (68/12/20) -> DENSE_SPLIT_FRACTION_ATOL=0.15.
  Estimand: max-of-variants VAT column only.
- **patents revival SCOPED (G2)**: RUNBOOK prereg conditions = (1) restrict label to
  §102/§103, (2) rebuild candidate sets symmetrically w/ randomized slots, (3) real
  multi-criterion A bank from online-rubrics, (4) claim-ordinal declared Track-B
  nuisance; quote Δ over V+A+STRUCT only. Statute field is NOT in labels.parquet —
  lives claim-level (option3_claims_gemma_scale.jsonl / office-action data). Phased
  plan: conditions 1+2+4 + honest-T rebuild first (no new bank needed for T);
  A-bank rebuild only if the cell re-enters mining. Queued behind BBC/V3 GPU jobs.

## 2026-08-12 — peer_revealed audit continuation: TEMPORAL SPLIT of the residual
- **Memorization-suspicion CPU battery LANDED** (peer_identity_audit/
  analyze_memorization_cpu.py + boot_bands.py; descriptive, frozen F2 stack OOFs,
  paired row bootstrap, rows=groups on this cell). y construction verified: top-vs-
  bottom quartile of citation percentile WITHIN venue x year (build_v3_s2.py);
  year-alone on E = .458 (chance, as constructed).
- **HEADLINE: the +.093 deconfounded residual is ERA-LOCALIZED.** (d)-(c) by year band:
  **2013-2019 +.168 [+.090,+.251] P=1.00 (n=134) · 2020-2021 +.135 [+.059,+.215]
  P=1.00 (n=136) · 2022-2023 +.025 [-.018,+.070] P=.87 NOT significant (n=208).**
  On the era where Llama-3.1's pretraining could not contain settled citation
  outcomes (2022-2023), the residual is the same order as the program's small-residual
  cells. Mirror pattern: bank+nuis (c) is much STRONGER on 2022-2023 (.864) than on
  2013-2019 (.715) while dense raw is flat (.87-.90 everywhere).
- **Within-class fame tracking (suspicion lever (c))**: within y=1, raw dense tracks
  each paper's OWN realized citation count at Spearman **.293** vs bank+nuis stack
  **.026** (oof_d .170); within y=0 all ~.07. The trained-on-binary model orders
  positives by graded fame the articulated instrument cannot see.
- **Fame split**: 2020-2021 low-fame half (median 2 cites) residual +.371 — consistent
  with RECOGNITION-ABSENCE as signal (unrecognized -> predict bottom quartile).
- **Two candidate mechanisms recorded, NOT adjudicated**: (i) pretraining
  memorization/recognition of settled outcomes; (ii) bank era-miscalibration
  (criteria fail on old-era conventions; dense generalizes). Discriminator RUNNING:
  base-Llama-3.1-8B abstract-NLL recognition probe (peer_recognition_probe.py, sk3
  GPU0 stacked) -> recognition_readout.py splits the residual by seen-ness within
  year bands.
- **Quoting discipline (immediate)**: the .884/.888 community headline and the
  "RR 1.78 ROBUST" E-value row now carry an era caveat — the residual is a pre-2022
  phenomenon; never quote the pooled +.093 without the band split.

## 2026-08-12 — peer_revealed RECOGNITION PROBE: memorization DISFAVORED
- Probe LANDED (478/478 abstract-NLLs, base Llama-3.1-8B, sk3 GPU0 stacked;
  peer_identity_audit/peer_recognition_nll.jsonl + recognition_readout.json).
  Instrument valid: NLL tracks fame (Spearman vs own cites -.435 in 2013-2019 band,
  -.313 overall) — famous papers ARE more "seen".
- **Residual concentrates in the UNRECOGNIZED half — the OPPOSITE of the memorization
  prediction.** Per band, (d)-(c) recognized-vs-unrecognized: 2013-2019 +.087
  [-.059,+.237] n.s. vs **+.234 [+.099,+.368] sig**; 2020-2021 +.095 vs +.146;
  2022-2023 +.017 vs +.031 (both n.s.); ALL +.061 vs +.110.
- Reading (descriptive): if dense recalled memorized outcomes, its edge should live
  where the model has seen the paper. It lives where it hasn't. Combined with (c)
  being much weaker on old-era rows (.715 vs .864), the leading account is
  **bank era/obscurity-miscalibration** — the articulated instrument fails on
  old-convention and obscure papers; dense generalizes there. The within-positives
  fame-tracking (rho .293) remains a recorded anomaly not fully explained by either
  account (graded quality visible in text is the honest interpretation consistent
  with the probe).
- Kitchen-sink conditioning arm (nuis 57 + identity 4 + NLL + year) running —
  results/f2_kitchensink_peer_revealed.json when landed.

## 2026-08-12 — peer_revealed KITCHEN-SINK arm: residual survives ALL named channels
- results/f2_kitchensink_peer_revealed.json: conditioning = bank + nuisance(57) +
  identity(4) + recognition-NLL + year. (c'') .8136 -> (d'') .8786; residual
  **+.0721 [+.0376,+.1065] P=1.000** (ref +.0927). NLL-alone .378 (reversed .622 —
  recognized papers skew y=1, consistent instrument). AUDIT VERDICT (G4 CLOSED,
  descriptive): the .88 level = honest bibliometric predictability + a residual that
  no named channel absorbs; residual is era-localized (pre-2022) and concentrated on
  UNRECOGNIZED/obscure papers = articulated-instrument miscalibration on old-era/
  obscure text, NOT identity leakage or outcome memorization. Open anomalies recorded:
  within-y=1 fame-tracking rho .293 (dense) vs .026 (bank); T0-null.
- Quoting rules going forward: (1) pooled +.093 only WITH the band split; (2) the
  2022-2023 band residual +.025 n.s. is the honest "current-era" number; (3) identity
  + NLL + year now belong in this cell's nuisance set for any future round.

## 2026-08-13 — peer_revealed TOPIC arm (user: "some fields just get more citations")
- Prior art confirmed on file: closure-time job1 (k-means bge-large strata k=5/10/20
  train-fit + trend deciles + year strata + PC-50 covariate control) — topic-alone
  .7415 on the same 478 rows; dense over topic+bank +.119 [+.083,+.154]; over
  topic+trend+bank +.089 [+.057,+.119]; stratified Delta RISES at k=20 (topic floor
  was the BANK's crutch). Topic was a READOUT robustness check then, not an F2
  conditioning covariate.
- NEW unified arm results/f2_topic_kitchensink_peer_revealed.json: bank + nuis(57)
  + identity(4) + NLL + year + **topic PCs(50, cached bge-large embeddings, 478/478
  joined)**: (c3) .7969 / (d3) .8724 -> residual **+.0728 [+.0372,+.1082] P=1.000**
  (vs +.0721 without topic, +.0927 unconditioned). Topic lifts the bank-side level,
  absorbs NONE of the dense edge. Audit conclusion unchanged; topic joins the
  cell's standing nuisance set with identity/NLL/year.

## 2026-08-13 — BBC r1 scoring DIED + RELAUNCHED (chunked/resumable)
- First run's engine got an external SIGTERM ~1 min after init (56K in-flight
  aborted); the single 1.27M-prompt llm.chat call was unresumable — total loss.
  Death moment coincides exactly with the harness reaping the hung launcher ssh.
- Orphaned EngineCore (172GB, GPU5) killed by PID after ownership check; scorer
  PATCHED: 26 chunks x ~2,000 texts, per-chunk rawchunk json + skip-on-relaunch.
- RELAUNCHED 23:27:34Z; init 28.6s; GPU5 100%; launcher ssh severed EARLY by
  killing local client PIDs (remote setsid job survives clean disconnect —
  verified). Lesson filed: memory/reference_vllm_batch_chunking_launch_detach.md.

## 2026-08-13 — PATENTS REBUILD PHASE 1 LAUNCHED (claim-only V3 + honest T)
- User: "get me a trustworthy VAT number (preferably V3)". Design (post-mortem-
  licensed): **construct RENAMED to what the model measurably does** — "examiner
  rejected this claim element (any ground)", a decision-maker verdict on the claim's
  OWN text; the 8 candidate references are DROPPED from all inputs (kills the
  positives-carry-gold-reference construction asymmetry; placebo criticism moot
  because no reference-reading is claimed).
- Build: datasets/patents/v3_claimonly/ (build_patents_v3_claimonly.py) — SAME ROWS
  as dense_standard splits (47,949/5,994/5,994, app_id-grouped, pos .6014); claims
  median 216 chars. arm_t = claim text only -> honest T for this construct;
  arm_a = V3 block (V_claim + STRUCT: claim ordinal DECLARED per revival cond. 4,
  dependency flag, parent-claim number, length family, wherein/numeric counts) +
  claim text -> fused arm, max-of-variants VAT column only. rejection_type sidecar
  excluded from all inputs; harvest_strata_NEVER_AN_INPUT.csv holds it for the
  §102/§103 replicate readout only.
- Judged-A bank (revival condition 3, online-rubrics -> Gemma) = phase 2, queued on
  GPU5 behind BBC scoring; upgrades arm_a to full V+A+T when landed.
- Chain scripts/tools/patents_v3_chain.sh RUNNING on sk3 **GPU7** via stack runner
  (launched 23:59:54Z; ~1-1.5h/arm at max_len 768 batch 16); launcher ssh severed
  early per the new landmine discipline; watcher armed.

## 2026-08-13 — USER RULING: patents ordinal DECORRELATED, not banked in the block
- User caught a design deviation: my arm_a block included own claim ordinal +
  parent-claim number. The RUNBOOK prereg says ordinal = **declared TRACK-B
  NUISANCE** (quote Δ over V+A+STRUCT) — and own-ordinal is metadata invisible in
  the claim text, so blocking it would have INJECTED the .75 nuisance channel into
  the fused VAT bar (incommensurable with other cells' bars).
- FIXED before arm_a trained (arm_t was mid-training, unaffected): block = text-
  derived content features only (dependency flag kept — computed from a substring
  the model already sees); STRUCT (own ordinal, parent num, dependency, lengths)
  now lives in harvest_strata_NEVER_AN_INPUT.csv as the nuisance block for the
  F2-style deconfounded readouts at harvest.
- Recorded nuance (math.SE lexicalized-ordinal law): the parent-claim mention
  ("of claim 1") IS in the text, so T reads partial ordinal signal regardless —
  that is exactly what the harvest-side conditioning is for; never scrub the text.

## 2026-08-13 — USER RULING: no "STRUCT" tier, DECORRELATE (standing rule)
- "'STRUCT' is not a thing... these are all CONFOUNDS that should be DECORRELATED.
  Don't report 'V+A+STRUCT' as something special."
- Supersedes the patents RUNBOOK's "quote Δ over V+A+STRUCT" PHRASING (the math —
  conditioning on those channels — is unchanged). Patents harvest frame is now
  IDENTICAL to every other cell: nuisance block = {claim ordinal, parent-claim num,
  dependency flag, length family} in the F2-style stacked arms; report the
  **deconfounded residual (d)−(c)** with CI + WY band + matched-sampling check,
  channel list in the caption. Ladder stays V → V+A → fused → T only.
- Standing rule filed: memory/feedback_no_struct_tier_decorrelate.md (applies to
  ALL cells: no structural/positional/length bundle ever gets tier status).

## 2026-08-13 — code_competitions V3 arm LANDED: NEGATIIVE, VAT column unchanged
- Chain completed (5/5 folds, GPU1, ~2h) + scoring pass (score_eval_dense_v4,
  DENSE_SCORE_MAXLEN=4096). OOF over held-out folds, order-join asserted:
  **v3_aug .6554** vs same-rows references V_nl .7429 / VA_nl .7535 / dense .7241
  (block_oof ensembles). Rank-avg [V3, VA_nl] .7242 — no lift anywhere.
- Reading: on a cell already CLOSED as no-residual (bank > dense), pushing the
  criteria into the prompt does not rescue the dense reader. CAVEAT recorded:
  ~680 train rows/fold is thin for a LoRA arm (RR had ~1,000 and 48K-row cells
  train the design properly) — this is a negative result AT THIS n, not a general
  V3 verdict. **VAT column for coding curation stays fused=.7537 (max-of-variants;
  wash vs bank).** Artifacts: code_competitions/v3aug_harvest.json + fold preds.

## 2026-08-13 — V3-MAX launched (user: "why so few training examples? train with more;
## the V3 must see VA+VA_new scores + full coding training data")
- Root cause of thin-n: the v3_aug arm trained only on the 999-row bank-scored AC
  intersection (~680/fold). INVENTORY: full labeled four-platform pool = **6,353 rows**
  (AC strict-L1 2,495 / LC 1,995 / CC 995 / CF 868; the CF-rebuild's 2,255
  bank-scored pairs are UNLABELED — L1 labeling died at 145/2,255 in June, so
  excluded). Judged-A coverage 1,867/6,353 (AC-999 + CF-869).
- **V3-MAX build** (build_code_competitions_v3max.py): block = ALL 27 V/V_new
  deterministic features (computable on every row) + ALL 139 judged criteria WITH
  registry names (global aspect_names.json covers the competitions ids; e.g. a520
  "Competitive-DS algorithm tag count"); real scores where judged, NA elsewhere
  (judge-NA semantics). Folds StratifiedGroupKFold(5,shuffle,rs=0) by platform-
  prefixed canonical_pid over the UNION -> ~4,400-4,500 train rows/fold (6.5x).
  Readout = same-rows AC-999 OOF (primary) + all-platform OOF (secondary).
  Block+code median 7,584 chars, p99 19,904; max_len 6144 block-first (~1% code
  tails truncated). CHAIN RUNNING sk3 GPU5 from 00:57:39Z (~10-15h).
- BBC r1 Gemma scoring COMPLETED (26/26 chunks; anchors pos-vs-neg .539 /
  coherent-vs-scrambled .936 PASS; 2 collapsed criteria; NA 3.0%); scores.npz
  landed; post-completion python reaped; round-1 readout = next CPU item.
- Patents note: arm_a already sees the FULL available articulated set (the 8
  content features — the ref-based V/A died with the references) and trains on all
  47,949 rows; the judged-A upgrade (phase 2, online-rubrics -> Gemma on GPU5 after
  v3max) triggers an arm_a_v2 retrain with criteria scores in the block.

## 2026-08-13 — PATENTS CLAIM-ONLY PHASE 1 HARVEST LANDED (trustworthy ladder v1)
- Construct: "examiner rejected this claim element (any ground)" — claim text only,
  references dropped, same rows/splits as dense_standard (eval+test n=11,988,
  6,128 app_id groups). Artifacts: datasets/patents/v3_claimonly/harvest_v3_claimonly.json.
- **Honest T .7194 eval / .7834 test** (arm_t). **V3 fused arm .7198 / .7859**
  (arm_a; VAT max-of-variants column; block wash vs T pre-A-bank, as expected —
  text-derived features are redundant with text).
- Confound strength (decorrelated, never a tier): NUIS block alone
  {claim ordinal, parent num, dependency, char/word len} = **.7547**;
  claim_num alone .2516 (= .748 reversed). V_content .6198; V+NUIS .7676; +T .7907.
- **DECONFOUNDED RESIDUAL (d)-(c) = +.0229 [+.0171, +.0285] P=1.000** — a small,
  real dense increment beyond content features + every structural confound.
- **§102/§103 replicate: STABLE** — T .7639, residual +.0237 [+.0184,+.0288]
  (n=10,259). The claim-only construct does NOT have the placebo problem by design
  (no reference-reading claimed), and the residual is ground-insensitive.
- vs the old cell: old T .7965/.8389 carried ~.05-.08 of reference/asymmetry
  channel now gone; old ref-based V .601/VA .626 RETIRED to appendix with the old
  construct. Phase 2 (judged-A bank from online-rubrics -> Gemma) fills V+A and
  upgrades the fused arm; queued for next free GPU window.

## 2026-08-13 — CODEX BACK (user caught it: usage panel 100% remaining)
- Probe: `codex exec --skip-git-repo-check` returns CODEX_ALIVE. The "Aug 18" reset
  message was pessimistic — weekly window rolled. Judge-checks-use-codex rule back
  in force; memory updated. UNBLOCKED: patents (V+A)_new 2-family mining fleet,
  SI Phase 2 (mining vs pairwise residual), BBC Track-B second-judge upgrade,
  V_new compilation with the original codex compiler.
- Patents bank_v1 CURATED (GLM sweep 10x150 -> 193 concepts -> merged 30 criteria,
  label-blind; datasets/patents/v3_claimonly/bank_v1.json): clarity/definiteness,
  antecedent basis, relative-terminology, prepositional clarity families etc.
  Next: Gemma scoring of eval+test claims (11,988 x 30 + anchors) on a free GPU.

## 2026-08-13 — PATENTS V+A LANDED: the bank absorbs most of the dense residual
- Harvest (datasets/patents/v3_claimonly/harvest_va_patents.json; bank_v1 30 criteria,
  4 collapsed, all levels pooled eval+test grouped-OOF unless split-marked):
  **A_nl .6648** (judged bank alone BEATS V_content .620) · **V+A_nl .6677**
  (per-split OOF .6165 eval / .7147 test) · fused stack [VA,T] **.7544**
  (.7171 eval / .7909 test) · honest T pooled .7513 (.7194/.7834).
- **DECONFOUNDED RESIDUAL COLLAPSES with the bank in (c): +.0229 -> +.0090
  [+.0049,+.0130] P=1.0** (c=V+A+confounds .7834, d=+T .7927). §102/§103 replicate
  +.0073 [+.0036,+.0112] — stable. Reading: the 30 articulated criteria captured
  ~60% of what dense knew beyond content+structure; the remaining dense edge on this
  construct is tiny but real. VAT column (max-of-variants): eval .720 (V3 arm) /
  test .791 (fused stack).
- Ladder status: V .620 / V+A .668 / fused .754 / T .751 (pooled frame); V_new +
  (V+A)_new = compilation + 2-family mining fleet (codex back), queued behind
  train-side scoring (GPU6, running) and bank certification write-up.

## 2026-08-13 — BBC most-read ROUND 1 READOUT LANDED (mining productive; r2 required)
- readout_r1_bbc.py on sk3 (readout_r1_results.json). TRACK A: the 16 A-routed mined
  criteria lift the bank **VA .7502 -> .7662 on MONITOR** (seeds .7634-.7656);
  **Delta .0729 -> .0570; gain +.0159 > eps .005 = NOT sub-eps** -> round 2 proceeds
  (rule couldn't fire at r1 regardless). Convention note: Delta_0 here uses
  VA0=seedmean_pred .7502 (T_MON .8231); the registry's earlier +.0749 used
  mean-of-seed-AUCs .7482 — both recorded, same instrument.
- SPURIOUS: all 9 B channels weak — top "Named-public-figure lead" .5534, joint B
  .5785 (vs bank .722 lin); dense over B+bank_r1 linear stack +.0960 on the 10,147
  dense-held-out rows. SWAP: C+ .794 / C- .392 — the bank actively disagrees with
  dense exactly where dense errs (strong complementarity signature).
- Notebook journalism community row updated (note only — MONITOR-frame numbers are
  never mixed into the E-frame bars).

## 2026-08-13 — CORRECTION: BBC r1 readout used PRE-ARBITER routing (superseded, rerunning)
- The r2 slice builder's own assertion (expected 16 A-routed, got 18) exposed it:
  routing_final.json carries BOTH `audit_track` (pre-arbiter, A=18/B=7) and
  `final_route` (arbiter-final, A=16/B=9). The r1 readout used `audit_track`, so
  VA1 .7662 included 2 arbiter-demoted nuisance channels and the B map missed 2.
- **Numbers from the first readout (VA1 .7662, gain +.0159) are SUPERSEDED** —
  json renamed readout_r1_results.AUDITTRACK_SUPERSEDED.json; both scripts fixed to
  `final_route`; readout RERUNNING on sk3. Notebook note will be refreshed on landing.
- Lesson (transferable): finalize artifacts that retain pre-arbiter fields need the
  consumer to name the FINAL field explicitly; the r2 builder's count assertion is
  what caught it — keep count assertions in every routing consumer.

## 2026-08-13 — BBC r1 readout v2 (ARBITER-FINAL routing) — CORRECTED NUMBERS OF RECORD
- With final_route (16A/9B): **VA .7502 -> .7586 on MONITOR; Delta .0729 -> .0645;
  gain +.0084 > eps .005 — NOT sub-eps, r2 proceeds.** The superseded audit_track
  readout had shown gain +.0159: the 2 arbiter-demoted channels were carrying ~half
  the apparent bank gain — the demotions were load-bearing, arbiter vindicated.
- Joint B (9 channels) .6305; dense over B+bank_r1 +.0960 unchanged.
- r2 slice BUILT with corrected routing + r1 bank state + prior-round row ban
  (slice_r2.json). PATENTS train-side bank scoring COMPLETE (47,949 claims x 30) —
  arm_a_v2 refit unblocked.

## 2026-08-13 — PATENTS V_new LANDED (compilation protocol; 4-column ladder complete)
- patents_vnew.py (vnew_pilot protocol; compiler codex gpt-5.6-luna, label-blind;
  cert on EVAL: rho>=.30 + modal<=.98): triage 18/26 codable -> 18 compiled ->
  **8 certified** (pb04 terms-of-degree, pb06 prepositional clarity, pb07/pb17/
  pb20/pb23/pb24/pb26). Yield 44% of compiled vs the pilot's 6.5% — structural/
  grammatical criteria compile; the pilot's codability law replicates from the
  opposite side.
- Ladder (pooled eval+test OOF): **V .620 -> V_new .627** (+.008; test .651->.654).
  **V_new+A .668 ~= V+A .667** — compiled columns are INERT on top of the judged
  bank (pilot finding replicates: translations of judged criteria add nothing to
  the judge). Deconf residual with V_new+A in (c): +.0110 [+.0068,+.0149].
- Vocabulary note: this fills the COMPILED-columns sense of "new". The MINED sense
  ((V+A)_new via discovery fleet) is the remaining open column — 2-family fleet
  queued as the next campaign (morning item; BBC r2 has fleet priority tonight).

## 2026-08-13 — BBC ROUND 2 executed end-to-end to the scoring boundary (one night)
- Fleet: 16 sealed prompts (8 proposers x 2 tracks, codex-luna x4 + GLM x4), 200/200
  collected (120A/80B), P x k count-check clean. Species pre-merge: A S_obs=70
  M_hat .400 recapture .31 · B S_obs=51 M_hat .525 (masses FALLING vs r1 —
  consumption continues).
- Blind merge: CROSS-FAMILY BOTH TRACKS this round (sol + GLM; r1's single-judge-B
  flag cleared for r2): strict edges A=47 / B=27; anchors 4/4 both tracks; applied
  via species_merge dual-verdict mode (NOTE: apply takes comma-separated judge
  files and does the strict intersection itself — my hand-rolled intersection file
  had dropped the anchor pids, caught and redone properly).
- Routing audit: fresh probe draw (stable-hash, prior-round ban), auditor
  gpt-5.6-sol (family rotated from r1's GLM): misrouting 6/25, **probes 4/4**;
  GLM arbiter ruled all 6 disputes -> **final A=14 / B=11 (6 mixed)**.
- Gemma scoring of the 25 r2 criteria LAUNCHED (GPU6, chunked scorer). Readout r2
  after: X2 = [V, A, A1(16), A2(14)]; Delta_2 vs Delta_1 +.0645; a sub-eps gain
  here = FIRST sub-eps round (rule needs two consecutive).

## 2026-08-13 — BBC r2 READOUT: gain +.0067, still super-eps — ROUND 3 REQUIRED
- Track A: VA .7586 -> .7653 MONITOR (the 14 r2 criteria add +.0067); Delta .0645 ->
  .0578; **gain +.0067 > eps .005** — NOT sub-eps. No sub-eps round yet; the
  two-consecutive rule cannot fire before r4 at the earliest. Gain trajectory
  .0084 -> .0067 = geometric shrink toward the eps floor.
- Spurious: joint B (20 channels r1+r2) .6532; dense over B+bank .0845.
- ROUND 3 LAUNCHING: slice with r2 bank state + ban of r1+r2 slice rows; same
  2-family fleet + cross-family merge + fresh-probe audit flow.

## 2026-08-13 — BBC r3 to scoring boundary; patents arm_a2 training
- r3 fleet 200/200 (after a 4th parse-variant extension for glm_d's bare-mixed pipe
  format — parse-only, sealed content untouched); species: A-mass ROSE .400->.517
  (real-space churn) while B flat .525 (consumed) — the program's known asymmetry
  in-round. Cross-family strict merge anchors 4/4 (A edges 39 / B 31); audit
  probes 4/4, arbiter ruled 5 -> **final A=11 / B=14** — first round where nuisance
  outnumbers substance (plateau signal). Gemma scoring on GPU7.
- Patents arm_a2 (V3 block WITH 26 judged criteria scores, full 59,937 rows) built
  after fixing an NpzFile lazy-decompression trap (z["X"] per-access = 1.8M
  decompressions; materialize once) and TRAINING on GPU6 (~2-3h at max_len 1280).

## 2026-08-13 — BBC r3 READOUT: gain +.0058, super-eps by .0008 — ROUND 4 REQUIRED
- Track A: VA .7653 -> .7712 MONITOR; Delta .0578 -> .0520; gain +.0058 vs eps .005.
  Gain trajectory .0084 -> .0067 -> .0058: geometric approach to the eps floor, each
  round still individually resolvable (paired-seed noise .00252). No sub-eps round
  yet; two-consecutive rule cannot fire before r5. Round cap = 5 (prior campaigns).
- Spurious: joint B (34 channels) .6685 rising as B accumulates; dense over B+bank
  +.0733 (was .0960 -> .0845 -> .0733 — the deconfounded dense edge is being eaten
  by the growing bank+nuisance set, as designed).
- ROUND 4 LAUNCHING (same flow).

## 2026-08-13 — BBC r4 READOUT: gain +.0067 super-eps AGAIN — r5 = FINAL (cap)
- VA .7712 -> .7779; Delta .0520 -> .0452; gain +.0067 (up-tick from r3's .0058).
  Gains r1-r4: .0084/.0067/.0058/.0067 — NO sub-eps round in four; the bank rises
  +.028 total while the eps floor is never crossed. With the round cap at 5 and the
  rule needing TWO consecutive sub-eps, the rule CANNOT fire -> the campaign ends at
  r5 as **TERMINAL BY CAP, STILL GAINING** (caption-finalist pattern; terminal bank
  = LOWER bound on articulability).
- Spurious: joint B (47 channels) .6884; dense over B+bank +.0676 (trajectory
  .0960 -> .0845 -> .0733 -> .0676 — monotone consumption of the dense edge).
- ROUND 5 (final) LAUNCHING.

## 2026-08-13 — BBC MOST-READ CAMPAIGN TERMINAL (5 rounds, one overnight)
- **VERDICT: TERMINAL BY CAP — stopping rule never fired** (gains +.0084/+.0067/
  +.0058/+.0067/−.0019; only r5 sub-eps). Bank VA .7502 -> .7779 peak (r4); Delta
  .0729 -> ~.046: **~37% of the best-powered residual in the grid closed by mining;
  the remaining ~+.046 MONITOR residual is REAL, resolvable, and NOT closed** —
  terminal bank = LOWER bound (caption-finalist pattern at much higher power).
- 66 arbiter-final mined A criteria added; 71 B channels mapped (joint .6908; top
  "Headline length" .598; question/explainer-framing family ANTI .428 — the
  homepage "less distinctive wins" inversion family recurs on the crowd side).
- Dense-over-everything-named: .0960 -> .0636 monotone. FUSED (B+bank+dense stack,
  10,147 held-out rows): **.8170** — the journalism community VAT column candidate.
- Swap C+ .835 / C- .381: strongest complementarity signature in the program.
- Instrument record: anchors 4/4 x5 rounds x2 tracks; probes 4/4 x5; cross-family
  strict merges throughout; misrouting 6/25 -> 2/25. Ledger:
  closure/bbc_mostread/bbc_mostread_TERMINAL_LEDGER.json

## 2026-08-13 — V3-MAX LANDED: well-powered NEGATIVE on code competitions
- All 5 folds trained on the 6,353-row four-platform pool (~4,470 train/fold, full
  27-V + 139-named-A block) + scored. **AC same-rows .6846** vs VA_nl .7535 / dense
  .7241 / thin arm .6554: the data+block upgrade bought +.029 but the arm still
  loses to BOTH parents. Thin-n caveat RESOLVED — this is now a powered negative:
  criteria-in-prompt underperforms external stacking on this cell; §11 fused stack
  (.7537) remains the VAT column. All-platform OOF .8180 recorded as
  COMPOSITION-INFLATED (platform pos-rates .19-.84), never a cell number.
  Artifacts: code_competitions/v3max_harvest.json.

## 2026-08-13 — PATENTS arm_a_v2 LANDED: wash — V3-design verdict now CROSS-CELL
- arm_a2 (26 judged criteria scores in the block, full 59,937 rows): eval .7167 /
  test .7808 ~= arm_t plain text .7194/.7834 ~= arm_a content-block .7198/.7859.
  **Criteria-in-prompt adds NOTHING on patents** — same verdict as V3-MAX on code
  competitions (.6846 < stack .7535). CROSS-CELL FINDING: at both data scales and
  on two fields, the V3 fused design (scores in prompt -> LoRA) never beats
  external stacking; the §11-style stack remains the program's fusion of record.
  Patents VAT column unchanged: fused stack .754 pooled (.717/.791).
- PATENTS COLUMN STATUS: V .620 / V_new .627 / V+A .668 / V_new+A .668 / fused
  .754 / T .751; deconfounded residual +.0090 [+.0049,+.0130] (~2x eps).
  RECOMMENDATION for the remaining (V+A)_new-mined column: the bank already
  absorbed the dense edge down to ~.009 — a mining fleet would chase eps-scale
  headroom. Propose declaring the cell CLOSED at phase 2 unless the formal mined
  column is wanted; awaiting user.

## 2026-08-13 — homepage fused bar + peer-curation deconfounding check (user Qs)
- **Homepage fused LANDED**: grouped-OOF stack [VA_nl OOF, dense seed-mean] on the
  2,631 dense-held-out rows = **.7402 pooled (.7258 eval / .7544 test)** — beats
  both parents (VA .7143 / dense .7340 same rows). Fills the last empty journalism-
  curation column; closure unchanged (terminal r0). V+A_new stays empty BY DESIGN:
  terminal at round 0 = no mining rounds = no mined criteria.
  results/homepage_fused_stack.json.
- **Peer curation deconfounding audit (question-answer)**: F2 primary is NULL
  (-.0006, P=.48) over bank + 52 mined nuisance channels. Venue is FIXED by
  construction (population 100% ICLR); **year-alone AUC .4951 = chance** (checked
  today; pos-rate wobbles by year but composition washes out); nuisance_struct=0
  (no explicit year/venue covariates) BUT the null-residual asymmetry applies:
  adding covariates to (c) can only shrink (d)-(c) further — a null cannot be
  manufactured by missing confounds. The fleet itself mined the identity-adjacent
  channel (r1:B09 "Identity-revealing URLs, handles and organisation-named host").
  LEVELS are raw instrument AUCs (ladder convention, all cells).

## 2026-08-13 — peer community: ERA-REWEIGHT + JUDGE-DRIFT hypotheses KILLED
- era_reweight_test.py (peer_identity_audit/): bank stack FIT ONLY ON PRE-2022 rows
  (n=1,357, full V17+A154 matrix, grouped-OOF, seeds 0-2) reads **.7165** on
  pre-2022 — WORSE than the pooled-fit stack on the same band (.7353); dense on
  held-out pre-2022 = .8840. **Criterion nonstationarity/re-weighting does NOT
  explain the residual** — era-specific weights recover nothing.
- Judge calibration by era: per-criterion dispersion pre-2022 vs 2022+ essentially
  identical (SD .130 vs .121; NA .654 both). **Judge-drift disfavored.**
- SURVIVING EXPLANATIONS (2): (i) era-specific MISSING VOCABULARY — the fleets
  mined on a recent-heavy corpus; pre-2022 quality conventions may need their own
  criteria (testable: era-targeted mining round on pre-2022 disagreement slices);
  (ii) CONFIGURAL/GESTALT reading — quality carried by combinations/global
  coherence no marginal criterion expresses (Noah weak-form; also fits the
  within-positives fame-gradation anomaly rho .293 vs bank .026).

## 2026-08-13 — peer community: ERA-VOCABULARY-AS-LEAK tested at two strengths — disfavored
- User concern: era-targeted mining could re-encode era voice = topic leak. Tested
  as an explicit spurious channel BEFORE any mining:
  (a) scalar predicted-year (grouped-OOF ridge on embeddings; rho .432 — user
  correctly flagged coarseness): overall residual +.0728 -> +.0662; pre-2022 band
  +.116 -> +.114 (untouched).
  (b) STRENGTHENED subspace: 3-class era-band probabilities (OOF acc .521 vs ~.44
  chance) + 200 topic PCs: overall +.0730 -> +.0734 (nothing); pre-2022 +.130 ->
  +.119 (~.01, within band noise at n=134).
- ALSO verified: y does NOT cluster by year (pos rate .475-.533 across all 11
  years; year-alone AUC .487 = chance; label is within venue x year quartiles by
  construction) — the E-band pos-rate wobble (.58 old band) is subset composition
  and cancels in the paired (c)/(d) band readouts.
- VERDICT: era-voice conditioning absorbs essentially nothing at either strength.
  Caveat recorded: bge-large era-recoverability is modest (.52 acc) so an 8B could
  read finer era cues — but recognition-NLL (already conditioned) covers the
  adjacent channel. The era-targeted mining round remains the discriminator between
  the two survivors (missing quality vocabulary vs configural), with the standing
  instruction that any mined era-correlated criterion must pass the same routing
  audit (quality vs incidental) as every other criterion — the audit+arbiter IS the
  topic-leak gate. Artifacts: results/f2_eravocab{,2}_peer_revealed.json.

## 2026-08-13/14 — code competitions FULL-UNION rebuild, part 1 (user: "use ALL the data")
- CONFIRMED the starvation charge: canonical ladder used AC-999 only (16% of 6,353
  labeled rows) and its dense T was ModernBERT-base 150M (not the Llama standard).
- FIXES RUN: (a) coded-metric bank (the "A" here = 154 deterministic CPU metrics,
  not LLM-judged) scored on ALL 6,353 rows; (b) full-union V/A/VA ladder (frozen
  stack, folds by platform-prefixed canonical_pid, WITHIN-PLATFORM readout);
  (c) union plain-text Llama LoRA dense TRAINING (same folds as v3max).
- PART-1 RESULT — the union HURTS the articulated instruments: within-platform
  n-weighted V .5994 / A .6143 / VA .6201 (per-platform VA: AC .677 / CF .523 /
  LC .595 / CC .612). Apples-to-apples on the IDENTICAL AC-999 rows: union-fit
  VA .7093 vs AC-fit VA .7535 (V .7150 vs .7429). **Cross-platform pooled fitting
  degrades AC readouts — the four platforms are NOT one construct** (label
  provenance differs: AC strict-L1 vs claude_label elsewhere; CF sits near chance).
- OPEN (the real starvation test): the union-trained Llama dense on AC-999 —
  dense transfer can win where feature stacks lose. If union-dense > .754 the cell
  verdict flips; if ~.70 the bank>=dense verdict stands with the objection
  properly answered. Chain running; artifacts union_va_ladder.json/union_va_oof.npz.

## 2026-08-14 — peer r6-ERA probe: CLEANEST AUDIT OF THE PROGRAM; scoring queued
- Chain through routing complete under the frozen prereg (ae7b7c329): slice 60
  cards (pre-2022 M rows, r1-r5 banned), 3-family fleet 150 proposals (12/12
  slots; claude legs = sealed fresh `claude -p` sessions — DEVIATION from the
  prereg'd 2-family P=8, recorded), species A .333/B .400, strict sol+GLM merge
  (anchors TRUE, edges 33/33), audit packet 29 items (4 planted).
- **AUDIT: misrouting 0/25, probes 4/4, disputes 0 — no arbiter round needed**
  (first campaign round anywhere with a zero-dispute audit). final A=15 B=10
  (4 mixed). Topic-leak gate applied: all 15 A-routed criteria passed as genuine
  quality properties, none as era/topic markers.
- Gemma scoring launched via gpu_runner (job peer_r6_maps, waits for a free GPU;
  log closure/peer_revealed/peer_revealed_r6_scoring.log). Prereg'd readout
  STAGED: peer_identity_audit/r6era_band_readout.py (both frames, banded
  residuals + paired gboot, declared verdicts; runs on scores landing).

## 2026-08-14 — BBC E-value chain STAGED (t0 rows + F2 adapter); patents E-value BLOCKED
- fusion/t0_build_rows_bbc.py (homepage-pattern post-hoc builder): t0_rows/
  bbc_mostread.{npz,texts.jsonl.gz,meta.json}; n_E=10,147 dense-held-out rows,
  T(seedmean pooled) .8218; gates = dense_join order-proof + bank-id coverage +
  split-vs-population y equality.
- f2_cells.py: bbc_mostread SPECIAL adapter (cells.py is metadata-only; loads via
  round0_bbc + scaleupC_layer1, shared _round_blocks). Smoke: bank 103 cols
  ([V,A]+66 mined A over r1-r5), nuis 59 B channels, E=10,147, family
  impute_perfold. Remaining chain: t0_score_vllm (GPU, queued BEHIND r6 scoring
  per the one-GPU rule) -> f2_deconf -> f2_evalue (species artifacts resolve:
  r5 strict preferred).
- **Patents E-value: NOT COMPUTABLE under the frozen definition** — closure/
  patents_claimonly has r0 selection artifacts only, no Good-Turing Track-B
  species (the claim-only construct never ran a sealed mining fleet). The row is
  contingent on the un-launched patents (V+A)_new fleet (recommendation on file:
  CLOSED at phase 2, ~eps headroom). Folded into that open user decision.

## 2026-08-14 — r6-ERA PROBE READOUT: prereg verdict = H-CONFIGURAL STRENGTHENED
- Scoring landed (anchors pos-vs-neg .662 / coherent-vs-scrambled .950 PASS,
  0/25 collapsed, NA 1.6%). Readout ran under the frozen prereg (ae7b7c329);
  artifact results/r6era_band_readout_peer_revealed.json.
- **Baseline reproduction exact** (frame-identity gate): plain 2013-19 +.1676 vs
  quoted +.168; kitchen-sink +.1298 exact.
- **PRIMARY (2013-19 band residual, r6 A-routed 15 appended to (c)):**
  plain +.1676 -> +.1692 (fall -.0016); kitchen-sink +.1298 -> +.1644 (fall
  -.0346). Both falls < .02 -> **frozen rule fires: H-configural STRENGTHENED**.
  SECONDARY: 2022-23 moves -.0123 / -.0165 (stable, within .04 window).
- Sharpest descriptive fact: on 2013-19 the (c) arm is UNCHANGED at .715 with the
  15 era-mined criteria added — the criteria mined FROM pre-2022 disagreement rows
  add zero discrimination on that band. Partial vocabulary recovery only on
  2020-21 (plain: bank .720->.750, residual +.135->+.105).
- Reading (descriptive): the pre-2022 dense edge is not missing-vocabulary at
  P=12/3-family mining strength; the configural/gestalt account is the last
  survivor of the 8-account audit. Terminal ledger untouched (registered
  post-terminal probe).
- BBC t0 scoring launched behind it (gpu_runner bbc_t0, env vllm_latest — first
  launch used a nonexistent env path, killed by PID and relaunched).

## 2026-08-14/15 — UNION DENSE HARVEST: coding-curation "bank >= dense" verdict STANDS
- code_uniont chain (plain-text Llama LoRA, ALL 6,353 four-platform rows, v3max
  folds, max_len 4096) trained 5/5 + scored (test = held-out fifth; eval slice is
  INSIDE train groups, so the OOF is honest — no select-on-heldout).
- **Within-platform n-wtd .6471** (ac .651 / cf .597 / lc .687 / cc .602); pooled
  .? NEVER QUOTED (composition). Artifact code_competitions/uniont_harvest.json.
- **AC-999 same rows: union dense .6744 < AC-fit dense .7241 < union-fit VA .7093*
  < AC-fit VA_nl .7535 = the bank.** (*union-fit VA recomputed on identical rows
  from union_va_oof.npz.) 6.4x more training data made the dense arm WORSE on
  AC — cross-platform transfer hurts; the platforms are not one construct (both
  instrument families now agree). Data-starvation objection ANSWERED; the cell's
  verdict does not flip. LC is the union dense's best platform (.687).

## 2026-08-15 — BBC T0 COLLAPSE RESOLVED: genuine saturation, recorded not patched
- BBC t0 p_yes was one-hot ({0,1}, 2 distinct/10,147). Differential probe
  (t0_probe_logprobs.py; homepage-100 = known-continuous control, bbc-100, modes
  default AND raw_logprobs): **homepage continuous in both modes (33/36 distinct),
  BBC all-1.0 in both modes** -> the env is the same instrument; the collapse is
  DATA-driven (base Llama hyper-confident on short headline prompts; loser-token
  mass underflows float64 beyond ~36 nats). Same category as the original
  battery's recorded collapses (hashtagwars 7-distinct; 4 cells median-saturated)
  -> per that precedent the scores STAND, flagged COLLAPSE, never patched.
- f2_deconf --cell bbc_mostread RUNNING on sk3 (n_E=10,147, bank 103, nuis 59);
  the T0 (e) secondary will read "uninformative" — the PRIMARY (d)-(c) does not
  touch t0. f2_evalue next on landing.

## 2026-08-16 — UNIFIED-X PROGRAM CUED (user directive): X -> {y_verdict, y_curated, y_community}
- Goal (user): move away from one-domain-per-y; same corpus carries all three label
  types. Four domains cued (tasks U1-U4):
  * **RoyalRoad** (U1): X = fiction descriptions. 135,598 listings (community:
    rating_pct/followers/views) + 1,584 STUB market-pickup (verdict, popularity-
    confounded DECLARED) + magazine contest 10 editions/2,173 chapter files
    (curated; y-definition needs toc spot-check). All local under
    datasets/creative-writing/royalroad_expansion/. Deep pages have description +
    full stats (verified on 100005.html.gz).
  * **r/Jokes** (U2): X = joke text. Mod-removal verdict (wayback stageC fetch LIVE
    on sk3: 28.8K/64K queue, 33% text recovery -> build on landing) vs kept
    universe 184,774 w/ scores; awards metadata 999,836 rows = second community y.
  * **math.SE** (U3): bounty-close = CURATED y (38,748) on the same answers X as
    the existing accepted (verdict) + vote_score (community) cells. 7z EXTRACTED:
    data/se_dumps/mathse_extracted/{Posts,Votes}.xml.
  * **StackOverflow** (U4): accepted 12.44M (verdict) + bounty-close 290,786
    (curated) on the V6 so_votes X. Votes.xml EXTRACTED (23GB).
- Vote-type decode confirmed from censuses: 1=accepted, 9=bounty-close, 2=up.
- Already-scored (NOT rebuilt): jokes_community upvotes, mathse accepted + votes,
  so_votes community (V6 live), cw_community, Wigleaf.
- Order of build (dataset-first protocol each): U3 (unblocked, smallest design
  risk) -> U1 community+verdict (local parse) -> U1 curated (after toc check) ->
  U4 (behind V6) -> U2 (behind the live fetch).

## 2026-08-16 — U3+U4 POPULATIONS BUILT (user: prioritize math.SE + SO, careful/precise)
- **Bounty award-mode audit FIRST (the precision step both cells needed):** SE
  bounties are awarded MANUALLY (full amount, deliberate curation) or AUTO at
  expiry (system half-award to top-scored answer = community signal, NOT
  curation). Classifier = close-amount vs start-amount join (full=manual,
  floor(half)=auto); validated by the day-gap histogram (auto piles at day 8 =
  expiry+grace; manual spreads 1-8). Auto-award questions DROPPED entirely.
  * math.SE: MANUAL 8,700 / AUTO 1,248 (87/13). Winner-is-top-scored: manual .64
    vs auto .75 — 36% of deliberate awards pick a NON-top answer = the curated-vs-
    community divergence the cell measures. data/se_dumps/mathse_bounty_award_mode_audit.json.
  * SO(python): MANUAL 12,214 / AUTO 2,230 (85/15) — replicates.
- **math.SE bounty population** (U3, data/se_dumps/mathse_bounty_manual_population
  .jsonl.gz): within-question (winner vs other answers, same design family as the
  vote cell), 8,368 questions / 24,317 rows / pos .354 / median answer 1,525 chars;
  16,360/37,765 close votes dropped as expired-or-question-attached (correct).
  210 multi-winner questions (multiple bounties) retained, flagged.
- **SO bounty population** (U4, so_bounty_manual_population.jsonl.gz): same
  design on the V6 python corpus (so_python parquets = same X-space as so_votes):
  7,128 questions / 23,942 rows / pos .301. BUG CAUGHT on run 1: parquet ParentId
  is float64 -> astype(str) gives "337.0" and silently breaks the XML join (all
  closes -> OTHER); fixed with int64 cast + notna assert. Spot-check: winner
  score 15 beat competing score 24 — curation != votes, live in the data.
- **SO verdict = free on V6's rows**: population.csv.gz already carries
  y_accepted as a separate never-merged column on the same X/splits.
- Next: text-prep MATCHED to sibling cells (check v2_va formatting), splits,
  V features, A-bank reuse check (bounty answers may already be Gemma-scored in
  the vote cells), then ladders.

## 2026-08-16 — U3/U4 CELLS BUILT (dense_standard layout; instruments NOT yet run)
- Sibling-convention identity enforced by IMPORT, never reimplementation:
  math text = build_multiy_v2.clean_body + QUESTION:/ANSWER: template + 50/12,000
  gates + sha1(qid|aid)[:20] row ids; SO text = V6 convention verbatim (body.strip,
  answer-Id row ids; ONE deviation recorded: 12,000-char max added for math parity).
  Splits = the frozen stable_hash_bucket_map (hashtagwars lineage) on question
  groups, both cells.
- **mathse_bounty** (datasets/math-se/mathse_bounty/): 23,972 rows / 8,278 q /
  pos .3547; splits 19,177/2,399/2,396 pos-matched to 4 decimals; char gate
  dropped 184.
- **so_bounty** (datasets/stackoverflow-votes/so_bounty/): 23,498 rows / 7,036 q /
  pos .3023; splits pos-matched; gate dropped 163.
- **so_accepted** (datasets/stackoverflow-votes/so_accepted/): V6 rows/text/splits
  VERBATIM, judgement = y_accepted (asserted: binary, exactly one accepted per
  question); 16,001 rows, pos .375/.366/.368 by split.
- A-score reuse checked and REJECTED for rows (overlap ~1.4% both domains:
  math 344/24,317 by answer_id, SO 285/23,942) — criteria BANKS will be reused,
  rows scored fresh.
- NEXT (the actual VAT runs): V features (sibling extractors), A = Gemma pass of
  the sibling banks on the new rows (one vLLM load, multi-job), T = dense chains
  (3 seeds x 3 cells, sequential one-GPU), then ladders + F2.

## 2026-08-16 — VAT LAUNCHED for U3/U4 (A-pass smoke-gated); U5 WritingPrompts cued
- **A-bank scorer** datasets/stackoverflow-votes/score_bounty_banks.py: two bank
  jobs in ONE Gemma load — mathse_bounty scored with the mathse_multiy sibling
  bank (SYS/trunc/ctx IMPORTED from score_scaleupC_banks, verbatim), so_bounty
  with the so_votes sibling bank (SYS/ctx incl. load-bearing question BODY
  imported from score_so_votes_bank). vvec = sibling v_features (V columns come
  free from the framework). Anchors = own-y pos/neg/scram per shard.
  Outputs -> outputs/va_gemma_banks_{mathse,so}_bounty/. **SMOKE (40 items/cell)
  RUNNING via gpu_runner (job bounty_smoke) — full pass gated on NA/modal/anchor
  inspection per the standing smoke rule.** so_accepted needs NO scoring (V6
  shards cover its rows).
- Dense chains staged conceptually; BLOCKED on one design check before launch:
  V6's T for so_votes is the QTRUNC variant — the so_accepted/so_bounty T must
  match the sibling convention (check dense_standard_so_votes_qtrunc args) for
  commensurable T columns. mathse T = sibling defaults.
- **U5 (WritingPrompts) located + cued (task #78):** datasets/creative-writing/
  bestof_writingprompts/ (scraper + award_threads/ present). USER RULING logged
  on jokes (task #75): awards = COMMUNITY not curation (gilded .19%, median
  score of gilded 14,438 — virality function); votes community on the FULL
  184,774 kept universe; removal verdict waits on the live wayback fetch.

## 2026-08-17 — SYNC TO sk1 + DENSE CHAINS LAUNCHED THERE (user directive)
- **Who holds sk3**: all 8 GPUs occupied by other users — l1ly (6 procs) +
  nntruong (incl. the 115GB GPU4 engine; earlier attribution of that one to us
  was WRONG — it is nntruong's, untouched). Our bounty_smoke gpu_runner stays
  queued on sk3 for the Gemma A-pass (Gemma-4-31b weights are sk3-only).
- **sk1 (skampere1, A100-80GB)**: GPUs 6/7 free; norm-research checkout present;
  Llama-3.1-8B in shared_hf_cache (HUB-layout dir -> HF_HUB_CACHE, not HF_HOME —
  first relaunch failed on this). unified_v1 env lacked pandas/peft/sklearn/
  datasets/accelerate -> installed. SYNCED: methods/dense, math-se +
  stackoverflow-votes builders/scorer, all three cell dense_standard bundles
  (sk3 -> sk1 direct rsync). sk2 untouched (no shared-code edits in its area —
  divergence note per the sync rule).
- **so_accepted_qtrunc bundle built**: V6's qtrunc variant data VERBATIM
  (12,202 rows — the sibling T-frame subset), judgement=y_accepted by row_id
  (pos .428/.451/.464 by split). Same-frame T comparability with so_votes.
- **unified_dense_chain_sk1.sh RUNNING on sk1 GPU6** (verified: pgrep + 27.7GB
  on GPU6 + seed42 START): mathse_bounty (3 seeds) -> so_accepted_qtrunc
  (3 seeds), frozen scaleupC recipe, scoring pass per cell. so_bounty dense
  deferred until its own qtrunc bundle is built with V6's builder logic.

## 2026-08-17 — SMOKE GATE PASSED; FULL A-PASS RUNNING STACKED (GPU5)
- Smoke (40 items/cell, stacked GPU5 util .50): mathse_bounty NA .345 mean .726,
  no pathological columns; so_bounty NA .487 mean .644 with 4 high-NA criteria —
  **gate check against the SIBLING'S FULL 16K-row matrix: same profile** (sibling
  overall NA .387; its top-NA criteria are the IDENTICAL ones — "Complexity claim
  matches the code" .96 vs our .95, "Performance claim is substantiated" .92 vs
  .90). Bank's known shape, not an instrument break; bountied questions read
  slightly harder (NA +.10), consistent with construct.
- **FULL pass launched** (1.7M prompts, both cells, one Gemma load, stacked GPU5
  alongside l1ly's tenant, 142GB/183GB total): 8 shards/cell + anchor batteries
  + V vectors. Watcher armed. Expect slower-than-dedicated wall clock (compute
  contention).
- STANDING RULE FILED (memory feedback_gpu_usage.md): co-tenant stacking OK when
  tenant is very long-running or one user hogs 5-6/8 GPUs; size mem fraction to
  free headroom, STACKED-ok ledger entries, co-tenants never touched.

## 2026-08-17 — FIRST UNIFIED-X LADDER LANDED: so_accepted (SO VERDICT) V/A/VA
- CPU-only on the V6 scored matrix (16,001 rows, y=accepted; collapse gate
  dropped 1 criterion -> 39): **V .639 / A .704 / VA .717** (nl mean-3; within-
  question pair-weighted .654/.706/.723). vs the so_votes COMMUNITY sibling on
  the same rows: V .638/VA .710 — verdict and community read near-identically at
  the VA level on this corpus (the cross-y contrast sharpens at T + fused).
  Artifacts: results/so_accepted_{ledger.json,va_oof.npz}.
- bounty_layer1.py staged for both curated cells (community-overlap covariate
  line = answer_score within-question); fires when the stacked A-pass lands.

## 2026-08-17 — BOTH BOUNTY VA LADDERS LANDED (full-speed batch complete)
- **mathse_bounty (math CURATED)**: V .691 / A .696 / VA .723 (within-q
  .680/.686/.711); collapse gate dropped 1 -> 31 criteria.
- **so_bounty (SO CURATED)**: V .739 / A .753 / VA .779 (within-q .744/.746/.779);
  39 criteria.
- **HEADLINE (descriptive): the community vote score predicts the manual bounty
  award WITHIN-QUESTION better than any text instrument** — ascore w-q .776 vs
  VA .711 (math), **.852 vs .779 (SO)**. Consistent with the award-mode audit
  (manual winner = top-scored 64%). The curated construct is heavily community-
  correlated; the live question for T/fused: does text add anything BEYOND the
  vote channel (votes = observed covariate line, NEVER a feature — position-
  ordinal discipline applies).
- Batch of three unified-X ladders now on the board (so_accepted V .639/A .704/
  VA .717 + these two). T columns next (sk1 chains); then VAT V3 fused; then
  (V+A)_new fleets off the dense-vs-VA disagreement slices.

## 2026-08-18 — VAT V3 PUSH + NEW-DATASET SURVEY (user directive; "double up on sk3" OK'd)
- **All three T legs now training in parallel**: mathse_bounty seed2 on sk1 GPU6
  (seeds 42+1 done, ~5.8h each); so_accepted_qtrunc on sk3 GPU1; so_bounty on
  sk3 GPU3 (its T frame = the cell's own title+answer text — differs from
  so_votes' qtrunc frame, recorded). First stacked attempt on GPU0 OOM'd —
  l1ly's tenant GREW to 166GB overnight (stale-survey lesson); resurvey found
  GPUs 1/3/5 fully FREE and both chains run dedicated. sk1's so_accepted leg
  skip-markered to prevent duplicate training.
- **unified_fused_stack.py staged** (VAT V3 = §11 grouped-OOF stack [VA_nl, T
  seedmean] on dense-held-out rows; no criteria-in-prompt arm per the cross-cell
  ruling). Fires per cell as T lands.
- **Survey of the 18 datasets — not-yet-run status:**
  * READY NOW: Kindle Scout (U7: 726 accept/reject verdicts + ~31K-char
    excerpts, campaigns_parsed.jsonl) — CW verdict candidate.
  * SCRAPE LIVE: Reedsy (U6: 18,811/57,832 stories parsed so far) — build on
    settle. RoyalRoad (U1) — data in hand, next build. Jokes removal (U2) —
    wayback at ~63%. WP prospective (#5) — labels start Aug 21.
  * BLOCKED/PARTIAL: McSweeney's (#13, manual TOC step pending); WP removal
    texts (#1, n=423 — pair with kept-side contrast #3); mathlib TMIM (#18,
    bors caveat); GH PR reactions (#16) + Rust highlights (#17) — queued behind
    the CW/humor builds (code field already has 3 columns).
  * DONE/RUNNING: #14/#15 censuses (U3/U4 cells), #4/#9/#11/#12 cued in U2/U5.

## 2026-08-18 — FIRST VAT V3 LANDED: mathse_bounty fused .768
- mathse_bounty T complete (sk1, 3 seeds, ~5.8h each, scored; preds synced to
  sk3). Fused stack on 4,795 dense-held-out rows (join 4795/4795):
  **VA .736 / T .766 / VAT(fused) .768** (.798 eval / .737 test).
- Math CURATED ladder now: V .691 / A .696 / VA .723 / **VAT .768** / T .766.
  T carries +.030 over VA on held-out; fused adds only +.003 over T — thin
  bank/dense complementarity on this cell (contrast BBC's strong swap signature).
  The votes-channel conditioning (does T survive the vote-score covariate?) is
  the F2 stage, queued after all three fused arms land.
- so_accepted + so_bounty T training on sk3 GPUs 1/3; fused fires on landing.

## 2026-08-18 — ALL THREE VAT V3 ARMS LANDED (unified-X batch complete thru VAT)
- so_accepted: VA .742 / T .752 / **VAT .772** (2,440 held-out; fusion +.020 over
  T — the strongest complementarity of the three).
- so_bounty: VA .790 / T .804 / **VAT .811** (4,698 held-out; +.007).
- mathse_bounty: VA .736 / T .766 / **VAT .768** (4,795 held-out; +.003).
- sk3 B200 chains ran 3 seeds in ~2.5-5h vs sk1 A100's 17.5h. Notebook UNIFIED-X
  table updated with full ladders. NEXT: F2 deconf w/ vote-score covariate
  (does text survive the vote channel on curated y?), then (V+A)_new fleets off
  dense-vs-VA disagreement slices; Kindle Scout + RoyalRoad builds in parallel.

## 2026-08-18 — CORRECTION (user catch): math.SE accepted V+A = .632, NOT .574
- The .574 in the master ladder's Math verdict row was the E-REFIT VA (fullgrid
  frame, 2,600 dense-held-out rows) pasted beside the full-population V (.591) —
  frame mixing. Ledger same-frame values: V_nl .591 / **VA_nl .632** (seeds
  .6318/.6325/.6318). VA > V holds; "VA < V pathology" retracted. Fixed in BOTH
  notebook cells (master ladder dict + mathcs comparison). NEVER quote .574 as
  the math verdict VA.
- Correctness criteria on math.SE: present in the bank, JUDGE-SATURATED (means
  .81-.84, modal .84) -> alone-AUC .525-.534 — the ceiling pattern, a measurement
  limit not a construct verdict.

## 2026-08-18 — F2P BALANCED ARM: UNMEASURABLE AT CURRENT INTERSECTION (n=37)
- User asked for 50/50 transition-vs-random VAT. Finding: only **37 rows with any
  F2P/P2F transition survive the join to A-scored PRs** (37/2,686 = 1.4%); the
  balanced set is 74 rows over 108 repos — every AUC is noise (pre-kill checklist:
  minority-class count fails). "F2P is useless" RETRACTED as a conclusion; the
  correct statement is F2P utility is UNMEASURED — the 3,264-signal transition
  universe barely overlaps the 68K A-scored universe. UNBLOCK = A-score the
  transition PRs (a ~3.3K-row Gemma pass, queued) then rerun the balanced arm +
  VAT (dense preds join also pending, row_id=batch:paper_id mapping).
- results/pr_testexec_ladder.json carries the full arms incl. balanced.

## 2026-08-18 — ARTICULATED-SHARE normalization of the residual-gap channel plot
- User challenge: curated≈0 raw gap is confounded with per-channel predictability; the claim
  we actually want is "% of recoverable shared preference that is articulated."
- New readout (notebook cells `articshare0818md`/`articshare0818`, fig_articulated_share.png):
  per cell, CEILING = max(all instruments)−.5; ARTICULATED SHARE = (max(V,V_new,VA,VA_new)−.5)/ceiling.
- Medians: curated .954 / verdict .875 / community .797. Ceilings: community .280 > verdict .244 > curated .197.
- CORRECTIONS to prior readings, on record:
  1. Humor-verdict raw-gap outlier (.20) was an instrument-maturity artifact — raw plot used
     pre-mining V+A (.529); mined V+A_new .703 ⇒ share .875. Never quote the .20 gap without this note.
  2. Peer-review within-field channel contrast (raw gaps .065/.12/.23) is mostly a CEILING
     story (.139/.277/.388), not articulation: shares nearly flat (.640/.606/.577). Retract the
     "strongest within-field evidence that the channel drives articulability" framing (2026-08-18 chat).
- Channel ordering survives normalization but is much softer than the raw plot suggests.
- Caveats carried: mixed frames per master-ladder notes; 5/19 cells lack V+A_new (share = lower
  bound there); ceiling<.15 cells (CW-Wigleaf, peer-oral, math-verdict) flagged ratio-unstable;
  descriptive only, no deconf.

## 2026-08-18 — articulated-share SENSITIVITY battery ("are you sure?" check)
- Numerator variants (max / declared-single / plain-VA), d-prime rescale, MWU+KW rank tests.
- ROBUST: curated median share .95 under every variant (curated cells never had mined columns
  to inflate them; their shares are lower bounds). Ceiling ordering community>verdict>curated robust.
- FRAGILE: verdict-vs-community share separation exists only via mined V+A_new columns
  (plain-VA: .66/.66 identical); no channel ordering significant (best p=.064, n=6-7).
- CELL CORRECTIONS: Humor .703 = campaign frame (†c); peer-community share .577 uses pooled
  T=.888 (NEVER-QUOTE without era band) — in-band share ~.90. Post-correction low-share list:
  CW upvotes .565, peer verdict .606, patents verdict .661.
- Methods note on record: under symmetric label noise, AUC excess over .5 scales by (1−2·fliprate)
  for every instrument equally ⇒ the SHARE is noise-invariant while the RAW GAP is not; channels
  with cleaner labels (aggregated votes) mechanically show bigger raw gaps. Raw-gap channel plot
  is attenuation-confounded; share plot is the defensible cross-channel readout.
- 2026-08-18: articulated-share figure ORDERED curated→verdict→community (user); TikZ wrapfigure landed in paper-3 as fig:articulated-share (figures/fig_articulated_share.tex, committed inner repo; main.tex 2-line wiring left dirty for paper agent). Raw-gap boxplot cell REMOVED from notebook by user — share readout supersedes it; fig_residual_gap_distributions.png retired.
- 2026-08-18 (cont): dense-ceiling panel (T medians .674/.751/.770 cur/ver/comm) + n-confound
  check added to articshare0818. Channel-level n confound REAL (median n 7.5K/28K/75K) but
  cell-level Spearman share~logn −.16 n.s.; extremes cross it (peer-cites 2.4K = largest
  ceiling, homepage 184K dense ≤ bank). U8 scaling ladders (two curated bounty cells, running)
  = decisive test. Gated-vs-open curated gradient noted (oral .640 / N&C .713 vs AoPS .954 /
  homepage .954 / comps 1.0). U12 created: hard-pair stratified share.
- 2026-08-18 (scaling early read, NOT final — 2 of 4 fractions): mathse_bounty (CURATED) dense
  T on the fixed 2,396-row test split, seed-mean: f12.5 .635 / f25 .660 / f100 .727 (f50 mid-run,
  1 seed done; SO chain on sk1 still at f12.5). NO PLATEAU — T still climbing at 100% (~+.033/
  doubling at the top end) ⇒ for this curated cell the dense ceiling IS data-limited, supporting
  the user's n-confound concern at cell level; with more data its share (currently ~.88: VA_nl
  .736 / best .768 same-rows) would fall. Counterweight in same cell: declared-channel deconf
  (d)−(c) = +.013 only — much of the climbing T rides declared channels (answer_score/
  n_answers/char_len); the paper-relevant scaling readout is (d)−(c) PER FRACTION, queued for
  chain completion. Never quote the fraction curve against the fused/same-rows frame numbers.

## 2026-08-19 — PR transition FULL ladder (canonical machinery) + wave-D launches
- pr_transition cell (results/pr_transition_full_ladder.json, va_oof saved): n=7,563 tests-ran
  rows / 184 repos / pos .748; 83 criteria scored, collapse gate -> 49; NA .41 (Title+Diff frame).
- WITHIN-REPO (quotable): **A .765** | V_exec .569 | VA .646. VA << A is an INSTRUMENT note:
  exec features (esp. baseline_n = repo fingerprint) poison within-repo ranking via pooled
  fit — quote A as the text instrument for this cell; never quote VA without this note.
- POWERED transition answer (3,533 transition rows): f2p alone within-repo .448, p2f .425,
  has_transition .423 — test EXECUTION outcomes are at/below chance for merge within-repo.
  Top criterion: "Tests included with changes" .799 within-repo — reviewers reward SHIPPING
  tests, not passing them. Balanced 50/50 arm concurs (A .755 wr).
- dense_coverage = 0: v3 grouped test preds do not overlap this population -> VAT pending a
  dense run on THIS population (queued); VA_new absent (no mining yet).
- Launches: jokes_removal bank scoring sk3/GPU7 (47 rubrics, shard0 NA .308, healthy);
  kindle_scout sk3/GPU3 (relaunch after cd-precedence path bug — first proc died on AFS path);
  SO-bounty scaling seed2 parallel chain sk1/GPU5. All under scaleupD scorer (newest machinery).
- Missing-value whisker (user request): Track-B Z (evalue jsons) is the SPURIOUS-side bound —
  NOT valid as a VA upper whisker. Track-A mass exists only in raw campaign round artifacts;
  estimator (mass x M2-backtest value-per-species) needs a frozen definition + harvest -> U13.
- 2026-08-19 LEAK CAUGHT + RETRACTED same-day: jokes_removal ladder (V .988/VA .992) INVALID —
  source↔class perfectly confounded (removed = wayback fetch: raw HTML entities, torn fragments,
  median 58 chars, no markdown; kept = live API: clean, 105 chars, markdown line structure).
  Caught by post-hoc spot-check; dataset-first rule was violated (population scored before
  text spot-check) — process note filed. A-bank scores (jokes_removal shards) also contaminated
  (judge sees entities/fragments) — bank NOT reusable as-is. Fix path: normalize BOTH classes
  through one renderer (unescape, whitespace/markdown collapse), drop torn fragments, re-probe
  V locally; if fingerprint persists, matched-pipeline rebuild (fetch kept jokes via wayback too).
  Anchor battery direction note (kept .914 > removed .619) still qualitatively fine but
  magnitude untrustworthy. NEVER quote any jokes_removal number from the 2026-08-19 scoring.

## 2026-08-19 — U8 RESIDUAL SCALING CURVES (deconfounded, corrected gboot arg order)
results/u8_scaling_residual_curves.json; frames: fixed eval+test rows, (b)/(c)/(d) arms per
unified_deconf_declared, T = per-fraction seed-mean. PRIMARY (d)−(c) with grouped bootstrap:
- mathse_bounty: +.0012 [−.002,.004] → +.0027 [−.001,.006] → +.0053 [.001,.009] → +.0125 [.007,.018]
  (f12.5→f100; significant from f50)
- so_bounty:     −.0008 → +.0020 → +.0026 → +.0064 [.002,.011] (significant at f100; 2 seeds
  f125-f50, 3 at f100 — seed2 chain still training, rerun readout when it lands)
- READING: the deconfounded dense residual GROWS with dense training data on BOTH curated
  bounty cells — near-zero at 12.5%, significant at full data, roughly doubling per doubling
  at the top end. Raw T grew ~10x faster (+.07) — bank+declared channels absorb most of the
  scaling, but NOT all. Articulated shares therefore fall (slowly) with data; the curated-.95
  headline needs the data-asymptote caveat (already in fig caption as "lower bound").
- First gboot run had (y,g,d,c) arg order — WRONG (signature y,new,old,g); numbers −.27 etc.
  never recorded anywhere; corrected same hour.
- 2026-08-19 (paper org, user): Fig 2 REPLACED with community-channel chart from the notebook
  master ladder (unified frame) — stacked V/V_new + VA/(VA_new increment) bars, U13 whiskers
  (humor +.041 / BBC +.040 / math votes +.011 / peer cites +.058), fields ordered ASCENDING by
  articulability gap (code .02 → peer .16), N&C + patents excluded; old verdict chart moved to
  appendix; U8 residual-scaling figure added to scaling appendix (old Llama-8B sweep kept
  provenance-only, predates current corpora). User message truncated mid-sentence ("Also,
  please ma…") — remainder pending.
- 2026-08-19 U13 whisker BACKTEST (truncation holdout, BBC+jokes, 7 truncations,
  notes/2026-08-19__u13-whisker-backtest.json): bound holds 5/7; both violations are the
  EARLIEST BBC truncations (tiny noisy gains); from mid-campaign the bound holds with
  realized <= 25% of it — the figure's last-round/terminal regime is the validated regime.
  KEY: Track-A mass does NOT fall with more rounds (BBC .45-.51 across 5 rounds; long tail
  of rare-but-real criteria) ⇒ more mining rounds will NOT shrink these whiskers much;
  value flow dries up while mass plateaus (mass ≠ value — same lesson as the A-bank
  degeneracy audit). Calibrated (estimate-not-bound) alternative: terminal-regime realized
  future gain ≈ 0-.005 — a "calibrated tick" could be drawn inside the bound whisker;
  NEVER silently relabel the bound as an estimate.
- 2026-08-19 ROUND 6 CUED (user: shrink whiskers) for bbc_mostread + jokes_community:
  stage-1 slices built (BBC: bank unchanged [V,A,A1-A4], bans r1-r5, assertion/ban loops
  split since r5 kept 0; jokes: 60 rows, median gap .542, oof_fitmine refreshed); 16 sealed
  prompts per cell; CODEX legs RUNNING (laptop, resume-safe); GLM legs RATE-LIMITED —
  weekly quota exhausted, resets 2026-08-22 03:51 — runners checkpoint-stop; RESUME the
  same commands after reset (resume-by-output-file, repeats nothing). Family floor: if
  round must close before Aug 22, add sealed Claude CLI legs (precedent: unified r1);
  NEVER refill GLM slots with another family. Whisker expectation ON RECORD (backtest):
  more rounds likely will NOT shrink the bound much (Track-A mass plateaus); the round's
  real value = extending the flat-gain record + any real criteria it surfaces.
- 2026-08-20 USER POLICY: maintain enough mining rounds that Fig-2 whiskers stay small;
  whisker size = campaign scheduling priority. Display rule: >=2 consecutive flat rounds
  -> calibrated tick quotable in figure, bound to appendix (never silent relabeling).
  Widest whiskers currently: peer-citation .058 (peer_revealed campaign, 6 rounds run,
  r6 candidacy AFTER current BBC/jokes r6), journalism .040 + humor .041 (r6 IN FLIGHT).

## 2026-08-20 — FULL-SPEED VAT sweep + kindle_scout FIRST NUMBERS
- SPUR-BATTERY amended per user: paraphrase intervention REMOVED (kills tacit-real features
  too); battery = invariance/transport + within-vs-between stratum + discriminant loading + panel.
- kindle_scout (CW VERDICT, PILOT-n 726, results/kindle_scout_ledger.json): V .612 / A .729 /
  VA .725 nl; within-genre A .719; genre-identity alone .527 (weak floor — signal is not genre
  composition); battery passed (accept .735 anchors, 0 all-NA). FIRST CW verdict cell since
  RoyalRoad retirement; A > V by +.12 — publisher taste substantially articulable at pilot n.
- LAUNCHES (all verified alive, setsid detach — earlier silent deaths were ssh-timeout
  casualties + a vanished scratchpad file, reshipped): jokes_removal_v2 rescore GPU7 (leak-fixed
  bank, 5+ shards in); pr_transition dense arm GPU3 (seed42 training; split 7,959/171/1,221
  grouped by repo — eval leg SMALL at 171/15 repos, selection noise flagged); kindle L1 done.
- Two old-session background agents stopped by user ("VAT stack dense-above-bank",
  "decorrelated-training planted battery") — noted, no orphan GPU state from either.
- 2026-08-20 OVERNIGHT SUPERVISOR armed on sk3 (logs/overnight_supervisor.sh, marker-gated,
  600s poll, 16h cap): (A) jokes_v2 scoring done -> jokes_removal_v2 layer1; (B) pr_transition
  dense done -> unified_fused_stack --cell pr_transition -> kindle dense chain on GPU3 (bs8/
  ga2/3ep for PILOT-n) -> kindle fused stack. CELLS registry extended (pr_transition,
  kindle_scout, jokes_removal_v2). Also still running: SO seed2 (sk1 GPU5), GLM r6 legs
  resume Fri 03:51. Morning harvest: jokes_v2 ladder (leak-fix verdict numbers), PR VAT
  column, kindle VAT column, U8 final curves.
- 2026-08-20 U9 DESIGN (user): two arms — (1) metadata-anchored subfields (tags/sections/
  genres/repos), per-subfield ladder w/ existing bank -> mine only where residual is large;
  (2) residual-matched subfamilies: match items on existing-feature profile, find DISCORDANT-y
  matched sets, cluster them, test whether newly-mined features (not existing ones) separate
  the clusters = novel subfamilies. Permutation null per #60 recipe.
- 2026-08-20 (cont): jokes_removal_v2 VAT chain armed (waits for v2 scoring -> dense splits
  by month -> 3-seed Llama-8B max_length 512 -> score -> fused stack; GPU preference 7).
  Jokes COMMUNITY VAT already exists (.760 fused) — this fills the VERDICT VAT. SPUR-BATTERY
  readout (tests 1-3) launched on the 13 word-probe channels; panel deferred to GLM window.
- 2026-08-20 SPUR-BATTERY first pass: T1+T3 informative, T2 DEGENERATE (BBC section constant
  "news") — rerun with T2 = within-DAY (primary; era-cohort channels are day-constant -> ~.5)
  + TF-IDF KMeans-30 topic proxy (secondary), deviation recorded. First-pass signals: most
  channels weak-alone (.50-.54); Leisure-Vertical Discount pooled .584 spread .086 q-load .378
  (transport-UNSTABLE but QUALITY-loading = MIXED candidate); Trump-Election cohort spread
  .057 (unstable, as predicted); Comment-Enabled Marker dead (constant). Nuisance-factor PC1
  weakly loading everywhere — refine to per-covariate max |corr| in panel cards if needed.
- 2026-08-20 SPUR-BATTERY COMPLETE (corrected T2; results/spur_battery_wordprobe.json):
  DECLARED CRITERION MET — clean bimodal transport split: era-unstable {Trump .057, Forensic
  .035, Prospective .043, Leisure .086} vs stable-weak {8 channels, .004-.020}; Comment-Enabled
  dead (constant). KEY REINTERPRETATION: within-day ≈ pooled for ALL channels — cohort channels
  are NOT between-day composition; they are ERA-LOCAL TOPICAL ATTENTION (predictive within-day
  during their window, absent outside). Leisure (.378) + Prospective (.217) quality-loading =
  MIXED per spec, never auto-verdict. Panel (3-family) waits for GLM window Fri; absorption
  test (channels added to (c) arm vs BBC +.046 residual) = morning list.
- 2026-08-20 jokes_removal_v2 VAT COMPLETE (chain found GPU4 free): same-rows held-out
  (n=3,562 = KNOWN-month strata only — unknown-month stratum entirely in train by grouped
  hash, distribution-shift caveat DECLARED): VA_nl .895 / dense T .916 / fused VAT .914
  (stack ≈ T, wash). Full-pop pooled ladder: V .603 / A .662 / VA .701; within-month .90;
  month-identity .857 (metadata confound: missing created_utc ↔ wayback ↔ removed — declared,
  unfixable without capture metadata). Articulated share on held-out frame ≈ .95 — humor
  VERDICT highly articulable, consistent with removal = offensive-content/repost signal.
  U2 verdict leg: INSTRUMENTED end-to-end (v1 leak retracted -> v2 clean same day).
- 2026-08-20 U8 COMPLETE (3 seeds both cells, all fractions; appendix fig refreshed):
  deconfounded residual (d)−(c) by fraction — math .0012/.0027/.0053*/.0125* (sig from f50);
  SO −.0001/.0007/.0038/.0064* (sig at f100). Both monotone-growing; raw T grows ~10x faster.
  HEADLINE: the dense residual beyond bank+declared channels GROWS with dense training data
  on both curated cells — small but systematic; articulated shares decline slowly with data.
- 2026-08-20 pr_transition dense + fused COMPLETE — WEAK/UNSTABLE, instrument-limited:
  dense test AUCs .536/.466/.581 (seed2 sub-chance), same-rows T .575 / VA .609 / fused .582
  (fused < VA — stack hurt by noisy T). ROOT CAUSE DECLARED: dense max_length 1024 tokens on
  Title+Diff = frame-starved (Gemma bank reads 11,600 tok; within-repo A .765). For long-doc
  cells T@1024 is NOT a ceiling — record as instrument note, NEVER quote this T as the cell's
  dense bound; the cell's headline instrument stays the A bank. Held-out n also small (1,563,
  17-repo test leg). If a real T is wanted here: longer-context dense arm (4K-8K) = new run.

## 2026-08-20 — humor deep-dive (user session)
- Humor gap row CORRECTED (caption cells dropped as quarantined): n=3 {staff picks .029,
  removal-v2 .021, upvotes .039} mean .030 sd .009 — still lowest-tier with code/math/journ.
- HARD-SLICE (results/jokes_removal_v2_hard_slice.json): removed 11,596 -> hard 2,452 (21%;
  filters: lexicon 939, near-dup 923, length-out 8,833 — LENGTH dominates). Ladder: hard VA nl
  .825 / within-month .898 vs easy .803/.952 — the "hard" (rule-clean) removals are ~AS
  articulable as easy ones (−.05 within-month). "Too easy" concern ANSWERED at battery level;
  residual caveats: battery ≠ ground-truth reason; latency (prospective collector) = true
  auto/human separator, accumulating.
- CHANDRASEKHARAN (datasets/prior_norms/reddit-norm-violations): 99-sub study INCLUDES r/funny,
  Showerthoughts, tifu; NOT r/Jokes. Data = macro-norm-VIOLATION removals only (by violation
  type: slurs/hate/attacks) = labeled-EASY stratum — useful as reason-labeled control, NOT a
  hard-verdict source.
- NEWSJACK: raw jsonl = forum CHATTER-contaminated (7/8 spot-checked rows are meta-discussion,
  not submitted jokes) — caught PRE-scoring (dataset-first). VAT blocked on an LLM extraction
  pass (segment submitted material from posts) — queued as the pipeline's stage 0.
- SNL cut-for-time: STRONG design (producer accept/reject at matched production; 2,549 aired /
  87 cut) but 38/2,636 transcripts — transcript fetch LAUNCHED (laptop, resume-state).
- GPU BAN: sk3 GPUs 3 AND 7 off-limits few days (user); kindle VAT rebuilt (ungrouped row-hash
  split, deviation recorded — 726 independent campaigns, genre=covariate; grouped split gave
  24-row test leg) + requeued on excl-3/7 watcher with ATOL override; stale watchers killed.
