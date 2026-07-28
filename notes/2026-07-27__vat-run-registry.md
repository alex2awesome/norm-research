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

## DENSE CHAIN COMPLETE (2026-07-28 00:27; all test-split interim, eval pass running)
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

## ROUND-2 DENSE CHAIN LAUNCHED (2026-07-28): cap_finalist → cap_crowd → cw_full
(WritingPrompts clean grouped, full data) — after eval pass drains. Watcher armed.

## In-flight jobs right now (sk3)
- Dense-standard chain GPU1: nc_responded → peer curation → peer verdict (watcher armed).
- Caption A-bank scorer GPU2: 5 shards, 364 rubrics × 18,838 captions (watcher armed).
- After chain: clean-eval scoring pass over all 6 runs (test-split numbers are interim).

## Task-list index
#43 V1 captions multi-y (▶) · #44 V2 math.SE votes · #45 V3 legal citation-pct ·
#46 V4 dense-standard re-runs+queue · #47 V5 CW/humor mature banks · #48 V6 SO votes ·
#49 V7 patents forward cites · #50 V8 N&C co-signing.
