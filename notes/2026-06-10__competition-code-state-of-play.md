# Code track — state of play (updated 2026-06-11 PM)

## 2026-06-11 PM — three-track push (user: "do all 3")

1. **Round-9b factory fixes DONE**: out-of-corpus reject augmentation now EXISTS (prep_repo.py:1955-2240, `--augment-rejects`, cap 10, append-only); accept-gate counts one no_change_all_pass+verified-rejected as clean reject (capped +1, flagged); R9-F2 not_installable real install; R9-F3 requires-python op-aware padding; R8-F1/F3/F5/F6 (attrs pin, Go gold/silver veracity parity, gopath build-time deps, go.mod version). test_round9b_fixes.py 119/119; all suites green. pilot_log.md "Round 9b" at line 2006. nuxeo/OpenPype/fairlearn all unblocked for the next supervised batch.
2. **SO js+sql v2 DONE** (`outputs/v2_analysis/so_js_sql_build_2026_06_11.md`): combined SO balanced = **151,222** (py 59,996 / js 60,004 / sql 31,222), all q-floors pass first build (0.474/0.471/0.448), answer margins 0.670/0.648/0.661, pairwise 270,487. js stricter than py (fenced-code-block filter — mirror stores MARKDOWN not HTML). sql widening option documented (dialect-only tags ≈ double).
3. **CF rebuild cell** (`outputs/v2_analysis/cf_rebuild_2026_06_11.md` + `cf_rebuild_cell/`): v2 render scrape complete; section-aware extraction + execution-gated editorials → 1,543 joinable problems (old cell: 155); 2,255 pairs/1,051 problems, bank scored (139 metrics). L1 labeling resumed after session-limit kill (145/2,255 at restart; headless sonnet workers, append-only results). Then LR/RF/ens AUC closes the competition table.

oauth2-proxy 52-PR local sweep still running (PID 28346) with self-finalizer.

## WIND-DOWN SNAPSHOT (2026-06-11, limit hit mid-plumbing)

**Still running WITHOUT tokens (survive everything):**
- oauth2-proxy 52-PR local sweep: nohup PID 28346 on laptop; 8/52 done, 5/8 strong incl. first-ever Go `fix` verdicts; on finish run `monitor_health` + `accept` per the commands at the end of factory/pilot_log.md
- fairlearn sk3 verdict runner (check sk3:/lfs/.../pr_factory/batch_runs/fairlearn/verdicts.jsonl; was 45/47)
- CF render scrapers v1+v2 on sk3 (datasets/codeforces_delta/render_scrape*_log.jsonl)

**Interrupted mid-task by the limit (resume first):**
- phys_id→denylist plumbing in run_sif.sh/run_local.sh (the gold-tier gap; agent died partway — all files left syntax-consistent, suites green, but the plumbing is INCOMPLETE: verdicts still carry phys_id=None; finish per the batch-1 section of pilot_log.md)
- batch-1 completion: nuxeo-drive + OpenPype re-runs (their blockers R8-F1/F2 are FIXED), chia from phase D, fairlearn accept audit

**Queued on user word:** SE ~100K build (age-gap matching + [javascript]/[sql] slices — all scripts exist, see so_python_v2 dir); CR.SE claim-pipeline 3 fixes; round-9 factory fixes (R8-F1 attrs pin, R8-F3 go veracity, R8-F5 gopath build-time deps, R8-F6 go_version derivation)

**Headline numbers as of wind-down:** competition AC 0.73/CC 0.73/LC 0.57 (CF awaiting scrape-rebuild); PR correlation answered (tekton-only effect, verdicts→per-repo V-features); CR.SE v2 floor-clean n=14,302 margin 0.590; SO[python] v2 STRICT n=59,996 margin 0.670 (+pairwise 138K); factory: gold tier proven, integrity proven, supervised-batch conditional GO.

Framing (do not drift): editorial-similarity is a PROXY label for good taste/style. Headline number = candidate-only bank AUC (predictor never sees editorial/cosine/problem). Relational features forbidden at prediction time.

## Honest numbers (grouped splits) — TABLE COMPLETE 2026-06-11 PM

| Platform | LR | RF (new best) | Ens(RF+LR) | Notes |
|---|---:|---:|---:|---|
| AC | 0.696 | 0.721 | **0.731** | after L1-discipline relabel (0.597 with old labels) |
| CC | 0.685 | 0.718 | **0.729** | clean 401 cell (48 editorial-derived candidates EXCLUDED; old headline 0.729 was inflated) |
| CF | 0.610 | 0.619 | **0.627** | REBUILT cell: rendered-scrape editorials, exec-gated, 1,543 joinable problems (was 155), n=2,255, pos 0.870, inter-run label agreement 91.8% |
| LC | 0.554 | 0.567 | **0.568** | organic rebuild (copy-contamination removed) |

Platform heterogeneity (0.57–0.73) stands as a finding. CF detail: `outputs/v2_analysis/cf_rebuild_2026_06_11.md`.

## Key findings today

1. **Label discipline >> prompt design**: strict-R4 "L1" relabel (aggressive boilerplate strip) lifted AC +0.10. L2 (same-realization) decompresses pos rate but does NOT lift AUC.
2. **CC was inflated** by `editorial_code_as_candidate` rows (48/449, all label=1, 41% of high-cosine decile). RULE: exclude editorial-derived candidates from all eval cells.
3. **Language match explains ~0** of platform gaps; C++ metrics a600-a614 add 0 AUC (coverage parity only).
4. **CF editorial code was lost at scrape time** (spoilers render client-side; static HTML has no <pre>). JS re-scrape running on sk3.
5. **LC caveat**: the 2K cell's positive candidates are largely renamed editorial COPIES from lc_discuss — error anatomy says 11/12 errors are this pattern. Candidate-only metrics can't see copy-ness. Either document 0.68 as LC's intrinsic ceiling or rebuild LC pairs from organic (taco-verified) submissions.
6. AC/CF cells verified CLEAN of editorial-derived candidates.

## In flight (survives laptop shutdown — all sk3 nohup)

- CF render scrape v1 (old-era 680 blogs): PID 3634835, ~10h left, log `datasets/codeforces_delta/render_scrape_log.jsonl`
- CF discovery v2 (917 modern contests, 100% tutorial hit rate) + render v2 + supervisor: PIDs 3682170 / 3694796 / 3694799, ~16h
- Modern blogs ~81% contain code in spoilers (vs 5.3% in old text dump)

## In flight (LOCAL agents — die if laptop shuts; relaunch from disk state)

- Dense-ceiling agent (#162): frozen bge probe + ModernBERT FT, 4 cells → `outputs/v2_analysis/dense_ceiling/`
- CC L1 relabel: 2 Sonnet shards over `outputs/v2_analysis/cc_l1_relabel/shards/` (390 pairs) → results/ (append-only, resumable)
- ac2→v1 distill chain orchestrator

## Next steps (ranked, from headroom audit `outputs/v2_analysis/headroom_audit/report-in-agent-result`)

1. DONE-ish: RF + rank-avg ensemble = new baseline (mechanical, +0.03-0.05)
2. CC L1 relabel (running) → expected CC ~0.74-0.76 under RF
3. After CC/LC relabels: retrain distill cross-encoder on L1 labels → bulk-label 100K+/platform → dense reward model = real dense ceiling C
4. CF chain: re-scrape → extract from rendered <pre> → EXECUTION-GATE editorials (only keep those passing own tests; text-dump version had only 15% pass) → fresh pairs → L1 labels
5. New metrics targeting AC's feature gaps: algorithmic-paradigm (closed-form vs iterative), complexity tier, DS-role (speculative +0.01-0.03)
6. LC: decide ceiling-vs-rebuild (see caveat 5 above)
7. SKIP: LC relabel (~0 EV), C++ metrics in superset (0)

## Dense ceiling results (landed 2026-06-10 late night; `outputs/v2_analysis/dense_ceiling/report.md`)

| Cell | Bank (best) | Frozen bge probe | ModernBERT FT |
|---|---:|---:|---:|
| AC (L1 labels) | 0.721 RF / 0.731 ens | 0.576 | 0.690 |
| CC | 0.718 RF / 0.729 ens (clean cell) | 0.666 | 0.680 |
| LC | 0.678 RF | 0.589 | 0.593 |
| CF | 0.545 LR | 0.526 | 0.515 |

PROVISIONAL CONCLUSION: at current cell sizes, the static bank (RF/ensemble) matches or EXCEEDS the dense estimate on every platform — the articulability gap for competition code is ≈0 (opposite of press releases, where dense 0.71 >> rubrics 0.58). Competition-code taste appears highly articulable. CAVEATS: (a) the agent's "dense exceeds bank on AC" claim compared dense-on-L1 vs bank-on-ORIGINAL labels — corrected here using bank-on-L1 numbers; (b) ModernBERT-base FT on ≤2.5K pairs is a weak/conservative C estimate — the real ceiling test is the scale-out (distill → 100K bulk labels → dense reward model); (c) dense CC ran on the 448 cell incl. editorial-derived pairs.

## Final tiered conclusion (2026-06-11, after paradigm-metrics + exec V-layer rounds)

- Paradigm/complexity/DS metrics (a650-a664): Δ≈0 on AC/CC (collinear with lizard/radon bank features), +0.009 LC, 0/36 targeted errors flipped. `outputs/v2_analysis/paradigm_metrics/`
- Exec V-layer (8 features incl. pass-rate): Δ ≤ +0.007 everywhere; V-only at/below chance CC (0.43)/AC (0.50, verdict 99.3% constant — "no V-layer"), 0.53 LC. `outputs/v2_analysis/exec_vlayer/`
- VERDICT (3 independent rounds + dense agree): bank RF/ens ~0.73 IS the candidate-only ceiling for AC/CC; remaining error variance is relational (which approach the editorial picked), structurally inaccessible to candidate-only predictors. V ⊥ A confirmed on current cells.

## Other tracks

- CR.SE: balanced_v1 built (Math.SE v3.3 port; question floor dead 0.458; position floor 0.653 SURVIVES — needs Math.SE-window decision on handling). Next: bank+dense ladder on balanced data.
- GitHub PRs: PAUSED by user. Resumable: rdk 137/189, tekton+iceberg staged+imaged. Headline analysis pending: verdict↔accept/reject cross-tab across ≥3 repos.
