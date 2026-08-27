# Press-release newsworthiness — progress report

*Task: predict whether a corporate press release is picked up by tracked news outlets.
Status as of 2026-07-03: **the corpus is a V/A/taste showcase once the label is thresholded at
k≥3 outlets** (broad/consensus coverage). The original `≥1 outlet` label was 88% single-outlet
noise and produced a misleading "no signal" null. **Decision: model at k≥3.** This report is the
single consolidated reference; detailed chronology lives in `notes/2026-06-25__press-release-audit.md`
(§5a–§5n).*

---

## 1. Corpus & label

- **Source corpus:** 128,131 press releases → deconfounded to **72,315** clean English, length-capped
  rows (`press_release_deconfounded.parquet`, cols: `id, judgement, text, model_len, company, group,
  split, year, topic, topic_label`). 17 tracked coverage outlets.
- **Label:** `judgement = 1` iff ≥k distinct outlets covered the PR. Originally k=1.
- **Splits:** stable **company-hash** 80/10/10 (`group == company`, **0 of 6,967 companies straddle
  a split**). All evaluation is **company-grouped** (StratifiedGroupKFold) + within-topic.

## 2. The key finding — label threshold k flips floor → showcase

The `≥1 outlet` label is **88% single-outlet pickups** (28,735 of 32,789 covered PRs covered by
exactly one outlet) — i.e. dominated by wire-republication / automation / single-editor noise.

| k (outlets) | positives (deconf 72k) | balance | dense grouped AUC |
|---|---|---|---|
| ≥1 (old) | 32,789 | 1:1.2 | 0.584 |
| ≥2 | 4,054 | 1:17 | 0.675 |
| **≥3 (chosen)** | **1,478** | **1:27** | **0.705** |
| ≥4 | 711 | 1:101 | ~0.70 |

Raising the bar to "consensus coverage" lifts the dense ceiling 0.584 → 0.705 (within-topic too:
0.584 → 0.705, so **not** publisher/topic confound returning). **Decision: keep k≥3.**

## 3. The V/A/dense battery

### At k=1 (the old, noisy label — "floor" picture)
| layer | grouped | within-topic |
|---|---|---|
| V — numbers only ($,%,counts) | ~0.51 | ~0.50 |
| V — cheap-feature bag (74) | 0.554 | 0.534 |
| A — 309 rubrics (70B judge) | 0.543 | 0.515 |
| dense — bge-m3 embeddings | 0.584 | 0.556 |
| relational — novelty/density/prominence/claims | all ≈ 0.50 | all ≈ 0.50 |
| dense + all relational stacked | 0.583 | 0.555 |

Everything ≈ 0.55–0.58; no articulability gap (A≤V); no tacit residual (dense=cheap); relational
adds nothing. Covered vs not-covered PRs are **identical** on announcer prominence (log-pageviews
5.66 vs 5.65) and verifiable-claim count (13.4 vs 13.4). → *floor case.*

### At k≥3 (the chosen label — **showcase** picture)
| layer | grouped | within-topic |
|---|---|---|
| V (cheap counts) | 0.628 | 0.627 |
| **A (40 rubrics, 70B)** | **0.648** | **0.645** |
| dense (bge-m3) | 0.705 | 0.705 |

**Clean V < A < dense ladder.** Articulated rubrics add ~0.02 over cheap counts; dense adds ~0.06
over A (a real tacit residual). This is the showcase structure the k=1 analysis missed.

**Top newsworthy-discriminating rubrics** (univariate AUC, k≥3): Company boilerplate completeness
(0.585) · Uncertainty communication (0.573) · Limitations/caveats (0.571) · Calls-to-action/media
facilitation (0.557) · ESG/sustainability disclosure (0.546) · Lede/5-Ws (0.544) · Original research
as newsworthy (0.534) · Investor clarity (0.533) — i.e. **professional PR-craft dimensions**.

## 4. Infilling machinery (global + ctree/MOB), k=1

Ran both engines per `methods/metrics_tree_infilling/AGENT_PLAYBOOK.md`: 28-metric code V-bank,
70B judge, GLM-5.2 proposer, company-grouped discover 989 / guard 585 / test 626, curated
z=[topic, length], n_perm=999.
- Baseline bank guard 0.499 (≈chance) / test 0.549.
- **Global:** scored ~3 GLM-proposed candidates via 70B; **kept 0** (none beat +0.03 gate).
- **ctree/MOB:** **stumped** (no topic/length split survived Bonferroni); root gap, GLM proposed
  "formal_business_register" → 70B scored auc_gain **−0.009** → dropped.
- **MCC certificate:** V_bits=+1.05 on n=626 → **N_lower=1** (≈1 metric-equivalent); **N_upper
  right-censored** (no dense scaling curve). Verdict = stump, not saturated.
- *Note: infilling was run at k=1 (pre-threshold reframe). Worth re-running at k≥3 now that signal
  exists — a region-specific (i) residual may surface.*

## 5. GEPA prompt optimization (Stage 2, k≥3-era)

`make_roles_mixed`: **Gemma-4-32B judge** (served, gemma4 env) + **GLM-5.2 proposer/reconstructor**
(z.ai `zai_anthropic`), objective = `fidelity_scalar` (reconstruction R + reliability + …).
6 viable rubrics × 2 rounds:
- **1 accepted:** "Lede uses concrete details" seed_fid 0.635 → **0.746 (+0.11)**.
- Others ran but mutants didn't consistently beat the cross-family acceptance gate (some regressed).
- POC scale (head-of-file 40 rubrics, 2 rounds, GLM-quota-bound). Machinery works end-to-end.

## 6. Per-outlet selection profiles (descriptive, k=1)

Outlets have distinct *selection* profiles (but presence ≠ coverage predictive power):
- **Financial desks** (cnbc/wsj/foxbusiness/marketwatch) select number-dense PRs (cnbc $ 1.42×,
  foxbusiness % 1.36×, wsj numbers 1.29×).
- **WSJ** = hard-financial signature (number-dense, fewest quotes 0.59×, least wire-fed 0.33× — scoops).
- **MarketWatch** = most wire-distributed (1.89×; republisher / Zacks).
- **TechCrunch** = startup/funding (fewest quotes 0.50×, most crowded announcement space 1.72×).
- **General outlets** (cnn/cbsnews/wapo) select number-light feature PRs.
- Prominence and novelty ~flat across all outlets.

## 7. Deconfounding (done once, foundational)

The raw dense 0.71 was mostly confound: publisher identity AUC **0.673**, topic **0.610** (each
near 0.71 alone). Honest deconfounded linear ceiling ~0.584 grouped / ~0.546 within-topic. Confounds
addressed: company-hash split, topic-balancing, language filter, length-cap, extraction-failure
boilerplate (~7% of rows, weakly label-skewed → noise). Data-quality: ~7.2% rows are extraction-failure
boilerplate; the gzip-truncated canonical CSV (only 41,607/128,131 readable) was reconstructed from
intact sources (not deleted).

## 8. Open items / next steps

1. **Re-run infilling (global + ctree) at k≥3** — signal now exists; a region-specific (i) residual
   may finally surface (it was absent at k=1).
2. **Scale GEPA**: coverage-selected metrics (not head-of-file), more rounds, then re-score the
   optimized A-layer — does it close the dense−A gap (~0.06)?
3. **A-layer NA rate 65%** — applicability-weighted scoring (vs impute-to-0.5) should sharpen A.
4. **k≥3 imbalance 1:27** — currently handled (stratified+grouped CV, AUC threshold-free); consider
   a second k≥2 run (n=4,054, 1:17) as a more-powered companion.
5. MCC certificate at k≥3 (not yet computed) — with V<A<dense, the dense-dominance gate may finally
   pass, giving a non-right-censored N_upper.

## 9. Artifacts

Deconfounded dataset (synced locally): `datasets/press-releases/press_release_deconfounded.parquet`.
Scripts + scores on sk3 at `/lfs/skampere3/0/alexspan/norm-research/datasets/press-releases/`:
- `run_A_layer_k3.py` → `pr_A_k3_scores.npz` (40 rubrics × 2956 items, k≥3).
- `run_gepa_pr.py` (Gemma+GLM GEPA), `build_pr_A3.py`, `build_pr_A2.py` → `pr_A_scores.npz`.
- `dense_embed.py` → `pr_dense_emb.npz` (bge-m3, 72,315×1024).
- `build_relational_offline.py` → `relational_offline.parquet`; `build_pr_claims.py` → `pr_claims_scores.npz`.
- Outlet grid: `outlet_category_grid.tsv`. Rubrics: `rubrics.jsonl` (309), `online-rubrics/` (49,612).
Full chronology + tables: `notes/2026-06-25__press-release-audit.md` §5a–§5n.
