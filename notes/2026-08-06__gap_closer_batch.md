# Gap-closer batch (task D6) — four diagnostic verdicts

Date: 2026-08-06. Parent: notes/2026-08-05__taste-decomposition-design.md §10
(full-grid drive plan, "GAP-CLOSER batch" line) + notes/2026-08-05__layer3-closure-prereg.md
(FROZEN prereg). Four small, sequential, diagnostic-only jobs — no rescoring, no new
Layer-3 mining rounds. Each verdict below is a stop-and-report, not a forced fix.

---

## 1. PRESS GATE CHECK — verdict: GATE CHECK ITSELF WAS ALREADY DONE AND CORRECT; the
"gap" the task brief quotes is a false premise (comparing two different protocols)

The apparent gap (V .6168 vs published .628, A .6691 vs published .648) is **not** the
gate-reproduction numbers — it's `methods/taste_decomposition/results/press_verdict_layer1.json`'s
`linear` block, which is deliberately computed under a DIFFERENT, campaign-uniform CV
protocol (unshuffled `GroupKFold(5)` outer folds + pooled-OOF AUC, the same protocol every
other Layer-1 cell uses) rather than this cell's own historical splitter. That substitution
is disclosed in-script and was necessary because this cell's V+A stack was **never fit
before** (`VA_first_fit: true`) — there is no published VA protocol to mirror, so the
campaign standard was used for uniformity, incidentally also changing the V-only/A-only
numbers away from their historical values.

The **actual** gate check — `a_gate_live()`/`v_gate_live()` in
`methods/taste_decomposition/press_verdict_layer1.py`, which exactly mirror each historical
script's own splitter (`StratifiedGroupKFold(5, shuffle=True, random_state=0)`) — reproduces
the published numbers correctly, to within `GATE_TOL=.006`:
- A: live .6481 vs published .648 (diff .0001) under sklearn 1.8.0/1.9.0.
- V: live .6307 vs published .628 (diff .0027) under sklearn 1.7.2.

Both PASS, **each under a different, independently-verified, historically-correct sklearn
version** — a documented, audited LANDMINE (module docstring in `press_verdict_layer1.py`):
`StratifiedGroupKFold`/`GroupKFold` fold assignments are not reproducible across sklearn
releases even at identical `random_state`. A ran on sk3's `gemma4` env (sklearn 1.9.0, per
`run_A_layer_k3.py`'s July-2 log); V's lost source script ran on this laptop's base Python
(sklearn 1.7.2, dated June-26) — the two published numbers were never computed in one common
environment, and demanding one would over-fit the gate to an assumption the historical
record doesn't support. **GATE_PASSED_PROCEED is legitimate, not a false pass.**

**Canonical numbers:** published V .628/A .648 (grouped) remain canonical for the cell's own
historical claim (confirmed reproduced, sklearn-version-pinned per component). The campaign's
`linear` ledger (V .6168/A .6691/VA-first-fit .6712) is a **separate, intentionally-different**
computation — canonical only as the input to *this cell's own* Δ_interact/Δ_beyond pipeline,
never as a stand-in for .628/.648 in a cross-cell table.

**Is VA .6712 (first-fit) trustworthy?** Qualified yes, with three flags to carry whenever
quoted:
1. V features are **bytecode-reconstructed** (source script `codex_pr_vrescue.py` lost;
   reconstruction sanity-checked on the full 72K-row corpus at .562 vs the audit's own
   quoted .554 — close, not exact; small unquantified reconstruction uncertainty on the
   V side only).
2. `T_provisional=.679` is itself flagged provisional (population-mismatched vs this cell's
   exact 2,956-row A/V population, not a same-rows rescore).
3. `VA_nl=.701 > T_provisional=.679` → `Delta_beyond_provisional = -.022` — this cell needs
   **no Layer-3 closure work** regardless of the exact VA_lin precision (design doc §7 item 3
   already flags this: "press... NOTE VA_nl>T there → no closure needed, Layer-2 only").

Use VA .6712/Δ_interact=+.030 (bootstrap CI [.017,.057] group-level, [.022,.053] row-level,
both clearly >0) as the cell's internal Δ_interact anchor — it is well-supported and internally
consistent. Do not headline any Δ_beyond number from this cell without the `T_provisional`
flag. **No rescoring done, per instructions.**

---

## 2. MATHLIB T VERIFY — verdict: **T=.770 is UNUSABLE, NEEDS-RERUN** (not merely
"split-unverified" — it turns out to be a different, non-canonical population entirely)

`methods/taste_decomposition/results/mathlib_verdict_layer1.json` documents the current
V′/A/VA population (n_pool_VA=7,921, n_eval_VA=811) as using a **SINGLE fixed train/eval
split from the parquet's own `split` column, explicitly "stratified NOT grouped" by area**
(group_column = top-level Mathlib area, 31 distinct). So even the cell's own canonical
numbers do not use an area-disjoint split.

But T=.770 itself does not trace to any reproducible number in the repo at all. Searched
`datasets/math/VAT_CLOSURE.md` (canonical math-vertical writeup) and
`notebooks/data/math_vat_summary.json` (the official portfolio-summary artifact). Two real
candidates, neither equal to .770:
- **Official canonical dense-C**: topic-residualized TF-IDF proxy = **.736** (matches
  `math_vat_summary.json`'s `mathlib.C` field exactly). This IS computed on the current
  canonical de-confounded slice (`accept_reject_clean_deconf.parquet`, n=7,956, the same
  population the Layer-1 V′=.680/A~.52/VA=.668 numbers use) — but it's a linear TF-IDF proxy,
  not a genuine neural dense/Llama run, and the area confound is handled by *residualization*
  (raw C .750 → topic-resid C .736), not by area-disjoint splitting.
- **Full-data Llama-8B LoRA dense run** (title+diff, 28,424 train; VAT_CLOSURE.md lines
  232-234): eval-split **.7315**, test-split **.767** — VAT_CLOSURE.md's own text flags the
  test number as "the ~0.035 gap is checkpoint-selection optimism" (i.e. select-on-test).
  .767 rounds close to .770, but this run predates the 2026-06-25 canonical-clean-slice audit
  by over a week and ran on the OLD, non-deconfounded ~35,796-row pool — **a different
  population from the one the current V′/A/VA Layer-1 numbers are computed on**, not merely a
  different split of the same rows.

**Verdict: T=.770 is unusable as currently cited.** Its likely source (.767) is (a) the
audit's own flagged-optimistic test-split number, not eval, and (b) computed on a
population that predates and does not match the canonical de-confounded slice this cell's
V/A numbers use — so any same-rows/leakage question about it is moot; it was never same-rows
to begin with.

**Rerun spec (not run, per instructions):** Llama-8B LoRA `dense_standard` recipe
(`methods/dense/run_dense_standard.sh`) on `accept_reject_clean_deconf.parquet` (n=7,956,
matching the exact Layer-1 population), with an **area-grouped** split (GroupKFold or
held-out areas over the 31 `area` groups) replacing the current label-stratified split,
**select-on-eval** per the campaign dense standard (this cell's own history already shows a
~.035 select-on-test inflation — do not repeat it), reporting eval AUC as canonical and test
as a secondary/flagged number. Until that reruns, the best *currently available* dense-side
comparator is the topic-residualized TF-IDF proxy **C=.736**, quoted explicitly as "TF-IDF
proxy, not neural dense" — never .770.

---

## 3. CODE COMPETITIONS LAYER 1 — verdict: the TRUE bank matrix behind "AC strict-L1 bank
ens .731" is **NOT FOUND locally** — a background search+run pass found and used the WRONG
(smaller, earlier, non-matching) matrix; corrected here after independent verification

**Two-pass job.** A background search agent located artifacts and — per this job's own
instructions ("if found: run the frozen Layer-1 protocol") — went ahead and ran it,
producing `methods/taste_decomposition/code_competitions_layer1.py` +
`methods/taste_decomposition/results/code_competitions_layer1.json`. On review, that run's
"gate soft-fail" framing turned out to rest on a **wrong premise** about which matrix
produces the published number. Correcting that premise changes the verdict materially, so
this section documents the correction (the JSON now carries a
`VERIFICATION_ADDENDUM_post_hoc` field with the same detail; kept as an honest record rather
than deleted).

**What the published `AC strict-L1 bank ens .731` (registry: "CODE curation = competitions
DONE ... AC strict-L1 bank ens .731 / dense FT .690") actually traces to:**
`outputs/v2_analysis/cf_rebuild_2026_06_11.md` line 148 and
`notes/2026-06-10__competition-code-state-of-play.md` lines 32/74 — both give the identical
row **LR .696 / RF .721 / ENS .731, n=2,495, pos rate .836** ("L1-discipline relabel"). n and
pos-rate here are **byte-identical to the dense FT population** (`outputs/v2_analysis/
dense_ceiling/report.md`'s AC-strict-L1 row: n=2,495, pos=0.836, FT 3-seed mean .6896 ≈
registry's .690) — so **the published bank/dense comparison IS same-population at the
aggregate level**; it is not the "999 vs 2,495" mismatch the first pass reported.

**But the per-example bank-score matrix for that exact 2,495-row L1-relabeled population
cannot be located anywhere locally**, despite checking every `outputs/v2_analysis/**/*.parquet`
with `aNNN_score`-style columns (candidate-only bank convention). What DOES exist —
`outputs/v2_analysis/comp_fourplatform_cells/ac_bank_scores.parquet` (n=1,000/999-joined,
what the first pass used as "the" bank matrix) — is a documented, EARLIER, SMALLER pass:
`outputs/v2_analysis/comp_gap_audit_2026_06_10.md`'s own data-sources table says "2495 ok
shard labels in total, 999 land in the bank-scored eval cell" — i.e. bank scoring was only
ever run for 999 of the 2,495 *originally-labeled* (pre-L1-relabel) AC pairs, a resource
choice from an earlier stage. Direct evidence this 999-row matrix is not the .731 source:
`outputs/v2_analysis/ac_l1_relabel/auc_compare.json` already ran the natural check — same
999-row bank-scored subset, L1 labels substituted in — and got **LR=.700**, not .696/.721/.731
under any leg. (The 999-row live rerun in `code_competitions_layer1.json` gets LR=.687/RF=.7217/
ENS=.7193 — the RF leg's apparent near-match to published .721, diff .0007, is very plausibly
coincidence given neither the population nor the label-swap check agree.) By contrast, CF's
published row (n=2,255) is fully traceable and verified: `cf_rebuild_cell/cf2_bank_scores.parquet`
is exactly 2,255 rows and `scripts/comp_cf_rebuild_eval.py` documents its exact recipe — so
this is specifically an **AC-L1 matrix-loss problem**, not a systemic four-platform one.

**Corrected verdict: matrix NOT FOUND (AC-L1 cell specifically) — same category as the
registry's separately-flagged "comp dense .7112 ... NOT FOUND" case.** Per this job's own
stop rule ("if the matrix can't be found in ~30 min of search, write the discovery log and
stop that job"), the right outcome is a discovery log, not a gate-check verdict — the .012
"soft fail" number in the JSON is not a meaningful reproduction attempt of the true cell and
should not be quoted as such; it only characterizes the smaller, differently-labeled,
999-row original-pool subset. **Do not headline "AC strict-L1: strongest bank>dense
exemplar" or attach any Δ_interact/Δ_beyond number to it** until either (a) the true 2,495-row
AC-L1 bank-score matrix is located (check sk3 — several AC pipeline steps in
`cf_rebuild_2026_06_11.md` reference `sk3:datasets/codeforces_delta/...` paths for sibling
platforms; an AC-L1 equivalent may exist there unshipped, mirroring how "comp dense .7112"
was an sk3-only laptop-unfound artifact) or (b) the bank is rescored from scratch on the
existing 139-criterion candidate-only metric definitions over all 2,495 L1-labeled
`candidate_code` rows already sitting in `cell_ac_l1.parquet` (inputs are present; only the
bank-scoring pass itself is missing). Spec only, not run here — scoring is out of this
diagnostic job's scope.

---

## 4. N&C AGREE MAP PREP — verdict: inputs sufficient; brief already written

Found `methods/taste_decomposition/closure/nc_agree_brief.md` already present on disk,
dated 2026-08-06, matching this job's exact spec (population/splits/T-caveats brief,
map-round-only, no round run). Independently re-verified its key claims against source
files before accepting it as the deliverable:
- `results/nc_agree_layer1.json` gate PASSES to machine precision (diff 3.15e-7 on VA).
- `closure/samerows_preds/nc_agree_dense_preds_slim.csv` (5,046 rows) + `results/samerows_T_nc_agree.json`:
  dense-held-out AUC **.6034** (n=1,009), sitting between the divergent registry eval
  (.566)/test (.639) numbers — confirmed by direct read.
- `datasets/notice-and-comment/v4/nc_scores_shard{0..4}.npz` (198-rubric pre-GEPA Gemma-4-31B
  A-bank) present and confirmed as the exact matrix `X (1500,198)` per shard, `docket`-keyed.
- `methods/taste_decomposition/closure/maps_batch1/cells.py`'s `LOADERS` dict verified to
  stop at `{peer_curation, peer_revealed, cap_crowd, cap_finalist, nc_outcome}` (line 198-203)
  — **no `nc_agree` loader entry yet**, confirmed by direct grep; `nc_layer1_stack.py` verified
  to already expose the needed `valid_agr`/`y_agr_by_id`/`docket_m`/`X_m`/`text_m` attributes
  (lines 213-227) that a new loader would wire up — mechanical addition, no new scoring.

**T caveats carried in the brief** (both independently confirmed): (1) unstable-y eval/test
divergence (.566 vs .639, reconfirmed on identical held-out rows split apart: eval-only
.5660 vs test-only .6411, a .075 gap within one nominally-homogeneous set); (2) docket-identity
near-chance-collapsing confound — Layer 2 (`results/layer2_nc_agree.json`) shows
within-docket VA_nl AUC = **.4934 (chance)** vs docket-identity-alone AUC **.8616** — the
cell's entire predictive edge is cross-docket, not comment-content. Both make N&C agree a
**map-focused** (Track-B/nuisance-emphasis), not full-closure-curve, cell — consistent with
the freeze declaration's roster. No round was run; the maps-batch agent picks it up from the
existing brief.

---

## Four-line summary

1. **Press gate**: no real gap — gate already passed correctly (V/A each reproduce under
   their own historically-correct sklearn version); VA .6712 first-fit is a trustworthy
   Δ_interact anchor (+.030, CI clearly >0) but needs its `T_provisional` flag whenever quoted.
2. **Mathlib T**: T=.770 is unusable — doesn't match any reproducible number; nearest
   candidate (.767) is a flagged-optimistic test-split AUC on a *different, pre-deconfounding*
   population. Needs a fresh area-grouped, select-on-eval Llama rerun on the canonical
   n=7,956 slice (spec'd, not run); use topic-resid TF-IDF C=.736 as the interim comparator.
3. **Code competitions**: the true per-example bank matrix behind "AC strict-L1 bank .731"
   is NOT FOUND locally — only a smaller, earlier, differently-labeled 999-row subset
   survives, confirmed NOT to reproduce .731 under any label version (a first pass mistook
   it for the source and ran Layer-1 on it; superseded here). The published .731/.690
   comparison IS same-population at the aggregate level (both n=2,495, pos .836) — the
   matrix, not the population, is missing; check sk3 or rescore before any Δ is quoted.
4. **N&C agree map prep**: inputs sufficient, brief already exists and independently verified
   accurate (`methods/taste_decomposition/closure/nc_agree_brief.md`) — map-focused round only,
   flagged for the unstable eval/test split and the chance-level within-docket signal;
   handed off to the maps-batch agent, no round run here.
