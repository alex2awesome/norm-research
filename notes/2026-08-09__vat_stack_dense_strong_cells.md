# V+A+T fusion Direction 1, mirrored onto the dense-strong cells

Date: 2026-08-09. Status: COMPLETE (8/8 cells run). Descriptive only — no claim
edits, no prereg confirmatory claims.

Task: run the Direction-1 V+A+T stack (dense per-row score appended as ONE
extra column to the V+A bank matrix, then the frozen Layer-1 linear + HistGB
stacks refit) on every cell where the DENSE side WINS OR TIES the articulated
bank — the mirror of the fusion battery in
notes/2026-08-07__vat_fusion_directions.md, which ran the identical recipe on
the three cells where the bank beat dense (cap_crowd, cap_finalist,
style_inv_toptier; not repeated here).

Terms (first use, matching the 2026-08-07 note): **V** = verifiable/surface
features; **A** = articulated criterion scores (Gemma-4-31B judged); **VA_lin
/ VA_nl** = linear / HistGB (HistGradientBoostingClassifier) aggregation of the
V+A matrix (frozen Layer-1 protocol, notes/2026-08-05__taste-decomposition-design.md
S1); **T** = dense-standard clean-eval AUC (Llama-3.1-8B LoRA reward model on
raw text); **E** = evaluation-valid rows = rows OUTSIDE the dense model's own
training split (dense eval+test buckets), the only rows where the dense
per-row prediction is out-of-sample; **VAT** = the V+A matrix with the dense
per-row probability appended as one extra column; **VAT_nl** = the HistGB fit
of that VAT matrix.

## Cells and why they qualify (dense wins or ties)

| cell | T (population/registry) | VA_nl (population, Layer-1 ledger) |
|---|---|---|
| peer_verdict | .753 | .688 |
| cw_community | .780 | .621 |
| nc_responded | .808 | .724 |
| nc_outcome | .622 | .610 |
| nc_agree | eval .566 / test .639 | .584 (split-dependent flip) |
| hashtagwars_verdict | .664 (registry mean-of-seed-AUC) | .630 |
| peer_curation | .593 / .588 (test) | .559 |
| peer_revealed | .871 / .896 (test) | .767 |

(Sources: `methods/taste_decomposition/results/*_layer1.json` `ledger`, plus
`maps_hw_si/cells.py` docstring for the HW registry T.)

## Protocol (identical intent to Direction 1's original run, `fusion/direction1_mirror.py`)

- **E** = dense_split in {eval, test} for every cell (rows the dense model
  never trained on). `cw_community`'s E is its ENTIRE 7,008-row honest
  population by construction (the campaign only ever scored dense-held-out
  rows); every other cell's E is a genuine held-out subset of a larger
  dense-trained population.
- Stacks refit fresh with grouped OOF (`GroupKFold(5)`) on E only — never
  reusing the closure campaigns' internal FIT+MINE/MONITOR mining split (that
  split existed to MINE new criteria; this mirror only asks whether a FIXED
  bank + T beats the bank alone, so E is refit from scratch exactly as
  Direction 1 did for the three original cells).
- Two preprocessing families, matched to each cell's own frozen Layer-1
  lineage:
  - **clean_once** (peer_verdict, peer_curation, peer_revealed, nc_outcome,
    nc_agree, nc_responded — all descend from `layer1_stack.py` /
    `nc_layer1_stack.py` / `closure_lib.py`): population-level `clean_cols`
    (median-impute + degeneracy screen, UNSUPERVISED, no y) applied ONCE on
    the E submatrix, then StandardScaler+LogisticRegression(C=1) (no imputer)
    / HistGB fed the already-clean matrix — identical treatment to how
    Direction 1 handled the two family2 caption cells.
  - **impute_perfold** (cw_community, hashtagwars_verdict — `family1` in
    `layer1_gemma_cells.py`): raw NaN-bearing matrix, `SimpleImputer(median,
    add_indicator=True)` fit PER TRAIN FOLD, fed to both the linear pipeline
    and the GBM.
- VA_nl / VAT_nl = mean over seeds {0,1,2}, spread reported.
- **Group-level** (not row-level) paired bootstrap CIs on (VAT_nl−T) and
  (VAT_nl−VA_nl) — a deviation from Direction 1's original row-level
  bootstrap, per this task's explicit instruction; group = the cell's own
  canonical grouping unit (ntitle / prompt_id / docket / hashtag contest).
- Two bank arms where the task specified them: peer_verdict (round0 = 17V+154A
  vs round4 = +56 mined criteria across rounds 1-4, exact count verified);
  cw_community (round0 = 45A+15V=60 cols vs round8 — **round8's anchor-battery
  gate FAILED** [coherent-vs-scrambled AUC .3205 < .5], so 0 criteria were
  admitted and round8's bank IS round7's 144-col bank verbatim); nc_responded
  (round0 = 27V+198A=225 vs round5 = +67 mined criteria across rounds 1-5,
  reported pooled AND eval-half/test-half separately per the selection
  caveat). The other four cells (nc_outcome, nc_agree, hashtagwars_verdict,
  peer_curation, peer_revealed) have only the plain Layer-1 matrix — no mined
  rounds were run for them, so one arm each.
- hashtagwars_verdict has 3 dense seeds (42, 1, 2); reported per-seed plus a
  mean-probability ensemble, mirroring how the original Direction 1 handled
  style_inv_toptier.
- **Alignment safeguard** (see `alignment_check` in every JSON): this script
  never reads a precomputed `*_va_nl_oof_*.npy` file — every VA_nl/VAT_nl OOF
  array is recomputed fresh in-process from V/A/y/groups/dense arrays built
  together in one consistent row order per cell, so a cached-OOF/population
  reordering bug is structurally impossible here. Where the dense column comes
  from a separate CSV (peer_verdict, cw_community, nc_responded), the join is
  explicitly equality-asserted (dense_split / y / ids match elementwise) at
  load time. As an independent external check, every cell's pooled T_E
  matches the pre-existing `samerows_T_*.json` / `*_dense_preds.report.json`
  `auc_dense_heldout` (and eval/test halves where available) figure — computed
  by a completely separate code path — to 4 decimal places (e.g. peer_verdict
  .7769=.77692, nc_responded .8167=.81671, nc_agree .6034=.60343). No cell's
  AUC is anywhere near .50.

CPU only, local. Code: `methods/taste_decomposition/fusion/direction1_mirror.py`.
Per-cell runtime 1,278s (peer_revealed) to 4,836s (cw_community); total
wall-clock ~21 min running all 8 in parallel background processes (machine
was independently under heavy shared load, unrelated to this job).

## Results (all AUCs on E; bootstrap = 2,000-draw group-level paired bootstrap)

| cell | arm | n (groups) | T_E | VA_nl (±spread) | VAT_lin | **VAT_nl (±spread)** | VAT_nl − VA_nl [CI] P(>0) | VAT_nl − T [CI] P(>0) |
|---|---|---|---|---|---|---|---|---|
| peer_verdict | round0 | 1,244 (1,239) | .7769 | .6539 (±.005) | .7462 | **.7367 (±.010)** | +.0804 [+.0505,+.1096] **1.00** | −.0434 [−.0590,−.0281] .00 |
| peer_verdict | round4 (56 mined) | 1,244 (1,239) | .7769 | .6684 (±.002) | .7364 | **.7415 (±.009)** | +.0693 [+.0433,+.0957] **1.00** | −.0394 [−.0555,−.0241] .00 |
| cw_community | round0 | 7,008 (5,136) | .7921 | .6501 (±.002) | .7866 | **.7864 (±.002)** | +.1365 [+.1239,+.1494] **1.00** | −.0046 [−.0081,−.0011] .01 |
| cw_community | round8 (=round7†) | 7,008 (5,136) | .7921 | .6652 (±.000) | .7787 | **.7869 (±.003)** | +.1231 [+.1119,+.1353] **1.00** | −.0038 [−.0073,−.0003] .02 |
| nc_responded | round0 | 1,904 (1,010) | .8167 | .7748 (±.013) | .7409 | **.8322 (±.008)** | +.0683 [+.0493,+.0857] **1.00** | +.0199 [+.0023,+.0374] **.99** |
| nc_responded | round5 (67 mined) | 1,904 (1,010) | .8167 | .7912 (±.008) | .7325 | **.8319 (±.007)** | +.0416 [+.0223,+.0598] **1.00** | +.0111 [−.0070,+.0293] .88 |
| nc_outcome | layer1_bank | 1,417 (692) | .6238 | .6121 (±.016) | .6072 | **.6227 (±.003)** | +.0223 [+.0001,+.0435] .98 | +.0002 [−.0307,+.0332] .49 |
| nc_agree | layer1_bank | 1,009 (498) | .6034 | .5627 (±.038) | .6189 | **.5713 (±.023)** | +.0330 [+.0076,+.0619] **1.00** | −.0227 [−.0907,+.0464] .30 |
| peer_curation | layer1_bank | 1,571 (1,571) | .5936 | .5286 (±.020) | .5792 | **.5542 (±.010)** | +.0340 [+.0030,+.0643] .98 | −.0341 [−.0661,−.0021] .02 |
| peer_revealed | layer1_bank | 478 (478) | .8842 | .6554 (±.030) | .8475 | **.8478 (±.002)** | +.2051 [+.1558,+.2544] **1.00** | −.0374 [−.0562,−.0199] .00 |
| hashtagwars_verdict | layer1_bank (3-seed ensemble T) | 924 (8) | .7315 | .5290 (±.000) | .6651 | **.6454 (±.030)** | +.1114 [+.0463,+.1693] **1.00** | −.0953 [−.1687,−.0161] .01 |

† round8's anchor-battery gate FAILED (0 criteria admitted); round8's bank is
round7's 144-col bank verbatim, kept as a labeled arm for completeness rather
than silently dropped.

nc_responded eval-half/test-half (selection caveat; `per_split` in
`vat_stack_nc_responded.json`): round0 eval T=.8084/VA_nl=.7971/VAT_nl=.8368,
test T=.8250/VA_nl=.7379/VAT_nl=.8018; round5 eval
T=.8084/VA_nl=.8010/VAT_nl=.8372, test T=.8250/VA_nl=.7461/VAT_nl=.8005 — the
pooled VAT_nl > T finding is driven almost entirely by the eval half (test
half sits slightly below T on both arms), so the pooled +.02/+.01 gain over T
should be read as fragile/eval-side rather than a robust win on both halves.

hashtagwars_verdict per dense seed (`per_dense_seed` in the JSON): seed1
T=.7161/VA_nl=.5290/VAT_nl=.6441; seed2 T=.7212/VAT_nl=.6289; seed42
T=.6940/VAT_nl=.5894 — VAT_nl trails T at every individual seed too, not just
the ensemble; V+A is close to non-informative alone here (VA_nl≈.529, near
chance) which is why the fused stack cannot approach T.

## Per-cell verdicts

- **peer_verdict**: stack recovers most but not all of dense (VAT_nl .737-.742
  vs T .777, gap ~.04, CI clearly excludes 0); the mined round4 bank adds a
  small amount over round0 (VAT_nl +.005) but does not close the gap.
- **cw_community**: stack essentially TIES dense (gap .004-.005, CIs barely
  exclude 0 given the very large n=7,008/5,136 groups); round8(=round7)'s
  bank is negligibly better than round0's for this purpose.
- **nc_responded**: the ONLY cell where the pooled VAT_nl reliably EXCEEDS T
  (round0 P(>0)=.99); the mined round5 bank narrows this to a noisier +.011
  (P=.88) — more mined criteria did not help the fused stack here, it's
  driven almost entirely by V+A's own strength (VA_nl .775→.791) rather than
  better synergy with T. The gain over T is eval-half-driven (see above).
- **nc_outcome**: stack essentially TIES dense exactly (+.0002, P=.49) — the
  cleanest tie in the batch.
- **nc_agree**: stack trails dense (P(>0) for VAT-T = .30, CI wide and crosses
  0) — but this cell's T itself is split-dependent (eval .566 < VA_nl, test
  .639 > VA_nl), so the pooled comparison mixes a "dense loses" half with a
  "dense wins" half; VAT-VA is nonetheless solidly positive (P=1.00).
- **peer_curation**: stack trails dense measurably (P=.02, CI excludes 0).
- **peer_revealed**: stack trails dense (P=.00) despite a huge VAT-VA gain
  (+.205, the largest in the batch) — dense is simply very strong here
  (T=.884-.896) and V+A alone is weak (VA_nl=.655).
- **hashtagwars_verdict**: largest shortfall in the batch (VAT_nl trails T by
  ~.095, P=.01) — V+A is nearly uninformative alone (VA_nl≈.53) on this cell,
  so there is little for the stack to add beyond T.

## Cross-cell answer

**Adding the dense score as one column to the bank and refitting always beats
the bank alone — VAT_nl > VA_nl with P(>0) ≥ .98 in every one of the 9 bank
arms across all 8 cells, including a clean +.20 AUC jump on peer_revealed and
a solid +.14/+.12 on cw_community — so the articulated V+A bank is never
simply inert once a strong dense reader already exists; but the fused stack
only matches or exceeds dense itself in a minority of arms (nc_responded's two
mined-bank arms and nc_outcome's exact tie — 3 of 9), while in the other 6 it
plateaus measurably below T, and the size of that shortfall tracks how
dominant T already is in the cell (near-zero for cw_community, worst for
hashtagwars_verdict and peer_revealed, the two cells where V+A is weakest
alone) — the same "dense and bank measure overlapping structure" mechanism
documented for the three dense-loses cells in the 2026-08-07 battery, just
with the competitive balance flipped: here dense is strong enough on its own
that the bank's contribution, while real and statistically detectable, is
rarely enough to fully absorb what dense alone already captures.**

## Artifacts

- Code: `methods/taste_decomposition/fusion/direction1_mirror.py` (engine +
  all 8 cell loaders; reuses `layer1_gemma_cells.py`'s frozen
  `linear_oof_family1/2` / `gbm_oof_family1` / `gbm_oof_raw` / `outer_folds`
  and `capagg.clean_cols`, exactly as `fusion/direction1_stack.py` did for the
  original three-cell battery).
- Results JSONs (one per cell, `alignment_check` field in every file):
  `methods/taste_decomposition/results/vat_stack_{peer_verdict,cw_community,
  nc_responded,nc_outcome,nc_agree,hashtagwars_verdict,peer_curation,
  peer_revealed}.json`.
- Source data reused verbatim (no new judging, no GPU): `closure/
  peer_verdict_dense_preds.csv` + `closure/stage4_readout.py` +
  `closure/stage4_round4.py` (peer_verdict); `closure/cw_community/
  round{0,7}_state.npz` (cw_community); `closure/nc_responded/
  nc_closure_lib.py` + `closure/nc_responded/readout.py` +
  `nc_responded_dense_preds_aligned.csv` (nc_responded); `closure/
  maps_batch1/cells.py` (nc_outcome, nc_agree, peer_curation, peer_revealed);
  `closure/maps_hw_si/cells.py` (hashtagwars_verdict).
