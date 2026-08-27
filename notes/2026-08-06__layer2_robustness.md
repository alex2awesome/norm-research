# Layer 2 — robustness appendix (grouped transfer + nuisance-stratified readouts)

Date: 2026-08-06. Design: `notes/2026-08-05__taste-decomposition-design.md` §2
(spec) / §0 (quantity ledger). Driver: `methods/taste_decomposition/layer2_robustness.py`.
Per-cell JSONs: `methods/taste_decomposition/results/layer2_<cell>.json`.

Cells run, in the order completed: peer verdict, N&C responded, N&C outcome,
CW community, humor caption crowd (cap crowd), humor caption finalist (cap
finalist), peer curation, peer revealed, N&C agree (added mid-run by the
coordinator's health-check directive; N&C agree was not in the original §4
priority list but is the third N&C expert label and fits the same loader).

## 0. What ran, what didn't (skip log)

**Nothing was skipped.** All 9 candidate cells had both a usable VA_nl
(nonlinear V+A stack) and locally-available raw text, so every cell got the
full part (a) + part (b) treatment. This is a stronger outcome than the task
anticipated ("skip+report otherwise") — the local-file audit before coding
found that every priority cell's canonical text lives on disk already:
peer-review `verdict.jsonl` / `curation.jsonl` / `revealed.jsonl` (has `text`,
`year`), N&C `nc_vat_sample.jsonl` / `nc_unmatched_sample.jsonl` (has `text`,
`year`), CW `datasets/creative-writing/va_bank_v2/writingprompts_modeling_clean_reconstructed.csv.gz`
(exact-text-hash-verified against the Layer-1 item ids, 2,000/2,000 matched),
and `datasets/humor/caption_multiy/caption_contest_v2.jsonl` (id scheme
verified against the Layer-1 loader's own hash construction). No sk3 corpus
fetch was needed for any of the 9 cells.

**T (dense) is row-level only for peer verdict.** `methods/taste_decomposition/
closure/peer_verdict_dense_preds.csv` (built by the Layer-3 pilot's
`rescore_dense_same_rows.py`) has a per-row dense probability on the exact
6,030-row population, positionally verified against `layer1_stack.load_cell
('verdict')`'s row order (spot-checked rows 0, 1, 2, 3000, 6028, 6029 — all
matched on `judgement` and `ntitle`). Restricted to `dense_split ∈
{eval,test}` (n=1,244) per freeze change #2 (train rows are in-sample). For
the other 8 cells, sk3's `datasets/*/dense_llama/*/rm_out` directories (N&C
×3, peer curation, peer revealed, CW community `wp_clean_rm_out`, caption
crowd/finalist) contain **LoRA adapter checkpoints only** — no saved per-row
prediction file exists anywhere, locally or on sk3. Producing one is a new
GPU scoring pass (the design's own queued "cross-cutting GPU batch" in §4b),
which is out of Layer 2's CPU-only scope. Per the task's explicit fallback,
those 8 cells run part (a)/(b) on **VA_nl only**, with `T_available: false`
and a cell-specific `T_note` in the JSON citing the aggregate eval/test AUC
that IS on file (so the reader isn't left guessing what T "is", just that no
row-level score exists to stratify).

**VA_nl is recomputed fresh here (seed 0), not read from the saved
`*_va_nl_oof_*.npy` files.** Two of the eight cells' Layer-1 loaders build
their row order from Python **set** iteration
(`nc_layer1_stack.NCData.valid_out` / `.valid_agr`), which is not provably
stable against an independently-constructed id list across processes. Rather
than risk a silent row-misalignment between a saved OOF array and freshly
captured ids/text, this script recomputes VA_nl OOF (frozen Layer-1 grid,
seed 0, same grouped outer folds) **inside the same loop that captures ids**,
guaranteeing internal consistency by construction. Sanity gate: recomputed
VA_nl(seed0) vs. the Layer-1 ledger value (3-seed mean where available):

| cell | VA_nl (seed0, this script) | Layer-1 ledger VA_nl | diff |
|---|---:|---:|---:|
| peer verdict | .6876 | .6876 | +.0000 |
| N&C responded | .7257 | .7244 | +.0013 |
| N&C outcome | .6106 | .6102 | +.0005 |
| CW community | .6244 | .6207 | +.0038 |
| cap crowd | .6638 | .6656 | −.0018 |
| cap finalist | .6787 | .6800 | −.0013 |
| peer curation | .5610 | .5588 | +.0022 |
| peer revealed | .7653 | .7667 | −.0015 |
| N&C agree | .5863 | .5844 | +.0019 |

All within ≤.004 — consistent with ordinary seed-0-vs-3-seed-mean spread
(FREEZE CHANGE 1), not a row-alignment error.

**One bug, fixed mid-run:** the first `nc_outcome` attempt crashed
(`ValueError: Input X contains NaN`) inside the nuisance-alone logistic
regression fit — 7/7,084 docket comments have no parsed publication year, and
unlike `HistGradientBoostingClassifier`, `LogisticRegression` isn't
NaN-tolerant. The process died in a `nohup`-backgrounded shell; the failure
sat in an unread log for several hours before a coordinator health-check
caught it. Fixed by median-imputing the "date" nuisance column specifically
for the alone/joint LR fit (decile **stratification** still uses the raw,
unimputed years — a missing-year row just falls out of every date stratum,
the same treatment `stratified_auc` already gives any other NaN). Re-ran
clean; all 9 cells below reflect the fixed code. Process note for next time:
foreground runs with the log tailed directly, not background-and-poll.

## 1. Part (a) — grouped-transfer table

Per cell: **pooled AUC** vs. **within-group AUC** (n-weighted mean over the
cell's canonical grouping unit, groups with ≥20 rows and both classes) vs.
**group-identity-alone AUC** (the group's positive rate as the score,
estimated out-of-fold via a row-level — not group-level — K-fold so a row's
own label never informs its own score; groups unseen in a fold's train side
fall back to that fold's global rate).

| cell | n | group unit (n groups) | score | pooled AUC | within-group AUC (n qualifying / total) | group-identity-alone AUC |
|---|---:|---|---|---:|---|---:|
| peer verdict | 6,030 | ntitle (5,999) | VA_nl | .688 | n/a (0/5,999 ≥20) | .487 |
| peer verdict | 1,244† | ntitle (1,239) | T | .777 | n/a (0/1,239 ≥20) | .475 |
| N&C responded | 9,521 | docket (1,814) | VA_nl | .726 | **.675** (55/1,814) | **.916** |
| N&C outcome | 7,084 | docket (1,216) | VA_nl | .611 | **.558** (38/1,216) | **.856** |
| N&C agree | 5,046 | docket (944) | VA_nl | .586 | **.493** (20/944) | **.862** |
| CW community | 2,000 | prompt_id (1,500) | VA_nl | .624 | n/a (0/1,500 ≥20) | .579 |
| cap crowd | 10,893 | contest (223) | VA_nl | .664 | .674 (223/223) | .335 |
| cap finalist | 5,218 | contest (227) | VA_nl | .679 | .683 (227/227) | .178 |
| peer curation | 7,941 | ntitle (7,941) | VA_nl | .561 | n/a (0/7,941 ≥20) | .494 |
| peer revealed | 2,387 | ntitle (2,387) | VA_nl | .765 | n/a (0/2,387 ≥20) | .482 |

† T restricted to `dense_split ∈ {eval,test}` (freeze change #2); the pooled
row for VA_nl at the same n=1,244 subset is not separately reported here — the
n=6,030 pooled VA_nl AUC above is the one comparable to the Layer-1 ledger.

**Reading.** Two structurally different regimes:
- **Near-singleton grouping units** (peer review's `ntitle`, CW's
  `prompt_id`) have essentially no groups with ≥20 rows — within-group AUC is
  undefined by construction, and group-identity-alone AUC sits at chance
  (.48–.58), confirming these units carry no leakage risk because they carry
  almost no repeated-identity signal at all. This is itself informative, not
  a null result: it rules out an identity-leakage explanation for these five
  cells' Δ values.
- **Coarser grouping units diverge sharply.** N&C's `docket` (all three
  cells) shows group-identity-alone AUC of **.86–.92** — far above VA_nl's
  pooled AUC (.59–.73) — while within-docket AUC **drops** to .49–.68. This
  is the code-review repo-identity-leak pattern the design doc names as the
  exemplar: which docket a comment belongs to is a much stronger predictor of
  "responded / outcome / agree" than the comment's own V+A content, and most
  of VA_nl's pooled advantage over chance is a **between-docket** effect, not
  a within-docket one. **N&C agree is the most severe case**: within-docket
  AUC (.493) is at chance, i.e. VA_nl's entire .086 edge over .5 pooled is a
  cross-docket effect. Caption `contest` groups show the *opposite* and
  reassuring pattern: within-group AUC (.674–.683) is *not lower* than pooled
  (in fact marginally higher), and group-identity-alone is *below* chance-ish
  informativeness in a useful sense (.18–.34, i.e. knowing only which contest
  a caption is from is anti-informative about role/crowd-preference once
  estimated out-of-fold) — captions' V+A signal genuinely transfers across
  contests.

## 2. Part (b) — nuisance-stratified readouts

Uniform nuisance set: **length** (char length + log token-count proxy),
**format** (linebreak rate + markdown/list-marker line rate), **topic**
(k=20 KMeans on base `BAAI/bge-large-en-v1.5` embeddings, fit train-side-only
inside the cell's own grouped folds — an out-of-fold cluster label per row),
**date** (publication year; only peer-review and N&C carry one — CW and
caption cells have no date field, noted not imputed).

### 2a. Nuisance-alone AUC (grouped-OOF logistic regression: can the nuisance predict y by itself?)

| cell | length | format | topic | date | joint (all available) |
|---|---:|---:|---:|---:|---:|
| peer verdict | .536 | .497 | .508 | .519 | .521 |
| N&C responded | .621 | .516 | .525 | — | .634 |
| N&C outcome | .580 | .559 | .536 | .460 | .580 |
| N&C agree | .511 | .518 | .498 | .550 | .543 |
| CW community | .473 | .578 | .497 | — | .549 |
| cap crowd | .530 | .500 | .490 | — | .518 |
| cap finalist | .596 | .499 | .498 | — | .578 |
| peer curation | .534 | .476 | .490 | .598 | .593 |
| peer revealed | .559 | .548 | .438 | .502 | .475 |

(`—` = no date field for that corpus; N&C responded's date coverage fell
below the 90% finite-value gate because the unmatched-comment pool has sparser
metadata, so date was dropped for that cell specifically, not imputed.)

None of the nuisance dimensions come close to VA_nl's own pooled AUC on any
cell (joint nuisance-alone AUC tops out at .634, on N&C responded, vs. VA_nl
pooled .726) — the nuisance channels are real but modest confounds on their
own, not surrogates for the V+A signal.

### 2b. Decile/cluster-stratified AUC of VA_nl (n-weighted mean over strata with ≥20 rows + both classes; survival = |stratified − pooled| ≤ .02)

| cell | pooled AUC | length: stratified (drop) | format: stratified (drop) | topic: stratified (drop) | date: stratified (drop) |
|---|---:|---|---|---|---|
| peer verdict | .688 | .684 (.004) ✓ | .687 (.001) ✓ | .688 (.000) ✓ | .687 (.000) ✓ |
| N&C responded | .726 | .653 (**.073**) ✗ | .716 (.010) ✓ | .723 (.002) ✓ | — |
| N&C outcome | .611 | .564 (**.047**) ✗ | .587 (**.023**) ✗ | .596 (.014) ✓ | .610 (.001) ✓ |
| N&C agree | .586 | .567 (.019) ✓ | .566 (**.021**) ✗ | .581 (.005) ✓ | .578 (.008) ✓ |
| CW community | .624 | .623 (.002) ✓ | .597 (**.028**) ✗ | .623 (.002) ✓ | — |
| cap crowd | .664 | .658 (.006) ✓ | no qualifying strata* | .664 (.000) ✓ | — |
| cap finalist | .679 | .643 (**.036**) ✗ | no qualifying strata* | .681 (−.002) ✓ | — |
| peer curation | .561 | .552 (.009) ✓ | .554 (.007) ✓ | .560 (.001) ✓ | .536 (**.025**) ✗ |
| peer revealed | .765 | .758 (.007) ✓ | .758 (.007) ✓ | .761 (.004) ✓ | .764 (.001) ✓ |

*Captions are single-line text (contest-entry captions), so linebreak rate is
uniformly 0 — the decile cut degenerates to a single undefined bin (pandas
`qcut` on a zero-variance array returns no valid bin edges). The
format-**alone** AUC (table 2a) still computed fine because it also uses
markdown-rate as a second column; only the *stratified* readout, which needs
`linebreak_rate` deciles specifically, is undefined for these two cells. Not
a bug — a genuine corpus property (documented in `layer2_robustness.py`'s
`_MARK_RE`/`linebreak_rate` comments).

### 2c. Decile/cluster-stratified AUC of T (peer verdict only — the one cell with row-level T)

| dimension | pooled T (n=1,244) | stratified T | drop | survives |
|---|---:|---:|---:|---|
| length | .777 | .771 | .006 | ✓ |
| format | .777 | .777 | .000 | ✓ |
| topic | .777 | .775 | .002 | ✓ |
| date | .777 | .775 | .002 | ✓ |

T is essentially nuisance-invariant on the one cell where it could be tested.

## 3. Survival-flag failures — the interesting cells

Seven (dimension, cell) pairs, spread across 6 of the 9 cells, fail the .02
survival tolerance, all on VA_nl (T only exists, and survives cleanly, for
peer verdict):

| cell | dimension | pooled | stratified | drop | reading |
|---|---|---:|---:|---:|---|
| **N&C responded** | length | .726 | .653 | **.073** | largest failure in the whole set — comment length is doing real, non-nuisance-alone work inside VA_nl's advantage |
| **N&C outcome** | length | .611 | .564 | .047 | same direction as responded, smaller cell |
| **cap finalist** | length | .679 | .643 | .036 | editor-pick finalist selection leans on caption length more than the pooled number shows |
| **CW community** | format | .624 | .597 | .028 | markdown/linebreak density (fan-fiction formatting habits) partly rides along with VA_nl |
| **peer curation** | date | .561 | .536 | .025 | oral/spotlight selection has drifted by publication year (venue norms changing across ICLR years) enough to matter at the .02 threshold |
| **N&C outcome** | format | .611 | .587 | .023 | secondary to its length failure |
| **N&C agree** | format | .586 | .566 | .021 | just over the line |

**Reading (descriptive).** Length is the dominant nuisance failure mode —
it fails on 3 of the 6 cells that fail anything at all (N&C responded, N&C
outcome, cap finalist), and by the largest margins (.036–.073, vs. .019–.026
for the format/date failures). This lines up with part (a)'s finding that
N&C's docket-identity leak is real: longer comments may correlate with which
dockets get responses (agencies engaging more with substantive, longer
submissions), so the length-stratification failure and the group-identity-
alone finding are plausibly the same underlying structure seen two ways.
Format fails on 3 cells (N&C outcome, CW community, N&C agree) at smaller
margins (.021–.028); date fails once (peer curation, .025). **Three of nine
cells — peer verdict, cap crowd, peer revealed — show zero nuisance
failures on any dimension**; the other six each fail at least one dimension
(five fail exactly one; N&C outcome fails two: length and format). So
VA_nl's pooled AUC is not a length/format/topic/date artifact on a third of
the program's cells, but on the other six (all three N&C cells, CW
community, cap finalist, peer curation) at least one nuisance dimension is
doing more work than the pooled number alone would suggest.

## 4. Artifacts

- Driver: `methods/taste_decomposition/layer2_robustness.py`
- Per-cell JSON: `methods/taste_decomposition/results/layer2_{peer_verdict,
  nc_responded,nc_outcome,cw_community,cap_crowd,cap_finalist,peer_curation,
  peer_revealed,nc_agree}.json`
- Embedding cache (reused across cells sharing a text pool, sha1-keyed):
  `methods/taste_decomposition/data_cache/bge_large_embed_cache.npz`
  (38,936 cached vectors after all 9 cells; BAAI/bge-large-en-v1.5, plain
  `transformers` AutoModel + CLS-pooling + L2-normalize on MPS — not
  sentence-transformers, whose import chain pulls in a broken local
  tensorflow build on this machine).
- T source for peer verdict: `methods/taste_decomposition/closure/
  peer_verdict_dense_preds.csv` (+ `.report.json`), from the Layer-3 pilot's
  `rescore_dense_same_rows.py`.
